from __future__ import annotations

"""Paper trading runner for the first US equities Alpaca demo."""

import argparse
import json
import math
import time as wall_time
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

import pandas as pd

from stockmachine.backtest.protocols import AccountSnapshot, ExecutionPolicy, MarketBar, PortfolioPolicy, SignalModel
from stockmachine.data.loaders import load_us_equities_dataset
from stockmachine.domain.models import OrderIntent, Signal, TargetPosition
from stockmachine.execution import NextOpenOrderExecutionPolicy
from stockmachine.execution.brokers import AlpacaTradeUpdateStream, AlpacaTradingAdapter
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.live import (
    AccountSyncResult,
    AlpacaAccountSync,
    BrokerOrderSnapshot,
    PollingOrderReconciler,
    SessionGuard,
    SessionGuardRequest,
    TradeUpdateMessageSource,
    recover_open_orders,
    stream_trade_updates,
)
from stockmachine.monitoring.reports import (
    PaperRunFailure,
    PaperRunManifest,
    PaperRunReport,
    build_paper_artifact_link,
    build_paper_run_manifest,
    build_paper_run_report,
)
from stockmachine.portfolio import RiskAwareTopKPortfolioPolicy
from stockmachine.research.us_equities_baseline import (
    BENCHMARK_SYMBOL,
    DEFAULT_UNIVERSE,
    OverlayConfig,
    build_metadata_from_silver,
    build_price_panel_from_silver,
    build_research_frame,
    generate_walk_forward_predictions,
)
from stockmachine.risk import (
    CLIENT_ORDER_ID_PREFIX,
    BrokerAwareOrderRiskPolicy,
    OrderValidationIssue,
    OrderValidationResult,
    build_client_order_id,
    extract_client_order_id,
)
from stockmachine.state import (
    EquitySnapshotRecord,
    FillAuditRecord,
    LocalLedger,
    OrderDecisionRecord,
    OrderRecord,
    RunRecord,
    SignalRecord,
    TargetRecord,
)


class UniverseProvider(Protocol):
    def get_universe(self, session_date: date) -> Sequence[str]:
        """Return the symbols eligible for this session."""


class AccountProvider(Protocol):
    def get_account_snapshot(self, session_date: date) -> AccountSnapshot:
        """Return broker or paper-account state for the session."""


class AccountSyncProvider(Protocol):
    def sync(self, session_date: date | None = None) -> AccountSyncResult:
        """Return broker account, positions, clock, and project snapshot."""


class MarketDataProvider(Protocol):
    def get_bars(self, session_date: date, universe: Sequence[str]) -> Mapping[str, MarketBar]:
        """Return session bars used by the execution policy."""


class OrderSubmitter(Protocol):
    def submit_orders(self, orders: Sequence[OrderIntent]) -> Sequence[Mapping[str, Any] | object]:
        """Submit orders to the broker or paper gateway."""


class OpenOrderProvider(Protocol):
    def list_orders(
        self,
        *,
        status: str | None = None,
        symbols: Sequence[str] | None = None,
    ) -> Sequence[Mapping[str, Any] | object]:
        """Return broker orders for validation and reconciliation."""


class OrderStatusProvider(Protocol):
    def get_order(self, order_id: str) -> Mapping[str, Any] | object:
        """Return one broker order snapshot by id."""


class OrderRiskPolicy(Protocol):
    def validate(
        self,
        *,
        orders: Sequence[OrderIntent],
        account_sync: object | None = None,
        open_orders: Sequence[Mapping[str, Any] | object] = (),
    ) -> OrderValidationResult:
        """Validate orders against broker-aware rules."""


@dataclass(slots=True, frozen=True)
class PaperRunConfig:
    session_date: date
    dry_run: bool = True
    universe: tuple[str, ...] = ()
    run_name: str = "paper-demo"
    artifact_dir: str | None = None
    execution_equity_cap: float | None = None
    post_submit_poll_seconds: float = 15.0
    post_submit_poll_interval_seconds: float = 2.0


@dataclass(slots=True)
class PaperRunDependencies:
    signal_model: SignalModel
    portfolio_policy: PortfolioPolicy
    execution_policy: ExecutionPolicy
    universe_provider: UniverseProvider
    account_provider: AccountProvider
    market_data_provider: MarketDataProvider
    order_submitter: OrderSubmitter | None = None
    account_sync_provider: AccountSyncProvider | None = None
    open_order_provider: OpenOrderProvider | None = None
    order_status_provider: OrderStatusProvider | None = None
    trade_update_stream: TradeUpdateMessageSource | None = None
    risk_policy: OrderRiskPolicy | None = None
    ledger: LocalLedger | None = None
    reconciler: PollingOrderReconciler | None = None
    market: str = "US"


@dataclass(slots=True)
class StaticUniverseProvider:
    symbols: tuple[str, ...] = ()

    def get_universe(self, session_date: date) -> Sequence[str]:
        return self.symbols


@dataclass(slots=True)
class StaticAccountProvider:
    cash: float = 0.0
    equity: float = 0.0

    def get_account_snapshot(self, session_date: date) -> AccountSnapshot:
        return AccountSnapshot(session_date=session_date, cash=self.cash, equity=self.equity, gross_exposure=0.0)


@dataclass(slots=True)
class StaticMarketDataProvider:
    def get_bars(self, session_date: date, universe: Sequence[str]) -> Mapping[str, MarketBar]:
        return {}


@dataclass(slots=True)
class NullSignalModel:
    horizon_bars: int = 5

    def predict(self, session_date: date, universe: Sequence[str]) -> Sequence[Signal]:
        return []


@dataclass(slots=True)
class SilverDatasetCache:
    """Lazy-loaded silver dataset used by the paper runner."""

    layout: StorageLayout = field(default_factory=StorageLayout)
    _dataset: dict[str, pd.DataFrame] | None = field(default=None, init=False, repr=False)

    def load(self) -> dict[str, pd.DataFrame]:
        if self._dataset is None:
            self._dataset = load_us_equities_dataset(layout=self.layout)
        return self._dataset

    def resolve_session_date(self, requested_date: date) -> date:
        dataset = self.load()
        daily_bar = dataset["daily_bar"]
        if daily_bar.empty:
            raise RuntimeError("Silver daily_bar table is empty; cannot resolve paper session date.")
        available_dates = sorted(pd.to_datetime(daily_bar["session_date"]).dt.date.unique())
        not_after = [current for current in available_dates if current <= requested_date]
        if not_after:
            return not_after[-1]
        return available_dates[0]


@dataclass(slots=True)
class LatestSilverUniverseProvider:
    dataset_cache: SilverDatasetCache

    def get_universe(self, session_date: date) -> Sequence[str]:
        dataset = self.dataset_cache.load()
        daily_bar = dataset["daily_bar"]
        if daily_bar.empty:
            return DEFAULT_UNIVERSE

        effective_date = self.dataset_cache.resolve_session_date(session_date)
        rows = daily_bar[pd.to_datetime(daily_bar["session_date"]).dt.date == effective_date].copy()
        if rows.empty:
            return DEFAULT_UNIVERSE
        symbols = sorted(symbol for symbol in rows["symbol"].dropna().unique().tolist() if symbol != BENCHMARK_SYMBOL)
        return tuple(symbols or DEFAULT_UNIVERSE)


@dataclass(slots=True)
class LatestSilverBarProvider:
    dataset_cache: SilverDatasetCache

    def get_bars(self, session_date: date, universe: Sequence[str]) -> Mapping[str, MarketBar]:
        dataset = self.dataset_cache.load()
        daily_bar = dataset["daily_bar"]
        if daily_bar.empty:
            return {}

        frame = daily_bar[daily_bar["symbol"].isin(list(universe))].copy()
        if frame.empty:
            return {}

        frame["session_date"] = pd.to_datetime(frame["session_date"])
        frame = frame[frame["session_date"].dt.date <= session_date]
        if frame.empty:
            return {}

        latest = (
            frame.sort_values(["symbol", "session_date"])
            .drop_duplicates(subset=["symbol"], keep="last")
            .reset_index(drop=True)
        )
        return {
            row.symbol: MarketBar(
                session_date=row.session_date.date(),
                symbol=row.symbol,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                vwap=float(row.vwap) if getattr(row, "vwap", None) is not None and pd.notna(row.vwap) else None,
                adj_open=float(row.adj_open) if getattr(row, "adj_open", None) is not None and pd.notna(row.adj_open) else None,
            )
            for row in latest.itertuples(index=False)
        }


@dataclass(slots=True)
class SilverWalkForwardSignalModel:
    """Build latest walk-forward signals directly from silver tables."""

    dataset_cache: SilverDatasetCache
    model_name: str = "hist_gbm"
    horizon_bars: int = 5

    def predict(self, session_date: date, universe: Sequence[str]) -> Sequence[Signal]:
        dataset = self.dataset_cache.load()
        effective_date = self.dataset_cache.resolve_session_date(session_date)

        price_data = build_price_panel_from_silver(dataset)
        metadata = build_metadata_from_silver(dataset)
        research_frame = build_research_frame(
            price_data,
            benchmark_symbol=BENCHMARK_SYMBOL,
            horizon=self.horizon_bars,
            symbol_metadata=metadata,
        )
        prediction_month_start = effective_date.replace(day=1).isoformat()
        predictions = generate_walk_forward_predictions(
            research_frame,
            predict_start=prediction_month_start,
        )
        model_predictions = predictions[predictions["model"] == self.model_name].copy()
        if model_predictions.empty:
            return []

        available_dates = sorted(
            current_date
            for current_date in model_predictions["date"].dt.date.unique()
            if current_date <= effective_date
        )
        if not available_dates:
            available_dates = sorted(model_predictions["date"].dt.date.unique())
        if not available_dates:
            return []

        prediction_date = available_dates[-1]
        current = model_predictions[
            (model_predictions["date"].dt.date == prediction_date)
            & (model_predictions["symbol"].isin(list(universe)))
        ].copy()
        if current.empty:
            return []

        timestamp = datetime.combine(prediction_date, time(16, 0))
        return [
            Signal(
                symbol=row.symbol,
                side="LONG",
                score=float(row.score),
                confidence=float(row.confidence),
                horizon_bars=self.horizon_bars,
                timestamp=timestamp,
                meta={
                    "prediction_date": prediction_date.isoformat(),
                    "sector": row.sector,
                    "industry": row.industry,
                    "close": float(row.close),
                    "vol_20": float(row.vol_20),
                    "median_dollar_volume_20": float(row.median_dollar_volume_20),
                    "reference_price": float(row.close),
                },
            )
            for row in current.itertuples(index=False)
        ]


@dataclass(slots=True)
class SyncedAccountProvider:
    account_sync: AccountSyncProvider

    def get_account_snapshot(self, session_date: date) -> AccountSnapshot:
        return self.account_sync.sync(session_date).snapshot


@dataclass(slots=True)
class AlpacaOrderSubmitter:
    broker: AlpacaTradingAdapter

    def submit_orders(self, orders: Sequence[OrderIntent]) -> Sequence[Mapping[str, Any] | object]:
        submitted = []
        for order in orders:
            order_type, time_in_force = self._map_order_type(order)
            submitted.append(
                self.broker.submit_order(
                    symbol=order.symbol,
                    side=order.side.lower(),
                    quantity=float(order.quantity),
                    order_type=order_type,
                    time_in_force=time_in_force,
                    limit_price=order.limit_price,
                    client_order_id=order.meta.get("client_order_id") if isinstance(order.meta, dict) else None,
                )
            )
        return tuple(submitted)

    def _map_order_type(self, order: OrderIntent) -> tuple[str, str]:
        order_type = order.order_type.lower()
        if order_type == "market_on_open":
            return "market", "opg"
        if order_type == "limit":
            return "limit", "day"
        return "market", "day"


@dataclass(slots=True)
class PaperRunner:
    dependencies: PaperRunDependencies

    def run(self, config: PaperRunConfig) -> PaperRunReport:
        failures: list[PaperRunFailure] = []
        run_id = uuid4().hex
        meta: dict[str, Any] = {"run_name": config.run_name}
        manifest: PaperRunManifest | None = None
        counts = {
            "universe_size": 0,
            "signals": 0,
            "targets": 0,
            "orders": 0,
            "approved_orders": 0,
            "blocked_orders": 0,
            "submitted_batches": 0,
            "reconciled_orders": 0,
        }

        try:
            ledger = self.dependencies.ledger
            requested_session_date = config.session_date
            runtime_session_date = config.session_date
            if ledger is not None:
                ledger.initialize()
                ledger.record_run(
                    RunRecord(
                        run_id=run_id,
                        strategy_name=config.run_name,
                        market=self.dependencies.market,
                        created_at_utc=datetime.now(timezone.utc),
                        status="running",
                        meta={
                            "dry_run": config.dry_run,
                            "session_date": requested_session_date.isoformat(),
                            "run_mode": "dry_run" if config.dry_run else "execute",
                        },
                    )
                )

            guard_result = self._evaluate_session_guard(ledger=ledger, config=config)
            if guard_result is not None:
                meta["session_guard"] = guard_result.to_dict()
                if guard_result.effective_session_date is not None:
                    runtime_session_date = guard_result.effective_session_date
                    meta["effective_session_date"] = runtime_session_date.isoformat()
                if not guard_result.allowed:
                    manifest = self._build_run_manifest(
                        run_id=run_id,
                        config=config,
                        universe=tuple(config.universe),
                        effective_session_date=runtime_session_date,
                        session_guard_result=guard_result.to_dict(),
                    )
                    if ledger is not None:
                        ledger.record_run_manifest(manifest.to_record())
                        ledger.finish_run(
                            run_id,
                            finished_at_utc=datetime.now(timezone.utc),
                            status="blocked",
                        )
                    failures.extend(self._guard_failures(guard_result))
                    return build_paper_run_report(
                        session_date=requested_session_date,
                        dry_run=config.dry_run,
                        stage="blocked",
                        counts=counts,
                        failures=tuple(failures),
                        meta=meta,
                        run_id=run_id,
                        manifest=manifest,
                    )

            if ledger is not None:
                ledger.record_run(
                    RunRecord(
                        run_id=run_id,
                        strategy_name=config.run_name,
                        market=self.dependencies.market,
                        created_at_utc=datetime.now(timezone.utc),
                        status="running",
                        meta={
                            "dry_run": config.dry_run,
                            "session_date": requested_session_date.isoformat(),
                            "effective_session_date": runtime_session_date.isoformat(),
                            "run_mode": "dry_run" if config.dry_run else "execute",
                        },
                    )
                )

            universe = tuple(config.universe) or tuple(self.dependencies.universe_provider.get_universe(runtime_session_date))
            counts["universe_size"] = len(universe)
            manifest = self._build_run_manifest(
                run_id=run_id,
                config=config,
                universe=universe,
                effective_session_date=runtime_session_date,
                session_guard_result=meta.get("session_guard"),
            )
            if ledger is not None:
                ledger.record_run_manifest(manifest.to_record())

            account_sync_result: AccountSyncResult | None = None
            if self.dependencies.account_sync_provider is not None:
                account_sync_result = self.dependencies.account_sync_provider.sync(runtime_session_date)
                account = account_sync_result.snapshot
                meta["market_open"] = bool(account_sync_result.clock.is_open)
                meta["buying_power"] = float(getattr(account_sync_result.broker_account, "buying_power", 0.0))
            else:
                account = self.dependencies.account_provider.get_account_snapshot(runtime_session_date)

            recovered_open_orders: tuple[Mapping[str, Any] | object, ...] = ()
            if ledger is not None and self.dependencies.open_order_provider is not None:
                broker_open_orders = tuple(self.dependencies.open_order_provider.list_orders(status="open"))
                recovered_open_orders = broker_open_orders
                recovery_result = recover_open_orders(
                    ledger,
                    broker_open_orders,
                    reconciler=self.dependencies.reconciler or PollingOrderReconciler(ledger),
                )
                meta["recovery"] = {
                    "aligned_open_orders": recovery_result.plan.aligned_count,
                    "orphan_broker_orders": recovery_result.plan.orphan_count,
                    "stale_ledger_orders": recovery_result.plan.stale_count,
                    "reconciled_created_orders": recovery_result.reconciliation.created_orders,
                    "reconciled_updated_orders": recovery_result.reconciliation.updated_orders,
                    "reconciled_fill_events": recovery_result.reconciliation.fill_events_created,
                    "status_counts": recovery_result.reconciliation.status_counts,
                }

            if ledger is not None:
                ledger.record_equity_snapshot(
                    EquitySnapshotRecord(
                        run_id=run_id,
                        session_date=runtime_session_date,
                        timestamp_utc=datetime.now(timezone.utc),
                        cash=float(account.cash),
                        equity=float(account.equity),
                        gross_exposure=float(account.gross_exposure),
                        payload=self._account_payload(account_sync_result, account),
                    )
                )

            signals = list(self.dependencies.signal_model.predict(runtime_session_date, universe))
            counts["signals"] = len(signals)
            if ledger is not None:
                for signal in signals:
                    ledger.append_signal(
                        SignalRecord(
                            run_id=run_id,
                            session_date=runtime_session_date,
                            symbol=signal.symbol,
                            side=signal.side,
                            score=signal.score,
                            confidence=signal.confidence,
                            horizon_bars=signal.horizon_bars,
                            timestamp_utc=signal.timestamp,
                            meta=dict(signal.meta),
                        )
                    )

            targets = list(self.dependencies.portfolio_policy.build_targets(runtime_session_date, signals, account))
            counts["targets"] = len(targets)
            if ledger is not None:
                for target in targets:
                    ledger.append_target(
                        TargetRecord(
                            run_id=run_id,
                            session_date=runtime_session_date,
                            symbol=target.symbol,
                            target_weight=target.target_weight,
                            max_weight=target.max_weight,
                            reason=target.reason,
                            timestamp_utc=target.timestamp,
                            meta=dict(target.meta),
                        )
                    )

            bars = dict(self.dependencies.market_data_provider.get_bars(runtime_session_date, universe))
            execution_account = self._execution_account_snapshot(account, config.execution_equity_cap)
            if config.execution_equity_cap is not None:
                meta["execution_equity_cap"] = float(config.execution_equity_cap)
            orders = list(
                self.dependencies.execution_policy.generate_orders(
                    runtime_session_date,
                    targets,
                    bars,
                    execution_account,
                )
            )
            orders = list(self._with_client_order_ids(orders, session_date=runtime_session_date, run_id=run_id))
            counts["orders"] = len(orders)

            open_orders: Sequence[Mapping[str, Any] | object] = ()
            approved_orders = tuple(orders)
            validation_issues: tuple[OrderValidationIssue, ...] = ()
            if self.dependencies.risk_policy is not None:
                if orders:
                    symbols = {order.symbol.upper() for order in orders}
                    if recovered_open_orders:
                        open_orders = tuple(
                            order
                            for order in recovered_open_orders
                            if str(_payload_value(order, "symbol", default="")).upper() in symbols
                        )
                    elif self.dependencies.open_order_provider is not None:
                        open_orders = tuple(
                            self.dependencies.open_order_provider.list_orders(status="open", symbols=sorted(symbols))
                        )
                validation = self.dependencies.risk_policy.validate(
                    orders=orders,
                    account_sync=account_sync_result,
                    open_orders=open_orders,
                )
                approved_orders = validation.approved_orders
                validation_issues = validation.issues
                counts["approved_orders"] = len(validation.approved_orders)
                counts["blocked_orders"] = len(validation.blocked_orders)
                meta["estimated_notional"] = validation.estimated_notional
                meta["open_order_symbols"] = list(validation.open_order_symbols)
                failures.extend(self._issues_to_failures(validation))
            else:
                counts["approved_orders"] = len(orders)

            if ledger is not None and orders:
                self._record_order_decisions(
                    ledger=ledger,
                    run_id=run_id,
                    session_date=runtime_session_date,
                    orders=orders,
                    approved_orders=approved_orders,
                    issues=validation_issues,
                )

            submitted: Sequence[Mapping[str, Any] | object] = ()
            reconciliation_result = None
            if not config.dry_run:
                if self.dependencies.order_submitter is None:
                    failures.append(
                        PaperRunFailure(
                            stage="submit_orders",
                            reason="missing_order_submitter",
                            details={"dry_run": config.dry_run},
                        )
                    )
                elif approved_orders:
                    submitted = tuple(self.dependencies.order_submitter.submit_orders(approved_orders))
                    counts["submitted_batches"] = len(submitted)
                    if ledger is not None:
                        order_contexts = self._seed_submitted_orders(
                            ledger=ledger,
                            run_id=run_id,
                            session_date=runtime_session_date,
                            orders=approved_orders,
                            submitted=submitted,
                        )
                        reconciler = self.dependencies.reconciler or PollingOrderReconciler(ledger)
                        reconciliation_result = reconciler.reconcile_orders(submitted)
                        counts["reconciled_orders"] = reconciliation_result.created_orders + reconciliation_result.updated_orders
                        meta["reconciliation"] = {
                            "created_orders": reconciliation_result.created_orders,
                            "updated_orders": reconciliation_result.updated_orders,
                            "fill_events_created": reconciliation_result.fill_events_created,
                            "status_counts": reconciliation_result.status_counts,
                        }
                        if self.dependencies.order_status_provider is not None and config.post_submit_poll_seconds > 0:
                            follow_up = self._poll_submitted_orders(
                                submitted,
                                reconciler=reconciler,
                                poll_seconds=config.post_submit_poll_seconds,
                                poll_interval_seconds=config.post_submit_poll_interval_seconds,
                            )
                            counts["reconciled_orders"] += follow_up["reconciled_orders"]
                            meta["post_submit_poll"] = follow_up["meta"]
                        fill_audits_created = self._record_fill_audits(
                            ledger=ledger,
                            run_id=run_id,
                            session_date=runtime_session_date,
                            order_contexts=order_contexts,
                        )
                        meta["reconciliation"]["fill_audits_created"] = fill_audits_created
            elif not approved_orders:
                meta["dry_run_note"] = "no_approved_orders_generated"

            stage = "completed" if not failures else "completed_with_warnings"
            if ledger is not None:
                ledger.finish_run(
                    run_id,
                    finished_at_utc=datetime.now(timezone.utc),
                    status="finished" if not failures else "finished_with_warnings",
                )

            return build_paper_run_report(
                session_date=config.session_date,
                dry_run=config.dry_run,
                stage=stage,
                counts=counts,
                failures=tuple(failures),
                meta=meta,
                run_id=run_id,
                manifest=manifest,
            )
        except Exception as exc:  # pragma: no cover - defensive wrapper for CLI use
            if self.dependencies.ledger is not None:
                self.dependencies.ledger.finish_run(
                    run_id,
                    finished_at_utc=datetime.now(timezone.utc),
                    status="failed",
                )
            failures.append(
                PaperRunFailure(
                    stage="run",
                    reason=type(exc).__name__,
                    details={"message": str(exc)},
                )
            )
            return build_paper_run_report(
                session_date=config.session_date,
                dry_run=config.dry_run,
                stage="failed",
                counts=counts,
                failures=tuple(failures),
                meta=meta,
                run_id=run_id,
                manifest=manifest,
            )

    def close(self) -> None:
        if self.dependencies.ledger is not None:
            self.dependencies.ledger.close()

    def _execution_account_snapshot(
        self,
        account: AccountSnapshot,
        execution_equity_cap: float | None,
    ) -> AccountSnapshot:
        if execution_equity_cap is None:
            return account
        capped_equity = max(0.0, min(float(account.equity), float(execution_equity_cap)))
        capped_cash = max(0.0, min(float(account.cash), capped_equity))
        return AccountSnapshot(
            session_date=account.session_date,
            cash=capped_cash,
            equity=capped_equity,
            gross_exposure=account.gross_exposure,
            positions=account.positions,
        )

    def _issues_to_failures(self, validation: OrderValidationResult) -> list[PaperRunFailure]:
        return [
            PaperRunFailure(
                stage="risk_gate",
                reason=issue.code,
                details={"message": issue.message, "symbol": issue.symbol, **issue.details},
            )
            for issue in validation.issues
        ]

    def _build_run_manifest(
        self,
        *,
        run_id: str,
        config: PaperRunConfig,
        universe: Sequence[str],
        effective_session_date: date | None = None,
        session_guard_result: Mapping[str, Any] | None = None,
    ) -> PaperRunManifest:
        risk_policy = self.dependencies.risk_policy
        execution_policy = self.dependencies.execution_policy
        data_snapshot = {
            "market": self.dependencies.market,
            "session_date": config.session_date.isoformat(),
            "effective_session_date": (
                effective_session_date.isoformat() if effective_session_date is not None else config.session_date.isoformat()
            ),
            "universe_size": len(universe),
            "universe_symbols": list(universe),
            "signal_model": type(self.dependencies.signal_model).__name__,
            "portfolio_policy": type(self.dependencies.portfolio_policy).__name__,
        }
        if session_guard_result is not None:
            data_snapshot["session_guard"] = dict(session_guard_result)
        risk_config = {
            "policy": type(risk_policy).__name__ if risk_policy is not None else None,
            "require_market_open": bool(getattr(risk_policy, "require_market_open", False)),
            "block_duplicate_symbols": bool(getattr(risk_policy, "block_duplicate_symbols", False)),
            "require_client_order_id": bool(getattr(risk_policy, "require_client_order_id", False)),
            "min_buying_power_buffer": float(getattr(risk_policy, "min_buying_power_buffer", 0.0) or 0.0),
            "max_order_notional": getattr(risk_policy, "max_order_notional", None),
            "max_total_notional": getattr(risk_policy, "max_total_notional", None),
            "max_total_orders": getattr(risk_policy, "max_total_orders", None),
        }
        execution_config = {
            "policy": type(execution_policy).__name__,
            "execution_equity_cap": config.execution_equity_cap,
            "post_submit_poll_seconds": config.post_submit_poll_seconds,
            "post_submit_poll_interval_seconds": config.post_submit_poll_interval_seconds,
        }
        ledger_path = str(self.dependencies.ledger.path) if self.dependencies.ledger is not None else None
        return build_paper_run_manifest(
            run_id=run_id,
            session_date=config.session_date,
            strategy_name=config.run_name,
            model_name=self._resolve_model_name(),
            dry_run=config.dry_run,
            client_order_id_prefix=CLIENT_ORDER_ID_PREFIX,
            data_snapshot=data_snapshot,
            risk_policy=risk_config,
            execution_policy=execution_config,
            meta={
                "market": self.dependencies.market,
                "ledger_path": ledger_path,
                "artifact_dir": config.artifact_dir,
                "requested_session_date": config.session_date.isoformat(),
                "effective_session_date": (
                    effective_session_date.isoformat() if effective_session_date is not None else config.session_date.isoformat()
                ),
                "artifact_link": build_paper_artifact_link(
                    artifact_dir=config.artifact_dir,
                    model_name=self._resolve_model_name(),
                    session_date=config.session_date,
                ).to_dict(),
            },
        )

    def _evaluate_session_guard(
        self,
        *,
        ledger: LocalLedger | None,
        config: PaperRunConfig,
    ):
        if ledger is None:
            return None
        silver_session_dates = self._available_silver_session_dates()
        if silver_session_dates is None:
            return None
        return SessionGuard(ledger).evaluate(
            SessionGuardRequest(
                ledger=ledger,
                strategy_name=config.run_name,
                session_date=config.session_date,
                dry_run=config.dry_run,
                silver_session_dates=silver_session_dates,
            )
        )

    def _available_silver_session_dates(self) -> tuple[date, ...] | None:
        for candidate in (
            getattr(self.dependencies.signal_model, "dataset_cache", None),
            getattr(self.dependencies.market_data_provider, "dataset_cache", None),
            getattr(self.dependencies.universe_provider, "dataset_cache", None),
        ):
            if candidate is None or not hasattr(candidate, "load"):
                continue
            try:
                dataset = candidate.load()
            except Exception:
                continue
            daily_bar = dataset.get("daily_bar")
            if daily_bar is None or daily_bar.empty:
                return ()
            return tuple(sorted(pd.to_datetime(daily_bar["session_date"]).dt.date.unique()))
        return None

    def _guard_failures(self, guard_result) -> list[PaperRunFailure]:
        details = dict(guard_result.data_freshness_meta)
        if guard_result.duplicate_run_ids:
            details["duplicate_run_ids"] = list(guard_result.duplicate_run_ids)
        return [
            PaperRunFailure(
                stage="session_guard",
                reason=reason,
                details=details,
            )
            for reason in guard_result.reasons
        ]

    def _resolve_model_name(self) -> str:
        explicit_name = getattr(self.dependencies.signal_model, "model_name", None)
        if explicit_name is None:
            explicit_name = getattr(self.dependencies.signal_model, "__name__", None)
        return str(explicit_name or type(self.dependencies.signal_model).__name__)

    def _with_client_order_ids(
        self,
        orders: Sequence[OrderIntent],
        *,
        session_date: date,
        run_id: str,
    ) -> tuple[OrderIntent, ...]:
        resolved_orders: list[OrderIntent] = []
        for sequence, order in enumerate(orders, start=1):
            client_order_id = extract_client_order_id(order) or build_client_order_id(
                session_date=session_date,
                symbol=order.symbol,
                side=order.side,
                sequence=sequence,
                run_id=run_id,
            )
            order_meta = dict(order.meta)
            order_meta["client_order_id"] = client_order_id
            resolved_orders.append(replace(order, meta=order_meta))
        return tuple(resolved_orders)

    def _record_order_decisions(
        self,
        *,
        ledger: LocalLedger,
        run_id: str,
        session_date: date,
        orders: Sequence[OrderIntent],
        approved_orders: Sequence[OrderIntent],
        issues: Sequence[OrderValidationIssue],
    ) -> None:
        approved_ids = {
            extract_client_order_id(order)
            for order in approved_orders
            if extract_client_order_id(order) is not None
        }
        issues_by_symbol: dict[str, list[OrderValidationIssue]] = {}
        for issue in issues:
            if issue.symbol is None:
                continue
            issues_by_symbol.setdefault(issue.symbol.upper(), []).append(issue)

        for order in orders:
            client_order_id = extract_client_order_id(order)
            if client_order_id is None:
                continue
            matching_issues = issues_by_symbol.get(order.symbol.upper(), [])
            approved = client_order_id in approved_ids
            reference_price = self._estimate_order_reference_price(order)
            ledger.record_order_decision(
                OrderDecisionRecord(
                    decision_id=uuid4().hex,
                    run_id=run_id,
                    session_date=session_date,
                    client_order_id=client_order_id,
                    symbol=order.symbol.upper(),
                    side=order.side.upper(),
                    decision_type="risk_gate",
                    decision_price=reference_price,
                    estimated_notional=self._estimate_order_notional(order),
                    approved=approved,
                    reason="approved" if approved else self._decision_reason(matching_issues),
                    decision_at_utc=datetime.now(timezone.utc),
                    meta={
                        "order_type": order.order_type,
                        "quantity": int(order.quantity),
                        "limit_price": order.limit_price,
                        "issue_codes": [issue.code for issue in matching_issues],
                    },
                )
            )

    def _seed_submitted_orders(
        self,
        *,
        ledger: LocalLedger,
        run_id: str,
        session_date: date,
        orders: Sequence[OrderIntent],
        submitted: Sequence[Mapping[str, Any] | object],
    ) -> dict[str, dict[str, Any]]:
        order_contexts: dict[str, dict[str, Any]] = {}
        for order, payload in zip(orders, submitted):
            snapshot = payload if isinstance(payload, BrokerOrderSnapshot) else BrokerOrderSnapshot.from_payload(payload)
            client_order_id = extract_client_order_id(order) or snapshot.client_order_id
            ledger.upsert_order(
                OrderRecord(
                    order_id=snapshot.order_id,
                    run_id=run_id,
                    session_date=session_date,
                    client_order_id=client_order_id,
                    symbol=snapshot.symbol or order.symbol,
                    side=(snapshot.side or order.side).lower(),
                    quantity=snapshot.quantity or int(order.quantity),
                    order_type=snapshot.order_type or order.order_type,
                    limit_price=snapshot.limit_price if snapshot.limit_price is not None else order.limit_price,
                    status=snapshot.status,
                    filled_quantity=0,
                    avg_fill_price=None,
                    submitted_at_utc=snapshot.submitted_at_utc,
                    updated_at_utc=snapshot.updated_at_utc,
                    broker_payload=snapshot.raw,
                )
            )
            order_contexts[snapshot.order_id] = {
                "client_order_id": client_order_id,
                "expected_price": self._estimate_order_reference_price(order),
                "side": order.side,
                "order_meta": dict(order.meta),
            }
        return order_contexts

    def _record_fill_audits(
        self,
        *,
        ledger: LocalLedger,
        run_id: str,
        session_date: date,
        order_contexts: Mapping[str, Mapping[str, Any]],
    ) -> int:
        existing_audit_ids = {audit.audit_id for audit in ledger.list_fill_audits(run_id=run_id)}
        created = 0

        for order_id, context in order_contexts.items():
            order_record = ledger.get_order(order_id)
            if order_record is None:
                continue
            fills = ledger.list_fills(order_id=order_id, run_id=run_id)
            for fill in fills:
                audit_id = f"audit:{fill.fill_id}"
                if audit_id in existing_audit_ids:
                    continue
                expected_price = self._coerce_optional_float(context.get("expected_price"))
                slippage = self._estimate_fill_slippage(
                    side=str(context.get("side") or fill.side),
                    expected_price=expected_price,
                    fill_price=float(fill.price),
                )
                fee = self._coerce_optional_float(context.get("order_meta", {}).get("fee_estimate"))
                ledger.record_fill_audit(
                    FillAuditRecord(
                        audit_id=audit_id,
                        order_id=fill.order_id,
                        run_id=run_id,
                        session_date=order_record.session_date or session_date,
                        client_order_id=str(context.get("client_order_id") or order_record.client_order_id or ""),
                        symbol=fill.symbol,
                        side=fill.side,
                        quantity=fill.quantity,
                        expected_price=expected_price,
                        price=fill.price,
                        slippage=slippage,
                        fee=fee,
                        filled_at_utc=fill.filled_at_utc,
                        meta={
                            "broker_order_status": order_record.status,
                        },
                    )
                )
                existing_audit_ids.add(audit_id)
                created += 1

        return created

    def _decision_reason(self, issues: Sequence[OrderValidationIssue]) -> str:
        if not issues:
            return "blocked"
        return ",".join(issue.code for issue in issues)

    def _estimate_order_reference_price(self, order: OrderIntent) -> float | None:
        if order.limit_price is not None and order.limit_price > 0:
            return float(order.limit_price)
        for key in ("reference_price", "close", "open", "last_price"):
            value = order.meta.get(key)
            if value is None:
                continue
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            if price > 0:
                return price
        return None

    def _estimate_order_notional(self, order: OrderIntent) -> float:
        reference_price = self._estimate_order_reference_price(order)
        if reference_price is None:
            return 0.0
        return float(order.quantity) * float(reference_price)

    def _estimate_fill_slippage(
        self,
        *,
        side: str,
        expected_price: float | None,
        fill_price: float,
    ) -> float | None:
        if expected_price is None:
            return None
        normalized_side = side.upper()
        if normalized_side == "SELL":
            return float(expected_price) - float(fill_price)
        return float(fill_price) - float(expected_price)

    def _coerce_optional_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _poll_submitted_orders(
        self,
        submitted: Sequence[Mapping[str, Any] | object],
        *,
        reconciler: PollingOrderReconciler,
        poll_seconds: float,
        poll_interval_seconds: float,
    ) -> dict[str, Any]:
        if self.dependencies.order_status_provider is None:
            return {
                "reconciled_orders": 0,
                "meta": {
                    "attempts": 0,
                    "status_counts": {},
                    "terminal_orders": 0,
                    "remaining_open_orders": 0,
                    "poll_seconds": poll_seconds,
                    "poll_interval_seconds": poll_interval_seconds,
                    "source_mode": "none",
                    "fallback_used": False,
                    "websocket_messages": 0,
                    "websocket_error": None,
                    "seed_order_ids": [],
                },
            }

        pending_order_ids = [
            order_id
            for order_id in (_payload_value(payload, "order_id", "id") for payload in submitted)
            if order_id
        ]
        effective_poll_step = poll_interval_seconds if poll_interval_seconds > 0 else 0.25
        max_polls = 1
        if poll_seconds > 0:
            max_polls = max(1, int(math.ceil(poll_seconds / effective_poll_step)))

        def _fetch_orders() -> Sequence[Mapping[str, Any] | object]:
            return [self.dependencies.order_status_provider.get_order(str(order_id)) for order_id in pending_order_ids]

        stream_result = stream_trade_updates(
            stream_source=self.dependencies.trade_update_stream,
            fetch_orders=_fetch_orders,
            max_messages=max(1, len(pending_order_ids) * 8),
            max_polls=max_polls,
            idle_messages_to_stop=1,
            idle_polls_to_stop=max_polls,
            sleep_seconds=max(0.0, poll_interval_seconds),
        )
        snapshots = [self.dependencies.order_status_provider.get_order(str(order_id)) for order_id in pending_order_ids]
        result = reconciler.reconcile_orders(snapshots)
        reconciled_orders = result.created_orders + result.updated_orders
        latest_status_counts = dict(result.status_counts)
        pending_order_ids = [
            str(_payload_value(snapshot, "order_id", "id"))
            for snapshot in snapshots
            if str(_payload_value(snapshot, "status", default="")).lower()
            not in {"filled", "canceled", "cancelled", "rejected", "expired"}
        ]

        return {
            "reconciled_orders": reconciled_orders,
            "meta": {
                "attempts": stream_result.polls,
                "status_counts": latest_status_counts,
                "terminal_orders": len(submitted) - len(pending_order_ids),
                "remaining_open_orders": len(pending_order_ids),
                "poll_seconds": poll_seconds,
                "poll_interval_seconds": poll_interval_seconds,
                "source_mode": stream_result.source_mode,
                "fallback_used": stream_result.fallback_used,
                "websocket_messages": stream_result.websocket_messages,
                "websocket_error": stream_result.websocket_error,
                "seed_order_ids": list(stream_result.seed_order_ids),
            },
        }

    def _account_payload(self, account_sync_result: AccountSyncResult | None, account: AccountSnapshot) -> dict[str, Any]:
        if account_sync_result is None:
            return {
                "session_date": account.session_date.isoformat(),
                "cash": float(account.cash),
                "equity": float(account.equity),
                "gross_exposure": float(account.gross_exposure),
            }
        return {
            "session_date": account.session_date.isoformat(),
            "cash": float(account.cash),
            "equity": float(account.equity),
            "gross_exposure": float(account.gross_exposure),
            "broker_account": _payload_to_dict(account_sync_result.broker_account),
            "clock": _payload_to_dict(account_sync_result.clock),
            "positions": [_payload_to_dict(position) for position in account_sync_result.broker_positions],
        }


@dataclass(slots=True)
class _DefaultPortfolioPolicy:
    def build_targets(self, session_date: date, signals: Sequence[Signal], account: AccountSnapshot) -> Sequence[TargetPosition]:
        return []


@dataclass(slots=True)
class _DefaultExecutionPolicy:
    def generate_orders(
        self,
        session_date: date,
        targets: Sequence[TargetPosition],
        bars: Mapping[str, MarketBar],
        account: AccountSnapshot,
    ) -> Sequence[OrderIntent]:
        return []


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the first Alpaca paper-trading demo.")
    parser.add_argument("--session-date", default=date.today().isoformat())
    parser.add_argument("--universe", nargs="*", default=[])
    parser.add_argument("--run-name", default="paper-demo")
    parser.add_argument("--execute", action="store_true", help="Submit orders instead of dry-run mode.")
    parser.add_argument("--output-format", choices=("json",), default="json")
    parser.add_argument("--demo-mode", action="store_true", help="Use the static no-op dependencies.")
    parser.add_argument("--model", default="hist_gbm")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--ledger-path", default="artifacts/paper_demo/paper_ledger.sqlite3")
    parser.add_argument("--min-close", type=float, default=10.0)
    parser.add_argument("--min-median-dollar-volume-20", type=float, default=50_000_000.0)
    parser.add_argument("--max-vol-20", type=float, default=0.04)
    parser.add_argument("--max-positions-per-sector", type=int, default=2)
    parser.add_argument("--disable-sector-neutral", action="store_true")
    parser.add_argument("--require-market-open", action="store_true")
    parser.add_argument("--min-buying-power-buffer", type=float, default=0.0)
    parser.add_argument("--max-order-notional", type=float, default=None)
    parser.add_argument("--max-total-notional", type=float, default=None)
    parser.add_argument("--max-total-orders", type=int, default=None)
    parser.add_argument("--execution-equity-cap", type=float, default=None)
    parser.add_argument("--post-submit-poll-seconds", type=float, default=15.0)
    parser.add_argument("--post-submit-poll-interval-seconds", type=float, default=2.0)
    return parser


def build_demo_runner(universe: Sequence[str] | None = None) -> tuple[PaperRunner, PaperRunConfig]:
    resolved_universe = tuple(universe or ())
    dependencies = PaperRunDependencies(
        signal_model=NullSignalModel(),
        portfolio_policy=_DefaultPortfolioPolicy(),
        execution_policy=_DefaultExecutionPolicy(),
        universe_provider=StaticUniverseProvider(symbols=resolved_universe),
        account_provider=StaticAccountProvider(),
        market_data_provider=StaticMarketDataProvider(),
        order_submitter=None,
    )
    config = PaperRunConfig(
        session_date=date.today(),
        dry_run=True,
        universe=resolved_universe,
        execution_equity_cap=None,
        post_submit_poll_seconds=0.0,
        post_submit_poll_interval_seconds=0.0,
    )
    return PaperRunner(dependencies=dependencies), config


def build_alpaca_paper_runner(
    *,
    universe: Sequence[str] | None = None,
    session_date: date | None = None,
    model_name: str = "hist_gbm",
    top_k: int = 10,
    horizon: int = 5,
    overlay_config: OverlayConfig | None = None,
    layout: StorageLayout | None = None,
    ledger_path: str | Path = "artifacts/paper_demo/paper_ledger.sqlite3",
    require_market_open: bool = False,
    min_buying_power_buffer: float = 0.0,
    max_order_notional: float | None = None,
    max_total_notional: float | None = None,
    max_total_orders: int | None = None,
    require_paper_environment: bool = True,
) -> tuple[PaperRunner, PaperRunConfig]:
    config = overlay_config or OverlayConfig()
    storage = layout or StorageLayout()
    dataset_cache = SilverDatasetCache(layout=storage)
    broker = AlpacaTradingAdapter.from_env()
    if require_paper_environment and not broker.is_paper_trading_environment():
        raise RuntimeError(
            f"Refusing to build paper runner against non-paper trading endpoint: {broker.credentials.trading_base_url}"
        )
    account_sync = AlpacaAccountSync(broker)
    ledger = LocalLedger(ledger_path)
    ledger.initialize()

    dependencies = PaperRunDependencies(
        signal_model=SilverWalkForwardSignalModel(
            dataset_cache=dataset_cache,
            model_name=model_name,
            horizon_bars=horizon,
        ),
        portfolio_policy=RiskAwareTopKPortfolioPolicy(
            top_k=top_k,
            min_close=config.min_close,
            min_median_dollar_volume_20=config.min_median_dollar_volume_20,
            max_vol_20=config.max_vol_20,
            max_positions_per_sector=config.max_positions_per_sector,
            sector_neutral=config.sector_neutral,
        ),
        execution_policy=NextOpenOrderExecutionPolicy(),
        universe_provider=StaticUniverseProvider(tuple(universe)) if universe else LatestSilverUniverseProvider(dataset_cache),
        account_provider=SyncedAccountProvider(account_sync),
        market_data_provider=LatestSilverBarProvider(dataset_cache),
        order_submitter=AlpacaOrderSubmitter(broker),
        account_sync_provider=account_sync,
        open_order_provider=broker,
        order_status_provider=broker,
        trade_update_stream=AlpacaTradeUpdateStream(broker.credentials),
        risk_policy=BrokerAwareOrderRiskPolicy(
            require_market_open=require_market_open,
            require_client_order_id=True,
            min_buying_power_buffer=min_buying_power_buffer,
            max_order_notional=max_order_notional,
            max_total_notional=max_total_notional,
            max_total_orders=max_total_orders,
        ),
        ledger=ledger,
        reconciler=PollingOrderReconciler(ledger),
        market="US",
    )
    paper_config = PaperRunConfig(
        session_date=session_date or date.today(),
        dry_run=True,
        universe=tuple(universe or ()),
        run_name=f"{model_name}-paper-demo",
        artifact_dir=None,
        execution_equity_cap=None,
        post_submit_poll_seconds=15.0,
        post_submit_poll_interval_seconds=2.0,
    )
    return PaperRunner(dependencies=dependencies), paper_config


def parse_session_date(raw_value: str) -> date:
    return date.fromisoformat(raw_value)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    session_date = parse_session_date(args.session_date)

    runner: PaperRunner | None = None
    try:
        if args.demo_mode:
            runner, _ = build_demo_runner(args.universe)
        else:
            overlay_config = OverlayConfig(
                min_close=args.min_close,
                min_median_dollar_volume_20=args.min_median_dollar_volume_20,
                max_vol_20=args.max_vol_20,
                max_positions_per_sector=args.max_positions_per_sector,
                sector_neutral=not args.disable_sector_neutral,
            )
            runner, _ = build_alpaca_paper_runner(
                universe=args.universe,
                session_date=session_date,
                model_name=args.model,
                top_k=args.top_k,
                horizon=args.horizon,
                overlay_config=overlay_config,
                layout=StorageLayout(root=Path(args.data_root)),
                ledger_path=args.ledger_path,
                require_market_open=args.require_market_open,
                min_buying_power_buffer=args.min_buying_power_buffer,
                max_order_notional=args.max_order_notional,
                max_total_notional=args.max_total_notional,
                max_total_orders=args.max_total_orders,
            )

        config = PaperRunConfig(
            session_date=session_date,
            dry_run=not args.execute,
            universe=tuple(args.universe),
            run_name=args.run_name,
            execution_equity_cap=args.execution_equity_cap,
            post_submit_poll_seconds=args.post_submit_poll_seconds,
            post_submit_poll_interval_seconds=args.post_submit_poll_interval_seconds,
        )
        report = runner.run(config)
    except Exception as exc:
        report = build_paper_run_report(
            session_date=session_date,
            dry_run=not args.execute,
            stage="initialization_failed",
            counts={
                "universe_size": 0,
                "signals": 0,
                "targets": 0,
                "orders": 0,
                "approved_orders": 0,
                "blocked_orders": 0,
                "submitted_batches": 0,
                "reconciled_orders": 0,
            },
            failures=(
                PaperRunFailure(
                    stage="initialize_runner",
                    reason=type(exc).__name__,
                    details={"message": str(exc)},
                ),
            ),
        )
    finally:
        if runner is not None:
            runner.close()

    print(json.dumps(report.to_dict(), indent=2))


def _payload_to_dict(payload: Mapping[str, Any] | object) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return {str(key): _json_safe(value) for key, value in payload.items()}
    if is_dataclass(payload):
        return _payload_to_dict(asdict(payload))
    if hasattr(payload, "__dict__"):
        return {
            key: _json_safe(value)
            for key, value in vars(payload).items()
            if not key.startswith("_")
        }
    return {"repr": repr(payload)}


def _payload_value(payload: Mapping[str, Any] | object, *names: str, default: Any = None) -> Any:
    if isinstance(payload, Mapping):
        for name in names:
            if name in payload:
                return payload[name]
        return default
    for name in names:
        if hasattr(payload, name):
            return getattr(payload, name)
    return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _payload_to_dict(value)
    return value


if __name__ == "__main__":
    main()
