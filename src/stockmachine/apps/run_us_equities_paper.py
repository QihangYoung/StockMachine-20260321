from __future__ import annotations

"""Paper trading runner for the first US equities Alpaca demo."""

import argparse
import json
import math
import time as wall_time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

import pandas as pd

from stockmachine.alpha import assert_supported_alpha_expert, list_alpha_expert_names
from stockmachine.backtest.protocols import AccountSnapshot, ExecutionPolicy, MarketBar, PortfolioPolicy, SignalModel
from stockmachine.data.loaders import load_us_equities_dataset
from stockmachine.domain.models import OrderIntent, Signal, TargetPosition
from stockmachine.execution import NextOpenOrderExecutionPolicy, SameSessionMarketOrderExecutionPolicy
from stockmachine.execution.brokers import AlpacaTradeUpdateStream, AlpacaTradingAdapter
from stockmachine.execution.brokers import classify_buy_retry_reason, is_retryable_buy_rejection, shrink_quantity_for_retry
from stockmachine.execution.models import SubmissionRetryRecord, SubmissionRetryReport
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.live import (
    AccountSyncResult,
    AlpacaAccountSync,
    BrokerOrderSnapshot,
    PollingOrderReconciler,
    SessionGuard,
    SessionGuardRequest,
    TradeUpdateMessageSource,
    backfill_order_statuses,
    recover_open_orders,
    record_fill_audits_for_orders,
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
    build_point_in_time_metadata_history,
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
    horizon_bars: int | None = None
    strategy_profile: str | None = None
    strategy_family: str | None = None
    strategy_project: str | None = None
    strategy_horizon_bucket: str | None = None
    strategy_track: str | None = None
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
    last_prediction_context: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_name = assert_supported_alpha_expert(self.model_name)

    def predict(self, session_date: date, universe: Sequence[str]) -> Sequence[Signal]:
        self.last_prediction_context = None
        dataset = self.dataset_cache.load()
        effective_date = self.dataset_cache.resolve_session_date(session_date)

        price_data = build_price_panel_from_silver(dataset)
        metadata = build_point_in_time_metadata_history(
            pd.Index(price_data.loc[price_data["symbol"] != BENCHMARK_SYMBOL, "date"].drop_duplicates().sort_values()),
            symbol_master_frame=dataset["symbol_master"],
            industry_membership_frame=dataset["industry_membership"],
        )
        research_frame = build_research_frame(
            price_data,
            benchmark_symbol=BENCHMARK_SYMBOL,
            horizon=self.horizon_bars,
            symbol_metadata=metadata,
            drop_unlabeled_rows=False,
        )
        prediction_month_start = effective_date.replace(day=1).isoformat()
        predictions = generate_walk_forward_predictions(
            research_frame,
            predict_start=prediction_month_start,
            requested_models=(self.model_name,),
            include_partial_current_test=True,
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
        prediction_slice = model_predictions[model_predictions["date"].dt.date == prediction_date].copy()
        self.last_prediction_context = _build_prediction_context(
            prediction_frame=prediction_slice,
            prediction_date=prediction_date,
        )
        current = prediction_slice[prediction_slice["symbol"].isin(list(universe))].copy()
        if current.empty:
            return []

        timestamp = datetime.combine(prediction_date, time(16, 0))
        prediction_context = self.last_prediction_context or {}
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
                    "training_window_start": prediction_context.get("training_window", {}).get("start"),
                    "training_window_end": prediction_context.get("training_window", {}).get("end"),
                    "validation_window_start": prediction_context.get("validation_window", {}).get("start"),
                    "validation_window_end": prediction_context.get("validation_window", {}).get("end"),
                    "prediction_window_start": prediction_context.get("prediction_window", {}).get("start"),
                    "prediction_window_end": prediction_context.get("prediction_window", {}).get("end"),
                },
            )
            for row in current.itertuples(index=False)
        ]


def _build_prediction_context(
    *,
    prediction_frame: pd.DataFrame,
    prediction_date: date,
) -> dict[str, Any] | None:
    if prediction_frame.empty:
        return None
    row = prediction_frame.iloc[0]
    return {
        "prediction_date": prediction_date.isoformat(),
        "split_fold_index": int(row["split_fold_index"]) if "split_fold_index" in prediction_frame.columns and pd.notna(row.get("split_fold_index")) else None,
        "split_anchor_date": _iso_date_or_none(row.get("split_anchor_date")),
        "training_window": {
            "start": _iso_date_or_none(row.get("split_train_start")),
            "end": _iso_date_or_none(row.get("split_train_end")),
        },
        "validation_window": {
            "start": _iso_date_or_none(row.get("split_validation_start")),
            "end": _iso_date_or_none(row.get("split_validation_end")),
        },
        "prediction_window": {
            "start": _iso_date_or_none(row.get("split_test_start")),
            "end": _iso_date_or_none(row.get("split_test_end")),
        },
    }


def _iso_date_or_none(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.date().isoformat()


@dataclass(slots=True)
class SyncedAccountProvider:
    account_sync: AccountSyncProvider

    def get_account_snapshot(self, session_date: date) -> AccountSnapshot:
        return self.account_sync.sync(session_date).snapshot


@dataclass(slots=True)
class AlpacaOrderSubmitter:
    broker: AlpacaTradingAdapter
    retry_shrink_ratio: float = 0.5
    last_submission_report: SubmissionRetryReport | None = field(default=None, init=False, repr=False)

    def submit_orders(self, orders: Sequence[OrderIntent]) -> Sequence[Mapping[str, Any] | object]:
        submitted = []
        records: list[SubmissionRetryRecord] = []
        attempted_orders = 0
        failed_orders = 0
        self.last_submission_report = SubmissionRetryReport(
            attempted_orders=len(orders),
            submitted_orders=0,
            retried_orders=0,
            failed_orders=0,
            records=(),
        )
        for order in orders:
            attempted_orders += 1
            try:
                broker_order, record = self._submit_one(order)
            except Exception as exc:
                failed_orders += 1
                records.append(self._build_failure_record(order, exc))
                self.last_submission_report = self._build_submission_report(
                    records=records,
                    attempted_orders=attempted_orders,
                    submitted_orders=len(submitted),
                    failed_orders=failed_orders,
                )
                raise
            submitted.append(broker_order)
            records.append(record)
        self.last_submission_report = self._build_submission_report(
            records=records,
            attempted_orders=attempted_orders,
            submitted_orders=len(submitted),
            failed_orders=failed_orders,
        )
        return tuple(submitted)

    def _submit_one(self, order: OrderIntent) -> tuple[Mapping[str, Any] | object, SubmissionRetryRecord]:
        order_type, time_in_force = self._map_order_type(order)
        original_quantity = int(order.quantity)
        initial_notional = self._estimate_order_notional(order)
        client_order_id = order.meta.get("client_order_id") if isinstance(order.meta, dict) else None

        try:
            submitted = self.broker.submit_order(
                symbol=order.symbol,
                side=order.side.lower(),
                quantity=float(order.quantity),
                order_type=order_type,
                time_in_force=time_in_force,
                limit_price=order.limit_price,
                client_order_id=client_order_id,
            )
            return submitted, self._build_success_record(
                order=order,
                submitted=submitted,
                original_quantity=original_quantity,
                initial_notional=initial_notional,
            )
        except Exception as exc:
            if not is_retryable_buy_rejection(exc, side=order.side) or original_quantity <= 1:
                raise

            retry_reason = classify_buy_retry_reason(exc) or "buy_rejection"
            retry_quantity = shrink_quantity_for_retry(original_quantity, shrink_ratio=self.retry_shrink_ratio)
            if retry_quantity <= 0 or retry_quantity >= original_quantity:
                raise

            retry_meta = dict(order.meta)
            retry_meta.update(
                {
                    "submission_retry_attempt": 1,
                    "submission_retry_reason": retry_reason,
                    "submission_retry_shrink_ratio": float(self.retry_shrink_ratio),
                    "submission_retry_original_quantity": original_quantity,
                    "submission_retry_quantity": retry_quantity,
                    "submission_retry_initial_error": str(exc),
                }
            )
            retry_order = replace(order, quantity=retry_quantity, meta=retry_meta)
            try:
                retried = self.broker.submit_order(
                    symbol=retry_order.symbol,
                    side=retry_order.side.lower(),
                    quantity=float(retry_order.quantity),
                    order_type=order_type,
                    time_in_force=time_in_force,
                    limit_price=retry_order.limit_price,
                    client_order_id=client_order_id,
                )
            except Exception as retry_exc:
                raise retry_exc from exc
            return retried, self._build_retry_record(
                order=order,
                submitted=retried,
                original_quantity=original_quantity,
                final_quantity=retry_quantity,
                retry_reason=retry_reason,
                initial_error=str(exc),
                initial_notional=initial_notional,
                retry_scale=self.retry_shrink_ratio,
            )

    def _map_order_type(self, order: OrderIntent) -> tuple[str, str]:
        order_type = order.order_type.lower()
        if order_type == "market_on_open":
            return "market", "opg"
        if order_type == "limit":
            return "limit", "day"
        return "market", "day"

    def _build_success_record(
        self,
        *,
        order: OrderIntent,
        submitted: Mapping[str, Any] | object,
        original_quantity: int,
        initial_notional: float | None,
    ) -> SubmissionRetryRecord:
        snapshot = submitted if isinstance(submitted, Mapping) else _payload_to_dict(submitted)
        final_quantity = int(float(_payload_value(snapshot, "qty", "quantity", default=order.quantity)))
        final_notional = self._estimate_quantity_notional(order, final_quantity)
        return SubmissionRetryRecord(
            symbol=order.symbol,
            side=order.side.upper(),
            client_order_id=extract_client_order_id(order),
            original_quantity=original_quantity,
            final_quantity=final_quantity,
            retry_used=False,
            initial_notional=initial_notional,
            final_notional=final_notional,
            final_order_id=str(_payload_value(snapshot, "order_id", "id", default="")),
            final_status=str(_payload_value(snapshot, "status", default="")),
            meta={
                "order_type": order.order_type,
            },
        )

    def _build_retry_record(
        self,
        *,
        order: OrderIntent,
        submitted: Mapping[str, Any] | object,
        original_quantity: int,
        final_quantity: int,
        retry_reason: str,
        initial_error: str,
        initial_notional: float | None,
        retry_scale: float,
    ) -> SubmissionRetryRecord:
        snapshot = submitted if isinstance(submitted, Mapping) else _payload_to_dict(submitted)
        final_notional = self._estimate_quantity_notional(order, final_quantity)
        return SubmissionRetryRecord(
            symbol=order.symbol,
            side=order.side.upper(),
            client_order_id=extract_client_order_id(order),
            original_quantity=original_quantity,
            final_quantity=final_quantity,
            retry_used=True,
            retry_reason=retry_reason,
            retry_scale=retry_scale,
            initial_error=initial_error,
            final_order_id=str(_payload_value(snapshot, "order_id", "id", default="")),
            final_status=str(_payload_value(snapshot, "status", default="")),
            initial_notional=initial_notional,
            final_notional=final_notional,
            meta={
                "order_type": order.order_type,
                "retry_quantity": final_quantity,
            },
        )

    def _build_failure_record(self, order: OrderIntent, error: Exception) -> SubmissionRetryRecord:
        original_quantity = int(order.quantity)
        initial_notional = self._estimate_order_notional(order)
        retry_reason = classify_buy_retry_reason(error) if is_retryable_buy_rejection(error, side=order.side) else None
        initial_error = str(getattr(error, "__cause__", None) or error)
        final_error = str(error)
        return SubmissionRetryRecord(
            symbol=order.symbol,
            side=order.side.upper(),
            client_order_id=extract_client_order_id(order),
            original_quantity=original_quantity,
            final_quantity=original_quantity,
            retry_used=bool(retry_reason and original_quantity > 1),
            retry_reason=retry_reason,
            initial_error=initial_error,
            final_error=final_error,
            initial_notional=initial_notional,
            final_notional=initial_notional,
            meta={
                "order_type": order.order_type,
            },
        )

    def _build_submission_report(
        self,
        *,
        records: Sequence[SubmissionRetryRecord],
        attempted_orders: int,
        submitted_orders: int,
        failed_orders: int,
    ) -> SubmissionRetryReport:
        return SubmissionRetryReport(
            attempted_orders=attempted_orders,
            submitted_orders=submitted_orders,
            retried_orders=sum(1 for record in records if record.retry_used),
            failed_orders=failed_orders,
            records=tuple(records),
        )

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

    def _estimate_quantity_notional(self, order: OrderIntent, quantity: int) -> float:
        reference_price = self._estimate_order_reference_price(order)
        if reference_price is None:
            return 0.0
        return float(quantity) * float(reference_price)


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
            ledger_open_orders = tuple(ledger.list_open_orders()) if ledger is not None else ()
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
                if ledger_open_orders and self.dependencies.order_status_provider is not None:
                    history_backfill = backfill_order_statuses(
                        ledger,
                        self.dependencies.order_status_provider,
                        candidate_orders=ledger_open_orders,
                        reconciler=self.dependencies.reconciler or PollingOrderReconciler(ledger),
                    )
                    meta["recovery"]["history_backfill"] = history_backfill.to_dict()
                    meta["recovery"]["reconciled_fill_events"] += history_backfill.reconciliation.fill_events_created

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
            prediction_context = self._prediction_context_from_signal_model(signals)
            if prediction_context is not None:
                meta["prediction_context"] = prediction_context
                if manifest is not None:
                    manifest = replace(
                        manifest,
                        meta={
                            **dict(manifest.meta),
                            "prediction_context": prediction_context,
                        },
                    )
                    if ledger is not None:
                        ledger.record_run_manifest(manifest.to_record())
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

            position_symbols = tuple(
                sorted(
                    {
                        str(getattr(position, "symbol", "")).upper()
                        for position in getattr(account, "positions", ())
                        if str(getattr(position, "symbol", "")).strip()
                    }
                )
            )
            bar_symbols = tuple(sorted(set(universe) | set(position_symbols)))
            bars = dict(self.dependencies.market_data_provider.get_bars(runtime_session_date, bar_symbols))
            exit_plan = self._build_position_exit_plan(
                session_date=runtime_session_date,
                account_sync_result=account_sync_result,
                bars=bars,
                available_session_dates=self._available_silver_session_dates() or (),
            )
            meta["position_exit_plan"] = exit_plan["meta"]
            execution_account = self._execution_account_snapshot(account, config.execution_equity_cap)
            if config.execution_equity_cap is not None:
                meta["execution_equity_cap"] = float(config.execution_equity_cap)
            buy_orders = list(
                self.dependencies.execution_policy.generate_orders(
                    runtime_session_date,
                    targets,
                    bars,
                    execution_account,
                )
            )
            exit_orders = list(exit_plan["orders"])
            if not bool(exit_plan["allow_new_buys"]):
                buy_orders = []
            exit_orders, buy_orders, carry_forward_symbols = self._net_rebalance_orders(
                exit_orders=exit_orders,
                buy_orders=buy_orders,
            )
            exit_plan_meta = dict(meta["position_exit_plan"])
            exit_plan_meta["carry_forward_symbols"] = carry_forward_symbols
            meta["position_exit_plan"] = exit_plan_meta
            orders = [*exit_orders, *buy_orders]
            orders = list(self._with_client_order_ids(orders, session_date=runtime_session_date, run_id=run_id))
            counts["orders"] = len(orders)
            counts["exit_orders"] = len(exit_orders)
            counts["buy_orders"] = len(buy_orders)

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
                    submission_retry_report = getattr(self.dependencies.order_submitter, "last_submission_report", None)
                    if submission_retry_report is not None:
                        meta["submission_retry"] = _payload_to_dict(submission_retry_report)
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
                        fill_audits_created = record_fill_audits_for_orders(
                            ledger,
                            candidate_orders=[
                                order_record
                                for order_id in order_contexts
                                for order_record in [ledger.get_order(order_id)]
                                if order_record is not None
                            ],
                            context_overrides=order_contexts,
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
            submission_retry_report = getattr(self.dependencies.order_submitter, "last_submission_report", None)
            if submission_retry_report is not None and "submission_retry" not in meta:
                meta["submission_retry"] = _payload_to_dict(submission_retry_report)
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
            "horizon_bars": int(config.horizon_bars or getattr(self.dependencies.signal_model, "horizon_bars", 0) or 0),
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
        strategy_lineage = self._strategy_lineage_from_config(config)
        if strategy_lineage:
            data_snapshot["strategy_lineage"] = dict(strategy_lineage)
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
                    strategy_project=config.strategy_project,
                ).to_dict(),
                "strategy_profile": config.strategy_profile,
                "strategy_lineage": strategy_lineage,
            },
        )

    def _strategy_lineage_from_config(self, config: PaperRunConfig) -> dict[str, Any]:
        horizon_bars = int(config.horizon_bars or getattr(self.dependencies.signal_model, "horizon_bars", 0) or 0)
        strategy_horizon_bucket = config.strategy_horizon_bucket or (f"h{horizon_bars}" if horizon_bars > 0 else None)
        strategy_family = config.strategy_family or "us_equities"
        strategy_project = config.strategy_project
        if strategy_project in (None, "") and strategy_horizon_bucket not in (None, ""):
            strategy_project = f"{strategy_family}_{strategy_horizon_bucket}"
        lineage: dict[str, Any] = {
            "strategy_horizon_bucket": strategy_horizon_bucket,
            "strategy_family": strategy_family,
            "strategy_project": strategy_project,
            "strategy_track": config.strategy_track,
        }
        return {key: value for key, value in lineage.items() if value not in (None, "")}

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

    def _prediction_context_from_signal_model(self, signals: Sequence[Signal]) -> dict[str, Any] | None:
        signal_model_context = getattr(self.dependencies.signal_model, "last_prediction_context", None)
        if isinstance(signal_model_context, Mapping) and signal_model_context:
            return dict(signal_model_context)

        if not signals:
            return None
        signal_meta = dict(getattr(signals[0], "meta", {}) or {})
        prediction_date = signal_meta.get("prediction_date")
        training_window_start = signal_meta.get("training_window_start")
        training_window_end = signal_meta.get("training_window_end")
        validation_window_start = signal_meta.get("validation_window_start")
        validation_window_end = signal_meta.get("validation_window_end")
        prediction_window_start = signal_meta.get("prediction_window_start")
        prediction_window_end = signal_meta.get("prediction_window_end")
        if prediction_date is None and prediction_window_start is None and prediction_window_end is None:
            return None
        return {
            "prediction_date": prediction_date,
            "training_window": {
                "start": training_window_start,
                "end": training_window_end,
            },
            "validation_window": {
                "start": validation_window_start,
                "end": validation_window_end,
            },
            "prediction_window": {
                "start": prediction_window_start,
                "end": prediction_window_end,
            },
        }

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

    def _build_position_exit_plan(
        self,
        *,
        session_date: date,
        account_sync_result: AccountSyncResult | None,
        bars: Mapping[str, MarketBar],
        available_session_dates: Sequence[date],
    ) -> dict[str, Any]:
        broker_positions = tuple(getattr(account_sync_result, "broker_positions", ()) or ())
        if not broker_positions:
            return {
                "orders": (),
                "allow_new_buys": True,
                "meta": {
                    "current_positions": 0,
                    "mature_symbols": [],
                    "immature_symbols": [],
                    "partial_symbols": [],
                    "unresolved_symbols": [],
                    "rebalance_due": True,
                    "history_source": "none",
                },
            }

        if not available_session_dates:
            return {
                "orders": (),
                "allow_new_buys": False,
                "meta": {
                    "current_positions": len(broker_positions),
                    "mature_symbols": [],
                    "immature_symbols": [],
                    "partial_symbols": [],
                    "unresolved_symbols": [str(getattr(position, "symbol", "")).upper() for position in broker_positions],
                    "rebalance_due": False,
                    "history_source": "missing_session_calendar",
                },
            }

        horizon_bars = int(getattr(self.dependencies.signal_model, "horizon_bars", 5) or 5)
        symbols = tuple(
            sorted(
                {
                    str(getattr(position, "symbol", "")).upper()
                    for position in broker_positions
                    if str(getattr(position, "symbol", "")).strip()
                }
            )
        )
        lots_by_symbol, history_source = self._load_open_position_lots(
            symbols=symbols,
            available_session_dates=available_session_dates,
        )
        session_index = {current: index for index, current in enumerate(available_session_dates)}
        current_index = session_index.get(session_date)
        if current_index is None:
            current_index = max(index for current, index in session_index.items() if current <= session_date)

        exit_orders: list[OrderIntent] = []
        mature_symbols: list[str] = []
        immature_symbols: list[str] = []
        partial_symbols: list[str] = []
        unresolved_symbols: list[str] = []
        timestamp = datetime.combine(session_date, time(9, 30))

        for broker_position in broker_positions:
            symbol = str(getattr(broker_position, "symbol", "")).upper()
            quantity = abs(int(float(getattr(broker_position, "quantity", 0.0) or 0.0)))
            if not symbol or quantity <= 0:
                continue

            open_lots = list(lots_by_symbol.get(symbol, ()))
            if not open_lots:
                unresolved_symbols.append(symbol)
                continue

            mature_lots: list[tuple[int, date, int]] = []
            immature_quantity = 0
            mature_quantity = 0
            tracked_quantity = 0
            for lot_quantity, entry_session_date in open_lots:
                tracked_quantity += int(lot_quantity)
                entry_index = session_index.get(entry_session_date)
                if entry_index is None:
                    unresolved_symbols.append(symbol)
                    mature_lots = []
                    mature_quantity = 0
                    immature_quantity = 0
                    tracked_quantity = 0
                    break
                age_sessions = current_index - entry_index
                if age_sessions >= horizon_bars:
                    mature_lots.append((int(lot_quantity), entry_session_date, age_sessions))
                    mature_quantity += int(lot_quantity)
                else:
                    immature_quantity += int(lot_quantity)

            if tracked_quantity < quantity:
                unresolved_symbols.append(symbol)
                continue

            sell_quantity = min(quantity, mature_quantity)
            remaining_after_sell = max(quantity - sell_quantity, 0)
            if sell_quantity > 0:
                if remaining_after_sell > 0:
                    partial_symbols.append(symbol)
                else:
                    mature_symbols.append(symbol)
                reference_price = self._position_reference_price(symbol=symbol, bars=bars, broker_position=broker_position)
                oldest_entry = min((entry for _, entry, _ in mature_lots), default=session_date)
                max_age_sessions = max((age for _, _, age in mature_lots), default=horizon_bars)
                exit_orders.append(
                    OrderIntent(
                        symbol=symbol,
                        side="SELL",
                        quantity=sell_quantity,
                        order_type="market",
                        limit_price=None,
                        timestamp=timestamp,
                        meta={
                            "reference_price": reference_price,
                            "close": reference_price,
                            "holding_entry_session_date": oldest_entry.isoformat(),
                            "holding_age_sessions": max_age_sessions,
                            "auto_exit": True,
                            "exit_reason": "horizon_exit",
                        },
                    )
                )

            if remaining_after_sell > 0 and symbol not in partial_symbols:
                immature_symbols.append(symbol)
            elif sell_quantity == 0:
                immature_symbols.append(symbol)

        allow_new_buys = not (immature_symbols or partial_symbols or unresolved_symbols)
        return {
            "orders": tuple(exit_orders),
            "allow_new_buys": allow_new_buys,
            "meta": {
                "current_positions": len(broker_positions),
                "mature_symbols": mature_symbols,
                "immature_symbols": immature_symbols,
                "partial_symbols": partial_symbols,
                "unresolved_symbols": unresolved_symbols,
                "rebalance_due": allow_new_buys,
                "history_source": history_source,
            },
        }

    def _load_open_position_lots(
        self,
        *,
        symbols: Sequence[str],
        available_session_dates: Sequence[date],
    ) -> tuple[dict[str, tuple[tuple[int, date], ...]], str]:
        order_payloads = self._list_position_history_orders(symbols=symbols)
        if not order_payloads:
            return {}, "broker_history_empty"

        events_by_symbol: dict[str, list[BrokerOrderSnapshot]] = defaultdict(list)
        for payload in order_payloads:
            snapshot = payload if isinstance(payload, BrokerOrderSnapshot) else BrokerOrderSnapshot.from_payload(payload)
            if not snapshot.symbol or snapshot.filled_quantity <= 0:
                continue
            side = str(snapshot.side).upper()
            if side not in {"BUY", "SELL"}:
                continue
            events_by_symbol[snapshot.symbol.upper()].append(snapshot)

        lots_by_symbol: dict[str, tuple[tuple[int, date], ...]] = {}
        for symbol, events in events_by_symbol.items():
            events.sort(key=lambda item: (item.updated_at_utc, item.order_id))
            lots: deque[tuple[int, date]] = deque()
            for event in events:
                event_session_date = self._resolve_session_date(
                    event.updated_at_utc.date(),
                    available_session_dates=available_session_dates,
                )
                if event_session_date is None:
                    continue
                quantity = int(event.filled_quantity)
                if quantity <= 0:
                    continue
                if str(event.side).upper() == "BUY":
                    lots.append((quantity, event_session_date))
                    continue

                remaining = quantity
                while remaining > 0 and lots:
                    lot_quantity, lot_session_date = lots[0]
                    if lot_quantity <= remaining:
                        remaining -= lot_quantity
                        lots.popleft()
                    else:
                        lots[0] = (lot_quantity - remaining, lot_session_date)
                        remaining = 0
            lots_by_symbol[symbol] = tuple(lots)
        return lots_by_symbol, "broker_order_history"

    def _list_position_history_orders(self, *, symbols: Sequence[str]) -> tuple[Mapping[str, Any] | object, ...]:
        provider = self.dependencies.open_order_provider
        if provider is None or not symbols:
            return ()
        list_orders = getattr(provider, "list_orders", None)
        if not callable(list_orders):
            return ()

        attempts = (
            {"status": "all", "symbols": list(symbols), "limit": 500, "direction": "desc"},
            {"status": "all", "symbols": list(symbols), "limit": 500},
            {"status": "all", "symbols": list(symbols)},
        )
        for kwargs in attempts:
            try:
                payload = list_orders(**kwargs)
                return tuple(payload)
            except TypeError:
                continue
        return ()

    def _resolve_session_date(
        self,
        candidate: date,
        *,
        available_session_dates: Sequence[date],
    ) -> date | None:
        not_after = [current for current in available_session_dates if current <= candidate]
        if not not_after:
            return None
        return not_after[-1]

    def _position_reference_price(
        self,
        *,
        symbol: str,
        bars: Mapping[str, MarketBar],
        broker_position: object,
    ) -> float | None:
        bar = bars.get(symbol)
        if bar is not None:
            return float(bar.adj_open) if bar.adj_open is not None else float(bar.open)
        current_price = getattr(broker_position, "current_price", None)
        if current_price is None:
            current_price = getattr(broker_position, "avg_entry_price", None)
        try:
            price = float(current_price)
        except (TypeError, ValueError):
            return None
        return price if price > 0 else None

    def _net_rebalance_orders(
        self,
        *,
        exit_orders: Sequence[OrderIntent],
        buy_orders: Sequence[OrderIntent],
    ) -> tuple[list[OrderIntent], list[OrderIntent], list[str]]:
        sell_by_symbol = {
            order.symbol.upper()
            for order in exit_orders
            if str(order.side).upper() == "SELL"
        }
        buy_by_symbol = {
            order.symbol.upper()
            for order in buy_orders
            if str(order.side).upper() == "BUY"
        }
        carry_forward_symbols = sorted(sell_by_symbol & buy_by_symbol)
        if not carry_forward_symbols:
            return list(exit_orders), list(buy_orders), []

        carry_forward = set(carry_forward_symbols)
        filtered_exit_orders = [
            order
            for order in exit_orders
            if not (str(order.side).upper() == "SELL" and order.symbol.upper() in carry_forward)
        ]
        filtered_buy_orders = [
            order
            for order in buy_orders
            if not (str(order.side).upper() == "BUY" and order.symbol.upper() in carry_forward)
        ]
        return filtered_exit_orders, filtered_buy_orders, carry_forward_symbols

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

    def _estimate_quantity_notional(self, order: OrderIntent, quantity: int) -> float:
        reference_price = self._estimate_order_reference_price(order)
        if reference_price is None:
            return 0.0
        return float(quantity) * float(reference_price)

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
    parser.add_argument("--strategy-profile", default=None)
    parser.add_argument("--strategy-family", default=None)
    parser.add_argument("--strategy-project", default=None)
    parser.add_argument("--strategy-horizon-bucket", default=None)
    parser.add_argument("--strategy-track", default=None)
    parser.add_argument("--execute", action="store_true", help="Submit orders instead of dry-run mode.")
    parser.add_argument("--output-format", choices=("json",), default="json")
    parser.add_argument("--demo-mode", action="store_true", help="Use the static no-op dependencies.")
    parser.add_argument("--model", default="hist_gbm", choices=list_alpha_expert_names())
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
        horizon_bars=5,
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
    resolved_model_name = assert_supported_alpha_expert(model_name)
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
            model_name=resolved_model_name,
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
        execution_policy=SameSessionMarketOrderExecutionPolicy(),
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
        run_name=f"{resolved_model_name}-paper-demo",
        horizon_bars=horizon,
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
            horizon_bars=args.horizon,
            strategy_profile=getattr(args, "strategy_profile", None),
            strategy_family=getattr(args, "strategy_family", None),
            strategy_project=getattr(args, "strategy_project", None),
            strategy_horizon_bucket=getattr(args, "strategy_horizon_bucket", None),
            strategy_track=getattr(args, "strategy_track", None),
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
