from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

import pandas as pd

from stockmachine.apps.run_us_equities_paper import (
    LatestSilverUniverseProvider,
    SilverDatasetCache,
    SilverWalkForwardSignalModel,
)
from stockmachine.backtest.protocols import AccountSnapshot
from stockmachine.data.vendors import AlpacaCredentials, AlpacaHttpClient
from stockmachine.domain.models import TargetPosition
from stockmachine.execution.brokers import AlpacaTradingAdapter, BrokerAccount, BrokerClock, BrokerOrder, BrokerPosition
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.live.account_sync import AlpacaAccountSync
from stockmachine.portfolio import RiskAwareTopKPortfolioPolicy

_DEFAULT_REBALANCE_INTERVAL_SESSIONS = 5
_DEFAULT_BAR_LOOKBACK_DAYS = 21
_DEFAULT_MIN_TRADE_NOTIONAL = 25.0


class TradingBroker(Protocol):
    def is_paper_trading_environment(self) -> bool: ...
    def get_account(self) -> BrokerAccount: ...
    def get_clock(self) -> BrokerClock: ...
    def list_positions(self) -> Sequence[BrokerPosition]: ...
    def list_orders(self, *, status: str | None = None, symbols: Sequence[str] | None = None) -> Sequence[BrokerOrder]: ...
    def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float | None = None,
        notional: float | None = None,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        client_order_id: str | None = None,
        extended_hours: bool | None = None,
    ) -> BrokerOrder: ...


class MarketDataClient(Protocol):
    def list_assets(self, *, status: str = "active", asset_class: str = "us_equity") -> list[dict[str, Any]]: ...
    def get_stock_bars(
        self,
        *,
        symbols: list[str],
        start: str,
        end: str,
        timeframe: str = "1Day",
        adjustment: str = "raw",
        feed: str = "iex",
        limit: int = 10000,
        asof: str | None = None,
    ) -> list[dict[str, Any]]: ...


@dataclass(slots=True, frozen=True)
class HybridPaperConfig:
    profile_id: str
    run_name: str
    model_name: str
    top_k: int
    horizon: int
    min_close: float
    min_median_dollar_volume_20: float
    max_vol_20: float
    max_positions_per_sector: int
    sector_neutral: bool
    core_weights: tuple[tuple[str, float], ...]
    sleeve_weight: float
    rebalance_interval_sessions: int = _DEFAULT_REBALANCE_INTERVAL_SESSIONS
    min_trade_notional: float = _DEFAULT_MIN_TRADE_NOTIONAL
    market_data_feed: str = "iex"

    @property
    def core_symbols(self) -> tuple[str, ...]:
        return tuple(symbol for symbol, _ in self.core_weights)


@dataclass(slots=True, frozen=True)
class HybridState:
    last_rebalance_effective_session_date: date | None = None
    last_rebalance_run_id: str | None = None
    last_rebalance_submitted_order_count: int = 0
    updated_at_utc: str | None = None


@dataclass(slots=True, frozen=True)
class SleeveSelection:
    universe_size: int
    signal_count: int
    target_count: int
    selected_symbols: tuple[str, ...]
    targets: tuple[TargetPosition, ...]
    prediction_context: Mapping[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class HybridOrderPlanEntry:
    symbol: str
    side: str
    reference_price: float
    current_market_value: float
    target_weight: float
    target_notional: float
    delta_notional: float
    quantity: float | None = None
    notional: float | None = None
    asset_fractionable: bool = False
    asset_tradable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "reference_price": self.reference_price,
            "current_market_value": self.current_market_value,
            "target_weight": self.target_weight,
            "target_notional": self.target_notional,
            "delta_notional": self.delta_notional,
            "quantity": self.quantity,
            "notional": self.notional,
            "asset_fractionable": self.asset_fractionable,
            "asset_tradable": self.asset_tradable,
        }


@dataclass(slots=True, frozen=True)
class HybridPlan:
    effective_session_date: date
    rebalance_due: bool
    sessions_since_last_rebalance: int | None
    last_rebalance_effective_session_date: date | None
    next_rebalance_effective_session_date: date | None
    sleeve_selection: SleeveSelection
    combined_target_weights: Mapping[str, float]
    latest_prices: Mapping[str, float]
    open_orders: tuple[BrokerOrder, ...]
    orders: tuple[HybridOrderPlanEntry, ...]
    account_snapshot: AccountSnapshot
    broker_account: BrokerAccount
    broker_positions: tuple[BrokerPosition, ...]
    clock: BrokerClock
    warnings: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class HybridRunResult:
    ok: bool
    stage: str
    run_id: str
    execute_requested: bool
    session_date: date
    effective_session_date: date
    rebalance_due: bool
    sessions_since_last_rebalance: int | None
    last_rebalance_effective_session_date: date | None
    next_rebalance_effective_session_date: date | None
    account: Mapping[str, Any]
    market: Mapping[str, Any]
    counts: Mapping[str, Any]
    core_targets: tuple[Mapping[str, Any], ...]
    sleeve_targets: tuple[Mapping[str, Any], ...]
    combined_targets: tuple[Mapping[str, Any], ...]
    planned_orders: tuple[Mapping[str, Any], ...]
    submitted_orders: tuple[Mapping[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    error: Mapping[str, Any] | None = None
    prediction_context: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "stage": self.stage,
            "run_id": self.run_id,
            "execute_requested": self.execute_requested,
            "session_date": self.session_date.isoformat(),
            "effective_session_date": self.effective_session_date.isoformat(),
            "rebalance_due": self.rebalance_due,
            "sessions_since_last_rebalance": self.sessions_since_last_rebalance,
            "last_rebalance_effective_session_date": (
                self.last_rebalance_effective_session_date.isoformat()
                if self.last_rebalance_effective_session_date is not None
                else None
            ),
            "next_rebalance_effective_session_date": (
                self.next_rebalance_effective_session_date.isoformat()
                if self.next_rebalance_effective_session_date is not None
                else None
            ),
            "account": dict(self.account),
            "market": dict(self.market),
            "counts": dict(self.counts),
            "core_targets": [dict(row) for row in self.core_targets],
            "sleeve_targets": [dict(row) for row in self.sleeve_targets],
            "combined_targets": [dict(row) for row in self.combined_targets],
            "planned_orders": [dict(row) for row in self.planned_orders],
            "submitted_orders": [dict(row) for row in self.submitted_orders],
            "warnings": list(self.warnings),
        }
        if self.error is not None:
            payload["error"] = dict(self.error)
        if self.prediction_context is not None:
            payload["prediction_context"] = dict(self.prediction_context)
        return payload


@dataclass(slots=True)
class HybridSleevePlanner:
    config: HybridPaperConfig
    dataset_cache: SilverDatasetCache

    def plan(self, session_date: date, *, account_snapshot: AccountSnapshot) -> SleeveSelection:
        effective_session_date = self.dataset_cache.resolve_session_date(session_date)
        universe = tuple(LatestSilverUniverseProvider(self.dataset_cache).get_universe(effective_session_date))
        signal_model = SilverWalkForwardSignalModel(
            dataset_cache=self.dataset_cache,
            model_name=self.config.model_name,
            horizon_bars=self.config.horizon,
        )
        signals = tuple(signal_model.predict(effective_session_date, universe))
        policy = RiskAwareTopKPortfolioPolicy(
            top_k=self.config.top_k,
            min_close=self.config.min_close,
            min_median_dollar_volume_20=self.config.min_median_dollar_volume_20,
            max_vol_20=self.config.max_vol_20,
            max_positions_per_sector=self.config.max_positions_per_sector,
            sector_neutral=self.config.sector_neutral,
        )
        targets = tuple(policy.build_targets(effective_session_date, list(signals), account_snapshot))
        return SleeveSelection(
            universe_size=len(universe),
            signal_count=len(signals),
            target_count=len(targets),
            selected_symbols=tuple(target.symbol for target in targets),
            targets=targets,
            prediction_context=signal_model.last_prediction_context,
        )


@dataclass(slots=True)
class HybridPaperRunner:
    config: HybridPaperConfig
    broker: TradingBroker
    market_data: MarketDataClient
    dataset_cache: SilverDatasetCache
    sleeve_planner: HybridSleevePlanner

    def run(
        self,
        *,
        session_date: date,
        execute: bool,
        state_path: str | Path | None = None,
    ) -> HybridRunResult:
        run_id = uuid4().hex
        try:
            if not self.broker.is_paper_trading_environment():
                raise RuntimeError("Hybrid paper runner refuses to use a non-paper Alpaca trading endpoint.")

            state = load_hybrid_state(state_path)
            plan = self._build_plan(session_date=session_date, state=state)
            if execute and plan.rebalance_due and not plan.sleeve_selection.targets:
                return self._blocked_result(
                    run_id=run_id,
                    session_date=session_date,
                    plan=plan,
                    reason="missing_sleeve_targets",
                    message="No sleeve targets were generated on a due rebalance date; refusing to advance hybrid state.",
                )
            if execute and plan.rebalance_due and any(
                str(warning).startswith("Missing latest prices") for warning in plan.warnings
            ):
                return self._blocked_result(
                    run_id=run_id,
                    session_date=session_date,
                    plan=plan,
                    reason="missing_latest_prices",
                    message="Missing latest prices for one or more required symbols; refusing to submit incomplete hybrid orders.",
                )
            if execute and plan.open_orders:
                return self._blocked_result(
                    run_id=run_id,
                    session_date=session_date,
                    plan=plan,
                    reason="open_orders_present",
                    message="Open broker orders are already present for this account; refusing to submit duplicates.",
                )

            submitted_orders: tuple[Mapping[str, Any], ...] = ()
            stage = "planned"
            if execute and plan.rebalance_due and plan.orders:
                submitted_orders = self._submit_orders(plan.orders, run_id=run_id)
                stage = "executed"
            elif execute and plan.rebalance_due and not plan.orders:
                stage = "executed_no_orders"
            elif execute and not plan.rebalance_due:
                stage = "skipped_not_due"

            if execute and plan.rebalance_due:
                save_hybrid_state(
                    state_path,
                    HybridState(
                        last_rebalance_effective_session_date=plan.effective_session_date,
                        last_rebalance_run_id=run_id,
                        last_rebalance_submitted_order_count=len(submitted_orders),
                        updated_at_utc=datetime.now(timezone.utc).isoformat(),
                    ),
                )

            return build_hybrid_run_result(
                config=self.config,
                run_id=run_id,
                session_date=session_date,
                execute_requested=execute,
                plan=plan,
                stage=stage,
                submitted_orders=submitted_orders,
            )
        except Exception as exc:
            return HybridRunResult(
                ok=False,
                stage="failed",
                run_id=run_id,
                execute_requested=execute,
                session_date=session_date,
                effective_session_date=session_date,
                rebalance_due=False,
                sessions_since_last_rebalance=None,
                last_rebalance_effective_session_date=None,
                next_rebalance_effective_session_date=None,
                account={},
                market={},
                counts={},
                core_targets=(),
                sleeve_targets=(),
                combined_targets=(),
                planned_orders=(),
                warnings=(),
                error={"type": type(exc).__name__, "message": str(exc)},
            )

    def _build_plan(self, *, session_date: date, state: HybridState) -> HybridPlan:
        sync_result = AlpacaAccountSync(self.broker).sync(session_date=session_date)
        effective_session_date = self.dataset_cache.resolve_session_date(session_date)
        available_session_dates = load_available_session_dates(self.dataset_cache)
        rebalance_state = compute_rebalance_state(
            effective_session_date=effective_session_date,
            available_session_dates=available_session_dates,
            last_rebalance_effective_session_date=state.last_rebalance_effective_session_date,
            has_existing_positions=bool(sync_result.broker_positions),
            interval_sessions=self.config.rebalance_interval_sessions,
        )
        sleeve_selection = self.sleeve_planner.plan(
            effective_session_date,
            account_snapshot=sync_result.snapshot,
        )
        target_weights = build_combined_target_weights(
            core_weights=self.config.core_weights,
            sleeve_targets=sleeve_selection.targets,
            sleeve_weight=self.config.sleeve_weight,
        )
        target_symbols = tuple(
            sorted(set(target_weights) | {position.symbol.upper() for position in sync_result.broker_positions})
        )
        latest_prices = fetch_latest_prices(
            self.market_data,
            symbols=target_symbols,
            session_date=session_date,
            lookback_days=_DEFAULT_BAR_LOOKBACK_DAYS,
            feed=self.config.market_data_feed,
        )
        asset_metadata = load_asset_metadata(self.market_data, symbols=target_symbols)
        open_orders = tuple(self.broker.list_orders(status="open"))
        warnings: list[str] = []
        if not sleeve_selection.targets:
            warnings.append("No sleeve targets were generated for the requested session.")
        missing_prices = sorted(symbol for symbol in target_symbols if symbol not in latest_prices)
        if missing_prices:
            warnings.append(f"Missing latest prices for: {', '.join(missing_prices)}")

        orders: tuple[HybridOrderPlanEntry, ...] = ()
        if rebalance_state["rebalance_due"] and not missing_prices and sleeve_selection.targets:
            orders = tuple(
                build_order_plan(
                    account_equity=float(sync_result.broker_account.equity),
                    broker_positions=sync_result.broker_positions,
                    target_weights=target_weights,
                    latest_prices=latest_prices,
                    asset_metadata=asset_metadata,
                    min_trade_notional=self.config.min_trade_notional,
                )
            )

        return HybridPlan(
            effective_session_date=effective_session_date,
            rebalance_due=bool(rebalance_state["rebalance_due"]),
            sessions_since_last_rebalance=rebalance_state["sessions_since_last_rebalance"],
            last_rebalance_effective_session_date=rebalance_state["last_rebalance_effective_session_date"],
            next_rebalance_effective_session_date=rebalance_state["next_rebalance_effective_session_date"],
            sleeve_selection=sleeve_selection,
            combined_target_weights=target_weights,
            latest_prices=latest_prices,
            open_orders=open_orders,
            orders=orders,
            account_snapshot=sync_result.snapshot,
            broker_account=sync_result.broker_account,
            broker_positions=sync_result.broker_positions,
            clock=sync_result.clock,
            warnings=tuple(warnings),
        )

    def _submit_orders(
        self,
        orders: Sequence[HybridOrderPlanEntry],
        *,
        run_id: str,
    ) -> tuple[Mapping[str, Any], ...]:
        submitted: list[Mapping[str, Any]] = []
        sell_orders = [order for order in orders if order.side == "sell"]
        buy_orders = [order for order in orders if order.side == "buy"]
        for index, order in enumerate([*sell_orders, *buy_orders], start=1):
            client_order_id = f"cs-{run_id[:10]}-{index:03d}-{order.symbol.lower()}"
            broker_order = self.broker.submit_order(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                notional=order.notional,
                order_type="market",
                time_in_force="day",
                client_order_id=client_order_id,
            )
            submitted.append(
                {
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "notional": order.notional,
                    "client_order_id": client_order_id,
                    "order_id": broker_order.order_id,
                    "status": broker_order.status,
                    "time_in_force": broker_order.time_in_force,
                    "order_type": broker_order.order_type,
                }
            )
        return tuple(submitted)

    def _blocked_result(
        self,
        *,
        run_id: str,
        session_date: date,
        plan: HybridPlan,
        reason: str,
        message: str,
    ) -> HybridRunResult:
        base = build_hybrid_run_result(
            config=self.config,
            run_id=run_id,
            session_date=session_date,
            execute_requested=True,
            plan=plan,
            stage="blocked",
            submitted_orders=(),
        )
        return HybridRunResult(
            ok=False,
            stage=base.stage,
            run_id=base.run_id,
            execute_requested=base.execute_requested,
            session_date=base.session_date,
            effective_session_date=base.effective_session_date,
            rebalance_due=base.rebalance_due,
            sessions_since_last_rebalance=base.sessions_since_last_rebalance,
            last_rebalance_effective_session_date=base.last_rebalance_effective_session_date,
            next_rebalance_effective_session_date=base.next_rebalance_effective_session_date,
            account=base.account,
            market=base.market,
            counts=base.counts,
            core_targets=base.core_targets,
            sleeve_targets=base.sleeve_targets,
            combined_targets=base.combined_targets,
            planned_orders=base.planned_orders,
            submitted_orders=(),
            warnings=base.warnings,
            prediction_context=base.prediction_context,
            error={"type": reason, "message": message},
        )


def load_hybrid_config(path: str | Path) -> HybridPaperConfig:
    payload = _load_json_file(path)
    core_policy = dict(payload.get("core_policy", {}))
    alpha_sleeve = dict(payload.get("alpha_sleeve", {}))
    core_weights_map = core_policy.get("weights_at_total_portfolio_level") or _expand_core_weights(core_policy)
    if not core_weights_map:
        raise ValueError("Missing core policy weights in hybrid experiment config.")
    return HybridPaperConfig(
        profile_id=str(payload.get("profile_id") or payload.get("run_name") or "core-sleeve-paper"),
        run_name=str(payload.get("run_name") or payload.get("profile_id") or "core-sleeve-paper"),
        model_name=str(alpha_sleeve.get("name") or payload.get("model") or "lightgbm_ranker"),
        top_k=int(alpha_sleeve.get("top_k") or payload.get("top_k") or 10),
        horizon=int(alpha_sleeve.get("horizon") or payload.get("horizon") or 5),
        min_close=float(alpha_sleeve.get("min_close") or payload.get("min_close") or 10.0),
        min_median_dollar_volume_20=float(
            alpha_sleeve.get("min_median_dollar_volume_20")
            or payload.get("min_median_dollar_volume_20")
            or 30_000_000.0
        ),
        max_vol_20=float(alpha_sleeve.get("max_vol_20") or payload.get("max_vol_20") or 0.065),
        max_positions_per_sector=int(
            alpha_sleeve.get("max_positions_per_sector") or payload.get("max_positions_per_sector") or 2
        ),
        sector_neutral=bool(alpha_sleeve.get("sector_neutral", not bool(payload.get("disable_sector_neutral", False)))),
        core_weights=tuple(
            (str(symbol).upper(), float(weight))
            for symbol, weight in dict(core_weights_map).items()
        ),
        sleeve_weight=float(alpha_sleeve.get("allocation_weight") or 0.06),
        rebalance_interval_sessions=int(payload.get("rebalance_interval_sessions") or _DEFAULT_REBALANCE_INTERVAL_SESSIONS),
        min_trade_notional=float(payload.get("min_trade_notional") or _DEFAULT_MIN_TRADE_NOTIONAL),
        market_data_feed=str(payload.get("market_data_feed") or "iex"),
    )


def _expand_core_weights(core_policy: Mapping[str, Any]) -> dict[str, float]:
    within_core = dict(core_policy.get("weights_within_core") or {})
    allocation_weight = float(core_policy.get("allocation_weight") or 1.0)
    return {
        str(symbol).upper(): allocation_weight * float(weight)
        for symbol, weight in within_core.items()
    }


def load_credentials_from_json(path: str | Path) -> AlpacaCredentials:
    payload = _load_json_file(path)
    api_key_id = str(payload.get("alpaca_api_key_id") or "").strip()
    api_secret_key = str(payload.get("alpaca_api_secret_key") or "").strip()
    if not api_key_id or not api_secret_key:
        raise RuntimeError(f"Missing Alpaca credentials in {Path(path)}.")
    return AlpacaCredentials(
        api_key_id=api_key_id,
        api_secret_key=api_secret_key,
        trading_base_url=str(payload.get("alpaca_trading_base_url") or "https://paper-api.alpaca.markets").rstrip("/"),
        data_base_url=str(payload.get("alpaca_data_base_url") or "https://data.alpaca.markets").rstrip("/"),
        request_timeout_seconds=float(payload.get("alpaca_http_timeout_seconds") or 60.0),
        max_retries=int(payload.get("alpaca_http_max_retries") or 2),
    )


def load_available_session_dates(dataset_cache: SilverDatasetCache) -> tuple[date, ...]:
    dataset = dataset_cache.load()
    daily_bar = dataset["daily_bar"]
    if daily_bar.empty:
        return ()
    return tuple(sorted(pd.to_datetime(daily_bar["session_date"]).dt.date.unique()))


def load_hybrid_state(path: str | Path | None) -> HybridState:
    if path is None:
        return HybridState()
    state_path = Path(path)
    if not state_path.exists():
        return HybridState()
    payload = _load_json_file(state_path)
    raw_date = payload.get("last_rebalance_effective_session_date")
    return HybridState(
        last_rebalance_effective_session_date=(date.fromisoformat(str(raw_date)) if raw_date else None),
        last_rebalance_run_id=payload.get("last_rebalance_run_id"),
        last_rebalance_submitted_order_count=int(payload.get("last_rebalance_submitted_order_count") or 0),
        updated_at_utc=payload.get("updated_at_utc"),
    )


def save_hybrid_state(path: str | Path | None, state: HybridState) -> None:
    if path is None:
        return
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(asdict(state), indent=2, default=str), encoding="utf-8")


def _load_json_file(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {Path(path)}.")
    return payload


def compute_rebalance_state(
    *,
    effective_session_date: date,
    available_session_dates: Sequence[date],
    last_rebalance_effective_session_date: date | None,
    has_existing_positions: bool,
    interval_sessions: int,
) -> dict[str, Any]:
    if not available_session_dates:
        return {
            "rebalance_due": True,
            "sessions_since_last_rebalance": None,
            "last_rebalance_effective_session_date": last_rebalance_effective_session_date,
            "next_rebalance_effective_session_date": None,
        }
    session_index = {current: index for index, current in enumerate(available_session_dates)}
    current_index = session_index.get(effective_session_date)
    if current_index is None:
        candidates = [index for current, index in session_index.items() if current <= effective_session_date]
        current_index = max(candidates) if candidates else len(available_session_dates) - 1
    if last_rebalance_effective_session_date is None:
        return {
            "rebalance_due": True,
            "sessions_since_last_rebalance": None if not has_existing_positions else len(available_session_dates),
            "last_rebalance_effective_session_date": None,
            "next_rebalance_effective_session_date": effective_session_date,
        }
    last_index = session_index.get(last_rebalance_effective_session_date)
    if last_index is None:
        return {
            "rebalance_due": True,
            "sessions_since_last_rebalance": None,
            "last_rebalance_effective_session_date": last_rebalance_effective_session_date,
            "next_rebalance_effective_session_date": effective_session_date,
        }
    sessions_since = max(0, current_index - last_index)
    next_index = last_index + interval_sessions
    return {
        "rebalance_due": sessions_since >= interval_sessions,
        "sessions_since_last_rebalance": sessions_since,
        "last_rebalance_effective_session_date": last_rebalance_effective_session_date,
        "next_rebalance_effective_session_date": (
            available_session_dates[next_index] if next_index < len(available_session_dates) else None
        ),
    }


def build_combined_target_weights(
    *,
    core_weights: Sequence[tuple[str, float]],
    sleeve_targets: Sequence[TargetPosition],
    sleeve_weight: float,
) -> dict[str, float]:
    weights = {str(symbol).upper(): float(weight) for symbol, weight in core_weights}
    for target in sleeve_targets:
        symbol = str(target.symbol).upper()
        weights[symbol] = weights.get(symbol, 0.0) + float(target.target_weight) * float(sleeve_weight)
    return dict(sorted(weights.items()))


def fetch_latest_prices(
    client: MarketDataClient,
    *,
    symbols: Sequence[str],
    session_date: date,
    lookback_days: int,
    feed: str,
) -> dict[str, float]:
    if not symbols:
        return {}
    bars = client.get_stock_bars(
        symbols=list(symbols),
        start=(session_date - timedelta(days=lookback_days)).isoformat(),
        end=session_date.isoformat(),
        feed=feed,
        adjustment="raw",
    )
    if not bars:
        return {}
    frame = pd.DataFrame(bars)
    if frame.empty or "symbol" not in frame.columns:
        return {}
    frame["_timestamp"] = pd.to_datetime(frame["t"], errors="coerce")
    frame = frame.dropna(subset=["_timestamp"]).sort_values(["symbol", "_timestamp"])
    latest = frame.drop_duplicates(subset=["symbol"], keep="last")
    return {
        str(getattr(row, "symbol")).upper(): float(getattr(row, "c"))
        for row in latest.itertuples(index=False)
        if float(getattr(row, "c")) > 0
    }


def load_asset_metadata(client: MarketDataClient, *, symbols: Sequence[str]) -> dict[str, dict[str, Any]]:
    symbols_upper = {str(symbol).upper() for symbol in symbols}
    assets = client.list_assets(status="active", asset_class="us_equity")
    return {
        str(asset.get("symbol")).upper(): {
            "tradable": bool(asset.get("tradable", True)),
            "fractionable": bool(asset.get("fractionable", False)),
        }
        for asset in assets
        if isinstance(asset, dict) and str(asset.get("symbol")).upper() in symbols_upper
    }


def build_order_plan(
    *,
    account_equity: float,
    broker_positions: Sequence[BrokerPosition],
    target_weights: Mapping[str, float],
    latest_prices: Mapping[str, float],
    asset_metadata: Mapping[str, Mapping[str, Any]],
    min_trade_notional: float,
) -> list[HybridOrderPlanEntry]:
    current_positions = {position.symbol.upper(): position for position in broker_positions}
    tracked_symbols = sorted(set(target_weights) | set(current_positions))
    orders: list[HybridOrderPlanEntry] = []
    for symbol in tracked_symbols:
        target_weight = float(target_weights.get(symbol, 0.0))
        target_notional = account_equity * target_weight
        position = current_positions.get(symbol)
        current_market_value = abs(float(getattr(position, "market_value", 0.0) or 0.0))
        delta_notional = target_notional - current_market_value
        reference_price = float(latest_prices.get(symbol) or getattr(position, "current_price", 0.0) or 0.0)
        if reference_price <= 0 or abs(delta_notional) < float(min_trade_notional):
            continue
        asset_info = dict(asset_metadata.get(symbol, {}))
        tradable = bool(asset_info.get("tradable", True))
        fractionable = bool(asset_info.get("fractionable", False))
        if not tradable:
            continue
        if delta_notional < 0:
            current_quantity = abs(float(getattr(position, "quantity", 0.0) or 0.0))
            quantity = min(current_quantity, _round_share_quantity(abs(delta_notional) / reference_price))
            if quantity <= 0:
                continue
            orders.append(
                HybridOrderPlanEntry(
                    symbol=symbol,
                    side="sell",
                    reference_price=reference_price,
                    current_market_value=current_market_value,
                    target_weight=target_weight,
                    target_notional=target_notional,
                    delta_notional=delta_notional,
                    quantity=quantity,
                    asset_fractionable=fractionable,
                    asset_tradable=tradable,
                )
            )
        else:
            notional = round(float(delta_notional), 2)
            quantity = None
            if not fractionable:
                quantity = math.floor(float(delta_notional) / reference_price)
                if quantity <= 0:
                    continue
                notional = None
            orders.append(
                HybridOrderPlanEntry(
                    symbol=symbol,
                    side="buy",
                    reference_price=reference_price,
                    current_market_value=current_market_value,
                    target_weight=target_weight,
                    target_notional=target_notional,
                    delta_notional=delta_notional,
                    quantity=quantity,
                    notional=notional,
                    asset_fractionable=fractionable,
                    asset_tradable=tradable,
                )
            )
    sells = sorted((order for order in orders if order.side == "sell"), key=lambda item: abs(item.delta_notional), reverse=True)
    buys = sorted((order for order in orders if order.side == "buy"), key=lambda item: abs(item.delta_notional), reverse=True)
    return [*sells, *buys]


def _round_share_quantity(value: float) -> float:
    if value <= 0:
        return 0.0
    if value >= 1.0:
        return float(math.floor(value))
    return round(value, 6)


def build_hybrid_run_result(
    *,
    config: HybridPaperConfig,
    run_id: str,
    session_date: date,
    execute_requested: bool,
    plan: HybridPlan,
    stage: str,
    submitted_orders: Sequence[Mapping[str, Any]],
) -> HybridRunResult:
    core_targets = tuple(
        {
            "symbol": symbol,
            "target_weight": float(weight),
            "target_notional": float(plan.broker_account.equity) * float(weight),
        }
        for symbol, weight in config.core_weights
    )
    sleeve_targets = tuple(
        {
            "symbol": target.symbol,
            "target_weight_within_sleeve": float(target.target_weight),
            "target_weight_at_total_portfolio_level": float(target.target_weight) * config.sleeve_weight,
            "adjusted_score": target.meta.get("adjusted_score"),
            "sector": target.meta.get("sector"),
            "industry": target.meta.get("industry"),
        }
        for target in plan.sleeve_selection.targets
    )
    combined_targets = tuple(
        {
            "symbol": symbol,
            "target_weight": float(plan.combined_target_weights[symbol]),
            "target_notional": float(plan.broker_account.equity) * float(plan.combined_target_weights[symbol]),
            "reference_price": float(plan.latest_prices.get(symbol, 0.0)),
        }
        for symbol in sorted(plan.combined_target_weights)
    )
    return HybridRunResult(
        ok=True,
        stage=stage,
        run_id=run_id,
        execute_requested=execute_requested,
        session_date=session_date,
        effective_session_date=plan.effective_session_date,
        rebalance_due=plan.rebalance_due,
        sessions_since_last_rebalance=plan.sessions_since_last_rebalance,
        last_rebalance_effective_session_date=plan.last_rebalance_effective_session_date,
        next_rebalance_effective_session_date=plan.next_rebalance_effective_session_date,
        account={
            "equity": float(plan.broker_account.equity),
            "cash": float(plan.broker_account.cash),
            "buying_power": float(plan.broker_account.buying_power),
            "portfolio_value": float(plan.broker_account.portfolio_value),
            "position_count": len(plan.broker_positions),
            "open_order_count": len(plan.open_orders),
        },
        market={
            "timestamp": plan.clock.timestamp.isoformat(),
            "is_open": bool(plan.clock.is_open),
            "next_open": plan.clock.next_open.isoformat() if plan.clock.next_open is not None else None,
            "next_close": plan.clock.next_close.isoformat() if plan.clock.next_close is not None else None,
        },
        counts={
            "core_targets": len(core_targets),
            "sleeve_signals": plan.sleeve_selection.signal_count,
            "sleeve_targets": plan.sleeve_selection.target_count,
            "combined_targets": len(combined_targets),
            "planned_orders": len(plan.orders),
            "planned_buy_orders": sum(1 for order in plan.orders if order.side == "buy"),
            "planned_sell_orders": sum(1 for order in plan.orders if order.side == "sell"),
            "submitted_orders": len(submitted_orders),
            "open_orders": len(plan.open_orders),
        },
        core_targets=core_targets,
        sleeve_targets=sleeve_targets,
        combined_targets=combined_targets,
        planned_orders=tuple(order.to_dict() for order in plan.orders),
        submitted_orders=tuple(dict(order) for order in submitted_orders),
        warnings=plan.warnings,
        prediction_context=plan.sleeve_selection.prediction_context,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a hybrid fixed-weight core plus stock-alpha sleeve paper strategy.")
    parser.add_argument("--config-path", required=True, help="Path to the hybrid experiment JSON config.")
    parser.add_argument("--credentials-path", required=True, help="Path to the version-local Alpaca credentials JSON.")
    parser.add_argument("--session-date", default=date.today().isoformat(), help="Requested trading date in YYYY-MM-DD.")
    parser.add_argument("--state-path", default=None, help="Path to the hybrid runtime state JSON.")
    parser.add_argument("--data-root", default="data", help="Research data root for silver inputs.")
    parser.add_argument("--execute", action="store_true", help="Submit live paper orders instead of only planning.")
    return parser


def parse_session_date(raw_value: str) -> date:
    return date.fromisoformat(raw_value)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_hybrid_config(args.config_path)
    credentials = load_credentials_from_json(args.credentials_path)
    dataset_cache = SilverDatasetCache(layout=StorageLayout(root=Path(args.data_root)))
    planner = HybridSleevePlanner(config=config, dataset_cache=dataset_cache)
    runner = HybridPaperRunner(
        config=config,
        broker=AlpacaTradingAdapter(credentials=credentials),
        market_data=AlpacaHttpClient(credentials),
        dataset_cache=dataset_cache,
        sleeve_planner=planner,
    )
    result = runner.run(
        session_date=parse_session_date(args.session_date),
        execute=bool(args.execute),
        state_path=args.state_path,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
