from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from stockmachine.apps.run_core_sleeve_paper import (
    HybridPaperConfig,
    HybridPaperRunner,
    HybridState,
    HybridOrderPlanEntry,
    SleeveSelection,
    build_combined_target_weights,
    build_order_plan,
    compute_rebalance_state,
    load_hybrid_config,
)
from stockmachine.backtest.protocols import AccountSnapshot
from stockmachine.domain.models import TargetPosition
from stockmachine.execution.brokers import BrokerAccount, BrokerClock, BrokerOrder, BrokerPosition


class _FakeDatasetCache:
    def __init__(self, sessions: list[date]) -> None:
        self._sessions = sessions

    def load(self) -> dict[str, object]:
        import pandas as pd

        return {"daily_bar": pd.DataFrame({"session_date": self._sessions})}

    def resolve_session_date(self, requested_date: date) -> date:
        eligible = [current for current in self._sessions if current <= requested_date]
        return eligible[-1] if eligible else self._sessions[0]


class _FakeMarketData:
    def __init__(self, prices: dict[str, float], *, fractionable: set[str] | None = None) -> None:
        self._prices = prices
        self._fractionable = {symbol.upper() for symbol in (fractionable or set())}

    def list_assets(self, *, status: str = "active", asset_class: str = "us_equity") -> list[dict[str, object]]:
        return [
            {
                "symbol": symbol,
                "tradable": True,
                "fractionable": symbol.upper() in self._fractionable,
            }
            for symbol in self._prices
        ]

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
    ) -> list[dict[str, object]]:
        return [
            {"symbol": symbol, "t": f"{end}T21:00:00Z", "c": self._prices[symbol]}
            for symbol in symbols
            if symbol in self._prices
        ]


class _FakeBroker:
    def __init__(self) -> None:
        self.submitted: list[dict[str, object]] = []
        self._clock = BrokerClock(
            timestamp=datetime(2026, 4, 7, 13, 0, tzinfo=timezone.utc),
            is_open=True,
            next_open=datetime(2026, 4, 8, 13, 30, tzinfo=timezone.utc),
            next_close=datetime(2026, 4, 7, 20, 0, tzinfo=timezone.utc),
            raw={},
        )
        self._account = BrokerAccount(
            account_id="paper-1",
            status="ACTIVE",
            cash=100000.0,
            equity=100000.0,
            buying_power=100000.0,
            portfolio_value=100000.0,
            long_market_value=0.0,
            short_market_value=0.0,
            pattern_day_trader=False,
            trading_blocked=False,
            account_blocked=False,
            raw={},
        )
        self._positions: tuple[BrokerPosition, ...] = ()

    def is_paper_trading_environment(self) -> bool:
        return True

    def get_account(self) -> BrokerAccount:
        return self._account

    def get_clock(self) -> BrokerClock:
        return self._clock

    def list_positions(self) -> tuple[BrokerPosition, ...]:
        return self._positions

    def list_orders(self, *, status: str | None = None, symbols: list[str] | None = None) -> tuple[BrokerOrder, ...]:
        return ()

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
    ) -> BrokerOrder:
        self.submitted.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "notional": notional,
                "client_order_id": client_order_id,
            }
        )
        return BrokerOrder(
            order_id=f"order-{len(self.submitted)}",
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            time_in_force=time_in_force,
            status="new",
            quantity=quantity,
            filled_quantity=0.0,
            filled_avg_price=None,
            limit_price=limit_price,
            submitted_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            raw={},
        )


class _FakeSleevePlanner:
    def plan(self, session_date: date, *, account_snapshot: AccountSnapshot) -> SleeveSelection:
        target = TargetPosition(
            symbol="AAPL",
            target_weight=1.0,
            max_weight=1.0,
            reason="test",
            timestamp=datetime(2026, 4, 6, 20, 0, tzinfo=timezone.utc),
            meta={"adjusted_score": 1.23, "sector": "Tech", "industry": "Hardware"},
        )
        return SleeveSelection(
            universe_size=100,
            signal_count=100,
            target_count=1,
            selected_symbols=("AAPL",),
            targets=(target,),
            prediction_context={"prediction_date": "2026-04-06"},
        )


def test_load_hybrid_config_expands_weights(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "profile_id": "x",
                "run_name": "x",
                "core_policy": {
                    "allocation_weight": 0.94,
                    "weights_within_core": {"SPY": 0.1, "AGG": 0.9},
                },
                "alpha_sleeve": {
                    "name": "lightgbm_ranker",
                    "allocation_weight": 0.06,
                    "top_k": 10,
                    "horizon": 5,
                    "sector_neutral": True,
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_hybrid_config(config_path)
    assert config.core_weights == (("SPY", 0.094), ("AGG", 0.846))
    assert config.sleeve_weight == 0.06


def test_compute_rebalance_state_respects_interval() -> None:
    sessions = [date(2026, 4, 1), date(2026, 4, 2), date(2026, 4, 3), date(2026, 4, 6), date(2026, 4, 7), date(2026, 4, 8)]
    state = compute_rebalance_state(
        effective_session_date=date(2026, 4, 7),
        available_session_dates=sessions,
        last_rebalance_effective_session_date=date(2026, 4, 1),
        has_existing_positions=True,
        interval_sessions=5,
    )
    assert state["rebalance_due"] is False
    assert state["sessions_since_last_rebalance"] == 4
    assert state["next_rebalance_effective_session_date"] == date(2026, 4, 8)


def test_build_order_plan_uses_fractional_notional_for_fractionable_assets() -> None:
    orders = build_order_plan(
        account_equity=100000.0,
        broker_positions=(),
        target_weights={"SPY": 0.94, "AAPL": 0.06},
        latest_prices={"SPY": 100.0, "AAPL": 200.0},
        asset_metadata={
            "SPY": {"tradable": True, "fractionable": True},
            "AAPL": {"tradable": True, "fractionable": False},
        },
        min_trade_notional=25.0,
    )
    assert len(orders) == 2
    spy_order = next(order for order in orders if order.symbol == "SPY")
    aapl_order = next(order for order in orders if order.symbol == "AAPL")
    assert spy_order.notional == 94000.0
    assert spy_order.quantity is None
    assert aapl_order.quantity == 30
    assert aapl_order.notional is None


def test_hybrid_runner_executes_and_updates_state(tmp_path: Path) -> None:
    config = HybridPaperConfig(
        profile_id="hybrid-test",
        run_name="hybrid-test",
        model_name="lightgbm_ranker",
        top_k=10,
        horizon=5,
        min_close=10.0,
        min_median_dollar_volume_20=30_000_000.0,
        max_vol_20=0.065,
        max_positions_per_sector=2,
        sector_neutral=True,
        core_weights=(("SPY", 0.94),),
        sleeve_weight=0.06,
    )
    state_path = tmp_path / "state.json"
    runner = HybridPaperRunner(
        config=config,
        broker=_FakeBroker(),
        market_data=_FakeMarketData({"SPY": 100.0, "AAPL": 200.0}, fractionable={"SPY"}),
        dataset_cache=_FakeDatasetCache([date(2026, 4, 1), date(2026, 4, 2), date(2026, 4, 3), date(2026, 4, 6)]),
        sleeve_planner=_FakeSleevePlanner(),
    )
    result = runner.run(session_date=date(2026, 4, 7), execute=True, state_path=state_path)
    assert result.ok is True
    assert result.stage == "executed"
    assert result.counts["submitted_orders"] == 2
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["last_rebalance_effective_session_date"] == "2026-04-06"
