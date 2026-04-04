from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from stockmachine.backtest import DataFrameSignalModel, DailyRebalanceOpenHoldBacktestEngine
from stockmachine.backtest.protocols import AccountSnapshot
from stockmachine.domain.models import Signal, TargetPosition
from stockmachine.execution import NextOpenOrderExecutionPolicy
from stockmachine.portfolio import RiskAwareTopKPortfolioPolicy


def test_daily_rebalance_engine_applies_turnover_controls() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-07",
                ]
            ),
            "symbol": ["AAA", "BBB"] * 4,
            "sector": ["Tech", "Health"] * 4,
            "industry": ["Tech", "Health"] * 4,
            "close": [100.0, 100.0] * 4,
            "vol_20": [0.02, 0.02] * 4,
            "median_dollar_volume_20": [100_000_000.0, 100_000_000.0] * 4,
            "target": [0.03, 0.0, 0.0, 0.03, 0.02, 0.0, 0.0, 0.01],
            "future_return": [0.03, 0.0, 0.0, 0.03, 0.02, 0.0, 0.0, 0.01],
            "benchmark_future_return": [0.01] * 8,
            "score": [0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9],
            "confidence": [0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9],
            "model": ["ridge"] * 8,
        }
    )
    daily_bar = pd.DataFrame(
        {
            "session_date": pd.to_datetime(
                ["2025-01-03", "2025-01-03", "2025-01-06", "2025-01-06", "2025-01-07", "2025-01-07"]
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "open": [100.0, 100.0, 101.0, 100.0, 102.0, 101.0],
            "high": [101.0, 101.0, 102.0, 101.0, 103.0, 102.0],
            "low": [99.0, 99.0, 100.0, 99.0, 101.0, 100.0],
            "close": [100.0, 100.0, 101.0, 100.0, 102.0, 101.0],
            "volume": [1_000_000.0] * 6,
            "vwap": [100.0, 100.0, 101.0, 100.0, 102.0, 101.0],
        }
    )
    benchmark_index = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"]),
            "symbol": ["SPY", "SPY", "SPY"],
            "open": [500.0, 501.0, 502.0],
            "high": [501.0, 502.0, 503.0],
            "low": [499.0, 500.0, 501.0],
            "close": [500.0, 501.0, 502.0],
            "volume": [1_000_000.0] * 3,
        }
    )

    engine = DailyRebalanceOpenHoldBacktestEngine(
        predictions=predictions,
        daily_bar=daily_bar,
        benchmark_index=benchmark_index,
        signal_model=DataFrameSignalModel(predictions, horizon_bars=1),
        portfolio_policy=RiskAwareTopKPortfolioPolicy(top_k=1, max_positions_per_sector=1),
        execution_policy=NextOpenOrderExecutionPolicy(),
        horizon_bars=1,
        cost_bps_per_side=0.0,
        max_turnover=0.5,
    )

    result = engine.run(start_date=datetime(2025, 1, 2).date(), end_date=datetime(2025, 1, 7).date())

    records = pd.DataFrame(result.meta["records"])
    assert result.sessions == 2
    assert "cap_max_turnover" in set(records["rebalance_action"])
    capped = records.loc[records["rebalance_action"] == "cap_max_turnover"].iloc[0]
    assert float(capped["turnover"]) < float(capped["raw_turnover"])


@dataclass
class _SessionWeightPolicy:
    weights_by_date: dict[str, dict[str, float]]

    def build_targets(self, session_date, signals: list[Signal], account: AccountSnapshot) -> list[TargetPosition]:
        weight_map = self.weights_by_date[session_date.isoformat()]
        return [
            TargetPosition(
                symbol=symbol,
                target_weight=weight,
                max_weight=weight,
                reason="test",
                timestamp=datetime.combine(session_date, datetime.min.time()),
                meta={"close": 100.0},
            )
            for symbol, weight in weight_map.items()
        ]


def test_daily_rebalance_engine_skips_small_weight_deltas() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-07",
                ]
            ),
            "symbol": ["AAA", "BBB"] * 4,
            "sector": ["Tech", "Health"] * 4,
            "industry": ["Tech", "Health"] * 4,
            "close": [100.0, 100.0] * 4,
            "vol_20": [0.02, 0.02] * 4,
            "median_dollar_volume_20": [100_000_000.0, 100_000_000.0] * 4,
            "target": [0.03, 0.0] * 4,
            "future_return": [0.03, 0.0] * 4,
            "benchmark_future_return": [0.01] * 8,
            "score": [0.9, 0.1] * 4,
            "confidence": [0.9, 0.1] * 4,
            "model": ["ridge"] * 8,
        }
    )
    daily_bar = pd.DataFrame(
        {
            "session_date": pd.to_datetime(
                ["2025-01-03", "2025-01-03", "2025-01-06", "2025-01-06", "2025-01-07", "2025-01-07"]
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "open": [100.0, 100.0, 101.0, 100.0, 102.0, 101.0],
            "high": [101.0, 101.0, 102.0, 101.0, 103.0, 102.0],
            "low": [99.0, 99.0, 100.0, 99.0, 101.0, 100.0],
            "close": [100.0, 100.0, 101.0, 100.0, 102.0, 101.0],
            "volume": [1_000_000.0] * 6,
            "vwap": [100.0, 100.0, 101.0, 100.0, 102.0, 101.0],
        }
    )
    benchmark_index = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"]),
            "symbol": ["SPY", "SPY", "SPY"],
            "open": [500.0, 501.0, 502.0],
            "high": [501.0, 502.0, 503.0],
            "low": [499.0, 500.0, 501.0],
            "close": [500.0, 501.0, 502.0],
            "volume": [1_000_000.0] * 3,
        }
    )

    engine = DailyRebalanceOpenHoldBacktestEngine(
        predictions=predictions,
        daily_bar=daily_bar,
        benchmark_index=benchmark_index,
        signal_model=DataFrameSignalModel(predictions, horizon_bars=1),
        portfolio_policy=_SessionWeightPolicy(
            weights_by_date={
                "2025-01-02": {"AAA": 0.50, "BBB": 0.50},
                "2025-01-03": {"AAA": 0.53, "BBB": 0.47},
            }
        ),
        execution_policy=NextOpenOrderExecutionPolicy(),
        horizon_bars=1,
        cost_bps_per_side=0.0,
        min_weight_change=0.05,
    )

    result = engine.run(start_date=datetime(2025, 1, 2).date(), end_date=datetime(2025, 1, 7).date())

    records = pd.DataFrame(result.meta["records"])
    assert result.sessions == 2
    assert "skip_small_weight_deltas" in set(records["rebalance_action"])
    skipped = records.loc[records["rebalance_action"] == "skip_small_weight_deltas"].iloc[0]
    assert float(skipped["turnover"]) == 0.0
    assert int(skipped["changed_symbols"]) == 0


def test_daily_rebalance_engine_normalizes_gross_exposure_after_small_delta_skips() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-03",
                    "2025-01-03",
                    "2025-01-03",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-06",
                    "2025-01-06",
                    "2025-01-06",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-07",
                    "2025-01-07",
                    "2025-01-07",
                    "2025-01-07",
                ]
            ),
            "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE"] * 4,
            "sector": ["Tech", "Health", "Industrials", "Utilities", "Energy"] * 4,
            "industry": ["Tech", "Health", "Industrials", "Utilities", "Energy"] * 4,
            "close": [100.0] * 20,
            "vol_20": [0.02] * 20,
            "median_dollar_volume_20": [100_000_000.0] * 20,
            "target": [0.03] * 20,
            "future_return": [0.03] * 20,
            "benchmark_future_return": [0.01] * 20,
            "score": [0.9, 0.8, 0.7, 0.6, 0.5] * 4,
            "confidence": [0.9, 0.8, 0.7, 0.6, 0.5] * 4,
            "model": ["ridge"] * 20,
        }
    )
    daily_bar = pd.DataFrame(
        {
            "session_date": pd.to_datetime(
                [
                    "2025-01-03",
                    "2025-01-03",
                    "2025-01-03",
                    "2025-01-03",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-06",
                    "2025-01-06",
                    "2025-01-06",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-07",
                    "2025-01-07",
                    "2025-01-07",
                    "2025-01-07",
                ]
            ),
            "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE"] * 3,
            "open": [100.0] * 15,
            "high": [101.0] * 15,
            "low": [99.0] * 15,
            "close": [100.0] * 15,
            "volume": [1_000_000.0] * 15,
            "vwap": [100.0] * 15,
        }
    )
    benchmark_index = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"]),
            "symbol": ["SPY", "SPY", "SPY"],
            "open": [500.0, 501.0, 502.0],
            "high": [501.0, 502.0, 503.0],
            "low": [499.0, 500.0, 501.0],
            "close": [500.0, 501.0, 502.0],
            "volume": [1_000_000.0] * 3,
        }
    )

    engine = DailyRebalanceOpenHoldBacktestEngine(
        predictions=predictions,
        daily_bar=daily_bar,
        benchmark_index=benchmark_index,
        signal_model=DataFrameSignalModel(predictions, horizon_bars=1),
        portfolio_policy=_SessionWeightPolicy(
            weights_by_date={
                "2025-01-02": {"AAA": 0.20, "BBB": 0.20, "CCC": 0.20, "DDD": 0.20, "EEE": 0.20},
                "2025-01-03": {"AAA": 0.40, "BBB": 0.15, "CCC": 0.15, "DDD": 0.15, "EEE": 0.15},
            }
        ),
        execution_policy=NextOpenOrderExecutionPolicy(),
        horizon_bars=1,
        cost_bps_per_side=0.0,
        min_weight_change=0.06,
    )

    result = engine.run(start_date=datetime(2025, 1, 2).date(), end_date=datetime(2025, 1, 7).date())

    records = pd.DataFrame(result.meta["records"])
    normalized = records.loc[records["rebalance_action"].str.contains("normalize_gross_exposure", regex=False)].iloc[0]
    assert abs(float(normalized["gross_exposure"]) - 1.0) < 1e-9


def test_daily_rebalance_engine_enforces_position_limit_under_turnover_cap() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-07",
                ]
            ),
            "symbol": ["AAA", "BBB"] * 4,
            "sector": ["Tech", "Health"] * 4,
            "industry": ["Tech", "Health"] * 4,
            "close": [100.0, 100.0] * 4,
            "vol_20": [0.02, 0.02] * 4,
            "median_dollar_volume_20": [100_000_000.0, 100_000_000.0] * 4,
            "target": [0.03, 0.0] * 4,
            "future_return": [0.03, 0.0] * 4,
            "benchmark_future_return": [0.01] * 8,
            "score": [0.9, 0.1] * 4,
            "confidence": [0.9, 0.1] * 4,
            "model": ["ridge"] * 8,
        }
    )
    daily_bar = pd.DataFrame(
        {
            "session_date": pd.to_datetime(
                ["2025-01-03", "2025-01-03", "2025-01-06", "2025-01-06", "2025-01-07", "2025-01-07"]
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "open": [100.0, 100.0, 101.0, 100.0, 102.0, 101.0],
            "high": [101.0, 101.0, 102.0, 101.0, 103.0, 102.0],
            "low": [99.0, 99.0, 100.0, 99.0, 101.0, 100.0],
            "close": [100.0, 100.0, 101.0, 100.0, 102.0, 101.0],
            "volume": [1_000_000.0] * 6,
            "vwap": [100.0, 100.0, 101.0, 100.0, 102.0, 101.0],
        }
    )
    benchmark_index = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"]),
            "symbol": ["SPY", "SPY", "SPY"],
            "open": [500.0, 501.0, 502.0],
            "high": [501.0, 502.0, 503.0],
            "low": [499.0, 500.0, 501.0],
            "close": [500.0, 501.0, 502.0],
            "volume": [1_000_000.0] * 3,
        }
    )

    engine = DailyRebalanceOpenHoldBacktestEngine(
        predictions=predictions,
        daily_bar=daily_bar,
        benchmark_index=benchmark_index,
        signal_model=DataFrameSignalModel(predictions, horizon_bars=1),
        portfolio_policy=_SessionWeightPolicy(
            weights_by_date={
                "2025-01-02": {"AAA": 1.0},
                "2025-01-03": {"BBB": 1.0},
            }
        ),
        execution_policy=NextOpenOrderExecutionPolicy(),
        horizon_bars=1,
        cost_bps_per_side=0.0,
        max_turnover=0.3,
    )

    result = engine.run(start_date=datetime(2025, 1, 2).date(), end_date=datetime(2025, 1, 7).date())

    records = pd.DataFrame(result.meta["records"])
    assert result.sessions == 2
    assert int(records["positions"].max()) == 1
    capped = records.loc[records["rebalance_action"].str.contains("enforce_position_limit", regex=False)].iloc[0]
    assert float(capped["gross_exposure"]) < 1.0
