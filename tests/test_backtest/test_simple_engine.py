from datetime import datetime

import pandas as pd

from stockmachine.backtest import (
    DailyOpenHoldBacktestEngine,
    DailyStaggeredOpenHoldBacktestEngine,
    DataFrameSignalModel,
)
from stockmachine.execution import NextOpenOrderExecutionPolicy
from stockmachine.portfolio import RiskAwareTopKPortfolioPolicy


def test_daily_open_hold_backtest_engine_runs_one_period() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03", "2025-01-10", "2025-01-10"]),
            "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "sector": ["Tech", "Health", "Tech", "Health", "Tech", "Health"],
            "industry": ["Tech", "Health", "Tech", "Health", "Tech", "Health"],
            "close": [100.0, 50.0, 100.0, 50.0, 110.0, 51.0],
            "vol_20": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            "median_dollar_volume_20": [100_000_000.0] * 6,
            "target": [0.05, 0.01, 0.04, 0.01, 0.0, 0.0],
            "future_return": [0.06, 0.01, 0.04, 0.01, 0.0, 0.0],
            "benchmark_future_return": [0.02, 0.02, 0.01, 0.01, 0.0, 0.0],
            "score": [0.9, 0.1, 0.8, 0.2, 0.1, 0.2],
            "confidence": [0.9, 0.1, 0.8, 0.2, 0.1, 0.2],
            "model": ["hist_gbm"] * 6,
        }
    )
    daily_bar = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2025-01-03", "2025-01-03", "2025-01-10", "2025-01-10"]),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "open": [100.0, 50.0, 110.0, 51.0],
            "high": [101.0, 51.0, 111.0, 52.0],
            "low": [99.0, 49.0, 109.0, 50.0],
            "close": [100.0, 50.0, 110.0, 51.0],
            "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0],
            "vwap": [100.0, 50.0, 110.0, 51.0],
        }
    )
    benchmark_index = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2025-01-03", "2025-01-10"]),
            "symbol": ["SPY", "SPY"],
            "open": [500.0, 505.0],
            "high": [501.0, 506.0],
            "low": [499.0, 504.0],
            "close": [500.0, 505.0],
            "volume": [1_000_000.0, 1_000_000.0],
        }
    )

    engine = DailyOpenHoldBacktestEngine(
        predictions=predictions,
        daily_bar=daily_bar,
        benchmark_index=benchmark_index,
        signal_model=DataFrameSignalModel(predictions, horizon_bars=1),
        portfolio_policy=RiskAwareTopKPortfolioPolicy(top_k=1, max_positions_per_sector=1),
        execution_policy=NextOpenOrderExecutionPolicy(),
        horizon_bars=1,
        cost_bps_per_side=0.0,
    )

    result = engine.run(start_date=datetime(2025, 1, 2).date(), end_date=datetime(2025, 1, 10).date())

    assert result.sessions == 1
    assert result.total_return > 0


def test_daily_open_hold_backtest_engine_uses_adjusted_open_for_realized_return() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03", "2025-01-10", "2025-01-10"]),
            "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "sector": ["Tech", "Health", "Tech", "Health", "Tech", "Health"],
            "industry": ["Tech", "Health", "Tech", "Health", "Tech", "Health"],
            "close": [100.0, 50.0, 100.0, 50.0, 100.0, 50.0],
            "vol_20": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            "median_dollar_volume_20": [100_000_000.0] * 6,
            "target": [0.05, 0.01, 0.05, 0.01, 0.0, 0.0],
            "future_return": [0.05, 0.01, 0.05, 0.01, 0.0, 0.0],
            "benchmark_future_return": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "score": [0.9, 0.1, 0.8, 0.2, 0.1, 0.2],
            "confidence": [0.9, 0.1, 0.8, 0.2, 0.1, 0.2],
            "model": ["hist_gbm"] * 6,
        }
    )
    daily_bar = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2025-01-03", "2025-01-03", "2025-01-10", "2025-01-10"]),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "open": [100.0, 50.0, 100.0, 50.0],
            "adj_open": [100.0, 50.0, 105.0, 50.0],
            "high": [101.0, 51.0, 101.0, 51.0],
            "low": [99.0, 49.0, 99.0, 49.0],
            "close": [100.0, 50.0, 100.0, 50.0],
            "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0],
            "vwap": [100.0, 50.0, 100.0, 50.0],
        }
    )
    benchmark_index = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2025-01-03", "2025-01-10"]),
            "symbol": ["SPY", "SPY"],
            "open": [500.0, 500.0],
            "adj_open": [500.0, 500.0],
            "high": [501.0, 501.0],
            "low": [499.0, 499.0],
            "close": [500.0, 500.0],
            "volume": [1_000_000.0, 1_000_000.0],
        }
    )

    engine = DailyOpenHoldBacktestEngine(
        predictions=predictions,
        daily_bar=daily_bar,
        benchmark_index=benchmark_index,
        signal_model=DataFrameSignalModel(predictions, horizon_bars=1),
        portfolio_policy=RiskAwareTopKPortfolioPolicy(top_k=1, max_positions_per_sector=1),
        execution_policy=NextOpenOrderExecutionPolicy(),
        horizon_bars=1,
        cost_bps_per_side=0.0,
    )

    result = engine.run(start_date=datetime(2025, 1, 2).date(), end_date=datetime(2025, 1, 10).date())

    assert result.sessions == 1
    assert round(result.total_return, 6) == 0.05


def test_daily_staggered_engine_reaches_full_exposure_and_nets_overlap_turnover() -> None:
    prediction_dates = pd.to_datetime(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"]
    )
    predictions = pd.DataFrame(
        {
            "date": list(prediction_dates) * 2,
            "symbol": ["AAA"] * len(prediction_dates) + ["BBB"] * len(prediction_dates),
            "sector": ["Tech"] * len(prediction_dates) + ["Health"] * len(prediction_dates),
            "industry": ["Tech"] * len(prediction_dates) + ["Health"] * len(prediction_dates),
            "close": [100.0] * (2 * len(prediction_dates)),
            "vol_20": [0.02] * (2 * len(prediction_dates)),
            "median_dollar_volume_20": [100_000_000.0] * (2 * len(prediction_dates)),
            "target": [0.05] * len(prediction_dates) + [0.01] * len(prediction_dates),
            "future_return": [0.05] * len(prediction_dates) + [0.01] * len(prediction_dates),
            "benchmark_future_return": [0.0] * (2 * len(prediction_dates)),
            "score": [0.9] * len(prediction_dates) + [0.1] * len(prediction_dates),
            "confidence": [0.9] * len(prediction_dates) + [0.1] * len(prediction_dates),
            "model": ["hist_gbm"] * (2 * len(prediction_dates)),
        }
    )
    daily_bar = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"]),
            "symbol": ["AAA", "AAA", "AAA", "AAA", "AAA"],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1_000_000.0] * 5,
            "vwap": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )
    benchmark_index = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"]),
            "symbol": ["SPY"] * 5,
            "open": [500.0, 501.0, 502.0, 503.0, 504.0],
            "high": [501.0, 502.0, 503.0, 504.0, 505.0],
            "low": [499.0, 500.0, 501.0, 502.0, 503.0],
            "close": [500.0, 501.0, 502.0, 503.0, 504.0],
            "volume": [1_000_000.0] * 5,
        }
    )

    engine = DailyStaggeredOpenHoldBacktestEngine(
        predictions=predictions,
        daily_bar=daily_bar,
        benchmark_index=benchmark_index,
        signal_model=DataFrameSignalModel(predictions, horizon_bars=2),
        portfolio_policy=RiskAwareTopKPortfolioPolicy(top_k=1, max_positions_per_sector=1, sector_neutral=False),
        execution_policy=NextOpenOrderExecutionPolicy(),
        horizon_bars=2,
        cost_bps_per_side=0.0,
    )

    result = engine.run(start_date=datetime(2025, 1, 2).date(), end_date=datetime(2025, 1, 9).date())

    records = pd.DataFrame(result.meta["records"])
    assert result.sessions == 4
    assert round(float(records.iloc[0]["gross_exposure"]), 6) == 0.5
    assert float(records.iloc[1]["gross_exposure"]) > 0.99
    assert float(records.iloc[2]["turnover"]) < 0.01
    assert result.total_return > 0


def test_daily_staggered_engine_counts_turnover_when_replacement_symbol_changes() -> None:
    prediction_dates = pd.to_datetime(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"]
    )
    score_map = {
        "2025-01-02": (0.9, 0.1),
        "2025-01-03": (0.9, 0.1),
        "2025-01-06": (0.1, 0.9),
        "2025-01-07": (0.1, 0.9),
        "2025-01-08": (0.1, 0.9),
        "2025-01-09": (0.1, 0.9),
    }
    rows: list[dict[str, object]] = []
    for current_date in prediction_dates:
        aaa_score, bbb_score = score_map[current_date.date().isoformat()]
        rows.append(
            {
                "date": current_date,
                "symbol": "AAA",
                "sector": "Tech",
                "industry": "Tech",
                "close": 100.0,
                "vol_20": 0.02,
                "median_dollar_volume_20": 100_000_000.0,
                "target": 0.05,
                "future_return": 0.05,
                "benchmark_future_return": 0.0,
                "score": aaa_score,
                "confidence": aaa_score,
                "model": "hist_gbm",
            }
        )
        rows.append(
            {
                "date": current_date,
                "symbol": "BBB",
                "sector": "Health",
                "industry": "Health",
                "close": 100.0,
                "vol_20": 0.02,
                "median_dollar_volume_20": 100_000_000.0,
                "target": 0.05,
                "future_return": 0.05,
                "benchmark_future_return": 0.0,
                "score": bbb_score,
                "confidence": bbb_score,
                "model": "hist_gbm",
            }
        )
    predictions = pd.DataFrame(rows)
    daily_bar = pd.DataFrame(
        {
            "session_date": pd.to_datetime(
                ["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"] * 2
            ),
            "symbol": ["AAA"] * 5 + ["BBB"] * 5,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 100.0, 100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 101.0, 101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 99.0, 99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 100.0, 100.0, 101.0, 102.0, 103.0],
            "volume": [1_000_000.0] * 10,
            "vwap": [100.0, 101.0, 102.0, 103.0, 104.0, 100.0, 100.0, 101.0, 102.0, 103.0],
        }
    )
    benchmark_index = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"]),
            "symbol": ["SPY"] * 5,
            "open": [500.0, 501.0, 502.0, 503.0, 504.0],
            "high": [501.0, 502.0, 503.0, 504.0, 505.0],
            "low": [499.0, 500.0, 501.0, 502.0, 503.0],
            "close": [500.0, 501.0, 502.0, 503.0, 504.0],
            "volume": [1_000_000.0] * 5,
        }
    )

    engine = DailyStaggeredOpenHoldBacktestEngine(
        predictions=predictions,
        daily_bar=daily_bar,
        benchmark_index=benchmark_index,
        signal_model=DataFrameSignalModel(predictions, horizon_bars=2),
        portfolio_policy=RiskAwareTopKPortfolioPolicy(top_k=1, max_positions_per_sector=1, sector_neutral=False),
        execution_policy=NextOpenOrderExecutionPolicy(),
        horizon_bars=2,
        cost_bps_per_side=0.0,
    )

    result = engine.run(start_date=datetime(2025, 1, 2).date(), end_date=datetime(2025, 1, 9).date())

    records = pd.DataFrame(result.meta["records"])
    assert result.sessions == 4
    assert float(records.iloc[2]["turnover"]) > 0.4
