from __future__ import annotations

import pandas as pd

from stockmachine.research.p1_rigor import (
    build_cost_stress_summary,
    build_period_stability_summary,
    summarize_backtest_records,
)


def _sample_records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_date": "2025-01-02",
                "entry_date": "2025-01-03",
                "exit_date": "2025-01-10",
                "gross_return": 0.02,
                "net_return": 0.019,
                "benchmark_return": 0.01,
                "turnover": 1.0,
                "cost_bps": 10.0,
                "positions": 10,
            },
            {
                "signal_date": "2025-02-03",
                "entry_date": "2025-02-04",
                "exit_date": "2025-02-11",
                "gross_return": -0.01,
                "net_return": -0.011,
                "benchmark_return": -0.005,
                "turnover": 1.0,
                "cost_bps": 10.0,
                "positions": 10,
            },
            {
                "signal_date": "2025-04-02",
                "entry_date": "2025-04-03",
                "exit_date": "2025-04-10",
                "gross_return": 0.03,
                "net_return": 0.028,
                "benchmark_return": 0.015,
                "turnover": 2.0,
                "cost_bps": 20.0,
                "positions": 10,
            },
        ]
    )


def test_summarize_backtest_records_uses_existing_net_returns_by_default() -> None:
    summary = summarize_backtest_records(_sample_records(), horizon=5)

    assert summary["sessions"] == 3
    assert summary["benchmark_total_return"] > 0
    assert summary["mean_turnover"] == 4 / 3
    assert summary["mean_cost_bps"] == 40 / 3


def test_summarize_backtest_records_can_revalue_with_new_cost() -> None:
    records = _sample_records()

    summary = summarize_backtest_records(records, horizon=5, cost_bps_per_side=40.0)

    expected_returns = records["gross_return"] - records["turnover"] * 0.004
    expected_total = float((1.0 + expected_returns).prod() - 1.0)
    assert abs(summary["total_return"] - expected_total) < 1e-12
    assert summary["mean_cost_bps"] == float((records["turnover"] * 40.0).mean())


def test_build_period_stability_summary_splits_year_and_quarter() -> None:
    records = _sample_records()

    yearly = build_period_stability_summary(records, model_name="hist_gbm", horizon=5, period="year")
    quarterly = build_period_stability_summary(records, model_name="hist_gbm", horizon=5, period="quarter")

    assert list(yearly["period_label"]) == ["2025"]
    assert set(quarterly["period_label"]) == {"2025Q1", "2025Q2"}
    assert set(yearly["model"]) == {"hist_gbm"}
    assert set(quarterly["period"]) == {"quarter"}


def test_build_cost_stress_summary_emits_one_row_per_cost_level() -> None:
    summary = build_cost_stress_summary(
        _sample_records(),
        model_name="hist_gbm",
        horizon=5,
        cost_levels_bps=(10.0, 20.0, 40.0),
    )

    assert list(summary["cost_bps_per_side"]) == [10.0, 20.0, 40.0]
    assert (summary["model"] == "hist_gbm").all()
    assert "excess_total_return" in summary.columns
