from __future__ import annotations

import pandas as pd

from stockmachine.apps.run_multi_asset_core_experiment import (
    run_default_experiment_suite,
    slice_return_frame,
    write_experiment_artifacts,
)
from stockmachine.research.multi_asset import CovarianceConfig, RollingAllocationConfig


def _bucket_returns() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=10, freq="B")
    return pd.DataFrame(
        {
            "equity_us": [0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02],
            "equity_ex_us": [0.00, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00],
            "duration": [0.00, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00, 0.01, -0.01],
            "credit": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, 0.01, -0.01, 0.00],
            "inflation_hedge": [0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01],
            "trend": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01, 0.00, 0.02],
            "cash": [0.0001] * 10,
        },
        index=dates,
    )


def _symbol_returns() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=10, freq="B")
    return pd.DataFrame(
        {
            "SPY": [0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02],
            "VXUS": [0.00, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00],
            "AGG": [0.00, -0.002, 0.001, 0.000, 0.001, -0.001, 0.001, 0.000, 0.001, -0.001],
            "CTA": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01, 0.00, 0.02],
            "GLDM": [0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01],
            "IEF": [0.00, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00, 0.01, -0.01],
            "LQD": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, 0.01, -0.01, 0.00],
            "SGOV": [0.0001] * 10,
        },
        index=dates,
    )


def test_slice_return_frame_applies_date_bounds() -> None:
    returns = _bucket_returns()

    sliced = slice_return_frame(
        returns,
        start_date="2026-01-06",
        end_date="2026-01-12",
    )

    assert sliced.index.min() == pd.Timestamp("2026-01-06")
    assert sliced.index.max() == pd.Timestamp("2026-01-12")


def test_run_default_experiment_suite_and_write_artifacts(tmp_path) -> None:
    returns = _bucket_returns()
    symbol_returns = _symbol_returns()
    covariance_config = CovarianceConfig(long_lookback=4, short_lookback=2, ewma_lambda=0.8)
    rolling_config = RollingAllocationConfig(rebalance_frequency=2, effective_lag=1, min_history=4)

    strategy_runs = run_default_experiment_suite(
        returns,
        symbol_returns=symbol_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column="equity_us",
        cost_bps_per_side=5.0,
    )

    assert [run.strategy_name for run in strategy_runs] == [
        "current_c_policy_core",
        "current_c_policy_core_periodic_rebalance",
        "static_equal_weight_non_cash",
        "rolling_erc_core",
        "rolling_c2_v0_core",
    ]
    assert all(run.summary["sessions"] > 0 for run in strategy_runs)

    write_experiment_artifacts(
        output_root=tmp_path,
        bucket_returns=returns,
        symbol_returns=symbol_returns,
        strategy_runs=strategy_runs,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column="equity_us",
        cost_bps_per_side=5.0,
    )

    assert (tmp_path / "summary_metrics.csv").exists()
    assert (tmp_path / "summary_metrics_common_window.csv").exists()
    assert (tmp_path / "symbol_returns.csv").exists()
    assert (tmp_path / "yearly_returns.csv").exists()
    assert (tmp_path / "yearly_returns_common_window.csv").exists()
    assert (tmp_path / "weight_summary.csv").exists()
    assert (tmp_path / "risk_share_summary_common_window.csv").exists()
    assert (tmp_path / "summary_report.md").exists()
    assert (tmp_path / "run_meta.json").exists()
    assert (tmp_path / "current_c_policy_core" / "backtest_records.csv").exists()
    assert (tmp_path / "current_c_policy_core_periodic_rebalance" / "backtest_records.csv").exists()
    assert (tmp_path / "rolling_erc_core" / "backtest_records.csv").exists()
    assert (tmp_path / "rolling_c2_v0_core" / "risk_share_diagnostics.csv").exists()

    common_summary = pd.read_csv(tmp_path / "summary_metrics_common_window.csv")
    assert set(common_summary["strategy_name"]) == {
        "current_c_policy_core",
        "current_c_policy_core_periodic_rebalance",
        "static_equal_weight_non_cash",
        "rolling_erc_core",
        "rolling_c2_v0_core",
    }
