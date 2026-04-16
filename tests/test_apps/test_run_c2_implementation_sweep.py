from __future__ import annotations

import pandas as pd

from stockmachine.apps.run_c2_implementation_sweep import (
    build_lead_c2_policy_config,
    build_live_window,
    parse_float_grid,
    parse_int_grid,
    run_implementation_suite,
    write_implementation_artifacts,
)
from stockmachine.research.multi_asset import CovarianceConfig, RollingAllocationConfig


def _bucket_returns() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=16, freq="B")
    return pd.DataFrame(
        {
            "equity_us": [0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.00, -0.01, 0.02],
            "equity_ex_us": [0.00, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00, 0.01, 0.00],
            "duration": [0.00, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00],
            "credit": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00, 0.01, 0.00],
            "inflation_hedge": [0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01],
            "trend": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01, 0.00, 0.02],
            "cash": [0.0001] * 16,
        },
        index=dates,
    )


def _symbol_returns() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=16, freq="B")
    return pd.DataFrame(
        {
            "SPY": [0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.00, -0.01, 0.02],
            "VXUS": [0.00, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00, 0.01, 0.00],
            "AGG": [0.00, -0.002, 0.001, 0.000, 0.001, -0.001, 0.001, 0.000, 0.001, -0.001, 0.001, 0.000, 0.001, -0.001, 0.001, 0.000],
            "CTA": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01, 0.00, 0.02],
            "GLDM": [0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01],
            "IEF": [0.00, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00],
            "LQD": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00, 0.01, 0.00],
            "SGOV": [0.0001] * 16,
        },
        index=dates,
    )


def test_parse_grid_helpers_and_lead_policy_builder() -> None:
    assert parse_float_grid("0.00, 0.05") == (0.0, 0.05)
    assert parse_int_grid("21,42") == (21, 42)

    policy = build_lead_c2_policy_config(reserve_cash_weight=0.05)
    assert policy.name == "c2_impl_lead_cash05"
    assert policy.reserve_capital_series()["cash"] == 0.05
    assert policy.normalized_risk_budgets().to_dict() == {
        "equity_us": 0.24,
        "equity_ex_us": 0.11,
        "duration": 0.15,
        "credit": 0.05,
        "inflation_hedge": 0.20,
        "trend": 0.25,
    }


def test_run_implementation_suite_and_write_artifacts(tmp_path) -> None:
    bucket_returns = _bucket_returns()
    symbol_returns = _symbol_returns()
    covariance_config = CovarianceConfig(long_lookback=4, short_lookback=2, ewma_lambda=0.8)
    rolling_config = RollingAllocationConfig(rebalance_frequency=2, effective_lag=1, min_history=4)

    strategy_runs, implementation_diagnostics, implementation_summary = run_implementation_suite(
        bucket_returns,
        symbol_returns=symbol_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column="equity_us",
        cost_bps_per_side=5.0,
        reserve_cash_grid=(0.0, 0.05),
        drift_threshold_grid=(0.04,),
        stale_cap_grid=(2,),
    )

    strategy_names = [run.strategy_name for run in strategy_runs]
    assert "current_c_policy_core_periodic_rebalance" in strategy_names
    assert "rolling_c2_v0_core" in strategy_names
    assert "c2_impl_lead_cash00_calendar" in strategy_names
    assert "c2_impl_lead_cash05_band_t04_s02" in strategy_names
    assert "c2_impl_lead_cash05_band_t04_s02" in implementation_diagnostics

    live_window = build_live_window(
        symbol_returns.loc[:, ["SPY", "VXUS", "AGG", "CTA", "GLDM", "IEF", "LQD", "SGOV"]],
        required_symbols=("SPY", "VXUS", "AGG", "CTA", "GLDM", "IEF", "LQD", "SGOV"),
    )

    write_implementation_artifacts(
        output_root=tmp_path,
        bucket_returns=bucket_returns,
        symbol_returns=symbol_returns,
        strategy_runs=strategy_runs,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column="equity_us",
        cost_bps_per_side=5.0,
        implementation_diagnostics=implementation_diagnostics,
        implementation_summary=implementation_summary,
        live_window=live_window,
    )

    assert (tmp_path / "summary_metrics.csv").exists()
    assert (tmp_path / "summary_metrics_common_window.csv").exists()
    assert (tmp_path / "summary_metrics_live_window.csv").exists()
    assert (tmp_path / "implementation_summary.csv").exists()
    assert (tmp_path / "implementation_report.md").exists()
    assert (tmp_path / "c2_impl_lead_cash05_band_t04_s02" / "implementation_diagnostics.csv").exists()

    implementation_frame = pd.read_csv(tmp_path / "implementation_summary.csv")
    assert set(implementation_frame["rebalance_mode"]) == {"calendar", "threshold"}
