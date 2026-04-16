from __future__ import annotations

import pandas as pd
import pytest

from stockmachine.apps.run_c2_risk_budget_sweep import (
    build_c2_region_policy_configs,
    parse_grid_argument,
    resolve_live_window,
    run_c2_sweep_suite,
    write_c2_sweep_artifacts,
)
from stockmachine.research.multi_asset import CovarianceConfig, RollingAllocationConfig


def _bucket_returns() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=12, freq="B")
    return pd.DataFrame(
        {
            "equity_us": [0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00],
            "equity_ex_us": [0.00, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00, 0.01, 0.00],
            "duration": [0.00, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00],
            "credit": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00],
            "inflation_hedge": [0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01],
            "trend": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01, 0.00, 0.02, -0.01, 0.00],
            "cash": [0.0001] * 12,
        },
        index=dates,
    )


def _symbol_returns() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=12, freq="B")
    return pd.DataFrame(
        {
            "SPY": [0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00],
            "VXUS": [0.00, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00, 0.01, 0.00],
            "AGG": [0.00, -0.002, 0.001, 0.000, 0.001, -0.001, 0.001, 0.000, 0.001, -0.001, 0.001, 0.000],
            "CTA": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01, 0.00, 0.02, -0.01, 0.00],
            "GLDM": [0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01],
            "IEF": [0.00, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00],
            "LQD": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00],
            "SGOV": [0.0001] * 12,
        },
        index=dates,
    )


def test_build_c2_region_policy_configs_skips_v0_center_and_sums_to_one() -> None:
    configs = build_c2_region_policy_configs(
        duration_grid=(0.15, 0.20),
        inflation_grid=(0.10, 0.15, 0.20),
    )

    assert "c2_region_e35_c10_d20_i15_t20" not in {config.name for config in configs}
    assert len(configs) == 5
    assert all(config.normalized_risk_budgets().sum() == pytest.approx(1.0) for config in configs)
    assert all(config.reserve_capital_series()["cash"] == pytest.approx(0.05) for config in configs)


def test_parse_grid_argument_parses_comma_separated_decimals() -> None:
    assert parse_grid_argument("0.35, 0.40,0.45") == pytest.approx((0.35, 0.40, 0.45))


def test_run_c2_sweep_suite_and_write_artifacts(tmp_path) -> None:
    bucket_returns = _bucket_returns()
    symbol_returns = _symbol_returns()
    covariance_config = CovarianceConfig(long_lookback=4, short_lookback=2, ewma_lambda=0.8)
    rolling_config = RollingAllocationConfig(rebalance_frequency=2, effective_lag=1, min_history=4)
    policy_configs = build_c2_region_policy_configs(
        duration_grid=(0.15, 0.20),
        inflation_grid=(0.10, 0.20),
    )

    strategy_runs = run_c2_sweep_suite(
        bucket_returns,
        symbol_returns=symbol_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column="equity_us",
        cost_bps_per_side=5.0,
        policy_configs=policy_configs,
    )

    strategy_names = [run.strategy_name for run in strategy_runs]
    assert "current_c_policy_core_periodic_rebalance" in strategy_names
    assert "rolling_c2_v0_core" in strategy_names
    assert any(name.startswith("rolling_c2_region_") for name in strategy_names)

    live_window = resolve_live_window(
        symbol_returns.loc[:, ["SPY", "VXUS", "AGG", "CTA", "GLDM", "IEF", "LQD", "SGOV"]],
        required_symbols=("SPY", "VXUS", "AGG", "CTA", "GLDM", "IEF", "LQD", "SGOV"),
    )

    write_c2_sweep_artifacts(
        output_root=tmp_path,
        bucket_returns=bucket_returns,
        symbol_returns=symbol_returns,
        strategy_runs=strategy_runs,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column="equity_us",
        cost_bps_per_side=5.0,
        policy_configs=policy_configs,
        live_window=live_window,
    )

    assert (tmp_path / "summary_metrics.csv").exists()
    assert (tmp_path / "summary_metrics_common_window.csv").exists()
    assert (tmp_path / "summary_metrics_live_window.csv").exists()
    assert (tmp_path / "summary_metrics_c2_candidates_common_window.csv").exists()
    assert (tmp_path / "policy_grid.csv").exists()
    assert (tmp_path / "sweep_report.md").exists()
    assert (tmp_path / "rolling_c2_v0_core" / "risk_share_diagnostics.csv").exists()

    policy_grid = pd.read_csv(tmp_path / "policy_grid.csv")
    assert set(policy_grid["strategy_name"]) == {"rolling_c2_v0_core"} | {
        f"rolling_{config.name}" for config in policy_configs
    }
    assert {"equity_total", "diversifier_total"}.issubset(policy_grid.columns)
