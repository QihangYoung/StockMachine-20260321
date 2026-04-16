from __future__ import annotations

import json

import pandas as pd
import pytest

from stockmachine.apps.run_fmf_validation_c2_grid_search import (
    build_fmf_c2_region_policy_configs,
    parse_grid_argument,
    run_fmf_c2_grid_search_suite,
    write_fmf_c2_grid_search_artifacts,
)
from stockmachine.research.multi_asset import CovarianceConfig, RollingAllocationConfig


def _bucket_returns() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=14, freq="B")
    return pd.DataFrame(
        {
            "equity_us": [0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02],
            "equity_ex_us": [0.00, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00],
            "duration": [0.00, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00, 0.01, -0.01],
            "credit": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, 0.01, -0.01],
            "inflation_hedge": [0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01],
            "trend": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01],
            "cash": [0.0001] * 14,
        },
        index=dates,
    )


def _symbol_returns() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=14, freq="B")
    return pd.DataFrame(
        {
            "SPY": [0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02],
            "VXUS": [0.00, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00],
            "IEF": [0.00, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00, 0.01, -0.01],
            "LQD": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, 0.01, -0.01],
            "GLD": [0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01],
            "FMF": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01],
            "BIL": [0.0001] * 14,
        },
        index=dates,
    )


def test_build_fmf_c2_region_policy_configs_skips_seed_and_sums_to_one() -> None:
    configs = build_fmf_c2_region_policy_configs(
        equity_total_grid=(0.35,),
        credit_grid=(0.10,),
        duration_grid=(0.15, 0.20),
        inflation_grid=(0.15, 0.20),
    )

    assert "fmf_c2_e35_c10_d20_i15_t20" not in {config.name for config in configs}
    assert len(configs) == 3
    assert all(config.normalized_risk_budgets().sum() == pytest.approx(1.0) for config in configs)
    assert all(config.reserve_capital_series()["cash"] == pytest.approx(0.05) for config in configs)


def test_parse_grid_argument_parses_comma_separated_decimals() -> None:
    assert parse_grid_argument("0.30, 0.35,0.40") == pytest.approx((0.30, 0.35, 0.40))


def test_run_fmf_c2_grid_search_suite_and_write_artifacts(tmp_path) -> None:
    bucket_returns = _bucket_returns()
    symbol_returns = _symbol_returns()
    covariance_config = CovarianceConfig(long_lookback=4, short_lookback=2, ewma_lambda=0.8)
    rolling_config = RollingAllocationConfig(rebalance_frequency=2, effective_lag=1, min_history=4)
    policy_configs = build_fmf_c2_region_policy_configs(
        equity_total_grid=(0.35,),
        credit_grid=(0.05, 0.10),
        duration_grid=(0.15, 0.20),
        inflation_grid=(0.15, 0.20),
    )

    strategy_runs = run_fmf_c2_grid_search_suite(
        bucket_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column="equity_us",
        cost_bps_per_side=5.0,
        policy_configs=policy_configs,
    )

    strategy_names = [run.strategy_name for run in strategy_runs]
    assert "static_equal_weight_non_cash" in strategy_names
    assert "rolling_c2_v0_seed_core" in strategy_names
    assert any(name.startswith("rolling_fmf_c2_") for name in strategy_names)

    write_fmf_c2_grid_search_artifacts(
        output_root=tmp_path,
        bucket_returns=bucket_returns,
        symbol_returns=symbol_returns,
        strategy_runs=strategy_runs,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column="equity_us",
        cost_bps_per_side=5.0,
        validation_start=pd.Timestamp("2026-01-02"),
        validation_end=pd.Timestamp("2026-01-21"),
        policy_configs=policy_configs,
    )

    assert (tmp_path / "summary_metrics.csv").exists()
    assert (tmp_path / "summary_metrics_common_window.csv").exists()
    assert (tmp_path / "summary_metrics_validation_window.csv").exists()
    assert (tmp_path / "summary_metrics_validation_window_common_window.csv").exists()
    assert (tmp_path / "summary_metrics_c2_candidates_validation_common_window.csv").exists()
    assert (tmp_path / "policy_grid.csv").exists()
    assert (tmp_path / "sweep_report.md").exists()
    assert not (tmp_path / "summary_metrics_test_window.csv").exists()

    run_meta = json.loads((tmp_path / "sweep_meta.json").read_text(encoding="utf-8"))
    assert run_meta["experiment_family"] == "fmf_validation_c2_grid_search"
    assert run_meta["lockbox_policy"]["test_window_locked"] is True
    assert run_meta["lockbox_policy"]["test_window_exposed"] is False
