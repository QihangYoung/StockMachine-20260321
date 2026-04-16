from __future__ import annotations

import json

import pandas as pd

from stockmachine.apps.run_fmf_validation_baseline_experiment import (
    _augment_run_meta,
    run_fmf_validation_baseline_suite,
    write_split_window_artifacts,
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
            "IEF": [0.00, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00, 0.01, -0.01, 0.01, 0.00],
            "LQD": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00, 0.01, -0.01, 0.00, 0.01, 0.00],
            "GLD": [0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01],
            "FMF": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01, 0.00, 0.02, -0.01, 0.00],
            "BIL": [0.0001] * 12,
        },
        index=dates,
    )


def test_run_fmf_validation_baseline_suite_and_split_outputs(tmp_path) -> None:
    bucket_returns = _bucket_returns()
    covariance_config = CovarianceConfig(long_lookback=4, short_lookback=2, ewma_lambda=0.8)
    rolling_config = RollingAllocationConfig(rebalance_frequency=2, effective_lag=1, min_history=4)

    strategy_runs = run_fmf_validation_baseline_suite(
        bucket_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        benchmark_column="equity_us",
        cost_bps_per_side=5.0,
    )

    assert [run.strategy_name for run in strategy_runs] == [
        "static_equal_weight_non_cash",
        "rolling_erc_core",
        "rolling_c2_v0_seed_core",
    ]

    validation_summary = write_split_window_artifacts(
        output_root=tmp_path,
        strategy_runs=strategy_runs,
        window_name="validation_window",
        start_date=pd.Timestamp("2026-01-02"),
        end_date=pd.Timestamp("2026-01-12"),
    )
    test_summary = write_split_window_artifacts(
        output_root=tmp_path,
        strategy_runs=strategy_runs,
        window_name="test_window",
        start_date=pd.Timestamp("2026-01-13"),
        end_date=pd.Timestamp("2026-01-19"),
    )

    assert set(validation_summary["strategy_name"]) == {
        "static_equal_weight_non_cash",
        "rolling_erc_core",
        "rolling_c2_v0_seed_core",
    }
    assert set(test_summary["strategy_name"]) == {
        "static_equal_weight_non_cash",
        "rolling_erc_core",
        "rolling_c2_v0_seed_core",
    }
    assert (tmp_path / "summary_metrics_validation_window.csv").exists()
    assert (tmp_path / "summary_metrics_validation_window_common_window.csv").exists()
    assert (tmp_path / "summary_metrics_test_window.csv").exists()
    assert (tmp_path / "summary_metrics_test_window_common_window.csv").exists()
    assert (tmp_path / "yearly_returns_validation_window.csv").exists()
    assert (tmp_path / "yearly_returns_test_window.csv").exists()
    assert (tmp_path / "risk_share_summary_validation_window.csv").exists()
    assert (tmp_path / "risk_share_summary_test_window.csv").exists()

    _augment_run_meta(
        path=tmp_path / "run_meta.json",
        bucket_returns=bucket_returns,
        symbol_returns=_symbol_returns(),
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        validation_start=pd.Timestamp("2026-01-02"),
        validation_end=pd.Timestamp("2026-01-12"),
        test_start=pd.Timestamp("2026-01-13"),
        validation_summary=validation_summary,
        test_summary=test_summary,
        include_test_window=True,
        test_end=pd.Timestamp("2026-01-19"),
    )

    run_meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert run_meta["experiment_family"] == "fmf_validation_baseline"
    assert run_meta["universe_name"] == "us_multi_asset_fmf_validation_v1"
    assert run_meta["validation_window"]["end_date"] == "2026-01-12"
    assert run_meta["test_window"]["start_date"] == "2026-01-13"
    assert run_meta["lockbox_policy"]["test_window_locked"] is False


def test_augment_run_meta_can_keep_test_window_locked(tmp_path) -> None:
    bucket_returns = _bucket_returns()
    covariance_config = CovarianceConfig(long_lookback=4, short_lookback=2, ewma_lambda=0.8)
    rolling_config = RollingAllocationConfig(rebalance_frequency=2, effective_lag=1, min_history=4)
    validation_summary = pd.DataFrame(
        [
            {
                "strategy_name": "rolling_c2_v0_seed_core",
                "sessions": 4,
                "total_return": 0.1,
                "annualized_return": 0.2,
                "annualized_volatility": 0.1,
                "sharpe": 1.0,
                "max_drawdown": -0.05,
                "benchmark_total_return": 0.08,
                "mean_turnover": 0.0,
                "mean_cost_bps": 0.0,
            }
        ]
    )

    _augment_run_meta(
        path=tmp_path / "run_meta.json",
        bucket_returns=bucket_returns,
        symbol_returns=_symbol_returns(),
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        validation_start=pd.Timestamp("2026-01-02"),
        validation_end=pd.Timestamp("2026-01-12"),
        test_start=pd.Timestamp("2026-01-13"),
        validation_summary=validation_summary,
        test_summary=None,
        include_test_window=False,
        test_end=pd.Timestamp("2026-01-19"),
    )

    run_meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert run_meta["lockbox_policy"]["test_window_locked"] is True
    assert run_meta["test_window"]["summary_rows"] == []
