from __future__ import annotations

import json

import pandas as pd

from stockmachine.apps.run_fmf_validation_target_vol_experiment import (
    _augment_run_meta,
    run_fmf_validation_target_vol_suite,
)
from stockmachine.research.multi_asset import (
    CovarianceConfig,
    RollingAllocationConfig,
    SharpeTargetVolConfig,
)


def _bucket_returns() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=12, freq="B")
    return pd.DataFrame(
        {
            "equity_us": [0.03, 0.02, -0.02, 0.03, 0.02, -0.01, 0.03, 0.02, -0.02, 0.03, 0.02, -0.01],
            "equity_ex_us": [0.02, 0.01, -0.01, 0.02, 0.01, -0.01, 0.02, 0.01, -0.01, 0.02, 0.01, -0.01],
            "duration": [0.00, -0.01, 0.01, 0.00, -0.01, 0.01, 0.00, -0.01, 0.01, 0.00, -0.01, 0.01],
            "credit": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01, -0.01, 0.00, 0.01, -0.01, 0.00, 0.01],
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
            "SPY": [0.03, 0.02, -0.02, 0.03, 0.02, -0.01, 0.03, 0.02, -0.02, 0.03, 0.02, -0.01],
            "VXUS": [0.02, 0.01, -0.01, 0.02, 0.01, -0.01, 0.02, 0.01, -0.01, 0.02, 0.01, -0.01],
            "IEF": [0.00, -0.01, 0.01, 0.00, -0.01, 0.01, 0.00, -0.01, 0.01, 0.00, -0.01, 0.01],
            "LQD": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01, -0.01, 0.00, 0.01, -0.01, 0.00, 0.01],
            "GLD": [0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01],
            "FMF": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01, 0.00, 0.02, -0.01, 0.00],
            "BIL": [0.0001] * 12,
        },
        index=dates,
    )


def test_run_fmf_validation_target_vol_suite_includes_target_vol_strategy() -> None:
    strategy_runs = run_fmf_validation_target_vol_suite(
        _bucket_returns(),
        covariance_config=CovarianceConfig(long_lookback=4, short_lookback=2, ewma_lambda=0.8),
        rolling_config=RollingAllocationConfig(rebalance_frequency=2, effective_lag=1, min_history=4),
        target_vol_config=SharpeTargetVolConfig(
            strategic_buckets=("equity_us", "equity_ex_us", "duration", "credit", "inflation_hedge", "trend"),
            cash_bucket="cash",
            score_lookback=4,
            target_volatility=0.10,
        ),
        benchmark_column="equity_us",
        cost_bps_per_side=5.0,
    )

    assert [run.strategy_name for run in strategy_runs] == [
        "static_equal_weight_non_cash",
        "rolling_erc_core",
        "rolling_c2_v0_seed_core",
        "rolling_fmf_c2_e42_c10_d22_i20_t06",
        "rolling_sharpe_target_vol_4d",
    ]
    target_vol_run = strategy_runs[-1]
    assert target_vol_run.diagnostics is not None
    assert "risk_scale" in target_vol_run.diagnostics.columns
    assert "cash_weight" in target_vol_run.diagnostics.columns
    assert target_vol_run.weight_schedule["cash"].between(0.0, 1.0).all()


def test_target_vol_run_meta_marks_test_window_locked(tmp_path) -> None:
    strategy_runs = run_fmf_validation_target_vol_suite(
        _bucket_returns(),
        covariance_config=CovarianceConfig(long_lookback=4, short_lookback=2, ewma_lambda=0.8),
        rolling_config=RollingAllocationConfig(rebalance_frequency=2, effective_lag=1, min_history=4),
        target_vol_config=SharpeTargetVolConfig(
            strategic_buckets=("equity_us", "equity_ex_us", "duration", "credit", "inflation_hedge", "trend"),
            cash_bucket="cash",
            score_lookback=4,
            target_volatility=0.10,
        ),
        benchmark_column="equity_us",
        cost_bps_per_side=0.0,
    )
    _augment_run_meta(
        path=tmp_path / "run_meta.json",
        bucket_returns=_bucket_returns(),
        symbol_returns=_symbol_returns(),
        covariance_config=CovarianceConfig(long_lookback=4, short_lookback=2, ewma_lambda=0.8),
        rolling_config=RollingAllocationConfig(rebalance_frequency=2, effective_lag=1, min_history=4),
        target_vol_config=SharpeTargetVolConfig(
            strategic_buckets=("equity_us", "equity_ex_us", "duration", "credit", "inflation_hedge", "trend"),
            cash_bucket="cash",
            score_lookback=4,
            target_volatility=0.10,
        ),
        validation_start=pd.Timestamp("2026-01-02"),
        validation_end=pd.Timestamp("2026-01-19"),
        strategy_runs=strategy_runs,
    )

    run_meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert run_meta["experiment_family"] == "fmf_validation_target_vol"
    assert run_meta["lockbox_policy"]["test_window_locked"] is True
    assert run_meta["lockbox_policy"]["test_window_exposed"] is False
