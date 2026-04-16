import pandas as pd

from stockmachine.apps.run_fmf_validation_regime_overlay_experiment import (
    build_regime_overlay_policy_map,
    build_validation_regime_state_frame,
    run_fmf_validation_regime_overlay_suite,
)
from stockmachine.research.multi_asset import CovarianceConfig, RollingAllocationConfig


def _bucket_returns() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=180, freq="B")
    equity_us = []
    for position in range(len(dates)):
        if position < 60:
            equity_us.append(0.002)
        elif position < 120:
            equity_us.append(-0.003)
        else:
            equity_us.append(0.0015)
    bucket_returns = pd.DataFrame(
        {
            "equity_us": equity_us,
            "equity_ex_us": [value * 0.6 for value in equity_us],
            "duration": [0.0008 if position >= 60 else -0.0002 for position in range(len(dates))],
            "credit": [0.0005 if position < 120 else 0.0002 for position in range(len(dates))],
            "inflation_hedge": [0.0003 if position % 2 == 0 else -0.0001 for position in range(len(dates))],
            "trend": [0.0004 if position >= 60 else 0.0001 for position in range(len(dates))],
            "cash": [0.00005] * len(dates),
        },
        index=dates,
    )
    return bucket_returns


def test_build_validation_regime_state_frame_maps_to_overlay_states() -> None:
    regime_frame = build_validation_regime_state_frame(
        _bucket_returns(),
        benchmark_column="equity_us",
        trend_lookback=20,
        vol_lookback=10,
        drawdown_threshold=-0.03,
        high_vol_threshold=0.03,
    )

    assert set(regime_frame["overlay_state"]).issubset({"risk_on", "neutral", "defensive"})
    assert regime_frame["overlay_state"].isin(["risk_on", "defensive"]).any()


def test_run_fmf_validation_regime_overlay_suite_adds_static_lead_and_overlay() -> None:
    bucket_returns = _bucket_returns()
    regime_frame = build_validation_regime_state_frame(
        bucket_returns,
        benchmark_column="equity_us",
        trend_lookback=20,
        vol_lookback=10,
        drawdown_threshold=-0.03,
        high_vol_threshold=0.03,
    )
    strategy_runs, returned_regime_frame = run_fmf_validation_regime_overlay_suite(
        bucket_returns,
        covariance_config=CovarianceConfig(long_lookback=30, short_lookback=10, ewma_lambda=0.9),
        rolling_config=RollingAllocationConfig(rebalance_frequency=21, effective_lag=1, min_history=30),
        benchmark_column="equity_us",
        cost_bps_per_side=0.0,
        regime_state_frame=regime_frame,
        overlay_policy_map=build_regime_overlay_policy_map(equity_duration_shift=0.04),
    )

    strategy_names = [run.strategy_name for run in strategy_runs]
    assert "rolling_fmf_c2_e42_c10_d22_i20_t06" in strategy_names
    assert "rolling_fmf_c2_regime_overlay_v1" in strategy_names
    overlay_run = next(run for run in strategy_runs if run.strategy_name == "rolling_fmf_c2_regime_overlay_v1")
    assert overlay_run.diagnostics is not None
    assert set(overlay_run.diagnostics["state_name"]).issubset({"risk_on", "neutral", "defensive"})
    assert overlay_run.records["rebalanced"].any()
    assert returned_regime_frame.equals(regime_frame)
