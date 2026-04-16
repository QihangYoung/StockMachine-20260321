import pandas as pd

from stockmachine.apps.run_fmf_validation_regime_vnext_experiment import (
    build_regime_overlay_policy_map,
    build_validation_regime_state_frame_vnext,
    run_fmf_validation_regime_vnext_suite,
)
from stockmachine.research.multi_asset import CovarianceConfig, RollingAllocationConfig


def _bucket_returns() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=240, freq="B")
    rows = []
    for position, date in enumerate(dates):
        wiggle = ((position % 5) - 2) * 0.00005
        if position < 80:
            row = {
                "equity_us": 0.0018 + wiggle,
                "equity_ex_us": 0.0012 + wiggle * 0.8,
                "duration": -0.0003 - wiggle * 0.3,
                "credit": 0.0008 + wiggle * 0.6,
                "inflation_hedge": 0.0001 + wiggle * 0.2,
                "trend": 0.0006 + wiggle * 0.4,
                "cash": 0.00005,
            }
        elif position < 160:
            row = {
                "equity_us": -0.0028 + wiggle,
                "equity_ex_us": -0.0022 + wiggle * 0.8,
                "duration": 0.0010 - wiggle * 0.3,
                "credit": -0.0009 + wiggle * 0.5,
                "inflation_hedge": 0.0007 - wiggle * 0.2,
                "trend": 0.0008 - wiggle * 0.4,
                "cash": 0.00005,
            }
        else:
            row = {
                "equity_us": 0.0011 + wiggle,
                "equity_ex_us": 0.0008 + wiggle * 0.7,
                "duration": 0.0001 - wiggle * 0.2,
                "credit": 0.0005 + wiggle * 0.5,
                "inflation_hedge": 0.0002 + wiggle * 0.2,
                "trend": 0.0003 + wiggle * 0.3,
                "cash": 0.00005,
            }
        rows.append({"date": date, **row})
    return pd.DataFrame(rows).set_index("date")


def test_build_validation_regime_state_frame_vnext_outputs_scores_and_states() -> None:
    regime_frame = build_validation_regime_state_frame_vnext(
        _bucket_returns(),
        benchmark_column="equity_us",
        trend_lookback=20,
        relative_lookback=15,
        vol_lookback=10,
        correlation_lookback=15,
        normalization_lookback=20,
        score_smoothing_halflife=5.0,
        risk_on_min_score=0.02,
        defensive_min_score=0.02,
        risk_on_net_threshold=0.0,
        defensive_net_threshold=0.0,
    )

    assert {
        "regime_risk_on_score",
        "regime_defensive_score",
        "regime_net_score",
        "overlay_state",
    }.issubset(regime_frame.columns)
    assert set(regime_frame["overlay_state"]).issubset({"risk_on", "neutral", "defensive"})
    assert regime_frame["regime_risk_on_score"].notna().sum() > 0
    assert regime_frame["regime_defensive_score"].notna().sum() > 0


def test_run_fmf_validation_regime_vnext_suite_adds_overlay() -> None:
    bucket_returns = _bucket_returns()
    regime_frame = build_validation_regime_state_frame_vnext(
        bucket_returns,
        benchmark_column="equity_us",
        trend_lookback=20,
        relative_lookback=15,
        vol_lookback=10,
        correlation_lookback=15,
        normalization_lookback=20,
        score_smoothing_halflife=5.0,
        risk_on_min_score=0.02,
        defensive_min_score=0.02,
        risk_on_net_threshold=0.0,
        defensive_net_threshold=0.0,
    )
    strategy_runs, returned_regime_frame = run_fmf_validation_regime_vnext_suite(
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
    assert "rolling_fmf_c2_regime_overlay_vnext" in strategy_names
    overlay_run = next(run for run in strategy_runs if run.strategy_name == "rolling_fmf_c2_regime_overlay_vnext")
    assert overlay_run.diagnostics is not None
    assert set(overlay_run.diagnostics["state_name"]).issubset({"risk_on", "neutral", "defensive"})
    assert overlay_run.records["rebalanced"].any()
    assert returned_regime_frame.equals(regime_frame)
