from __future__ import annotations

import pandas as pd

from stockmachine.apps.run_c_policy_sleeve_experiment import (
    apply_risk_match,
    align_core_and_sleeve_records,
    build_sleeve_blend_records,
    summarize_regime_gated_sleeve_blend,
    summarize_risk_matched_sleeve_blend,
    summarize_sleeve_blend,
)
from stockmachine.risk import BenchmarkTrendDrawdownRegimeDetector, RegimeGatePolicy


def _records(net_returns: list[float], *, turnover: float = 1.0) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=len(net_returns), freq="5B")
    return pd.DataFrame(
        {
            "signal_date": dates - pd.offsets.BDay(1),
            "entry_date": dates,
            "exit_date": dates + pd.offsets.BDay(5),
            "gross_return": net_returns,
            "net_return": net_returns,
            "benchmark_return": [0.0] * len(net_returns),
            "turnover": [turnover] * len(net_returns),
            "cost_bps": [turnover * 10.0] * len(net_returns),
            "positions": [8] * len(net_returns),
        }
    )


def test_align_core_and_sleeve_records_inner_joins_on_entry_date() -> None:
    core = _records([0.01, 0.02, 0.03])
    sleeve = _records([0.04, 0.05, 0.06]).iloc[1:].copy()

    aligned = align_core_and_sleeve_records(core, sleeve)

    assert len(aligned) == 2
    assert aligned["entry_date"].tolist() == sleeve["entry_date"].tolist()


def test_build_sleeve_blend_records_mixes_core_and_sleeve_returns() -> None:
    core = _records([0.00, 0.10])
    sleeve = _records([0.20, 0.00], turnover=2.0)
    aligned = align_core_and_sleeve_records(core, sleeve)

    blended = build_sleeve_blend_records(aligned, sleeve_weight=0.25)

    assert blended["net_return"].tolist() == [0.05, 0.07500000000000001]
    assert blended["turnover"].tolist() == [1.25, 1.25]


def test_build_sleeve_blend_records_accepts_dynamic_weights() -> None:
    core = _records([0.00, 0.10, 0.00])
    sleeve = _records([0.20, 0.00, 0.20], turnover=2.0)
    aligned = align_core_and_sleeve_records(core, sleeve)

    blended = build_sleeve_blend_records(aligned, sleeve_weight=pd.Series([0.0, 0.50, 1.0]))

    assert blended["net_return"].tolist() == [0.0, 0.05, 0.2]
    assert blended["turnover"].tolist() == [1.0, 1.5, 2.0]


def test_summarize_sleeve_blend_reports_positive_excess_when_sleeve_helps() -> None:
    core = _records([0.00, 0.00, 0.00, 0.00])
    sleeve = _records([0.05, 0.05, 0.05, 0.05])
    aligned = align_core_and_sleeve_records(core, sleeve)

    summary = summarize_sleeve_blend(
        aligned,
        sleeve_name="test_sleeve",
        sleeve_weight=0.20,
        horizon=5,
    )

    assert summary["sleeve_name"] == "test_sleeve"
    assert summary["sleeve_weight"] == 0.20
    assert summary["excess_total_return_vs_core"] > 0.0


def test_apply_risk_match_scales_to_target_annualized_volatility() -> None:
    records = _records([0.02, -0.01, 0.03, 0.00, 0.01, -0.02])

    scaled, leverage_multiplier, _ = apply_risk_match(
        records,
        target_annualized_volatility=0.10,
        horizon=5,
    )

    scaled_summary = summarize_sleeve_blend(
        align_core_and_sleeve_records(scaled, scaled),
        sleeve_name="scaled",
        sleeve_weight=0.0,
        horizon=5,
    )
    assert leverage_multiplier > 0.0
    assert abs(scaled_summary["annualized_volatility"] - 0.10) < 1e-9


def test_summarize_risk_matched_sleeve_blend_includes_raw_metrics() -> None:
    core = _records([0.005, 0.004, 0.006, 0.003, 0.004, 0.005])
    sleeve = _records([0.03, -0.01, 0.04, 0.00, 0.02, -0.01])
    aligned = align_core_and_sleeve_records(core, sleeve)

    summary = summarize_risk_matched_sleeve_blend(
        aligned,
        sleeve_name="test_sleeve",
        sleeve_weight=0.05,
        horizon=5,
        target_annualized_volatility=0.07,
    )

    assert summary["risk_match_mode"] == "target_core_volatility"
    assert summary["target_annualized_volatility"] == 0.07
    assert summary["leverage_multiplier"] > 0.0
    assert "raw_sharpe" in summary
    assert "raw_annualized_volatility" in summary


def test_summarize_regime_gated_sleeve_blend_reports_gate_metadata() -> None:
    core = _records([0.01, 0.01, -0.02, -0.01, 0.01, 0.00])
    sleeve = _records([0.03, 0.02, -0.04, -0.03, 0.02, 0.01])
    aligned = align_core_and_sleeve_records(core, sleeve)
    detector = BenchmarkTrendDrawdownRegimeDetector(trend_lookback_windows=2, drawdown_threshold=-0.01)
    policy = RegimeGatePolicy(
        multipliers={
            "warmup": 1.0,
            "bull": 1.0,
            "correction": 0.0,
            "bear": 0.0,
            "rebound": 0.5,
        }
    )

    summary, gated_alignment = summarize_regime_gated_sleeve_blend(
        aligned,
        sleeve_name="test_sleeve",
        sleeve_weight=0.06,
        horizon=5,
        detector=detector,
        gate_policy=policy,
    )

    assert summary["regime_gate_enabled"] is True
    assert summary["regime_lookback_windows"] == 2
    assert summary["regime_bull_multiplier"] == 1.0
    assert summary["regime_rebound_multiplier"] == 0.5
    assert summary["effective_sleeve_weight_max"] == 0.06
    assert "regime_label" in gated_alignment.columns
