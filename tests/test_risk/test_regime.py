from __future__ import annotations

import pandas as pd

from stockmachine.risk import (
    BenchmarkForwardRegimeLabeler,
    BenchmarkTrendDrawdownRegimeDetector,
    BenchmarkTrendDrawdownVolRegimeDetector,
    BenchmarkTrendDrawdownVolCrossAssetRegimeDetector,
    RegimeGatePolicy,
    apply_regime_gate,
    build_regime_confusion_matrix,
)


def _frame(benchmark_returns: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=len(benchmark_returns), freq="5B")
    return pd.DataFrame(
        {
            "entry_date": dates,
            "benchmark_return": benchmark_returns,
        }
    )


def test_benchmark_trend_drawdown_detector_uses_only_lagged_history() -> None:
    frame = _frame([0.10, 0.10, -0.30, 0.15, 0.05, 0.00])
    detector = BenchmarkTrendDrawdownRegimeDetector(trend_lookback_windows=2, drawdown_threshold=-0.10)

    labeled = detector.label_frame(frame)

    assert labeled["regime_label"].tolist() == [
        "warmup",
        "warmup",
        "bull",
        "bear",
        "bear",
        "rebound",
    ]
    assert labeled.loc[2, "regime_trailing_return"] > 0.0
    assert labeled.loc[3, "regime_drawdown"] <= -0.10
    assert labeled.loc[5, "regime_drawdown"] <= -0.10
    assert labeled.loc[5, "regime_trailing_return"] > 0.0


def test_regime_gate_policy_maps_all_known_labels() -> None:
    policy = RegimeGatePolicy(
        multipliers={
            "warmup": 1.0,
            "bull": 1.0,
            "correction": 0.0,
            "bear": 0.0,
            "rebound": 0.5,
        }
    )

    mapped = policy.apply(pd.Series(["warmup", "bull", "correction", "bear", "rebound"]))

    assert mapped.tolist() == [1.0, 1.0, 0.0, 0.0, 0.5]


def test_apply_regime_gate_builds_effective_weights() -> None:
    frame = _frame([0.10, 0.10, -0.30, 0.15, 0.05, 0.00])
    detector = BenchmarkTrendDrawdownRegimeDetector(trend_lookback_windows=2, drawdown_threshold=-0.10)
    policy = RegimeGatePolicy(
        multipliers={
            "warmup": 1.0,
            "bull": 1.0,
            "correction": 0.0,
            "bear": 0.0,
            "rebound": 0.5,
        }
    )

    gated = apply_regime_gate(frame, base_sleeve_weight=0.06, detector=detector, gate_policy=policy)

    assert gated["effective_sleeve_weight"].round(4).tolist() == [0.06, 0.06, 0.06, 0.0, 0.0, 0.03]
    assert gated["effective_core_weight"].round(4).tolist() == [0.94, 0.94, 0.94, 1.0, 1.0, 0.97]


def test_benchmark_forward_regime_labeler_creates_market_centric_targets() -> None:
    frame = _frame([0.03, 0.02, 0.01, -0.02, -0.01, -0.01, -0.02, 0.03])
    labeler = BenchmarkForwardRegimeLabeler(forward_windows=3, drawdown_threshold=-0.02)

    labeled = labeler.label_frame(frame)

    assert labeled["forward_regime_label"].tolist() == [
        "bull",
        "rebound",
        "bear",
        "correction",
        "bear",
        "bear",
        "unlabeled",
        "unlabeled",
    ]
    assert labeled.loc[0, "forward_regime_total_return"] > 0.0
    assert labeled.loc[2, "forward_regime_total_return"] <= 0.0
    assert labeled.loc[2, "forward_regime_max_drawdown"] <= -0.02


def test_build_regime_confusion_matrix_counts_predicted_vs_target_labels() -> None:
    frame = pd.DataFrame(
        {
            "regime_label": ["bull", "bull", "bear", "correction"],
            "forward_regime_label": ["bull", "correction", "bear", "bear"],
        }
    )

    confusion = build_regime_confusion_matrix(frame)

    assert confusion.loc["bull", "bull"] == 1
    assert confusion.loc["bull", "correction"] == 1
    assert confusion.loc["bear", "bear"] == 1


def test_trend_drawdown_vol_detector_flags_high_volatility_as_stress() -> None:
    frame = _frame([0.01, 0.01, 0.01, -0.12, 0.12, -0.12, 0.12, 0.01])
    detector = BenchmarkTrendDrawdownVolRegimeDetector(
        trend_lookback_windows=2,
        drawdown_threshold=-0.05,
        vol_lookback_windows=2,
        high_vol_annualized_threshold=0.20,
        horizon_sessions=5,
    )

    labeled = detector.label_frame(frame)

    assert bool(labeled.loc[5, "regime_high_vol"]) is True
    assert labeled.loc[5, "regime_label"] in {"bear", "correction"}
    assert "regime_realized_vol_annualized" in labeled.columns


def test_cross_asset_detector_uses_lagged_risk_on_spread() -> None:
    frame = _frame([0.01, 0.01, 0.01, 0.01, -0.01, -0.01, 0.01, 0.01])
    frame["spy_ret"] = [0.02, 0.02, 0.02, 0.02, -0.03, -0.03, -0.03, -0.03]
    frame["vxus_ret"] = [0.02, 0.02, 0.02, 0.02, -0.03, -0.03, -0.03, -0.03]
    frame["agg_ret"] = [0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01]
    frame["cta_ret"] = [0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01]
    frame["gldm_ret"] = [0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01]
    frame["sgov_ret"] = [0.0] * len(frame)

    detector = BenchmarkTrendDrawdownVolCrossAssetRegimeDetector(
        trend_lookback_windows=2,
        drawdown_threshold=-0.05,
        vol_lookback_windows=2,
        high_vol_annualized_threshold=0.50,
        horizon_sessions=5,
        cross_asset_lookback_windows=2,
        cross_asset_risk_on_threshold=0.0,
    )

    labeled = detector.label_frame(frame)

    assert "regime_cross_asset_spread" in labeled.columns
    assert bool(labeled.loc[4, "regime_risk_on_supportive"]) is True
    assert bool(labeled.loc[7, "regime_risk_on_supportive"]) is False
    assert labeled.loc[7, "regime_label"] in {"correction", "bear"}
