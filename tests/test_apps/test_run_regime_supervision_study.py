from __future__ import annotations

import pandas as pd

from stockmachine.apps.run_regime_supervision_study import _summarize_by_label
from stockmachine.risk import (
    BenchmarkForwardRegimeLabeler,
    BenchmarkTrendDrawdownRegimeDetector,
    build_regime_confusion_matrix,
)


def _records(benchmark_returns: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=len(benchmark_returns), freq="5B")
    return pd.DataFrame(
        {
            "entry_date": dates,
            "benchmark_return": benchmark_returns,
        }
    )


def test_regime_supervision_components_build_expected_artifacts() -> None:
    records = _records([0.03, 0.02, 0.01, -0.02, -0.01, -0.01, -0.02, 0.03])
    detector = BenchmarkTrendDrawdownRegimeDetector(trend_lookback_windows=2, drawdown_threshold=-0.02)
    labeler = BenchmarkForwardRegimeLabeler(forward_windows=3, drawdown_threshold=-0.02)

    detected = detector.label_frame(records, return_column="benchmark_return", date_column="entry_date")
    forward = labeler.label_frame(records, return_column="benchmark_return", date_column="entry_date")

    combined = records.merge(detected, on="entry_date", how="left").merge(forward, on="entry_date", how="left")
    evaluable = combined.loc[
        (combined["regime_label"] != "warmup") & (combined["forward_regime_label"] != "unlabeled")
    ].reset_index(drop=True)

    confusion = build_regime_confusion_matrix(evaluable)
    detected_summary = _summarize_by_label(evaluable, "regime_label")
    forward_summary = _summarize_by_label(evaluable, "forward_regime_label")

    assert not evaluable.empty
    assert confusion.values.sum() == len(evaluable)
    assert "mean_forward_total_return" in detected_summary.columns
    assert "mean_forward_max_drawdown" in forward_summary.columns
