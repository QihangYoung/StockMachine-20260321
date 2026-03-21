import pandas as pd
import pytest

from stockmachine.research.comparison import (
    build_yearly_holdout_windows,
    comparison_summary_frame,
    slice_frame_by_windows,
    validate_aligned_frames,
)
from stockmachine.research.us_equities_baseline import PeriodMetrics


def test_build_yearly_holdout_windows_creates_open_test_window() -> None:
    validation, test = build_yearly_holdout_windows(validation_year=2024)

    assert validation.name == "validation"
    assert validation.start == "2024-01-01"
    assert validation.end == "2025-01-01"
    assert test.name == "test"
    assert test.start == "2025-01-01"
    assert test.end is None


def test_slice_frame_by_windows_keeps_boundary_dates_out_of_next_window() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-12-31", "2025-01-01", "2025-06-30"]),
            "symbol": ["AAA", "BBB", "CCC"],
            "score": [3.0, 2.0, 1.0],
        }
    )
    validation, test = build_yearly_holdout_windows(validation_year=2024)

    slices = slice_frame_by_windows(frame, (validation, test))

    assert slices["validation"]["symbol"].tolist() == ["AAA"]
    assert slices["test"]["symbol"].tolist() == ["BBB", "CCC"]


def test_validate_aligned_frames_rejects_misaligned_inputs() -> None:
    aligned_a = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "symbol": ["AAA", "BBB"],
            "score": [1.0, 2.0],
        }
    )
    aligned_b = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "symbol": ["AAA", "BBB"],
            "score": [2.0, 3.0],
        }
    )

    alignment = validate_aligned_frames({"factor": aligned_a, "ridge": aligned_b}, required_columns=("score",))

    assert alignment.frame_names == ("factor", "ridge")
    assert alignment.row_counts == (("factor", 2), ("ridge", 2))
    assert list(alignment.shared_dates) == list(pd.to_datetime(["2025-01-02", "2025-01-03"]))

    misaligned_dates = aligned_b.iloc[:-1].copy()
    with pytest.raises(ValueError, match="is not aligned"):
        validate_aligned_frames({"factor": aligned_a, "ridge": misaligned_dates}, required_columns=("score",))

    misaligned_schema = aligned_b.rename(columns={"score": "rank_score"})
    with pytest.raises(KeyError, match="missing required columns"):
        validate_aligned_frames({"factor": aligned_a, "ridge": misaligned_schema}, required_columns=("score",))


def test_comparison_summary_frame_flattens_dataclass_metrics() -> None:
    metrics = PeriodMetrics(
        days=2,
        samples=10,
        mean_rank_ic=0.1,
        rank_ic_ir=1.5,
        mean_top_bottom_spread=0.02,
        top_k_total_return=0.05,
        top_k_annualized_return=0.12,
        top_k_sharpe=1.25,
        benchmark_total_return=0.01,
        annualized_excess_return=0.11,
        hit_rate=0.6,
    )

    summary = comparison_summary_frame({"ridge": {"validation": metrics}})

    assert summary.columns.tolist() == [
        "model",
        "period",
        "days",
        "samples",
        "mean_rank_ic",
        "rank_ic_ir",
        "mean_top_bottom_spread",
        "top_k_total_return",
        "top_k_annualized_return",
        "top_k_sharpe",
        "benchmark_total_return",
        "annualized_excess_return",
        "hit_rate",
    ]
    assert summary.iloc[0]["model"] == "ridge"
    assert summary.iloc[0]["period"] == "validation"
    assert summary.iloc[0]["days"] == 2
