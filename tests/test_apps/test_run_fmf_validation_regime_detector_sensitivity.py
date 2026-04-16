import pandas as pd

from stockmachine.apps.run_fmf_validation_regime_detector_sensitivity import (
    build_reference_regime_segments,
    expand_segment_frame_to_daily,
    rank_detector_summary,
    summarize_detector_against_reference,
)


def test_expand_segment_frame_to_daily_and_summary() -> None:
    reference_segments = pd.DataFrame(
        [
            {
                "segment_start": pd.Timestamp("2026-01-01"),
                "segment_end": pd.Timestamp("2026-01-05"),
                "reference_state": "risk_on",
            },
            {
                "segment_start": pd.Timestamp("2026-01-06"),
                "segment_end": pd.Timestamp("2026-01-10"),
                "reference_state": "defensive",
            },
        ]
    )
    reference_daily = expand_segment_frame_to_daily(
        reference_segments,
        start_column="segment_start",
        end_column="segment_end",
        state_column="reference_state",
        start_date=pd.Timestamp("2026-01-01"),
        end_date=pd.Timestamp("2026-01-10"),
    )
    regime_state_frame = pd.DataFrame(
        {
            "entry_date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "overlay_state": [
                "risk_on",
                "risk_on",
                "neutral",
                "risk_on",
                "risk_on",
                "defensive",
                "defensive",
                "neutral",
                "defensive",
                "defensive",
            ],
        }
    )
    summary = summarize_detector_against_reference(
        regime_state_frame,
        reference_daily=reference_daily,
    )

    assert summary["days"] == 10
    assert summary["daily_match_ratio"] == 0.8
    assert summary["overlay_defensive_share"] == 0.4


def test_rank_detector_summary_sorts_by_weighted_score() -> None:
    summary_frame = pd.DataFrame(
        [
            {
                "trend_lookback": 42,
                "vol_lookback": 21,
                "drawdown_threshold": -0.05,
                "high_vol_threshold": 0.12,
                "daily_match_ratio": 0.60,
                "q4_2018_match_ratio": 0.70,
                "q4_2018_defensive_recall": 0.80,
                "q4_2018_defensive_precision": 0.75,
                "overlay_risk_on_share": 0.70,
                "overlay_defensive_share": 0.10,
                "reference_risk_on_share": 0.80,
                "reference_defensive_share": 0.20,
                "days": 100,
            },
            {
                "trend_lookback": 63,
                "vol_lookback": 42,
                "drawdown_threshold": -0.06,
                "high_vol_threshold": 0.14,
                "daily_match_ratio": 0.65,
                "q4_2018_match_ratio": 0.60,
                "q4_2018_defensive_recall": 0.60,
                "q4_2018_defensive_precision": 0.60,
                "overlay_risk_on_share": 0.60,
                "overlay_defensive_share": 0.15,
                "reference_risk_on_share": 0.80,
                "reference_defensive_share": 0.20,
                "days": 100,
            },
        ]
    )
    ranked = rank_detector_summary(summary_frame)

    assert ranked.iloc[0]["trend_lookback"] == 42
    assert ranked.iloc[0]["rank"] == 1


def test_build_reference_regime_segments_returns_expected_shape() -> None:
    segments = build_reference_regime_segments()
    assert list(segments.columns) == [
        "segment_start",
        "segment_end",
        "reference_state",
        "reference_label",
    ]
    assert len(segments) == 5
