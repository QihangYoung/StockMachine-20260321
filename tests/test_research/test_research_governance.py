from __future__ import annotations

import pandas as pd

from stockmachine.research.research_governance import (
    ResearchSourceInputs,
    build_research_data_coverage_assessment,
)


def _source_inputs(*, membership_last_date: str) -> ResearchSourceInputs:
    session_dates = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
        ]
    )
    dataset = {
        "daily_bar": pd.DataFrame(
            {
                "session_date": session_dates,
                "symbol": ["AAPL", "AAPL", "AAPL"],
            }
        ),
        "benchmark_index": pd.DataFrame(
            {
                "session_date": session_dates,
                "symbol": ["SPY", "SPY", "SPY"],
            }
        ),
        "adj_factor": pd.DataFrame(
            {
                "session_date": session_dates,
                "symbol": ["AAPL", "AAPL", "AAPL"],
            }
        ),
        "symbol_master": pd.DataFrame(
            {
                "as_of_date": pd.to_datetime(["2025-01-06"]),
                "symbol": ["AAPL"],
            }
        ),
        "industry_membership": pd.DataFrame(
            {
                "as_of_date": pd.to_datetime([membership_last_date]),
                "symbol": ["AAPL"],
                "industry_system": ["gics"],
            }
        ),
        "universe_membership": pd.DataFrame(
            {
                "session_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "universe_name": ["us_equities_research_v1", "us_equities_research_v1"],
                "symbol": ["AAPL", "AAPL"],
                "is_member": [True, True],
            }
        ),
    }
    price_data = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL", "SPY", "SPY", "SPY"],
            "date": pd.to_datetime(
                ["2025-01-02", "2025-01-03", "2025-01-06"] * 2
            ),
        }
    )
    return ResearchSourceInputs(
        dataset=dataset,
        price_data=price_data,
        session_dates=pd.Index(session_dates),
    )


def test_build_research_data_coverage_assessment_honors_expected_last_session() -> None:
    assessment = build_research_data_coverage_assessment(
        source_inputs=_source_inputs(membership_last_date="2025-01-03"),
        expected_last_session=pd.Timestamp("2025-01-03").date(),
    )

    assert assessment.ok is True
    assert assessment.research_last_session_date == pd.Timestamp("2025-01-03").date()
    assert assessment.coverage_meta["missing_session_count"] == 0


def test_build_research_data_coverage_assessment_flags_stale_tail_when_expected_last_session_extends() -> None:
    assessment = build_research_data_coverage_assessment(
        source_inputs=_source_inputs(membership_last_date="2025-01-03"),
        expected_last_session=pd.Timestamp("2025-01-06").date(),
    )

    assert assessment.ok is False
    assert "stale_industry_membership" in assessment.reasons
    assert "universe_membership_incomplete" in assessment.reasons
    assert assessment.coverage_meta["missing_sessions_preview"] == ["2025-01-06"]
