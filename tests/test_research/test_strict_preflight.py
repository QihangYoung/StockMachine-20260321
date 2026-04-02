from __future__ import annotations

import pandas as pd

from stockmachine.research.strict_preflight import (
    StrictResearchSourceInputs,
    assess_universe_membership_coverage,
    build_strict_research_preflight,
)


def _source_inputs(*, membership_last_date: str) -> StrictResearchSourceInputs:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    price_data = pd.DataFrame(
        [
            {
                "date": current_date,
                "symbol": symbol,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000_000.0,
                "adj_open": 100.0,
                "adj_close": 100.5,
            }
            for current_date in dates
            for symbol in ("AAA", "SPY")
        ]
    )
    dataset = {
        "daily_bar": pd.DataFrame({"session_date": dates, "symbol": ["AAA", "AAA", "AAA"]}),
        "benchmark_index": pd.DataFrame({"session_date": dates, "symbol": ["SPY", "SPY", "SPY"]}),
        "adj_factor": pd.DataFrame({"session_date": dates, "symbol": ["AAA", "AAA", "AAA"]}),
        "symbol_master": pd.DataFrame(
            [{"as_of_date": dates[-1], "symbol": "AAA", "is_active": True}]
        ),
        "industry_membership": pd.DataFrame(
            [{"as_of_date": dates[-1], "symbol": "AAA", "industry_system": "gics"}]
        ),
        "universe_membership": pd.DataFrame(
            [
                {
                    "session_date": session_date,
                    "symbol": "AAA",
                    "universe_name": "us_equities_research_v1",
                    "is_member": True,
                }
                for session_date in dates[dates <= pd.Timestamp(membership_last_date)]
            ]
        ),
    }
    session_dates = pd.Index(pd.to_datetime(dates).normalize())
    return StrictResearchSourceInputs(dataset=dataset, price_data=price_data, session_dates=session_dates)


def test_assess_universe_membership_coverage_reports_missing_tail_dates() -> None:
    session_dates = pd.Index(pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]))
    membership = pd.DataFrame(
        [
            {"session_date": "2025-01-02", "symbol": "AAA", "universe_name": "us_equities_research_v1", "is_member": True},
            {"session_date": "2025-01-03", "symbol": "AAA", "universe_name": "us_equities_research_v1", "is_member": True},
        ]
    )
    result = assess_universe_membership_coverage(
        session_dates=session_dates,
        universe_membership_frame=membership,
        universe_name="us_equities_research_v1",
    )

    assert result["ok"] is False
    assert result["reason"] == "universe_membership_incomplete"
    assert result["missing_session_count"] == 1
    assert result["missing_sessions_preview"] == ["2025-01-06"]


def test_build_strict_research_preflight_flags_stale_coverage_from_source_inputs() -> None:
    preflight = build_strict_research_preflight(
        horizon=1,
        strategy_project="us_equities_h1",
        source_inputs=_source_inputs(membership_last_date="2025-01-03"),
    )

    assert preflight.ok is False
    assert "universe_membership_incomplete" in preflight.reasons
    assert preflight.coverage_meta["missing_sessions_preview"] == ["2025-01-06"]
