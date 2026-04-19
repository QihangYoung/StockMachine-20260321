from __future__ import annotations

import pandas as pd

from stockmachine.apps.run_non_price_factor_phase1 import (
    _build_decayed_panel,
    _filing_red_flag_components,
    _next_session_after,
    _quarter_labels,
    _sec_date_to_iso,
)


def test_filing_red_flag_components_weight_nt_and_red_8k_items():
    record = {
        "form": "8-K",
        "items": "4.02,2.06,9.01",
        "filing_date": "2019-01-15",
        "report_date": "2018-12-31",
    }

    result = _filing_red_flag_components(record)

    assert result["red_8k_event_weight"] == 6.0
    assert result["filing_red_flag_event_weight"] == 6.0
    assert result["red_8k_items"] == "4.02|2.06"


def test_filing_red_flag_components_weight_nt_10q():
    result = _filing_red_flag_components({"form": "NT 10-Q", "items": ""})

    assert result["nt_10kq_event_weight"] == 3.0
    assert result["filing_red_flag_event_weight"] == 3.0


def test_next_session_after_is_strictly_after_event_date():
    sessions = ["2019-01-02", "2019-01-03", "2019-01-04"]

    assert _next_session_after("2019-01-02", sessions) == "2019-01-03"
    assert _next_session_after("2019-01-04", sessions) is None


def test_quarter_labels_cover_partial_years():
    assert _quarter_labels("2013-08-05", "2014-02-01") == ["2013q3", "2013q4", "2014q1"]


def test_sec_date_to_iso_parses_sec_quarterly_dataset_dates():
    assert _sec_date_to_iso("31-MAR-2026") == "2026-03-31"
    assert _sec_date_to_iso("2026-03-31T17:00:00.000Z") == "2026-03-31"


def test_build_decayed_panel_adds_score_and_event_counts():
    base = pd.DataFrame(
        [
            {"session_date": "2019-01-02", "symbol": "AAPL"},
            {"session_date": "2019-01-03", "symbol": "AAPL"},
            {"session_date": "2019-01-04", "symbol": "AAPL"},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "effective_session": "2019-01-03",
                "event_score": 2.0,
                "event_flag": 1.0,
            }
        ]
    )

    panel = _build_decayed_panel(
        base=base,
        events=events,
        session_index={"2019-01-02": 0, "2019-01-03": 1, "2019-01-04": 2},
        score_column="event_score",
        output_score_column="score",
        component_specs={"event_count_20d": ("event_flag", 20)},
        days_since_column="days_since_event",
        output_columns=["score", "event_count_20d", "days_since_event"],
        half_life_sessions=1.0,
        max_age_sessions=2,
    )

    assert panel["score"].tolist() == [0.0, 2.0, 1.0]
    assert panel["event_count_20d"].tolist() == [0.0, 1.0, 1.0]
    assert panel["days_since_event"].tolist() == [9999.0, 0.0, 1.0]
