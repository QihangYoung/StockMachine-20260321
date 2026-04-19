from __future__ import annotations

import pandas as pd

from stockmachine.apps.run_non_price_event_phase2 import (
    _build_cluster_buy_events,
    _build_hard_red_events,
    _event_summary,
    _hard_red_categories,
    _next_session_after,
)


def test_hard_red_categories_keeps_only_high_confidence_items():
    assert _hard_red_categories({"form": "8-K", "items": "2.02,9.01"}) == []
    assert _hard_red_categories({"form": "8-K", "items": "4.02,2.06,9.01"}) == [
        "material_impairment",
        "non_reliance_or_restatement",
    ]
    assert _hard_red_categories({"form": "NT 10-Q", "items": ""}) == ["late_10kq"]


def test_next_session_after_is_strictly_after_event_date():
    sessions = ["2019-01-02", "2019-01-03", "2019-01-04"]

    assert _next_session_after("2019-01-02", sessions) == "2019-01-03"
    assert _next_session_after("2019-01-04", sessions) is None


def test_build_hard_red_events_drops_stale_pre_window_filings(tmp_path):
    path = tmp_path / "filings.csv"
    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "cik": "0000320193",
                "accession_number": "old",
                "filing_date": "2018-12-01",
                "form": "8-K",
                "items": "4.02",
                "primary_doc_description": "",
            },
            {
                "symbol": "AAPL",
                "cik": "0000320193",
                "accession_number": "fresh",
                "filing_date": "2019-01-03",
                "form": "8-K",
                "items": "2.04",
                "primary_doc_description": "",
            },
        ]
    ).to_csv(path, index=False)

    events = _build_hard_red_events(
        filings_path=path,
        sessions=["2019-01-03", "2019-01-04"],
        signal_symbols={"AAPL"},
    )

    assert len(events) == 1
    assert events.iloc[0]["accession_number"] == "fresh"
    assert events.iloc[0]["effective_session"] == "2019-01-04"


def test_build_cluster_buy_events_requires_window_cluster(tmp_path):
    path = tmp_path / "insider_events.csv"
    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "cik": "0000320193",
                "accession_number": "a1",
                "filing_date": "2019-01-02",
                "effective_session": "2019-01-03",
                "buy_value_role_weighted": 1000.0,
                "buy_owner_count": 1,
                "insider_buy_intensity": 0.5,
            },
            {
                "symbol": "AAPL",
                "cik": "0000320193",
                "accession_number": "a2",
                "filing_date": "2019-01-04",
                "effective_session": "2019-01-07",
                "buy_value_role_weighted": 2000.0,
                "buy_owner_count": 1,
                "insider_buy_intensity": 0.7,
            },
            {
                "symbol": "MSFT",
                "cik": "0000789019",
                "accession_number": "m1",
                "filing_date": "2019-01-02",
                "effective_session": "2019-01-03",
                "buy_value_role_weighted": 1000.0,
                "buy_owner_count": 1,
                "insider_buy_intensity": 0.5,
            },
        ]
    ).to_csv(path, index=False)
    sessions = ["2019-01-02", "2019-01-03", "2019-01-04", "2019-01-07"]

    events = _build_cluster_buy_events(
        insider_events_path=path,
        sessions=sessions,
        session_index={session: index for index, session in enumerate(sessions)},
        signal_symbols={"AAPL", "MSFT"},
        cluster_window_sessions=3,
        min_cluster_buy_events=2,
        min_cluster_buy_owners=2,
    )

    assert len(events) == 1
    assert events.iloc[0]["symbol"] == "AAPL"
    assert events.iloc[0]["effective_session"] == "2019-01-07"
    assert events.iloc[0]["cluster_buy_event_count"] == 2


def test_event_summary_labels_supported_short_direction():
    events = pd.DataFrame(
        [
            {
                "event_family": "filing_hard_red_event_v2",
                "event_direction": "short",
                "event_subtype": "late_10kq",
                "symbol": "AAPL",
                "forward_beta_residual_return_5d": -0.02,
            },
            {
                "event_family": "filing_hard_red_event_v2",
                "event_direction": "short",
                "event_subtype": "late_10kq",
                "symbol": "MSFT",
                "forward_beta_residual_return_5d": -0.01,
            },
        ]
    )
    baseline = pd.DataFrame(
        [
            {
                "target_median": 0.001,
                "target_trimmed_mean_10_90": 0.001,
                "bottom_loser_share": 0.2,
                "top_winner_share": 0.2,
            }
        ]
    )

    summary = _event_summary(
        events,
        target_column="forward_beta_residual_return_5d",
        bottom_cutoff=-0.005,
        top_cutoff=0.005,
        baseline=baseline,
    )

    assert summary.iloc[0]["intended_direction_label"] == "intended_direction_supported"
