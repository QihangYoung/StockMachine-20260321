from __future__ import annotations

import pandas as pd

from stockmachine.apps.run_non_price_position_overlay_phase3 import (
    _add_signed_alpha,
    _event_intent,
    _overlay_summary,
)


def test_add_signed_alpha_flips_short_side():
    positions = pd.DataFrame(
        [
            {
                "session_date": "2019-01-02",
                "portfolio": "p1",
                "side": "long",
                "symbol": "AAPL",
                "forward_beta_residual_return_5d": 0.01,
            },
            {
                "session_date": "2019-01-02",
                "portfolio": "p1",
                "side": "short",
                "symbol": "MSFT",
                "forward_beta_residual_return_5d": -0.02,
            },
        ]
    )

    result = _add_signed_alpha(positions, target_column="forward_beta_residual_return_5d")

    assert result.loc[result["side"].eq("long"), "signed_alpha"].iloc[0] == 0.01
    assert result.loc[result["side"].eq("short"), "signed_alpha"].iloc[0] == 0.02


def test_event_intent_maps_family_and_side_to_confirmation_or_veto():
    assert _event_intent("filing_hard_red_event_v2", "short") == "short_confirmation"
    assert _event_intent("filing_hard_red_event_v2", "long") == "long_veto"
    assert _event_intent("insider_cluster_buy_event_v2", "long") == "long_confirmation"
    assert _event_intent("insider_cluster_buy_event_v2", "short") == "short_veto"


def test_overlay_summary_supports_short_confirmation():
    positions = pd.DataFrame(
        [
            {
                "session_date": "2019-01-02",
                "portfolio": "p1",
                "side": "short",
                "symbol": "AAPL",
                "signed_alpha": 0.03,
            },
            {
                "session_date": "2019-01-03",
                "portfolio": "p1",
                "side": "short",
                "symbol": "MSFT",
                "signed_alpha": 0.02,
            },
            {
                "session_date": "2019-01-02",
                "portfolio": "p1",
                "side": "short",
                "symbol": "NVDA",
                "signed_alpha": -0.01,
            },
            {
                "session_date": "2019-01-03",
                "portfolio": "p1",
                "side": "short",
                "symbol": "AMZN",
                "signed_alpha": 0.0,
            },
        ]
    )
    events = pd.DataFrame(
        [
            {
                "session_date": "2019-01-02",
                "symbol": "AAPL",
                "event_family": "filing_hard_red_event_v2",
                "event_count": 1,
                "event_strength": 1.0,
            },
            {
                "session_date": "2019-01-03",
                "symbol": "MSFT",
                "event_family": "filing_hard_red_event_v2",
                "event_count": 1,
                "event_strength": 1.0,
            },
        ]
    )

    summary = _overlay_summary(positions, events, min_event_rows=1)
    all_short = summary[summary["portfolio"].eq("ALL") & summary["side"].eq("short")].iloc[0]

    assert all_short["intended_use"] == "short_confirmation"
    assert all_short["edge_signed_alpha_mean"] > 0
    assert all_short["overlay_label"] == "overlay_supported"
