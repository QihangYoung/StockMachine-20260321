from __future__ import annotations

import pandas as pd

from stockmachine.apps.run_non_price_phase5n import (
    _basket_state_verdict,
    _build_state_frame,
    _factor_verdict_summary,
    _mean_available,
    _score_long_candidates,
    _window_rows,
)


def test_mean_available_ignores_missing_values():
    result = _mean_available(
        [
            pd.Series([1.0, None, 3.0]),
            pd.Series([3.0, 5.0, None]),
        ]
    )

    assert result.tolist() == [2.0, 5.0, 3.0]


def test_score_long_candidates_sets_distress_triggers():
    candidates = pd.DataFrame(
        [
            {
                "session_date": "2019-01-02",
                "variant": "top1000",
                "symbol": "AAPL",
                "beta": 1.0,
                "base_score": 0.9,
                "forward_beta_residual_return_5d": -0.02,
                "forward_return_5d": -0.01,
                "benchmark_forward_return_5d": 0.0,
                "candidate_rank": 1,
                "side": "long",
                "base_score_column": "reversal_5d",
            }
        ]
    )
    non_price = pd.DataFrame(
        [
            {
                "session_date": "2019-01-02",
                "symbol": "AAPL",
                "filing_red_flag_events_20d": 4.0,
                "filing_red_flag_events_20d_pct": 0.95,
                "red_8k_events_60d": 2.0,
                "red_8k_events_60d_pct": 0.90,
                "periodic_delay_events_252d": 1.0,
                "periodic_delay_events_252d_pct": 0.85,
                "days_since_last_filing_red_flag": 3.0,
                "days_since_last_filing_red_flag_inverse_pct": 0.98,
                "insider_sell_events_20d": 2.0,
                "insider_sell_events_20d_pct": 0.80,
                "insider_sell_intensity_60d": 1.5,
                "insider_sell_intensity_60d_pct": 0.88,
                "insider_net_buy_score": -10.0,
                "insider_net_buy_score_inverse_pct": 0.92,
            }
        ]
    )

    scored = _score_long_candidates(
        candidates=candidates,
        non_price=non_price,
        recent_filing_days=20,
    )

    assert scored.iloc[0]["recent_filing_red_flag"] == 1.0
    assert scored.iloc[0]["long_filing_distress_score"] > 0.9
    assert scored.iloc[0]["long_insider_sell_pressure_score"] > 0.8
    assert scored.iloc[0]["trigger_long_non_price_distress"]


def test_build_state_frame_combines_long_and_short_phi():
    long_candidates = pd.DataFrame(
        [
            {
                "session_date": "2019-01-02",
                "long_filing_distress_score": 0.8,
                "long_insider_sell_pressure_score": 0.7,
                "long_non_price_distress_score": 0.9,
            },
            {
                "session_date": "2019-01-02",
                "long_filing_distress_score": 0.2,
                "long_insider_sell_pressure_score": 0.1,
                "long_non_price_distress_score": 0.3,
            },
        ]
    )
    short_candidates = pd.DataFrame(
        [
            {
                "session_date": "2019-01-02",
                "short_insider_buy_support_score": 0.8,
                "recent_insider_buy": 1.0,
                "insider_cluster_buy_events_60d": 2.0,
                "short_non_price_improver_score": 0.85,
                "short_filing_confirmation_score": 0.1,
            },
            {
                "session_date": "2019-01-02",
                "short_insider_buy_support_score": 0.2,
                "recent_insider_buy": 0.0,
                "insider_cluster_buy_events_60d": 0.0,
                "short_non_price_improver_score": 0.25,
                "short_filing_confirmation_score": 0.9,
            },
        ]
    )
    price_state = pd.DataFrame(
        [
            {
                "session_date": "2019-01-02",
                "whitebox_state_score_a": 0.4,
                "whitebox_spy_ret20": -0.01,
            }
        ]
    )

    state = _build_state_frame(
        long_candidates=long_candidates,
        short_candidates=short_candidates,
        trigger_threshold=0.75,
        price_state=price_state,
    )

    assert len(state) == 1
    assert state.iloc[0]["share_long_filing_distress_hi"] == 0.5
    assert state.iloc[0]["share_short_insider_buy_support_hi"] == 0.5
    assert state.iloc[0]["phi_long_non_price"] > 0.0
    assert state.iloc[0]["phi_short_non_price"] > 0.0


def test_window_rows_labels_long_and_short_support_correctly():
    long_frame = pd.DataFrame(
        [
            {
                "long_non_price_distress_score": 0.9,
                "trigger_long_non_price_distress": True,
                "forward_beta_residual_return_5d": -0.03,
            },
            {
                "long_non_price_distress_score": 0.1,
                "trigger_long_non_price_distress": False,
                "forward_beta_residual_return_5d": 0.01,
            },
        ]
    )
    short_frame = pd.DataFrame(
        [
            {
                "short_non_price_improver_score": 0.9,
                "trigger_short_non_price_improver": True,
                "forward_beta_residual_return_5d": 0.03,
            },
            {
                "short_non_price_improver_score": 0.1,
                "trigger_short_non_price_improver": False,
                "forward_beta_residual_return_5d": -0.01,
            },
        ]
    )

    long_rows = _window_rows(
        long_frame,
        window="w1",
        side="long",
        factor_specs=(("long_non_price_distress_score", "trigger_long_non_price_distress"),),
        trigger_threshold=0.75,
    )
    short_rows = _window_rows(
        short_frame,
        window="w2",
        side="short",
        factor_specs=(("short_non_price_improver_score", "trigger_short_non_price_improver"),),
        trigger_threshold=0.75,
    )

    assert long_rows[0]["label"] == "culprit_supported"
    assert short_rows[0]["label"] == "culprit_supported"


def test_factor_verdict_summary_counts_supported_windows_and_trigger_share():
    candidate_frame = pd.DataFrame(
        [
            {"side": "long", "trigger_long_filing_distress": True},
            {"side": "long", "trigger_long_filing_distress": False},
            {"side": "long", "trigger_long_filing_distress": True},
            {"side": "short", "trigger_short_filing_confirmation": False},
            {"side": "short", "trigger_short_filing_confirmation": True},
        ]
    )
    culprit_summary = pd.DataFrame(
        [
            {
                "factor_name": "long_filing_distress_score",
                "label": "culprit_supported",
                "trigger_edge_vs_rest": -0.01,
            },
            {
                "factor_name": "long_filing_distress_score",
                "label": "culprit_not_supported",
                "trigger_edge_vs_rest": 0.02,
            },
            {
                "factor_name": "short_filing_confirmation_score",
                "label": "culprit_supported",
                "trigger_edge_vs_rest": 0.01,
            },
        ]
    )

    verdict = _factor_verdict_summary(candidate_frame, culprit_summary)

    long_row = verdict[verdict["factor_name"].eq("long_filing_distress_score")].iloc[0]
    short_row = verdict[verdict["factor_name"].eq("short_filing_confirmation_score")].iloc[0]
    assert long_row["overall_trigger_share"] == 2 / 3
    assert long_row["supported_windows"] == 1
    assert long_row["verdict"] == "partial_support"
    assert short_row["overall_trigger_share"] == 0.5
    assert short_row["verdict"] == "partial_support"


def test_basket_state_verdict_requires_majority_of_windows():
    window_state_summary = pd.DataFrame(
        [
            {"side": "long", "phi_edge_vs_full_mean": 0.01},
            {"side": "long", "phi_edge_vs_full_mean": 0.02},
            {"side": "long", "phi_edge_vs_full_mean": -0.01},
            {"side": "long", "phi_edge_vs_full_mean": -0.02},
            {"side": "short", "phi_edge_vs_full_mean": 0.01},
            {"side": "short", "phi_edge_vs_full_mean": -0.02},
            {"side": "short", "phi_edge_vs_full_mean": -0.03},
        ]
    )

    verdict = _basket_state_verdict(window_state_summary)

    long_row = verdict[verdict["side"].eq("long")].iloc[0]
    short_row = verdict[verdict["side"].eq("short")].iloc[0]
    assert long_row["windows_with_phi_rise"] == 2
    assert long_row["verdict"] == "mixed_signal"
    assert short_row["windows_with_phi_rise"] == 1
    assert short_row["verdict"] == "mixed_signal"
