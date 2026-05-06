from __future__ import annotations

from pathlib import Path

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase5o import (
    _add_non_price_long_scores,
    _position_non_price_exposure_summary,
)


def test_add_non_price_long_scores_builds_veto_and_soft_overlay(tmp_path: Path):
    panel = pd.DataFrame(
        [
            {
                "session_date": "2019-01-02",
                "variant": "top1000_clean_core_beta_full",
                "symbol": "AAA",
                "reversal_5d": 3.0,
                "momentum_20d": -0.20,
                "momentum_60d": -0.30,
                "beta": 0.80,
            },
            {
                "session_date": "2019-01-02",
                "variant": "top1000_clean_core_beta_full",
                "symbol": "BBB",
                "reversal_5d": 2.0,
                "momentum_20d": 0.30,
                "momentum_60d": 0.40,
                "beta": 1.50,
            },
            {
                "session_date": "2019-01-02",
                "variant": "top1000_clean_core_beta_full",
                "symbol": "CCC",
                "reversal_5d": 1.0,
                "momentum_20d": 0.00,
                "momentum_60d": 0.10,
                "beta": 1.00,
            },
        ]
    )
    non_price = pd.DataFrame(
        [
            {
                "session_date": "2019-01-02",
                "symbol": "AAA",
                "filing_red_flag_events_20d": 3.0,
                "red_8k_events_60d": 2.0,
                "periodic_delay_events_252d": 1.0,
                "days_since_last_filing_red_flag": 1.0,
                "insider_sell_events_20d": 0.0,
                "insider_sell_intensity_60d": 0.0,
                "insider_net_buy_score": 1.0,
            },
            {
                "session_date": "2019-01-02",
                "symbol": "BBB",
                "filing_red_flag_events_20d": 0.0,
                "red_8k_events_60d": 0.0,
                "periodic_delay_events_252d": 0.0,
                "days_since_last_filing_red_flag": 9999.0,
                "insider_sell_events_20d": 0.0,
                "insider_sell_intensity_60d": 0.0,
                "insider_net_buy_score": 2.0,
            },
            {
                "session_date": "2019-01-02",
                "symbol": "CCC",
                "filing_red_flag_events_20d": 0.0,
                "red_8k_events_60d": 0.0,
                "periodic_delay_events_252d": 0.0,
                "days_since_last_filing_red_flag": 9999.0,
                "insider_sell_events_20d": 0.0,
                "insider_sell_intensity_60d": 0.0,
                "insider_net_buy_score": 3.0,
            },
        ]
    )
    non_price_path = tmp_path / "non_price.csv"
    non_price.to_csv(non_price_path, index=False)

    scored = _add_non_price_long_scores(
        panel,
        long_non_price_panel_path=non_price_path,
        long_variant="top1000_clean_core_beta_full",
        trigger_threshold=0.75,
        recent_filing_days=20,
        soft_penalty_weight=0.75,
    )

    aaa = scored[scored["symbol"].eq("AAA")].iloc[0]
    bbb = scored[scored["symbol"].eq("BBB")].iloc[0]
    assert aaa["trigger_long_filing_distress"]
    assert pd.isna(aaa["long_np_filing_veto_t075"])
    assert bbb["long_np_filing_veto_t075"] == 2.0
    assert scored["long_np_promising_soft_overlay"].notna().all()


def test_add_non_price_long_scores_builds_fundamental_fragility_variants(tmp_path: Path):
    panel = pd.DataFrame(
        [
            {
                "session_date": "2019-01-02",
                "variant": "top1000_clean_core_beta_full",
                "symbol": "AAA",
                "reversal_5d": 3.0,
                "momentum_20d": -0.20,
                "momentum_60d": -0.30,
                "beta": 0.80,
            },
            {
                "session_date": "2019-01-02",
                "variant": "top1000_clean_core_beta_full",
                "symbol": "BBB",
                "reversal_5d": 2.0,
                "momentum_20d": 0.30,
                "momentum_60d": 0.40,
                "beta": 1.50,
            },
            {
                "session_date": "2019-01-02",
                "variant": "top1000_clean_core_beta_full",
                "symbol": "CCC",
                "reversal_5d": 1.0,
                "momentum_20d": 0.00,
                "momentum_60d": 0.10,
                "beta": 1.00,
            },
        ]
    )
    non_price = pd.DataFrame(
        [
            {
                "session_date": "2019-01-02",
                "symbol": symbol,
                "filing_red_flag_events_20d": 0.0,
                "red_8k_events_60d": 0.0,
                "periodic_delay_events_252d": 0.0,
                "days_since_last_filing_red_flag": 9999.0,
                "insider_sell_events_20d": 0.0,
                "insider_sell_intensity_60d": 0.0,
                "insider_net_buy_score": 0.0,
            }
            for symbol in ("AAA", "BBB", "CCC")
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {
                "session_date": "2019-01-02",
                "symbol": "AAA",
                "fundamental_leverage_pressure_score": 0.9,
                "fundamental_profit_stress_score": 0.8,
                "fundamental_fragility_score": 0.9,
            },
            {
                "session_date": "2019-01-02",
                "symbol": "BBB",
                "fundamental_leverage_pressure_score": 0.7,
                "fundamental_profit_stress_score": 0.74,
                "fundamental_fragility_score": 0.72,
            },
        ]
    )
    non_price_path = tmp_path / "non_price.csv"
    fundamental_path = tmp_path / "fundamentals.csv"
    non_price.to_csv(non_price_path, index=False)
    fundamentals.to_csv(fundamental_path, index=False)

    scored = _add_non_price_long_scores(
        panel,
        long_non_price_panel_path=non_price_path,
        long_fundamental_panel_path=fundamental_path,
        long_variant="top1000_clean_core_beta_full",
        trigger_threshold=0.75,
        fundamental_trigger_threshold=0.75,
        recent_filing_days=20,
        soft_penalty_weight=0.75,
    )

    aaa = scored[scored["symbol"].eq("AAA")].iloc[0]
    bbb = scored[scored["symbol"].eq("BBB")].iloc[0]
    ccc = scored[scored["symbol"].eq("CCC")].iloc[0]
    assert aaa["trigger_long_fundamental_fragility"]
    assert not aaa["trigger_long_filing_and_fundamental"]
    assert aaa["trigger_long_filing_or_fundamental"]
    assert pd.isna(aaa["long_np_fund_fragility_veto_t075"])
    assert aaa["long_np_filing_and_fund_veto_t075"] == 3.0
    assert pd.isna(aaa["long_np_filing_or_fund_veto_t075"])
    assert pd.isna(aaa["long_np_fof_low_mom20_veto_t075"])
    assert pd.isna(aaa["long_np_fof_low_mom60_veto_t075"])
    assert pd.isna(aaa["long_np_fof_low_beta_veto_t075"])
    assert pd.isna(aaa["long_np_fof_low_mom60_low_beta_veto_t075"])
    assert bbb["long_np_fund_fragility_veto_t075"] == 2.0
    assert pd.isna(bbb["long_np_filing_or_fund_veto_t070"])
    assert bbb["long_np_filing_or_fund_veto_t080"] == 2.0
    assert bbb["long_np_fof_low_mom20_veto_t075"] == 2.0
    assert bbb["long_np_fof_low_mom60_veto_t075"] == 2.0
    assert bbb["long_np_fof_low_beta_veto_t075"] == 2.0
    assert not ccc["has_long_fundamental_fragility_score"]
    assert scored["long_np_promising_plus_fund_soft_overlay"].notna().all()


def test_position_non_price_exposure_summary_uses_position_weights():
    positions = pd.DataFrame(
        [
            {
                "session_date": "2019-01-02",
                "portfolio": "p1",
                "side": "long",
                "symbol": "AAA",
                "side_weight": 0.75,
            },
            {
                "session_date": "2019-01-02",
                "portfolio": "p1",
                "side": "long",
                "symbol": "BBB",
                "side_weight": 0.25,
            },
            {
                "session_date": "2019-01-02",
                "portfolio": "p1",
                "side": "short",
                "symbol": "ZZZ",
                "side_weight": 1.0,
            },
        ]
    )
    panel = pd.DataFrame(
        [
            {
                "session_date": "2019-01-02",
                "symbol": "AAA",
                "long_filing_distress_score": 1.0,
                "long_insider_sell_pressure_score": 0.0,
                "long_np_promising_score": 0.5,
                "trigger_long_filing_distress": True,
                "trigger_long_insider_sell_pressure": False,
                "trigger_long_np_any_promising": True,
            },
            {
                "session_date": "2019-01-02",
                "symbol": "BBB",
                "long_filing_distress_score": 0.0,
                "long_insider_sell_pressure_score": 0.0,
                "long_np_promising_score": 0.0,
                "trigger_long_filing_distress": False,
                "trigger_long_insider_sell_pressure": False,
                "trigger_long_np_any_promising": False,
            },
        ]
    )

    summary = _position_non_price_exposure_summary(positions=positions, panel=panel)

    row = summary.iloc[0]
    assert row["portfolio"] == "p1"
    assert row["mean_selected_long_filing_trigger_weight_share"] == 0.75
    assert row["mean_selected_long_any_trigger_weight_share"] == 0.75
    assert row["mean_selected_long_np_promising_score"] == 0.375
