from __future__ import annotations

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4l import (
    build_phase4l_top2000_feasibility_artifacts,
)


def test_phase4l_blocks_top2000_without_signal_panel_membership(tmp_path) -> None:
    panel_path = _write_signal_panel(tmp_path)
    phase1_path = _write_phase1_summary(tmp_path)
    phase4k_path = _write_phase4k_summary(tmp_path)

    result = build_phase4l_top2000_feasibility_artifacts(
        signal_panel_path=panel_path,
        phase1_summary_path=phase1_path,
        phase4k_summary_path=phase4k_path,
        output_root=tmp_path / "phase4l",
    )
    gate = pd.read_csv(tmp_path / "phase4l" / "phase4l_top2000_gate_validation.csv")

    top2000 = gate.loc[gate["variant"] == "top2000_clean_core_beta_full"].iloc[0]
    assert result["top2000_status"] == "blocked"
    assert top2000["available_in_phase3_panel"] == False
    assert top2000["primary_blocker"] == "missing_from_phase3_signal_panel"
    assert gate["test_window_used"].eq(False).all()


def test_phase4l_writes_proxy_evidence_when_phase4k_exists(tmp_path) -> None:
    build_phase4l_top2000_feasibility_artifacts(
        signal_panel_path=_write_signal_panel(tmp_path),
        phase1_summary_path=_write_phase1_summary(tmp_path),
        phase4k_summary_path=_write_phase4k_summary(tmp_path),
        output_root=tmp_path / "phase4l",
    )
    proxy = pd.read_csv(tmp_path / "phase4l" / "phase4l_proxy_universe_evidence_validation.csv")

    assert proxy.loc[0, "pair"] == "top1000_clean_core_beta_full__short_adv30m_clean_core_beta_full"
    assert proxy.loc[0, "proxy_rank"] == 1


def _write_signal_panel(tmp_path):
    rows = []
    for index in range(3):
        rows.append(
            {
                "session_date": "2020-01-02",
                "variant": "top1000_clean_core_beta_full",
                "symbol": f"SYM{index:03d}",
                "lagged_close": 20.0 + index,
                "trailing_median_dollar_volume_20": 1_000_000.0,
                "liquidity_rank": index + 1,
                "beta": 1.0,
                "forward_return_5d": 0.01,
            }
        )
    path = tmp_path / "signal_panel.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    return path


def _write_phase1_summary(tmp_path):
    rows = [
        {
            "variant": "top1000_clean_core_beta_full",
            "data_status": "supported_current_top1000_scope",
            "median_members": 600,
            "sessions_with_members": 1,
        },
        {
            "variant": "top1500_clean_core_beta_full",
            "data_status": "blocked_current_backfill_has_only_top1000_symbols",
            "median_members": 0,
            "sessions_with_members": 0,
        },
        {
            "variant": "top2000_clean_core_beta_full",
            "data_status": "blocked_current_backfill_has_only_top1000_symbols",
            "median_members": 0,
            "sessions_with_members": 0,
        },
        {
            "variant": "top3000_clean_core_beta_full",
            "data_status": "blocked_current_backfill_has_only_top1000_symbols",
            "median_members": 0,
            "sessions_with_members": 0,
        },
    ]
    path = tmp_path / "phase1_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_phase4k_summary(tmp_path):
    rows = [
        {
            "pair": "top1000_clean_core_beta_full__short_adv30m_clean_core_beta_full",
            "mean_spread_cs_demeaned_residual": 0.002,
            "mean_spread_return": 0.001,
            "mean_short_cs_demeaned_residual_contribution": 0.0015,
            "mean_abs_net_beta": 0.0,
            "test_window_used": False,
        },
        {
            "pair": "top1000_clean_core_beta_full__short_adv20m_clean_core_beta_full",
            "mean_spread_cs_demeaned_residual": 0.001,
            "mean_spread_return": 0.0005,
            "mean_short_cs_demeaned_residual_contribution": 0.0012,
            "mean_abs_net_beta": 0.0,
            "test_window_used": False,
        },
    ]
    path = tmp_path / "phase4k_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
