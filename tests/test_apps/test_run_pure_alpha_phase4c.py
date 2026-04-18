from __future__ import annotations

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4c import build_phase4c_residual_loser_lab


def test_phase4c_identifies_selector_that_captures_residual_losers(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_residual_loser_rows())

    result = build_phase4c_residual_loser_lab(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4c",
        candidate_count=5,
        loser_quantile=0.25,
    )
    summary = pd.read_csv(
        tmp_path / "phase4c" / "phase4c_residual_loser_selector_summary_validation.csv"
    )
    improvement = pd.read_csv(
        tmp_path / "phase4c" / "phase4c_residual_loser_improvement_vs_reversal_validation.csv"
    )
    weak20 = summary[summary["selector"] == "short_weak_momentum_20"].iloc[0]
    reversal = summary[summary["selector"] == "short_reversal_winner"].iloc[0]
    weak20_improvement = improvement[improvement["selector"] == "short_weak_momentum_20"].iloc[0]

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert weak20["mean_short_contribution_beta_residual_return"] > 0
    assert weak20["mean_oracle_overlap_rate"] > reversal["mean_oracle_overlap_rate"]
    assert weak20_improvement["beats_reversal_on_residual"] in (True, "True")


def test_phase4c_writes_opportunity_and_feature_roadmap(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_residual_loser_rows())

    build_phase4c_residual_loser_lab(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4c",
        candidate_count=5,
    )
    opportunity = pd.read_csv(
        tmp_path / "phase4c" / "phase4c_residual_loser_universe_opportunity_validation.csv"
    )
    roadmap = pd.read_csv(tmp_path / "phase4c" / "phase4c_residual_loser_feature_roadmap.csv")

    assert opportunity.loc[0, "mean_negative_residual_share"] > 0
    assert "fundamental_quality" in set(roadmap["feature_family"])
    assert "needs_new_data" in set(roadmap["availability"])


def _residual_loser_rows() -> list[dict[str, object]]:
    rows = []
    for session in ("2020-01-02", "2020-01-03"):
        for index in range(40):
            is_loser = index < 5
            rows.append(
                {
                    "session_date": session,
                    "variant": "top500_clean_core_beta_full",
                    "symbol": f"SYM{index:03d}",
                    "lagged_close": 20.0 + index,
                    "trailing_median_dollar_volume_20": 50_000_000.0 + index * 1_000_000.0,
                    "liquidity_rank": index + 1,
                    "beta": 1.0 + index * 0.001,
                    "reversal_5d": 100.0 + index if is_loser else float(index),
                    "momentum_20d": -1.0 if is_loser else 1.0 + index * 0.01,
                    "momentum_60d": 0.5 + index * 0.01,
                    "beta_residual_momentum_20d": -0.5 if is_loser else 0.5,
                    "vol_adjusted_momentum_20d": -0.5 if is_loser else 0.5,
                    "forward_return_5d": -0.03 if is_loser else 0.01,
                    "forward_beta_residual_return_5d": -0.04 if is_loser else 0.01,
                    "random_control": index / 40.0,
                }
            )
    return rows


def _write_signal_panel(tmp_path, *, rows: list[dict[str, object]]):
    path = tmp_path / "signal_panel.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    return path
