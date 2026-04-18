from __future__ import annotations

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4b import build_phase4b_short_book_diagnostics


def test_phase4b_labels_reversal_short_failure_before_beta_matching(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_diagnosis_rows())
    positions_path = _write_positions(tmp_path)
    skipped_path = _write_skipped(tmp_path)

    build_phase4b_short_book_diagnostics(
        signal_panel_path=signal_panel_path,
        phase4_positions_path=positions_path,
        phase4_skipped_path=skipped_path,
        output_root=tmp_path / "phase4b",
        signal_names=("reversal_5d", "momentum_20d"),
        candidate_count=10,
    )
    matrix = pd.read_csv(tmp_path / "phase4b" / "phase4b_short_failure_matrix_validation.csv")
    reversal = matrix[matrix["signal"] == "reversal_5d"].iloc[0]
    momentum = matrix[matrix["signal"] == "momentum_20d"].iloc[0]

    assert reversal["diagnosis_label"] == "short_selection_fails_before_beta_matching"
    assert reversal["equal_short_contribution_beta_residual_return"] < 0
    assert momentum["diagnosis_label"] == "short_signal_candidate_viable"
    assert momentum["equal_short_contribution_beta_residual_return"] > 0
    assert matrix["test_window_used"].eq(False).all()


def test_phase4b_writes_weighted_side_and_skip_summaries(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_diagnosis_rows())
    positions_path = _write_positions(tmp_path)
    skipped_path = _write_skipped(tmp_path)

    result = build_phase4b_short_book_diagnostics(
        signal_panel_path=signal_panel_path,
        phase4_positions_path=positions_path,
        phase4_skipped_path=skipped_path,
        output_root=tmp_path / "phase4b",
        signal_names=("reversal_5d",),
        candidate_count=10,
    )
    summary = pd.read_csv(tmp_path / "phase4b" / "phase4b_short_book_side_summary_validation.csv")
    skip_summary = pd.read_csv(tmp_path / "phase4b" / "phase4b_phase4_skip_summary_validation.csv")
    weighted_short = summary[
        (summary["source"] == "phase4_weighted_positions") & (summary["side"] == "short")
    ].iloc[0]

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert weighted_short["mean_side_contribution_return"] < 0
    assert skip_summary.loc[0, "skip_reason"] == "beta_range_no_overlap"
    assert skip_summary.loc[0, "skipped_sessions"] == 1


def _diagnosis_rows() -> list[dict[str, object]]:
    rows = []
    for session in ("2020-01-02", "2020-01-03"):
        for index in range(40):
            forward_return = 0.0
            if index < 10:
                forward_return = 0.02
            elif 10 <= index < 20:
                forward_return = -0.02
            elif index >= 30:
                forward_return = 0.01
            rows.append(
                {
                    "session_date": session,
                    "variant": "top500_clean_core_beta_full",
                    "symbol": f"SYM{index:03d}",
                    "beta": 1.0,
                    "forward_return_5d": forward_return,
                    "forward_beta_residual_return_5d": forward_return - 0.001,
                    "reversal_5d": float(40 - index),
                    "momentum_20d": _momentum_score(index),
                }
            )
    return rows


def _momentum_score(index: int) -> float:
    if 10 <= index < 20:
        return -100.0 + index
    if 20 <= index < 30:
        return 100.0 - index
    return 0.0


def _write_signal_panel(tmp_path, *, rows: list[dict[str, object]]):
    path = tmp_path / "signal_panel.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    return path


def _write_positions(tmp_path):
    path = tmp_path / "positions.csv.gz"
    rows = []
    for side, sign, forward_return in (("long", 1.0, 0.02), ("short", -1.0, 0.01)):
        for index in range(10):
            rows.append(
                {
                    "session_date": "2020-01-02",
                    "variant": "top500_clean_core_beta_full",
                    "signal": "reversal_5d",
                    "side": side,
                    "symbol": f"{side.upper()}{index:03d}",
                    "score": float(index),
                    "beta": 1.0,
                    "side_weight": 0.1,
                    "signed_weight": sign * 0.1,
                    "forward_return_5d": forward_return,
                    "forward_beta_residual_return_5d": forward_return - 0.001,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    return path


def _write_skipped(tmp_path):
    path = tmp_path / "skipped.csv"
    pd.DataFrame(
        [
            {
                "session_date": "2020-01-03",
                "variant": "top500_clean_core_beta_full",
                "signal": "reversal_5d",
                "eligible_names": 40,
                "skip_reason": "beta_range_no_overlap",
                "test_window_used": False,
            }
        ]
    ).to_csv(path, index=False)
    return path
