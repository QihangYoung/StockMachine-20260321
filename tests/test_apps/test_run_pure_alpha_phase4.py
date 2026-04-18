from __future__ import annotations

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4 import build_phase4_portfolio_artifacts


def test_phase4_constructor_beta_matches_sides_and_writes_positions(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_beta_match_rows("2020-01-02"))

    result = build_phase4_portfolio_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4",
        signal_names=("reversal_5d",),
        max_names_per_side=30,
        max_single_name_side_weight=0.05,
        beta_match_tolerance=0.01,
    )
    positions = pd.read_csv(tmp_path / "phase4" / "phase4_beta_matched_positions_validation.csv.gz")
    diagnostics = pd.read_csv(tmp_path / "phase4" / "phase4_portfolio_daily_diagnostics_validation.csv")

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert len(diagnostics) == 1
    assert abs(diagnostics.loc[0, "net_beta"]) <= 0.01
    assert diagnostics.loc[0, "long_count"] >= 20
    assert diagnostics.loc[0, "short_count"] >= 20
    assert abs(positions.loc[positions["side"] == "long", "signed_weight"].sum() - 1.0) < 1e-12
    assert abs(positions.loc[positions["side"] == "short", "signed_weight"].sum() + 1.0) < 1e-12
    assert positions["test_window_used"].eq(False).all()


def test_phase4_constructor_records_skip_when_candidates_are_insufficient(tmp_path) -> None:
    rows = _beta_match_rows("2020-01-02")[:30]
    signal_panel_path = _write_signal_panel(tmp_path, rows=rows)

    build_phase4_portfolio_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4",
        signal_names=("reversal_5d",),
        max_names_per_side=30,
        max_single_name_side_weight=0.05,
    )
    skipped = pd.read_csv(tmp_path / "phase4" / "phase4_skipped_sessions_validation.csv")
    diagnostics = pd.read_csv(tmp_path / "phase4" / "phase4_portfolio_daily_diagnostics_validation.csv")

    assert diagnostics.empty
    assert skipped.loc[0, "skip_reason"] == "insufficient_non_overlapping_candidates"
    assert skipped["test_window_used"].eq(False).all()


def test_phase4_constructor_records_skip_when_beta_ranges_do_not_overlap(tmp_path) -> None:
    rows = []
    for index in range(60):
        rows.append(
            _row(
                session_date="2020-01-02",
                symbol=f"SYM{index:03d}",
                score=float(100 - index),
                beta=0.4 if index < 30 else 2.2,
            )
        )
    signal_panel_path = _write_signal_panel(tmp_path, rows=rows)

    build_phase4_portfolio_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4",
        signal_names=("reversal_5d",),
        max_names_per_side=30,
        max_single_name_side_weight=0.05,
    )
    skipped = pd.read_csv(tmp_path / "phase4" / "phase4_skipped_sessions_validation.csv")

    assert skipped.loc[0, "skip_reason"] == "beta_range_no_overlap"


def _beta_match_rows(session_date: str) -> list[dict[str, object]]:
    rows = []
    for index in range(60):
        rows.append(
            _row(
                session_date=session_date,
                symbol=f"SYM{index:03d}",
                score=float(100 - index),
                beta=0.8 + (index % 30) * 0.02,
                forward_return=0.01 if index < 30 else -0.01,
            )
        )
    return rows


def _row(
    *,
    session_date: str,
    symbol: str,
    score: float,
    beta: float,
    forward_return: float = 0.0,
) -> dict[str, object]:
    benchmark_return = 0.002
    return {
        "session_date": session_date,
        "variant": "top500_clean_core_beta_full",
        "symbol": symbol,
        "beta": beta,
        "forward_return_5d": forward_return,
        "benchmark_forward_return_5d": benchmark_return,
        "forward_beta_residual_return_5d": forward_return - beta * benchmark_return,
        "reversal_5d": score,
    }


def _write_signal_panel(tmp_path, *, rows: list[dict[str, object]]):
    path = tmp_path / "signal_panel.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    return path
