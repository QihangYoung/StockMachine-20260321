from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4m import (
    RIDGE_SELECTOR,
    build_phase4m_ridge_short_selector_artifacts,
)


def test_phase4m_ridge_selector_learns_synthetic_residual_losers(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_ridge_rows())

    result = build_phase4m_ridge_short_selector_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4m",
        long_variants=("top1000_clean_core_beta_full",),
        short_variants=("adv30m_clean_core_beta_full",),
        residual_targets=("cs_demeaned_beta_residual",),
        candidate_count=3,
        initial_train_sessions=4,
        label_embargo_sessions=1,
        block_sessions=3,
        max_train_rows=500,
        min_regression_rows=8,
        min_train_rows=20,
        max_names_per_side=5,
        max_single_name_side_weight=0.5,
    )
    summary = pd.read_csv(
        tmp_path / "phase4m" / "phase4m_ridge_selector_summary_validation.csv"
    )
    ridge = summary.loc[summary["selector"] == RIDGE_SELECTOR].iloc[0]

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert result["primary_evaluation"] == "standalone_short_selector_performance"
    assert ridge["mean_short_contribution_target"] > 0
    assert ridge["mean_selected_negative_target_share"] > 0
    assert summary["test_window_used"].eq(False).all()


def test_phase4m_writes_portfolio_comparison_outputs(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_ridge_rows())

    build_phase4m_ridge_short_selector_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4m",
        long_variants=("top1000_clean_core_beta_full",),
        short_variants=("adv30m_clean_core_beta_full",),
        residual_targets=("cs_demeaned_beta_residual",),
        candidate_count=3,
        initial_train_sessions=4,
        label_embargo_sessions=1,
        block_sessions=3,
        max_train_rows=500,
        min_regression_rows=8,
        min_train_rows=20,
        max_names_per_side=5,
        max_single_name_side_weight=0.5,
        beta_match_tolerance=0.01,
    )
    portfolio = pd.read_csv(
        tmp_path / "phase4m" / "phase4m_ridge_portfolio_summary_validation.csv"
    )
    common = pd.read_csv(
        tmp_path / "phase4m" / "phase4m_ridge_portfolio_common_session_validation.csv"
    )
    coefficients = pd.read_csv(
        tmp_path / "phase4m" / "phase4m_ridge_coefficients_validation.csv"
    )
    memo = (tmp_path / "phase4m" / "phase4m_ridge_short_selector_memo.md").read_text(
        encoding="utf-8"
    )

    assert {"ridge", "baseline"}.issubset(set(portfolio["selector_kind"]))
    assert portfolio["test_window_used"].eq(False).all()
    assert not common.empty
    assert common["test_window_used"].eq(False).all()
    assert not coefficients.empty
    assert coefficients["test_window_used"].eq(False).all()
    assert "Primary decision standard: standalone top-N short selector performance" in memo
    assert "Secondary diagnostic only" in memo


def _ridge_rows() -> list[dict[str, object]]:
    rows = []
    start = date(2020, 1, 1)
    for day in range(12):
        session = (start + timedelta(days=day)).isoformat()
        for variant, prefix in (
            ("top1000_clean_core_beta_full", "L"),
            ("adv30m_clean_core_beta_full", "S"),
        ):
            for index in range(12):
                short_loser = variant.startswith("adv30m") and index >= 9
                long_winner = variant.startswith("top1000") and index < 5
                residual = -0.05 if short_loser else 0.03 if long_winner else 0.01
                extension = 0.30 if short_loser else 0.01 * index
                reversal = 0.20 if long_winner else -0.05 if short_loser else 0.01
                beta = 0.9 + (index % 6) * 0.02
                benchmark_forward = 0.002
                rows.append(
                    {
                        "session_date": session,
                        "variant": variant,
                        "symbol": f"{prefix}{index:03d}",
                        "lagged_close": 20.0 + index,
                        "trailing_median_dollar_volume_20": 100_000_000.0 + index,
                        "liquidity_rank": index + 1,
                        "beta": beta,
                        "reversal_5d": reversal,
                        "momentum_20d": extension,
                        "momentum_60d": extension / 2.0,
                        "beta_residual_momentum_20d": extension,
                        "vol_adjusted_momentum_20d": 2.0 if short_loser else 0.1 * index,
                        "forward_return_5d": residual + beta * benchmark_forward,
                        "benchmark_forward_return_5d": benchmark_forward,
                        "forward_beta_residual_return_5d": residual,
                    }
                )
    return rows


def _write_signal_panel(tmp_path, *, rows: list[dict[str, object]]):
    path = tmp_path / "signal_panel.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    return path
