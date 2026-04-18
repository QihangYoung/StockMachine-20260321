from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4n import (
    build_phase4n_pca_ridge_standalone_artifacts,
)


def test_phase4n_pca_ridge_writes_standalone_short_outputs(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_pca_rows())

    result = build_phase4n_pca_ridge_standalone_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4n",
        short_variant="adv30m_clean_core_beta_full",
        residual_target="cs_demeaned_beta_residual",
        pca_components=(3,),
        candidate_count=3,
        initial_train_sessions=4,
        label_embargo_sessions=1,
        block_sessions=3,
        max_train_rows=500,
        min_regression_rows=8,
        min_train_rows=20,
    )
    summary = pd.read_csv(
        tmp_path / "phase4n" / "phase4n_pca_ridge_selector_summary_validation.csv"
    )
    folds = pd.read_csv(
        tmp_path / "phase4n" / "phase4n_pca_ridge_fold_manifest_validation.csv"
    )
    loadings = pd.read_csv(tmp_path / "phase4n" / "phase4n_pca_ridge_loadings_validation.csv")
    memo = (tmp_path / "phase4n" / "phase4n_pca_ridge_standalone_short_memo.md").read_text(
        encoding="utf-8"
    )

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert result["portfolio_diagnostics_computed"] is False
    assert result["primary_evaluation"] == "standalone_short_selector_performance"
    assert {"pca_ridge_pc3", "short_core_plus_overextension"}.issubset(
        set(summary["selector"])
    )
    assert summary["test_window_used"].eq(False).all()
    assert folds["pca_components"].eq(3).all()
    assert folds["test_window_used"].eq(False).all()
    assert not loadings.empty
    assert loadings["test_window_used"].eq(False).all()
    assert "No beta-matched portfolio diagnostics are computed" in memo


def _pca_rows() -> list[dict[str, object]]:
    rows = []
    start = date(2020, 1, 1)
    for day in range(12):
        session = (start + timedelta(days=day)).isoformat()
        for index in range(12):
            short_loser = index >= 9
            residual = -0.05 if short_loser else 0.025
            extension = 0.35 if short_loser else 0.01 * index
            beta = 0.9 + (index % 6) * 0.02
            benchmark_forward = 0.002
            rows.append(
                {
                    "session_date": session,
                    "variant": "adv30m_clean_core_beta_full",
                    "symbol": f"S{index:03d}",
                    "lagged_close": 20.0 + index,
                    "trailing_median_dollar_volume_20": 100_000_000.0 + index,
                    "liquidity_rank": index + 1,
                    "beta": beta,
                    "reversal_5d": -0.05 if short_loser else 0.01,
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
