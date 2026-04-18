from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import build_phase4d_tree_selector_artifacts


def test_phase4d_tree_selector_uses_prior_folds_and_writes_summary(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_tree_rows())

    result = build_phase4d_tree_selector_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4d",
        variant_names=("top1000_clean_core_beta_full",),
        candidate_count=2,
        initial_train_sessions=4,
        label_embargo_sessions=1,
        block_sessions=3,
        max_train_rows=200,
        n_estimators=12,
        max_depth=3,
        min_samples_leaf=1,
        random_state=7,
    )
    summary = pd.read_csv(tmp_path / "phase4d" / "phase4d_tree_selector_summary_validation.csv")
    folds = pd.read_csv(tmp_path / "phase4d" / "phase4d_tree_selector_fold_manifest_validation.csv")

    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert not summary.empty
    assert summary.loc[0, "mean_short_contribution_beta_residual_return"] > 0
    assert folds.loc[0, "train_end_session"] < folds.loc[0, "predict_start_session"]
    assert folds["test_window_used"].eq(False).all()


def test_phase4d_tree_selector_writes_predictions_and_feature_importance(tmp_path) -> None:
    signal_panel_path = _write_signal_panel(tmp_path, rows=_tree_rows())

    build_phase4d_tree_selector_artifacts(
        signal_panel_path=signal_panel_path,
        output_root=tmp_path / "phase4d",
        variant_names=("top1000_clean_core_beta_full",),
        candidate_count=2,
        initial_train_sessions=4,
        label_embargo_sessions=1,
        block_sessions=3,
        max_train_rows=200,
        n_estimators=12,
        max_depth=3,
        min_samples_leaf=1,
        random_state=7,
    )
    predictions = pd.read_csv(
        tmp_path / "phase4d" / "phase4d_tree_selector_predictions_validation.csv.gz"
    )
    importance = pd.read_csv(
        tmp_path / "phase4d" / "phase4d_tree_selector_feature_importance_validation.csv"
    )

    assert {"tree_residual_loser_score", "predicted_forward_beta_residual_return_5d"}.issubset(
        predictions.columns
    )
    assert not importance.empty
    assert importance["test_window_used"].eq(False).all()


def _tree_rows() -> list[dict[str, object]]:
    rows = []
    start = date(2020, 1, 1)
    for day in range(12):
        session = (start + timedelta(days=day)).isoformat()
        for index in range(8):
            loser = index < 2
            rows.append(
                {
                    "session_date": session,
                    "variant": "top1000_clean_core_beta_full",
                    "symbol": f"SYM{index:03d}",
                    "lagged_close": 20.0 + index,
                    "trailing_median_dollar_volume_20": 100_000_000.0 + index * 1_000_000.0,
                    "liquidity_rank": index + 1,
                    "beta": 1.0 + index * 0.01,
                    "reversal_5d": -0.10 if loser else 0.01 * index,
                    "momentum_20d": 0.20 if loser else -0.01 * index,
                    "momentum_60d": 0.15 if loser else 0.01 * index,
                    "beta_residual_momentum_20d": 0.18 if loser else -0.01 * index,
                    "vol_adjusted_momentum_20d": 2.0 if loser else -0.1 * index,
                    "forward_return_5d": -0.04 if loser else 0.01,
                    "forward_beta_residual_return_5d": -0.05 if loser else 0.02,
                }
            )
    return rows


def _write_signal_panel(tmp_path, *, rows: list[dict[str, object]]):
    path = tmp_path / "signal_panel.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    return path
