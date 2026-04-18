from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


PROJECT_ID = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT_ID / "research"
DEFAULT_PHASE3_SIGNAL_PANEL = (
    RESEARCH_ROOT
    / "phase3_baseline_signals_20260418"
    / "phase3_baseline_signal_panel_validation.csv.gz"
)
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4d_tree_residual_loser_selector_20260418"
DEFAULT_VARIANTS = ("top1000_clean_core_beta_full",)
TARGET_COLUMN = "forward_beta_residual_return_5d"
BASE_COLUMNS = (
    "session_date",
    "variant",
    "symbol",
    "lagged_close",
    "trailing_median_dollar_volume_20",
    "liquidity_rank",
    "beta",
    "reversal_5d",
    "momentum_20d",
    "momentum_60d",
    "beta_residual_momentum_20d",
    "vol_adjusted_momentum_20d",
    "forward_return_5d",
    "forward_beta_residual_return_5d",
)
MODEL_FEATURES = (
    "lagged_close_log",
    "trailing_median_dollar_volume_20_log",
    "liquidity_rank",
    "beta",
    "return_5d",
    "momentum_20d",
    "momentum_60d",
    "beta_residual_momentum_20d",
    "vol_adjusted_momentum_20d",
    "return_5d_z",
    "momentum_20d_z",
    "momentum_60d_z",
    "beta_residual_momentum_20d_z",
    "vol_adjusted_momentum_20d_z",
    "beta_z",
    "liquidity_rank_z",
    "exhausted_winner_20_5",
    "residual_overextension_20_5",
    "fragile_winner_proxy",
)


def build_phase4d_tree_selector_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_PHASE3_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant_names: Sequence[str] = DEFAULT_VARIANTS,
    candidate_count: int = 30,
    initial_train_sessions: int = 252,
    label_embargo_sessions: int = 5,
    block_sessions: int = 126,
    max_train_rows: int = 150_000,
    n_estimators: int = 96,
    max_depth: int | None = 5,
    min_samples_leaf: int = 200,
    random_state: int = 260321,
) -> dict[str, Any]:
    """Train a rolling tree model to rank likely future beta-residual losers."""

    _validate_settings(
        variant_names=variant_names,
        candidate_count=candidate_count,
        initial_train_sessions=initial_train_sessions,
        label_embargo_sessions=label_embargo_sessions,
        block_sessions=block_sessions,
        max_train_rows=max_train_rows,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
    )
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_feature_panel(signal_panel_path, variant_names=variant_names)
    predictions, daily, folds, importance = _rolling_tree_predictions(
        panel,
        candidate_count=candidate_count,
        initial_train_sessions=initial_train_sessions,
        label_embargo_sessions=label_embargo_sessions,
        block_sessions=block_sessions,
        max_train_rows=max_train_rows,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    summary = _selector_summary(daily)

    predictions_path = output_dir / "phase4d_tree_selector_predictions_validation.csv.gz"
    daily_path = output_dir / "phase4d_tree_selector_daily_validation.csv"
    summary_path = output_dir / "phase4d_tree_selector_summary_validation.csv"
    folds_path = output_dir / "phase4d_tree_selector_fold_manifest_validation.csv"
    importance_path = output_dir / "phase4d_tree_selector_feature_importance_validation.csv"
    memo_path = output_dir / "phase4d_tree_selector_memo.md"
    rollup_path = output_dir / "phase4d_tree_selector_rollup.json"

    predictions.to_csv(predictions_path, index=False, compression="gzip")
    daily.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    folds.to_csv(folds_path, index=False)
    importance.to_csv(importance_path, index=False)
    memo_path.write_text(
        _tree_memo(
            summary,
            folds,
            importance,
            candidate_count=candidate_count,
            initial_train_sessions=initial_train_sessions,
            label_embargo_sessions=label_embargo_sessions,
            block_sessions=block_sessions,
            max_train_rows=max_train_rows,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "variant_names": list(variant_names),
        "candidate_count": int(candidate_count),
        "initial_train_sessions": int(initial_train_sessions),
        "label_embargo_sessions": int(label_embargo_sessions),
        "block_sessions": int(block_sessions),
        "max_train_rows": int(max_train_rows),
        "model_type": "sklearn.ensemble.ExtraTreesRegressor",
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": int(random_state),
        "features": list(MODEL_FEATURES),
        "validation_start": _first_or_none(daily["session_date"]) if not daily.empty else None,
        "validation_end": _last_or_none(daily["session_date"]) if not daily.empty else None,
        "feature_panel_rows_loaded": int(len(panel)),
        "prediction_rows": int(len(predictions)),
        "daily_rows": int(len(daily)),
        "summary_rows": int(len(summary)),
        "fold_rows": int(len(folds)),
        "feature_importance_rows": int(len(importance)),
        "predictions_artifact": predictions_path.as_posix(),
        "daily_artifact": daily_path.as_posix(),
        "summary_artifact": summary_path.as_posix(),
        "fold_manifest_artifact": folds_path.as_posix(),
        "feature_importance_artifact": importance_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "This is a rolling validation-only selector diagnostic, not a portfolio constructor.",
            "Tree predictions use only prior labeled sessions with a label embargo.",
            "No universe expansion, transaction costs, borrow costs, candidate freeze, or test-window performance are computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _validate_settings(
    *,
    variant_names: Sequence[str],
    candidate_count: int,
    initial_train_sessions: int,
    label_embargo_sessions: int,
    block_sessions: int,
    max_train_rows: int,
    n_estimators: int,
    min_samples_leaf: int,
) -> None:
    if not variant_names:
        raise ValueError("At least one variant is required.")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive.")
    if initial_train_sessions <= 0:
        raise ValueError("initial_train_sessions must be positive.")
    if label_embargo_sessions < 0:
        raise ValueError("label_embargo_sessions must be non-negative.")
    if block_sessions <= 0:
        raise ValueError("block_sessions must be positive.")
    if max_train_rows <= 0:
        raise ValueError("max_train_rows must be positive.")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive.")
    if min_samples_leaf <= 0:
        raise ValueError("min_samples_leaf must be positive.")


def _load_feature_panel(
    signal_panel_path: str | Path,
    *,
    variant_names: Sequence[str],
) -> pd.DataFrame:
    panel = pd.read_csv(signal_panel_path, usecols=list(BASE_COLUMNS), low_memory=False)
    panel["session_date"] = pd.to_datetime(panel["session_date"]).dt.date.astype(str)
    panel["variant"] = panel["variant"].astype(str)
    panel["symbol"] = panel["symbol"].astype(str)
    panel = panel[panel["variant"].isin(set(variant_names))].copy()
    for column in BASE_COLUMNS[3:]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.dropna(subset=["beta", TARGET_COLUMN, "forward_return_5d"])
    panel = _add_model_features(panel)
    return panel.sort_values(["variant", "session_date", "symbol"]).reset_index(drop=True)


def _add_model_features(panel: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in panel.groupby(["variant", "session_date"], sort=False):
        frame = group.copy()
        safe_price = frame["lagged_close"].where(frame["lagged_close"] > 0)
        safe_adv = frame["trailing_median_dollar_volume_20"].where(
            frame["trailing_median_dollar_volume_20"] > 0
        )
        frame["lagged_close_log"] = np.log(safe_price)
        frame["trailing_median_dollar_volume_20_log"] = np.log(safe_adv)
        frame["return_5d"] = -frame["reversal_5d"]
        frame["return_5d_z"] = _zscore(frame["return_5d"])
        frame["momentum_20d_z"] = _zscore(frame["momentum_20d"])
        frame["momentum_60d_z"] = _zscore(frame["momentum_60d"])
        frame["beta_residual_momentum_20d_z"] = _zscore(frame["beta_residual_momentum_20d"])
        frame["vol_adjusted_momentum_20d_z"] = _zscore(frame["vol_adjusted_momentum_20d"])
        frame["beta_z"] = _zscore(frame["beta"])
        frame["liquidity_rank_z"] = _zscore(frame["liquidity_rank"])
        frame["exhausted_winner_20_5"] = frame["momentum_20d_z"] + frame["return_5d_z"]
        frame["residual_overextension_20_5"] = (
            frame["beta_residual_momentum_20d_z"] + frame["return_5d_z"]
        )
        frame["fragile_winner_proxy"] = (
            frame["return_5d_z"]
            + frame["beta_z"]
            + _zscore(-frame["trailing_median_dollar_volume_20_log"])
        )
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True) if pieces else panel


def _rolling_tree_predictions(
    panel: pd.DataFrame,
    *,
    candidate_count: int,
    initial_train_sessions: int,
    label_embargo_sessions: int,
    block_sessions: int,
    max_train_rows: int,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for variant, variant_panel in panel.groupby("variant", sort=True):
        sessions = tuple(sorted(variant_panel["session_date"].unique()))
        start = initial_train_sessions + label_embargo_sessions
        fold_id = 0
        while start < len(sessions):
            predict_sessions = sessions[start : min(start + block_sessions, len(sessions))]
            train_end_index = start - label_embargo_sessions
            train_sessions = sessions[:train_end_index]
            if len(train_sessions) < initial_train_sessions or not predict_sessions:
                break
            train = variant_panel[variant_panel["session_date"].isin(train_sessions)].copy()
            predict = variant_panel[variant_panel["session_date"].isin(predict_sessions)].copy()
            train = train.dropna(subset=[*MODEL_FEATURES, TARGET_COLUMN])
            predict = predict.dropna(subset=[*MODEL_FEATURES, TARGET_COLUMN])
            if train.empty or predict.empty:
                start += block_sessions
                fold_id += 1
                continue
            train = _deterministic_sample(train, max_rows=max_train_rows, random_state=random_state + fold_id)
            fill_values = train[list(MODEL_FEATURES)].median(numeric_only=True).fillna(0.0)
            x_train = train[list(MODEL_FEATURES)].fillna(fill_values)
            y_train = train[TARGET_COLUMN]
            x_predict = predict[list(MODEL_FEATURES)].fillna(fill_values)
            model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state + fold_id,
                n_jobs=-1,
            )
            model.fit(x_train, y_train)
            predict = predict.copy()
            predict["predicted_forward_beta_residual_return_5d"] = model.predict(x_predict)
            predict["tree_residual_loser_score"] = -predict[
                "predicted_forward_beta_residual_return_5d"
            ]
            predict["model_fold_id"] = fold_id
            prediction_rows.append(
                predict[
                    [
                        "session_date",
                        "variant",
                        "symbol",
                        "model_fold_id",
                        "tree_residual_loser_score",
                        "predicted_forward_beta_residual_return_5d",
                        TARGET_COLUMN,
                        "forward_return_5d",
                    ]
                ]
            )
            daily_rows.extend(_daily_selector_rows(predict, candidate_count=candidate_count))
            fold_rows.append(
                {
                    "variant": variant,
                    "model_fold_id": fold_id,
                    "train_start_session": train_sessions[0],
                    "train_end_session": train_sessions[-1],
                    "predict_start_session": predict_sessions[0],
                    "predict_end_session": predict_sessions[-1],
                    "train_sessions": int(len(train_sessions)),
                    "predict_sessions": int(len(predict_sessions)),
                    "train_rows": int(len(train)),
                    "predict_rows": int(len(predict)),
                    "label_embargo_sessions": int(label_embargo_sessions),
                    "test_window_used": False,
                }
            )
            for feature, value in zip(MODEL_FEATURES, model.feature_importances_):
                importance_rows.append(
                    {
                        "variant": variant,
                        "model_fold_id": fold_id,
                        "feature": feature,
                        "feature_importance": float(value),
                        "test_window_used": False,
                    }
                )
            start += block_sessions
            fold_id += 1
    predictions = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame(columns=_prediction_columns())
    )
    daily = pd.DataFrame(daily_rows, columns=_daily_columns())
    folds = pd.DataFrame(fold_rows, columns=_fold_columns())
    importance = _feature_importance_summary(pd.DataFrame(importance_rows))
    return predictions, daily, folds, importance


def _daily_selector_rows(predict: pd.DataFrame, *, candidate_count: int) -> list[dict[str, Any]]:
    rows = []
    for (variant, session_date), group in predict.groupby(["variant", "session_date"], sort=True):
        if len(group) < candidate_count:
            continue
        selected = group.sort_values(
            ["tree_residual_loser_score", "symbol"],
            ascending=[False, True],
        ).head(candidate_count)
        oracle = group.nsmallest(candidate_count, TARGET_COLUMN)
        oracle_symbols = set(oracle["symbol"])
        residual = selected[TARGET_COLUMN]
        raw_return = selected["forward_return_5d"]
        all_scores = group[["tree_residual_loser_score", TARGET_COLUMN]].dropna()
        rank_ic = all_scores["tree_residual_loser_score"].corr(
            all_scores[TARGET_COLUMN],
            method="spearman",
        )
        rows.append(
            {
                "session_date": session_date,
                "variant": variant,
                "selector": "extra_trees_residual_loser_score",
                "model_fold_id": int(selected["model_fold_id"].iloc[0]),
                "eligible_names": int(len(group)),
                "selected_names": int(len(selected)),
                "selected_mean_predicted_residual_return": float(
                    selected["predicted_forward_beta_residual_return_5d"].mean()
                ),
                "selected_mean_forward_return_5d": float(raw_return.mean()),
                "selected_mean_forward_beta_residual_return_5d": float(residual.mean()),
                "short_contribution_return": float(-raw_return.mean()),
                "short_contribution_beta_residual_return": float(-residual.mean()),
                "selected_negative_residual_share": float((residual < 0).mean()),
                "oracle_short_contribution_beta_residual_return": float(
                    -oracle[TARGET_COLUMN].mean()
                ),
                "oracle_overlap_rate": float(
                    len(set(selected["symbol"]).intersection(oracle_symbols)) / candidate_count
                ),
                "rank_ic_loser_score_vs_forward_residual": float(rank_ic),
                "test_window_used": False,
            }
        )
    return rows


def _selector_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=_summary_columns())
    rows = []
    for (variant, selector), group in daily.groupby(["variant", "selector"], sort=True):
        contribution = group["short_contribution_beta_residual_return"]
        oracle_gap = group["oracle_short_contribution_beta_residual_return"] - contribution
        rows.append(
            {
                "variant": variant,
                "selector": selector,
                "sessions": int(len(group)),
                "mean_short_contribution_return": float(group["short_contribution_return"].mean()),
                "mean_short_contribution_beta_residual_return": float(contribution.mean()),
                "short_residual_hit_rate": float((contribution > 0).mean()),
                "mean_selected_negative_residual_share": float(
                    group["selected_negative_residual_share"].mean()
                ),
                "mean_selected_predicted_residual_return": float(
                    group["selected_mean_predicted_residual_return"].mean()
                ),
                "mean_oracle_short_contribution_beta_residual_return": float(
                    group["oracle_short_contribution_beta_residual_return"].mean()
                ),
                "mean_oracle_gap_beta_residual_return": float(oracle_gap.mean()),
                "mean_oracle_overlap_rate": float(group["oracle_overlap_rate"].mean()),
                "mean_rank_ic_loser_score_vs_forward_residual": float(
                    group["rank_ic_loser_score_vs_forward_residual"].mean()
                ),
                "rank_ic_negative_share": float(
                    (group["rank_ic_loser_score_vs_forward_residual"] < 0).mean()
                ),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows, columns=_summary_columns())


def _feature_importance_summary(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame(columns=_importance_columns())
    summary = (
        importance.groupby(["variant", "feature"], sort=True)
        .agg(
            folds=("model_fold_id", "nunique"),
            mean_feature_importance=("feature_importance", "mean"),
            max_feature_importance=("feature_importance", "max"),
        )
        .reset_index()
        .sort_values(["variant", "mean_feature_importance"], ascending=[True, False])
    )
    summary["test_window_used"] = False
    return summary[_importance_columns()]


def _deterministic_sample(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=random_state).sort_values(
        ["session_date", "symbol"]
    )


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.nan, index=series.index)
    return (values - values.mean()) / std


def _tree_memo(
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    importance: pd.DataFrame,
    *,
    candidate_count: int,
    initial_train_sessions: int,
    label_embargo_sessions: int,
    block_sessions: int,
    max_train_rows: int,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
) -> str:
    lines = [
        "# Pure Alpha Phase 4D Tree Residual-Loser Selector Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "This is a validation-only rolling tree-model selector diagnostic. It trains "
        "only on prior labeled validation sessions, applies a label embargo, and "
        "does not expand the universe or inspect the test lockbox.",
        "",
        "## Model",
        "",
        "- model: `sklearn.ensemble.ExtraTreesRegressor`",
        f"- selected candidates per session: `{candidate_count}`",
        f"- initial train sessions: `{initial_train_sessions}`",
        f"- label embargo sessions: `{label_embargo_sessions}`",
        f"- prediction block sessions: `{block_sessions}`",
        f"- max train rows per fold: `{max_train_rows}`",
        f"- n estimators: `{n_estimators}`",
        f"- max depth: `{max_depth}`",
        f"- min samples leaf: `{min_samples_leaf}`",
        "",
        "## Summary",
        "",
        "| Variant | Sessions | Mean Short Residual Contribution | Hit Rate | Oracle Overlap | RankIC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['variant']} | {row['sessions']} | "
            f"{row['mean_short_contribution_beta_residual_return']} | "
            f"{row['short_residual_hit_rate']} | {row['mean_oracle_overlap_rate']} | "
            f"{row['mean_rank_ic_loser_score_vs_forward_residual']} |"
        )
    lines.extend(
        [
            "",
            "## Fold Manifest",
            "",
            "| Variant | Fold | Train End | Predict Start | Predict End |",
            "|---|---:|---|---|---|",
        ]
    )
    for _, row in folds.head(24).iterrows():
        lines.append(
            f"| {row['variant']} | {row['model_fold_id']} | {row['train_end_session']} | "
            f"{row['predict_start_session']} | {row['predict_end_session']} |"
        )
    lines.extend(
        [
            "",
            "## Top Feature Importances",
            "",
            "| Variant | Feature | Mean Importance |",
            "|---|---|---:|",
        ]
    )
    for _, row in importance.head(24).iterrows():
        lines.append(
            f"| {row['variant']} | {row['feature']} | {row['mean_feature_importance']} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This is a selector diagnostic, not a beta-matched portfolio.",
            "- Tree output must be compared with Phase 4C price-only selectors before promotion.",
            "- Any promoted tree selector needs a new Phase 4 construction pass and cost review.",
            "- Test lockbox remains closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _prediction_columns() -> list[str]:
    return [
        "session_date",
        "variant",
        "symbol",
        "model_fold_id",
        "tree_residual_loser_score",
        "predicted_forward_beta_residual_return_5d",
        TARGET_COLUMN,
        "forward_return_5d",
    ]


def _daily_columns() -> list[str]:
    return [
        "session_date",
        "variant",
        "selector",
        "model_fold_id",
        "eligible_names",
        "selected_names",
        "selected_mean_predicted_residual_return",
        "selected_mean_forward_return_5d",
        "selected_mean_forward_beta_residual_return_5d",
        "short_contribution_return",
        "short_contribution_beta_residual_return",
        "selected_negative_residual_share",
        "oracle_short_contribution_beta_residual_return",
        "oracle_overlap_rate",
        "rank_ic_loser_score_vs_forward_residual",
        "test_window_used",
    ]


def _summary_columns() -> list[str]:
    return [
        "variant",
        "selector",
        "sessions",
        "mean_short_contribution_return",
        "mean_short_contribution_beta_residual_return",
        "short_residual_hit_rate",
        "mean_selected_negative_residual_share",
        "mean_selected_predicted_residual_return",
        "mean_oracle_short_contribution_beta_residual_return",
        "mean_oracle_gap_beta_residual_return",
        "mean_oracle_overlap_rate",
        "mean_rank_ic_loser_score_vs_forward_residual",
        "rank_ic_negative_share",
        "test_window_used",
    ]


def _fold_columns() -> list[str]:
    return [
        "variant",
        "model_fold_id",
        "train_start_session",
        "train_end_session",
        "predict_start_session",
        "predict_end_session",
        "train_sessions",
        "predict_sessions",
        "train_rows",
        "predict_rows",
        "label_embargo_sessions",
        "test_window_used",
    ]


def _importance_columns() -> list[str]:
    return [
        "variant",
        "feature",
        "folds",
        "mean_feature_importance",
        "max_feature_importance",
        "test_window_used",
    ]


def _first_or_none(series: pd.Series) -> str | None:
    if len(series) == 0:
        return None
    return str(series.min())


def _last_or_none(series: pd.Series) -> str | None:
    if len(series) == 0:
        return None
    return str(series.max())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pure-alpha Phase 4D tree selector.")
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_PHASE3_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--initial-train-sessions", type=int, default=252)
    parser.add_argument("--label-embargo-sessions", type=int, default=5)
    parser.add_argument("--block-sessions", type=int, default=126)
    parser.add_argument("--max-train-rows", type=int, default=150_000)
    parser.add_argument("--n-estimators", type=int, default=96)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=260321)
    args = parser.parse_args(argv)

    result = build_phase4d_tree_selector_artifacts(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        variant_names=tuple(args.variants or DEFAULT_VARIANTS),
        candidate_count=args.candidate_count,
        initial_train_sessions=args.initial_train_sessions,
        label_embargo_sessions=args.label_embargo_sessions,
        block_sessions=args.block_sessions,
        max_train_rows=args.max_train_rows,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
