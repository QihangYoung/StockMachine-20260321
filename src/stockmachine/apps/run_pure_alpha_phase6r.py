"""Validation-only model-complexity ladder for strict residual pairwise ranking.

Phase6R keeps Phase6Q's anti-overfit discipline fixed and varies only model
capacity. The goal is to separate underfitting from weak-signal overfitting for
the strict style-residual target.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase6p import BASE_FEATURES, LABEL_COLUMN, _markdown_table
from stockmachine.apps.run_pure_alpha_phase6q import (
    FOLD_SPECS,
    MAX_TRAIN_PAIRS_PER_FOLD,
    PAIRS_PER_DATE,
    PAIR_STOCK_APPEARANCE_CAP,
    PURGE_SESSIONS,
    RANKING_REFERENCE_COUNT,
    LabelSpec,
    ModelSpec,
    _build_eval_ranking_scores,
    _cap_train_pairs,
    _feature_importance_rows,
    _fold_date_sets,
    _load_phase6p_panel,
    _make_model,
    _pairwise_metric_row,
    _sample_capped_sparse_pairs,
    _stable_seed,
    _summarize_daily_ranking,
)


OUTDIR = RESEARCH_ROOT / "phase6r_pairwise_complexity_ladder_20260511"
STRICT_LABEL = LabelSpec(
    name="strict_style_residual",
    column=LABEL_COLUMN,
    description="Beta, true size, liquidity, reversal/momentum style, and SIC2 residual.",
)

COMPLEXITY_MODEL_SPECS = (
    ModelSpec(
        "tiny_lgbm",
        {
            "objective": "binary",
            "n_estimators": 60,
            "learning_rate": 0.05,
            "num_leaves": 3,
            "max_depth": 2,
            "min_child_samples": 5000,
            "subsample": 0.70,
            "subsample_freq": 1,
            "colsample_bytree": 0.70,
            "reg_alpha": 1.0,
            "reg_lambda": 100.0,
        },
    ),
    ModelSpec(
        "shallow_lgbm",
        {
            "objective": "binary",
            "n_estimators": 120,
            "learning_rate": 0.04,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 2000,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.50,
            "reg_lambda": 25.0,
        },
    ),
    ModelSpec(
        "medium_lgbm",
        {
            "objective": "binary",
            "n_estimators": 220,
            "learning_rate": 0.035,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 800,
            "subsample": 0.80,
            "subsample_freq": 1,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.15,
            "reg_lambda": 8.0,
        },
    ),
    ModelSpec(
        "phase6p_like_lgbm",
        {
            "objective": "binary",
            "n_estimators": 400,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 250,
            "subsample": 0.80,
            "subsample_freq": 1,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
        },
    ),
)


def main(argv: Sequence[str] | None = None) -> None:
    rollup = build_phase6r_pairwise_complexity_ladder()
    print(json.dumps(rollup, indent=2, ensure_ascii=True))


def build_phase6r_pairwise_complexity_ladder() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = _load_phase6p_panel()
    all_dates = pd.Index(sorted(panel["session_date"].unique()))
    pairs, sampling = _sample_capped_sparse_pairs(panel, STRICT_LABEL)
    pair_dates = pd.to_datetime(pairs["session_date"])
    feature_cols = [f"d_{feature}" for feature in BASE_FEATURES]

    pairwise_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for fold_spec in FOLD_SPECS:
        train_dates, eval_dates = _fold_date_sets(all_dates, fold_spec)
        train_pairs = pairs.loc[pair_dates.isin(train_dates)].copy()
        eval_pairs = pairs.loc[pair_dates.isin(eval_dates)].copy()
        train_pairs = _cap_train_pairs(train_pairs, STRICT_LABEL, fold_spec)
        eval_panel = panel[panel["session_date"].isin(eval_dates)].copy()

        for model_spec in COMPLEXITY_MODEL_SPECS:
            model = _make_model(model_spec)
            model.fit(train_pairs[feature_cols], train_pairs["label"].astype(int))
            pairwise_rows.append(
                _pairwise_metric_row(
                    model=model,
                    frame=train_pairs,
                    feature_cols=feature_cols,
                    label_spec=STRICT_LABEL,
                    fold_spec=fold_spec,
                    model_spec=model_spec,
                    sample_split="train_used",
                )
            )
            pairwise_rows.append(
                _pairwise_metric_row(
                    model=model,
                    frame=eval_pairs,
                    feature_cols=feature_cols,
                    label_spec=STRICT_LABEL,
                    fold_spec=fold_spec,
                    model_spec=model_spec,
                    sample_split="eval_oos",
                )
            )
            ranking_panel = _build_eval_ranking_scores(
                panel=eval_panel,
                label_spec=STRICT_LABEL,
                model=model,
                seed=_stable_seed(STRICT_LABEL.name, fold_spec.name, model_spec.name, "phase6r"),
            )
            ranking_rows.append(
                _summarize_daily_ranking(
                    ranking_panel,
                    label_spec=STRICT_LABEL,
                    score_column="pairwise_rank_score",
                    score_label=model_spec.name,
                    fold_spec=fold_spec,
                    model_name=model_spec.name,
                )
            )
            importance_rows.extend(
                _feature_importance_rows(model, STRICT_LABEL, fold_spec, model_spec)
            )

    pairwise_metrics = pd.DataFrame(pairwise_rows)
    ranking_metrics = pd.DataFrame(ranking_rows)
    sampling_summary = pd.DataFrame(sampling)
    feature_importance = pd.DataFrame(importance_rows)
    degradation = _build_complexity_degradation(pairwise_metrics, ranking_metrics)

    pairwise_path = OUTDIR / "phase6r_pairwise_metrics.csv"
    ranking_path = OUTDIR / "phase6r_oos_ranking_metrics.csv"
    sampling_path = OUTDIR / "phase6r_pair_sampling_summary.csv"
    importance_path = OUTDIR / "phase6r_feature_importance.csv"
    degradation_path = OUTDIR / "phase6r_complexity_degradation_summary.csv"
    plot_path = OUTDIR / "phase6r_complexity_ladder_plot.png"
    memo_path = OUTDIR / "phase6r_complexity_ladder_memo.md"
    rollup_path = OUTDIR / "phase6r_rollup.json"

    pairwise_metrics.to_csv(pairwise_path, index=False)
    ranking_metrics.to_csv(ranking_path, index=False)
    sampling_summary.to_csv(sampling_path, index=False)
    feature_importance.to_csv(importance_path, index=False)
    degradation.to_csv(degradation_path, index=False)
    _plot_complexity_ladder(degradation, plot_path)
    memo_path.write_text(
        _memo(
            ranking_metrics=ranking_metrics,
            degradation=degradation,
            feature_importance=feature_importance,
            plot_path=plot_path,
        ),
        encoding="utf-8",
    )

    rollup = {
        "phase": "phase6r_pairwise_complexity_ladder",
        "created_at_utc": _utc_now(),
        "scope": "validation_only",
        "test_lockbox_used": False,
        "label": STRICT_LABEL.__dict__,
        "models": [model.name for model in COMPLEXITY_MODEL_SPECS],
        "folds": [fold.__dict__ for fold in FOLD_SPECS],
        "anti_overfit_controls": {
            "pairs_per_date": PAIRS_PER_DATE,
            "pair_stock_appearance_cap": PAIR_STOCK_APPEARANCE_CAP,
            "purge_sessions": PURGE_SESSIONS,
            "max_train_pairs_per_fold": MAX_TRAIN_PAIRS_PER_FOLD,
            "ranking_reference_count": RANKING_REFERENCE_COUNT,
        },
        "rows": {
            "panel": int(len(panel)),
            "pairs": int(len(pairs)),
            "pairwise_metrics": int(len(pairwise_metrics)),
            "ranking_metrics": int(len(ranking_metrics)),
        },
        "outputs": {
            "pairwise_metrics": str(pairwise_path),
            "ranking_metrics": str(ranking_path),
            "sampling_summary": str(sampling_path),
            "feature_importance": str(importance_path),
            "degradation": str(degradation_path),
            "plot": str(plot_path),
            "memo": str(memo_path),
        },
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _build_complexity_degradation(
    pairwise_metrics: pd.DataFrame,
    ranking_metrics: pd.DataFrame,
) -> pd.DataFrame:
    train = pairwise_metrics[pairwise_metrics["sample_split"].eq("train_used")].copy()
    eval_oos = pairwise_metrics[pairwise_metrics["sample_split"].eq("eval_oos")].copy()
    merged = eval_oos.merge(
        train[["fold", "model", "auc", "logloss", "accuracy_0p5"]].rename(
            columns={
                "auc": "train_auc",
                "logloss": "train_logloss",
                "accuracy_0p5": "train_accuracy_0p5",
            }
        ),
        on=["fold", "model"],
        how="left",
    )
    merged = merged.rename(
        columns={
            "auc": "eval_auc",
            "logloss": "eval_logloss",
            "accuracy_0p5": "eval_accuracy_0p5",
        }
    )
    merged["auc_gap_train_minus_eval"] = merged["train_auc"] - merged["eval_auc"]
    return merged.merge(
        ranking_metrics[
            [
                "fold",
                "model",
                "mean_rank_ic",
                "rank_ic_tstat",
                "mean_top_bottom_spread_bps",
                "spread_positive_rate",
                "spread_p10_bps",
            ]
        ],
        on=["fold", "model"],
        how="left",
    )


def _plot_complexity_ladder(degradation: pd.DataFrame, output_path: Path) -> None:
    order = [model.name for model in COMPLEXITY_MODEL_SPECS]
    plot = degradation.copy()
    plot["model"] = pd.Categorical(plot["model"], categories=order, ordered=True)
    plot = plot.sort_values(["fold", "model"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharex=True)
    for fold, group in plot.groupby("fold", sort=False):
        axes[0, 0].plot(group["model"].astype(str), group["eval_auc"], marker="o", label=fold)
        axes[0, 1].plot(
            group["model"].astype(str),
            group["auc_gap_train_minus_eval"],
            marker="o",
            label=fold,
        )
        axes[1, 0].plot(
            group["model"].astype(str),
            group["mean_top_bottom_spread_bps"],
            marker="o",
            label=fold,
        )
        axes[1, 1].plot(
            group["model"].astype(str),
            group["mean_rank_ic"],
            marker="o",
            label=fold,
        )

    axes[0, 0].set_title("OOS Pairwise AUC")
    axes[0, 1].set_title("Train - OOS AUC Gap")
    axes[1, 0].set_title("OOS Top-Bottom Strict Residual Spread")
    axes[1, 0].set_ylabel("bps")
    axes[1, 1].set_title("OOS Mean Rank IC")
    for ax in axes.ravel():
        ax.axhline(0.0, color="#111827", linestyle="--", linewidth=0.9)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(loc="best")
        ax.grid(True, linewidth=0.5, alpha=0.35)
    fig.suptitle("Phase6R Strict-Residual Pairwise Ranker Complexity Ladder")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _memo(
    *,
    ranking_metrics: pd.DataFrame,
    degradation: pd.DataFrame,
    feature_importance: pd.DataFrame,
    plot_path: Path,
) -> str:
    order = [model.name for model in COMPLEXITY_MODEL_SPECS]
    display = degradation.copy()
    display["model"] = pd.Categorical(display["model"], categories=order, ordered=True)
    display = display.sort_values(["fold", "model"])
    feature_rollup = (
        feature_importance.groupby(["model", "feature"], as_index=False)["importance_gain"]
        .mean()
        .sort_values(["model", "importance_gain"], ascending=[True, False])
    )
    lines = [
        "# Phase6R Strict-Residual Pairwise Complexity Ladder",
        "",
        "Validation-only. Test lockbox is not used.",
        "",
        "## Objective",
        "",
        "Hold Phase6Q's anti-overfit sampling and walk-forward controls fixed, then increase model capacity from tiny to Phase6P-like strong. If OOS strict residual spread recovers smoothly without a large train-OOS AUC gap, the Phase6Q drop was underfitting. If train fit rises but OOS spread does not, strict residual signal is weak and the stronger model is mostly fitting noise.",
        "",
        "## Complexity Ladder",
        "",
        _markdown_table(
            display[
                [
                    "fold",
                    "model",
                    "train_auc",
                    "eval_auc",
                    "auc_gap_train_minus_eval",
                    "mean_rank_ic",
                    "rank_ic_tstat",
                    "mean_top_bottom_spread_bps",
                    "spread_positive_rate",
                    "spread_p10_bps",
                ]
            ]
        ),
        "",
        "## Ranking Metrics",
        "",
        _markdown_table(ranking_metrics.sort_values(["fold", "model"])),
        "",
        "## Top Mean Feature Importance",
        "",
        _markdown_table(feature_rollup.groupby("model").head(8)),
        "",
        f"Plot: `{plot_path.as_posix()}`",
        "",
        "## Interpretation Rule",
        "",
        "A true underfitting diagnosis requires OOS spread and rank IC to improve with complexity while the train-OOS AUC gap remains controlled. A weak-signal diagnosis is favored if the Phase6P-like model mainly expands the train-OOS gap or produces unstable fold-specific OOS gains.",
    ]
    return "\n".join(lines) + "\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
