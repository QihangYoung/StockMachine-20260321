"""Validation-only sparse pairwise residual-alpha ranking experiment.

Phase6P tests whether nonlinear price-state features can rank future
multi-factor residual winners and losers without directly relying on the
linear reversal/momentum style payoff used by the hand-built selectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT, TARGET_COLUMN
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_LONG_VARIANT,
    DEFAULT_SHORT_VARIANT,
    NUMERIC_STYLE_RISK_FEATURES,
    _daily_residualize,
    _zscore,
)
from stockmachine.apps.run_pure_alpha_phase6e import _load_feature_panel


OUTDIR = RESEARCH_ROOT / "phase6p_sparse_pairwise_ranker_20260511"
SIZE_PANEL_PATH = (
    RESEARCH_ROOT
    / "phase6j_true_size_mvp_20260508"
    / "phase6j_true_size_panel_validation.csv.gz"
)

LABEL_COLUMN = "true_size_style_sic2_residual"
RANDOM_SEED = 20260511
PAIRS_PER_DATE = 2500
RANKING_REFERENCE_COUNT = 96
MARGIN_IQR_MULTIPLIER = 0.25
ADV_FLOOR_USD = 1_000_000.0

TRAIN_END = "2017-12-31"
MODEL_VALID_START = "2018-01-01"
MODEL_VALID_END = "2018-12-31"
FINAL_VALID_START = "2019-01-01"
FINAL_VALID_END = "2019-12-31"

BASE_FEATURES = (
    "beta",
    "reversal_5d",
    "momentum_20d",
    "momentum_60d",
    "beta_residual_momentum_20d",
    "vol_adjusted_momentum_20d",
    "lagged_close_log",
    "trailing_median_dollar_volume_20_log",
    "liquidity_rank_z",
    "market_cap_log_z",
    "ret_5_over_20",
    "ret_20_over_60",
    "overextension_20_5",
    "residual_overextension_20_5",
    "abs_reversal_5d",
    "abs_momentum_20d",
)

BASELINE_SCORE_COLUMNS = (
    "reversal_5d",
    "anti_momentum_20d_score",
    "anti_beta_residual_momentum_20d_score",
    "anti_vol_adjusted_momentum_20d_score",
    "transparent_composite",
)


@dataclass(frozen=True)
class SplitSpec:
    split: str
    start: str | None
    end: str | None


SPLITS = (
    SplitSpec("train", None, TRAIN_END),
    SplitSpec("model_validation_2018", MODEL_VALID_START, MODEL_VALID_END),
    SplitSpec("final_validation_2019", FINAL_VALID_START, FINAL_VALID_END),
)


def main(argv: Sequence[str] | None = None) -> None:
    rollup = build_phase6p_sparse_pairwise_ranker()
    print(json.dumps(rollup, indent=2, ensure_ascii=True))


def build_phase6p_sparse_pairwise_ranker() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    panel = _load_training_panel()
    pairs = _sample_sparse_pairs(panel)
    model, model_metrics = _fit_pairwise_model(pairs)
    score_panel = _build_ranking_scores(panel, model)
    ranking_metrics = _build_ranking_metrics(score_panel)
    baseline_metrics = _build_baseline_ranking_metrics(panel)
    feature_importance = _feature_importance(model)

    pair_path = OUTDIR / "phase6p_sparse_pair_sample.csv.gz"
    panel_path = OUTDIR / "phase6p_label_feature_panel.csv.gz"
    score_path = OUTDIR / "phase6p_pairwise_score_panel.csv.gz"
    model_metrics_path = OUTDIR / "phase6p_pairwise_model_metrics.csv"
    ranking_metrics_path = OUTDIR / "phase6p_ranking_metrics.csv"
    baseline_metrics_path = OUTDIR / "phase6p_baseline_ranking_metrics.csv"
    feature_importance_path = OUTDIR / "phase6p_feature_importance.csv"
    model_path = OUTDIR / "phase6p_lgbm_pairwise_model.joblib"
    plot_path = OUTDIR / "phase6p_ranking_diagnostics_plot.png"
    memo_path = OUTDIR / "phase6p_sparse_pairwise_ranker_memo.md"
    rollup_path = OUTDIR / "phase6p_rollup.json"

    panel.to_csv(panel_path, index=False, compression="gzip")
    pairs.to_csv(pair_path, index=False, compression="gzip")
    score_panel.to_csv(score_path, index=False, compression="gzip")
    model_metrics.to_csv(model_metrics_path, index=False)
    ranking_metrics.to_csv(ranking_metrics_path, index=False)
    baseline_metrics.to_csv(baseline_metrics_path, index=False)
    feature_importance.to_csv(feature_importance_path, index=False)
    joblib.dump(model, model_path)
    _plot_ranking_diagnostics(ranking_metrics, baseline_metrics, plot_path)
    memo_path.write_text(
        _memo(
            model_metrics=model_metrics,
            ranking_metrics=ranking_metrics,
            baseline_metrics=baseline_metrics,
            feature_importance=feature_importance,
            plot_path=plot_path,
        ),
        encoding="utf-8",
    )

    rollup = {
        "phase": "phase6p_sparse_pairwise_ranker",
        "created_at_utc": _utc_now(),
        "scope": "validation_only",
        "test_lockbox_used": False,
        "label_column": LABEL_COLUMN,
        "target_source_column": TARGET_COLUMN,
        "feature_columns": list(BASE_FEATURES),
        "pairs_per_date": PAIRS_PER_DATE,
        "ranking_reference_count": RANKING_REFERENCE_COUNT,
        "margin_iqr_multiplier": MARGIN_IQR_MULTIPLIER,
        "adv_floor_usd": ADV_FLOOR_USD,
        "splits": [split.__dict__ for split in SPLITS],
        "rows": {
            "panel": int(len(panel)),
            "pairs": int(len(pairs)),
            "score_panel": int(len(score_panel)),
            "model_metrics": int(len(model_metrics)),
            "ranking_metrics": int(len(ranking_metrics)),
        },
        "outputs": {
            "label_feature_panel": str(panel_path),
            "pair_sample": str(pair_path),
            "score_panel": str(score_path),
            "model_metrics": str(model_metrics_path),
            "ranking_metrics": str(ranking_metrics_path),
            "baseline_metrics": str(baseline_metrics_path),
            "feature_importance": str(feature_importance_path),
            "model": str(model_path),
            "plot": str(plot_path),
            "memo": str(memo_path),
        },
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_training_panel() -> pd.DataFrame:
    panel = _load_feature_panel()
    panel = panel[panel["variant"].astype(str).eq(DEFAULT_LONG_VARIANT)].copy()
    panel["session_date"] = pd.to_datetime(panel["session_date"])
    panel["symbol"] = panel["symbol"].astype(str)
    if "test_window_used" in panel.columns and panel["test_window_used"].map(_is_true).any():
        raise ValueError("Refusing to run: panel includes test-window rows.")

    size = pd.read_csv(
        SIZE_PANEL_PATH,
        usecols=[
            "variant",
            "session_date",
            "symbol",
            "market_cap_log_z",
            "market_cap_mvp_available",
        ],
        low_memory=False,
    )
    size = size[size["variant"].astype(str).eq(DEFAULT_LONG_VARIANT)].copy()
    size["session_date"] = pd.to_datetime(size["session_date"])
    size["symbol"] = size["symbol"].astype(str)
    size["market_cap_log_z"] = pd.to_numeric(size["market_cap_log_z"], errors="coerce")
    panel = panel.merge(
        size[["session_date", "symbol", "market_cap_log_z", "market_cap_mvp_available"]],
        on=["session_date", "symbol"],
        how="left",
    )

    for column in [
        "beta",
        "lagged_close",
        "trailing_median_dollar_volume_20",
        "liquidity_rank",
        "reversal_5d",
        "momentum_20d",
        "momentum_60d",
        "beta_residual_momentum_20d",
        "vol_adjusted_momentum_20d",
        TARGET_COLUMN,
        "market_cap_log_z",
    ]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

    panel = panel[panel["trailing_median_dollar_volume_20"].fillna(0.0) >= ADV_FLOOR_USD].copy()
    panel["lagged_close_log"] = np.log(panel["lagged_close"].where(panel["lagged_close"] > 0))
    panel["trailing_median_dollar_volume_20_log"] = np.log(
        panel["trailing_median_dollar_volume_20"].where(
            panel["trailing_median_dollar_volume_20"] > 0
        )
    )
    panel["liquidity_rank_z"] = panel.groupby("session_date", sort=False)[
        "liquidity_rank"
    ].transform(_zscore)
    panel["ret_5_over_20"] = panel["reversal_5d"] / panel["momentum_20d"].abs().replace(0.0, np.nan)
    panel["ret_20_over_60"] = panel["momentum_20d"] / panel["momentum_60d"].abs().replace(0.0, np.nan)
    panel["overextension_20_5"] = panel["momentum_20d"] - panel["reversal_5d"]
    panel["residual_overextension_20_5"] = panel["beta_residual_momentum_20d"] - panel["reversal_5d"]
    panel["abs_reversal_5d"] = panel["reversal_5d"].abs()
    panel["abs_momentum_20d"] = panel["momentum_20d"].abs()
    panel["anti_momentum_20d_score"] = -panel["momentum_20d"]
    panel["anti_beta_residual_momentum_20d_score"] = -panel["beta_residual_momentum_20d"]
    panel["anti_vol_adjusted_momentum_20d_score"] = -panel["vol_adjusted_momentum_20d"]

    # A stricter realized residual-alpha label: beta + true size + liquidity +
    # reversal/momentum style + SIC2 residual. All exposures are known at t.
    label_features = tuple(
        dict.fromkeys(
            [
                "beta",
                "market_cap_log_z",
                "trailing_median_dollar_volume_20_log",
                "liquidity_rank",
                "reversal_5d",
                "momentum_20d",
                "momentum_60d",
                "beta_residual_momentum_20d",
                "vol_adjusted_momentum_20d",
            ]
        )
    )
    panel[LABEL_COLUMN] = _daily_residualize(
        panel,
        target_column=TARGET_COLUMN,
        numeric_features=label_features,
        categorical_features=("sic2_sector",),
        min_regression_rows=80,
        min_dummy_count=5,
    )
    for feature in BASE_FEATURES:
        panel[feature] = pd.to_numeric(panel[feature], errors="coerce")
        panel[f"{feature}_x"] = panel.groupby("session_date", sort=False)[feature].transform(_zscore)

    feature_cols = [f"{feature}_x" for feature in BASE_FEATURES]
    required = ["session_date", "symbol", LABEL_COLUMN, *feature_cols]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    panel["split"] = panel["session_date"].map(_split_for_date)
    panel = panel[panel["split"].notna()].copy()
    return panel.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _sample_sparse_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    feature_cols = [f"{feature}_x" for feature in BASE_FEATURES]
    rows: list[pd.DataFrame] = []
    for session_date, group in panel.groupby("session_date", sort=True):
        group = group.reset_index(drop=True)
        y = group[LABEL_COLUMN].to_numpy(dtype=float)
        if len(group) < 100 or np.nanstd(y) <= 1e-12:
            continue
        iqr = float(np.nanquantile(y, 0.75) - np.nanquantile(y, 0.25))
        margin_threshold = max(iqr * MARGIN_IQR_MULTIPLIER, 1e-8)
        sampled = _sample_pairs_for_group(group, y, margin_threshold, rng)
        if sampled.empty:
            continue
        values_i = group.loc[sampled["idx_i"].to_numpy(), feature_cols].to_numpy(dtype=float)
        values_j = group.loc[sampled["idx_j"].to_numpy(), feature_cols].to_numpy(dtype=float)
        diff = values_i - values_j
        pair_frame = pd.DataFrame(diff, columns=[f"d_{feature}" for feature in BASE_FEATURES])
        pair_frame.insert(0, "split", str(group["split"].iloc[0]))
        pair_frame.insert(0, "session_date", session_date)
        pair_frame.insert(1, "symbol_i", group.loc[sampled["idx_i"].to_numpy(), "symbol"].to_numpy())
        pair_frame.insert(2, "symbol_j", group.loc[sampled["idx_j"].to_numpy(), "symbol"].to_numpy())
        pair_frame["label"] = sampled["label"].to_numpy(dtype=int)
        pair_frame["margin"] = sampled["margin"].to_numpy(dtype=float)
        pair_frame["sample_type"] = sampled["sample_type"].astype(str).to_numpy()
        rows.append(pair_frame)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    return result.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def _sample_pairs_for_group(
    group: pd.DataFrame,
    y: np.ndarray,
    margin_threshold: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(group)
    target_tail = int(PAIRS_PER_DATE * 0.65)
    target_random = PAIRS_PER_DATE - target_tail
    rows: list[dict[str, Any]] = []

    q20 = float(np.nanquantile(y, 0.20))
    q80 = float(np.nanquantile(y, 0.80))
    winners = np.flatnonzero(y >= q80)
    losers = np.flatnonzero(y <= q20)
    if len(winners) and len(losers):
        win = rng.choice(winners, size=target_tail, replace=True)
        lose = rng.choice(losers, size=target_tail, replace=True)
        flip = rng.random(target_tail) < 0.5
        for w_idx, l_idx, do_flip in zip(win, lose, flip):
            margin = float(abs(y[w_idx] - y[l_idx]))
            if margin <= margin_threshold:
                continue
            if do_flip:
                rows.append(
                    {
                        "idx_i": int(l_idx),
                        "idx_j": int(w_idx),
                        "label": 0,
                        "margin": margin,
                        "sample_type": "tail",
                    }
                )
            else:
                rows.append(
                    {
                        "idx_i": int(w_idx),
                        "idx_j": int(l_idx),
                        "label": 1,
                        "margin": margin,
                        "sample_type": "tail",
                    }
                )

    attempts = 0
    max_attempts = target_random * 20
    while len(rows) < PAIRS_PER_DATE and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        margin = float(abs(y[i] - y[j]))
        if margin <= margin_threshold:
            continue
        label = int(y[i] > y[j])
        rows.append(
            {
                "idx_i": i,
                "idx_j": j,
                "label": label,
                "margin": margin,
                "sample_type": "random_margin",
            }
        )
    return pd.DataFrame(rows[:PAIRS_PER_DATE])


def _fit_pairwise_model(pairs: pd.DataFrame) -> tuple[LGBMClassifier, pd.DataFrame]:
    feature_cols = [f"d_{feature}" for feature in BASE_FEATURES]
    train = pairs[pairs["split"].eq("train")].copy()
    model_valid = pairs[pairs["split"].eq("model_validation_2018")].copy()
    final_valid = pairs[pairs["split"].eq("final_validation_2019")].copy()
    model = LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=250,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        train[feature_cols],
        train["label"].astype(int),
        eval_set=[(model_valid[feature_cols], model_valid["label"].astype(int))],
        eval_metric="binary_logloss",
    )
    metrics = []
    for split, frame in [
        ("train", train),
        ("model_validation_2018", model_valid),
        ("final_validation_2019", final_valid),
    ]:
        metrics.append(_pairwise_metric_row(model, frame, feature_cols, split))
    return model, pd.DataFrame(metrics)


def _pairwise_metric_row(
    model: LGBMClassifier,
    frame: pd.DataFrame,
    feature_cols: list[str],
    split: str,
) -> dict[str, Any]:
    proba = model.predict_proba(frame[feature_cols])[:, 1]
    labels = frame["label"].astype(int).to_numpy()
    return {
        "split": split,
        "pairs": int(len(frame)),
        "positive_rate": float(labels.mean()),
        "auc": float(roc_auc_score(labels, proba)) if len(set(labels)) == 2 else np.nan,
        "logloss": float(log_loss(labels, np.clip(proba, 1e-6, 1.0 - 1e-6))),
        "accuracy_0p5": float(accuracy_score(labels, proba >= 0.5)),
        "mean_margin_bps": float(frame["margin"].mean() * 10000.0),
    }


def _build_ranking_scores(panel: pd.DataFrame, model: LGBMClassifier) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 17)
    stock_feature_cols = [f"{feature}_x" for feature in BASE_FEATURES]
    model_feature_cols = [f"d_{feature}" for feature in BASE_FEATURES]
    rows: list[pd.DataFrame] = []
    for session_date, group in panel.groupby("session_date", sort=True):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < 50:
            continue
        ref_n = min(RANKING_REFERENCE_COUNT, n)
        ref_idx = rng.choice(np.arange(n), size=ref_n, replace=False)
        features = group[stock_feature_cols].to_numpy(dtype=float)
        ref_features = features[ref_idx]
        diff = (features[:, None, :] - ref_features[None, :, :]).reshape(n * ref_n, len(stock_feature_cols))
        diff_frame = pd.DataFrame(diff, columns=model_feature_cols)
        proba = model.predict_proba(diff_frame)[:, 1].reshape(n, ref_n)
        scores = (proba - 0.5).mean(axis=1)
        output_columns = [
            "session_date",
            "symbol",
            "split",
            LABEL_COLUMN,
            "reversal_5d",
            "anti_momentum_20d_score",
            "anti_beta_residual_momentum_20d_score",
            "anti_vol_adjusted_momentum_20d_score",
        ]
        if "transparent_composite" in group.columns:
            output_columns.append("transparent_composite")
        out = group[output_columns].copy()
        out["pairwise_rank_score"] = scores
        out["reference_count"] = ref_n
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_ranking_metrics(score_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, session_date), group in score_panel.groupby(["split", "session_date"], sort=True):
        rows.append(
            _ranking_metric_row(
                group,
                split=split,
                session_date=session_date,
                score_column="pairwise_rank_score",
                score_label="pairwise_rank_score",
            )
        )
    daily = pd.DataFrame(rows)
    summary = _summarize_ranking_metrics(daily)
    daily_path = OUTDIR / "phase6p_pairwise_ranking_daily_metrics.csv"
    daily.to_csv(daily_path, index=False)
    return summary


def _build_baseline_ranking_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    daily_rows = []
    for score_column in BASELINE_SCORE_COLUMNS:
        if score_column not in panel.columns:
            continue
        for (split, session_date), group in panel.groupby(["split", "session_date"], sort=True):
            daily_rows.append(
                _ranking_metric_row(
                    group,
                    split=split,
                    session_date=session_date,
                    score_column=score_column,
                    score_label=score_column,
                )
            )
    daily = pd.DataFrame(daily_rows)
    daily_path = OUTDIR / "phase6p_baseline_ranking_daily_metrics.csv"
    daily.to_csv(daily_path, index=False)
    return _summarize_ranking_metrics(daily)


def _ranking_metric_row(
    group: pd.DataFrame,
    *,
    split: str,
    session_date: Any,
    score_column: str,
    score_label: str,
) -> dict[str, Any]:
    frame = group[[score_column, LABEL_COLUMN]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 50:
        return {
            "split": split,
            "session_date": session_date,
            "score": score_label,
            "names": int(len(frame)),
            "rank_ic": np.nan,
            "top_bottom_spread": np.nan,
            "top_mean": np.nan,
            "bottom_mean": np.nan,
        }
    rank_ic = spearmanr(frame[score_column], frame[LABEL_COLUMN]).correlation
    low = frame[score_column].quantile(0.20)
    high = frame[score_column].quantile(0.80)
    top = frame[frame[score_column] >= high][LABEL_COLUMN]
    bottom = frame[frame[score_column] <= low][LABEL_COLUMN]
    return {
        "split": split,
        "session_date": session_date,
        "score": score_label,
        "names": int(len(frame)),
        "rank_ic": float(rank_ic) if np.isfinite(rank_ic) else np.nan,
        "top_bottom_spread": float(top.mean() - bottom.mean()),
        "top_mean": float(top.mean()),
        "bottom_mean": float(bottom.mean()),
    }


def _summarize_ranking_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    rows = []
    for (split, score), group in daily.groupby(["split", "score"], sort=True):
        rank_ic = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        spread = pd.to_numeric(group["top_bottom_spread"], errors="coerce").dropna()
        rows.append(
            {
                "split": split,
                "score": score,
                "sessions": int(len(group)),
                "mean_rank_ic": float(rank_ic.mean()) if not rank_ic.empty else np.nan,
                "rank_ic_tstat": float(rank_ic.mean() / rank_ic.std(ddof=1) * np.sqrt(len(rank_ic)))
                if len(rank_ic) > 2 and rank_ic.std(ddof=1) > 0
                else np.nan,
                "mean_top_bottom_spread_bps": float(spread.mean() * 10000.0)
                if not spread.empty
                else np.nan,
                "spread_positive_rate": float((spread > 0).mean()) if not spread.empty else np.nan,
                "spread_p10_bps": float(spread.quantile(0.10) * 10000.0)
                if not spread.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _feature_importance(model: LGBMClassifier) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [f"d_{feature}" for feature in BASE_FEATURES],
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)


def _plot_ranking_diagnostics(
    ranking_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    combined = pd.concat([ranking_metrics, baseline_metrics], ignore_index=True)
    plot = combined[
        combined["split"].isin(["model_validation_2018", "final_validation_2019"])
    ].copy()
    if plot.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    order = ["pairwise_rank_score", *BASELINE_SCORE_COLUMNS]
    plot["score"] = pd.Categorical(plot["score"], categories=order, ordered=True)
    plot = plot.sort_values(["split", "score"])
    for split, group in plot.groupby("split", sort=False):
        axes[0].plot(group["score"].astype(str), group["mean_rank_ic"], marker="o", label=split)
        axes[1].plot(
            group["score"].astype(str),
            group["mean_top_bottom_spread_bps"],
            marker="o",
            label=split,
        )
    axes[0].axhline(0.0, color="#111827", linestyle="--", linewidth=0.9)
    axes[0].set_title("Mean Rank IC")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].axhline(0.0, color="#111827", linestyle="--", linewidth=0.9)
    axes[1].set_title("Top 20% - Bottom 20% Residual Spread")
    axes[1].set_ylabel("bps / h10 label")
    axes[1].tick_params(axis="x", rotation=35)
    for ax in axes:
        ax.legend(loc="best")
        ax.grid(True, linewidth=0.5, alpha=0.35)
    fig.suptitle("Phase6P Sparse Pairwise Ranker Diagnostics (Validation Only)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _memo(
    *,
    model_metrics: pd.DataFrame,
    ranking_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    plot_path: Path,
) -> str:
    lines = [
        "# Phase6P Sparse Pairwise Ranker Memo",
        "",
        "Validation-only. Test lockbox is not used.",
        "",
        "## Objective",
        "",
        "Train a nonlinear sparse pairwise classifier to rank future h10 realized residual alpha, where the label residualizes beta, true size, liquidity, reversal/momentum style variables, and SIC2.",
        "",
        "## Model Quality",
        "",
        _markdown_table(model_metrics),
        "",
        "## Pairwise Ranking Summary",
        "",
        _markdown_table(ranking_metrics),
        "",
        "## Baseline Ranking Summary",
        "",
        _markdown_table(baseline_metrics),
        "",
        "## Top Feature Importance",
        "",
        _markdown_table(feature_importance.head(12)),
        "",
        f"Plot: `{plot_path.as_posix()}`",
        "",
        "## Interpretation Rule",
        "",
        "This experiment is promising only if the pairwise score beats the hand-built price-style baselines on 2018 and 2019 rank IC / top-bottom residual spread. If it does not, price-only nonlinear ranking is likely still dominated by public style payoff noise.",
    ]
    return "\n".join(lines) + "\n"


def _split_for_date(value: Any) -> str | None:
    date = pd.Timestamp(value)
    if date <= pd.Timestamp(TRAIN_END):
        return "train"
    if pd.Timestamp(MODEL_VALID_START) <= date <= pd.Timestamp(MODEL_VALID_END):
        return "model_validation_2018"
    if pd.Timestamp(FINAL_VALID_START) <= date <= pd.Timestamp(FINAL_VALID_END):
        return "final_validation_2019"
    return None


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6g}")
        else:
            display[column] = display[column].astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str)]
    return "\n".join([header, separator, *rows])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
