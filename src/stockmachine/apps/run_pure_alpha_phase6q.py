"""Validation-only anti-overfit audit for sparse pairwise alpha ranking.

Phase6Q keeps the Phase6P data/label construction fixed, then asks a narrower
question: after stronger train/test discipline and much weaker models, does the
pairwise price-state ranker still retain out-of-sample residual-alpha signal?
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT, TARGET_COLUMN
from stockmachine.apps.run_pure_alpha_phase4s import _daily_residualize
from stockmachine.apps.run_pure_alpha_phase6p import (
    BASE_FEATURES,
    BASELINE_SCORE_COLUMNS,
    LABEL_COLUMN,
    _markdown_table,
)


PHASE6P_OUTDIR = RESEARCH_ROOT / "phase6p_sparse_pairwise_ranker_20260511"
PHASE6P_PANEL_PATH = PHASE6P_OUTDIR / "phase6p_label_feature_panel.csv.gz"
OUTDIR = RESEARCH_ROOT / "phase6q_sparse_pairwise_overfit_audit_20260511"

RANDOM_SEED = 20260511
PAIRS_PER_DATE = 700
PAIR_STOCK_APPEARANCE_CAP = 6
RANKING_REFERENCE_COUNT = 64
MARGIN_IQR_MULTIPLIER = 0.25
PURGE_SESSIONS = 10
MAX_TRAIN_PAIRS_PER_FOLD = 350_000


@dataclass(frozen=True)
class LabelSpec:
    name: str
    column: str
    description: str


@dataclass(frozen=True)
class FoldSpec:
    name: str
    eval_start: str
    eval_end: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: dict[str, Any]


LABEL_SPECS = (
    LabelSpec(
        name="beta_residual",
        column=TARGET_COLUMN,
        description="Forward return after the existing beta-residual adjustment.",
    ),
    LabelSpec(
        name="beta_size_liquidity_sic2_residual",
        column="phase6q_beta_size_liquidity_sic2_residual",
        description="Beta residual further neutralized to true size, liquidity, and SIC2.",
    ),
    LabelSpec(
        name="strict_style_residual",
        column=LABEL_COLUMN,
        description="Phase6P strict label: beta, true size, liquidity, reversal/momentum style, and SIC2 residual.",
    ),
)

FOLD_SPECS = (
    FoldSpec("wf_2017", "2017-01-01", "2017-12-31"),
    FoldSpec("wf_2018", "2018-01-01", "2018-12-31"),
    FoldSpec("wf_2019", "2019-01-01", "2019-12-31"),
)

MODEL_SPECS = (
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
)


def main(argv: Sequence[str] | None = None) -> None:
    rollup = build_phase6q_sparse_pairwise_overfit_audit()
    print(json.dumps(rollup, indent=2, ensure_ascii=True))


def build_phase6q_sparse_pairwise_overfit_audit() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    panel = _load_phase6p_panel()
    all_dates = pd.Index(sorted(panel["session_date"].unique()))
    pairwise_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for label_spec in LABEL_SPECS:
        pairs, sampling = _sample_capped_sparse_pairs(panel, label_spec)
        sampling_rows.extend(sampling)
        feature_cols = [f"d_{feature}" for feature in BASE_FEATURES]
        pair_dates = pd.to_datetime(pairs["session_date"])

        for fold_spec in FOLD_SPECS:
            train_dates, eval_dates = _fold_date_sets(all_dates, fold_spec)
            train_mask = pair_dates.isin(train_dates)
            eval_mask = pair_dates.isin(eval_dates)
            train_pairs = pairs.loc[train_mask].copy()
            eval_pairs = pairs.loc[eval_mask].copy()
            train_pairs = _cap_train_pairs(train_pairs, label_spec, fold_spec)
            eval_panel = panel[panel["session_date"].isin(eval_dates)].copy()

            for baseline_score in BASELINE_SCORE_COLUMNS:
                if baseline_score in eval_panel.columns:
                    baseline_rows.append(
                        _summarize_daily_ranking(
                            eval_panel,
                            label_spec=label_spec,
                            score_column=baseline_score,
                            score_label=baseline_score,
                            fold_spec=fold_spec,
                            model_name="baseline",
                        )
                    )

            for model_spec in MODEL_SPECS:
                if train_pairs.empty or eval_pairs.empty:
                    continue
                model = _make_model(model_spec)
                model.fit(train_pairs[feature_cols], train_pairs["label"].astype(int))
                pairwise_rows.append(
                    _pairwise_metric_row(
                        model=model,
                        frame=train_pairs,
                        feature_cols=feature_cols,
                        label_spec=label_spec,
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
                        label_spec=label_spec,
                        fold_spec=fold_spec,
                        model_spec=model_spec,
                        sample_split="eval_oos",
                    )
                )
                ranking_panel = _build_eval_ranking_scores(
                    panel=eval_panel,
                    label_spec=label_spec,
                    model=model,
                    seed=_stable_seed(label_spec.name, fold_spec.name, model_spec.name, offset=31),
                )
                ranking_rows.append(
                    _summarize_daily_ranking(
                        ranking_panel,
                        label_spec=label_spec,
                        score_column="pairwise_rank_score",
                        score_label=model_spec.name,
                        fold_spec=fold_spec,
                        model_name=model_spec.name,
                    )
                )
                importance_rows.extend(_feature_importance_rows(model, label_spec, fold_spec, model_spec))

    pairwise_metrics = pd.DataFrame(pairwise_rows)
    ranking_metrics = pd.DataFrame(ranking_rows)
    baseline_metrics = pd.DataFrame(baseline_rows)
    sampling_summary = pd.DataFrame(sampling_rows)
    feature_importance = pd.DataFrame(importance_rows)
    degradation = _build_degradation_summary(pairwise_metrics, ranking_metrics)

    pairwise_path = OUTDIR / "phase6q_pairwise_metrics.csv"
    ranking_path = OUTDIR / "phase6q_oos_ranking_metrics.csv"
    baseline_path = OUTDIR / "phase6q_oos_baseline_metrics.csv"
    sampling_path = OUTDIR / "phase6q_pair_sampling_summary.csv"
    importance_path = OUTDIR / "phase6q_feature_importance.csv"
    degradation_path = OUTDIR / "phase6q_overfit_degradation_summary.csv"
    plot_path = OUTDIR / "phase6q_oos_spread_audit.png"
    memo_path = OUTDIR / "phase6q_sparse_pairwise_overfit_audit_memo.md"
    rollup_path = OUTDIR / "phase6q_rollup.json"

    pairwise_metrics.to_csv(pairwise_path, index=False)
    ranking_metrics.to_csv(ranking_path, index=False)
    baseline_metrics.to_csv(baseline_path, index=False)
    sampling_summary.to_csv(sampling_path, index=False)
    feature_importance.to_csv(importance_path, index=False)
    degradation.to_csv(degradation_path, index=False)
    _plot_oos_spread_audit(ranking_metrics, baseline_metrics, plot_path)
    memo_path.write_text(
        _memo(
            pairwise_metrics=pairwise_metrics,
            ranking_metrics=ranking_metrics,
            baseline_metrics=baseline_metrics,
            degradation=degradation,
            sampling_summary=sampling_summary,
            feature_importance=feature_importance,
            plot_path=plot_path,
        ),
        encoding="utf-8",
    )

    rollup = {
        "phase": "phase6q_sparse_pairwise_overfit_audit",
        "created_at_utc": _utc_now(),
        "scope": "validation_only",
        "test_lockbox_used": False,
        "source_panel": str(PHASE6P_PANEL_PATH),
        "labels": [label.__dict__ for label in LABEL_SPECS],
        "folds": [fold.__dict__ for fold in FOLD_SPECS],
        "models": [model.name for model in MODEL_SPECS],
        "pairs_per_date": PAIRS_PER_DATE,
        "pair_stock_appearance_cap": PAIR_STOCK_APPEARANCE_CAP,
        "ranking_reference_count": RANKING_REFERENCE_COUNT,
        "purge_sessions": PURGE_SESSIONS,
        "max_train_pairs_per_fold": MAX_TRAIN_PAIRS_PER_FOLD,
        "rows": {
            "panel": int(len(panel)),
            "pairwise_metrics": int(len(pairwise_metrics)),
            "ranking_metrics": int(len(ranking_metrics)),
            "baseline_metrics": int(len(baseline_metrics)),
        },
        "outputs": {
            "pairwise_metrics": str(pairwise_path),
            "ranking_metrics": str(ranking_path),
            "baseline_metrics": str(baseline_path),
            "sampling_summary": str(sampling_path),
            "feature_importance": str(importance_path),
            "degradation": str(degradation_path),
            "plot": str(plot_path),
            "memo": str(memo_path),
        },
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_phase6p_panel() -> pd.DataFrame:
    if not PHASE6P_PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Phase6P panel is required before Phase6Q audit: {PHASE6P_PANEL_PATH}"
        )
    feature_cols = [f"{feature}_x" for feature in BASE_FEATURES]
    needed = {
        "variant",
        "session_date",
        "symbol",
        "sic2_sector",
        TARGET_COLUMN,
        LABEL_COLUMN,
        "beta",
        "market_cap_log_z",
        "trailing_median_dollar_volume_20_log",
        "liquidity_rank",
        *feature_cols,
        *BASELINE_SCORE_COLUMNS,
    }
    available = pd.read_csv(PHASE6P_PANEL_PATH, nrows=0).columns
    usecols = [column for column in available if column in needed]
    panel = pd.read_csv(PHASE6P_PANEL_PATH, usecols=usecols, low_memory=False)
    panel["session_date"] = pd.to_datetime(panel["session_date"])
    panel["symbol"] = panel["symbol"].astype(str)
    for column in panel.columns:
        if column not in {"variant", "session_date", "symbol", "sic2_sector"}:
            panel[column] = pd.to_numeric(panel[column], errors="coerce")

    panel["phase6q_beta_size_liquidity_sic2_residual"] = _daily_residualize(
        panel,
        target_column=TARGET_COLUMN,
        numeric_features=(
            "beta",
            "market_cap_log_z",
            "trailing_median_dollar_volume_20_log",
            "liquidity_rank",
        ),
        categorical_features=("sic2_sector",),
        min_regression_rows=80,
        min_dummy_count=5,
    )
    required = ["variant", "session_date", "symbol", *feature_cols, *(label.column for label in LABEL_SPECS)]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    return panel.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _sample_capped_sparse_pairs(
    panel: pd.DataFrame,
    label_spec: LabelSpec,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rng = np.random.default_rng(_stable_seed(label_spec.name, "pairs"))
    feature_cols = [f"{feature}_x" for feature in BASE_FEATURES]
    model_feature_cols = [f"d_{feature}" for feature in BASE_FEATURES]
    pair_frames: list[pd.DataFrame] = []
    sampling_rows: list[dict[str, Any]] = []

    for session_date, group in panel.groupby("session_date", sort=True):
        group = group.reset_index(drop=True)
        y = group[label_spec.column].to_numpy(dtype=float)
        if len(group) < 100 or np.nanstd(y) <= 1e-12:
            continue
        iqr = float(np.nanquantile(y, 0.75) - np.nanquantile(y, 0.25))
        margin_threshold = max(iqr * MARGIN_IQR_MULTIPLIER, 1e-8)
        sampled = _sample_pairs_for_group_capped(y, margin_threshold, rng)
        if sampled.empty:
            continue
        values_i = group.loc[sampled["idx_i"].to_numpy(), feature_cols].to_numpy(dtype=np.float32)
        values_j = group.loc[sampled["idx_j"].to_numpy(), feature_cols].to_numpy(dtype=np.float32)
        pair_frame = pd.DataFrame(values_i - values_j, columns=model_feature_cols)
        pair_frame.insert(0, "session_date", session_date)
        pair_frame["label"] = sampled["label"].to_numpy(dtype=np.int8)
        pair_frame["margin"] = sampled["margin"].to_numpy(dtype=np.float32)
        pair_frame["sample_type"] = sampled["sample_type"].astype(str).to_numpy()
        pair_frames.append(pair_frame)
        sampling_rows.append(
            {
                "label": label_spec.name,
                "session_date": session_date,
                "names": int(len(group)),
                "pairs": int(len(pair_frame)),
                "tail_pairs": int((pair_frame["sample_type"] == "tail").sum()),
                "random_margin_pairs": int((pair_frame["sample_type"] == "random_margin").sum()),
                "mean_margin_bps": float(pair_frame["margin"].mean() * 10000.0),
            }
        )

    if not pair_frames:
        return pd.DataFrame(), sampling_rows
    pairs = pd.concat(pair_frames, ignore_index=True)
    pairs = pairs.sample(frac=1.0, random_state=_stable_seed(label_spec.name, "shuffle"))
    return pairs.reset_index(drop=True), sampling_rows


def _sample_pairs_for_group_capped(
    y: np.ndarray,
    margin_threshold: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(y)
    counts = np.zeros(n, dtype=np.int16)
    rows: list[dict[str, Any]] = []
    target_tail = int(PAIRS_PER_DATE * 0.50)

    q20 = float(np.nanquantile(y, 0.20))
    q80 = float(np.nanquantile(y, 0.80))
    winners = np.flatnonzero(y >= q80)
    losers = np.flatnonzero(y <= q20)
    attempts = 0
    max_tail_attempts = max(target_tail * 80, 1)
    while len(rows) < target_tail and attempts < max_tail_attempts and len(winners) and len(losers):
        attempts += 1
        i = int(rng.choice(winners))
        j = int(rng.choice(losers))
        _try_add_pair(rows, counts, y, i, j, margin_threshold, "tail", rng)

    attempts = 0
    max_random_attempts = max(PAIRS_PER_DATE * 100, 1)
    while len(rows) < PAIRS_PER_DATE and attempts < max_random_attempts:
        attempts += 1
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        _try_add_pair(rows, counts, y, i, j, margin_threshold, "random_margin", rng)
    return pd.DataFrame(rows)


def _try_add_pair(
    rows: list[dict[str, Any]],
    counts: np.ndarray,
    y: np.ndarray,
    i: int,
    j: int,
    margin_threshold: float,
    sample_type: str,
    rng: np.random.Generator,
) -> bool:
    if i == j or counts[i] >= PAIR_STOCK_APPEARANCE_CAP or counts[j] >= PAIR_STOCK_APPEARANCE_CAP:
        return False
    margin = float(abs(y[i] - y[j]))
    if margin <= margin_threshold:
        return False
    if rng.random() < 0.5:
        idx_i, idx_j = i, j
    else:
        idx_i, idx_j = j, i
    rows.append(
        {
            "idx_i": idx_i,
            "idx_j": idx_j,
            "label": int(y[idx_i] > y[idx_j]),
            "margin": margin,
            "sample_type": sample_type,
        }
    )
    counts[i] += 1
    counts[j] += 1
    return True


def _fold_date_sets(
    all_dates: pd.Index,
    fold_spec: FoldSpec,
) -> tuple[pd.Index, pd.Index]:
    eval_start = pd.Timestamp(fold_spec.eval_start)
    eval_end = pd.Timestamp(fold_spec.eval_end)
    eval_dates = all_dates[(all_dates >= eval_start) & (all_dates <= eval_end)]
    first_eval_idx = int(all_dates.searchsorted(eval_start))
    train_cutoff_idx = max(0, first_eval_idx - PURGE_SESSIONS - 1)
    train_cutoff = all_dates[train_cutoff_idx]
    train_dates = all_dates[all_dates <= train_cutoff]
    return train_dates, eval_dates


def _cap_train_pairs(
    train_pairs: pd.DataFrame,
    label_spec: LabelSpec,
    fold_spec: FoldSpec,
) -> pd.DataFrame:
    if len(train_pairs) <= MAX_TRAIN_PAIRS_PER_FOLD:
        return train_pairs
    return train_pairs.sample(
        n=MAX_TRAIN_PAIRS_PER_FOLD,
        random_state=_stable_seed(label_spec.name, fold_spec.name, "train_cap"),
    ).reset_index(drop=True)


def _make_model(model_spec: ModelSpec) -> LGBMClassifier:
    return LGBMClassifier(
        **model_spec.params,
        random_state=_stable_seed(model_spec.name, "model"),
        n_jobs=-1,
        verbose=-1,
    )


def _pairwise_metric_row(
    *,
    model: LGBMClassifier,
    frame: pd.DataFrame,
    feature_cols: list[str],
    label_spec: LabelSpec,
    fold_spec: FoldSpec,
    model_spec: ModelSpec,
    sample_split: str,
) -> dict[str, Any]:
    proba = model.predict_proba(frame[feature_cols])[:, 1]
    labels = frame["label"].astype(int).to_numpy()
    return {
        "label": label_spec.name,
        "fold": fold_spec.name,
        "model": model_spec.name,
        "sample_split": sample_split,
        "pairs": int(len(frame)),
        "positive_rate": float(labels.mean()) if len(labels) else np.nan,
        "auc": float(roc_auc_score(labels, proba)) if len(set(labels)) == 2 else np.nan,
        "logloss": float(log_loss(labels, np.clip(proba, 1e-6, 1.0 - 1e-6)))
        if len(set(labels)) == 2
        else np.nan,
        "accuracy_0p5": float(accuracy_score(labels, proba >= 0.5)) if len(labels) else np.nan,
        "mean_margin_bps": float(frame["margin"].mean() * 10000.0) if len(frame) else np.nan,
    }


def _build_eval_ranking_scores(
    *,
    panel: pd.DataFrame,
    label_spec: LabelSpec,
    model: LGBMClassifier,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
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
        features = group[stock_feature_cols].to_numpy(dtype=np.float32)
        ref_features = features[ref_idx]
        diff = (features[:, None, :] - ref_features[None, :, :]).reshape(n * ref_n, len(stock_feature_cols))
        diff_frame = pd.DataFrame(diff, columns=model_feature_cols)
        proba = model.predict_proba(diff_frame)[:, 1].reshape(n, ref_n)
        out = group[["session_date", "symbol", label_spec.column]].copy()
        out["pairwise_rank_score"] = (proba - 0.5).mean(axis=1)
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _summarize_daily_ranking(
    frame: pd.DataFrame,
    *,
    label_spec: LabelSpec,
    score_column: str,
    score_label: str,
    fold_spec: FoldSpec,
    model_name: str,
) -> dict[str, Any]:
    daily_rows = []
    for session_date, group in frame.groupby("session_date", sort=True):
        metric = _daily_ranking_metric(group, label_spec.column, score_column)
        metric["session_date"] = session_date
        daily_rows.append(metric)
    daily = pd.DataFrame(daily_rows)
    rank_ic = pd.to_numeric(daily["rank_ic"], errors="coerce").dropna()
    spread = pd.to_numeric(daily["top_bottom_spread"], errors="coerce").dropna()
    return {
        "label": label_spec.name,
        "fold": fold_spec.name,
        "model": model_name,
        "score": score_label,
        "sessions": int(len(daily)),
        "mean_rank_ic": float(rank_ic.mean()) if not rank_ic.empty else np.nan,
        "rank_ic_tstat": float(rank_ic.mean() / rank_ic.std(ddof=1) * np.sqrt(len(rank_ic)))
        if len(rank_ic) > 2 and rank_ic.std(ddof=1) > 0
        else np.nan,
        "mean_top_bottom_spread_bps": float(spread.mean() * 10000.0)
        if not spread.empty
        else np.nan,
        "spread_positive_rate": float((spread > 0).mean()) if not spread.empty else np.nan,
        "spread_p10_bps": float(spread.quantile(0.10) * 10000.0) if not spread.empty else np.nan,
    }


def _daily_ranking_metric(
    group: pd.DataFrame,
    label_column: str,
    score_column: str,
) -> dict[str, Any]:
    frame = group[[score_column, label_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 50:
        return {
            "rank_ic": np.nan,
            "top_bottom_spread": np.nan,
            "top_mean": np.nan,
            "bottom_mean": np.nan,
        }
    rank_ic = spearmanr(frame[score_column], frame[label_column]).correlation
    low = frame[score_column].quantile(0.20)
    high = frame[score_column].quantile(0.80)
    top = frame[frame[score_column] >= high][label_column]
    bottom = frame[frame[score_column] <= low][label_column]
    return {
        "rank_ic": float(rank_ic) if np.isfinite(rank_ic) else np.nan,
        "top_bottom_spread": float(top.mean() - bottom.mean()),
        "top_mean": float(top.mean()),
        "bottom_mean": float(bottom.mean()),
    }


def _feature_importance_rows(
    model: LGBMClassifier,
    label_spec: LabelSpec,
    fold_spec: FoldSpec,
    model_spec: ModelSpec,
) -> list[dict[str, Any]]:
    booster = model.booster_
    gains = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")
    return [
        {
            "label": label_spec.name,
            "fold": fold_spec.name,
            "model": model_spec.name,
            "feature": f"d_{feature}",
            "importance_gain": float(gain),
            "importance_split": int(split),
        }
        for feature, gain, split in zip(BASE_FEATURES, gains, splits)
    ]


def _build_degradation_summary(
    pairwise_metrics: pd.DataFrame,
    ranking_metrics: pd.DataFrame,
) -> pd.DataFrame:
    eval_pairwise = pairwise_metrics[pairwise_metrics["sample_split"].eq("eval_oos")].copy()
    train_pairwise = pairwise_metrics[pairwise_metrics["sample_split"].eq("train_used")].copy()
    merged = eval_pairwise.merge(
        train_pairwise[["label", "fold", "model", "auc"]].rename(columns={"auc": "train_auc"}),
        on=["label", "fold", "model"],
        how="left",
    )
    merged = merged.rename(columns={"auc": "eval_auc"})
    merged["auc_gap_train_minus_eval"] = merged["train_auc"] - merged["eval_auc"]
    ranking = ranking_metrics[
        ["label", "fold", "model", "mean_top_bottom_spread_bps", "mean_rank_ic", "spread_positive_rate"]
    ].copy()
    return merged.merge(ranking, on=["label", "fold", "model"], how="left")


def _plot_oos_spread_audit(
    ranking_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    if ranking_metrics.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    strict = ranking_metrics[ranking_metrics["label"].eq("strict_style_residual")].copy()
    for model, group in strict.groupby("model", sort=True):
        axes[0].plot(
            group["fold"],
            group["mean_top_bottom_spread_bps"],
            marker="o",
            linewidth=2,
            label=model,
        )
    strict_baseline = baseline_metrics[
        baseline_metrics["label"].eq("strict_style_residual")
        & baseline_metrics["score"].isin(
            ["reversal_5d", "anti_beta_residual_momentum_20d_score"]
        )
    ].copy()
    for score, group in strict_baseline.groupby("score", sort=True):
        axes[0].plot(
            group["fold"],
            group["mean_top_bottom_spread_bps"],
            marker="x",
            linestyle="--",
            linewidth=1.5,
            label=score,
        )
    axes[0].axhline(0.0, color="#111827", linestyle="--", linewidth=0.9)
    axes[0].set_title("Strict Residual OOS Spread")
    axes[0].set_ylabel("Top 20% - Bottom 20%, bps")
    axes[0].legend(loc="best")

    shallow = ranking_metrics[ranking_metrics["model"].eq("shallow_lgbm")].copy()
    for label, group in shallow.groupby("label", sort=True):
        axes[1].plot(
            group["fold"],
            group["mean_top_bottom_spread_bps"],
            marker="o",
            linewidth=2,
            label=label,
        )
    axes[1].axhline(0.0, color="#111827", linestyle="--", linewidth=0.9)
    axes[1].set_title("Label Ladder, Shallow LGBM")
    axes[1].set_ylabel("Top 20% - Bottom 20%, bps")
    axes[1].legend(loc="best")

    for ax in axes:
        ax.grid(True, linewidth=0.5, alpha=0.35)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Phase6Q Sparse Pairwise Ranker Anti-Overfit Audit (Validation Only)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _memo(
    *,
    pairwise_metrics: pd.DataFrame,
    ranking_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    degradation: pd.DataFrame,
    sampling_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    plot_path: Path,
) -> str:
    strict_oos = ranking_metrics[ranking_metrics["label"].eq("strict_style_residual")].copy()
    final_2019 = ranking_metrics[ranking_metrics["fold"].eq("wf_2019")].copy()
    baseline_2019 = baseline_metrics[baseline_metrics["fold"].eq("wf_2019")].copy()
    sampling_rollup = (
        sampling_summary.groupby("label", as_index=False)
        .agg(
            sessions=("session_date", "nunique"),
            mean_pairs_per_session=("pairs", "mean"),
            mean_margin_bps=("mean_margin_bps", "mean"),
        )
        .sort_values("label")
    )
    feature_rollup = (
        feature_importance.groupby(["label", "model", "feature"], as_index=False)["importance_gain"]
        .mean()
        .sort_values(["label", "model", "importance_gain"], ascending=[True, True, False])
    )
    lines = [
        "# Phase6Q Sparse Pairwise Ranker Anti-Overfit Audit",
        "",
        "Validation-only. Test lockbox is not used.",
        "",
        "## Why This Exists",
        "",
        "Phase6P showed a large train ranking spread but weak 2019 validation spread. Phase6Q therefore uses capped sparse pairs, purged walk-forward folds, and deliberately weak LightGBM models to test whether the price-only nonlinear ranker survives stricter anti-overfit discipline.",
        "",
        "## Controls",
        "",
        f"- Pairs per date: {PAIRS_PER_DATE}",
        f"- Per-stock pair appearance cap per date: {PAIR_STOCK_APPEARANCE_CAP}",
        f"- Purged gap before each OOS year: {PURGE_SESSIONS} sessions",
        f"- Max train pairs per fold/model/label: {MAX_TRAIN_PAIRS_PER_FOLD:,}",
        f"- Ranking reference count: {RANKING_REFERENCE_COUNT}",
        "",
        "## Pair Sampling Summary",
        "",
        _markdown_table(sampling_rollup),
        "",
        "## Pairwise AUC / Overfit Gap",
        "",
        _markdown_table(
            degradation[
                [
                    "label",
                    "fold",
                    "model",
                    "train_auc",
                    "eval_auc",
                    "auc_gap_train_minus_eval",
                    "mean_top_bottom_spread_bps",
                    "mean_rank_ic",
                    "spread_positive_rate",
                ]
            ].sort_values(["label", "fold", "model"])
        ),
        "",
        "## Strict Residual OOS Ranking",
        "",
        _markdown_table(strict_oos.sort_values(["fold", "model"])),
        "",
        "## 2019 Label Ladder",
        "",
        _markdown_table(final_2019.sort_values(["label", "model"])),
        "",
        "## 2019 Baselines",
        "",
        _markdown_table(baseline_2019.sort_values(["label", "score"])),
        "",
        "## Mean Feature Importance",
        "",
        _markdown_table(feature_rollup.groupby(["label", "model"]).head(8)),
        "",
        f"Plot: `{plot_path.as_posix()}`",
        "",
        "## Interpretation Rule",
        "",
        "If strict-style-residual OOS spread stays near zero under weak models, the pairwise ML idea is not yet a tradable pure-alpha selector. If beta-residual labels look materially stronger than strict labels, the original price signal is mostly harvesting known style payoff rather than a robust residual alpha.",
    ]
    return "\n".join(lines) + "\n"


def _stable_seed(*parts: str, offset: int = 0) -> int:
    value = RANDOM_SEED + offset
    for part in parts:
        for char in str(part):
            value = (value * 131 + ord(char)) % 2_147_483_647
    return int(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
