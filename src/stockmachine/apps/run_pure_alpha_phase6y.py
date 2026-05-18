"""Phase6Y: rolling h1 ranker with profit-consistency model selection.

Phase6X used LightGBM early stopping on pairwise logloss. Phase6Y keeps the
same rolling 12m/3m/1m setup, but selects the model iteration using a day1
profit-path utility on both the training and early-stopping windows.

The utility rewards mean top-bottom residual spread and penalizes:
- left-tail daily spread,
- consecutive loss streaks,
- train/validation overfit gaps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from stockmachine.apps import run_pure_alpha_phase6x as phase6x
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT


OUTDIR = RESEARCH_ROOT / "phase6y_profit_consistency_rolling_h1_20260514"

PANEL_PATH = phase6x.PANEL_PATH
LABEL_COLUMN = phase6x.LABEL_COLUMN
FEATURE_COLUMNS = phase6x.FEATURE_COLUMNS
FEATURE_X_COLUMNS = phase6x.FEATURE_X_COLUMNS

PREDICT_START = phase6x.PREDICT_START
PREDICT_END = phase6x.PREDICT_END
TRAIN_MONTHS = phase6x.TRAIN_MONTHS
EARLY_STOP_MONTHS = phase6x.EARLY_STOP_MONTHS
PREDICT_MONTHS = phase6x.PREDICT_MONTHS
PAIRS_PER_DATE = phase6x.PAIRS_PER_DATE
RANKING_REFERENCE_COUNT = phase6x.RANKING_REFERENCE_COUNT
TOP_BOTTOM_FRACTION = phase6x.TOP_BOTTOM_FRACTION
RANDOM_SEED = 20260514

CANDIDATE_ITERATIONS = (1, 2, 4, 8, 16, 32, 64, 96, 128, 192)
UTILITY_REFERENCE_COUNT = 16
UTILITY_MAX_NAMES_PER_DATE = 240

LEFT_TAIL_PENALTY_WEIGHT = 0.20
LOSS_STREAK_PENALTY_WEIGHT = 0.05
POSITIVE_RATE_FLOOR = 0.50
POSITIVE_RATE_PENALTY_BPS = 100.0
EARLY_UTILITY_WEIGHT = 0.70
MIN_UTILITY_WEIGHT = 0.30
OVERFIT_MEAN_GAP_WEIGHT = 0.10


def main(argv: Sequence[str] | None = None) -> None:
    rollup = build_phase6y_profit_consistency_rolling_h1()
    print(json.dumps(rollup, indent=2, ensure_ascii=True))


def build_phase6y_profit_consistency_rolling_h1() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = phase6x._load_panel()
    windows = phase6x._rolling_windows(PREDICT_START, PREDICT_END)

    daily_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    monthly_score_rows: list[pd.DataFrame] = []

    for window_id, window in enumerate(windows, start=1):
        train = phase6x._slice(panel, window["train_start"], window["train_end"])
        early = phase6x._slice(panel, window["early_stop_start"], window["early_stop_end"])
        predict = phase6x._slice(panel, window["predict_start"], window["predict_end"])
        if train.empty or early.empty or predict.empty:
            continue

        rng = np.random.default_rng(RANDOM_SEED + window_id * 101)
        train_pairs = phase6x._sample_sparse_pairs(train, rng)
        early_pairs = phase6x._sample_sparse_pairs(early, rng)
        if train_pairs.empty or early_pairs.empty:
            continue

        model = _fit_full_model(train_pairs)
        selected, candidates = _select_iteration_by_profit_consistency(
            model=model,
            train_panel=train,
            early_panel=early,
            train_pairs=train_pairs,
            early_pairs=early_pairs,
            rng=rng,
        )
        for row in candidates:
            row.update(window)
            row["window_id"] = window_id
        candidate_rows.extend(candidates)

        score_panel = _build_prediction_scores(predict, model, rng, num_iteration=selected)
        monthly_score_rows.append(score_panel)

        daily_metrics = phase6x._daily_day1_metrics(score_panel)
        for row in daily_metrics:
            row.update(window)
            row["window_id"] = window_id
            row["selected_iteration"] = selected
        daily_rows.extend(daily_metrics)

        model_rows.append(
            _model_row(
                window=window,
                window_id=window_id,
                model=model,
                selected_iteration=selected,
                train_pairs=train_pairs,
                early_pairs=early_pairs,
                score_panel=score_panel,
                selected_candidate=[r for r in candidates if r["candidate_iteration"] == selected][0],
            )
        )

    daily = pd.DataFrame(daily_rows)
    monthly = phase6x._monthly_summary(daily)
    model_diagnostics = pd.DataFrame(model_rows)
    candidate_utility = pd.DataFrame(candidate_rows)
    score_panel = pd.concat(monthly_score_rows, ignore_index=True) if monthly_score_rows else pd.DataFrame()
    overall = phase6x._overall_summary(daily)

    daily_path = OUTDIR / "phase6y_day1_daily_metrics.csv"
    monthly_path = OUTDIR / "phase6y_day1_monthly_summary.csv"
    model_path = OUTDIR / "phase6y_model_diagnostics.csv"
    candidates_path = OUTDIR / "phase6y_candidate_utility.csv"
    score_path = OUTDIR / "phase6y_score_panel.csv.gz"
    overall_path = OUTDIR / "phase6y_day1_overall_summary.csv"
    plot_path = OUTDIR / "phase6y_day1_cumulative_spread.png"
    memo_path = OUTDIR / "phase6y_profit_consistency_memo.md"
    rollup_path = OUTDIR / "phase6y_profit_consistency_rollup.json"

    daily.to_csv(daily_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    model_diagnostics.to_csv(model_path, index=False)
    candidate_utility.to_csv(candidates_path, index=False)
    overall.to_csv(overall_path, index=False)
    score_panel.to_csv(score_path, index=False, compression="gzip")
    _plot_daily_spread(daily, plot_path)
    memo_path.write_text(
        _memo(
            overall=overall,
            monthly=monthly,
            model_diagnostics=model_diagnostics,
            candidate_utility=candidate_utility,
            plot_path=plot_path,
        ),
        encoding="utf-8",
    )

    rollup = {
        "phase": "phase6y_profit_consistency_rolling_h1",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "input_panel": str(PANEL_PATH),
        "label_column": LABEL_COLUMN,
        "feature_columns": list(FEATURE_COLUMNS),
        "train_months": TRAIN_MONTHS,
        "early_stop_months": EARLY_STOP_MONTHS,
        "predict_months": PREDICT_MONTHS,
        "pairs_per_date": PAIRS_PER_DATE,
        "ranking_reference_count": RANKING_REFERENCE_COUNT,
        "utility_reference_count": UTILITY_REFERENCE_COUNT,
        "utility_max_names_per_date": UTILITY_MAX_NAMES_PER_DATE,
        "candidate_iterations": list(CANDIDATE_ITERATIONS),
        "top_bottom_fraction": TOP_BOTTOM_FRACTION,
        "prediction_start": PREDICT_START,
        "prediction_end": PREDICT_END,
        "rows": {
            "input_panel": int(len(panel)),
            "windows": int(len(model_diagnostics)),
            "daily_metrics": int(len(daily)),
            "candidate_utility": int(len(candidate_utility)),
            "score_panel": int(len(score_panel)),
        },
        "outputs": {
            "daily_metrics": str(daily_path),
            "monthly_summary": str(monthly_path),
            "model_diagnostics": str(model_path),
            "candidate_utility": str(candidates_path),
            "overall_summary": str(overall_path),
            "score_panel": str(score_path),
            "plot": str(plot_path),
            "memo": str(memo_path),
        },
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _fit_full_model(train_pairs: pd.DataFrame) -> LGBMClassifier:
    feature_cols = [f"d_{feature}" for feature in FEATURE_COLUMNS]
    model = LGBMClassifier(
        objective="binary",
        n_estimators=max(CANDIDATE_ITERATIONS),
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
    model.fit(train_pairs[feature_cols], train_pairs["label"].astype(int))
    return model


def _select_iteration_by_profit_consistency(
    *,
    model: LGBMClassifier,
    train_panel: pd.DataFrame,
    early_panel: pd.DataFrame,
    train_pairs: pd.DataFrame,
    early_pairs: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[int, list[dict[str, Any]]]:
    train_utility_panel = _make_utility_panel(train_panel, rng)
    early_utility_panel = _make_utility_panel(early_panel, rng)
    rows: list[dict[str, Any]] = []
    for iteration in CANDIDATE_ITERATIONS:
        train_scores = _build_utility_scores(train_utility_panel, model, rng, num_iteration=iteration)
        early_scores = _build_utility_scores(early_utility_panel, model, rng, num_iteration=iteration)
        train_daily = pd.DataFrame(phase6x._daily_day1_metrics(train_scores))
        early_daily = pd.DataFrame(phase6x._daily_day1_metrics(early_scores))
        train_stats = _profit_utility(train_daily)
        early_stats = _profit_utility(early_daily)
        train_pair_auc = _pair_auc(model, train_pairs, num_iteration=iteration)
        early_pair_auc = _pair_auc(model, early_pairs, num_iteration=iteration)
        overfit_gap = max(0.0, train_stats["mean_spread_bps"] - early_stats["mean_spread_bps"])
        objective = (
            EARLY_UTILITY_WEIGHT * early_stats["utility"]
            + MIN_UTILITY_WEIGHT * min(train_stats["utility"], early_stats["utility"])
            - OVERFIT_MEAN_GAP_WEIGHT * overfit_gap
        )
        rows.append(
            {
                "candidate_iteration": int(iteration),
                "objective": float(objective),
                "train_pair_auc": train_pair_auc,
                "early_pair_auc": early_pair_auc,
                "overfit_mean_gap_bps": float(overfit_gap),
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"early_{k}": v for k, v in early_stats.items()},
            }
        )
    selected = int(max(rows, key=lambda row: (row["objective"], -row["candidate_iteration"]))["candidate_iteration"])
    return selected, rows


def _make_utility_panel(panel: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for session_date, group in panel.groupby("session_date", sort=True):
        if len(group) > UTILITY_MAX_NAMES_PER_DATE:
            seed = int(rng.integers(0, 2**31 - 1))
            group = group.sample(n=UTILITY_MAX_NAMES_PER_DATE, random_state=seed)
        rows.append(group[["session_date", "symbol", LABEL_COLUMN, *FEATURE_X_COLUMNS]].copy())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_utility_scores(
    panel: pd.DataFrame,
    model: LGBMClassifier,
    rng: np.random.Generator,
    *,
    num_iteration: int,
) -> pd.DataFrame:
    return _build_scores_with_reference_count(
        panel,
        model,
        rng,
        reference_count=UTILITY_REFERENCE_COUNT,
        num_iteration=num_iteration,
    )


def _build_prediction_scores(
    predict: pd.DataFrame,
    model: LGBMClassifier,
    rng: np.random.Generator,
    *,
    num_iteration: int,
) -> pd.DataFrame:
    return _build_scores_with_reference_count(
        predict,
        model,
        rng,
        reference_count=RANKING_REFERENCE_COUNT,
        num_iteration=num_iteration,
    )


def _build_scores_with_reference_count(
    panel: pd.DataFrame,
    model: LGBMClassifier,
    rng: np.random.Generator,
    *,
    reference_count: int,
    num_iteration: int,
) -> pd.DataFrame:
    model_feature_cols = [f"d_{feature}" for feature in FEATURE_COLUMNS]
    rows: list[pd.DataFrame] = []
    for session_date, group in panel.groupby("session_date", sort=True):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < 50:
            continue
        ref_n = min(reference_count, n)
        ref_idx = rng.choice(np.arange(n), size=ref_n, replace=False)
        features = group[list(FEATURE_X_COLUMNS)].to_numpy(dtype=float)
        ref_features = features[ref_idx]
        diff = (features[:, None, :] - ref_features[None, :, :]).reshape(n * ref_n, len(FEATURE_X_COLUMNS))
        diff_frame = pd.DataFrame(diff, columns=model_feature_cols)
        proba = model.predict_proba(diff_frame, num_iteration=num_iteration)[:, 1].reshape(n, ref_n)
        out = group[["session_date", "symbol", LABEL_COLUMN]].copy()
        out["pairwise_rank_score"] = (proba - 0.5).mean(axis=1)
        out["reference_count"] = ref_n
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _profit_utility(daily: pd.DataFrame) -> dict[str, float]:
    if daily.empty:
        return {
            "sessions": 0.0,
            "mean_spread_bps": np.nan,
            "spread_tstat": np.nan,
            "positive_rate": np.nan,
            "left_tail_p10_bps": np.nan,
            "worst_loss_streak_bps": np.nan,
            "max_consecutive_loss_days": np.nan,
            "utility": -1e9,
        }
    spread = daily["top_bottom_spread_bps"].dropna().astype(float)
    mean = float(spread.mean())
    p10 = float(spread.quantile(0.10))
    positive_rate = float((spread > 0).mean())
    worst_streak_bps, max_loss_days = _worst_loss_streak(spread)
    left_tail_penalty = max(0.0, -p10)
    positive_rate_penalty = max(0.0, POSITIVE_RATE_FLOOR - positive_rate) * POSITIVE_RATE_PENALTY_BPS
    utility = (
        mean
        - LEFT_TAIL_PENALTY_WEIGHT * left_tail_penalty
        - LOSS_STREAK_PENALTY_WEIGHT * worst_streak_bps
        - positive_rate_penalty
    )
    return {
        "sessions": float(len(spread)),
        "mean_spread_bps": mean,
        "spread_tstat": phase6x._tstat(spread),
        "positive_rate": positive_rate,
        "left_tail_p10_bps": p10,
        "worst_loss_streak_bps": worst_streak_bps,
        "max_consecutive_loss_days": float(max_loss_days),
        "utility": float(utility),
    }


def _worst_loss_streak(spread: pd.Series) -> tuple[float, int]:
    current_loss = 0.0
    current_days = 0
    worst_loss = 0.0
    worst_days = 0
    for value in spread:
        if value < 0:
            current_loss += -float(value)
            current_days += 1
            if current_loss > worst_loss:
                worst_loss = current_loss
                worst_days = current_days
        else:
            current_loss = 0.0
            current_days = 0
    return float(worst_loss), int(worst_days)


def _pair_auc(model: LGBMClassifier, pairs: pd.DataFrame, *, num_iteration: int) -> float:
    feature_cols = [f"d_{feature}" for feature in FEATURE_COLUMNS]
    labels = pairs["label"].astype(int).to_numpy()
    if len(set(labels)) < 2:
        return np.nan
    pred = model.predict_proba(pairs[feature_cols], num_iteration=num_iteration)[:, 1]
    return float(roc_auc_score(labels, pred))


def _model_row(
    *,
    window: dict[str, str],
    window_id: int,
    model: LGBMClassifier,
    selected_iteration: int,
    train_pairs: pd.DataFrame,
    early_pairs: pd.DataFrame,
    score_panel: pd.DataFrame,
    selected_candidate: dict[str, Any],
) -> dict[str, Any]:
    daily = pd.DataFrame(phase6x._daily_day1_metrics(score_panel))
    return {
        "window_id": window_id,
        **window,
        "train_pairs": int(len(train_pairs)),
        "early_stop_pairs": int(len(early_pairs)),
        "prediction_rows": int(len(score_panel)),
        "prediction_sessions": int(score_panel["session_date"].nunique()) if not score_panel.empty else 0,
        "selected_iteration": int(selected_iteration),
        "selected_objective": float(selected_candidate["objective"]),
        "selected_train_utility": float(selected_candidate["train_utility"]),
        "selected_early_utility": float(selected_candidate["early_utility"]),
        "selected_train_mean_spread_bps": float(selected_candidate["train_mean_spread_bps"]),
        "selected_early_mean_spread_bps": float(selected_candidate["early_mean_spread_bps"]),
        "selected_train_left_tail_p10_bps": float(selected_candidate["train_left_tail_p10_bps"]),
        "selected_early_left_tail_p10_bps": float(selected_candidate["early_left_tail_p10_bps"]),
        "selected_train_worst_loss_streak_bps": float(selected_candidate["train_worst_loss_streak_bps"]),
        "selected_early_worst_loss_streak_bps": float(selected_candidate["early_worst_loss_streak_bps"]),
        "train_pair_auc": _pair_auc(model, train_pairs, num_iteration=selected_iteration),
        "early_stop_pair_auc": _pair_auc(model, early_pairs, num_iteration=selected_iteration),
        "predict_day1_mean_spread_bps": float(daily["top_bottom_spread_bps"].mean()) if not daily.empty else np.nan,
        "predict_day1_positive_rate": float((daily["top_bottom_spread_bps"] > 0).mean()) if not daily.empty else np.nan,
    }


def _plot_daily_spread(daily: pd.DataFrame, outpath: Path) -> None:
    if daily.empty:
        return
    plot = daily.sort_values("session_date").copy()
    plot["cum_spread_bps"] = plot["top_bottom_spread_bps"].fillna(0.0).cumsum()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(plot["session_date"], plot["cum_spread_bps"], color="#7c2d12", linewidth=1.6)
    axes[0].axhline(0, color="#111827", linestyle="--", linewidth=0.8)
    axes[0].set_title("Phase6Y profit-consistency rolling h1: cumulative day1 spread")
    axes[0].set_ylabel("Cumulative bps")
    axes[1].bar(plot["session_date"], plot["top_bottom_spread_bps"], color="#c2410c", width=1.0)
    axes[1].axhline(0, color="#111827", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Daily bps")
    axes[1].set_xlabel("Prediction date")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _memo(
    *,
    overall: pd.DataFrame,
    monthly: pd.DataFrame,
    model_diagnostics: pd.DataFrame,
    candidate_utility: pd.DataFrame,
    plot_path: Path,
) -> str:
    selected_counts = (
        model_diagnostics["selected_iteration"].value_counts().sort_index().to_frame("count")
        if not model_diagnostics.empty
        else pd.DataFrame()
    )
    return "\n".join(
        [
            "# Phase6Y Profit-Consistency Rolling H1 Ranker",
            "",
            "Date: 2026-05-14",
            "",
            "Scope: validation-only walk-forward. The test lockbox is not used.",
            "",
            "## Design",
            "",
            "- Base setup: Phase6X 12m train / 3m validation / 1m prediction.",
            "- Label: day1 strict residual.",
            "- Candidate iterations: " + ", ".join(map(str, CANDIDATE_ITERATIONS)) + ".",
            "- Selection utility rewards day1 top-bottom mean spread and penalizes left tail, consecutive losses, and train/validation overfit gap.",
            "- Final report remains prediction-month day1 only.",
            "",
            "## Overall Day1 Result",
            "",
            "```text",
            overall.to_string(index=False) if not overall.empty else "No rows.",
            "```",
            "",
            "## Selected Iteration Counts",
            "",
            "```text",
            selected_counts.to_string() if not selected_counts.empty else "No rows.",
            "```",
            "",
            "## Monthly Day1 Result",
            "",
            "```text",
            monthly.to_string(index=False) if not monthly.empty else "No rows.",
            "```",
            "",
            "## Model Diagnostics",
            "",
            "```text",
            model_diagnostics.to_string(index=False) if not model_diagnostics.empty else "No rows.",
            "```",
            "",
            "## Candidate Utility Sample",
            "",
            "```text",
            candidate_utility.head(40).to_string(index=False) if not candidate_utility.empty else "No rows.",
            "```",
            "",
            "## Plot",
            "",
            f"`{plot_path}`",
            "",
        ]
    )


if __name__ == "__main__":
    main()
