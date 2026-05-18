"""Phase6X: rolling h1 enhanced-feature pairwise ranker.

Each prediction month is trained with only prior information:

    12 calendar months train -> 3 calendar months early-stopping validation
    -> 1 calendar month prediction

The evaluation is day1-only. This avoids the misleading day1-10 sum view for
short-horizon labels and checks whether walk-forward validation can control the
overfit seen in Phase6W.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score

from stockmachine.apps import run_pure_alpha_phase6w as phase6w
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT


OUTDIR = RESEARCH_ROOT / "phase6x_rolling_h1_enhanced_pairwise_20260513"
PANEL_PATH = phase6w.OUTDIR / f"{phase6w.OUTPUT_PREFIX}_label_feature_panel.csv.gz"
LABEL_COLUMN = phase6w.LABEL_COLUMN
FEATURE_COLUMNS = tuple(phase6w.ENHANCED_FEATURES)
FEATURE_X_COLUMNS = tuple(f"{feature}_x" for feature in FEATURE_COLUMNS)

PREDICT_START = "2018-01-01"
PREDICT_END = "2019-12-31"
TRAIN_MONTHS = 12
EARLY_STOP_MONTHS = 3
PREDICT_MONTHS = 1
PAIRS_PER_DATE = 1000
RANKING_REFERENCE_COUNT = 96
TOP_BOTTOM_FRACTION = 0.20
RANDOM_SEED = 20260513


def main(argv: Sequence[str] | None = None) -> None:
    rollup = build_phase6x_rolling_h1_enhanced_pairwise()
    print(json.dumps(rollup, indent=2, ensure_ascii=True))


def build_phase6x_rolling_h1_enhanced_pairwise() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = _load_panel()
    windows = _rolling_windows(PREDICT_START, PREDICT_END)

    daily_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    monthly_score_rows: list[pd.DataFrame] = []

    for window_id, window in enumerate(windows, start=1):
        train = _slice(panel, window["train_start"], window["train_end"])
        early = _slice(panel, window["early_stop_start"], window["early_stop_end"])
        predict = _slice(panel, window["predict_start"], window["predict_end"])
        if train.empty or early.empty or predict.empty:
            continue

        rng = np.random.default_rng(RANDOM_SEED + window_id * 101)
        train_pairs = _sample_sparse_pairs(train, rng)
        early_pairs = _sample_sparse_pairs(early, rng)
        if train_pairs.empty or early_pairs.empty:
            continue

        model = _fit_model(train_pairs, early_pairs)
        score_panel = _build_prediction_scores(predict, model, rng)
        monthly_score_rows.append(score_panel)
        daily_metrics = _daily_day1_metrics(score_panel)
        for row in daily_metrics:
            row.update(window)
            row["window_id"] = window_id
        daily_rows.extend(daily_metrics)
        model_rows.append(
            _model_row(
                window=window,
                window_id=window_id,
                model=model,
                train_pairs=train_pairs,
                early_pairs=early_pairs,
                score_panel=score_panel,
            )
        )

    daily = pd.DataFrame(daily_rows)
    monthly = _monthly_summary(daily)
    model_diagnostics = pd.DataFrame(model_rows)
    score_panel = pd.concat(monthly_score_rows, ignore_index=True) if monthly_score_rows else pd.DataFrame()
    overall = _overall_summary(daily)

    daily_path = OUTDIR / "phase6x_rolling_day1_daily_metrics.csv"
    monthly_path = OUTDIR / "phase6x_rolling_day1_monthly_summary.csv"
    model_path = OUTDIR / "phase6x_rolling_model_diagnostics.csv"
    score_path = OUTDIR / "phase6x_rolling_score_panel.csv.gz"
    overall_path = OUTDIR / "phase6x_rolling_day1_overall_summary.csv"
    plot_path = OUTDIR / "phase6x_rolling_day1_cumulative_spread.png"
    memo_path = OUTDIR / "phase6x_rolling_h1_enhanced_pairwise_memo.md"
    rollup_path = OUTDIR / "phase6x_rolling_h1_enhanced_pairwise_rollup.json"

    daily.to_csv(daily_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    model_diagnostics.to_csv(model_path, index=False)
    overall.to_csv(overall_path, index=False)
    score_panel.to_csv(score_path, index=False, compression="gzip")
    _plot_daily_spread(daily, plot_path)
    memo_path.write_text(
        _memo(
            overall=overall,
            monthly=monthly,
            model_diagnostics=model_diagnostics,
            plot_path=plot_path,
        ),
        encoding="utf-8",
    )

    rollup = {
        "phase": "phase6x_rolling_h1_enhanced_pairwise",
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
        "top_bottom_fraction": TOP_BOTTOM_FRACTION,
        "prediction_start": PREDICT_START,
        "prediction_end": PREDICT_END,
        "rows": {
            "input_panel": int(len(panel)),
            "windows": int(len(model_diagnostics)),
            "daily_metrics": int(len(daily)),
            "score_panel": int(len(score_panel)),
        },
        "outputs": {
            "daily_metrics": str(daily_path),
            "monthly_summary": str(monthly_path),
            "model_diagnostics": str(model_path),
            "overall_summary": str(overall_path),
            "score_panel": str(score_path),
            "plot": str(plot_path),
            "memo": str(memo_path),
        },
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_panel() -> pd.DataFrame:
    usecols = ["session_date", "symbol", LABEL_COLUMN, *FEATURE_X_COLUMNS]
    panel = pd.read_csv(PANEL_PATH, usecols=usecols, parse_dates=["session_date"], low_memory=False)
    panel["symbol"] = panel["symbol"].astype(str)
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[LABEL_COLUMN, *FEATURE_X_COLUMNS]
    )
    return panel.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _rolling_windows(start: str, end: str) -> list[dict[str, str]]:
    starts = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS")
    windows: list[dict[str, str]] = []
    for predict_start in starts:
        predict_end = predict_start + pd.DateOffset(months=PREDICT_MONTHS) - pd.DateOffset(days=1)
        if predict_end > pd.Timestamp(end):
            predict_end = pd.Timestamp(end)
        early_stop_start = predict_start - pd.DateOffset(months=EARLY_STOP_MONTHS)
        early_stop_end = predict_start - pd.DateOffset(days=1)
        train_start = early_stop_start - pd.DateOffset(months=TRAIN_MONTHS)
        train_end = early_stop_start - pd.DateOffset(days=1)
        windows.append(
            {
                "predict_month": predict_start.strftime("%Y-%m"),
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "early_stop_start": early_stop_start.strftime("%Y-%m-%d"),
                "early_stop_end": early_stop_end.strftime("%Y-%m-%d"),
                "predict_start": predict_start.strftime("%Y-%m-%d"),
                "predict_end": predict_end.strftime("%Y-%m-%d"),
            }
        )
    return windows


def _slice(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = panel["session_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    return panel.loc[mask].copy()


def _sample_sparse_pairs(panel: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    feature_cols = list(FEATURE_X_COLUMNS)
    for session_date, group in panel.groupby("session_date", sort=True):
        group = group.reset_index(drop=True)
        y = group[LABEL_COLUMN].to_numpy(dtype=float)
        if len(group) < 100 or np.nanstd(y) <= 1e-12:
            continue
        iqr = float(np.nanquantile(y, 0.75) - np.nanquantile(y, 0.25))
        margin_threshold = max(iqr * 0.25, 1e-8)
        sampled = _sample_pairs_for_group(y, margin_threshold, rng)
        if sampled.empty:
            continue
        values_i = group.loc[sampled["idx_i"].to_numpy(), feature_cols].to_numpy(dtype=float)
        values_j = group.loc[sampled["idx_j"].to_numpy(), feature_cols].to_numpy(dtype=float)
        diff = values_i - values_j
        pair_frame = pd.DataFrame(diff, columns=[f"d_{feature}" for feature in FEATURE_COLUMNS])
        pair_frame.insert(0, "session_date", session_date)
        pair_frame["label"] = sampled["label"].to_numpy(dtype=int)
        pair_frame["margin"] = sampled["margin"].to_numpy(dtype=float)
        rows.append(pair_frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=RANDOM_SEED)


def _sample_pairs_for_group(
    y: np.ndarray,
    margin_threshold: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(y)
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
            rows.append(
                {
                    "idx_i": int(l_idx if do_flip else w_idx),
                    "idx_j": int(w_idx if do_flip else l_idx),
                    "label": int(not do_flip),
                    "margin": margin,
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
        rows.append({"idx_i": i, "idx_j": j, "label": int(y[i] > y[j]), "margin": margin})
    return pd.DataFrame(rows[:PAIRS_PER_DATE])


def _fit_model(train_pairs: pd.DataFrame, early_pairs: pd.DataFrame) -> LGBMClassifier:
    feature_cols = [f"d_{feature}" for feature in FEATURE_COLUMNS]
    model = LGBMClassifier(
        objective="binary",
        n_estimators=700,
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
        train_pairs[feature_cols],
        train_pairs["label"].astype(int),
        eval_set=[(early_pairs[feature_cols], early_pairs["label"].astype(int))],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(40, verbose=False), log_evaluation(0)],
    )
    return model


def _build_prediction_scores(
    predict: pd.DataFrame,
    model: LGBMClassifier,
    rng: np.random.Generator,
) -> pd.DataFrame:
    model_feature_cols = [f"d_{feature}" for feature in FEATURE_COLUMNS]
    rows: list[pd.DataFrame] = []
    for session_date, group in predict.groupby("session_date", sort=True):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < 50:
            continue
        ref_n = min(RANKING_REFERENCE_COUNT, n)
        ref_idx = rng.choice(np.arange(n), size=ref_n, replace=False)
        features = group[list(FEATURE_X_COLUMNS)].to_numpy(dtype=float)
        ref_features = features[ref_idx]
        diff = (features[:, None, :] - ref_features[None, :, :]).reshape(n * ref_n, len(FEATURE_X_COLUMNS))
        diff_frame = pd.DataFrame(diff, columns=model_feature_cols)
        proba = model.predict_proba(diff_frame)[:, 1].reshape(n, ref_n)
        out = group[["session_date", "symbol", LABEL_COLUMN]].copy()
        out["pairwise_rank_score"] = (proba - 0.5).mean(axis=1)
        out["reference_count"] = ref_n
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _daily_day1_metrics(score_panel: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session_date, group in score_panel.groupby("session_date", sort=True):
        frame = group[["pairwise_rank_score", LABEL_COLUMN]].dropna()
        if len(frame) < 50:
            continue
        low = frame["pairwise_rank_score"].quantile(TOP_BOTTOM_FRACTION)
        high = frame["pairwise_rank_score"].quantile(1.0 - TOP_BOTTOM_FRACTION)
        top = frame.loc[frame["pairwise_rank_score"] >= high, LABEL_COLUMN]
        bottom = frame.loc[frame["pairwise_rank_score"] <= low, LABEL_COLUMN]
        rank_ic = frame["pairwise_rank_score"].rank().corr(frame[LABEL_COLUMN].rank())
        spread = float(top.mean() - bottom.mean())
        rows.append(
            {
                "session_date": session_date,
                "names": int(len(frame)),
                "rank_ic": float(rank_ic) if np.isfinite(rank_ic) else np.nan,
                "top_mean_bps": float(top.mean() * 10_000.0),
                "bottom_mean_bps": float(bottom.mean() * 10_000.0),
                "top_bottom_spread_bps": float(spread * 10_000.0),
            }
        )
    return rows


def _model_row(
    *,
    window: dict[str, str],
    window_id: int,
    model: LGBMClassifier,
    train_pairs: pd.DataFrame,
    early_pairs: pd.DataFrame,
    score_panel: pd.DataFrame,
) -> dict[str, Any]:
    feature_cols = [f"d_{feature}" for feature in FEATURE_COLUMNS]

    def _auc(frame: pd.DataFrame) -> float:
        labels = frame["label"].astype(int).to_numpy()
        if len(set(labels)) < 2:
            return np.nan
        pred = model.predict_proba(frame[feature_cols])[:, 1]
        return float(roc_auc_score(labels, pred))

    daily = pd.DataFrame(_daily_day1_metrics(score_panel))
    return {
        "window_id": window_id,
        **window,
        "train_pairs": int(len(train_pairs)),
        "early_stop_pairs": int(len(early_pairs)),
        "prediction_rows": int(len(score_panel)),
        "prediction_sessions": int(score_panel["session_date"].nunique()) if not score_panel.empty else 0,
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "train_pair_auc": _auc(train_pairs),
        "early_stop_pair_auc": _auc(early_pairs),
        "predict_day1_mean_spread_bps": float(daily["top_bottom_spread_bps"].mean()) if not daily.empty else np.nan,
        "predict_day1_positive_rate": float((daily["top_bottom_spread_bps"] > 0).mean()) if not daily.empty else np.nan,
    }


def _monthly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    grouped = daily.groupby("predict_month", sort=True)
    return grouped.agg(
        sessions=("session_date", "nunique"),
        mean_rank_ic=("rank_ic", "mean"),
        mean_top_mean_bps=("top_mean_bps", "mean"),
        mean_bottom_mean_bps=("bottom_mean_bps", "mean"),
        mean_top_bottom_spread_bps=("top_bottom_spread_bps", "mean"),
        spread_positive_rate=("top_bottom_spread_bps", lambda x: float((x > 0).mean())),
        spread_p10_bps=("top_bottom_spread_bps", lambda x: float(x.quantile(0.10))),
        spread_p50_bps=("top_bottom_spread_bps", "median"),
        spread_p90_bps=("top_bottom_spread_bps", lambda x: float(x.quantile(0.90))),
    ).reset_index()


def _overall_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    spread = daily["top_bottom_spread_bps"].dropna()
    rank_ic = daily["rank_ic"].dropna()
    return pd.DataFrame(
        [
            {
                "prediction_start": PREDICT_START,
                "prediction_end": PREDICT_END,
                "sessions": int(daily["session_date"].nunique()),
                "mean_rank_ic": float(rank_ic.mean()) if not rank_ic.empty else np.nan,
                "rank_ic_tstat": _tstat(rank_ic),
                "mean_top_bottom_spread_bps": float(spread.mean()) if not spread.empty else np.nan,
                "spread_tstat": _tstat(spread),
                "spread_positive_rate": float((spread > 0).mean()) if not spread.empty else np.nan,
                "spread_p10_bps": float(spread.quantile(0.10)) if not spread.empty else np.nan,
                "spread_p50_bps": float(spread.median()) if not spread.empty else np.nan,
                "spread_p90_bps": float(spread.quantile(0.90)) if not spread.empty else np.nan,
                "cumulative_spread_bps": float(spread.sum()) if not spread.empty else np.nan,
            }
        ]
    )


def _tstat(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    std = series.std(ddof=1)
    if std == 0 or pd.isna(std):
        return np.nan
    return float(series.mean() / (std / np.sqrt(len(series))))


def _plot_daily_spread(daily: pd.DataFrame, outpath: Path) -> None:
    if daily.empty:
        return
    plot = daily.sort_values("session_date").copy()
    plot["cum_spread_bps"] = plot["top_bottom_spread_bps"].fillna(0.0).cumsum()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(plot["session_date"], plot["cum_spread_bps"], color="#0f766e", linewidth=1.6)
    axes[0].axhline(0, color="#111827", linestyle="--", linewidth=0.8)
    axes[0].set_title("Phase6X rolling h1 enhanced pairwise: cumulative day1 spread")
    axes[0].set_ylabel("Cumulative bps")
    axes[1].bar(plot["session_date"], plot["top_bottom_spread_bps"], color="#2563eb", width=1.0)
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
    plot_path: Path,
) -> str:
    return "\n".join(
        [
            "# Phase6X Rolling H1 Enhanced Pairwise Ranker",
            "",
            "Date: 2026-05-13",
            "",
            "Scope: validation-only walk-forward. The test lockbox is not used.",
            "",
            "## Design",
            "",
            "- Train window: 12 calendar months.",
            "- Early-stopping validation window: next 3 calendar months.",
            "- Prediction window: next 1 calendar month.",
            "- Prediction months: 2018-01 through 2019-12.",
            "- Evaluation target: day1 strict residual only.",
            "- Bucket: Top20% minus Bottom20% by rolling pairwise score.",
            "",
            "## Overall Day1 Result",
            "",
            "```text",
            overall.to_string(index=False) if not overall.empty else "No rows.",
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
            "## Plot",
            "",
            f"`{plot_path}`",
            "",
        ]
    )


if __name__ == "__main__":
    main()
