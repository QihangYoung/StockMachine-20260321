"""Phase6V: sparse pairwise ranker supervised on h1 residual returns.

Phase6P trained the same pairwise model on an aggregate h10 strict residual
label. Phase6U showed the validation edge is mostly front-loaded, so this run
keeps the model family fixed and changes only the supervision target to the
first holding-day residual return.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from stockmachine.apps import run_pure_alpha_phase6p as phase6p
from stockmachine.apps.run_pure_alpha_phase3 import (
    DEFAULT_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_DAILY_GLOBS,
    DEFAULT_DAILY_GLOBS,
)
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4s import _daily_residualize, _zscore
from stockmachine.apps.run_pure_alpha_phase4z import _load_open_to_open_returns


OUTDIR = RESEARCH_ROOT / "phase6v_h1_sparse_pairwise_ranker_20260513"
PHASE_NAME = "phase6v_h1_sparse_pairwise_ranker"
OUTPUT_PREFIX = "phase6v_h1"
MEMO_TITLE = "Phase6V H1 Sparse Pairwise Ranker"
LABEL_COLUMN = "true_size_style_sic2_h1_residual"
H1_TARGET_COLUMN = "h1_beta_residual_oto_return"
H1_RAW_RETURN_COLUMN = "h1_raw_oto_return"
H1_BENCHMARK_RETURN_COLUMN = "h1_benchmark_oto_return"
RANDOM_SEED = 20260513
VALIDATION_PRICE_END = "2019-12-31"
FEATURE_COLUMNS = phase6p.BASE_FEATURES
PREBUILT_PANEL_PATH: Path | None = None
WRITE_PAIR_SAMPLE = os.environ.get("PHASE6V_WRITE_PAIR_SAMPLE", "1") != "0"


def main(argv: Sequence[str] | None = None) -> None:
    rollup = build_phase6v_h1_sparse_pairwise_ranker()
    print(json.dumps(rollup, indent=2, ensure_ascii=True))


def build_phase6v_h1_sparse_pairwise_ranker() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    _patch_phase6p_globals()

    panel = _load_or_build_panel()
    pairs = phase6p._sample_sparse_pairs(panel)
    model, model_metrics = phase6p._fit_pairwise_model(pairs)
    score_panel = phase6p._build_ranking_scores(panel, model)
    ranking_metrics = phase6p._build_ranking_metrics(score_panel)
    baseline_metrics = phase6p._build_baseline_ranking_metrics(panel)
    feature_importance = phase6p._feature_importance(model)

    pair_path = OUTDIR / f"{OUTPUT_PREFIX}_sparse_pair_sample.csv.gz"
    panel_path = OUTDIR / f"{OUTPUT_PREFIX}_label_feature_panel.csv.gz"
    score_path = OUTDIR / f"{OUTPUT_PREFIX}_pairwise_score_panel.csv.gz"
    model_metrics_path = OUTDIR / f"{OUTPUT_PREFIX}_pairwise_model_metrics.csv"
    ranking_metrics_path = OUTDIR / f"{OUTPUT_PREFIX}_ranking_metrics.csv"
    baseline_metrics_path = OUTDIR / f"{OUTPUT_PREFIX}_baseline_ranking_metrics.csv"
    feature_importance_path = OUTDIR / f"{OUTPUT_PREFIX}_feature_importance.csv"
    model_path = OUTDIR / f"{OUTPUT_PREFIX}_lgbm_pairwise_model.joblib"
    plot_path = OUTDIR / f"{OUTPUT_PREFIX}_ranking_diagnostics_plot.png"
    memo_path = OUTDIR / f"{OUTPUT_PREFIX}_sparse_pairwise_ranker_memo.md"
    rollup_path = OUTDIR / f"{OUTPUT_PREFIX}_rollup.json"

    panel.to_csv(panel_path, index=False, compression="gzip")
    if WRITE_PAIR_SAMPLE:
        pairs.to_csv(pair_path, index=False, compression="gzip")
    score_panel.to_csv(score_path, index=False, compression="gzip")
    model_metrics.to_csv(model_metrics_path, index=False)
    ranking_metrics.to_csv(ranking_metrics_path, index=False)
    baseline_metrics.to_csv(baseline_metrics_path, index=False)
    feature_importance.to_csv(feature_importance_path, index=False)
    joblib.dump(model, model_path)
    phase6p._plot_ranking_diagnostics(ranking_metrics, baseline_metrics, plot_path)
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
        "phase": PHASE_NAME,
        "created_at_utc": phase6p._utc_now(),
        "scope": "validation_only",
        "test_lockbox_used": False,
        "label_column": LABEL_COLUMN,
        "target_source_column": H1_TARGET_COLUMN,
        "feature_columns": list(FEATURE_COLUMNS),
        "pairs_per_date": phase6p.PAIRS_PER_DATE,
        "ranking_reference_count": phase6p.RANKING_REFERENCE_COUNT,
        "margin_iqr_multiplier": phase6p.MARGIN_IQR_MULTIPLIER,
        "adv_floor_usd": phase6p.ADV_FLOOR_USD,
        "splits": [split.__dict__ for split in phase6p.SPLITS],
        "rows": {
            "panel": int(len(panel)),
            "pairs": int(len(pairs)),
            "score_panel": int(len(score_panel)),
            "model_metrics": int(len(model_metrics)),
            "ranking_metrics": int(len(ranking_metrics)),
        },
        "outputs": {
            "label_feature_panel": str(panel_path),
            "pair_sample": str(pair_path) if WRITE_PAIR_SAMPLE else None,
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


def _load_or_build_panel() -> pd.DataFrame:
    if PREBUILT_PANEL_PATH is not None and PREBUILT_PANEL_PATH.exists():
        return pd.read_csv(PREBUILT_PANEL_PATH, parse_dates=["session_date"], low_memory=False)
    return _load_h1_training_panel()


def _patch_phase6p_globals() -> None:
    phase6p.OUTDIR = OUTDIR
    phase6p.LABEL_COLUMN = LABEL_COLUMN
    phase6p.RANDOM_SEED = RANDOM_SEED
    phase6p.BASE_FEATURES = FEATURE_COLUMNS


def _load_h1_training_panel() -> pd.DataFrame:
    panel = phase6p._load_feature_panel()
    panel = panel[panel["variant"].astype(str).eq(phase6p.DEFAULT_LONG_VARIANT)].copy()
    panel["session_date"] = pd.to_datetime(panel["session_date"])
    panel["symbol"] = panel["symbol"].astype(str)
    if "test_window_used" in panel.columns and panel["test_window_used"].map(phase6p._is_true).any():
        raise ValueError("Refusing to run: panel includes test-window rows.")

    size = pd.read_csv(
        phase6p.SIZE_PANEL_PATH,
        usecols=[
            "variant",
            "session_date",
            "symbol",
            "market_cap_log_z",
            "market_cap_mvp_available",
        ],
        low_memory=False,
    )
    size = size[size["variant"].astype(str).eq(phase6p.DEFAULT_LONG_VARIANT)].copy()
    size["session_date"] = pd.to_datetime(size["session_date"])
    size["symbol"] = size["symbol"].astype(str)
    size["market_cap_log_z"] = pd.to_numeric(size["market_cap_log_z"], errors="coerce")
    panel = panel.merge(
        size[["session_date", "symbol", "market_cap_log_z", "market_cap_mvp_available"]],
        on=["session_date", "symbol"],
        how="left",
    )

    numeric_columns = [
        "beta",
        "lagged_close",
        "trailing_median_dollar_volume_20",
        "liquidity_rank",
        "reversal_5d",
        "momentum_20d",
        "momentum_60d",
        "beta_residual_momentum_20d",
        "vol_adjusted_momentum_20d",
        "market_cap_log_z",
    ]
    for column in numeric_columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

    panel = panel[panel["trailing_median_dollar_volume_20"].fillna(0.0) >= phase6p.ADV_FLOOR_USD].copy()
    panel = _attach_h1_returns(panel)
    panel = _add_price_state_features(panel)
    panel = _add_extra_features(panel)

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
        target_column=H1_TARGET_COLUMN,
        numeric_features=label_features,
        categorical_features=("sic2_sector",),
        min_regression_rows=80,
        min_dummy_count=5,
    )

    for feature in FEATURE_COLUMNS:
        panel[feature] = pd.to_numeric(panel[feature], errors="coerce")
        panel[f"{feature}_x"] = panel.groupby("session_date", sort=False)[feature].transform(_zscore)

    feature_cols = [f"{feature}_x" for feature in FEATURE_COLUMNS]
    required = ["session_date", "symbol", LABEL_COLUMN, *feature_cols]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    panel["split"] = panel["session_date"].map(phase6p._split_for_date)
    panel = panel[panel["split"].notna()].copy()
    return panel.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _attach_h1_returns(panel: pd.DataFrame) -> pd.DataFrame:
    stock_returns = _load_open_to_open_returns(
        daily_globs=DEFAULT_DAILY_GLOBS,
        adj_factor_globs=DEFAULT_ADJ_FACTOR_GLOBS,
        symbols=panel["symbol"].unique(),
        end_date=VALIDATION_PRICE_END,
    ).rename(columns={"session_date": "return_date", "oto_return": H1_RAW_RETURN_COLUMN})
    benchmark_returns = _load_open_to_open_returns(
        daily_globs=DEFAULT_BENCHMARK_DAILY_GLOBS,
        adj_factor_globs=DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
        symbols=["SPY"],
        end_date=VALIDATION_PRICE_END,
    ).rename(columns={"session_date": "return_date", "oto_return": H1_BENCHMARK_RETURN_COLUMN})
    benchmark_returns = benchmark_returns[["return_date", H1_BENCHMARK_RETURN_COLUMN]].drop_duplicates()

    calendar_map = _make_h1_calendar_map(panel["session_date"], benchmark_returns)
    frame = panel.merge(calendar_map, on="session_date", how="inner")
    frame = frame.merge(stock_returns, on=["return_date", "symbol"], how="inner")
    frame = frame.merge(benchmark_returns, on="return_date", how="inner")
    frame[H1_TARGET_COLUMN] = frame[H1_RAW_RETURN_COLUMN] - frame["beta"] * frame[H1_BENCHMARK_RETURN_COLUMN]
    return frame


def _make_h1_calendar_map(decision_dates: pd.Series, benchmark_returns: pd.DataFrame) -> pd.DataFrame:
    calendar = benchmark_returns["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
    calendar_index = {date: idx for idx, date in enumerate(calendar)}
    rows: list[dict[str, pd.Timestamp]] = []
    for session_date in sorted(decision_dates.drop_duplicates()):
        idx = calendar_index.get(session_date)
        if idx is None:
            continue
        return_idx = idx + 2
        if return_idx >= len(calendar):
            continue
        rows.append({"session_date": session_date, "return_date": calendar.iloc[return_idx]})
    return pd.DataFrame(rows)


def _add_price_state_features(panel: pd.DataFrame) -> pd.DataFrame:
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
    return panel


def _add_extra_features(panel: pd.DataFrame) -> pd.DataFrame:
    return panel


def _memo(
    *,
    model_metrics: pd.DataFrame,
    ranking_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    plot_path: Path,
) -> str:
    return "\n".join(
        [
            f"# {MEMO_TITLE}",
            "",
            "Date: 2026-05-13",
            "",
            "Scope: validation only. The test lockbox is not used.",
            "",
            "## Objective",
            "",
            "Keep the Phase6P pairwise model family fixed, but change the supervised label from aggregate h10 strict residual return to the first holding-day strict residual return.",
            "",
            "## Model Quality",
            "",
            phase6p._markdown_table(model_metrics),
            "",
            "## Ranking Quality on H1 Label",
            "",
            phase6p._markdown_table(ranking_metrics),
            "",
            "## Baseline Ranking Quality on H1 Label",
            "",
            phase6p._markdown_table(baseline_metrics),
            "",
            "## Top Feature Importances",
            "",
            phase6p._markdown_table(feature_importance.head(20)),
            "",
            "## Plot",
            "",
            f"`{plot_path}`",
            "",
            "## Interpretation Rule",
            "",
            "This experiment is useful only if h1 supervision improves validation day1 behavior without destroying the day1-10 forward profile.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
