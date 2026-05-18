"""Phase6U: daily forward-horizon profile for the sparse pairwise ranker.

The Phase6P pairwise model was originally judged on an h10 aggregate target.
This diagnostic decomposes the same score into holding-day 1..10 outcomes, so
we can see whether the small validation h10 spread is diffuse decay or a few
specific days giving back gains.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase3 import (
    DEFAULT_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_DAILY_GLOBS,
    DEFAULT_DAILY_GLOBS,
)
from stockmachine.apps.run_pure_alpha_phase4s import _daily_residualize
from stockmachine.apps.run_pure_alpha_phase4z import _load_open_to_open_returns


PROJECT_DIR = Path("artifacts/strategy_projects/us_equities_pure_alpha_h5")
PHASE6P_DIR = PROJECT_DIR / "research/phase6p_sparse_pairwise_ranker_20260511"
SOURCE_DIR = Path(os.environ.get("PHASE6U_SOURCE_DIR", str(PHASE6P_DIR)))


def _configured_top_bottom_fraction() -> float:
    fraction = float(os.environ.get("PHASE6U_TOP_BOTTOM_FRACTION", "0.20"))
    if not 0.0 < fraction < 0.50:
        raise ValueError("PHASE6U_TOP_BOTTOM_FRACTION must be between 0 and 0.50.")
    return fraction


TOP_BOTTOM_FRACTION = _configured_top_bottom_fraction()
TOP_BOTTOM_PERCENT = int(round(TOP_BOTTOM_FRACTION * 100.0))
_OUTPUT_DIR_NAME = "phase6u_pairwise_ranker_daily_horizon_profile_20260513"
if os.environ.get("PHASE6U_OUTPUT_DIR"):
    OUTPUT_DIR = Path(os.environ["PHASE6U_OUTPUT_DIR"])
elif TOP_BOTTOM_PERCENT == 20:
    OUTPUT_DIR = PROJECT_DIR / f"research/{_OUTPUT_DIR_NAME}"
else:
    OUTPUT_DIR = PROJECT_DIR / f"research/{_OUTPUT_DIR_NAME}_top{TOP_BOTTOM_PERCENT:02d}"

SCORE_PANEL_PATH = Path(
    os.environ.get(
        "PHASE6U_SCORE_PANEL_PATH",
        str(SOURCE_DIR / "phase6p_pairwise_score_panel.csv.gz"),
    )
)
LABEL_FEATURE_PANEL_PATH = Path(
    os.environ.get(
        "PHASE6U_LABEL_FEATURE_PANEL_PATH",
        str(SOURCE_DIR / "phase6p_label_feature_panel.csv.gz"),
    )
)

VALIDATION_PRICE_END = "2019-12-31"
HOLDING_DAYS = tuple(range(1, 11))

SCORE_COLUMNS = {
    "pairwise_rank_score": "pairwise_rank_score",
    "reversal_5d": "reversal_5d",
    "anti_momentum_20d": "anti_momentum_20d_score",
    "anti_beta_residual_momentum_20d": "anti_beta_residual_momentum_20d_score",
    "anti_vol_adjusted_momentum_20d": "anti_vol_adjusted_momentum_20d_score",
}

TARGET_COLUMNS = {
    "raw_oto": "raw_day_return",
    "beta_residual_oto": "beta_residual_day_return",
    "strict_style_sic2_residual_oto": "strict_style_sic2_day_residual",
}

STYLE_RESIDUAL_NUMERIC_FEATURES = (
    "beta",
    "lagged_close_log",
    "trailing_median_dollar_volume_20_log",
    "liquidity_rank",
    "market_cap_log_z",
    "reversal_5d",
    "momentum_20d",
    "momentum_60d",
    "beta_residual_momentum_20d",
    "vol_adjusted_momentum_20d",
)

STYLE_RESIDUAL_CATEGORICAL_FEATURES = ("sic2_sector",)


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_score_and_feature_panel() -> pd.DataFrame:
    score_usecols = [
        "session_date",
        "symbol",
        "split",
        *SCORE_COLUMNS.values(),
    ]
    feature_usecols = [
        "session_date",
        "symbol",
        "beta",
        "lagged_close_log",
        "trailing_median_dollar_volume_20_log",
        "liquidity_rank",
        "market_cap_log_z",
        "momentum_20d",
        "momentum_60d",
        "beta_residual_momentum_20d",
        "vol_adjusted_momentum_20d",
        "sic2_sector",
    ]

    scores = pd.read_csv(SCORE_PANEL_PATH, usecols=score_usecols, parse_dates=["session_date"])
    features = pd.read_csv(
        LABEL_FEATURE_PANEL_PATH,
        usecols=feature_usecols,
        parse_dates=["session_date"],
    )
    panel = scores.merge(features, on=["session_date", "symbol"], how="inner", validate="one_to_one")
    panel = panel.replace([np.inf, -np.inf], np.nan)

    required = [
        "split",
        *SCORE_COLUMNS.values(),
        *STYLE_RESIDUAL_NUMERIC_FEATURES,
        *STYLE_RESIDUAL_CATEGORICAL_FEATURES,
    ]
    panel = panel.dropna(subset=required).copy()
    panel["sic2_sector"] = panel["sic2_sector"].astype(str)
    return panel


def _load_validation_returns(symbols: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_returns = _load_open_to_open_returns(
        daily_globs=DEFAULT_DAILY_GLOBS,
        adj_factor_globs=DEFAULT_ADJ_FACTOR_GLOBS,
        symbols=symbols,
        end_date=VALIDATION_PRICE_END,
    ).rename(columns={"session_date": "return_date", "oto_return": "raw_day_return"})
    benchmark_returns = _load_open_to_open_returns(
        daily_globs=DEFAULT_BENCHMARK_DAILY_GLOBS,
        adj_factor_globs=DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
        symbols=["SPY"],
        end_date=VALIDATION_PRICE_END,
    ).rename(columns={"session_date": "return_date", "oto_return": "benchmark_day_return"})
    benchmark_returns = benchmark_returns[["return_date", "benchmark_day_return"]].drop_duplicates()
    return stock_returns, benchmark_returns


def _make_horizon_calendar_map(
    decision_dates: pd.Series,
    benchmark_returns: pd.DataFrame,
    horizon_day: int,
) -> pd.DataFrame:
    calendar = benchmark_returns["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
    calendar_index = {date: idx for idx, date in enumerate(calendar)}

    rows: list[dict[str, pd.Timestamp]] = []
    for session_date in sorted(decision_dates.drop_duplicates()):
        idx = calendar_index.get(session_date)
        if idx is None:
            continue
        return_idx = idx + 1 + horizon_day
        if return_idx >= len(calendar):
            continue
        rows.append({"session_date": session_date, "return_date": calendar.iloc[return_idx]})
    return pd.DataFrame(rows)


def _attach_forward_day_returns(
    base_panel: pd.DataFrame,
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    horizon_day: int,
) -> pd.DataFrame:
    calendar_map = _make_horizon_calendar_map(
        base_panel["session_date"],
        benchmark_returns,
        horizon_day=horizon_day,
    )
    frame = base_panel.merge(calendar_map, on="session_date", how="inner")
    frame = frame.merge(stock_returns, on=["return_date", "symbol"], how="inner")
    frame = frame.merge(benchmark_returns, on="return_date", how="inner")
    frame["variant"] = "phase6p_top1000_pairwise_panel"
    frame["beta_residual_day_return"] = (
        frame["raw_day_return"] - frame["beta"] * frame["benchmark_day_return"]
    )
    frame["strict_style_sic2_day_residual"] = _daily_residualize(
        frame,
        target_column="beta_residual_day_return",
        numeric_features=STYLE_RESIDUAL_NUMERIC_FEATURES,
        categorical_features=STYLE_RESIDUAL_CATEGORICAL_FEATURES,
        min_regression_rows=80,
        min_dummy_count=5,
    )
    frame["horizon_day"] = horizon_day
    return frame.replace([np.inf, -np.inf], np.nan)


def _rank_ic(x: pd.Series, y: pd.Series) -> float:
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return np.nan
    return float(x.rank().corr(y.rank()))


def _daily_top_bottom_metrics(
    group: pd.DataFrame,
    score_col: str,
    target_col: str,
) -> dict[str, float] | None:
    data = group[[score_col, target_col]].dropna()
    if len(data) < 50:
        return None

    low_cut = data[score_col].quantile(TOP_BOTTOM_FRACTION)
    high_cut = data[score_col].quantile(1.0 - TOP_BOTTOM_FRACTION)
    top = data.loc[data[score_col] >= high_cut, target_col]
    bottom = data.loc[data[score_col] <= low_cut, target_col]
    if len(top) == 0 or len(bottom) == 0:
        return None

    top_mean = float(top.mean())
    bottom_mean = float(bottom.mean())
    spread = top_mean - bottom_mean
    return {
        "stock_count": float(len(data)),
        "top_count": float(len(top)),
        "bottom_count": float(len(bottom)),
        "rank_ic": _rank_ic(data[score_col], data[target_col]),
        "top_mean_bps": top_mean * 10_000.0,
        "bottom_mean_bps": bottom_mean * 10_000.0,
        "top_bottom_spread_bps": spread * 10_000.0,
    }


def _build_daily_metric_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(["split", "session_date", "horizon_day"], observed=True)
    for (split, session_date, horizon_day), group in grouped:
        for score_name, score_col in SCORE_COLUMNS.items():
            for target_name, target_col in TARGET_COLUMNS.items():
                metrics = _daily_top_bottom_metrics(group, score_col=score_col, target_col=target_col)
                if metrics is None:
                    continue
                rows.append(
                    {
                        "split": split,
                        "session_date": session_date,
                        "horizon_day": int(horizon_day),
                        "score_name": score_name,
                        "target_name": target_name,
                        **metrics,
                    }
                )
    return rows


def _summarize_daily_metrics(daily_metrics: pd.DataFrame) -> pd.DataFrame:
    def _tstat(series: pd.Series) -> float:
        series = series.dropna()
        if len(series) < 2:
            return np.nan
        std = series.std(ddof=1)
        if std == 0 or pd.isna(std):
            return np.nan
        return float(series.mean() / (std / np.sqrt(len(series))))

    grouped = daily_metrics.groupby(
        ["split", "score_name", "target_name", "horizon_day"],
        observed=True,
    )
    summary = grouped.agg(
        sessions=("session_date", "nunique"),
        mean_stock_count=("stock_count", "mean"),
        mean_rank_ic=("rank_ic", "mean"),
        rank_ic_tstat=("rank_ic", _tstat),
        mean_top_mean_bps=("top_mean_bps", "mean"),
        mean_bottom_mean_bps=("bottom_mean_bps", "mean"),
        mean_top_bottom_spread_bps=("top_bottom_spread_bps", "mean"),
        spread_tstat=("top_bottom_spread_bps", _tstat),
        spread_positive_rate=("top_bottom_spread_bps", lambda x: float((x > 0).mean())),
        spread_p10_bps=("top_bottom_spread_bps", lambda x: float(x.quantile(0.10))),
        spread_p50_bps=("top_bottom_spread_bps", "median"),
        spread_p90_bps=("top_bottom_spread_bps", lambda x: float(x.quantile(0.90))),
    ).reset_index()
    return summary


def _build_cumulative_summary(summary: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in summary.sort_values("horizon_day").groupby(
        ["split", "score_name", "target_name"],
        observed=True,
    ):
        group = group.copy()
        group["cumulative_mean_spread_bps"] = group["mean_top_bottom_spread_bps"].cumsum()
        group["cumulative_mean_top_bps"] = group["mean_top_mean_bps"].cumsum()
        group["cumulative_mean_bottom_bps"] = group["mean_bottom_mean_bps"].cumsum()
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def _plot_pairwise_profiles(summary: pd.DataFrame, cumulative: pd.DataFrame) -> dict[str, str]:
    paths: dict[str, str] = {}
    pairwise = summary[summary["score_name"].eq("pairwise_rank_score")]
    pairwise_cum = cumulative[cumulative["score_name"].eq("pairwise_rank_score")]

    for target_name in TARGET_COLUMNS:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        target_daily = pairwise[pairwise["target_name"].eq(target_name)]
        target_cum = pairwise_cum[pairwise_cum["target_name"].eq(target_name)]
        for split, group in target_daily.groupby("split", observed=True):
            group = group.sort_values("horizon_day")
            axes[0].plot(
                group["horizon_day"],
                group["mean_top_bottom_spread_bps"],
                marker="o",
                label=str(split),
            )
        for split, group in target_cum.groupby("split", observed=True):
            group = group.sort_values("horizon_day")
            axes[1].plot(
                group["horizon_day"],
                group["cumulative_mean_spread_bps"],
                marker="o",
                label=str(split),
            )

        axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_title(f"Phase6U pairwise score: daily top-bottom spread ({target_name})")
        axes[0].set_ylabel("Daily spread, bps")
        axes[1].set_ylabel("Cumulative spread, bps")
        axes[1].set_xlabel("Forward holding day")
        axes[0].legend(loc="best")
        axes[1].legend(loc="best")
        fig.tight_layout()
        path = OUTPUT_DIR / f"phase6u_pairwise_{target_name}_daily_profile.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths[target_name] = str(path)

    return paths


def _format_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    table = df.loc[:, columns].head(max_rows).copy()
    for column in table.select_dtypes(include=[np.number]).columns:
        table[column] = table[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _write_markdown_report(summary: pd.DataFrame, cumulative: pd.DataFrame, plot_paths: dict[str, str]) -> Path:
    report_path = OUTPUT_DIR / "phase6u_pairwise_ranker_daily_horizon_profile.md"
    pairwise_strict = summary[
        summary["score_name"].eq("pairwise_rank_score")
        & summary["target_name"].eq("strict_style_sic2_residual_oto")
    ].sort_values(["split", "horizon_day"])
    pairwise_strict_cum = cumulative[
        cumulative["score_name"].eq("pairwise_rank_score")
        & cumulative["target_name"].eq("strict_style_sic2_residual_oto")
    ].sort_values(["split", "horizon_day"])

    final_day = pairwise_strict_cum[pairwise_strict_cum["horizon_day"].eq(10)]
    report = f"""# Phase6U Pairwise Ranker Daily Horizon Profile

Date: 2026-05-13

Scope: validation-era decomposition of the Phase6P sparse pairwise ranker. This
does not use the test lockbox.

## Question

The Phase6P model had strong training top-bottom h10 residual spread but only a
small 2019 validation spread. This run decomposes the same ranking into forward
holding day 1 through 10, so we can see whether the h10 result is persistent,
front-loaded, or offset by later-day giveback.

## Method

- Universe/scores: Phase6P score panel.
- Buckets: daily Top {TOP_BOTTOM_PERCENT}% minus Bottom {TOP_BOTTOM_PERCENT}% by score.
- Forward day 1: adjusted-open return from decision date +1 to +2.
- Forward day 10: adjusted-open return from decision date +10 to +11.
- Targets: raw open-to-open return, beta residual return, and daily strict
  size/style/SIC2 residual return.
- Strict daily residual controls: beta, true size proxy, liquidity, reversal,
  momentum, beta-residual momentum, volatility-adjusted momentum, and SIC2.

## Pairwise Score, Strict Residual Daily Spread

{_format_table(pairwise_strict, [
    "split",
    "horizon_day",
    "sessions",
    "mean_rank_ic",
    "mean_top_bottom_spread_bps",
    "spread_tstat",
    "spread_positive_rate",
])}

## Pairwise Score, Strict Residual Cumulative Spread

{_format_table(pairwise_strict_cum, [
    "split",
    "horizon_day",
    "cumulative_mean_spread_bps",
    "cumulative_mean_top_bps",
    "cumulative_mean_bottom_bps",
])}

## Day-10 Cumulative Check

{_format_table(final_day, [
    "split",
    "target_name",
    "horizon_day",
    "cumulative_mean_spread_bps",
    "cumulative_mean_top_bps",
    "cumulative_mean_bottom_bps",
])}

## Plots

"""
    for target_name, path in plot_paths.items():
        report += f"- {target_name}: `{path}`\n"

    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    _ensure_output_dir()

    base_panel = _load_score_and_feature_panel()
    stock_returns, benchmark_returns = _load_validation_returns(base_panel["symbol"].unique())

    metric_rows: list[dict[str, object]] = []
    for horizon_day in HOLDING_DAYS:
        frame = _attach_forward_day_returns(
            base_panel,
            stock_returns,
            benchmark_returns,
            horizon_day=horizon_day,
        )
        metric_rows.extend(_build_daily_metric_rows(frame))

    daily_metrics = pd.DataFrame(metric_rows)
    summary = _summarize_daily_metrics(daily_metrics)
    cumulative = _build_cumulative_summary(summary)

    daily_metrics_path = OUTPUT_DIR / "phase6u_daily_horizon_metrics.csv"
    summary_path = OUTPUT_DIR / "phase6u_horizon_summary.csv"
    cumulative_path = OUTPUT_DIR / "phase6u_horizon_cumulative_summary.csv"
    rollup_path = OUTPUT_DIR / "phase6u_run_summary.json"

    daily_metrics.to_csv(daily_metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    cumulative.to_csv(cumulative_path, index=False)
    plot_paths = _plot_pairwise_profiles(summary, cumulative)
    report_path = _write_markdown_report(summary, cumulative, plot_paths)

    pairwise_day10 = cumulative[
        cumulative["score_name"].eq("pairwise_rank_score")
        & cumulative["horizon_day"].eq(10)
    ][
        [
            "split",
            "target_name",
            "cumulative_mean_spread_bps",
            "cumulative_mean_top_bps",
            "cumulative_mean_bottom_bps",
        ]
    ].to_dict(orient="records")
    rollup = {
        "output_dir": str(OUTPUT_DIR),
        "input_score_panel": str(SCORE_PANEL_PATH),
        "input_label_feature_panel": str(LABEL_FEATURE_PANEL_PATH),
        "daily_metrics_path": str(daily_metrics_path),
        "summary_path": str(summary_path),
        "cumulative_path": str(cumulative_path),
        "report_path": str(report_path),
        "plot_paths": plot_paths,
        "pairwise_day10_cumulative": pairwise_day10,
        "top_bottom_fraction": TOP_BOTTOM_FRACTION,
        "top_bottom_percent": TOP_BOTTOM_PERCENT,
        "base_rows": int(len(base_panel)),
        "symbols": int(base_panel["symbol"].nunique()),
    }
    rollup_path.write_text(json.dumps(rollup, indent=2), encoding="utf-8")
    print(json.dumps(rollup, indent=2))


if __name__ == "__main__":
    main()
