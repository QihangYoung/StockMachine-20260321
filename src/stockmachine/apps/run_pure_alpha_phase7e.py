"""Phase7E momentum/trend sleeve payoff diagnostics.

Phase7E separates short-term continuation from true intermediate momentum.
It computes validation-only long-short payoff sleeves and correlations against
the current reversal sleeve. The test lockbox is not used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT, TARGET_COLUMN
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_CIK_MAPPING,
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_SEC_SUBMISSIONS_DIR,
)
from stockmachine.apps.run_pure_alpha_phase7b import (
    DEFAULT_END,
    DEFAULT_EVAL_START,
    DEFAULT_FUNDAMENTAL_PANEL,
    DEFAULT_NON_PRICE_PANEL,
    DEFAULT_QUANTILE,
    DEFAULT_SIZE_PANEL,
    DEFAULT_START,
    DEFAULT_VARIANT,
    _load_joined_panel,
    _markdown_table,
    _safe_corr,
    _t_stat,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7e_momentum_trend_sleeves_20260517"
DEFAULT_ROLLING_CORR_WINDOW = 252
DEFAULT_ROLLING_CORR_MIN_PERIODS = 120
REVERSAL_STYLE = "reversal_5d_loser_minus_winner"


@dataclass(frozen=True)
class StyleSpec:
    style: str
    source_column: str
    direction: float
    family: str
    description: str


STYLE_SPECS = (
    StyleSpec(
        REVERSAL_STYLE,
        "reversal_5d",
        1.0,
        "short_horizon_contrarian",
        "Long recent 5-session losers, short recent winners.",
    ),
    StyleSpec(
        "short_term_continuation_5d",
        "return_5d",
        1.0,
        "short_horizon_continuation",
        "Long recent 5-session winners, short recent losers; exact mirror of reversal_5d.",
    ),
    StyleSpec(
        "momentum_20d_winner_minus_loser",
        "momentum_20d",
        1.0,
        "intermediate_momentum",
        "Long 20-session winners, short 20-session losers.",
    ),
    StyleSpec(
        "momentum_60d_winner_minus_loser",
        "momentum_60d",
        1.0,
        "intermediate_momentum",
        "Long 60-session winners, short 60-session losers.",
    ),
    StyleSpec(
        "momentum_20d_ex_recent5",
        "momentum_20d_ex_recent5",
        1.0,
        "intermediate_momentum_ex_recent",
        "Long 20-session ex-recent-5 winners, short ex-recent losers.",
    ),
    StyleSpec(
        "momentum_60d_ex_recent5",
        "momentum_60d_ex_recent5",
        1.0,
        "intermediate_momentum_ex_recent",
        "Long 60-session ex-recent-5 winners, short ex-recent losers.",
    ),
    StyleSpec(
        "beta_residual_momentum_20d",
        "beta_residual_momentum_20d",
        1.0,
        "residual_momentum",
        "Long beta-residual 20-session winners, short residual losers.",
    ),
    StyleSpec(
        "vol_adjusted_momentum_20d",
        "vol_adjusted_momentum_20d",
        1.0,
        "risk_adjusted_momentum",
        "Long volatility-adjusted 20-session winners, short losers.",
    ),
)

REFERENCE_SCORE_PAIRS = (
    ("reversal_5d", "return_5d"),
    ("reversal_5d", "momentum_20d"),
    ("reversal_5d", "momentum_60d"),
    ("reversal_5d", "momentum_20d_ex_recent5"),
    ("reversal_5d", "momentum_60d_ex_recent5"),
    ("reversal_5d", "beta_residual_momentum_20d"),
    ("reversal_5d", "vol_adjusted_momentum_20d"),
    ("momentum_20d", "momentum_60d"),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase7E momentum/trend sleeve payoff diagnostics."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--size-panel-path", default=str(DEFAULT_SIZE_PANEL))
    parser.add_argument("--non-price-panel-path", default=str(DEFAULT_NON_PRICE_PANEL))
    parser.add_argument("--fundamental-panel-path", default=str(DEFAULT_FUNDAMENTAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--eval-start", default=DEFAULT_EVAL_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--top-bottom-quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--rolling-corr-window", type=int, default=DEFAULT_ROLLING_CORR_WINDOW)
    parser.add_argument(
        "--rolling-corr-min-periods",
        type=int,
        default=DEFAULT_ROLLING_CORR_MIN_PERIODS,
    )
    args = parser.parse_args(argv)

    rollup = build_phase7e_momentum_trend_sleeves(
        signal_panel_path=args.signal_panel_path,
        size_panel_path=args.size_panel_path,
        non_price_panel_path=args.non_price_panel_path,
        fundamental_panel_path=args.fundamental_panel_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        output_root=args.output_root,
        variant=args.variant,
        start=args.start,
        eval_start=args.eval_start,
        end=args.end,
        top_bottom_quantile=args.top_bottom_quantile,
        rolling_corr_window=args.rolling_corr_window,
        rolling_corr_min_periods=args.rolling_corr_min_periods,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


def build_phase7e_momentum_trend_sleeves(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    size_panel_path: str | Path = DEFAULT_SIZE_PANEL,
    non_price_panel_path: str | Path = DEFAULT_NON_PRICE_PANEL,
    fundamental_panel_path: str | Path = DEFAULT_FUNDAMENTAL_PANEL,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant: str = DEFAULT_VARIANT,
    start: str = DEFAULT_START,
    eval_start: str = DEFAULT_EVAL_START,
    end: str = DEFAULT_END,
    top_bottom_quantile: float = DEFAULT_QUANTILE,
    rolling_corr_window: int = DEFAULT_ROLLING_CORR_WINDOW,
    rolling_corr_min_periods: int = DEFAULT_ROLLING_CORR_MIN_PERIODS,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_joined_panel(
        signal_panel_path=signal_panel_path,
        size_panel_path=size_panel_path,
        non_price_panel_path=non_price_panel_path,
        fundamental_panel_path=fundamental_panel_path,
        cik_mapping_path=cik_mapping_path,
        sec_submissions_dir=sec_submissions_dir,
        variant=variant,
        start=start,
        end=end,
    )
    panel = _add_style_features(panel)
    payoff = _build_style_payoff_panel(
        panel,
        top_bottom_quantile=top_bottom_quantile,
    )
    eval_payoff = payoff[
        payoff["session_date"].ge(eval_start) & payoff["session_date"].le(end)
    ].copy()
    metrics = _style_metrics(eval_payoff)
    payoff_corr = _payoff_correlations(eval_payoff)
    rolling_corr = _rolling_reversal_correlations(
        eval_payoff,
        window=rolling_corr_window,
        min_periods=rolling_corr_min_periods,
    )
    rolling_summary = _rolling_corr_summary(rolling_corr)
    score_corr = _score_correlation_summary(
        panel[
            panel["session_date"].ge(eval_start) & panel["session_date"].le(end)
        ].copy()
    )
    orthogonality = _orthogonality_summary(metrics, payoff_corr, score_corr, rolling_summary)
    combo_metrics = _combo_metrics(eval_payoff)

    payoff_path = output_dir / "phase7e_style_payoff_panel.csv"
    metrics_path = output_dir / "phase7e_style_metrics.csv"
    payoff_corr_path = output_dir / "phase7e_payoff_correlations.csv"
    rolling_corr_path = output_dir / "phase7e_rolling_reversal_correlations.csv"
    rolling_summary_path = output_dir / "phase7e_rolling_reversal_correlation_summary.csv"
    score_corr_path = output_dir / "phase7e_score_correlation_summary.csv"
    orthogonality_path = output_dir / "phase7e_orthogonality_summary.csv"
    combo_metrics_path = output_dir / "phase7e_combo_metrics.csv"
    memo_path = output_dir / "phase7e_momentum_trend_sleeves_memo.md"
    rollup_path = output_dir / "phase7e_rollup.json"

    payoff.to_csv(payoff_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    payoff_corr.to_csv(payoff_corr_path, index=False)
    rolling_corr.to_csv(rolling_corr_path, index=False)
    rolling_summary.to_csv(rolling_summary_path, index=False)
    score_corr.to_csv(score_corr_path, index=False)
    orthogonality.to_csv(orthogonality_path, index=False)
    combo_metrics.to_csv(combo_metrics_path, index=False)
    memo_path.write_text(
        _memo(
            variant=variant,
            start=start,
            eval_start=eval_start,
            end=end,
            top_bottom_quantile=top_bottom_quantile,
            rolling_corr_window=rolling_corr_window,
            rolling_corr_min_periods=rolling_corr_min_periods,
            panel=panel,
            payoff=payoff,
            metrics=metrics,
            payoff_corr=payoff_corr,
            rolling_summary=rolling_summary,
            score_corr=score_corr,
            orthogonality=orthogonality,
            combo_metrics=combo_metrics,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "phase": "phase7e_momentum_trend_sleeves",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "variant": variant,
        "start": start,
        "eval_start": eval_start,
        "end": end,
        "top_bottom_quantile": float(top_bottom_quantile),
        "rolling_corr_window": int(rolling_corr_window),
        "rolling_corr_min_periods": int(rolling_corr_min_periods),
        "rows": {
            "panel": int(len(panel)),
            "payoff": int(len(payoff)),
            "eval_payoff": int(len(eval_payoff)),
            "styles": int(len(STYLE_SPECS)),
        },
        "outputs": {
            "style_payoff_panel": payoff_path.as_posix(),
            "style_metrics": metrics_path.as_posix(),
            "payoff_correlations": payoff_corr_path.as_posix(),
            "rolling_reversal_correlations": rolling_corr_path.as_posix(),
            "rolling_reversal_correlation_summary": rolling_summary_path.as_posix(),
            "score_correlation_summary": score_corr_path.as_posix(),
            "orthogonality_summary": orthogonality_path.as_posix(),
            "combo_metrics": combo_metrics_path.as_posix(),
            "memo": memo_path.as_posix(),
        },
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _add_style_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    numeric_columns = {
        "reversal_5d",
        "return_5d",
        "momentum_20d",
        "momentum_60d",
        "beta_residual_momentum_20d",
        "vol_adjusted_momentum_20d",
        TARGET_COLUMN,
    }
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["return_5d"] = -frame["reversal_5d"]
    frame["momentum_20d_ex_recent5"] = _compound_ex_recent(
        frame["momentum_20d"],
        frame["return_5d"],
    )
    frame["momentum_60d_ex_recent5"] = _compound_ex_recent(
        frame["momentum_60d"],
        frame["return_5d"],
    )
    return frame


def _build_style_payoff_panel(
    panel: pd.DataFrame,
    *,
    top_bottom_quantile: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = panel.dropna(subset=[TARGET_COLUMN]).copy()
    for session_date, group in frame.groupby("session_date", sort=True):
        for spec in STYLE_SPECS:
            feature = spec.direction * pd.to_numeric(group[spec.source_column], errors="coerce")
            valid = group.loc[feature.notna()].copy()
            valid_feature = feature.loc[valid.index]
            if len(valid) < 100:
                continue
            low = valid_feature.quantile(top_bottom_quantile)
            high = valid_feature.quantile(1.0 - top_bottom_quantile)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                continue
            long_frame = valid.loc[valid_feature >= high]
            short_frame = valid.loc[valid_feature <= low]
            if long_frame.empty or short_frame.empty:
                continue
            long_target = pd.to_numeric(long_frame[TARGET_COLUMN], errors="coerce")
            short_target = pd.to_numeric(short_frame[TARGET_COLUMN], errors="coerce")
            payoff = float(long_target.mean() - short_target.mean())
            rows.append(
                {
                    "session_date": str(session_date),
                    "style": spec.style,
                    "source_column": spec.source_column,
                    "family": spec.family,
                    "description": spec.description,
                    "names": int(len(valid)),
                    "long_names": int(len(long_frame)),
                    "short_names": int(len(short_frame)),
                    "long_source_mean": float(
                        pd.to_numeric(long_frame[spec.source_column], errors="coerce").mean()
                    ),
                    "short_source_mean": float(
                        pd.to_numeric(short_frame[spec.source_column], errors="coerce").mean()
                    ),
                    "long_target_mean": float(long_target.mean()),
                    "short_target_mean": float(short_target.mean()),
                    "payoff": payoff,
                    "payoff_bps": payoff * 10000.0,
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["session_date", "style"]).reset_index(drop=True)


def _style_metrics(payoff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for style, group in payoff.groupby("style", sort=True):
        spec = _style_spec(style)
        series = pd.to_numeric(group["payoff"], errors="coerce").dropna()
        rows.append(
            {
                "style": style,
                "family": spec.family,
                "sessions": int(len(series)),
                "mean_payoff_bps": float(series.mean() * 10000.0),
                "median_payoff_bps": float(series.median() * 10000.0),
                "hit_rate": float((series > 0).mean()),
                "t_stat": _t_stat(series),
                "description": spec.description,
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_payoff_bps", ascending=False)


def _payoff_correlations(payoff: pd.DataFrame) -> pd.DataFrame:
    pivot = _payoff_pivot(payoff)
    rows = []
    for left in pivot.columns:
        for right in pivot.columns:
            if left >= right:
                continue
            subset = pivot[[left, right]].dropna()
            rows.append(
                {
                    "left_style": left,
                    "right_style": right,
                    "sessions": int(len(subset)),
                    "pearson_corr": _safe_corr(subset[left], subset[right], method="pearson"),
                    "spearman_corr": _safe_corr(subset[left], subset[right], method="spearman"),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows)


def _rolling_reversal_correlations(
    payoff: pd.DataFrame,
    *,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    pivot = _payoff_pivot(payoff)
    if REVERSAL_STYLE not in pivot.columns:
        return pd.DataFrame()
    rows = []
    reversal = pivot[REVERSAL_STYLE]
    for style in pivot.columns:
        if style == REVERSAL_STYLE:
            continue
        rolling = pivot[style].rolling(window, min_periods=min_periods).corr(reversal)
        for session_date, value in rolling.dropna().items():
            rows.append(
                {
                    "session_date": str(session_date.date()),
                    "style": style,
                    "rolling_corr_vs_reversal": float(value),
                    "window": int(window),
                    "min_periods": int(min_periods),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows)


def _rolling_corr_summary(rolling: pd.DataFrame) -> pd.DataFrame:
    if rolling.empty:
        return pd.DataFrame()
    rows = []
    for style, group in rolling.groupby("style", sort=True):
        value = pd.to_numeric(group["rolling_corr_vs_reversal"], errors="coerce").dropna()
        rows.append(
            {
                "style": style,
                "sessions": int(len(value)),
                "mean_rolling_corr_vs_reversal": float(value.mean()),
                "median_rolling_corr_vs_reversal": float(value.median()),
                "p10_rolling_corr_vs_reversal": float(value.quantile(0.10)),
                "p90_rolling_corr_vs_reversal": float(value.quantile(0.90)),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_rolling_corr_vs_reversal")


def _score_correlation_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left, right in REFERENCE_SCORE_PAIRS:
        daily_values = []
        for _, group in panel.groupby("session_date", sort=True):
            subset = group[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(subset) < 100:
                continue
            value = subset[left].corr(subset[right], method="spearman")
            if np.isfinite(value):
                daily_values.append(float(value))
        values = pd.Series(daily_values, dtype=float)
        rows.append(
            {
                "left_score": left,
                "right_score": right,
                "sessions": int(len(values)),
                "mean_daily_spearman": float(values.mean()) if len(values) else np.nan,
                "median_daily_spearman": float(values.median()) if len(values) else np.nan,
                "p10_daily_spearman": float(values.quantile(0.10)) if len(values) else np.nan,
                "p90_daily_spearman": float(values.quantile(0.90)) if len(values) else np.nan,
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_daily_spearman")


def _orthogonality_summary(
    metrics: pd.DataFrame,
    payoff_corr: pd.DataFrame,
    score_corr: pd.DataFrame,
    rolling_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for spec in STYLE_SPECS:
        style = spec.style
        metric = metrics[metrics["style"].eq(style)]
        corr_row = _style_pair_row(payoff_corr, style, REVERSAL_STYLE)
        score_row = score_corr[
            score_corr["left_score"].eq("reversal_5d")
            & score_corr["right_score"].eq(spec.source_column)
        ]
        rolling_row = rolling_summary[rolling_summary["style"].eq(style)]
        rows.append(
            {
                "style": style,
                "family": spec.family,
                "mean_payoff_bps": _first_float(metric, "mean_payoff_bps"),
                "hit_rate": _first_float(metric, "hit_rate"),
                "payoff_pearson_corr_vs_reversal": _first_float(corr_row, "pearson_corr"),
                "payoff_spearman_corr_vs_reversal": _first_float(corr_row, "spearman_corr"),
                "mean_score_spearman_vs_reversal_score": _first_float(
                    score_row,
                    "mean_daily_spearman",
                ),
                "mean_rolling_payoff_corr_vs_reversal": _first_float(
                    rolling_row,
                    "mean_rolling_corr_vs_reversal",
                ),
                "description": spec.description,
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values("payoff_pearson_corr_vs_reversal")


def _combo_metrics(payoff: pd.DataFrame) -> pd.DataFrame:
    pivot = _payoff_pivot(payoff)
    combos: dict[str, dict[str, float]] = {
        "reversal_only": {REVERSAL_STYLE: 1.0},
        "rev75_mom60_ex_recent5_25": {
            REVERSAL_STYLE: 0.75,
            "momentum_60d_ex_recent5": 0.25,
        },
        "rev50_mom60_ex_recent5_50": {
            REVERSAL_STYLE: 0.50,
            "momentum_60d_ex_recent5": 0.50,
        },
        "rev50_mom60_raw_50": {
            REVERSAL_STYLE: 0.50,
            "momentum_60d_winner_minus_loser": 0.50,
        },
        "rev75_mom20_ex_recent5_25": {
            REVERSAL_STYLE: 0.75,
            "momentum_20d_ex_recent5": 0.25,
        },
    }
    rows = []
    reversal = pivot[REVERSAL_STYLE]
    for combo, weights in combos.items():
        required = [style for style in weights if style in pivot.columns]
        if len(required) != len(weights):
            continue
        series = sum(pivot[style] * weight for style, weight in weights.items())
        series = series.dropna()
        rows.append(
            {
                "combo": combo,
                "weights": json.dumps(weights, sort_keys=True),
                "sessions": int(len(series)),
                "mean_payoff_bps": float(series.mean()),
                "median_payoff_bps": float(series.median()),
                "hit_rate": float((series > 0).mean()),
                "std_payoff_bps": float(series.std(ddof=1)),
                "t_stat": _t_stat(series / 10000.0),
                "pearson_corr_vs_reversal": _safe_corr(
                    series,
                    reversal.reindex(series.index),
                    method="pearson",
                ),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values("t_stat", ascending=False)


def _payoff_pivot(payoff: pd.DataFrame) -> pd.DataFrame:
    frame = payoff.copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    return frame.pivot_table(
        index="session_date",
        columns="style",
        values="payoff_bps",
        aggfunc="mean",
    ).sort_index()


def _compound_ex_recent(total_return: pd.Series, recent_return: pd.Series) -> pd.Series:
    total = pd.to_numeric(total_return, errors="coerce")
    recent = pd.to_numeric(recent_return, errors="coerce")
    denominator = 1.0 + recent
    value = (1.0 + total) / denominator.where(denominator > 0.0) - 1.0
    return value.replace([np.inf, -np.inf], np.nan)


def _style_spec(style: str) -> StyleSpec:
    for spec in STYLE_SPECS:
        if spec.style == style:
            return spec
    raise KeyError(style)


def _style_pair_row(frame: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    if left == right:
        return pd.DataFrame(
            [
                {
                    "pearson_corr": 1.0,
                    "spearman_corr": 1.0,
                }
            ]
        )
    mask = (
        frame["left_style"].eq(left) & frame["right_style"].eq(right)
    ) | (
        frame["left_style"].eq(right) & frame["right_style"].eq(left)
    )
    return frame[mask]


def _first_float(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    value = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(value.iloc[0]) if len(value) else np.nan


def _memo(
    *,
    variant: str,
    start: str,
    eval_start: str,
    end: str,
    top_bottom_quantile: float,
    rolling_corr_window: int,
    rolling_corr_min_periods: int,
    panel: pd.DataFrame,
    payoff: pd.DataFrame,
    metrics: pd.DataFrame,
    payoff_corr: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    score_corr: pd.DataFrame,
    orthogonality: pd.DataFrame,
    combo_metrics: pd.DataFrame,
) -> str:
    reversal_vs_return = score_corr[
        score_corr["left_score"].eq("reversal_5d")
        & score_corr["right_score"].eq("return_5d")
    ]
    lines = [
        "# Phase7E Momentum/Trend Sleeve Diagnostics",
        "",
        f"Generated: {_utc_now()}",
        "",
        "Validation-only. The test lockbox is not used.",
        "",
        "## Setup",
        "",
        f"- universe variant: `{variant}`",
        f"- data window: `{start}` through `{end}`",
        f"- evaluation window: `{eval_start}` through `{end}`",
        f"- top/bottom quantile: `{top_bottom_quantile}`",
        f"- rolling payoff correlation window: `{rolling_corr_window}` sessions",
        f"- rolling payoff correlation min periods: `{rolling_corr_min_periods}` sessions",
        f"- joined panel rows: `{len(panel)}`",
        f"- payoff rows: `{len(payoff)}`",
        "",
        "## Key Point",
        "",
        "`short_term_continuation_5d` is the exact mirror of `reversal_5d` because",
        "`return_5d = -reversal_5d`. Intermediate momentum uses different horizons.",
        "",
        "Daily score correlation for `reversal_5d` vs `return_5d`:",
        "",
        _markdown_table(reversal_vs_return),
        "",
        "## Style Metrics",
        "",
        _markdown_table(metrics),
        "",
        "## Orthogonality Summary",
        "",
        _markdown_table(orthogonality),
        "",
        "## Static Combo Diagnostics",
        "",
        "These combinations are validation diagnostics only, not optimized live allocations.",
        "",
        _markdown_table(combo_metrics),
        "",
        "## Payoff Correlations",
        "",
        _markdown_table(payoff_corr),
        "",
        "## Rolling Correlation Summary",
        "",
        _markdown_table(rolling_summary),
        "",
        "## Score Correlation Summary",
        "",
        _markdown_table(score_corr),
        "",
    ]
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
