"""Phase7F momentum formation/holding horizon grid.

Phase7F rebuilds forward labels directly from adjusted prices so the target
horizon is explicit. It tests when cross-sectional momentum pays across
formation lookbacks, skip windows, and holding horizons. The test lockbox is not
used.
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

from stockmachine.apps.run_pure_alpha_phase3 import (
    DEFAULT_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_DAILY_GLOBS,
    DEFAULT_DAILY_GLOBS,
    DEFAULT_PHASE1_MEMBERSHIP,
    DEFAULT_PHASE2_BETA_PANEL,
    _load_adjusted_prices,
)
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase7b import (
    DEFAULT_END,
    DEFAULT_EVAL_START,
    DEFAULT_QUANTILE,
    DEFAULT_START,
    DEFAULT_VARIANT,
    _markdown_table,
    _safe_corr,
    _t_stat,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7f_momentum_horizon_grid_20260517"
DEFAULT_FORMATION_LOOKBACKS = (20, 60, 120, 252)
DEFAULT_SKIP_WINDOWS = (0, 5, 20)
DEFAULT_HOLDING_HORIZONS = (5, 10, 20, 60)
DEFAULT_ROLLING_CORR_WINDOW = 252
DEFAULT_ROLLING_CORR_MIN_PERIODS = 120
REVERSAL_STYLE_PREFIX = "reversal_5d"
SHORT_CONTINUATION_STYLE_PREFIX = "short_continuation_5d"


@dataclass(frozen=True)
class StyleSpec:
    style: str
    family: str
    signal_column: str
    score_sign: float
    formation_lookback: int
    skip: int
    holding: int
    description: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7F momentum horizon grid.")
    parser.add_argument("--membership-path", default=str(DEFAULT_PHASE1_MEMBERSHIP))
    parser.add_argument("--beta-panel-path", default=str(DEFAULT_PHASE2_BETA_PANEL))
    parser.add_argument("--daily-glob", action="append", dest="daily_globs")
    parser.add_argument("--adj-factor-glob", action="append", dest="adj_factor_globs")
    parser.add_argument("--benchmark-daily-glob", action="append", dest="benchmark_daily_globs")
    parser.add_argument(
        "--benchmark-adj-factor-glob",
        action="append",
        dest="benchmark_adj_factor_globs",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--benchmark-symbol", default="SPY")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--eval-start", default=DEFAULT_EVAL_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--top-bottom-quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument(
        "--formation-lookbacks",
        default=",".join(str(value) for value in DEFAULT_FORMATION_LOOKBACKS),
    )
    parser.add_argument(
        "--skip-windows",
        default=",".join(str(value) for value in DEFAULT_SKIP_WINDOWS),
    )
    parser.add_argument(
        "--holding-horizons",
        default=",".join(str(value) for value in DEFAULT_HOLDING_HORIZONS),
    )
    parser.add_argument("--rolling-corr-window", type=int, default=DEFAULT_ROLLING_CORR_WINDOW)
    parser.add_argument(
        "--rolling-corr-min-periods",
        type=int,
        default=DEFAULT_ROLLING_CORR_MIN_PERIODS,
    )
    args = parser.parse_args(argv)

    rollup = build_phase7f_momentum_horizon_grid(
        membership_path=args.membership_path,
        beta_panel_path=args.beta_panel_path,
        daily_globs=tuple(args.daily_globs or DEFAULT_DAILY_GLOBS),
        adj_factor_globs=tuple(args.adj_factor_globs or DEFAULT_ADJ_FACTOR_GLOBS),
        benchmark_daily_globs=tuple(args.benchmark_daily_globs or DEFAULT_BENCHMARK_DAILY_GLOBS),
        benchmark_adj_factor_globs=tuple(
            args.benchmark_adj_factor_globs or DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS
        ),
        output_root=args.output_root,
        benchmark_symbol=args.benchmark_symbol,
        variant=args.variant,
        start=args.start,
        eval_start=args.eval_start,
        end=args.end,
        top_bottom_quantile=args.top_bottom_quantile,
        formation_lookbacks=_parse_int_grid(args.formation_lookbacks),
        skip_windows=_parse_int_grid(args.skip_windows),
        holding_horizons=_parse_int_grid(args.holding_horizons),
        rolling_corr_window=args.rolling_corr_window,
        rolling_corr_min_periods=args.rolling_corr_min_periods,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


def build_phase7f_momentum_horizon_grid(
    *,
    membership_path: str | Path = DEFAULT_PHASE1_MEMBERSHIP,
    beta_panel_path: str | Path = DEFAULT_PHASE2_BETA_PANEL,
    daily_globs: Sequence[str | Path] = DEFAULT_DAILY_GLOBS,
    adj_factor_globs: Sequence[str | Path] = DEFAULT_ADJ_FACTOR_GLOBS,
    benchmark_daily_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_DAILY_GLOBS,
    benchmark_adj_factor_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    benchmark_symbol: str = "SPY",
    variant: str = DEFAULT_VARIANT,
    start: str = DEFAULT_START,
    eval_start: str = DEFAULT_EVAL_START,
    end: str = DEFAULT_END,
    top_bottom_quantile: float = DEFAULT_QUANTILE,
    formation_lookbacks: Sequence[int] = DEFAULT_FORMATION_LOOKBACKS,
    skip_windows: Sequence[int] = DEFAULT_SKIP_WINDOWS,
    holding_horizons: Sequence[int] = DEFAULT_HOLDING_HORIZONS,
    rolling_corr_window: int = DEFAULT_ROLLING_CORR_WINDOW,
    rolling_corr_min_periods: int = DEFAULT_ROLLING_CORR_MIN_PERIODS,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    membership = _load_membership(membership_path, variant=variant, start=start, end=end)
    beta = _load_beta(beta_panel_path)
    symbols = tuple(sorted(membership["symbol"].astype(str).unique()))
    stock_prices = _load_adjusted_prices(
        daily_globs=daily_globs,
        adj_factor_globs=adj_factor_globs,
        symbols=symbols,
        end_date=end,
    )
    benchmark_prices = _load_adjusted_prices(
        daily_globs=benchmark_daily_globs,
        adj_factor_globs=benchmark_adj_factor_globs,
        symbols=(benchmark_symbol,),
        end_date=end,
    )
    panel = _build_feature_target_panel(
        membership=membership,
        beta=beta,
        stock_prices=stock_prices,
        benchmark_prices=benchmark_prices,
        benchmark_symbol=benchmark_symbol,
        formation_lookbacks=formation_lookbacks,
        skip_windows=skip_windows,
        holding_horizons=holding_horizons,
    )
    specs = _style_specs(
        formation_lookbacks=formation_lookbacks,
        skip_windows=skip_windows,
        holding_horizons=holding_horizons,
    )
    payoff = _style_payoffs(panel, specs=specs, top_bottom_quantile=top_bottom_quantile)
    eval_payoff = payoff[
        payoff["session_date"].ge(eval_start) & payoff["session_date"].le(end)
    ].copy()
    metrics = _style_metrics(eval_payoff)
    corr = _correlation_vs_reversal(eval_payoff)
    rolling_corr = _rolling_correlation_vs_reversal(
        eval_payoff,
        window=rolling_corr_window,
        min_periods=rolling_corr_min_periods,
    )
    rolling_summary = _rolling_corr_summary(rolling_corr)
    metrics = metrics.merge(corr, on=["holding", "style"], how="left").merge(
        rolling_summary,
        on=["holding", "style"],
        how="left",
    )
    horizon_summary = _horizon_summary(metrics)

    panel_path = output_dir / "phase7f_feature_target_panel_sample.csv"
    payoff_path = output_dir / "phase7f_style_payoff_panel.csv"
    metrics_path = output_dir / "phase7f_style_metrics.csv"
    horizon_summary_path = output_dir / "phase7f_horizon_summary.csv"
    corr_path = output_dir / "phase7f_payoff_corr_vs_reversal.csv"
    rolling_corr_path = output_dir / "phase7f_rolling_corr_vs_reversal.csv"
    memo_path = output_dir / "phase7f_momentum_horizon_grid_memo.md"
    rollup_path = output_dir / "phase7f_rollup.json"

    panel.head(5000).to_csv(panel_path, index=False)
    payoff.to_csv(payoff_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    horizon_summary.to_csv(horizon_summary_path, index=False)
    corr.to_csv(corr_path, index=False)
    rolling_corr.to_csv(rolling_corr_path, index=False)
    memo_path.write_text(
        _memo(
            variant=variant,
            start=start,
            eval_start=eval_start,
            end=end,
            top_bottom_quantile=top_bottom_quantile,
            formation_lookbacks=formation_lookbacks,
            skip_windows=skip_windows,
            holding_horizons=holding_horizons,
            panel=panel,
            payoff=payoff,
            metrics=metrics,
            horizon_summary=horizon_summary,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "phase": "phase7f_momentum_horizon_grid",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "variant": variant,
        "start": start,
        "eval_start": eval_start,
        "end": end,
        "top_bottom_quantile": float(top_bottom_quantile),
        "formation_lookbacks": [int(value) for value in formation_lookbacks],
        "skip_windows": [int(value) for value in skip_windows],
        "holding_horizons": [int(value) for value in holding_horizons],
        "rows": {
            "membership": int(len(membership)),
            "panel": int(len(panel)),
            "payoff": int(len(payoff)),
            "eval_payoff": int(len(eval_payoff)),
            "styles": int(len(specs)),
        },
        "outputs": {
            "feature_target_panel_sample": panel_path.as_posix(),
            "style_payoff_panel": payoff_path.as_posix(),
            "style_metrics": metrics_path.as_posix(),
            "horizon_summary": horizon_summary_path.as_posix(),
            "payoff_corr_vs_reversal": corr_path.as_posix(),
            "rolling_corr_vs_reversal": rolling_corr_path.as_posix(),
            "memo": memo_path.as_posix(),
        },
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_membership(
    path: str | Path,
    *,
    variant: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["session_date", "variant", "symbol", "liquidity_rank"])
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    frame["variant"] = frame["variant"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame[
        frame["variant"].eq(variant)
        & frame["session_date"].ge(start)
        & frame["session_date"].le(end)
    ].copy()
    return frame.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _load_beta(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["session_date", "symbol", "beta", "beta_available"])
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["beta"] = pd.to_numeric(frame["beta"], errors="coerce")
    return frame[["session_date", "symbol", "beta", "beta_available"]]


def _build_feature_target_panel(
    *,
    membership: pd.DataFrame,
    beta: pd.DataFrame,
    stock_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    benchmark_symbol: str,
    formation_lookbacks: Sequence[int],
    skip_windows: Sequence[int],
    holding_horizons: Sequence[int],
) -> pd.DataFrame:
    stock_features = _stock_feature_frame(
        stock_prices,
        formation_lookbacks=formation_lookbacks,
        skip_windows=skip_windows,
        holding_horizons=holding_horizons,
    )
    benchmark_features = _benchmark_feature_frame(
        benchmark_prices,
        benchmark_symbol=benchmark_symbol,
        holding_horizons=holding_horizons,
    )
    panel = (
        membership.merge(stock_features, on=["session_date", "symbol"], how="left")
        .merge(beta, on=["session_date", "symbol"], how="left")
        .merge(benchmark_features, on="session_date", how="left")
    )
    for holding in holding_horizons:
        panel[f"forward_beta_residual_return_h{holding}"] = (
            panel[f"forward_return_h{holding}"]
            - panel["beta"] * panel[f"benchmark_forward_return_h{holding}"]
        )
    return panel.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _stock_feature_frame(
    prices: pd.DataFrame,
    *,
    formation_lookbacks: Sequence[int],
    skip_windows: Sequence[int],
    holding_horizons: Sequence[int],
) -> pd.DataFrame:
    frame = prices.sort_values(["symbol", "session_date"]).copy()
    grouped = frame.groupby("symbol", group_keys=False)
    for lookback in sorted(set(formation_lookbacks).union({5})):
        frame[f"return_{lookback}d"] = grouped["adjusted_close"].pct_change(lookback)
    for lookback in formation_lookbacks:
        for skip in skip_windows:
            numerator = grouped["adjusted_close"].shift(skip)
            denominator = grouped["adjusted_close"].shift(skip + lookback)
            frame[f"momentum_l{lookback}_s{skip}"] = numerator / denominator - 1.0
    frame["next_adjusted_open"] = grouped["adjusted_open"].shift(-1)
    for holding in holding_horizons:
        exit_open = grouped["adjusted_open"].shift(-(holding + 1))
        frame[f"forward_return_h{holding}"] = exit_open / frame["next_adjusted_open"] - 1.0
    keep = ["session_date", "symbol", "return_5d"]
    keep.extend(f"momentum_l{lookback}_s{skip}" for lookback in formation_lookbacks for skip in skip_windows)
    keep.extend(f"forward_return_h{holding}" for holding in holding_horizons)
    return frame[keep]


def _benchmark_feature_frame(
    prices: pd.DataFrame,
    *,
    benchmark_symbol: str,
    holding_horizons: Sequence[int],
) -> pd.DataFrame:
    frame = prices[prices["symbol"].astype(str).eq(benchmark_symbol)].sort_values("session_date").copy()
    frame["benchmark_next_adjusted_open"] = frame["adjusted_open"].shift(-1)
    for holding in holding_horizons:
        exit_open = frame["adjusted_open"].shift(-(holding + 1))
        frame[f"benchmark_forward_return_h{holding}"] = (
            exit_open / frame["benchmark_next_adjusted_open"] - 1.0
        )
    keep = ["session_date"]
    keep.extend(f"benchmark_forward_return_h{holding}" for holding in holding_horizons)
    return frame[keep]


def _style_specs(
    *,
    formation_lookbacks: Sequence[int],
    skip_windows: Sequence[int],
    holding_horizons: Sequence[int],
) -> list[StyleSpec]:
    specs: list[StyleSpec] = []
    for holding in holding_horizons:
        specs.append(
            StyleSpec(
                style=f"{REVERSAL_STYLE_PREFIX}_h{holding}",
                family="short_horizon_reversal",
                signal_column="return_5d",
                score_sign=-1.0,
                formation_lookback=5,
                skip=0,
                holding=holding,
                description="Long recent 5-session losers, short recent winners.",
            )
        )
        specs.append(
            StyleSpec(
                style=f"{SHORT_CONTINUATION_STYLE_PREFIX}_h{holding}",
                family="short_horizon_continuation",
                signal_column="return_5d",
                score_sign=1.0,
                formation_lookback=5,
                skip=0,
                holding=holding,
                description="Long recent 5-session winners, short recent losers.",
            )
        )
        for lookback in formation_lookbacks:
            for skip in skip_windows:
                specs.append(
                    StyleSpec(
                        style=f"momentum_l{lookback}_s{skip}_h{holding}",
                        family="momentum",
                        signal_column=f"momentum_l{lookback}_s{skip}",
                        score_sign=1.0,
                        formation_lookback=lookback,
                        skip=skip,
                        holding=holding,
                        description=(
                            f"Long {lookback}-session winners ending {skip} sessions ago, "
                            f"short losers; forward holding h{holding}."
                        ),
                    )
                )
    return specs


def _style_payoffs(
    panel: pd.DataFrame,
    *,
    specs: Sequence[StyleSpec],
    top_bottom_quantile: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session_date, group in panel.groupby("session_date", sort=True):
        for spec in specs:
            target_column = f"forward_beta_residual_return_h{spec.holding}"
            subset = group[[spec.signal_column, target_column]].copy()
            subset["score"] = spec.score_sign * pd.to_numeric(
                subset[spec.signal_column],
                errors="coerce",
            )
            subset[target_column] = pd.to_numeric(subset[target_column], errors="coerce")
            subset = subset.dropna(subset=["score", target_column])
            if len(subset) < 100:
                continue
            low = subset["score"].quantile(top_bottom_quantile)
            high = subset["score"].quantile(1.0 - top_bottom_quantile)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                continue
            long = subset[subset["score"] >= high]
            short = subset[subset["score"] <= low]
            if long.empty or short.empty:
                continue
            long_target = long[target_column].astype(float)
            short_target = short[target_column].astype(float)
            payoff = float(long_target.mean() - short_target.mean())
            rows.append(
                {
                    "session_date": str(session_date),
                    "style": spec.style,
                    "family": spec.family,
                    "formation_lookback": int(spec.formation_lookback),
                    "skip": int(spec.skip),
                    "holding": int(spec.holding),
                    "signal_column": spec.signal_column,
                    "score_sign": float(spec.score_sign),
                    "names": int(len(subset)),
                    "long_names": int(len(long)),
                    "short_names": int(len(short)),
                    "long_signal_mean": float(long[spec.signal_column].mean()),
                    "short_signal_mean": float(short[spec.signal_column].mean()),
                    "long_target_mean": float(long_target.mean()),
                    "short_target_mean": float(short_target.mean()),
                    "payoff": payoff,
                    "payoff_bps": payoff * 10000.0,
                    "description": spec.description,
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["session_date", "holding", "style"]).reset_index(drop=True)


def _style_metrics(payoff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (holding, style), group in payoff.groupby(["holding", "style"], sort=True):
        series = pd.to_numeric(group["payoff"], errors="coerce").dropna()
        first = group.iloc[0]
        rows.append(
            {
                "style": style,
                "family": first["family"],
                "formation_lookback": int(first["formation_lookback"]),
                "skip": int(first["skip"]),
                "holding": int(holding),
                "sessions": int(len(series)),
                "mean_payoff_bps": float(series.mean() * 10000.0),
                "median_payoff_bps": float(series.median() * 10000.0),
                "hit_rate": float((series > 0).mean()),
                "t_stat": _t_stat(series),
                "newey_west_lag": int(max(0, holding - 1)),
                "newey_west_t_stat": _newey_west_t_stat(series, lag=max(0, holding - 1)),
                "description": first["description"],
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["holding", "mean_payoff_bps"], ascending=[True, False])


def _correlation_vs_reversal(payoff: pd.DataFrame) -> pd.DataFrame:
    pivot = _payoff_pivot(payoff)
    rows = []
    for holding in sorted(payoff["holding"].dropna().astype(int).unique()):
        reversal_style = f"{REVERSAL_STYLE_PREFIX}_h{holding}"
        if reversal_style not in pivot.columns:
            continue
        reversal = pivot[reversal_style]
        for style in sorted(payoff.loc[payoff["holding"].eq(holding), "style"].unique()):
            if style not in pivot.columns:
                continue
            subset = pd.concat([pivot[style], reversal], axis=1).dropna()
            subset.columns = ["style_payoff_bps", "reversal_payoff_bps"]
            rows.append(
                {
                    "holding": int(holding),
                    "style": style,
                    "corr_sessions": int(len(subset)),
                    "payoff_pearson_corr_vs_reversal": _safe_corr(
                        subset["style_payoff_bps"],
                        subset["reversal_payoff_bps"],
                        method="pearson",
                    ),
                    "payoff_spearman_corr_vs_reversal": _safe_corr(
                        subset["style_payoff_bps"],
                        subset["reversal_payoff_bps"],
                        method="spearman",
                    ),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows)


def _rolling_correlation_vs_reversal(
    payoff: pd.DataFrame,
    *,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    pivot = _payoff_pivot(payoff)
    rows = []
    for holding in sorted(payoff["holding"].dropna().astype(int).unique()):
        reversal_style = f"{REVERSAL_STYLE_PREFIX}_h{holding}"
        if reversal_style not in pivot.columns:
            continue
        reversal = pivot[reversal_style]
        for style in sorted(payoff.loc[payoff["holding"].eq(holding), "style"].unique()):
            if style == reversal_style or style not in pivot.columns:
                continue
            rolling = pivot[style].rolling(window, min_periods=min_periods).corr(reversal)
            for session_date, value in rolling.dropna().items():
                rows.append(
                    {
                        "session_date": str(session_date.date()),
                        "holding": int(holding),
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
        return pd.DataFrame(columns=["holding", "style"])
    rows = []
    for (holding, style), group in rolling.groupby(["holding", "style"], sort=True):
        value = pd.to_numeric(group["rolling_corr_vs_reversal"], errors="coerce").dropna()
        rows.append(
            {
                "holding": int(holding),
                "style": style,
                "mean_rolling_corr_vs_reversal": float(value.mean()),
                "median_rolling_corr_vs_reversal": float(value.median()),
                "p10_rolling_corr_vs_reversal": float(value.quantile(0.10)),
                "p90_rolling_corr_vs_reversal": float(value.quantile(0.90)),
                "rolling_corr_sessions": int(len(value)),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows)


def _horizon_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    momentum = metrics[metrics["family"].eq("momentum")].copy()
    rows = []
    for holding, group in momentum.groupby("holding", sort=True):
        best_mean = group.sort_values("mean_payoff_bps", ascending=False).iloc[0]
        best_t = group.sort_values("t_stat", ascending=False).iloc[0]
        best_nw = group.sort_values("newey_west_t_stat", ascending=False).iloc[0]
        rows.append(
            {
                "holding": int(holding),
                "best_mean_style": best_mean["style"],
                "best_mean_payoff_bps": float(best_mean["mean_payoff_bps"]),
                "best_mean_t_stat": float(best_mean["t_stat"]),
                "best_mean_newey_west_t_stat": float(best_mean["newey_west_t_stat"]),
                "best_t_style": best_t["style"],
                "best_t_payoff_bps": float(best_t["mean_payoff_bps"]),
                "best_t_stat": float(best_t["t_stat"]),
                "best_t_newey_west_t_stat": float(best_t["newey_west_t_stat"]),
                "best_newey_west_style": best_nw["style"],
                "best_newey_west_payoff_bps": float(best_nw["mean_payoff_bps"]),
                "best_newey_west_t_stat": float(best_nw["newey_west_t_stat"]),
                "positive_momentum_styles": int((group["mean_payoff_bps"] > 0).sum()),
                "tested_momentum_styles": int(len(group)),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows)


def _payoff_pivot(payoff: pd.DataFrame) -> pd.DataFrame:
    frame = payoff.copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    return frame.pivot_table(
        index="session_date",
        columns="style",
        values="payoff_bps",
        aggfunc="mean",
    ).sort_index()


def _memo(
    *,
    variant: str,
    start: str,
    eval_start: str,
    end: str,
    top_bottom_quantile: float,
    formation_lookbacks: Sequence[int],
    skip_windows: Sequence[int],
    holding_horizons: Sequence[int],
    panel: pd.DataFrame,
    payoff: pd.DataFrame,
    metrics: pd.DataFrame,
    horizon_summary: pd.DataFrame,
) -> str:
    best = metrics[metrics["family"].eq("momentum")].sort_values(
        "t_stat",
        ascending=False,
    ).head(20)
    baseline = metrics[metrics["family"].isin(["short_horizon_reversal", "short_horizon_continuation"])]
    lines = [
        "# Phase7F Momentum Horizon Grid Results",
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
        f"- formation lookbacks: `{list(formation_lookbacks)}` sessions",
        f"- skip windows: `{list(skip_windows)}` sessions",
        f"- holding horizons: `{list(holding_horizons)}` sessions",
        f"- panel rows: `{len(panel)}`",
        f"- payoff rows: `{len(payoff)}`",
        "",
        "## Horizon Summary",
        "",
        _markdown_table(horizon_summary),
        "",
        "## Top Momentum Grid Rows",
        "",
        _markdown_table(best),
        "",
        "## Reversal / Short-Term Continuation Baselines",
        "",
        _markdown_table(baseline),
        "",
    ]
    return "\n".join(lines)


def _newey_west_t_stat(series: pd.Series, *, lag: int) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 2:
        return np.nan
    mean = float(values.mean())
    centered = values - mean
    lag = min(max(int(lag), 0), n - 1)
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    for step in range(1, lag + 1):
        weight = 1.0 - step / (lag + 1.0)
        gamma = float(np.dot(centered[step:], centered[:-step]) / n)
        long_run_variance += 2.0 * weight * gamma
    if not np.isfinite(long_run_variance) or long_run_variance <= 0.0:
        return np.nan
    standard_error = float(np.sqrt(long_run_variance / n))
    if standard_error <= 0.0:
        return np.nan
    return mean / standard_error


def _parse_int_grid(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("grid cannot be empty")
    if any(part < 0 for part in parsed):
        raise ValueError("grid values must be non-negative")
    return parsed


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
