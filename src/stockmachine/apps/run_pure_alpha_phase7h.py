"""Phase7H sector-neutral non-momentum factor diagnostics.

Phase7H starts from the Phase7G factor panel and asks whether the best
non-momentum sleeves survive sector neutralization. The test lockbox is not
used.
"""

from __future__ import annotations

import argparse
import json
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
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_CIK_MAPPING,
    DEFAULT_SEC_SUBMISSIONS_DIR,
    _load_sec_sic_map,
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
    _markdown_table,
    _safe_corr,
    _t_stat,
)
from stockmachine.apps.run_pure_alpha_phase7f import _load_beta, _load_membership
from stockmachine.apps.run_pure_alpha_phase7g import (
    DEFAULT_HOLDING_HORIZONS,
    FUNDAMENTAL_COLUMNS,
    NON_PRICE_COLUMNS,
    SIZE_COLUMNS,
    _build_feature_target_panel,
    _factor_specs,
    _load_optional_panel,
    _load_size_panel,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7h_sector_neutral_factor_grid_20260517"
DEFAULT_MIN_SECTOR_NAMES = 30
DEFAULT_ROLLING_CORR_WINDOW = 252
DEFAULT_ROLLING_CORR_MIN_PERIODS = 120
NEUTRALIZATION_METHODS = ("raw", "sector_rank", "sector_balanced")
DEFAULT_STYLE_PREFIXES = (
    "reversal_5d",
    "size_small_cap",
    "quality_high_cash_to_assets",
    "defensive_low_beta",
    "defensive_low_vol_60d",
    "defensive_low_vol_120d",
    "value_book_to_market",
    "value_cash_flow_yield",
    "value_earnings_yield",
    "quality_low_fragility",
    "quality_low_leverage_pressure",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7H sector-neutral factor grid.")
    parser.add_argument("--membership-path", default=str(DEFAULT_PHASE1_MEMBERSHIP))
    parser.add_argument("--beta-panel-path", default=str(DEFAULT_PHASE2_BETA_PANEL))
    parser.add_argument("--size-panel-path", default=str(DEFAULT_SIZE_PANEL))
    parser.add_argument("--non-price-panel-path", default=str(DEFAULT_NON_PRICE_PANEL))
    parser.add_argument("--fundamental-panel-path", default=str(DEFAULT_FUNDAMENTAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
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
        "--holding-horizons",
        default=",".join(str(value) for value in DEFAULT_HOLDING_HORIZONS),
    )
    parser.add_argument(
        "--style-prefixes",
        default=",".join(DEFAULT_STYLE_PREFIXES),
        help="Comma-separated style prefixes to test. Use 'all' to run the full Phase7G set.",
    )
    parser.add_argument("--min-sector-names", type=int, default=DEFAULT_MIN_SECTOR_NAMES)
    parser.add_argument("--rolling-corr-window", type=int, default=DEFAULT_ROLLING_CORR_WINDOW)
    parser.add_argument(
        "--rolling-corr-min-periods",
        type=int,
        default=DEFAULT_ROLLING_CORR_MIN_PERIODS,
    )
    args = parser.parse_args(argv)

    rollup = build_phase7h_sector_neutral_factor_grid(
        membership_path=args.membership_path,
        beta_panel_path=args.beta_panel_path,
        size_panel_path=args.size_panel_path,
        non_price_panel_path=args.non_price_panel_path,
        fundamental_panel_path=args.fundamental_panel_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
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
        holding_horizons=_parse_int_grid(args.holding_horizons),
        style_prefixes=_parse_string_grid(args.style_prefixes),
        min_sector_names=args.min_sector_names,
        rolling_corr_window=args.rolling_corr_window,
        rolling_corr_min_periods=args.rolling_corr_min_periods,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


def build_phase7h_sector_neutral_factor_grid(
    *,
    membership_path: str | Path = DEFAULT_PHASE1_MEMBERSHIP,
    beta_panel_path: str | Path = DEFAULT_PHASE2_BETA_PANEL,
    size_panel_path: str | Path = DEFAULT_SIZE_PANEL,
    non_price_panel_path: str | Path = DEFAULT_NON_PRICE_PANEL,
    fundamental_panel_path: str | Path = DEFAULT_FUNDAMENTAL_PANEL,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
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
    holding_horizons: Sequence[int] = DEFAULT_HOLDING_HORIZONS,
    style_prefixes: Sequence[str] = DEFAULT_STYLE_PREFIXES,
    min_sector_names: int = DEFAULT_MIN_SECTOR_NAMES,
    rolling_corr_window: int = DEFAULT_ROLLING_CORR_WINDOW,
    rolling_corr_min_periods: int = DEFAULT_ROLLING_CORR_MIN_PERIODS,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    membership = _load_membership(membership_path, variant=variant, start=start, end=end)
    beta = _load_beta(beta_panel_path)
    size = _load_size_panel(size_panel_path, variant=variant, start=start, end=end)
    non_price = _load_optional_panel(non_price_panel_path, NON_PRICE_COLUMNS, start=start, end=end)
    fundamentals = _load_optional_panel(
        fundamental_panel_path,
        FUNDAMENTAL_COLUMNS,
        start=start,
        end=end,
    )
    sic = _load_sec_sic_map(cik_mapping_path, sec_submissions_dir)
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
        size=size,
        non_price=non_price,
        fundamentals=fundamentals,
        stock_prices=stock_prices,
        benchmark_prices=benchmark_prices,
        benchmark_symbol=benchmark_symbol,
        holding_horizons=holding_horizons,
    )
    panel = panel.merge(
        sic[["symbol", "sic2_sector", "sic4_industry"]],
        on="symbol",
        how="left",
    )
    panel["sic2_sector"] = panel["sic2_sector"].fillna("UNKNOWN").astype(str)
    panel["sic4_industry"] = panel["sic4_industry"].fillna("UNKNOWN").astype(str)

    specs = _filter_specs(
        _factor_specs(holding_horizons=holding_horizons),
        style_prefixes=style_prefixes,
    )
    if not specs:
        raise ValueError(f"No styles matched prefixes: {style_prefixes}")
    payoff = _sector_neutral_payoffs(
        panel,
        specs=specs,
        top_bottom_quantile=top_bottom_quantile,
        min_sector_names=min_sector_names,
    )
    eval_payoff = payoff[
        payoff["session_date"].ge(eval_start) & payoff["session_date"].le(end)
    ].copy()
    metrics = _neutralized_metrics(eval_payoff)
    corr = _correlation_diagnostics(eval_payoff)
    rolling_corr = _rolling_correlation_diagnostics(
        eval_payoff,
        window=rolling_corr_window,
        min_periods=rolling_corr_min_periods,
    )
    rolling_summary = _rolling_corr_summary(rolling_corr)
    metrics = metrics.merge(corr, on=["neutralization", "holding", "style"], how="left").merge(
        rolling_summary,
        on=["neutralization", "holding", "style"],
        how="left",
    )
    robustness = _robustness_table(metrics)
    family_summary = _family_summary(metrics)
    shortlist = _shortlist(robustness)

    panel_path = output_dir / "phase7h_feature_target_panel_sample.csv"
    payoff_path = output_dir / "phase7h_sector_neutral_payoff_panel.csv"
    metrics_path = output_dir / "phase7h_sector_neutral_metrics.csv"
    robustness_path = output_dir / "phase7h_factor_robustness.csv"
    family_summary_path = output_dir / "phase7h_family_summary.csv"
    shortlist_path = output_dir / "phase7h_shortlist.csv"
    corr_path = output_dir / "phase7h_correlation_diagnostics.csv"
    rolling_corr_path = output_dir / "phase7h_rolling_correlation_diagnostics.csv"
    memo_path = output_dir / "phase7h_sector_neutral_factor_grid_memo.md"
    rollup_path = output_dir / "phase7h_rollup.json"

    panel.head(5000).to_csv(panel_path, index=False)
    payoff.to_csv(payoff_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    robustness.to_csv(robustness_path, index=False)
    family_summary.to_csv(family_summary_path, index=False)
    shortlist.to_csv(shortlist_path, index=False)
    corr.to_csv(corr_path, index=False)
    rolling_corr.to_csv(rolling_corr_path, index=False)
    memo_path.write_text(
        _memo(
            variant=variant,
            start=start,
            eval_start=eval_start,
            end=end,
            top_bottom_quantile=top_bottom_quantile,
            min_sector_names=min_sector_names,
            style_prefixes=style_prefixes,
            panel=panel,
            payoff=payoff,
            metrics=metrics,
            robustness=robustness,
            family_summary=family_summary,
            shortlist=shortlist,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "phase": "phase7h_sector_neutral_factor_grid",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "variant": variant,
        "start": start,
        "eval_start": eval_start,
        "end": end,
        "top_bottom_quantile": float(top_bottom_quantile),
        "holding_horizons": [int(value) for value in holding_horizons],
        "style_prefixes": list(style_prefixes),
        "min_sector_names": int(min_sector_names),
        "rows": {
            "membership": int(len(membership)),
            "panel": int(len(panel)),
            "payoff": int(len(payoff)),
            "eval_payoff": int(len(eval_payoff)),
            "styles": int(len(specs)),
        },
        "outputs": {
            "feature_target_panel_sample": panel_path.as_posix(),
            "sector_neutral_payoff_panel": payoff_path.as_posix(),
            "sector_neutral_metrics": metrics_path.as_posix(),
            "factor_robustness": robustness_path.as_posix(),
            "family_summary": family_summary_path.as_posix(),
            "shortlist": shortlist_path.as_posix(),
            "correlation_diagnostics": corr_path.as_posix(),
            "rolling_correlation_diagnostics": rolling_corr_path.as_posix(),
            "memo": memo_path.as_posix(),
        },
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _filter_specs(specs: Sequence[Any], *, style_prefixes: Sequence[str]) -> list[Any]:
    normalized = tuple(prefix.strip() for prefix in style_prefixes if prefix.strip())
    if not normalized or normalized == ("all",):
        return list(specs)
    return [spec for spec in specs if spec.style.startswith(normalized)]


def _sector_neutral_payoffs(
    panel: pd.DataFrame,
    *,
    specs: Sequence[Any],
    top_bottom_quantile: float,
    min_sector_names: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session_date, group in panel.groupby("session_date", sort=True):
        for spec in specs:
            target_column = f"forward_beta_residual_return_h{spec.holding}"
            subset = group[
                ["sic2_sector", spec.signal_column, target_column]
            ].copy()
            subset["raw_score"] = spec.score_sign * pd.to_numeric(
                subset[spec.signal_column],
                errors="coerce",
            )
            subset[target_column] = pd.to_numeric(subset[target_column], errors="coerce")
            subset = subset.dropna(subset=["raw_score", target_column, "sic2_sector"])
            if len(subset) < 100:
                continue
            rows.extend(
                _raw_or_rank_rows(
                    subset,
                    session_date=session_date,
                    spec=spec,
                    target_column=target_column,
                    top_bottom_quantile=top_bottom_quantile,
                    neutralization="raw",
                    score_column="raw_score",
                )
            )
            ranked = subset.copy()
            ranked["sector_rank_score"] = ranked.groupby("sic2_sector")["raw_score"].rank(
                pct=True,
                method="average",
            ) - 0.5
            rows.extend(
                _raw_or_rank_rows(
                    ranked,
                    session_date=session_date,
                    spec=spec,
                    target_column=target_column,
                    top_bottom_quantile=top_bottom_quantile,
                    neutralization="sector_rank",
                    score_column="sector_rank_score",
                )
            )
            balanced = _sector_balanced_row(
                subset,
                session_date=session_date,
                spec=spec,
                target_column=target_column,
                top_bottom_quantile=top_bottom_quantile,
                min_sector_names=min_sector_names,
            )
            if balanced is not None:
                rows.append(balanced)
    return pd.DataFrame(rows).sort_values(
        ["session_date", "neutralization", "holding", "style"]
    ).reset_index(drop=True)


def _raw_or_rank_rows(
    subset: pd.DataFrame,
    *,
    session_date: str,
    spec: Any,
    target_column: str,
    top_bottom_quantile: float,
    neutralization: str,
    score_column: str,
) -> list[dict[str, Any]]:
    score = pd.to_numeric(subset[score_column], errors="coerce")
    low = score.quantile(top_bottom_quantile)
    high = score.quantile(1.0 - top_bottom_quantile)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return []
    long = subset[score >= high]
    short = subset[score <= low]
    if long.empty or short.empty:
        return []
    return [
        _payoff_row(
            session_date=session_date,
            spec=spec,
            neutralization=neutralization,
            names=len(subset),
            sectors=int(subset["sic2_sector"].nunique()),
            long_names=len(long),
            short_names=len(short),
            long_sectors=int(long["sic2_sector"].nunique()),
            short_sectors=int(short["sic2_sector"].nunique()),
            long_signal_mean=float(long[spec.signal_column].mean()),
            short_signal_mean=float(short[spec.signal_column].mean()),
            payoff=float(long[target_column].mean() - short[target_column].mean()),
            name_weighted_payoff=np.nan,
            description=spec.description,
        )
    ]


def _sector_balanced_row(
    subset: pd.DataFrame,
    *,
    session_date: str,
    spec: Any,
    target_column: str,
    top_bottom_quantile: float,
    min_sector_names: int,
) -> dict[str, Any] | None:
    sector_rows: list[dict[str, float]] = []
    for sector, sector_group in subset.groupby("sic2_sector", sort=True):
        if sector == "UNKNOWN" or len(sector_group) < min_sector_names:
            continue
        low = sector_group["raw_score"].quantile(top_bottom_quantile)
        high = sector_group["raw_score"].quantile(1.0 - top_bottom_quantile)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            continue
        long = sector_group[sector_group["raw_score"] >= high]
        short = sector_group[sector_group["raw_score"] <= low]
        if long.empty or short.empty:
            continue
        sector_rows.append(
            {
                "sector": sector,
                "names": float(len(sector_group)),
                "long_names": float(len(long)),
                "short_names": float(len(short)),
                "long_signal_mean": float(long[spec.signal_column].mean()),
                "short_signal_mean": float(short[spec.signal_column].mean()),
                "payoff": float(long[target_column].mean() - short[target_column].mean()),
            }
        )
    if not sector_rows:
        return None
    sector_frame = pd.DataFrame(sector_rows)
    name_weighted = np.average(sector_frame["payoff"], weights=sector_frame["names"])
    return _payoff_row(
        session_date=session_date,
        spec=spec,
        neutralization="sector_balanced",
        names=int(sector_frame["names"].sum()),
        sectors=int(len(sector_frame)),
        long_names=int(sector_frame["long_names"].sum()),
        short_names=int(sector_frame["short_names"].sum()),
        long_sectors=int(len(sector_frame)),
        short_sectors=int(len(sector_frame)),
        long_signal_mean=float(
            np.average(sector_frame["long_signal_mean"], weights=sector_frame["names"])
        ),
        short_signal_mean=float(
            np.average(sector_frame["short_signal_mean"], weights=sector_frame["names"])
        ),
        payoff=float(sector_frame["payoff"].mean()),
        name_weighted_payoff=float(name_weighted),
        description=spec.description,
    )


def _payoff_row(
    *,
    session_date: str,
    spec: Any,
    neutralization: str,
    names: int,
    sectors: int,
    long_names: int,
    short_names: int,
    long_sectors: int,
    short_sectors: int,
    long_signal_mean: float,
    short_signal_mean: float,
    payoff: float,
    name_weighted_payoff: float,
    description: str,
) -> dict[str, Any]:
    return {
        "session_date": str(session_date),
        "neutralization": neutralization,
        "style": spec.style,
        "family": spec.family,
        "formation_lookback": int(spec.formation_lookback),
        "skip": int(spec.skip),
        "holding": int(spec.holding),
        "signal_column": spec.signal_column,
        "score_sign": float(spec.score_sign),
        "names": int(names),
        "sectors": int(sectors),
        "long_names": int(long_names),
        "short_names": int(short_names),
        "long_sectors": int(long_sectors),
        "short_sectors": int(short_sectors),
        "long_signal_mean": float(long_signal_mean),
        "short_signal_mean": float(short_signal_mean),
        "payoff": float(payoff),
        "payoff_bps": float(payoff * 10000.0),
        "name_weighted_payoff": float(name_weighted_payoff)
        if np.isfinite(name_weighted_payoff)
        else np.nan,
        "name_weighted_payoff_bps": float(name_weighted_payoff * 10000.0)
        if np.isfinite(name_weighted_payoff)
        else np.nan,
        "description": description,
        "test_window_used": False,
    }


def _neutralized_metrics(payoff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (neutralization, holding, style), group in payoff.groupby(
        ["neutralization", "holding", "style"],
        sort=True,
    ):
        series = pd.to_numeric(group["payoff"], errors="coerce").dropna()
        if series.empty:
            continue
        first = group.iloc[0]
        rows.append(
            {
                "neutralization": neutralization,
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
                "avg_names": float(group["names"].mean()),
                "avg_sectors": float(group["sectors"].mean()),
                "avg_long_names": float(group["long_names"].mean()),
                "avg_short_names": float(group["short_names"].mean()),
                "description": first["description"],
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["neutralization", "holding", "mean_payoff_bps"],
        ascending=[True, True, False],
    )


def _correlation_diagnostics(payoff: pd.DataFrame) -> pd.DataFrame:
    pivot = _payoff_pivot(payoff)
    rows = []
    for (neutralization, holding), group in payoff.groupby(["neutralization", "holding"], sort=True):
        same_method_reversal = f"{neutralization}|reversal_5d_h{int(holding)}"
        raw_reversal = f"raw|reversal_5d_h{int(holding)}"
        for style in sorted(group["style"].unique()):
            key = f"{neutralization}|{style}"
            if key not in pivot.columns:
                continue
            row = {
                "neutralization": neutralization,
                "holding": int(holding),
                "style": style,
                "test_window_used": False,
            }
            if same_method_reversal in pivot.columns:
                subset = pd.concat([pivot[key], pivot[same_method_reversal]], axis=1).dropna()
                row["same_method_reversal_corr_sessions"] = int(len(subset))
                row["pearson_corr_vs_same_method_reversal"] = _safe_corr(
                    subset.iloc[:, 0],
                    subset.iloc[:, 1],
                    method="pearson",
                )
                row["spearman_corr_vs_same_method_reversal"] = _safe_corr(
                    subset.iloc[:, 0],
                    subset.iloc[:, 1],
                    method="spearman",
                )
            if raw_reversal in pivot.columns:
                subset = pd.concat([pivot[key], pivot[raw_reversal]], axis=1).dropna()
                row["raw_reversal_corr_sessions"] = int(len(subset))
                row["pearson_corr_vs_raw_reversal"] = _safe_corr(
                    subset.iloc[:, 0],
                    subset.iloc[:, 1],
                    method="pearson",
                )
                row["spearman_corr_vs_raw_reversal"] = _safe_corr(
                    subset.iloc[:, 0],
                    subset.iloc[:, 1],
                    method="spearman",
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _rolling_correlation_diagnostics(
    payoff: pd.DataFrame,
    *,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    pivot = _payoff_pivot(payoff)
    rows = []
    for (neutralization, holding), group in payoff.groupby(["neutralization", "holding"], sort=True):
        raw_reversal = f"raw|reversal_5d_h{int(holding)}"
        if raw_reversal not in pivot.columns:
            continue
        baseline = pivot[raw_reversal]
        for style in sorted(group["style"].unique()):
            key = f"{neutralization}|{style}"
            if key == raw_reversal or key not in pivot.columns:
                continue
            rolling = pivot[key].rolling(window, min_periods=min_periods).corr(baseline)
            for session_date, value in rolling.dropna().items():
                rows.append(
                    {
                        "session_date": str(session_date.date()),
                        "neutralization": neutralization,
                        "holding": int(holding),
                        "style": style,
                        "rolling_corr_vs_raw_reversal": float(value),
                        "window": int(window),
                        "min_periods": int(min_periods),
                        "test_window_used": False,
                    }
                )
    return pd.DataFrame(rows)


def _rolling_corr_summary(rolling: pd.DataFrame) -> pd.DataFrame:
    if rolling.empty:
        return pd.DataFrame(columns=["neutralization", "holding", "style"])
    rows = []
    for (neutralization, holding, style), group in rolling.groupby(
        ["neutralization", "holding", "style"],
        sort=True,
    ):
        values = pd.to_numeric(group["rolling_corr_vs_raw_reversal"], errors="coerce").dropna()
        rows.append(
            {
                "neutralization": neutralization,
                "holding": int(holding),
                "style": style,
                "mean_rolling_corr_vs_raw_reversal": float(values.mean()),
                "median_rolling_corr_vs_raw_reversal": float(values.median()),
                "p10_rolling_corr_vs_raw_reversal": float(values.quantile(0.10)),
                "p90_rolling_corr_vs_raw_reversal": float(values.quantile(0.90)),
                "rolling_corr_sessions": int(len(values)),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows)


def _robustness_table(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "mean_payoff_bps",
        "newey_west_t_stat",
        "hit_rate",
        "sessions",
        "avg_names",
        "avg_sectors",
        "pearson_corr_vs_raw_reversal",
        "mean_rolling_corr_vs_raw_reversal",
    ]
    keys = ["style", "family", "holding", "signal_column"]
    frame = metrics.copy()
    if "signal_column" not in frame.columns:
        frame["signal_column"] = frame["style"].str.rsplit("_h", n=1).str[0]
    rows = []
    for (style, family, holding), group in frame.groupby(["style", "family", "holding"], sort=True):
        row: dict[str, Any] = {
            "style": style,
            "family": family,
            "holding": int(holding),
            "test_window_used": False,
        }
        for method in NEUTRALIZATION_METHODS:
            method_row = group[group["neutralization"].eq(method)]
            if method_row.empty:
                continue
            first = method_row.iloc[0]
            for column in metric_columns:
                if column in first:
                    row[f"{method}_{column}"] = first[column]
        if "raw_mean_payoff_bps" in row and "sector_rank_mean_payoff_bps" in row:
            row["sector_rank_minus_raw_mean_bps"] = (
                row["sector_rank_mean_payoff_bps"] - row["raw_mean_payoff_bps"]
            )
        if "raw_mean_payoff_bps" in row and "sector_balanced_mean_payoff_bps" in row:
            row["sector_balanced_minus_raw_mean_bps"] = (
                row["sector_balanced_mean_payoff_bps"] - row["raw_mean_payoff_bps"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["holding", "sector_balanced_newey_west_t_stat", "sector_rank_newey_west_t_stat"],
        ascending=[True, False, False],
    )


def _family_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    candidates = metrics[~metrics["family"].eq("short_horizon_reversal")].copy()
    rows = []
    for (neutralization, holding, family), group in candidates.groupby(
        ["neutralization", "holding", "family"],
        sort=True,
    ):
        best_nw = group.sort_values(
            ["newey_west_t_stat", "mean_payoff_bps"],
            ascending=[False, False],
        ).iloc[0]
        rows.append(
            {
                "neutralization": neutralization,
                "holding": int(holding),
                "family": family,
                "tested_styles": int(len(group)),
                "positive_styles": int((group["mean_payoff_bps"] > 0.0).sum()),
                "median_mean_payoff_bps": float(group["mean_payoff_bps"].median()),
                "best_newey_west_style": best_nw["style"],
                "best_newey_west_payoff_bps": float(best_nw["mean_payoff_bps"]),
                "best_newey_west_t_stat": float(best_nw["newey_west_t_stat"]),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["neutralization", "holding", "best_newey_west_t_stat"],
        ascending=[True, True, False],
    )


def _shortlist(robustness: pd.DataFrame) -> pd.DataFrame:
    keep_prefixes = (
        "size_small_cap",
        "quality_high_cash_to_assets",
        "defensive_low_beta",
        "defensive_low_vol_60d",
        "defensive_low_vol_120d",
        "value_book_to_market",
        "value_cash_flow_yield",
        "value_earnings_yield",
        "quality_low_fragility",
        "quality_low_leverage_pressure",
    )
    mask = robustness["style"].apply(lambda value: str(value).startswith(keep_prefixes))
    return robustness[mask].sort_values(
        ["holding", "sector_balanced_newey_west_t_stat", "sector_rank_newey_west_t_stat"],
        ascending=[True, False, False],
    )


def _payoff_pivot(payoff: pd.DataFrame) -> pd.DataFrame:
    frame = payoff.copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    frame["key"] = frame["neutralization"].astype(str) + "|" + frame["style"].astype(str)
    return frame.pivot_table(
        index="session_date",
        columns="key",
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
    min_sector_names: int,
    style_prefixes: Sequence[str],
    panel: pd.DataFrame,
    payoff: pd.DataFrame,
    metrics: pd.DataFrame,
    robustness: pd.DataFrame,
    family_summary: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> str:
    key_styles = shortlist[
        shortlist["style"].str.contains(
            "size_small_cap|quality_high_cash_to_assets|defensive_low_beta",
            regex=True,
        )
    ]
    baseline = metrics[metrics["family"].eq("short_horizon_reversal")]
    lines = [
        "# Phase7H Sector-Neutral Factor Grid Results",
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
        f"- tested style prefixes: `{list(style_prefixes)}`",
        f"- min sector names for sector-balanced sleeves: `{min_sector_names}`",
        f"- panel rows: `{len(panel)}`",
        f"- payoff rows: `{len(payoff)}`",
        "",
        "## Neutralization Methods",
        "",
        "- `raw`: ordinary cross-sectional top/bottom quintile.",
        "- `sector_rank`: rank the score inside each SIC2 sector, then take global top/bottom quintiles.",
        "- `sector_balanced`: form top/bottom quintiles inside each SIC2 sector, then equal-weight sector payoffs.",
        "",
        "## Family Summary",
        "",
        _markdown_table(family_summary),
        "",
        "## Key Candidate Robustness",
        "",
        _markdown_table(key_styles),
        "",
        "## Shortlist",
        "",
        _markdown_table(shortlist),
        "",
        "## Reversal Baseline",
        "",
        _markdown_table(baseline),
        "",
        "## Notes",
        "",
        "- Sector neutralization uses `sic2_sector` from SEC mapping; `UNKNOWN` sectors are skipped in the strict `sector_balanced` method.",
        "- `sector_balanced` payoff is equal-weighted across valid sectors, so it is a diagnostic for robustness, not a production-weighting prescription.",
        "- The payoff remains beta-residual forward return; transaction costs, borrow, capacity, and turnover are still outside this diagnostic.",
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
    if any(part <= 0 for part in parsed):
        raise ValueError("holding horizons must be positive")
    return parsed


def _parse_string_grid(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("style-prefixes cannot be empty")
    return parsed


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
