from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4aa import (
    _add_former_winner_scores,
    _fmt_bps,
    _fmt_float,
    _fmt_pct,
    _table,
)
from stockmachine.apps.run_pure_alpha_phase4ab import _add_hybrid_scores
from stockmachine.apps.run_pure_alpha_phase4af import _add_falling_knife_overlay_scores
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_LONG_VARIANT,
    DEFAULT_SHORT_VARIANT,
    _load_panel,
)
from stockmachine.apps.run_pure_alpha_phase4z import (
    DEFAULT_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_DAILY_GLOBS,
    DEFAULT_DAILY_GLOBS,
    _active_return_dates,
    _load_open_to_open_returns,
)


DEFAULT_SOURCE_ROOT = RESEARCH_ROOT / "phase5f_overlay_combo_on_turnover_aware_20260427"
DEFAULT_POSITIONS_PATH = DEFAULT_SOURCE_ROOT / "phase5f_positions_validation.csv.gz"
DEFAULT_STRICT_CURVE_PATH = DEFAULT_SOURCE_ROOT / "strict_h10_rebuild" / "phase4z_strict_h10_daily_curve.csv"
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5g_sota_loss_window_profiles_20260427"
DEFAULT_PORTFOLIO = "short_overlay_only"
DEFAULT_HOLDING_PERIOD_SESSIONS = 10
DEFAULT_VALIDATION_PRICE_END = "2019-12-31"
DEFAULT_TOP_N = 10
FEATURE_COLUMNS: tuple[str, ...] = (
    "beta",
    "beta_z",
    "reversal_5d",
    "return_5d",
    "return_5d_z",
    "momentum_20d_z",
    "momentum_60d_z",
    "beta_residual_momentum_20d_z",
    "vol_adjusted_momentum_20d_z",
    "former_winner_strength_mixed_20_60",
    "former_winner_recent_weakness",
    "former_winner_recent_is_down",
    "falling_knife_structural_weakness",
)
WINDOWS: tuple[tuple[str, str, str, str], ...] = (
    ("2015_peak_to_trough", "2015-06-04", "2015-08-21", "long_falling_knife"),
    ("2016_jan_feb", "2016-01-05", "2016-02-09", "long_falling_knife"),
    ("2017_spring_short_fail", "2017-03-20", "2017-05-31", "short_continuation"),
    ("2018_aug_sep_short_fail", "2018-08-08", "2018-09-12", "short_continuation"),
    ("2019_may", "2019-05-08", "2019-06-07", "mixed_stress"),
    ("2019_aug", "2019-07-30", "2019-08-28", "long_falling_knife"),
)


def build_phase5g_sota_loss_window_artifacts(
    *,
    positions_path: str | Path = DEFAULT_POSITIONS_PATH,
    strict_curve_path: str | Path = DEFAULT_STRICT_CURVE_PATH,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    portfolio: str = DEFAULT_PORTFOLIO,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    daily_globs: Sequence[str | Path] = DEFAULT_DAILY_GLOBS,
    adj_factor_globs: Sequence[str | Path] = DEFAULT_ADJ_FACTOR_GLOBS,
    benchmark_daily_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_DAILY_GLOBS,
    benchmark_adj_factor_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    benchmark_symbol: str = "SPY",
    holding_period_sessions: int = DEFAULT_HOLDING_PERIOD_SESSIONS,
    validation_price_end: str = DEFAULT_VALIDATION_PRICE_END,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = _load_sota_positions(positions_path, portfolio=portfolio)
    strict_curve = _load_strict_curve(strict_curve_path, portfolio=portfolio)
    signal_features = _load_signal_features(
        signal_panel_path=signal_panel_path,
        long_variant=long_variant,
        short_variant=short_variant,
    )
    active = _build_active_positions(
        positions,
        signal_features=signal_features,
        daily_globs=daily_globs,
        adj_factor_globs=adj_factor_globs,
        benchmark_daily_globs=benchmark_daily_globs,
        benchmark_adj_factor_globs=benchmark_adj_factor_globs,
        benchmark_symbol=benchmark_symbol,
        holding_period_sessions=holding_period_sessions,
        validation_price_end=validation_price_end,
    )

    window_summary = _window_summary(strict_curve)
    culprit_symbols = _top_culprit_symbols(active, top_n=top_n)
    feature_profile = _feature_profile(active, culprit_symbols)
    bucket_lifts = _bucket_lifts(active, culprit_symbols)
    repeat_offenders = _repeat_offenders(culprit_symbols)

    positions_out = output_dir / "phase5g_culprit_symbols.csv"
    window_out = output_dir / "phase5g_window_summary.csv"
    feature_out = output_dir / "phase5g_feature_profile.csv"
    bucket_out = output_dir / "phase5g_bucket_lifts.csv"
    repeat_out = output_dir / "phase5g_repeat_offenders.csv"
    memo_out = output_dir / "phase5g_sota_loss_window_profiles_memo.md"
    rollup_out = output_dir / "phase5g_rollup.json"

    culprit_symbols.to_csv(positions_out, index=False)
    window_summary.to_csv(window_out, index=False)
    feature_profile.to_csv(feature_out, index=False)
    bucket_lifts.to_csv(bucket_out, index=False)
    repeat_offenders.to_csv(repeat_out, index=False)
    memo_out.write_text(
        _memo(
            window_summary=window_summary,
            culprit_symbols=culprit_symbols,
            feature_profile=feature_profile,
            bucket_lifts=bucket_lifts,
            repeat_offenders=repeat_offenders,
            top_n=top_n,
            portfolio=portfolio,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "positions_path": Path(positions_path).as_posix(),
        "strict_curve_path": Path(strict_curve_path).as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "portfolio": portfolio,
        "holding_period_sessions": int(holding_period_sessions),
        "validation_price_end": validation_price_end,
        "top_n": int(top_n),
        "windows": [
            {
                "window": name,
                "start": start,
                "end": end,
                "theme": theme,
            }
            for name, start, end, theme in WINDOWS
        ],
        "feature_columns": list(FEATURE_COLUMNS),
        "window_summary_artifact": window_out.as_posix(),
        "culprit_symbols_artifact": positions_out.as_posix(),
        "feature_profile_artifact": feature_out.as_posix(),
        "bucket_lifts_artifact": bucket_out.as_posix(),
        "repeat_offenders_artifact": repeat_out.as_posix(),
        "memo_artifact": memo_out.as_posix(),
        "method": "sota_loss_window_culprit_feature_profiling",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "Profiles are descriptive diagnostics on the current SOTA portfolio, not causal attribution.",
            "Feature averages are absolute-weighted over active sleeve rows inside each window.",
            "The h10 signal panel keeps legacy *_5d names even though the contract is h10.",
            "No test-window rows are used.",
        ],
    }
    rollup_out.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_sota_positions(path: str | Path, *, portfolio: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["session_date"])
    frame = frame[frame["portfolio"].astype(str).eq(portfolio)].copy()
    if frame.empty:
        raise ValueError(f"No positions found for portfolio={portfolio!r} in {path}.")
    frame["merge_variant"] = np.where(
        frame["side"].astype(str).eq("long"),
        frame["long_variant"].astype(str),
        frame["short_variant"].astype(str),
    )
    return frame.sort_values(["session_date", "side", "symbol"]).reset_index(drop=True)


def _load_strict_curve(path: str | Path, *, portfolio: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["return_date"])
    frame = frame[frame["portfolio"].astype(str).eq(portfolio)].copy()
    if frame.empty:
        raise ValueError(f"No strict curve rows found for portfolio={portfolio!r} in {path}.")
    return frame.sort_values("return_date").reset_index(drop=True)


def _load_signal_features(
    *,
    signal_panel_path: str | Path,
    long_variant: str,
    short_variant: str,
) -> pd.DataFrame:
    panel = _load_panel(signal_panel_path, long_variant, short_variant)
    panel = _add_former_winner_scores(panel)
    panel = _add_hybrid_scores(panel)
    panel = _add_falling_knife_overlay_scores(panel)
    panel["session_date"] = pd.to_datetime(panel["session_date"])
    usecols = ["session_date", "symbol", "variant", *FEATURE_COLUMNS]
    return panel[usecols].copy()


def _build_active_positions(
    positions: pd.DataFrame,
    *,
    signal_features: pd.DataFrame,
    daily_globs: Sequence[str | Path],
    adj_factor_globs: Sequence[str | Path],
    benchmark_daily_globs: Sequence[str | Path],
    benchmark_adj_factor_globs: Sequence[str | Path],
    benchmark_symbol: str,
    holding_period_sessions: int,
    validation_price_end: str,
) -> pd.DataFrame:
    feature_rows = positions.merge(
        signal_features,
        left_on=["session_date", "symbol", "merge_variant"],
        right_on=["session_date", "symbol", "variant"],
        how="left",
    )
    feature_rows = feature_rows.drop(columns=["variant"])
    feature_rows = feature_rows.rename(columns={"session_date": "sleeve_session_date"})
    if "beta_x" in feature_rows.columns or "beta_y" in feature_rows.columns:
        feature_rows["beta"] = feature_rows.get("beta_x").fillna(feature_rows.get("beta_y"))
        feature_rows = feature_rows.drop(
            columns=[column for column in ("beta_x", "beta_y") if column in feature_rows.columns]
        )
    feature_lookup = feature_rows[
        ["sleeve_session_date", "side", "symbol", "merge_variant", *FEATURE_COLUMNS]
    ].drop_duplicates()

    symbols = tuple(sorted(feature_rows["symbol"].astype(str).unique()))
    stock_returns = _load_open_to_open_returns(
        daily_globs=daily_globs,
        adj_factor_globs=adj_factor_globs,
        symbols=symbols,
        end_date=validation_price_end,
    )
    benchmark_returns = _load_open_to_open_returns(
        daily_globs=benchmark_daily_globs,
        adj_factor_globs=benchmark_adj_factor_globs,
        symbols=(benchmark_symbol,),
        end_date=validation_price_end,
    )
    calendar = (
        benchmark_returns[benchmark_returns["symbol"].eq(benchmark_symbol)]["session_date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    active_map = _active_return_dates(
        positions[
            ["session_date", "portfolio", "side", "symbol", "signed_weight", "test_window_used"]
        ].copy(),
        calendar=calendar,
        holding_period_sessions=holding_period_sessions,
    )
    active = positions.merge(active_map, on="session_date", how="inner")
    active = active.rename(columns={"session_date": "sleeve_session_date"})
    active = active.merge(
        feature_lookup,
        on=["sleeve_session_date", "side", "symbol", "merge_variant"],
        how="left",
    )
    active = active.merge(
        stock_returns.rename(columns={"session_date": "return_date"}),
        on=["return_date", "symbol"],
        how="left",
    )
    active["missing_return"] = active["oto_return"].isna()
    active["weighted_return"] = active["signed_weight"] * active["oto_return"].fillna(0.0)
    complete_key = ["return_date", "sleeve_session_date"]
    complete = active.groupby(complete_key)["missing_return"].sum().reset_index()
    complete = complete[complete["missing_return"].eq(0)][complete_key]
    active = active.merge(complete, on=complete_key, how="inner")
    active["abs_weight"] = active["signed_weight"].abs()
    active["theme"] = None
    return active.sort_values(["return_date", "side", "symbol"]).reset_index(drop=True)


def _window_summary(strict_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window, start, end, theme in WINDOWS:
        subset = strict_curve[
            (strict_curve["return_date"] >= pd.Timestamp(start))
            & (strict_curve["return_date"] <= pd.Timestamp(end))
        ].copy()
        if subset.empty:
            continue
        equity = (1.0 + subset["gross_return"].fillna(0.0)).prod() - 1.0
        benchmark = (1.0 + subset["benchmark_oto_return"].fillna(0.0)).prod() - 1.0
        rows.append(
            {
                "window": window,
                "start": start,
                "end": end,
                "theme": theme,
                "sessions": int(len(subset)),
                "strategy_compound_return": float(equity),
                "strategy_long_sum": float(subset["long_gross_return"].sum()),
                "strategy_short_sum": float(subset["short_gross_return"].sum()),
                "spy_compound_return": float(benchmark),
            }
        )
    return pd.DataFrame(rows)


def _top_culprit_symbols(active: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for window, start, end, theme in WINDOWS:
        subset = active[
            (active["return_date"] >= pd.Timestamp(start))
            & (active["return_date"] <= pd.Timestamp(end))
        ].copy()
        for side in ("long", "short"):
            side_subset = subset[subset["side"].eq(side)].copy()
            if side_subset.empty:
                continue
            by_symbol = (
                side_subset.groupby("symbol", as_index=False)
                .agg(
                    total_contribution=("weighted_return", "sum"),
                    active_rows=("weighted_return", "size"),
                    avg_abs_weight=("abs_weight", "mean"),
                )
                .sort_values(["total_contribution", "symbol"], ascending=[True, True])
                .head(top_n)
                .reset_index(drop=True)
            )
            by_symbol["window"] = window
            by_symbol["start"] = start
            by_symbol["end"] = end
            by_symbol["theme"] = theme
            by_symbol["side"] = side
            by_symbol["rank"] = np.arange(1, len(by_symbol) + 1, dtype=int)
            rows.append(by_symbol)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)[
        [
            "window",
            "start",
            "end",
            "theme",
            "side",
            "rank",
            "symbol",
            "total_contribution",
            "active_rows",
            "avg_abs_weight",
        ]
    ]


def _feature_profile(active: pd.DataFrame, culprit_symbols: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    culprit_map = {
        (window, side): set(group["symbol"].astype(str))
        for (window, side), group in culprit_symbols.groupby(["window", "side"], sort=False)
    }
    for window, start, end, theme in WINDOWS:
        subset = active[
            (active["return_date"] >= pd.Timestamp(start))
            & (active["return_date"] <= pd.Timestamp(end))
        ].copy()
        for side in ("long", "short"):
            side_subset = subset[subset["side"].eq(side)].copy()
            if side_subset.empty:
                continue
            culprits = side_subset[
                side_subset["symbol"].astype(str).isin(culprit_map.get((window, side), set()))
            ].copy()
            row: dict[str, Any] = {
                "window": window,
                "start": start,
                "end": end,
                "theme": theme,
                "side": side,
                "active_rows": int(len(side_subset)),
                "culprit_rows": int(len(culprits)),
            }
            for column in FEATURE_COLUMNS:
                culprit_mean = _weighted_mean(culprits, column)
                universe_mean = _weighted_mean(side_subset, column)
                row[f"{column}_culprit"] = culprit_mean
                row[f"{column}_universe"] = universe_mean
                row[f"{column}_diff"] = (
                    culprit_mean - universe_mean
                    if pd.notna(culprit_mean) and pd.notna(universe_mean)
                    else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _bucket_lifts(active: pd.DataFrame, culprit_symbols: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    culprit_map = {
        (window, side): set(group["symbol"].astype(str))
        for (window, side), group in culprit_symbols.groupby(["window", "side"], sort=False)
    }
    for window, start, end, theme in WINDOWS:
        subset = active[
            (active["return_date"] >= pd.Timestamp(start))
            & (active["return_date"] <= pd.Timestamp(end))
        ].copy()
        for side in ("long", "short"):
            side_subset = subset[subset["side"].eq(side)].copy()
            if side_subset.empty:
                continue
            culprits = side_subset[
                side_subset["symbol"].astype(str).isin(culprit_map.get((window, side), set()))
            ].copy()
            for bucket, culprit_mask, universe_mask in _bucket_specs(culprits, side_subset, side):
                culprit_share = _weighted_share(culprits, culprit_mask)
                universe_share = _weighted_share(side_subset, universe_mask)
                rows.append(
                    {
                        "window": window,
                        "start": start,
                        "end": end,
                        "theme": theme,
                        "side": side,
                        "bucket": bucket,
                        "culprit_share": culprit_share,
                        "universe_share": universe_share,
                        "lift": culprit_share / universe_share
                        if pd.notna(culprit_share) and pd.notna(universe_share) and universe_share > 0
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _bucket_specs(
    culprits: pd.DataFrame,
    universe: pd.DataFrame,
    side: str,
) -> list[tuple[str, pd.Series, pd.Series]]:
    if side == "long":
        return [
            (
                "structural_weak_gt_050",
                culprits["falling_knife_structural_weakness"] > 0.50,
                universe["falling_knife_structural_weakness"] > 0.50,
            ),
            (
                "structural_weak_gt_075",
                culprits["falling_knife_structural_weakness"] > 0.75,
                universe["falling_knife_structural_weakness"] > 0.75,
            ),
            (
                "resid_mom_lt_m05",
                culprits["beta_residual_momentum_20d_z"] < -0.5,
                universe["beta_residual_momentum_20d_z"] < -0.5,
            ),
            (
                "mom60_lt_m05",
                culprits["momentum_60d_z"] < -0.5,
                universe["momentum_60d_z"] < -0.5,
            ),
            (
                "beta_gt_p05",
                culprits["beta_z"] > 0.5,
                universe["beta_z"] > 0.5,
            ),
        ]
    return [
        (
            "winner_strength_gt_p05",
            culprits["former_winner_strength_mixed_20_60"] > 0.5,
            universe["former_winner_strength_mixed_20_60"] > 0.5,
        ),
        (
            "winner_strength_gt_p10",
            culprits["former_winner_strength_mixed_20_60"] > 1.0,
            universe["former_winner_strength_mixed_20_60"] > 1.0,
        ),
        (
            "recent_not_down",
            ~culprits["former_winner_recent_is_down"].fillna(False),
            ~universe["former_winner_recent_is_down"].fillna(False),
        ),
        (
            "recent_weak_lt_p25",
            culprits["former_winner_recent_weakness"] < 0.25,
            universe["former_winner_recent_weakness"] < 0.25,
        ),
        (
            "mom20_gt_p05_and_beta_gt_p05",
            (culprits["momentum_20d_z"] > 0.5) & (culprits["beta_z"] > 0.5),
            (universe["momentum_20d_z"] > 0.5) & (universe["beta_z"] > 0.5),
        ),
    ]


def _repeat_offenders(culprit_symbols: pd.DataFrame) -> pd.DataFrame:
    if culprit_symbols.empty:
        return pd.DataFrame()
    frame = culprit_symbols.copy()
    return (
        frame.groupby(["side", "symbol"], as_index=False)
        .agg(
            windows=("window", "nunique"),
            windows_list=("window", lambda s: ", ".join(sorted(set(s)))),
            mean_rank=("rank", "mean"),
            total_contribution_sum=("total_contribution", "sum"),
        )
        .sort_values(["side", "windows", "total_contribution_sum"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def _weighted_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    valid = frame[[column, "abs_weight"]].dropna()
    if valid.empty:
        return np.nan
    total_weight = float(valid["abs_weight"].sum())
    if total_weight <= 0.0:
        return np.nan
    return float(np.average(valid[column], weights=valid["abs_weight"]))


def _weighted_share(frame: pd.DataFrame, mask: pd.Series) -> float:
    if frame.empty:
        return np.nan
    valid = frame[["abs_weight"]].copy()
    valid["mask"] = mask.astype(float).to_numpy()
    total_weight = float(valid["abs_weight"].sum())
    if total_weight <= 0.0:
        return np.nan
    return float(np.average(valid["mask"], weights=valid["abs_weight"]))


def _memo(
    *,
    window_summary: pd.DataFrame,
    culprit_symbols: pd.DataFrame,
    feature_profile: pd.DataFrame,
    bucket_lifts: pd.DataFrame,
    repeat_offenders: pd.DataFrame,
    top_n: int,
    portfolio: str,
) -> str:
    important_features = [
        "falling_knife_structural_weakness_diff",
        "momentum_60d_z_diff",
        "beta_residual_momentum_20d_z_diff",
        "beta_z_diff",
        "former_winner_strength_mixed_20_60_diff",
        "former_winner_recent_weakness_diff",
        "momentum_20d_z_diff",
    ]
    feature_view = feature_profile[
        [column for column in ["window", "side", *important_features] if column in feature_profile.columns]
    ].copy()
    bucket_view = (
        bucket_lifts.sort_values(["side", "window", "lift"], ascending=[True, True, False])
        .groupby(["window", "side"], as_index=False)
        .head(3)
        .reset_index(drop=True)
    )
    repeat_view = repeat_offenders[repeat_offenders["windows"] >= 2].copy()
    culprit_view = culprit_symbols[culprit_symbols["rank"] <= 5].copy()
    lines = [
        f"# Phase5G SOTA Loss-Window Culprit Profiles ({portfolio})",
        "",
        "## Window Summary",
        "",
        _table(window_summary),
        "",
        f"## Top-{top_n} Culprit Symbols",
        "",
        _table(culprit_view),
        "",
        "## Feature Diffs (culprit minus active universe)",
        "",
        _table(feature_view),
        "",
        "## Strongest Bucket Lifts",
        "",
        _table(bucket_view),
        "",
        "## Repeat Offenders (2+ windows)",
        "",
        _table(repeat_view),
        "",
        "## Reading",
        "",
        "- Long-side deep drawdowns still cluster in names that are simultaneously structurally weak, residual-momentum negative, and often above-average beta.",
        "- Short-side failures now look narrower: they are concentrated in strong former winners, especially high-momentum/high-beta growth names that have not really rolled over yet.",
        "- 2019 May is more mixed than 2015/2019 Aug: long culprits are still weak losers, but their feature lift is milder, while short culprits are clearly continuation-prone winners.",
        "- 2016 Jan-Feb remains a special case: the long book was hurt by a broad crash/rebound tape, so the culprit profile is less cleanly 'structural weak loser' than 2015 or 2019 Aug.",
        "",
        "## Key Numbers",
        "",
        f"- Worst long-dominant window: `2015-06-04 ~ 2015-08-21`, strategy compound {_fmt_pct(window_summary.loc[window_summary['window'].eq('2015_peak_to_trough'), 'strategy_compound_return'].iloc[0])}, long sum {_fmt_bps(window_summary.loc[window_summary['window'].eq('2015_peak_to_trough'), 'strategy_long_sum'].iloc[0])}, short sum {_fmt_bps(window_summary.loc[window_summary['window'].eq('2015_peak_to_trough'), 'strategy_short_sum'].iloc[0])}.",
        f"- Cleanest short-continuation window: `2017-03-20 ~ 2017-05-31`, strategy compound {_fmt_pct(window_summary.loc[window_summary['window'].eq('2017_spring_short_fail'), 'strategy_compound_return'].iloc[0])}, long sum {_fmt_bps(window_summary.loc[window_summary['window'].eq('2017_spring_short_fail'), 'strategy_long_sum'].iloc[0])}, short sum {_fmt_bps(window_summary.loc[window_summary['window'].eq('2017_spring_short_fail'), 'strategy_short_sum'].iloc[0])}.",
    ]
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile SOTA pure-alpha loss windows by culprit-symbol features.",
    )
    parser.add_argument("--positions-path", default=str(DEFAULT_POSITIONS_PATH))
    parser.add_argument("--strict-curve-path", default=str(DEFAULT_STRICT_CURVE_PATH))
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--portfolio", default=DEFAULT_PORTFOLIO)
    parser.add_argument("--holding-period-sessions", type=int, default=DEFAULT_HOLDING_PERIOD_SESSIONS)
    parser.add_argument("--validation-price-end", default=DEFAULT_VALIDATION_PRICE_END)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args(argv)

    rollup = build_phase5g_sota_loss_window_artifacts(
        positions_path=args.positions_path,
        strict_curve_path=args.strict_curve_path,
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        portfolio=args.portfolio,
        holding_period_sessions=args.holding_period_sessions,
        validation_price_end=args.validation_price_end,
        top_n=args.top_n,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
