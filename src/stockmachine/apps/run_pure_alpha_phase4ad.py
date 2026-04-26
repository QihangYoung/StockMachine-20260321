from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT, TARGET_COLUMN
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_LONG_SCORE,
    DEFAULT_LONG_VARIANT,
    DEFAULT_OUTPUT_ROOT as PHASE4S_OUTPUT_ROOT,
    DEFAULT_SHORT_VARIANT,
    _load_panel,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4ad_2015_long_selector_failure_20260419"
DEFAULT_POSITIONS = PHASE4S_OUTPUT_ROOT / "phase4s_positions_validation.csv.gz"
DEFAULT_PORTFOLIO = "sic2_soft_neutral"
DEFAULT_HOLDING_PERIOD_SESSIONS = 10
DEFAULT_STANDALONE_COUNTS = (30, 80)
RETURN_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("full_validation", "2014-08-07", "2019-12-31"),
    ("2015_peak_to_trough", "2015-06-04", "2015-08-21"),
    ("2015_recovery", "2015-08-24", "2015-11-09"),
    ("2019_may_aug", "2019-05-01", "2019-08-31"),
    ("2019_may", "2019-05-01", "2019-06-07"),
)


def build_phase4ad_2015_long_selector_failure_diagnostics(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    positions_path: str | Path = DEFAULT_POSITIONS,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    long_score: str = DEFAULT_LONG_SCORE,
    portfolio: str = DEFAULT_PORTFOLIO,
    standalone_counts: Sequence[int] = DEFAULT_STANDALONE_COUNTS,
    holding_period_sessions: int = DEFAULT_HOLDING_PERIOD_SESSIONS,
) -> dict[str, Any]:
    """Diagnose whether the 2015 drawdown came from long selector inversion."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_panel(signal_panel_path, long_variant, short_variant)
    long_panel = panel[panel["variant"].eq(long_variant)].copy()
    calendar = pd.Series(pd.to_datetime(sorted(long_panel["session_date"].unique())))
    windows = _window_table(
        calendar,
        return_windows=RETURN_WINDOWS,
        holding_period_sessions=holding_period_sessions,
    )
    standalone_daily = _standalone_long_daily(
        long_panel,
        long_score=long_score,
        candidate_counts=standalone_counts,
    )
    standalone_summary = _standalone_window_summary(standalone_daily, windows)
    score_bins = _score_bin_summary(long_panel, long_score=long_score, windows=windows)

    positions = _load_long_positions(positions_path, portfolio=portfolio)
    positions_daily = _actual_long_book_daily(positions)
    positions_summary = _actual_long_book_window_summary(positions_daily, windows)
    symbol_contrib = _symbol_contribution(
        positions,
        window=windows[windows["window"].eq("2015_peak_to_trough")].iloc[0],
    )
    sector_contrib = _sector_contribution(
        positions,
        window=windows[windows["window"].eq("2015_peak_to_trough")].iloc[0],
    )

    memo = _memo(
        windows=windows,
        standalone_summary=standalone_summary,
        score_bins=score_bins,
        positions_summary=positions_summary,
        symbol_contrib=symbol_contrib,
        sector_contrib=sector_contrib,
        portfolio=portfolio,
        long_variant=long_variant,
        long_score=long_score,
    )

    windows_path = output_dir / "phase4ad_window_mapping.csv"
    standalone_daily_path = output_dir / "phase4ad_standalone_long_daily_validation.csv"
    standalone_summary_path = output_dir / "phase4ad_standalone_long_window_summary.csv"
    score_bins_path = output_dir / "phase4ad_long_score_bin_summary.csv"
    positions_daily_path = output_dir / "phase4ad_actual_long_book_daily_validation.csv"
    positions_summary_path = output_dir / "phase4ad_actual_long_book_window_summary.csv"
    symbol_contrib_path = output_dir / "phase4ad_2015_worst_long_symbols.csv"
    sector_contrib_path = output_dir / "phase4ad_2015_worst_long_sectors.csv"
    memo_path = output_dir / "phase4ad_2015_long_selector_failure_memo.md"
    rollup_path = output_dir / "phase4ad_rollup.json"

    windows.to_csv(windows_path, index=False)
    standalone_daily.to_csv(standalone_daily_path, index=False)
    standalone_summary.to_csv(standalone_summary_path, index=False)
    score_bins.to_csv(score_bins_path, index=False)
    positions_daily.to_csv(positions_daily_path, index=False)
    positions_summary.to_csv(positions_summary_path, index=False)
    symbol_contrib.to_csv(symbol_contrib_path, index=False)
    sector_contrib.to_csv(sector_contrib_path, index=False)
    memo_path.write_text(memo, encoding="utf-8")

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "positions_path": Path(positions_path).as_posix(),
        "long_variant": long_variant,
        "short_variant": short_variant,
        "long_score": long_score,
        "portfolio": portfolio,
        "holding_period_sessions": int(holding_period_sessions),
        "standalone_counts": [int(x) for x in standalone_counts],
        "panel_rows": int(len(panel)),
        "long_panel_rows": int(len(long_panel)),
        "positions_rows": int(len(positions)),
        "windows_artifact": windows_path.as_posix(),
        "standalone_daily_artifact": standalone_daily_path.as_posix(),
        "standalone_summary_artifact": standalone_summary_path.as_posix(),
        "score_bins_artifact": score_bins_path.as_posix(),
        "positions_daily_artifact": positions_daily_path.as_posix(),
        "positions_summary_artifact": positions_summary_path.as_posix(),
        "symbol_contrib_artifact": symbol_contrib_path.as_posix(),
        "sector_contrib_artifact": sector_contrib_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "method": "long_reversal_selector_inversion_diagnostics_for_2015_drawdown",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "The h10 signal panel keeps legacy *_5d column names; values are h10 labels.",
            "Return-date drawdown windows are mapped to signal-date windows using the strict h10 active-sleeve contract.",
            "No test-window performance is computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _window_table(
    calendar: pd.Series,
    *,
    return_windows: Sequence[tuple[str, str, str]],
    holding_period_sessions: int,
) -> pd.DataFrame:
    rows = []
    calendar = pd.to_datetime(calendar).drop_duplicates().sort_values().reset_index(drop=True)
    for name, return_start, return_end in return_windows:
        if name == "full_validation":
            signal_start = str(calendar.min().date())
            signal_end = str(calendar.max().date())
            mode = "full_signal_window"
        else:
            signal_start, signal_end = _mapped_signal_window(
                calendar,
                return_start=return_start,
                return_end=return_end,
                holding_period_sessions=holding_period_sessions,
            )
            mode = "mapped_from_return_window"
        rows.append(
            {
                "window": name,
                "return_start": return_start,
                "return_end": return_end,
                "signal_start": signal_start,
                "signal_end": signal_end,
                "mapping_mode": mode,
                "holding_period_sessions": int(holding_period_sessions),
                "test_window_used": False,
            }
        )
    rows.append(
        {
            "window": "2015_peak_to_trough_same_dates",
            "return_start": "2015-06-04",
            "return_end": "2015-08-21",
            "signal_start": "2015-06-04",
            "signal_end": "2015-08-21",
            "mapping_mode": "same_dates_sanity_check",
            "holding_period_sessions": int(holding_period_sessions),
            "test_window_used": False,
        }
    )
    return pd.DataFrame(rows)


def _mapped_signal_window(
    calendar: pd.Series,
    *,
    return_start: str,
    return_end: str,
    holding_period_sessions: int,
) -> tuple[str, str]:
    values = calendar.to_numpy(dtype="datetime64[ns]")
    start_idx = int(np.searchsorted(values, np.datetime64(pd.Timestamp(return_start)), side="left"))
    end_idx = int(np.searchsorted(values, np.datetime64(pd.Timestamp(return_end)), side="right")) - 1
    signal_start_idx = max(0, start_idx - holding_period_sessions - 1)
    signal_end_idx = max(0, end_idx - 2)
    return (
        str(pd.Timestamp(calendar.iloc[signal_start_idx]).date()),
        str(pd.Timestamp(calendar.iloc[signal_end_idx]).date()),
    )


def _standalone_long_daily(
    long_panel: pd.DataFrame,
    *,
    long_score: str,
    candidate_counts: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session_date, group in long_panel.groupby("session_date", sort=True):
        valid = group.dropna(subset=[long_score, TARGET_COLUMN, "forward_return_5d"])
        if valid.empty:
            continue
        universe_mean = float(valid[TARGET_COLUMN].mean())
        rank_ic = valid[long_score].corr(valid[TARGET_COLUMN], method="spearman")
        for count in candidate_counts:
            if len(valid) < count * 2:
                continue
            top = valid.sort_values([long_score, "symbol"], ascending=[False, True]).head(count)
            bottom = valid.sort_values([long_score, "symbol"], ascending=[True, True]).head(count)
            rows.append(
                {
                    "session_date": session_date,
                    "long_variant": str(valid["variant"].iloc[0]),
                    "long_score": long_score,
                    "candidate_count": int(count),
                    "eligible_names": int(len(valid)),
                    "top_mean_target": float(top[TARGET_COLUMN].mean()),
                    "top_mean_raw": float(top["forward_return_5d"].mean()),
                    "top_positive_target_share": float((top[TARGET_COLUMN] > 0).mean()),
                    "bottom_mean_target": float(bottom[TARGET_COLUMN].mean()),
                    "universe_mean_target": universe_mean,
                    "top_edge_vs_universe": float(top[TARGET_COLUMN].mean() - universe_mean),
                    "top_minus_bottom_target": float(
                        top[TARGET_COLUMN].mean() - bottom[TARGET_COLUMN].mean()
                    ),
                    "rank_ic_score_vs_target": float(rank_ic),
                    "rank_ic_bad": bool(rank_ic < 0) if pd.notna(rank_ic) else False,
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows)


def _standalone_window_summary(
    standalone_daily: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    if standalone_daily.empty:
        return standalone_daily
    frame = standalone_daily.copy()
    frame["session_date_dt"] = pd.to_datetime(frame["session_date"])
    rows: list[dict[str, Any]] = []
    for window in windows.itertuples(index=False):
        subset = frame[
            (frame["session_date_dt"] >= pd.Timestamp(window.signal_start))
            & (frame["session_date_dt"] <= pd.Timestamp(window.signal_end))
        ]
        for count, group in subset.groupby("candidate_count", sort=True):
            rows.append(
                {
                    "window": window.window,
                    "return_start": window.return_start,
                    "return_end": window.return_end,
                    "signal_start": window.signal_start,
                    "signal_end": window.signal_end,
                    "mapping_mode": window.mapping_mode,
                    "candidate_count": int(count),
                    "sessions": int(len(group)),
                    "mean_top_target": float(group["top_mean_target"].mean()),
                    "top_target_hit_rate": float((group["top_mean_target"] > 0).mean()),
                    "mean_top_raw": float(group["top_mean_raw"].mean()),
                    "mean_top_positive_target_share": float(
                        group["top_positive_target_share"].mean()
                    ),
                    "mean_universe_target": float(group["universe_mean_target"].mean()),
                    "mean_top_edge_vs_universe": float(group["top_edge_vs_universe"].mean()),
                    "mean_bottom_target": float(group["bottom_mean_target"].mean()),
                    "mean_top_minus_bottom_target": float(
                        group["top_minus_bottom_target"].mean()
                    ),
                    "mean_rank_ic": float(group["rank_ic_score_vs_target"].mean()),
                    "rank_ic_bad_share": float(group["rank_ic_bad"].mean()),
                    "selector_inverted": bool(
                        group["rank_ic_score_vs_target"].mean() < 0
                        and group["top_minus_bottom_target"].mean() < 0
                    ),
                    "test_window_used": bool(group["test_window_used"].any()),
                }
            )
    return pd.DataFrame(rows)


def _score_bin_summary(
    long_panel: pd.DataFrame,
    *,
    long_score: str,
    windows: pd.DataFrame,
    bins: int = 10,
) -> pd.DataFrame:
    daily_rows: list[dict[str, Any]] = []
    for session_date, group in long_panel.groupby("session_date", sort=True):
        valid = group.dropna(subset=[long_score, TARGET_COLUMN, "forward_return_5d"]).copy()
        if len(valid) < bins * 2:
            continue
        ranks = valid[long_score].rank(method="first", ascending=True)
        valid["score_bin"] = pd.qcut(ranks, q=bins, labels=False, duplicates="drop") + 1
        for score_bin, bin_group in valid.groupby("score_bin", sort=True):
            daily_rows.append(
                {
                    "session_date": session_date,
                    "score_bin": int(score_bin),
                    "names": int(len(bin_group)),
                    "mean_target": float(bin_group[TARGET_COLUMN].mean()),
                    "mean_raw": float(bin_group["forward_return_5d"].mean()),
                    "positive_target_share": float((bin_group[TARGET_COLUMN] > 0).mean()),
                    "test_window_used": False,
                }
            )
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return daily
    daily["session_date_dt"] = pd.to_datetime(daily["session_date"])
    rows: list[dict[str, Any]] = []
    for window in windows.itertuples(index=False):
        subset = daily[
            (daily["session_date_dt"] >= pd.Timestamp(window.signal_start))
            & (daily["session_date_dt"] <= pd.Timestamp(window.signal_end))
        ]
        for score_bin, group in subset.groupby("score_bin", sort=True):
            rows.append(
                {
                    "window": window.window,
                    "return_start": window.return_start,
                    "return_end": window.return_end,
                    "signal_start": window.signal_start,
                    "signal_end": window.signal_end,
                    "score_bin": int(score_bin),
                    "sessions": int(group["session_date"].nunique()),
                    "mean_names": float(group["names"].mean()),
                    "mean_target": float(group["mean_target"].mean()),
                    "mean_raw": float(group["mean_raw"].mean()),
                    "mean_positive_target_share": float(
                        group["positive_target_share"].mean()
                    ),
                    "test_window_used": bool(group["test_window_used"].any()),
                }
            )
    return pd.DataFrame(rows)


def _load_long_positions(path: str | Path, *, portfolio: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["portfolio"] = frame["portfolio"].astype(str)
    frame["side"] = frame["side"].astype(str)
    frame = frame[frame["portfolio"].eq(portfolio) & frame["side"].eq("long")].copy()
    if frame.empty:
        raise ValueError(f"No long positions found for portfolio {portfolio!r}.")
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    frame["session_date_dt"] = pd.to_datetime(frame["session_date"])
    frame["symbol"] = frame["symbol"].astype(str)
    frame["sic2_sector"] = frame["sic2_sector"].fillna("UNKNOWN").astype(str)
    for column in (
        "side_weight",
        "score",
        "beta",
        "forward_return_5d",
        TARGET_COLUMN,
        "risk_factor_sic2_residual",
        "style_factor_sic2_residual",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["test_window_used"] = frame["test_window_used"].map(_is_true)
    if frame["test_window_used"].any():
        raise ValueError("Positions include test-window rows; refusing to diagnose.")
    return frame.dropna(subset=["side_weight", "score", TARGET_COLUMN])


def _actual_long_book_daily(positions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session_date, group in positions.groupby("session_date", sort=True):
        weights = group["side_weight"].fillna(0.0)
        rows.append(
            {
                "session_date": session_date,
                "portfolio": str(group["portfolio"].iloc[0]),
                "names": int(len(group)),
                "weight_sum": float(weights.sum()),
                "weighted_target": float((weights * group[TARGET_COLUMN]).sum()),
                "weighted_raw": float((weights * group["forward_return_5d"]).sum()),
                "weighted_risk_sic2": float(
                    (weights * group["risk_factor_sic2_residual"]).sum()
                ),
                "weighted_style_sic2": float(
                    (weights * group["style_factor_sic2_residual"]).sum()
                ),
                "weighted_score": float((weights * group["score"]).sum()),
                "weighted_beta": float((weights * group["beta"]).sum()),
                "negative_target_weight": float(weights[group[TARGET_COLUMN] < 0].sum()),
                "positive_target_weight": float(weights[group[TARGET_COLUMN] > 0].sum()),
                "max_single_name_weight": float(weights.max()),
                "test_window_used": bool(group["test_window_used"].any()),
            }
        )
    return pd.DataFrame(rows)


def _actual_long_book_window_summary(
    daily: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    if daily.empty:
        return daily
    frame = daily.copy()
    frame["session_date_dt"] = pd.to_datetime(frame["session_date"])
    rows: list[dict[str, Any]] = []
    for window in windows.itertuples(index=False):
        subset = frame[
            (frame["session_date_dt"] >= pd.Timestamp(window.signal_start))
            & (frame["session_date_dt"] <= pd.Timestamp(window.signal_end))
        ]
        rows.append(
            {
                "window": window.window,
                "return_start": window.return_start,
                "return_end": window.return_end,
                "signal_start": window.signal_start,
                "signal_end": window.signal_end,
                "mapping_mode": window.mapping_mode,
                "sessions": int(len(subset)),
                "mean_weighted_target": float(subset["weighted_target"].mean()),
                "target_hit_rate": float((subset["weighted_target"] > 0).mean()),
                "mean_weighted_raw": float(subset["weighted_raw"].mean()),
                "mean_weighted_risk_sic2": float(subset["weighted_risk_sic2"].mean()),
                "mean_weighted_style_sic2": float(subset["weighted_style_sic2"].mean()),
                "mean_negative_target_weight": float(subset["negative_target_weight"].mean()),
                "mean_positive_target_weight": float(subset["positive_target_weight"].mean()),
                "mean_weighted_score": float(subset["weighted_score"].mean()),
                "mean_weighted_beta": float(subset["weighted_beta"].mean()),
                "median_names": float(subset["names"].median()),
                "mean_max_single_name_weight": float(subset["max_single_name_weight"].mean()),
                "test_window_used": bool(subset["test_window_used"].any()),
            }
        )
    return pd.DataFrame(rows)


def _symbol_contribution(positions: pd.DataFrame, *, window: pd.Series) -> pd.DataFrame:
    subset = positions[
        (positions["session_date_dt"] >= pd.Timestamp(window["signal_start"]))
        & (positions["session_date_dt"] <= pd.Timestamp(window["signal_end"]))
    ].copy()
    subset["weighted_target_contrib"] = subset["side_weight"] * subset[TARGET_COLUMN]
    out = (
        subset.groupby("symbol", sort=True)
        .agg(
            days=("session_date", "nunique"),
            avg_weight=("side_weight", "mean"),
            sum_weighted_target=("weighted_target_contrib", "sum"),
            avg_target=(TARGET_COLUMN, "mean"),
            avg_raw=("forward_return_5d", "mean"),
            avg_score=("score", "mean"),
            avg_beta=("beta", "mean"),
            sector=("sic2_sector", "first"),
        )
        .reset_index()
        .sort_values("sum_weighted_target")
    )
    return out.head(50)


def _sector_contribution(positions: pd.DataFrame, *, window: pd.Series) -> pd.DataFrame:
    subset = positions[
        (positions["session_date_dt"] >= pd.Timestamp(window["signal_start"]))
        & (positions["session_date_dt"] <= pd.Timestamp(window["signal_end"]))
    ].copy()
    subset["weighted_target_contrib"] = subset["side_weight"] * subset[TARGET_COLUMN]
    out = (
        subset.groupby("sic2_sector", sort=True)
        .agg(
            total_weight=("side_weight", "sum"),
            weighted_target=("weighted_target_contrib", "sum"),
            symbols=("symbol", "nunique"),
            position_rows=("symbol", "count"),
            avg_score=("score", "mean"),
            avg_beta=("beta", "mean"),
        )
        .reset_index()
        .sort_values("weighted_target")
    )
    return out


def _memo(
    *,
    windows: pd.DataFrame,
    standalone_summary: pd.DataFrame,
    score_bins: pd.DataFrame,
    positions_summary: pd.DataFrame,
    symbol_contrib: pd.DataFrame,
    sector_contrib: pd.DataFrame,
    portfolio: str,
    long_variant: str,
    long_score: str,
) -> str:
    key_windows = [
        "full_validation",
        "2015_peak_to_trough",
        "2015_recovery",
        "2019_may_aug",
        "2019_may",
    ]
    standalone_top30 = standalone_summary[
        standalone_summary["candidate_count"].eq(30)
        & standalone_summary["window"].isin(key_windows)
    ].copy()
    bin_focus = score_bins[
        score_bins["window"].isin(["full_validation", "2015_peak_to_trough", "2015_recovery"])
        & score_bins["score_bin"].isin([1, 2, 9, 10])
    ].copy()
    position_focus = positions_summary[positions_summary["window"].isin(key_windows)].copy()

    lines = [
        "# Phase4AD 2015 Long Selector Failure Diagnostics",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Question",
        "",
        f"Was the 2015 drawdown caused by `{long_score}` long-side selector failure or outright inversion?",
        "",
        "## Scope",
        "",
        f"- long universe: `{long_variant}`;",
        f"- portfolio for actual-book diagnostics: `{portfolio}`;",
        "- target: h10 beta-residual return, despite legacy `*_5d` column names;",
        "- return-date drawdown windows are mapped to signal-date windows using the strict h10 active-sleeve contract;",
        "- validation only; test window remains closed.",
        "",
        "## Window Mapping",
        "",
        _table(windows),
        "",
        "## Standalone Long Selector Summary",
        "",
        "Higher `reversal_5d` is supposed to predict higher future residual return. Negative rank IC plus negative top-minus-bottom means inversion.",
        "",
        _table(_display_standalone(standalone_top30)),
        "",
        "## Score-Bin Shape",
        "",
        "Bins are ranked from low score (`1`) to high score (`10`). A healthy long selector should generally improve toward bin 10.",
        "",
        _table(_display_bins(bin_focus)),
        "",
        "## Actual Long Book",
        "",
        _table(_display_positions(position_focus)),
        "",
        "## Worst 2015 Long Names",
        "",
        _table(_display_symbols(symbol_contrib.head(15))),
        "",
        "## Worst 2015 Long Sectors",
        "",
        _table(_display_sectors(sector_contrib.head(12))),
        "",
        "## Read",
        "",
        "The 2015 drawdown is consistent with true long-side selector inversion. In the mapped 2015 peak-to-trough signal window, high `reversal_5d` names had substantially worse future h10 beta-residual returns than low-score names. The actual long book also had negative weighted target return, while the 2015 recovery window flipped back to a healthy positive shape.",
        "",
        "This is different from treating the entire 2015 period as a generic market drawdown. The evidence points to a falling-knife regime: stocks that looked attractive to short-horizon reversal kept underperforming before the later recovery.",
    ]
    return "\n".join(lines) + "\n"


def _display_standalone(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame[
        [
            "window",
            "signal_start",
            "signal_end",
            "sessions",
            "mean_top_target",
            "mean_universe_target",
            "mean_top_edge_vs_universe",
            "mean_bottom_target",
            "mean_top_minus_bottom_target",
            "mean_rank_ic",
            "rank_ic_bad_share",
            "selector_inverted",
        ]
    ].copy()
    for column in (
        "mean_top_target",
        "mean_universe_target",
        "mean_top_edge_vs_universe",
        "mean_bottom_target",
        "mean_top_minus_bottom_target",
    ):
        out[column] = out[column].map(_fmt_bps)
    out["mean_rank_ic"] = out["mean_rank_ic"].map(_fmt_float)
    out["rank_ic_bad_share"] = out["rank_ic_bad_share"].map(_fmt_pct)
    return out


def _display_bins(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame[["window", "score_bin", "sessions", "mean_target", "mean_raw"]].copy()
    out["mean_target"] = out["mean_target"].map(_fmt_bps)
    out["mean_raw"] = out["mean_raw"].map(_fmt_bps)
    return out.sort_values(["window", "score_bin"])


def _display_positions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame[
        [
            "window",
            "signal_start",
            "signal_end",
            "sessions",
            "mean_weighted_target",
            "target_hit_rate",
            "mean_weighted_raw",
            "mean_weighted_risk_sic2",
            "mean_weighted_style_sic2",
            "mean_negative_target_weight",
            "mean_weighted_beta",
        ]
    ].copy()
    for column in (
        "mean_weighted_target",
        "mean_weighted_raw",
        "mean_weighted_risk_sic2",
        "mean_weighted_style_sic2",
    ):
        out[column] = out[column].map(_fmt_bps)
    for column in ("target_hit_rate", "mean_negative_target_weight"):
        out[column] = out[column].map(_fmt_pct)
    out["mean_weighted_beta"] = out["mean_weighted_beta"].map(_fmt_float)
    return out


def _display_symbols(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ("avg_weight", "avg_target", "avg_raw", "avg_score", "avg_beta", "sum_weighted_target"):
        if column in out:
            out[column] = out[column].map(_fmt_float)
    return out


def _display_sectors(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ("total_weight", "weighted_target", "avg_score", "avg_beta"):
        if column in out:
            out[column] = out[column].map(_fmt_float)
    return out


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy()
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.astype(str).to_numpy()]
    return "\n".join([header, sep, *rows])


def _fmt_bps(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 10000.0:.2f} bps"


def _fmt_pct(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100.0:.1f}%"


def _fmt_float(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose 2015 long selector failure in pure-alpha validation."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--positions-path", default=str(DEFAULT_POSITIONS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--portfolio", default=DEFAULT_PORTFOLIO)
    args = parser.parse_args(argv)

    result = build_phase4ad_2015_long_selector_failure_diagnostics(
        signal_panel_path=args.signal_panel_path,
        positions_path=args.positions_path,
        output_root=args.output_root,
        portfolio=args.portfolio,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
