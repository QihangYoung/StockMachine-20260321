from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4z import _active_return_dates
from stockmachine.apps.run_pure_alpha_phase5b import _fmt_float_like, _fmt_pct_like, _text_table
from stockmachine.apps.run_pure_alpha_phase6 import (
    DEFAULT_DAILY_DIAGNOSTICS_PATH,
    DEFAULT_HOLDING_PERIOD_SESSIONS,
    DEFAULT_MODEL_NAME,
    DEFAULT_PORTFOLIO,
    DEFAULT_POSITIONS_PATH,
    DEFAULT_STRICT_CURVE_PATH,
    _build_backtest_records,
    _build_daily_aggregate_positions,
    _build_rolling_beta,
    _daily_path_metrics,
    _is_true,
    _load_daily_diagnostics,
    _load_full_positions,
    _load_strict_curve,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase6b_realized_beta_diagnostics_20260507"
DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (60, 126, 252)
DEFAULT_HIGH_ABS_BETA_THRESHOLD = 0.20


def build_phase6b_realized_beta_diagnostics(
    *,
    positions_path: str | Path = DEFAULT_POSITIONS_PATH,
    daily_diagnostics_path: str | Path = DEFAULT_DAILY_DIAGNOSTICS_PATH,
    strict_curve_path: str | Path = DEFAULT_STRICT_CURVE_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    portfolio: str = DEFAULT_PORTFOLIO,
    model_name: str = DEFAULT_MODEL_NAME,
    holding_period_sessions: int = DEFAULT_HOLDING_PERIOD_SESSIONS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    high_abs_beta_threshold: float = DEFAULT_HIGH_ABS_BETA_THRESHOLD,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    curve = _load_strict_curve(strict_curve_path, portfolio=portfolio)
    diagnostics = _load_daily_diagnostics(daily_diagnostics_path, portfolio=portfolio)
    positions = _load_full_positions(positions_path, portfolio=portfolio)
    calendar = curve["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
    aggregate_positions, daily_exposure = _build_daily_aggregate_positions(
        positions,
        benchmark_calendar=calendar,
        holding_period_sessions=holding_period_sessions,
    )
    beta_exposure = _build_daily_ex_ante_beta_exposure(
        positions,
        benchmark_calendar=calendar,
        holding_period_sessions=holding_period_sessions,
    )
    records = _build_backtest_records(
        curve=curve,
        turnover=pd.DataFrame(
            {
                "return_date": curve["return_date"],
                "h10_turnover": np.nan,
                "h10_positions": np.nan,
            }
        ),
        daily_exposure=daily_exposure,
    ).merge(beta_exposure, left_on="entry_date", right_on="return_date", how="left")
    records = records.drop(columns=["return_date"], errors="ignore")
    records = _add_path_columns(records)

    rolling_beta = _build_rolling_beta(records, windows=rolling_windows)
    rolling_enriched = _enrich_rolling_beta(records, rolling_beta)
    high_beta_windows = _build_high_beta_windows(
        records,
        rolling_enriched,
        threshold=high_abs_beta_threshold,
    )
    leg_beta = _build_leg_realized_beta(curve=curve, records=records, threshold=high_abs_beta_threshold)
    state_beta = _build_market_state_beta(curve=curve, records=records)
    exante_alignment = _build_exante_realized_alignment(records=records, rolling=rolling_enriched)
    symbol_concentration = _build_high_beta_symbol_concentration(
        records=records,
        aggregate_positions=aggregate_positions,
        rolling=rolling_enriched,
        threshold=high_abs_beta_threshold,
    )
    sector_exposure = _build_high_beta_sector_exposure(
        records=records,
        aggregate_positions=aggregate_positions,
        rolling=rolling_enriched,
        threshold=high_abs_beta_threshold,
    )
    diagnostic_summary = _build_diagnostic_summary(
        records=records,
        rolling=rolling_enriched,
        high_beta_windows=high_beta_windows,
        leg_beta=leg_beta,
        diagnostics=diagnostics,
        threshold=high_abs_beta_threshold,
    )

    rolling_path = output_dir / "phase6b_rolling_beta_enriched.csv"
    high_windows_path = output_dir / "phase6b_high_beta_windows.csv"
    leg_beta_path = output_dir / "phase6b_leg_realized_beta.csv"
    state_beta_path = output_dir / "phase6b_market_state_leg_beta.csv"
    exante_path = output_dir / "phase6b_exante_vs_realized_beta.csv"
    symbol_path = output_dir / "phase6b_high_beta_symbol_concentration.csv"
    sector_path = output_dir / "phase6b_high_beta_sector_exposure.csv"
    summary_path = output_dir / "phase6b_diagnostic_summary.csv"
    plot_path = output_dir / "phase6b_realized_beta_diagnostics_plot.png"
    memo_path = output_dir / "phase6b_realized_beta_diagnostics_memo.md"
    rollup_path = output_dir / "phase6b_rollup.json"

    rolling_enriched.to_csv(rolling_path, index=False)
    high_beta_windows.to_csv(high_windows_path, index=False)
    leg_beta.to_csv(leg_beta_path, index=False)
    state_beta.to_csv(state_beta_path, index=False)
    exante_alignment.to_csv(exante_path, index=False)
    symbol_concentration.to_csv(symbol_path, index=False)
    sector_exposure.to_csv(sector_path, index=False)
    diagnostic_summary.to_csv(summary_path, index=False)
    _plot_diagnostics(records=records, rolling=rolling_enriched, high_windows=high_beta_windows, output_path=plot_path)
    memo_path.write_text(
        _memo(
            diagnostic_summary=diagnostic_summary,
            high_beta_windows=high_beta_windows,
            leg_beta=leg_beta,
            state_beta=state_beta,
            exante_alignment=exante_alignment,
            symbol_concentration=symbol_concentration,
            sector_exposure=sector_exposure,
            plot_path=plot_path,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "portfolio": portfolio,
        "model_name": model_name,
        "positions_path": Path(positions_path).as_posix(),
        "daily_diagnostics_path": Path(daily_diagnostics_path).as_posix(),
        "strict_curve_path": Path(strict_curve_path).as_posix(),
        "holding_period_sessions": int(holding_period_sessions),
        "rolling_windows": [int(value) for value in rolling_windows],
        "high_abs_beta_threshold": float(high_abs_beta_threshold),
        "rolling_beta_artifact": rolling_path.as_posix(),
        "high_beta_windows_artifact": high_windows_path.as_posix(),
        "leg_realized_beta_artifact": leg_beta_path.as_posix(),
        "market_state_leg_beta_artifact": state_beta_path.as_posix(),
        "exante_vs_realized_beta_artifact": exante_path.as_posix(),
        "high_beta_symbol_concentration_artifact": symbol_path.as_posix(),
        "high_beta_sector_exposure_artifact": sector_path.as_posix(),
        "diagnostic_summary_artifact": summary_path.as_posix(),
        "plot_artifact": plot_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "method": "phase6b_realized_beta_follow_up",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "This is diagnosis only; it does not change SOTA positions.",
            "Ex-ante beta exposure uses the beta column stored in target positions.",
            "Sector exposure is exposure-based, not sector return attribution.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _build_daily_ex_ante_beta_exposure(
    positions: pd.DataFrame,
    *,
    benchmark_calendar: pd.Series,
    holding_period_sessions: int,
) -> pd.DataFrame:
    active_map = _active_return_dates(
        positions,
        calendar=benchmark_calendar,
        holding_period_sessions=holding_period_sessions,
    )
    active = positions.merge(active_map, on="session_date", how="inner")
    active_sleeves = (
        active.groupby("return_date", as_index=False)["session_date"]
        .nunique()
        .rename(columns={"session_date": "active_sleeves"})
    )
    active = active.merge(active_sleeves, on="return_date", how="left")
    active["portfolio_weight"] = active["signed_weight"].astype(float) / active["active_sleeves"].astype(float)
    active["beta"] = pd.to_numeric(active["beta"], errors="coerce")
    active["beta_contribution"] = active["portfolio_weight"] * active["beta"]
    active["long_beta_contribution"] = np.where(
        active["portfolio_weight"] > 0,
        active["beta_contribution"],
        0.0,
    )
    active["short_beta_contribution"] = np.where(
        active["portfolio_weight"] < 0,
        active["beta_contribution"],
        0.0,
    )
    daily = (
        active.groupby("return_date", as_index=False)
        .agg(
            exante_net_beta=("beta_contribution", "sum"),
            exante_long_beta=("long_beta_contribution", "sum"),
            exante_short_beta=("short_beta_contribution", "sum"),
            exante_abs_beta_contribution=("beta_contribution", lambda s: float(np.abs(s).sum())),
            mean_active_sleeves=("active_sleeves", "first"),
        )
        .sort_values("return_date")
        .reset_index(drop=True)
    )
    daily["exante_beta_match_error"] = daily["exante_long_beta"] + daily["exante_short_beta"]
    return daily


def _add_path_columns(records: pd.DataFrame) -> pd.DataFrame:
    frame = records.sort_values("entry_date").reset_index(drop=True).copy()
    returns = frame["gross_return"].astype(float)
    benchmark = frame["benchmark_return"].astype(float)
    frame["equity"] = (1.0 + returns).cumprod()
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
    frame["spy_equity"] = (1.0 + benchmark).cumprod()
    frame["spy_drawdown"] = frame["spy_equity"] / frame["spy_equity"].cummax() - 1.0
    for window in (10, 20, 60, 126, 252):
        frame[f"rolling_{window}_strategy_return"] = (
            (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0
        )
        frame[f"rolling_{window}_spy_return"] = (
            (1.0 + benchmark).rolling(window).apply(np.prod, raw=True) - 1.0
        )
    return frame


def _enrich_rolling_beta(records: pd.DataFrame, rolling_beta: pd.DataFrame) -> pd.DataFrame:
    base = records[
        [
            "entry_date",
            "gross_return",
            "benchmark_return",
            "long_gross_return",
            "short_gross_return",
            "drawdown",
            "spy_drawdown",
            "exante_net_beta",
            "exante_long_beta",
            "exante_short_beta",
            "exante_abs_beta_contribution",
            "gross_exposure",
        ]
    ].copy()
    base["entry_date"] = pd.to_datetime(base["entry_date"])
    frame = rolling_beta.merge(base, on="entry_date", how="left")
    for window in sorted(frame["window"].dropna().unique()):
        mask = frame["window"].eq(window)
        w = int(window)
        frame.loc[mask, "rolling_strategy_return"] = (
            (1.0 + frame.loc[mask, "gross_return"].astype(float)).rolling(w).apply(np.prod, raw=True) - 1.0
        )
        frame.loc[mask, "rolling_spy_return"] = (
            (1.0 + frame.loc[mask, "benchmark_return"].astype(float)).rolling(w).apply(np.prod, raw=True) - 1.0
        )
        frame.loc[mask, "rolling_long_return"] = (
            (1.0 + frame.loc[mask, "long_gross_return"].astype(float)).rolling(w).apply(np.prod, raw=True) - 1.0
        )
        frame.loc[mask, "rolling_short_return"] = (
            (1.0 + frame.loc[mask, "short_gross_return"].astype(float)).rolling(w).apply(np.prod, raw=True) - 1.0
        )
        frame.loc[mask, "rolling_mean_exante_net_beta"] = (
            frame.loc[mask, "exante_net_beta"].astype(float).rolling(w).mean()
        )
        frame.loc[mask, "rolling_mean_gross_exposure"] = (
            frame.loc[mask, "gross_exposure"].astype(float).rolling(w).mean()
        )
    frame["abs_rolling_beta"] = frame["rolling_beta"].abs()
    frame["beta_sign"] = np.where(frame["rolling_beta"] >= 0, "positive", "negative")
    return frame.sort_values(["window", "entry_date"]).reset_index(drop=True)


def _build_high_beta_windows(
    records: pd.DataFrame,
    rolling: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    focus = rolling[rolling["window"].eq(60)].dropna(subset=["rolling_beta"]).copy()
    focus["is_high_abs_beta"] = focus["abs_rolling_beta"] >= float(threshold)
    focus = focus.sort_values("entry_date").reset_index(drop=True)
    groups: list[pd.DataFrame] = []
    current_group = -1
    previous_high = False
    for idx, is_high in enumerate(focus["is_high_abs_beta"].tolist()):
        if is_high and not previous_high:
            current_group += 1
        groups.append(current_group if is_high else np.nan)
        previous_high = bool(is_high)
    focus["cluster_id"] = groups
    rows: list[dict[str, Any]] = []
    record_frame = records.copy()
    record_frame["entry_date"] = pd.to_datetime(record_frame["entry_date"])
    for cluster_id, group in focus.dropna(subset=["cluster_id"]).groupby("cluster_id", sort=True):
        start = pd.Timestamp(group["entry_date"].min())
        end = pd.Timestamp(group["entry_date"].max())
        sub = record_frame[(record_frame["entry_date"] >= start) & (record_frame["entry_date"] <= end)].copy()
        if sub.empty:
            continue
        metrics = _daily_path_metrics(sub["gross_return"], benchmark=sub["benchmark_return"])
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "days": int(len(sub)),
                "mean_rolling_beta": float(group["rolling_beta"].mean()),
                "max_abs_rolling_beta": float(group["abs_rolling_beta"].max()),
                "mean_rolling_corr": float(group["rolling_corr"].mean()),
                "strategy_total_return": metrics["total_return"],
                "spy_total_return": float((1.0 + sub["benchmark_return"].astype(float)).prod() - 1.0),
                "long_total_return": float((1.0 + sub["long_gross_return"].astype(float)).prod() - 1.0),
                "short_total_return": float((1.0 + sub["short_gross_return"].astype(float)).prod() - 1.0),
                "mean_daily_return_bps": float(sub["gross_return"].astype(float).mean() * 10000.0),
                "mean_spy_bps": float(sub["benchmark_return"].astype(float).mean() * 10000.0),
                "mean_exante_net_beta": float(sub["exante_net_beta"].astype(float).mean()),
                "mean_abs_exante_net_beta": float(sub["exante_net_beta"].astype(float).abs().mean()),
                "mean_gross_exposure": float(sub["gross_exposure"].astype(float).mean()),
                "min_drawdown_in_window": float(sub["drawdown"].min()),
                "dominant_beta_sign": "positive" if group["rolling_beta"].mean() >= 0 else "negative",
            }
        )
    return pd.DataFrame(rows)


def _build_leg_realized_beta(
    *,
    curve: pd.DataFrame,
    records: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    frame = curve.merge(
        records[["entry_date", "benchmark_return"]].rename(columns={"entry_date": "return_date"}),
        on="return_date",
        how="left",
        suffixes=("", "_records"),
    )
    frame["return_date"] = pd.to_datetime(frame["return_date"])
    rolling60 = _build_rolling_beta(records, windows=(60,))
    high_dates = set(
        rolling60.loc[rolling60["rolling_beta"].abs() >= threshold, "entry_date"].dt.normalize().tolist()
    )
    spy_top = float(frame["benchmark_oto_return"].quantile(0.90))
    spy_bottom = float(frame["benchmark_oto_return"].quantile(0.10))
    states = [
        ("all", pd.Series(True, index=frame.index)),
        ("high_abs_rolling60_beta", frame["return_date"].dt.normalize().isin(high_dates)),
        ("normal_abs_rolling60_beta", ~frame["return_date"].dt.normalize().isin(high_dates)),
        ("spy_up", frame["benchmark_oto_return"] > 0),
        ("spy_down", frame["benchmark_oto_return"] < 0),
        ("spy_top_decile", frame["benchmark_oto_return"] >= spy_top),
        ("spy_bottom_decile", frame["benchmark_oto_return"] <= spy_bottom),
    ]
    legs = [
        ("spread", "gross_return"),
        ("long", "long_gross_return"),
        ("short", "short_gross_return"),
    ]
    rows: list[dict[str, Any]] = []
    for state, mask in states:
        sub = frame.loc[mask].copy()
        if sub.empty:
            continue
        benchmark = sub["benchmark_oto_return"].astype(float)
        for leg, column in legs:
            ret = sub[column].astype(float)
            rows.append(_beta_row(state=state, leg=leg, returns=ret, benchmark=benchmark))
    return pd.DataFrame(rows)


def _build_market_state_beta(*, curve: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    frame = curve.copy()
    frame["return_date"] = pd.to_datetime(frame["return_date"])
    frame["spy_abs_return"] = frame["benchmark_oto_return"].abs()
    frame["spy_vol20"] = frame["benchmark_oto_return"].rolling(20).std()
    high_vol_threshold = float(frame["spy_vol20"].quantile(0.75))
    states = [
        ("all", pd.Series(True, index=frame.index)),
        ("spy_up_low_vol", (frame["benchmark_oto_return"] > 0) & (frame["spy_vol20"] <= high_vol_threshold)),
        ("spy_up_high_vol", (frame["benchmark_oto_return"] > 0) & (frame["spy_vol20"] > high_vol_threshold)),
        ("spy_down_low_vol", (frame["benchmark_oto_return"] < 0) & (frame["spy_vol20"] <= high_vol_threshold)),
        ("spy_down_high_vol", (frame["benchmark_oto_return"] < 0) & (frame["spy_vol20"] > high_vol_threshold)),
        ("spy_large_move", frame["spy_abs_return"] >= float(frame["spy_abs_return"].quantile(0.90))),
    ]
    rows: list[dict[str, Any]] = []
    for state, mask in states:
        sub = frame.loc[mask].copy()
        if sub.empty:
            continue
        benchmark = sub["benchmark_oto_return"].astype(float)
        for leg, column in (
            ("spread", "gross_return"),
            ("long", "long_gross_return"),
            ("short", "short_gross_return"),
        ):
            rows.append(_beta_row(state=state, leg=leg, returns=sub[column].astype(float), benchmark=benchmark))
    return pd.DataFrame(rows)


def _beta_row(*, state: str, leg: str, returns: pd.Series, benchmark: pd.Series) -> dict[str, Any]:
    aligned = pd.concat([returns.rename("returns"), benchmark.rename("benchmark")], axis=1).dropna()
    beta = np.nan
    corr = np.nan
    if len(aligned) > 2 and aligned["benchmark"].var(ddof=1) > 0:
        beta = float(aligned["returns"].cov(aligned["benchmark"]) / aligned["benchmark"].var(ddof=1))
        corr = float(aligned["returns"].corr(aligned["benchmark"]))
    return {
        "state": state,
        "leg": leg,
        "days": int(len(aligned)),
        "mean_return_bps": float(aligned["returns"].mean() * 10000.0) if not aligned.empty else np.nan,
        "hit_rate": float((aligned["returns"] > 0).mean()) if not aligned.empty else np.nan,
        "realized_beta_to_spy": beta,
        "corr_to_spy": corr,
        "return_vol_ann": float(aligned["returns"].std(ddof=1) * np.sqrt(252.0)) if len(aligned) > 1 else np.nan,
    }


def _build_exante_realized_alignment(*, records: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    focus = rolling[rolling["window"].isin([60, 126, 252])].copy()
    rows: list[dict[str, Any]] = []
    for window, group in focus.dropna(subset=["rolling_beta"]).groupby("window", sort=True):
        aligned = group.dropna(subset=["rolling_beta", "rolling_mean_exante_net_beta"]).copy()
        if aligned.empty:
            continue
        rows.append(
            {
                "window": int(window),
                "observations": int(len(aligned)),
                "mean_rolling_realized_beta": float(aligned["rolling_beta"].mean()),
                "mean_abs_rolling_realized_beta": float(aligned["rolling_beta"].abs().mean()),
                "p90_abs_rolling_realized_beta": float(aligned["rolling_beta"].abs().quantile(0.90)),
                "mean_rolling_exante_net_beta": float(aligned["rolling_mean_exante_net_beta"].mean()),
                "mean_abs_rolling_exante_net_beta": float(aligned["rolling_mean_exante_net_beta"].abs().mean()),
                "corr_realized_beta_vs_exante_net_beta": float(
                    aligned["rolling_beta"].corr(aligned["rolling_mean_exante_net_beta"])
                )
                if aligned["rolling_mean_exante_net_beta"].std(ddof=1) > 0
                else np.nan,
                "corr_realized_beta_vs_spy_return": float(
                    aligned["rolling_beta"].corr(aligned["rolling_spy_return"])
                )
                if aligned["rolling_spy_return"].std(ddof=1) > 0
                else np.nan,
                "corr_abs_realized_beta_vs_gross_exposure": float(
                    aligned["rolling_beta"].abs().corr(aligned["rolling_mean_gross_exposure"])
                )
                if aligned["rolling_mean_gross_exposure"].std(ddof=1) > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_high_beta_symbol_concentration(
    *,
    records: pd.DataFrame,
    aggregate_positions: pd.DataFrame,
    rolling: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    high_dates = _high_beta_dates(rolling, threshold=threshold)
    aggregate = aggregate_positions.copy()
    aggregate["return_date"] = pd.to_datetime(aggregate["return_date"])
    aggregate["high_abs_rolling60_beta"] = aggregate["return_date"].dt.normalize().isin(high_dates)
    rows: list[pd.DataFrame] = []
    for label, mask in (
        ("high_abs_rolling60_beta", aggregate["high_abs_rolling60_beta"]),
        ("normal_abs_rolling60_beta", ~aggregate["high_abs_rolling60_beta"]),
    ):
        sub = aggregate.loc[mask].copy()
        if sub.empty:
            continue
        summary = (
            sub.groupby("symbol", as_index=False)
            .agg(
                mean_weight=("portfolio_weight", "mean"),
                mean_abs_weight=("portfolio_weight", lambda s: float(np.abs(s).mean())),
                days=("return_date", "nunique"),
                sic2_sector=("sic2_sector", "first"),
            )
            .sort_values("mean_abs_weight", ascending=False)
            .head(30)
        )
        summary.insert(0, "state", label)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_high_beta_sector_exposure(
    *,
    records: pd.DataFrame,
    aggregate_positions: pd.DataFrame,
    rolling: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    high_dates = _high_beta_dates(rolling, threshold=threshold)
    aggregate = aggregate_positions.copy()
    aggregate["return_date"] = pd.to_datetime(aggregate["return_date"])
    aggregate["state"] = np.where(
        aggregate["return_date"].dt.normalize().isin(high_dates),
        "high_abs_rolling60_beta",
        "normal_abs_rolling60_beta",
    )
    sector = (
        aggregate.groupby(["state", "return_date", "sic2_sector"], as_index=False)["portfolio_weight"]
        .sum()
        .rename(columns={"portfolio_weight": "net_sector_weight"})
    )
    summary = (
        sector.groupby(["state", "sic2_sector"], as_index=False)
        .agg(
            mean_net_sector_weight=("net_sector_weight", "mean"),
            mean_abs_sector_weight=("net_sector_weight", lambda s: float(np.abs(s).mean())),
            max_abs_sector_weight=("net_sector_weight", lambda s: float(np.abs(s).max())),
            days=("return_date", "nunique"),
        )
        .sort_values(["state", "mean_abs_sector_weight"], ascending=[True, False])
    )
    return summary.groupby("state", group_keys=False).head(15).reset_index(drop=True)


def _high_beta_dates(rolling: pd.DataFrame, *, threshold: float) -> set[pd.Timestamp]:
    focus = rolling[
        rolling["window"].eq(60)
        & rolling["rolling_beta"].notna()
        & (rolling["rolling_beta"].abs() >= float(threshold))
    ].copy()
    return set(pd.to_datetime(focus["entry_date"]).dt.normalize().tolist())


def _build_diagnostic_summary(
    *,
    records: pd.DataFrame,
    rolling: pd.DataFrame,
    high_beta_windows: pd.DataFrame,
    leg_beta: pd.DataFrame,
    diagnostics: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    all_spread = leg_beta[(leg_beta["state"].eq("all")) & (leg_beta["leg"].eq("spread"))]
    all_long = leg_beta[(leg_beta["state"].eq("all")) & (leg_beta["leg"].eq("long"))]
    all_short = leg_beta[(leg_beta["state"].eq("all")) & (leg_beta["leg"].eq("short"))]
    high_spread = leg_beta[(leg_beta["state"].eq("high_abs_rolling60_beta")) & (leg_beta["leg"].eq("spread"))]
    high_long = leg_beta[(leg_beta["state"].eq("high_abs_rolling60_beta")) & (leg_beta["leg"].eq("long"))]
    high_short = leg_beta[(leg_beta["state"].eq("high_abs_rolling60_beta")) & (leg_beta["leg"].eq("short"))]
    rolling60 = rolling[rolling["window"].eq(60)].dropna(subset=["rolling_beta"])
    rows = [
        {
            "metric": "full_sample_spread_realized_beta",
            "value": _first_float(all_spread, "realized_beta_to_spy"),
            "reading": "slightly_above_0p05_gate",
        },
        {
            "metric": "full_sample_long_realized_beta",
            "value": _first_float(all_long, "realized_beta_to_spy"),
            "reading": "long_leg_primary_positive_market_coupling",
        },
        {
            "metric": "full_sample_short_realized_beta",
            "value": _first_float(all_short, "realized_beta_to_spy"),
            "reading": "short_leg_offsets_part_of_long_market_coupling",
        },
        {
            "metric": "high_abs_rolling60_beta_day_share",
            "value": float((rolling60["rolling_beta"].abs() >= float(threshold)).mean())
            if not rolling60.empty
            else np.nan,
            "reading": "realized_beta_tail_frequency",
        },
        {
            "metric": "high_abs_rolling60_beta_clusters",
            "value": float(len(high_beta_windows)),
            "reading": "contiguous_risk_windows",
        },
        {
            "metric": "high_beta_spread_realized_beta",
            "value": _first_float(high_spread, "realized_beta_to_spy"),
            "reading": "conditional_beta_inside_risk_windows",
        },
        {
            "metric": "high_beta_long_realized_beta",
            "value": _first_float(high_long, "realized_beta_to_spy"),
            "reading": "conditional_long_beta_inside_risk_windows",
        },
        {
            "metric": "high_beta_short_realized_beta",
            "value": _first_float(high_short, "realized_beta_to_spy"),
            "reading": "conditional_short_beta_inside_risk_windows",
        },
        {
            "metric": "mean_abs_exante_net_beta_daily",
            "value": float(records["exante_net_beta"].astype(float).abs().mean()),
            "reading": "target_beta_matching_is_not_the_main_cause",
        },
        {
            "metric": "mean_exante_long_beta_daily",
            "value": float(records["exante_long_beta"].astype(float).mean()),
            "reading": "average_target_long_beta",
        },
        {
            "metric": "mean_exante_short_beta_daily",
            "value": float(records["exante_short_beta"].astype(float).mean()),
            "reading": "average_target_short_beta",
        },
        {
            "metric": "mean_decision_exante_net_beta",
            "value": float(diagnostics["net_beta"].astype(float).abs().mean()) if "net_beta" in diagnostics else np.nan,
            "reading": "optimizer_exante_beta_match_error",
        },
    ]
    return pd.DataFrame(rows)


def _first_float(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    value = frame[column].iloc[0]
    return float(value) if pd.notna(value) else np.nan


def _plot_diagnostics(
    *,
    records: pd.DataFrame,
    rolling: pd.DataFrame,
    high_windows: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = records.copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    rb60 = rolling[rolling["window"].eq(60)].copy()
    rb126 = rolling[rolling["window"].eq(126)].copy()
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(frame["entry_date"], frame["equity"], label="SOTA equity", color="#0f766e", linewidth=2.0)
    axes[0].plot(frame["entry_date"], frame["spy_equity"], label="SPY raw", color="#6b7280", alpha=0.75)
    axes[0].legend(loc="upper left")
    axes[0].set_ylabel("Growth")
    axes[1].plot(rb60["entry_date"], rb60["rolling_beta"], label="60d realized beta", color="#dc2626")
    axes[1].plot(rb126["entry_date"], rb126["rolling_beta"], label="126d realized beta", color="#2563eb")
    for y in (0.0, 0.05, -0.05, 0.20, -0.20):
        axes[1].axhline(y, color="gray" if y else "black", linestyle=":" if y else "--", linewidth=0.8)
    axes[1].legend(loc="upper left")
    axes[1].set_ylabel("Beta")
    axes[2].plot(frame["entry_date"], frame["long_gross_return"].rolling(20).mean() * 10000.0, label="long 20d avg bps")
    axes[2].plot(frame["entry_date"], frame["short_gross_return"].rolling(20).mean() * 10000.0, label="short 20d avg bps")
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[2].legend(loc="upper left")
    axes[2].set_ylabel("20d avg bps")
    axes[3].plot(frame["entry_date"], frame["exante_net_beta"], label="daily ex-ante net beta", color="#7c3aed")
    axes[3].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[3].set_ylabel("Ex-ante beta")
    axes[3].set_xlabel("Date")
    axes[3].legend(loc="upper left")
    for _, row in high_windows.iterrows():
        start = pd.Timestamp(row["start"])
        end = pd.Timestamp(row["end"])
        for ax in axes:
            ax.axvspan(start, end, color="#f97316", alpha=0.12)
    fig.suptitle("Phase6B Realized Beta Diagnostics (Validation Only)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _memo(
    *,
    diagnostic_summary: pd.DataFrame,
    high_beta_windows: pd.DataFrame,
    leg_beta: pd.DataFrame,
    state_beta: pd.DataFrame,
    exante_alignment: pd.DataFrame,
    symbol_concentration: pd.DataFrame,
    sector_exposure: pd.DataFrame,
    plot_path: Path,
) -> str:
    summary = diagnostic_summary.copy()
    summary["value"] = summary["value"].map(_fmt_float_like)

    windows = high_beta_windows.copy()
    for column in (
        "strategy_total_return",
        "spy_total_return",
        "long_total_return",
        "short_total_return",
        "min_drawdown_in_window",
    ):
        if column in windows:
            windows[column] = windows[column].map(_fmt_pct_like)
    for column in (
        "mean_rolling_beta",
        "max_abs_rolling_beta",
        "mean_rolling_corr",
        "mean_daily_return_bps",
        "mean_spy_bps",
        "mean_exante_net_beta",
        "mean_abs_exante_net_beta",
        "mean_gross_exposure",
    ):
        if column in windows:
            windows[column] = windows[column].map(_fmt_float_like)

    legs = leg_beta[leg_beta["state"].isin(["all", "high_abs_rolling60_beta", "normal_abs_rolling60_beta"])].copy()
    for column in ("mean_return_bps", "realized_beta_to_spy", "corr_to_spy", "return_vol_ann"):
        legs[column] = legs[column].map(_fmt_float_like)
    legs["hit_rate"] = legs["hit_rate"].map(_fmt_pct_like)

    states = state_beta.copy()
    for column in ("mean_return_bps", "realized_beta_to_spy", "corr_to_spy", "return_vol_ann"):
        states[column] = states[column].map(_fmt_float_like)
    states["hit_rate"] = states["hit_rate"].map(_fmt_pct_like)

    exante = exante_alignment.copy()
    for column in exante.columns:
        if column != "window":
            exante[column] = exante[column].map(_fmt_float_like)

    symbols = symbol_concentration.copy()
    for column in ("mean_weight", "mean_abs_weight"):
        if column in symbols:
            symbols[column] = symbols[column].map(_fmt_float_like)

    sectors = sector_exposure.copy()
    for column in ("mean_net_sector_weight", "mean_abs_sector_weight", "max_abs_sector_weight"):
        if column in sectors:
            sectors[column] = sectors[column].map(_fmt_float_like)

    lines = [
        "# Phase6B Realized Beta Follow-Up",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "- Candidate: current SOTA / Phase5F `short_overlay_only`.",
        "- Purpose: diagnose the realized beta red flag from Phase6.",
        "- This is validation-only and does not alter strategy positions.",
        "",
        "## Diagnostic Summary",
        "",
        _text_table(summary),
        "",
        "## High 60-Day Realized Beta Windows",
        "",
        _text_table(windows),
        "",
        "## Leg Realized Beta",
        "",
        _text_table(legs),
        "",
        "## Market-State Leg Beta",
        "",
        _text_table(states),
        "",
        "## Ex-Ante Vs Realized Beta Alignment",
        "",
        _text_table(exante),
        "",
        "## High-Beta Symbol Concentration",
        "",
        _text_table(symbols.head(20)),
        "",
        "## High-Beta SIC2 Exposure",
        "",
        _text_table(sectors.head(20)),
        "",
        "## Plot",
        "",
        f"![Phase6B realized beta diagnostics]({plot_path.as_posix()})",
        "",
        "## First Reading",
        "",
        "- Ex-ante beta matching is essentially exact, so the realized beta issue is mainly a realized co-movement / state problem, not a construction math bug.",
        "- The positive realized beta is mostly carried by the long leg; the short leg offsets part of it but not all of it.",
        "- The next useful step is to inspect the high-beta windows and decide whether to accept the small beta drift, add a realized-beta monitor, or test a lightweight ex-post beta hedge overlay before Phase7 freeze.",
    ]
    return "\n".join(lines) + "\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose realized beta for pure-alpha SOTA.")
    parser.add_argument("--positions-path", default=str(DEFAULT_POSITIONS_PATH))
    parser.add_argument("--daily-diagnostics-path", default=str(DEFAULT_DAILY_DIAGNOSTICS_PATH))
    parser.add_argument("--strict-curve-path", default=str(DEFAULT_STRICT_CURVE_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--portfolio", default=DEFAULT_PORTFOLIO)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--holding-period-sessions", type=int, default=DEFAULT_HOLDING_PERIOD_SESSIONS)
    parser.add_argument("--high-abs-beta-threshold", type=float, default=DEFAULT_HIGH_ABS_BETA_THRESHOLD)
    args = parser.parse_args(argv)

    result = build_phase6b_realized_beta_diagnostics(
        positions_path=args.positions_path,
        daily_diagnostics_path=args.daily_diagnostics_path,
        strict_curve_path=args.strict_curve_path,
        output_root=args.output_root,
        portfolio=args.portfolio,
        model_name=args.model_name,
        holding_period_sessions=args.holding_period_sessions,
        high_abs_beta_threshold=args.high_abs_beta_threshold,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
