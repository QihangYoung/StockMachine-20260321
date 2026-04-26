from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase3 import (
    DEFAULT_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_DAILY_GLOBS,
    DEFAULT_DAILY_GLOBS,
)
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4s import DEFAULT_H10_SIGNAL_PANEL
from stockmachine.apps.run_pure_alpha_phase4z import (
    DEFAULT_VALIDATION_PRICE_END,
    _load_open_to_open_returns,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5a_daily_rescoring_sticky_20260426"
DEFAULT_TARGET_POSITIONS_PATH = (
    RESEARCH_ROOT
    / "phase4ag_dual_overlay_combo_20260425"
    / "phase4ag_beta_matched_positions_validation.csv.gz"
)
DEFAULT_H10_COMPARE_CURVE_PATH = (
    RESEARCH_ROOT
    / "phase4ag_dual_overlay_combo_20260425"
    / "phase4ag_path_compare_curve.csv"
)
DEFAULT_H10_COMPARE_METRICS_PATH = (
    RESEARCH_ROOT
    / "phase4ag_dual_overlay_combo_20260425"
    / "phase4ag_path_compare_metrics.csv"
)
DEFAULT_TARGET_PORTFOLIO = "sic2_soft_neutral__short_hybrid_soft_fw_overlay"
DEFAULT_BASELINE_PORTFOLIO = "sic2_soft_neutral"
DEFAULT_ASSUMED_COST_BPS_PER_SIDE = 4.0


@dataclass(frozen=True, slots=True)
class StickyConfig:
    label: str
    min_hold_sessions: int
    no_trade_band: float
    max_turnover: float | None
    min_weight_change: float
    max_new_names_per_rebalance: int | None


DAILY_CONFIGS = (
    StickyConfig(
        label="daily_full_retarget",
        min_hold_sessions=1,
        no_trade_band=0.0,
        max_turnover=None,
        min_weight_change=0.0,
        max_new_names_per_rebalance=None,
    ),
    StickyConfig(
        label="daily_minhold3_sticky",
        min_hold_sessions=3,
        no_trade_band=0.05,
        max_turnover=1.00,
        min_weight_change=0.010,
        max_new_names_per_rebalance=12,
    ),
)
PLOT_COLORS = {
    "h10_baseline": "#0f766e",
    "h10_short_hybrid": "#2563eb",
    "daily_full_retarget": "#a855f7",
    "daily_minhold3_sticky": "#c2410c",
    "spy_raw": "#6b7280",
}


def build_phase5a_daily_rescoring_sticky_artifacts(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    target_positions_path: str | Path = DEFAULT_TARGET_POSITIONS_PATH,
    h10_compare_curve_path: str | Path = DEFAULT_H10_COMPARE_CURVE_PATH,
    h10_compare_metrics_path: str | Path = DEFAULT_H10_COMPARE_METRICS_PATH,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    daily_globs: Sequence[str | Path] = DEFAULT_DAILY_GLOBS,
    adj_factor_globs: Sequence[str | Path] = DEFAULT_ADJ_FACTOR_GLOBS,
    benchmark_daily_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_DAILY_GLOBS,
    benchmark_adj_factor_globs: Sequence[str | Path] = DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    benchmark_symbol: str = "SPY",
    target_portfolio: str = DEFAULT_TARGET_PORTFOLIO,
    baseline_portfolio: str = DEFAULT_BASELINE_PORTFOLIO,
    validation_price_end: str = DEFAULT_VALIDATION_PRICE_END,
    assumed_cost_bps_per_side: float = DEFAULT_ASSUMED_COST_BPS_PER_SIDE,
) -> dict[str, Any]:
    """Compare h10 lead paths against daily rescoring with sticky execution controls."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_net = _load_net_targets(target_positions_path, portfolio=target_portfolio)
    symbols = tuple(sorted(target_net["symbol"].astype(str).unique()))
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
    ).rename(columns={"oto_return": "benchmark_oto_return"})
    benchmark_calendar = (
        benchmark_returns[benchmark_returns["symbol"].eq(benchmark_symbol)]["session_date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    beta_lookup = _load_beta_lookup(signal_panel_path, symbols=symbols)
    return_map = _build_return_date_map(target_net["session_date"].drop_duplicates(), benchmark_calendar)

    h10_curve = _load_h10_compare_curve(
        h10_compare_curve_path,
        baseline_portfolio=baseline_portfolio,
        target_portfolio=target_portfolio,
    )
    h10_metrics = _load_h10_compare_metrics(h10_compare_metrics_path)

    stock_return_lookup = _return_lookup(stock_returns)
    benchmark_lookup = _benchmark_lookup(benchmark_returns)

    records_frames: list[pd.DataFrame] = []
    strategy_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    daily_curve = h10_curve[["return_date", "benchmark_oto_return"]].copy()

    for config in DAILY_CONFIGS:
        records = _simulate_daily_execution(
            target_net=target_net,
            return_map=return_map,
            stock_return_lookup=stock_return_lookup,
            benchmark_lookup=benchmark_lookup,
            beta_lookup=beta_lookup,
            config=config,
            assumed_cost_bps_per_side=assumed_cost_bps_per_side,
        )
        records_frames.append(records)
        metric_row, action_frame = _daily_metrics_and_actions(
            records,
            strategy_label=config.label,
            assumed_cost_bps_per_side=assumed_cost_bps_per_side,
        )
        strategy_rows.append(metric_row)
        action_rows.extend(action_frame.to_dict("records"))
        daily_curve = daily_curve.merge(
            records[
                [
                    "return_date",
                    "gross_return",
                    "equity_gross",
                    "drawdown_gross",
                    "rolling_60_gross",
                    "net_return_assumed_cost",
                    "equity_net_assumed_cost",
                    "drawdown_net_assumed_cost",
                    "rolling_60_net_assumed_cost",
                    "turnover",
                    "gross_exposure",
                    "net_beta",
                ]
            ].rename(
                columns={
                    "gross_return": f"{config.label}__gross_return",
                    "equity_gross": f"{config.label}__equity_gross",
                    "drawdown_gross": f"{config.label}__drawdown_gross",
                    "rolling_60_gross": f"{config.label}__rolling_60_gross",
                    "net_return_assumed_cost": f"{config.label}__net_return_assumed_cost",
                    "equity_net_assumed_cost": f"{config.label}__equity_net_assumed_cost",
                    "drawdown_net_assumed_cost": f"{config.label}__drawdown_net_assumed_cost",
                    "rolling_60_net_assumed_cost": f"{config.label}__rolling_60_net_assumed_cost",
                    "turnover": f"{config.label}__turnover",
                    "gross_exposure": f"{config.label}__gross_exposure",
                    "net_beta": f"{config.label}__net_beta",
                }
            ),
            on="return_date",
            how="inner",
        )

    metrics = _merge_metrics(h10_metrics, pd.DataFrame(strategy_rows))
    actions = pd.DataFrame(action_rows)
    records_all = pd.concat(records_frames, ignore_index=True) if records_frames else pd.DataFrame()
    comparison = _comparison_curve(h10_curve, daily_curve)

    records_path = output_dir / "phase5a_daily_records.csv.gz"
    metrics_path = output_dir / "phase5a_metrics.csv"
    actions_path = output_dir / "phase5a_action_summary.csv"
    curve_path = output_dir / "phase5a_comparison_curve.csv"
    plot_path = output_dir / "phase5a_comparison_plot.png"
    memo_path = output_dir / "phase5a_daily_rescoring_sticky_memo.md"
    rollup_path = output_dir / "phase5a_rollup.json"

    records_all.to_csv(records_path, index=False, compression="gzip")
    metrics.to_csv(metrics_path, index=False)
    actions.to_csv(actions_path, index=False)
    comparison.to_csv(curve_path, index=False)
    _plot_comparison(comparison, output_path=plot_path)
    memo_path.write_text(_memo(metrics, actions, plot_path), encoding="utf-8")

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "target_positions_path": Path(target_positions_path).as_posix(),
        "h10_compare_curve_path": Path(h10_compare_curve_path).as_posix(),
        "h10_compare_metrics_path": Path(h10_compare_metrics_path).as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "target_portfolio": target_portfolio,
        "baseline_portfolio": baseline_portfolio,
        "benchmark_symbol": benchmark_symbol,
        "validation_price_end": validation_price_end,
        "assumed_cost_bps_per_side": float(assumed_cost_bps_per_side),
        "daily_configs": [asdict(config) for config in DAILY_CONFIGS],
        "records_artifact": records_path.as_posix(),
        "metrics_artifact": metrics_path.as_posix(),
        "actions_artifact": actions_path.as_posix(),
        "curve_artifact": curve_path.as_posix(),
        "plot_artifact": plot_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "daily_rescoring_wrapper_with_min_hold_and_sticky_controls_on_current_h10_lead",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_net_targets(path: str | Path, *, portfolio: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["session_date", "portfolio", "symbol", "signed_weight", "test_window_used"],
        low_memory=False,
    )
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    frame["portfolio"] = frame["portfolio"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["signed_weight"] = pd.to_numeric(frame["signed_weight"], errors="coerce")
    frame["test_window_used"] = frame["test_window_used"].astype(str).str.lower().isin(
        {"1", "true", "yes", "y"}
    )
    frame = frame[frame["portfolio"].eq(portfolio)].copy()
    if frame.empty:
        raise ValueError(f"Portfolio {portfolio!r} not found in {path}.")
    if frame["test_window_used"].any():
        raise ValueError("Target positions include test-window rows; refusing to continue.")
    net = (
        frame.groupby(["session_date", "symbol"], as_index=False)["signed_weight"]
        .sum()
        .sort_values(["session_date", "symbol"])
        .reset_index(drop=True)
    )
    net = net[net["signed_weight"].abs() > 1e-12].copy()
    return net


def _load_beta_lookup(signal_panel_path: str | Path, *, symbols: Sequence[str]) -> dict[tuple[pd.Timestamp, str], float]:
    panel = pd.read_csv(
        signal_panel_path,
        usecols=["session_date", "symbol", "beta"],
        low_memory=False,
    )
    panel["session_date"] = pd.to_datetime(panel["session_date"])
    panel["symbol"] = panel["symbol"].astype(str)
    panel["beta"] = pd.to_numeric(panel["beta"], errors="coerce")
    panel = panel[panel["symbol"].isin(set(symbols))].dropna(subset=["beta"]).copy()
    grouped = (
        panel.groupby(["session_date", "symbol"], as_index=False)["beta"]
        .mean()
        .sort_values(["session_date", "symbol"])
    )
    return {
        (row.session_date, row.symbol): float(row.beta)
        for row in grouped.itertuples(index=False)
    }


def _build_return_date_map(
    signal_dates: pd.Series,
    benchmark_calendar: pd.Series,
) -> dict[pd.Timestamp, pd.Timestamp]:
    signal_dates = pd.to_datetime(signal_dates).drop_duplicates().sort_values().reset_index(drop=True)
    calendar = pd.to_datetime(benchmark_calendar).drop_duplicates().sort_values().reset_index(drop=True)
    calendar_index = {date: idx for idx, date in enumerate(calendar)}
    out: dict[pd.Timestamp, pd.Timestamp] = {}
    for signal_date in signal_dates:
        idx = calendar_index.get(signal_date)
        if idx is None or idx + 2 >= len(calendar):
            continue
        out[pd.Timestamp(signal_date)] = pd.Timestamp(calendar.iloc[idx + 2])
    return out


def _load_h10_compare_curve(
    path: str | Path,
    *,
    baseline_portfolio: str,
    target_portfolio: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["return_date"])
    needed = [
        "return_date",
        "benchmark_oto_return",
        f"{baseline_portfolio}__return",
        f"{baseline_portfolio}__equity",
        f"{baseline_portfolio}__drawdown",
        f"{baseline_portfolio}__rolling_60_return",
        f"{target_portfolio}__return",
        f"{target_portfolio}__equity",
        f"{target_portfolio}__drawdown",
        f"{target_portfolio}__rolling_60_return",
        "benchmark_equity",
        "benchmark_drawdown",
        "benchmark_rolling_60_return",
    ]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing h10 comparison columns in {path}: {missing}")
    return frame[needed].copy()


def _load_h10_compare_metrics(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame


def _return_lookup(frame: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], float]:
    return {
        (pd.Timestamp(row.session_date), str(row.symbol)): float(row.oto_return)
        for row in frame.itertuples(index=False)
    }


def _benchmark_lookup(frame: pd.DataFrame) -> dict[pd.Timestamp, float]:
    benchmark = frame[frame["symbol"].astype(str).eq(frame["symbol"].astype(str).iloc[0])].copy()
    return {
        pd.Timestamp(row.session_date): float(row.benchmark_oto_return)
        for row in benchmark.itertuples(index=False)
    }


def _simulate_daily_execution(
    *,
    target_net: pd.DataFrame,
    return_map: dict[pd.Timestamp, pd.Timestamp],
    stock_return_lookup: dict[tuple[pd.Timestamp, str], float],
    benchmark_lookup: dict[pd.Timestamp, float],
    beta_lookup: dict[tuple[pd.Timestamp, str], float],
    config: StickyConfig,
    assumed_cost_bps_per_side: float,
) -> pd.DataFrame:
    targets_by_date = {
        pd.Timestamp(session_date): {
            str(row.symbol): float(row.signed_weight)
            for row in group.itertuples(index=False)
            if abs(float(row.signed_weight)) > 1e-12
        }
        for session_date, group in target_net.groupby("session_date", sort=True)
    }
    current_weights: dict[str, float] = {}
    current_ages: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    for signal_date in sorted(targets_by_date):
        return_date = return_map.get(pd.Timestamp(signal_date))
        if return_date is None:
            continue
        raw_target = dict(targets_by_date[signal_date])
        target_long_gross = float(
            sum(weight for weight in raw_target.values() if float(weight) > 1e-12)
        )
        target_short_gross = float(
            sum(-weight for weight in raw_target.values() if float(weight) < -1e-12)
        )
        target_long_count = int(sum(1 for weight in raw_target.values() if float(weight) > 1e-12))
        target_short_count = int(sum(1 for weight in raw_target.values() if float(weight) < -1e-12))
        candidate = _apply_min_hold_and_entry_limit(
            previous_weights=current_weights,
            previous_ages=current_ages,
            raw_target=raw_target,
            config=config,
        )
        candidate, _ = _normalize_to_side_grosses(
            candidate,
            target_long_gross=target_long_gross,
            target_short_gross=target_short_gross,
        )
        raw_turnover = _weight_turnover(current_weights, candidate)
        applied, action = _apply_sticky_controls(
            previous_weights=current_weights,
            candidate_weights=candidate,
            config=config,
            target_long_gross=target_long_gross,
            target_short_gross=target_short_gross,
            target_long_count=target_long_count,
            target_short_count=target_short_count,
        )
        turnover = _weight_turnover(current_weights, applied)
        gross_return, missing_symbols = _portfolio_return(
            weights=applied,
            return_date=return_date,
            stock_return_lookup=stock_return_lookup,
        )
        assumed_cost = turnover * assumed_cost_bps_per_side / 10000.0
        net_return = gross_return - assumed_cost
        gross_exposure = float(sum(abs(weight) for weight in applied.values()))
        net_beta, beta_coverage = _net_beta(
            weights=applied,
            signal_date=signal_date,
            beta_lookup=beta_lookup,
        )
        rows.append(
            {
                "strategy": config.label,
                "signal_date": pd.Timestamp(signal_date),
                "return_date": pd.Timestamp(return_date),
                "gross_return": gross_return,
                "net_return_assumed_cost": net_return,
                "benchmark_oto_return": benchmark_lookup.get(pd.Timestamp(return_date), np.nan),
                "turnover": turnover,
                "raw_turnover": raw_turnover,
                "assumed_cost_bps": assumed_cost * 10000.0,
                "gross_exposure": gross_exposure,
                "net_beta": net_beta,
                "beta_coverage_ratio": beta_coverage,
                "positions": int(len(applied)),
                "changed_symbols": _count_changed_symbols(current_weights, applied),
                "new_symbols": _count_new_symbols(current_weights, applied),
                "missing_return_symbols": int(missing_symbols),
                "rebalance_action": action,
                "test_window_used": False,
            }
        )
        previous_weights = dict(current_weights)
        current_weights = dict(applied)
        current_ages = _update_ages(previous_weights, current_ages, current_weights)

    frame = pd.DataFrame(rows).sort_values("return_date").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No daily execution rows built for strategy {config.label}.")
    frame = _add_path_columns(frame)
    return frame


def _apply_min_hold_and_entry_limit(
    *,
    previous_weights: dict[str, float],
    previous_ages: dict[str, int],
    raw_target: dict[str, float],
    config: StickyConfig,
) -> dict[str, float]:
    candidate = {symbol: float(weight) for symbol, weight in raw_target.items() if abs(weight) > 1e-12}
    for symbol, previous_weight in previous_weights.items():
        age = int(previous_ages.get(symbol, 0))
        if age < config.min_hold_sessions and abs(previous_weight) > 1e-12:
            candidate[symbol] = float(previous_weight)

    if config.max_new_names_per_rebalance is not None and config.max_new_names_per_rebalance >= 0:
        new_symbols = [
            symbol
            for symbol, weight in candidate.items()
            if abs(weight) > 1e-12 and abs(previous_weights.get(symbol, 0.0)) <= 1e-12
        ]
        if len(new_symbols) > config.max_new_names_per_rebalance:
            keep = set(
                sorted(
                    new_symbols,
                    key=lambda symbol: (abs(candidate[symbol]), symbol),
                    reverse=True,
                )[: config.max_new_names_per_rebalance]
            )
            for symbol in new_symbols:
                if symbol not in keep:
                    candidate.pop(symbol, None)
    return {symbol: float(weight) for symbol, weight in candidate.items() if abs(weight) > 1e-12}


def _apply_sticky_controls(
    *,
    previous_weights: dict[str, float],
    candidate_weights: dict[str, float],
    config: StickyConfig,
    target_long_gross: float,
    target_short_gross: float,
    target_long_count: int,
    target_short_count: int,
) -> tuple[dict[str, float], str]:
    raw_turnover = _weight_turnover(previous_weights, candidate_weights)
    if raw_turnover <= config.no_trade_band:
        return dict(previous_weights), "skip_no_trade_band"

    adjusted = dict(candidate_weights)
    actions: list[str] = []
    if config.max_turnover is not None and raw_turnover > config.max_turnover and raw_turnover > 0:
        scale = float(config.max_turnover / raw_turnover)
        blended: dict[str, float] = {}
        for symbol in set(previous_weights) | set(candidate_weights):
            previous = previous_weights.get(symbol, 0.0)
            target = candidate_weights.get(symbol, 0.0)
            weight = previous + (target - previous) * scale
            if abs(weight) > 1e-12:
                blended[symbol] = float(weight)
        adjusted = blended
        actions.append("cap_max_turnover")

    if config.min_weight_change > 0:
        thresholded: dict[str, float] = {}
        skipped = 0
        for symbol in set(previous_weights) | set(adjusted):
            previous = previous_weights.get(symbol, 0.0)
            target = adjusted.get(symbol, 0.0)
            if abs(target) <= 1e-12 and abs(previous) <= config.min_weight_change:
                weight = 0.0
            elif abs(target - previous) < config.min_weight_change:
                weight = previous
                if abs(target - previous) > 1e-12:
                    skipped += 1
            else:
                weight = target
            if abs(weight) > 1e-12:
                thresholded[symbol] = float(weight)
        adjusted = thresholded
        if skipped > 0:
            actions.append("skip_small_weight_deltas")

    adjusted, dropped = _enforce_side_position_limits(
        adjusted,
        target_long_count=target_long_count,
        target_short_count=target_short_count,
    )
    if dropped > 0:
        actions.append("enforce_side_position_limit")

    adjusted, normalized = _normalize_to_side_grosses(
        adjusted,
        target_long_gross=target_long_gross,
        target_short_gross=target_short_gross,
    )
    if normalized:
        actions.append("normalize_side_gross")

    final_turnover = _weight_turnover(previous_weights, adjusted)
    if config.max_turnover is not None and final_turnover > config.max_turnover and final_turnover > 0:
        scale = float(config.max_turnover / final_turnover)
        capped: dict[str, float] = {}
        for symbol in set(previous_weights) | set(adjusted):
            previous = previous_weights.get(symbol, 0.0)
            target = adjusted.get(symbol, 0.0)
            weight = previous + (target - previous) * scale
            if abs(weight) > 1e-12:
                capped[symbol] = float(weight)
        adjusted = capped
        actions.append("final_cap_max_turnover")

    if _weight_turnover(previous_weights, adjusted) <= 1e-12:
        return dict(previous_weights), actions[0] if actions else "skip_small_weight_deltas"
    return adjusted, "+".join(actions) if actions else "full_retarget"


def _portfolio_return(
    *,
    weights: dict[str, float],
    return_date: pd.Timestamp,
    stock_return_lookup: dict[tuple[pd.Timestamp, str], float],
) -> tuple[float, int]:
    gross_return = 0.0
    missing = 0
    for symbol, weight in weights.items():
        value = stock_return_lookup.get((pd.Timestamp(return_date), symbol))
        if value is None or pd.isna(value):
            missing += 1
            continue
        gross_return += float(weight) * float(value)
    return float(gross_return), int(missing)


def _net_beta(
    *,
    weights: dict[str, float],
    signal_date: pd.Timestamp,
    beta_lookup: dict[tuple[pd.Timestamp, str], float],
) -> tuple[float, float]:
    if not weights:
        return 0.0, 1.0
    beta_sum = 0.0
    covered = 0
    for symbol, weight in weights.items():
        beta = beta_lookup.get((pd.Timestamp(signal_date), symbol))
        if beta is None or pd.isna(beta):
            continue
        covered += 1
        beta_sum += float(weight) * float(beta)
    coverage = float(covered / len(weights)) if weights else 1.0
    return float(beta_sum), coverage


def _update_ages(
    previous_weights: dict[str, float],
    previous_ages: dict[str, int],
    new_weights: dict[str, float],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for symbol, weight in new_weights.items():
        previous = previous_weights.get(symbol, 0.0)
        if abs(previous) > 1e-12 and np.sign(previous) == np.sign(weight):
            out[symbol] = int(previous_ages.get(symbol, 0)) + 1
        else:
            out[symbol] = 1
    return out


def _add_path_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["equity_gross"] = (1.0 + out["gross_return"].astype(float)).cumprod()
    out["drawdown_gross"] = out["equity_gross"] / out["equity_gross"].cummax() - 1.0
    out["rolling_60_gross"] = out["equity_gross"] / out["equity_gross"].shift(60) - 1.0
    out["equity_net_assumed_cost"] = (1.0 + out["net_return_assumed_cost"].astype(float)).cumprod()
    out["drawdown_net_assumed_cost"] = (
        out["equity_net_assumed_cost"] / out["equity_net_assumed_cost"].cummax() - 1.0
    )
    out["rolling_60_net_assumed_cost"] = (
        out["equity_net_assumed_cost"] / out["equity_net_assumed_cost"].shift(60) - 1.0
    )
    return out


def _daily_metrics_and_actions(
    records: pd.DataFrame,
    *,
    strategy_label: str,
    assumed_cost_bps_per_side: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    returns = records["gross_return"].astype(float)
    equity = records["equity_gross"].astype(float)
    rolling_60 = records["rolling_60_gross"].dropna().astype(float)
    spy_return = records["benchmark_oto_return"].astype(float)
    metric = {
        "series": strategy_label,
        "portfolio": strategy_label,
        "start": str(pd.to_datetime(records["return_date"].min()).date()),
        "end": str(pd.to_datetime(records["return_date"].max()).date()),
        "final_equity": float(equity.iloc[-1]),
        "annualized_return": float(equity.iloc[-1] ** (252.0 / len(equity)) - 1.0),
        "annualized_vol": float(returns.std(ddof=1) * np.sqrt(252.0)),
        "sharpe_no_rf": float(
            returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)
            if returns.std(ddof=1) > 0
            else np.nan
        ),
        "max_drawdown": float(records["drawdown_gross"].min()),
        "rolling_60_positive_rate": float((rolling_60 > 0).mean()),
        "corr_to_spy": float(returns.corr(spy_return)),
        "mean_daily_return_bps": float(returns.mean() * 10000.0),
        "mean_turnover": float(records["turnover"].astype(float).mean()),
        "mean_assumed_cost_bps": float(records["assumed_cost_bps"].astype(float).mean()),
        "mean_gross_exposure": float(records["gross_exposure"].astype(float).mean()),
        "mean_abs_net_beta": float(records["net_beta"].abs().astype(float).mean()),
        "beta_coverage_ratio": float(records["beta_coverage_ratio"].astype(float).mean()),
        "hit_rate_daily": float((returns > 0).mean()),
        "mean_positions": float(records["positions"].astype(float).mean()),
        "max_positions": int(records["positions"].max()),
        "assumed_cost_bps_per_side": float(assumed_cost_bps_per_side),
        "test_window_used": False,
    }
    actions = (
        records.groupby("rebalance_action", sort=True)
        .agg(
            days=("return_date", "count"),
            mean_turnover=("turnover", "mean"),
            mean_new_symbols=("new_symbols", "mean"),
            mean_changed_symbols=("changed_symbols", "mean"),
        )
        .reset_index()
    )
    actions.insert(0, "strategy", strategy_label)
    actions["test_window_used"] = False
    return metric, actions


def _merge_metrics(h10_metrics: pd.DataFrame, daily_metrics: pd.DataFrame) -> pd.DataFrame:
    h10 = h10_metrics.copy()
    h10 = h10[h10["series"].isin(["baseline", "short hybrid overlay", "spy_raw"])].copy()
    h10["mean_turnover"] = np.nan
    h10["mean_assumed_cost_bps"] = np.nan
    h10["mean_gross_exposure"] = np.nan
    h10["mean_abs_net_beta"] = np.nan
    h10["beta_coverage_ratio"] = np.nan
    h10["hit_rate_daily"] = np.nan
    h10["mean_positions"] = np.nan
    h10["max_positions"] = np.nan
    h10["assumed_cost_bps_per_side"] = np.nan
    return pd.concat([h10, daily_metrics], ignore_index=True, sort=False)


def _comparison_curve(h10_curve: pd.DataFrame, daily_curve: pd.DataFrame) -> pd.DataFrame:
    out = h10_curve.merge(daily_curve, on=["return_date", "benchmark_oto_return"], how="inner")
    out["test_window_used"] = False
    return out.sort_values("return_date").reset_index(drop=True)


def _plot_comparison(curve: pd.DataFrame, *, output_path: str | Path) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.8, 1.8]},
    )
    ax1, ax2, ax3 = axes
    dates = curve["return_date"]

    series_specs = [
        ("h10_baseline", "sic2_soft_neutral__equity", "sic2_soft_neutral__drawdown", "sic2_soft_neutral__rolling_60_return"),
        (
            "h10_short_hybrid",
            "sic2_soft_neutral__short_hybrid_soft_fw_overlay__equity",
            "sic2_soft_neutral__short_hybrid_soft_fw_overlay__drawdown",
            "sic2_soft_neutral__short_hybrid_soft_fw_overlay__rolling_60_return",
        ),
        ("daily_full_retarget", "daily_full_retarget__equity_gross", "daily_full_retarget__drawdown_gross", "daily_full_retarget__rolling_60_gross"),
        ("daily_minhold3_sticky", "daily_minhold3_sticky__equity_gross", "daily_minhold3_sticky__drawdown_gross", "daily_minhold3_sticky__rolling_60_gross"),
    ]
    for label, equity_col, drawdown_col, rolling_col in series_specs:
        color = PLOT_COLORS[label]
        ax1.plot(dates, curve[equity_col], label=label, color=color, linewidth=2.0)
        ax2.plot(dates, curve[drawdown_col] * 100.0, color=color, linewidth=1.8)
        ax3.plot(dates, curve[rolling_col] * 100.0, color=color, linewidth=1.8)

    ax1.plot(
        dates,
        curve["benchmark_equity"],
        label="spy_raw",
        color=PLOT_COLORS["spy_raw"],
        linestyle="--",
        linewidth=1.8,
    )
    ax2.plot(
        dates,
        curve["benchmark_drawdown"] * 100.0,
        color=PLOT_COLORS["spy_raw"],
        linestyle="--",
        linewidth=1.6,
    )
    ax3.plot(
        dates,
        curve["benchmark_rolling_60_return"] * 100.0,
        color=PLOT_COLORS["spy_raw"],
        linestyle="--",
        linewidth=1.6,
    )

    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax1.set_ylabel("Growth of $1")
    ax2.set_ylabel("Drawdown %")
    ax3.set_ylabel("Rolling 60-session %")
    ax3.set_xlabel("Date")
    ax1.set_title("Phase5A Daily Rescoring Wrapper vs Existing Rolling H10 Paths")
    for ax in axes:
        ax.grid(alpha=0.2)
    ax1.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _memo(metrics: pd.DataFrame, actions: pd.DataFrame, plot_path: Path) -> str:
    display = metrics.copy()
    for column in ("annualized_return", "annualized_vol", "max_drawdown", "rolling_60_positive_rate", "corr_to_spy", "hit_rate_daily", "beta_coverage_ratio"):
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct_like if column != "corr_to_spy" else _fmt_float_like)
    for column in ("sharpe_no_rf", "mean_daily_return_bps", "mean_turnover", "mean_assumed_cost_bps", "mean_gross_exposure", "mean_abs_net_beta"):
        if column in display.columns:
            display[column] = display[column].map(_fmt_float_like)

    action_display = actions.copy()
    for column in ("mean_turnover", "mean_new_symbols", "mean_changed_symbols"):
        if column in action_display.columns:
            action_display[column] = action_display[column].map(_fmt_float_like)

    return "\n".join(
        [
            "# Phase5A Daily Rescoring + Min-Hold-3 + Sticky Controls",
            "",
            f"Generated: {_utc_now()}",
            "",
            "## Scope",
            "",
            "This pass keeps the current lead short-hybrid signal fixed and only changes the execution wrapper.",
            "",
            "## Metrics",
            "",
            _text_table(display),
            "",
            "## Rebalance Actions",
            "",
            _text_table(action_display),
            "",
            "## Plot",
            "",
            f"![Phase5A comparison]({plot_path.as_posix()})",
            "",
        ]
    )


def _weight_turnover(previous: dict[str, float], new: dict[str, float]) -> float:
    symbols = set(previous) | set(new)
    return float(sum(abs(new.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols))


def _count_changed_symbols(previous: dict[str, float], new: dict[str, float]) -> int:
    symbols = set(previous) | set(new)
    return int(sum(1 for symbol in symbols if abs(new.get(symbol, 0.0) - previous.get(symbol, 0.0)) > 1e-12))


def _count_new_symbols(previous: dict[str, float], new: dict[str, float]) -> int:
    return int(
        sum(
            1
            for symbol, weight in new.items()
            if abs(weight) > 1e-12 and abs(previous.get(symbol, 0.0)) <= 1e-12
        )
    )


def _normalize_to_side_grosses(
    weights: dict[str, float],
    *,
    target_long_gross: float,
    target_short_gross: float,
) -> tuple[dict[str, float], bool]:
    positive = {
        symbol: float(weight)
        for symbol, weight in weights.items()
        if float(weight) > 1e-12
    }
    negative = {
        symbol: float(weight)
        for symbol, weight in weights.items()
        if float(weight) < -1e-12
    }
    positive_gross = float(sum(positive.values()))
    negative_gross = float(sum(-weight for weight in negative.values()))
    normalized = False

    if target_long_gross > 0 and positive_gross > 0 and abs(positive_gross - target_long_gross) > 1e-12:
        scale = float(target_long_gross / positive_gross)
        positive = {
            symbol: float(weight * scale)
            for symbol, weight in positive.items()
            if abs(weight * scale) > 1e-12
        }
        normalized = True
    if target_short_gross > 0 and negative_gross > 0 and abs(negative_gross - target_short_gross) > 1e-12:
        scale = float(target_short_gross / negative_gross)
        negative = {
            symbol: float(weight * scale)
            for symbol, weight in negative.items()
            if abs(weight * scale) > 1e-12
        }
        normalized = True

    out = dict(positive)
    out.update(negative)
    return out, normalized


def _enforce_side_position_limits(
    weights: dict[str, float],
    *,
    target_long_count: int,
    target_short_count: int,
) -> tuple[dict[str, float], int]:
    positive = [
        (symbol, float(weight))
        for symbol, weight in weights.items()
        if float(weight) > 1e-12
    ]
    negative = [
        (symbol, float(weight))
        for symbol, weight in weights.items()
        if float(weight) < -1e-12
    ]
    if target_long_count > 0 and len(positive) > target_long_count:
        positive = sorted(positive, key=lambda item: (abs(item[1]), item[0]), reverse=True)[
            :target_long_count
        ]
    if target_short_count > 0 and len(negative) > target_short_count:
        negative = sorted(negative, key=lambda item: (abs(item[1]), item[0]), reverse=True)[
            :target_short_count
        ]
    limited = {symbol: weight for symbol, weight in positive}
    limited.update({symbol: weight for symbol, weight in negative})
    dropped = int(len(weights) - len(limited))
    return limited, dropped


def _text_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    return frame.to_string(index=False)


def _fmt_pct_like(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100.0:.2f}%"


def _fmt_float_like(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase5A daily rescoring wrapper with min-hold and sticky controls."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--target-positions-path", default=str(DEFAULT_TARGET_POSITIONS_PATH))
    parser.add_argument("--h10-compare-curve-path", default=str(DEFAULT_H10_COMPARE_CURVE_PATH))
    parser.add_argument("--h10-compare-metrics-path", default=str(DEFAULT_H10_COMPARE_METRICS_PATH))
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--target-portfolio", default=DEFAULT_TARGET_PORTFOLIO)
    parser.add_argument("--baseline-portfolio", default=DEFAULT_BASELINE_PORTFOLIO)
    parser.add_argument("--benchmark-symbol", default="SPY")
    parser.add_argument("--validation-price-end", default=DEFAULT_VALIDATION_PRICE_END)
    parser.add_argument("--assumed-cost-bps-per-side", type=float, default=DEFAULT_ASSUMED_COST_BPS_PER_SIDE)
    args = parser.parse_args(argv)

    build_phase5a_daily_rescoring_sticky_artifacts(
        output_root=args.output_root,
        target_positions_path=args.target_positions_path,
        h10_compare_curve_path=args.h10_compare_curve_path,
        h10_compare_metrics_path=args.h10_compare_metrics_path,
        signal_panel_path=args.signal_panel_path,
        target_portfolio=args.target_portfolio,
        baseline_portfolio=args.baseline_portfolio,
        benchmark_symbol=args.benchmark_symbol,
        validation_price_end=args.validation_price_end,
        assumed_cost_bps_per_side=args.assumed_cost_bps_per_side,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
