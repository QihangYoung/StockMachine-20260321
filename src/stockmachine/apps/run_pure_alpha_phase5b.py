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
from stockmachine.apps.run_pure_alpha_phase4z import (
    DEFAULT_VALIDATION_PRICE_END,
    _active_return_dates,
    _load_open_to_open_returns,
    _load_positions,
)
from stockmachine.apps.run_pure_alpha_phase5a import (
    DEFAULT_ASSUMED_COST_BPS_PER_SIDE,
    DEFAULT_BASELINE_PORTFOLIO,
    DEFAULT_H10_COMPARE_CURVE_PATH,
    DEFAULT_H10_COMPARE_METRICS_PATH,
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_TARGET_PORTFOLIO,
    DEFAULT_TARGET_POSITIONS_PATH,
    StickyConfig,
    _add_path_columns,
    _apply_min_hold_and_entry_limit,
    _apply_sticky_controls,
    _benchmark_lookup,
    _count_changed_symbols,
    _count_new_symbols,
    _daily_metrics_and_actions,
    _load_beta_lookup,
    _load_h10_compare_curve,
    _load_h10_compare_metrics,
    _load_net_targets,
    _net_beta,
    _normalize_to_side_grosses,
    _portfolio_return,
    _return_lookup,
    _update_ages,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5b_minhold5_h10_turnover_cap_20260426"


@dataclass(frozen=True, slots=True)
class Phase5BConfig:
    label: str
    min_hold_sessions: int
    no_trade_band: float
    min_weight_change: float
    max_new_names_per_rebalance: int


SWEEP_CONFIGS: tuple[Phase5BConfig, ...] = tuple(
    Phase5BConfig(
        label=(
            f"mh5_nt{int(no_trade_band * 100):02d}_"
            f"mwc{int(min_weight_change * 1000):03d}_"
            f"new{max_new_names:02d}"
        ),
        min_hold_sessions=5,
        no_trade_band=no_trade_band,
        min_weight_change=min_weight_change,
        max_new_names_per_rebalance=max_new_names,
    )
    for no_trade_band in (0.00, 0.02, 0.05)
    for min_weight_change in (0.000, 0.005, 0.010)
    for max_new_names in (4, 8, 12, 16)
)

PLOT_COLORS = {
    "h10_baseline": "#0f766e",
    "h10_short_hybrid": "#2563eb",
    "minhold5_h10_cap": "#c2410c",
    "spy_raw": "#6b7280",
    "h10_turnover_budget": "#111827",
    "minhold5_turnover": "#dc2626",
}


def build_phase5b_minhold5_h10_cap_artifacts(
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
    stock_return_lookup = _return_lookup(stock_returns)
    benchmark_lookup = _benchmark_lookup(benchmark_returns)

    h10_curve = _load_h10_compare_curve(
        h10_compare_curve_path,
        baseline_portfolio=baseline_portfolio,
        target_portfolio=target_portfolio,
    )
    h10_metrics = _load_h10_compare_metrics(h10_compare_metrics_path)
    h10_turnover_curve = _build_h10_turnover_curve(
        positions_path=target_positions_path,
        portfolio=target_portfolio,
        benchmark_calendar=benchmark_calendar,
        holding_period_sessions=10,
    )
    turnover_budget_map = {
        pd.Timestamp(row.return_date): float(row.h10_turnover)
        for row in h10_turnover_curve.itertuples(index=False)
    }
    h10_mean_turnover = float(h10_turnover_curve["h10_turnover"].iloc[10:].mean())
    h10_mean_gross = float(h10_turnover_curve["h10_gross_exposure"].iloc[10:].mean())
    h10_mean_positions = float(h10_turnover_curve["h10_positions"].iloc[10:].mean())

    sweep_metrics: list[dict[str, Any]] = []
    records_by_label: dict[str, pd.DataFrame] = {}
    action_frames: list[pd.DataFrame] = []
    for config in SWEEP_CONFIGS:
        records = _simulate_daily_execution_with_budget(
            target_net=target_net,
            return_map=return_map,
            stock_return_lookup=stock_return_lookup,
            benchmark_lookup=benchmark_lookup,
            beta_lookup=beta_lookup,
            config=config,
            turnover_budget_map=turnover_budget_map,
            assumed_cost_bps_per_side=assumed_cost_bps_per_side,
        )
        metric, actions = _daily_metrics_and_actions(
            records,
            strategy_label=config.label,
            assumed_cost_bps_per_side=assumed_cost_bps_per_side,
        )
        metric["turnover_budget_mean"] = h10_mean_turnover
        metric["turnover_vs_h10_ratio"] = (
            float(metric["mean_turnover"] / h10_mean_turnover) if h10_mean_turnover > 0 else np.nan
        )
        metric["turnover_cap_breach_days"] = int(
            (records["turnover"] > records["h10_turnover_budget"] + 1e-12).sum()
        )
        metric["config_min_hold_sessions"] = int(config.min_hold_sessions)
        metric["config_no_trade_band"] = float(config.no_trade_band)
        metric["config_min_weight_change"] = float(config.min_weight_change)
        metric["config_max_new_names_per_rebalance"] = int(config.max_new_names_per_rebalance)
        sweep_metrics.append(metric)
        actions.insert(1, "config_label", config.label)
        action_frames.append(actions)
        records_by_label[config.label] = records

    sweep = pd.DataFrame(sweep_metrics).sort_values(
        ["annualized_return", "sharpe_no_rf"],
        ascending=[False, False],
    )
    eligible = sweep[
        sweep["turnover_cap_breach_days"].eq(0)
        & (sweep["mean_turnover"].astype(float) <= h10_mean_turnover + 1e-12)
    ].copy()
    if eligible.empty:
        raise ValueError("No eligible min-hold-5 configs stayed within the rolling h10 turnover budget.")
    best = eligible.sort_values(["annualized_return", "sharpe_no_rf"], ascending=[False, False]).iloc[0]
    best_label = str(best["series"])
    best_records = records_by_label[best_label].copy()

    comparison = _build_comparison_curve(h10_curve, best_records, h10_turnover_curve)
    metrics = _merge_metrics(
        comparison=comparison,
        best_metric_row=best.to_dict(),
        target_portfolio=target_portfolio,
        h10_mean_turnover=h10_mean_turnover,
        h10_mean_gross=h10_mean_gross,
        h10_mean_positions=h10_mean_positions,
    )
    actions = (
        pd.concat(action_frames, ignore_index=True)
        if action_frames
        else pd.DataFrame(columns=["strategy", "config_label", "rebalance_action"])
    )
    best_actions = actions[actions["config_label"].eq(best_label)].copy()

    sweep_path = output_dir / "phase5b_minhold5_sweep.csv"
    metrics_path = output_dir / "phase5b_metrics.csv"
    actions_path = output_dir / "phase5b_best_action_summary.csv"
    all_actions_path = output_dir / "phase5b_all_action_summary.csv"
    curve_path = output_dir / "phase5b_comparison_curve.csv"
    turnover_path = output_dir / "phase5b_h10_turnover_budget_curve.csv"
    records_path = output_dir / "phase5b_best_records.csv.gz"
    plot_path = output_dir / "phase5b_comparison_plot.png"
    memo_path = output_dir / "phase5b_minhold5_h10_turnover_cap_memo.md"
    rollup_path = output_dir / "phase5b_rollup.json"

    sweep.to_csv(sweep_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    best_actions.to_csv(actions_path, index=False)
    actions.to_csv(all_actions_path, index=False)
    comparison.to_csv(curve_path, index=False)
    h10_turnover_curve.to_csv(turnover_path, index=False)
    best_records.to_csv(records_path, index=False, compression="gzip")
    _plot_comparison(comparison, output_path=plot_path)
    memo_path.write_text(
        _memo(
            metrics=metrics,
            best_actions=best_actions,
            eligible=eligible,
            best_label=best_label,
            plot_path=plot_path,
            h10_mean_turnover=h10_mean_turnover,
        ),
        encoding="utf-8",
    )

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
        "sweep_configs": [asdict(config) for config in SWEEP_CONFIGS],
        "best_config_label": best_label,
        "h10_mean_turnover_budget": h10_mean_turnover,
        "sweep_artifact": sweep_path.as_posix(),
        "metrics_artifact": metrics_path.as_posix(),
        "best_action_summary_artifact": actions_path.as_posix(),
        "all_action_summary_artifact": all_actions_path.as_posix(),
        "comparison_curve_artifact": curve_path.as_posix(),
        "turnover_curve_artifact": turnover_path.as_posix(),
        "best_records_artifact": records_path.as_posix(),
        "plot_artifact": plot_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "method": "min_hold_5_daily_rescoring_capped_by_realized_rolling_h10_daily_turnover_budget",
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


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


def _build_h10_turnover_curve(
    *,
    positions_path: str | Path,
    portfolio: str,
    benchmark_calendar: pd.Series,
    holding_period_sessions: int,
) -> pd.DataFrame:
    positions = _load_positions(positions_path)
    positions = positions[positions["portfolio"].eq(portfolio)].copy()
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
    aggregated = (
        active.groupby(["return_date", "symbol"], as_index=False)["signed_weight"]
        .sum()
        .merge(active_sleeves, on="return_date", how="left")
    )
    aggregated["portfolio_weight"] = aggregated["signed_weight"] / aggregated["active_sleeves"]
    weights_by_date = {
        pd.Timestamp(return_date): {
            str(row.symbol): float(row.portfolio_weight)
            for row in group.itertuples(index=False)
            if abs(float(row.portfolio_weight)) > 1e-12
        }
        for return_date, group in aggregated.groupby("return_date", sort=True)
    }
    previous: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for return_date in sorted(weights_by_date):
        weights = weights_by_date[return_date]
        rows.append(
            {
                "return_date": pd.Timestamp(return_date),
                "h10_turnover": _weight_turnover(previous, weights),
                "h10_positions": int(len(weights)),
                "h10_gross_exposure": float(sum(abs(weight) for weight in weights.values())),
            }
        )
        previous = weights
    return pd.DataFrame(rows).sort_values("return_date").reset_index(drop=True)


def _simulate_daily_execution_with_budget(
    *,
    target_net: pd.DataFrame,
    return_map: dict[pd.Timestamp, pd.Timestamp],
    stock_return_lookup: dict[tuple[pd.Timestamp, str], float],
    benchmark_lookup: dict[pd.Timestamp, float],
    beta_lookup: dict[tuple[pd.Timestamp, str], float],
    config: Phase5BConfig,
    turnover_budget_map: dict[pd.Timestamp, float],
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
        turnover_budget = turnover_budget_map.get(pd.Timestamp(return_date))
        if turnover_budget is None or not np.isfinite(turnover_budget):
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
            config=StickyConfig(
                label=config.label,
                min_hold_sessions=config.min_hold_sessions,
                no_trade_band=config.no_trade_band,
                max_turnover=float(turnover_budget),
                min_weight_change=config.min_weight_change,
                max_new_names_per_rebalance=config.max_new_names_per_rebalance,
            ),
        )
        candidate, _ = _normalize_to_side_grosses(
            candidate,
            target_long_gross=target_long_gross,
            target_short_gross=target_short_gross,
        )
        applied, action = _apply_sticky_controls(
            previous_weights=current_weights,
            candidate_weights=candidate,
            config=StickyConfig(
                label=config.label,
                min_hold_sessions=config.min_hold_sessions,
                no_trade_band=config.no_trade_band,
                max_turnover=float(turnover_budget),
                min_weight_change=config.min_weight_change,
                max_new_names_per_rebalance=config.max_new_names_per_rebalance,
            ),
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
                "h10_turnover_budget": float(turnover_budget),
                "turnover_budget_ratio": float(turnover / turnover_budget) if turnover_budget > 0 else np.nan,
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
    return _add_path_columns(frame)


def _merge_metrics(
    *,
    comparison: pd.DataFrame,
    best_metric_row: dict[str, Any],
    target_portfolio: str,
    h10_mean_turnover: float,
    h10_mean_gross: float,
    h10_mean_positions: float,
) -> pd.DataFrame:
    h10 = _reference_metrics_from_comparison(
        comparison=comparison,
        target_portfolio=target_portfolio,
        h10_mean_turnover=h10_mean_turnover,
        h10_mean_gross=h10_mean_gross,
        h10_mean_positions=h10_mean_positions,
    )
    best_row = pd.DataFrame([best_metric_row])
    return pd.concat([h10, best_row], ignore_index=True, sort=False)


def _reference_metrics_from_comparison(
    *,
    comparison: pd.DataFrame,
    target_portfolio: str,
    h10_mean_turnover: float,
    h10_mean_gross: float,
    h10_mean_positions: float,
) -> pd.DataFrame:
    end = str(pd.to_datetime(comparison["return_date"].max()).date())
    rows: list[dict[str, Any]] = []
    specs = (
        ("baseline", "sic2_soft_neutral", "sic2_soft_neutral__return"),
        ("short hybrid overlay", target_portfolio, "sic2_soft_neutral__short_hybrid_soft_fw_overlay__return"),
        ("spy_raw", "SPY", "benchmark_oto_return"),
    )
    benchmark = comparison["benchmark_oto_return"].astype(float)
    for series, portfolio, return_col in specs:
        returns = comparison[return_col].astype(float)
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        rolling_60 = equity / equity.shift(60) - 1.0
        row = {
            "series": series,
            "portfolio": portfolio,
            "start": str(pd.to_datetime(comparison["return_date"].min()).date()),
            "end": end,
            "final_equity": float(equity.iloc[-1]),
            "annualized_return": float(equity.iloc[-1] ** (252.0 / len(equity)) - 1.0),
            "annualized_vol": float(returns.std(ddof=1) * np.sqrt(252.0)),
            "sharpe_no_rf": float(
                returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)
                if returns.std(ddof=1) > 0
                else np.nan
            ),
            "max_drawdown": float(drawdown.min()),
            "rolling_60_positive_rate": float((rolling_60 > 0).mean()),
            "corr_to_spy": float(returns.corr(benchmark)) if series != "spy_raw" else 1.0,
            "mean_daily_return_bps": float(returns.mean() * 10000.0),
            "test_window_used": False,
            "mean_turnover": np.nan,
            "mean_gross_exposure": np.nan,
            "mean_positions": np.nan,
            "turnover_budget_mean": np.nan,
            "turnover_vs_h10_ratio": np.nan,
            "turnover_cap_breach_days": np.nan,
            "config_min_hold_sessions": np.nan,
            "config_no_trade_band": np.nan,
            "config_min_weight_change": np.nan,
            "config_max_new_names_per_rebalance": np.nan,
            "mean_assumed_cost_bps": np.nan,
            "mean_abs_net_beta": np.nan,
            "beta_coverage_ratio": np.nan,
            "hit_rate_daily": np.nan,
            "assumed_cost_bps_per_side": np.nan,
            "max_positions": np.nan,
        }
        if portfolio == target_portfolio:
            row["mean_turnover"] = float(h10_mean_turnover)
            row["mean_gross_exposure"] = float(h10_mean_gross)
            row["mean_positions"] = float(h10_mean_positions)
            row["turnover_budget_mean"] = float(h10_mean_turnover)
            row["turnover_vs_h10_ratio"] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def _build_comparison_curve(
    h10_curve: pd.DataFrame,
    best_records: pd.DataFrame,
    h10_turnover_curve: pd.DataFrame,
) -> pd.DataFrame:
    out = h10_curve.merge(
        best_records[
            [
                "return_date",
                "gross_return",
                "equity_gross",
                "drawdown_gross",
                "rolling_60_gross",
                "turnover",
                "h10_turnover_budget",
            ]
        ].rename(
            columns={
                "gross_return": "minhold5_h10_cap__return",
                "equity_gross": "minhold5_h10_cap__equity",
                "drawdown_gross": "minhold5_h10_cap__drawdown",
                "rolling_60_gross": "minhold5_h10_cap__rolling_60_return",
                "turnover": "minhold5_h10_cap__turnover",
            }
        ),
        on="return_date",
        how="inner",
    )
    out = out.merge(
        h10_turnover_curve[["return_date", "h10_turnover"]],
        on="return_date",
        how="left",
    )
    out["test_window_used"] = False
    return out.sort_values("return_date").reset_index(drop=True)


def _plot_comparison(curve: pd.DataFrame, *, output_path: str | Path) -> None:
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 15),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.8, 1.8, 1.6]},
    )
    ax1, ax2, ax3, ax4 = axes
    dates = curve["return_date"]

    series_specs = [
        (
            "h10_baseline",
            "sic2_soft_neutral__equity",
            "sic2_soft_neutral__drawdown",
            "sic2_soft_neutral__rolling_60_return",
        ),
        (
            "h10_short_hybrid",
            "sic2_soft_neutral__short_hybrid_soft_fw_overlay__equity",
            "sic2_soft_neutral__short_hybrid_soft_fw_overlay__drawdown",
            "sic2_soft_neutral__short_hybrid_soft_fw_overlay__rolling_60_return",
        ),
        (
            "minhold5_h10_cap",
            "minhold5_h10_cap__equity",
            "minhold5_h10_cap__drawdown",
            "minhold5_h10_cap__rolling_60_return",
        ),
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
    ax4.plot(
        dates,
        curve["h10_turnover"],
        color=PLOT_COLORS["h10_turnover_budget"],
        linewidth=1.6,
        label="rolling_h10_turnover",
    )
    ax4.plot(
        dates,
        curve["minhold5_h10_cap__turnover"],
        color=PLOT_COLORS["minhold5_turnover"],
        linewidth=1.6,
        label="minhold5_turnover",
    )

    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax4.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax1.set_ylabel("Growth of $1")
    ax2.set_ylabel("Drawdown %")
    ax3.set_ylabel("Rolling 60-session %")
    ax4.set_ylabel("Daily Turnover")
    ax4.set_xlabel("Date")
    ax1.set_title("Phase5B Min-Hold-5 with Rolling H10 Turnover Cap")
    for ax in axes:
        ax.grid(alpha=0.2)
    ax1.legend(loc="upper left", fontsize=10)
    ax4.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _memo(
    *,
    metrics: pd.DataFrame,
    best_actions: pd.DataFrame,
    eligible: pd.DataFrame,
    best_label: str,
    plot_path: Path,
    h10_mean_turnover: float,
) -> str:
    display = metrics.copy()
    for column in (
        "annualized_return",
        "annualized_vol",
        "max_drawdown",
        "rolling_60_positive_rate",
        "corr_to_spy",
        "hit_rate_daily",
        "beta_coverage_ratio",
    ):
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct_like if column != "corr_to_spy" else _fmt_float_like)
    for column in (
        "sharpe_no_rf",
        "mean_daily_return_bps",
        "mean_turnover",
        "turnover_budget_mean",
        "turnover_vs_h10_ratio",
        "mean_assumed_cost_bps",
        "mean_gross_exposure",
        "mean_abs_net_beta",
        "mean_positions",
    ):
        if column in display.columns:
            display[column] = display[column].map(_fmt_float_like)

    actions = best_actions.copy()
    for column in ("mean_turnover", "mean_new_symbols", "mean_changed_symbols"):
        if column in actions.columns:
            actions[column] = actions[column].map(_fmt_float_like)

    shortlist = eligible[
        [
            "series",
            "annualized_return",
            "annualized_vol",
            "sharpe_no_rf",
            "mean_turnover",
            "turnover_vs_h10_ratio",
        ]
    ].copy()
    shortlist = shortlist.sort_values(["annualized_return", "sharpe_no_rf"], ascending=[False, False]).head(10)
    for column in ("annualized_return", "annualized_vol"):
        shortlist[column] = shortlist[column].map(_fmt_pct_like)
    for column in ("sharpe_no_rf", "mean_turnover", "turnover_vs_h10_ratio"):
        shortlist[column] = shortlist[column].map(_fmt_float_like)

    return "\n".join(
        [
            "# Phase5B Min-Hold-5 under Rolling H10 Turnover Cap",
            "",
            f"Generated: {_utc_now()}",
            "",
            f"Rolling h10 mean daily turnover benchmark = {h10_mean_turnover:.6f}",
            "",
            f"Best eligible config: {best_label}",
            "",
            "## Metrics",
            "",
            _text_table(display),
            "",
            "## Best Config Rebalance Actions",
            "",
            _text_table(actions),
            "",
            "## Top Eligible Configs",
            "",
            _text_table(shortlist),
            "",
            "## Plot",
            "",
            f"![Phase5B comparison]({plot_path.as_posix()})",
            "",
        ]
    )


def _weight_turnover(previous: dict[str, float], new: dict[str, float]) -> float:
    symbols = set(previous) | set(new)
    return float(sum(abs(new.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols))


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
        description="Run Phase5B min-hold-5 daily rescoring capped by rolling h10 turnover."
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

    build_phase5b_minhold5_h10_cap_artifacts(
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
