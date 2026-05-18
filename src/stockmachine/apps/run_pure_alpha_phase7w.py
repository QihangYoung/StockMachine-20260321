"""Phase7W walk-forward cross-asset insurance overlays.

Phase7W extends Phase7V by making insurance weights state-dependent. The
state detector only uses lagged shared_core and SPY path features, so the
overlay schedule is a walk-forward diagnostic rather than an ex-post fit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase7b import _markdown_table
from stockmachine.apps.run_pure_alpha_phase7v import (
    CANDIDATES,
    DEFAULT_ADJ_FACTOR_PATHS,
    DEFAULT_BENCHMARK_INDEX_PATHS,
    DEFAULT_DAILY_BAR_PATHS,
    DEFAULT_SHARED_CORE_CURVE,
    DEFAULT_SHARED_PORTFOLIO,
    Candidate,
    _build_candidate_returns,
    _format_components,
    _load_external_returns,
    _load_shared_core_returns,
    _performance_metrics,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7w_walk_forward_insurance_overlay_20260518"
TRADING_DAYS = 252.0


@dataclass(frozen=True)
class InsurancePolicy:
    name: str
    overlay_kind: str
    calm_weights: Mapping[str, float]
    watch_weights: Mapping[str, float]
    stress_weights: Mapping[str, float]
    description: str


POLICIES: tuple[InsurancePolicy, ...] = (
    InsurancePolicy(
        name="funded_gld_0_10_20",
        overlay_kind="funded",
        calm_weights={},
        watch_weights={"GLD_gold": 0.10},
        stress_weights={"GLD_gold": 0.20},
        description="turn on gold when shared_core/SPY path risk appears",
    ),
    InsurancePolicy(
        name="funded_gld_5_10_20",
        overlay_kind="funded",
        calm_weights={"GLD_gold": 0.05},
        watch_weights={"GLD_gold": 0.10},
        stress_weights={"GLD_gold": 0.20},
        description="keep a small always-on gold reserve and scale up in stress",
    ),
    InsurancePolicy(
        name="funded_bil_0_10_20",
        overlay_kind="funded",
        calm_weights={},
        watch_weights={"BIL_cash": 0.10},
        stress_weights={"BIL_cash": 0.20},
        description="turn on cash as pure de-risking",
    ),
    InsurancePolicy(
        name="funded_ief_0_10_20",
        overlay_kind="funded",
        calm_weights={},
        watch_weights={"IEF_duration": 0.10},
        stress_weights={"IEF_duration": 0.20},
        description="turn on duration for growth-scare/rate-cut protection",
    ),
    InsurancePolicy(
        name="funded_equal_defensive_0_10_20",
        overlay_kind="funded",
        calm_weights={},
        watch_weights={"equal_defensive": 0.10},
        stress_weights={"equal_defensive": 0.20},
        description="turn on a diversified defensive basket",
    ),
    InsurancePolicy(
        name="funded_duration_gold_trend_0_10_20",
        overlay_kind="funded",
        calm_weights={},
        watch_weights={"duration_gold_trend": 0.10},
        stress_weights={"duration_gold_trend": 0.20},
        description="turn on duration/gold/trend convexity basket",
    ),
    InsurancePolicy(
        name="funded_cash_gld_0_10_20",
        overlay_kind="funded",
        calm_weights={},
        watch_weights={"GLD_gold": 0.10},
        stress_weights={"GLD_gold": 0.10, "BIL_cash": 0.10},
        description="gold in watch state, gold plus cash in stress",
    ),
    InsurancePolicy(
        name="financed_gld_0_05_10",
        overlay_kind="financed",
        calm_weights={},
        watch_weights={"GLD_gold": 0.05},
        stress_weights={"GLD_gold": 0.10},
        description="small financed gold overlay",
    ),
    InsurancePolicy(
        name="financed_equal_defensive_0_05_10",
        overlay_kind="financed",
        calm_weights={},
        watch_weights={"equal_defensive": 0.05},
        stress_weights={"equal_defensive": 0.10},
        description="small financed defensive basket overlay",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7W walk-forward insurance overlays.")
    parser.add_argument("--shared-core-curve", default=str(DEFAULT_SHARED_CORE_CURVE))
    parser.add_argument("--shared-portfolio", default=DEFAULT_SHARED_PORTFOLIO)
    parser.add_argument("--daily-bar-path", action="append", dest="daily_bar_paths")
    parser.add_argument("--adj-factor-path", action="append", dest="adj_factor_paths")
    parser.add_argument("--benchmark-index-path", action="append", dest="benchmark_index_paths")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cash-column", default="BIL_cash")
    parser.add_argument("--overlay-cost-bps-per-traded-notional", type=float, default=1.0)
    parser.add_argument("--drawdown-watch", type=float, default=-0.03)
    parser.add_argument("--drawdown-stress", type=float, default=-0.06)
    parser.add_argument("--roll20-stress", type=float, default=-0.03)
    parser.add_argument("--vol63-stress", type=float, default=0.14)
    parser.add_argument("--spy-drawdown-stress", type=float, default=-0.08)
    parser.add_argument("--spy-roll20-stress", type=float, default=-0.06)
    args = parser.parse_args(argv)

    daily_bar_paths = _paths_or_default(args.daily_bar_paths, DEFAULT_DAILY_BAR_PATHS)
    adj_factor_paths = _paths_or_default(args.adj_factor_paths, DEFAULT_ADJ_FACTOR_PATHS)
    benchmark_index_paths = _paths_or_default(args.benchmark_index_paths, DEFAULT_BENCHMARK_INDEX_PATHS)

    rollup = build_phase7w_walk_forward_insurance_overlay(
        shared_core_curve=Path(args.shared_core_curve),
        shared_portfolio=str(args.shared_portfolio),
        daily_bar_paths=daily_bar_paths,
        adj_factor_paths=adj_factor_paths,
        benchmark_index_paths=benchmark_index_paths,
        output_root=Path(args.output_root),
        cash_column=str(args.cash_column),
        overlay_cost_bps_per_traded_notional=float(args.overlay_cost_bps_per_traded_notional),
        thresholds={
            "drawdown_watch": float(args.drawdown_watch),
            "drawdown_stress": float(args.drawdown_stress),
            "roll20_stress": float(args.roll20_stress),
            "vol63_stress": float(args.vol63_stress),
            "spy_drawdown_stress": float(args.spy_drawdown_stress),
            "spy_roll20_stress": float(args.spy_roll20_stress),
        },
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=False))
    return 0


def build_phase7w_walk_forward_insurance_overlay(
    *,
    shared_core_curve: Path,
    shared_portfolio: str,
    daily_bar_paths: Sequence[Path],
    adj_factor_paths: Sequence[Path],
    benchmark_index_paths: Sequence[Path],
    output_root: Path,
    cash_column: str,
    overlay_cost_bps_per_traded_notional: float,
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)

    shared = _load_shared_core_returns(shared_core_curve, shared_portfolio=shared_portfolio)
    required_symbols = tuple(
        sorted({symbol for candidate in CANDIDATES for symbol in candidate.components})
    )
    external = _load_external_returns(
        daily_bar_paths=daily_bar_paths,
        adj_factor_paths=adj_factor_paths,
        benchmark_index_paths=benchmark_index_paths,
        symbols=required_symbols,
    )
    candidate_returns = _build_candidate_returns(external, CANDIDATES)
    panel = shared.join(candidate_returns, how="inner").sort_index()
    if panel.empty:
        raise ValueError("No common dates remain between shared_core and external assets.")

    state = _build_lagged_state(panel, thresholds=thresholds)
    policy_returns, policy_weights, policy_costs = _run_policy_grid(
        panel,
        state=state,
        policies=POLICIES,
        cash_column=cash_column,
        overlay_cost_bps_per_traded_notional=overlay_cost_bps_per_traded_notional,
    )
    summary = _policy_summary(panel["shared_core"], policy_returns, policy_weights, policy_costs)
    state_summary = _state_summary(panel["shared_core"], state, policy_returns)
    tail_summary = _tail_window_summary(panel["shared_core"], policy_returns, candidate_returns)

    daily_panel = pd.concat([panel.loc[:, ["shared_core"]], state, policy_returns], axis=1)
    daily_panel.index.name = "date"
    policy_weights.index.name = "date"
    policy_costs.index.name = "date"

    daily_panel_path = output_root / "phase7w_daily_panel.csv"
    weights_path = output_root / "phase7w_policy_weights.csv"
    costs_path = output_root / "phase7w_policy_costs.csv"
    summary_path = output_root / "phase7w_policy_summary.csv"
    state_path = output_root / "phase7w_state_summary.csv"
    tail_path = output_root / "phase7w_tail_window_summary.csv"
    rollup_path = output_root / "phase7w_rollup.json"
    memo_path = output_root / "phase7w_walk_forward_insurance_overlay_memo_zh.md"

    daily_panel.reset_index().to_csv(daily_panel_path, index=False)
    policy_weights.reset_index().to_csv(weights_path, index=False)
    policy_costs.reset_index().to_csv(costs_path, index=False)
    summary.to_csv(summary_path, index=False)
    state_summary.to_csv(state_path, index=False)
    tail_summary.to_csv(tail_path, index=False)

    base_metrics = _performance_metrics(panel["shared_core"], name="shared_core_100")
    rollup: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shared_core_curve": str(shared_core_curve),
        "shared_portfolio": shared_portfolio,
        "daily_bar_paths": [str(path) for path in daily_bar_paths],
        "adj_factor_paths": [str(path) for path in adj_factor_paths],
        "benchmark_index_paths": [str(path) for path in benchmark_index_paths],
        "output_root": str(output_root),
        "start": str(panel.index.min().date()),
        "end": str(panel.index.max().date()),
        "days": int(len(panel)),
        "cash_column": cash_column,
        "overlay_cost_bps_per_traded_notional": float(overlay_cost_bps_per_traded_notional),
        "thresholds": {key: float(value) for key, value in thresholds.items()},
        "base_metrics": base_metrics,
        "best_by_sharpe": _best_row_payload(summary, "sharpe"),
        "best_by_drawdown": _best_row_payload(summary, "max_dd_reduction_vs_base"),
        "best_by_return_drag_constrained_dd": _best_return_drag_constrained(summary),
        "paths": {
            "daily_panel": str(daily_panel_path),
            "policy_weights": str(weights_path),
            "policy_costs": str(costs_path),
            "policy_summary": str(summary_path),
            "state_summary": str(state_path),
            "tail_window_summary": str(tail_path),
            "memo": str(memo_path),
        },
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_memo(
        memo_path=memo_path,
        rollup=rollup,
        summary=summary,
        state_summary=state_summary,
        tail_summary=tail_summary,
    )
    return rollup


def _build_lagged_state(panel: pd.DataFrame, *, thresholds: Mapping[str, float]) -> pd.DataFrame:
    base = panel["shared_core"].astype(float)
    spy = panel["SPY_us_equity"].astype(float)

    base_equity = (1.0 + base).cumprod()
    spy_equity = (1.0 + spy).cumprod()
    base_drawdown_lag = (base_equity / base_equity.cummax() - 1.0).shift(1)
    spy_drawdown_lag = (spy_equity / spy_equity.cummax() - 1.0).shift(1)
    base_roll20_lag = _rolling_compound_return(base, 20).shift(1)
    spy_roll20_lag = _rolling_compound_return(spy, 20).shift(1)
    base_vol63_lag = (base.rolling(63).std(ddof=1) * np.sqrt(TRADING_DAYS)).shift(1)

    risk_score = pd.Series(0, index=panel.index, dtype=int)
    risk_score += (base_drawdown_lag <= thresholds["drawdown_watch"]).fillna(False).astype(int)
    risk_score += (base_drawdown_lag <= thresholds["drawdown_stress"]).fillna(False).astype(int)
    risk_score += (base_roll20_lag <= thresholds["roll20_stress"]).fillna(False).astype(int)
    risk_score += (base_vol63_lag >= thresholds["vol63_stress"]).fillna(False).astype(int)
    risk_score += (spy_drawdown_lag <= thresholds["spy_drawdown_stress"]).fillna(False).astype(int)
    risk_score += (spy_roll20_lag <= thresholds["spy_roll20_stress"]).fillna(False).astype(int)

    risk_state = pd.Series("calm", index=panel.index, dtype=object)
    risk_state.loc[risk_score.eq(1)] = "watch"
    risk_state.loc[risk_score.ge(2)] = "stress"

    return pd.DataFrame(
        {
            "base_drawdown_lag": base_drawdown_lag,
            "base_roll20_lag": base_roll20_lag,
            "base_vol63_lag": base_vol63_lag,
            "spy_drawdown_lag": spy_drawdown_lag,
            "spy_roll20_lag": spy_roll20_lag,
            "risk_score": risk_score,
            "risk_state": risk_state,
        },
        index=panel.index,
    )


def _run_policy_grid(
    panel: pd.DataFrame,
    *,
    state: pd.DataFrame,
    policies: Sequence[InsurancePolicy],
    cash_column: str,
    overlay_cost_bps_per_traded_notional: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = pd.DataFrame(index=panel.index)
    weights = pd.DataFrame(index=panel.index)
    costs = pd.DataFrame(index=panel.index)
    base = panel["shared_core"].astype(float)
    cash = panel[cash_column].astype(float)

    for policy in policies:
        schedule = _policy_weight_schedule(state["risk_state"], policy)
        total_weight = schedule.sum(axis=1)
        if total_weight.gt(0.30 + 1e-12).any():
            raise ValueError(f"{policy.name} exceeds the diagnostic 30% overlay cap.")
        if policy.overlay_kind == "funded":
            gross_return = (1.0 - total_weight) * base
            for column in schedule.columns:
                gross_return = gross_return + schedule[column] * panel[column]
            turnover = _funded_turnover(schedule)
            gross_nav_exposure = pd.Series(1.0, index=panel.index)
        elif policy.overlay_kind == "financed":
            gross_return = base.copy()
            for column in schedule.columns:
                gross_return = gross_return + schedule[column] * (panel[column] - cash)
            turnover = schedule.diff().abs().sum(axis=1).fillna(schedule.abs().sum(axis=1))
            gross_nav_exposure = 1.0 + total_weight
        else:
            raise ValueError(f"Unsupported overlay kind: {policy.overlay_kind}")

        cost_return = turnover * overlay_cost_bps_per_traded_notional / 10000.0
        returns[policy.name] = gross_return - cost_return
        costs[f"{policy.name}__cost_return"] = cost_return
        costs[f"{policy.name}__turnover"] = turnover
        costs[f"{policy.name}__gross_nav_exposure"] = gross_nav_exposure
        weights[f"{policy.name}__total_overlay_weight"] = total_weight
        for column in schedule.columns:
            weights[f"{policy.name}__{column}"] = schedule[column]
    return returns, weights, costs


def _policy_weight_schedule(risk_state: pd.Series, policy: InsurancePolicy) -> pd.DataFrame:
    asset_names = sorted(
        set(policy.calm_weights)
        .union(policy.watch_weights)
        .union(policy.stress_weights)
    )
    schedule = pd.DataFrame(0.0, index=risk_state.index, columns=asset_names)
    state_to_weights = {
        "calm": policy.calm_weights,
        "watch": policy.watch_weights,
        "stress": policy.stress_weights,
    }
    for state_name, weights in state_to_weights.items():
        mask = risk_state.eq(state_name)
        for asset, weight in weights.items():
            schedule.loc[mask, asset] = float(weight)
    return schedule


def _funded_turnover(schedule: pd.DataFrame) -> pd.Series:
    total_weight = schedule.sum(axis=1)
    base_weight = 1.0 - total_weight
    full_schedule = schedule.copy()
    full_schedule.insert(0, "shared_core", base_weight)
    first = full_schedule.iloc[0].copy()
    first.loc[:] = 0.0
    first.loc["shared_core"] = 1.0
    previous = pd.concat([first.to_frame().T, full_schedule.iloc[:-1]], ignore_index=True)
    previous.index = full_schedule.index
    return (full_schedule - previous).abs().sum(axis=1)


def _policy_summary(
    base: pd.Series,
    policy_returns: pd.DataFrame,
    policy_weights: pd.DataFrame,
    policy_costs: pd.DataFrame,
) -> pd.DataFrame:
    base_metrics = _performance_metrics(base, name="shared_core_100")
    worst_decile_mask = base <= base.quantile(0.10)
    base_worst_decile_mean = float(base.loc[worst_decile_mask].mean() * 10000.0)
    rows: list[dict[str, object]] = []
    policy_by_name = {policy.name: policy for policy in POLICIES}
    for column in policy_returns.columns:
        metrics = _performance_metrics(policy_returns[column], name=column)
        policy = policy_by_name[column]
        mean_overlay = float(policy_weights[f"{column}__total_overlay_weight"].mean())
        active_rate = float(policy_weights[f"{column}__total_overlay_weight"].gt(0.0).mean())
        max_overlay = float(policy_weights[f"{column}__total_overlay_weight"].max())
        mean_cost_bps = float(policy_costs[f"{column}__cost_return"].mean() * 10000.0)
        mean_turnover = float(policy_costs[f"{column}__turnover"].mean())
        row: dict[str, object] = {
            "policy": column,
            "overlay_kind": policy.overlay_kind,
            "description": policy.description,
            "mean_overlay_weight": mean_overlay,
            "active_rate": active_rate,
            "max_overlay_weight": max_overlay,
            "mean_overlay_turnover": mean_turnover,
            "mean_overlay_cost_bps": mean_cost_bps,
            "ann_return_drag_vs_base": float(base_metrics["ann_return"] - metrics["ann_return"]),
            "vol_reduction_vs_base": float(base_metrics["vol"] - metrics["vol"]),
            "max_dd_reduction_vs_base": float(metrics["max_dd"] - base_metrics["max_dd"]),
            "cvar05_reduction_vs_base_bps": float(
                metrics["cvar05_1d_bps"] - base_metrics["cvar05_1d_bps"]
            ),
            "worst_20d_reduction_vs_base": float(metrics["worst_20d"] - base_metrics["worst_20d"]),
            "mean_return_on_base_worst_decile_bps": float(
                policy_returns[column].loc[worst_decile_mask].mean() * 10000.0
            ),
            "base_worst_decile_mean_bps": base_worst_decile_mean,
        }
        row["bad_day_improvement_bps"] = (
            row["mean_return_on_base_worst_decile_bps"] - base_worst_decile_mean
        )
        row.update({key: value for key, value in metrics.items() if key != "name"})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["max_dd_reduction_vs_base", "ann_return_drag_vs_base"],
        ascending=[False, True],
    )


def _state_summary(
    base: pd.Series,
    state: pd.DataFrame,
    policy_returns: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state_name in ("calm", "watch", "stress"):
        mask = state["risk_state"].eq(state_name)
        if not mask.any():
            continue
        row: dict[str, object] = {
            "risk_state": state_name,
            "days": int(mask.sum()),
            "day_rate": float(mask.mean()),
            "mean_base_return_bps": float(base.loc[mask].mean() * 10000.0),
            "base_hit_rate": float((base.loc[mask] > 0.0).mean()),
            "mean_risk_score": float(state.loc[mask, "risk_score"].mean()),
            "mean_base_drawdown_lag": float(state.loc[mask, "base_drawdown_lag"].mean()),
            "mean_base_roll20_lag": float(state.loc[mask, "base_roll20_lag"].mean()),
            "mean_base_vol63_lag": float(state.loc[mask, "base_vol63_lag"].mean()),
        }
        for policy in POLICIES:
            row[f"{policy.name}_mean_bps"] = float(policy_returns.loc[mask, policy.name].mean() * 10000.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _tail_window_summary(
    base: pd.Series,
    policy_returns: pd.DataFrame,
    candidate_returns: pd.DataFrame,
) -> pd.DataFrame:
    windows = {
        "base_worst_20d": _worst_window_dates(base, 20),
        "base_worst_60d": _worst_window_dates(base, 60),
        "base_worst_decile_days": base.index[base <= base.quantile(0.10)],
    }
    selected = [
        "shared_core",
        "BIL_cash",
        "IEF_duration",
        "GLD_gold",
        "equal_defensive",
        "duration_gold_trend",
        "funded_gld_0_10_20",
        "funded_gld_5_10_20",
        "funded_bil_0_10_20",
        "funded_equal_defensive_0_10_20",
        "funded_duration_gold_trend_0_10_20",
        "funded_cash_gld_0_10_20",
        "financed_gld_0_05_10",
    ]
    comparison = pd.concat(
        [
            base.rename("shared_core"),
            candidate_returns,
            policy_returns,
        ],
        axis=1,
    )
    rows: list[dict[str, object]] = []
    for window_name, dates in windows.items():
        dates = pd.Index(dates).intersection(comparison.index)
        for column in selected:
            if column not in comparison.columns:
                continue
            series = comparison.loc[dates, column].dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "window": window_name,
                    "series": column,
                    "days": int(series.shape[0]),
                    "cumulative_return": float((1.0 + series).prod() - 1.0),
                    "mean_daily_bps": float(series.mean() * 10000.0),
                    "hit_rate": float((series > 0.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _write_memo(
    *,
    memo_path: Path,
    rollup: Mapping[str, object],
    summary: pd.DataFrame,
    state_summary: pd.DataFrame,
    tail_summary: pd.DataFrame,
) -> None:
    base = rollup["base_metrics"]
    assert isinstance(base, Mapping)
    summary_view = summary.loc[
        :,
        [
            "policy",
            "overlay_kind",
            "mean_overlay_weight",
            "active_rate",
            "ann_return",
            "vol",
            "sharpe",
            "max_dd",
            "ann_return_drag_vs_base",
            "max_dd_reduction_vs_base",
            "bad_day_improvement_bps",
            "mean_overlay_cost_bps",
        ],
    ].sort_values("max_dd_reduction_vs_base", ascending=False)
    constrained = summary.loc[summary["ann_return_drag_vs_base"].le(0.03)].copy()
    constrained = constrained.sort_values(
        ["max_dd_reduction_vs_base", "sharpe"],
        ascending=[False, False],
    )
    tail_focus = tail_summary.loc[
        tail_summary["series"].isin(
            [
                "shared_core",
                "BIL_cash",
                "IEF_duration",
                "GLD_gold",
                "funded_gld_0_10_20",
                "funded_gld_5_10_20",
                "funded_equal_defensive_0_10_20",
                "funded_cash_gld_0_10_20",
                "financed_gld_0_05_10",
            ]
        )
    ].copy()
    lines = [
        "# Phase7W Walk-Forward 外部资产保险层",
        "",
        "Validation-only。未使用 test lockbox。",
        "",
        "## Setup",
        "",
        f"- shared_core curve: `{rollup['shared_core_curve']}`",
        f"- external data: `{', '.join(str(path) for path in rollup['daily_bar_paths'])}`",
        f"- overlap: `{rollup['start']}` to `{rollup['end']}`, `{rollup['days']}` days",
        f"- overlay cost: `{rollup['overlay_cost_bps_per_traded_notional']}` bps per traded notional",
        "- state detector: lagged shared_core drawdown, lagged shared_core 20d return, lagged shared_core 63d vol, lagged SPY drawdown, lagged SPY 20d return",
        "- calm/watch/stress weights are fixed structural rules; no in-sample optimized sizing.",
        "",
        "## Base Shared Core",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "days": base["days"],
                        "ann_return": base["ann_return"],
                        "vol": base["vol"],
                        "sharpe": base["sharpe"],
                        "max_dd": base["max_dd"],
                        "worst_20d": base["worst_20d"],
                        "cvar05_1d_bps": base["cvar05_1d_bps"],
                    }
                ]
            )
        ),
        "",
        "## Regime State",
        "",
        _markdown_table(state_summary),
        "",
        "## Policy Summary",
        "",
        _markdown_table(summary_view),
        "",
        "## Drag <= 3% Candidates",
        "",
        _markdown_table(
            constrained.loc[
                :,
                [
                    "policy",
                    "mean_overlay_weight",
                    "active_rate",
                    "ann_return",
                    "sharpe",
                    "max_dd",
                    "ann_return_drag_vs_base",
                    "max_dd_reduction_vs_base",
                    "bad_day_improvement_bps",
                ],
            ]
        ),
        "",
        "## Tail Windows",
        "",
        _markdown_table(tail_focus),
        "",
        "## Interpretation",
        "",
        "- Walk-forward gating reduces the insurance drag compared with always-on funded overlays, because insurance is mostly active in watch/stress states.",
        "- A policy passes only if it improves drawdown/tail metrics without paying too much annual return drag.",
        "- Funded insurance remains cleaner than financed insurance: financed overlays preserve shared_core exposure but add gross exposure and can fail to improve bad-state returns.",
        "- This version is still rule-based validation research, not production sizing.",
    ]
    memo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rolling_compound_return(returns: pd.Series, window: int) -> pd.Series:
    return (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0


def _worst_window_dates(returns: pd.Series, window: int) -> pd.Index:
    rolling = _rolling_compound_return(returns, window)
    if rolling.dropna().empty:
        return returns.index[:0]
    end_date = rolling.idxmin()
    end_position = returns.index.get_loc(end_date)
    start_position = max(0, int(end_position) - window + 1)
    return returns.index[start_position : int(end_position) + 1]


def _best_row_payload(frame: pd.DataFrame, column: str) -> dict[str, object]:
    if frame.empty:
        return {}
    row = frame.loc[frame[column].idxmax()]
    return row.to_dict()


def _best_return_drag_constrained(frame: pd.DataFrame, *, max_drag: float = 0.03) -> dict[str, object]:
    candidates = frame.loc[frame["ann_return_drag_vs_base"].le(max_drag)].copy()
    if candidates.empty:
        return {}
    row = candidates.loc[candidates["max_dd_reduction_vs_base"].idxmax()]
    return row.to_dict()


def _paths_or_default(values: Sequence[str] | None, defaults: Sequence[Path]) -> tuple[Path, ...]:
    if values:
        return tuple(Path(value) for value in values)
    return tuple(defaults)


if __name__ == "__main__":
    raise SystemExit(main())
