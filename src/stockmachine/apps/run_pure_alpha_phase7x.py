"""Phase7X threshold robustness for walk-forward insurance overlays.

Phase7X stress-tests Phase7W's lagged state detector over a small neighboring
grid of thresholds. The purpose is not to pick the best in-sample threshold,
but to check whether the insurance conclusion is stable around the default.
"""

from __future__ import annotations

import argparse
import itertools
import json
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
    _build_candidate_returns,
    _load_external_returns,
    _load_shared_core_returns,
)
from stockmachine.apps.run_pure_alpha_phase7w import (
    POLICIES,
    InsurancePolicy,
    _build_lagged_state,
    _policy_summary,
    _run_policy_grid,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7x_insurance_threshold_robustness_20260518"
DEFAULT_POLICY_NAMES = (
    "funded_gld_5_10_20",
    "funded_gld_0_10_20",
    "funded_cash_gld_0_10_20",
    "financed_gld_0_05_10",
)
DEFAULT_DRAWDOWN_WATCH_GRID = (-0.02, -0.03, -0.04)
DEFAULT_DRAWDOWN_STRESS_GRID = (-0.05, -0.06, -0.08)
DEFAULT_ROLL20_STRESS_GRID = (-0.02, -0.03, -0.04)
DEFAULT_VOL63_STRESS_GRID = (0.12, 0.14, 0.16)
DEFAULT_SPY_DRAWDOWN_STRESS_GRID = (-0.06, -0.08, -0.10)
DEFAULT_SPY_ROLL20_STRESS_GRID = (-0.04, -0.06, -0.08)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7X insurance threshold robustness grid.")
    parser.add_argument("--shared-core-curve", default=str(DEFAULT_SHARED_CORE_CURVE))
    parser.add_argument("--shared-portfolio", default=DEFAULT_SHARED_PORTFOLIO)
    parser.add_argument("--daily-bar-path", action="append", dest="daily_bar_paths")
    parser.add_argument("--adj-factor-path", action="append", dest="adj_factor_paths")
    parser.add_argument("--benchmark-index-path", action="append", dest="benchmark_index_paths")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cash-column", default="BIL_cash")
    parser.add_argument("--overlay-cost-bps-per-traded-notional", type=float, default=1.0)
    parser.add_argument("--policy-names", default=",".join(DEFAULT_POLICY_NAMES))
    parser.add_argument("--drawdown-watch-grid", default=_format_grid(DEFAULT_DRAWDOWN_WATCH_GRID))
    parser.add_argument("--drawdown-stress-grid", default=_format_grid(DEFAULT_DRAWDOWN_STRESS_GRID))
    parser.add_argument("--roll20-stress-grid", default=_format_grid(DEFAULT_ROLL20_STRESS_GRID))
    parser.add_argument("--vol63-stress-grid", default=_format_grid(DEFAULT_VOL63_STRESS_GRID))
    parser.add_argument("--spy-drawdown-stress-grid", default=_format_grid(DEFAULT_SPY_DRAWDOWN_STRESS_GRID))
    parser.add_argument("--spy-roll20-stress-grid", default=_format_grid(DEFAULT_SPY_ROLL20_STRESS_GRID))
    args = parser.parse_args(argv)

    rollup = build_phase7x_insurance_threshold_robustness(
        shared_core_curve=Path(args.shared_core_curve),
        shared_portfolio=str(args.shared_portfolio),
        daily_bar_paths=_paths_or_default(args.daily_bar_paths, DEFAULT_DAILY_BAR_PATHS),
        adj_factor_paths=_paths_or_default(args.adj_factor_paths, DEFAULT_ADJ_FACTOR_PATHS),
        benchmark_index_paths=_paths_or_default(args.benchmark_index_paths, DEFAULT_BENCHMARK_INDEX_PATHS),
        output_root=Path(args.output_root),
        cash_column=str(args.cash_column),
        overlay_cost_bps_per_traded_notional=float(args.overlay_cost_bps_per_traded_notional),
        policies=_select_policies(args.policy_names),
        drawdown_watch_grid=_parse_grid(args.drawdown_watch_grid),
        drawdown_stress_grid=_parse_grid(args.drawdown_stress_grid),
        roll20_stress_grid=_parse_grid(args.roll20_stress_grid),
        vol63_stress_grid=_parse_grid(args.vol63_stress_grid),
        spy_drawdown_stress_grid=_parse_grid(args.spy_drawdown_stress_grid),
        spy_roll20_stress_grid=_parse_grid(args.spy_roll20_stress_grid),
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=False))
    return 0


def build_phase7x_insurance_threshold_robustness(
    *,
    shared_core_curve: Path,
    shared_portfolio: str,
    daily_bar_paths: Sequence[Path],
    adj_factor_paths: Sequence[Path],
    benchmark_index_paths: Sequence[Path],
    output_root: Path,
    cash_column: str,
    overlay_cost_bps_per_traded_notional: float,
    policies: Sequence[InsurancePolicy],
    drawdown_watch_grid: Sequence[float],
    drawdown_stress_grid: Sequence[float],
    roll20_stress_grid: Sequence[float],
    vol63_stress_grid: Sequence[float],
    spy_drawdown_stress_grid: Sequence[float],
    spy_roll20_stress_grid: Sequence[float],
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)

    shared = _load_shared_core_returns(shared_core_curve, shared_portfolio=shared_portfolio)
    required_symbols = tuple(sorted({symbol for candidate in CANDIDATES for symbol in candidate.components}))
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

    grid_rows: list[dict[str, object]] = []
    grid_iter = _threshold_grid(
        drawdown_watch_grid=drawdown_watch_grid,
        drawdown_stress_grid=drawdown_stress_grid,
        roll20_stress_grid=roll20_stress_grid,
        vol63_stress_grid=vol63_stress_grid,
        spy_drawdown_stress_grid=spy_drawdown_stress_grid,
        spy_roll20_stress_grid=spy_roll20_stress_grid,
    )
    for grid_id, thresholds in enumerate(grid_iter, start=1):
        state = _build_lagged_state(panel, thresholds=thresholds)
        policy_returns, policy_weights, policy_costs = _run_policy_grid(
            panel,
            state=state,
            policies=policies,
            cash_column=cash_column,
            overlay_cost_bps_per_traded_notional=overlay_cost_bps_per_traded_notional,
        )
        summary = _policy_summary(panel["shared_core"], policy_returns, policy_weights, policy_costs)
        rates = _state_rates(state)
        for row in summary.to_dict(orient="records"):
            payload = {
                "grid_id": grid_id,
                **thresholds,
                **rates,
                **row,
            }
            payload["passes_drag_2pct"] = bool(payload["ann_return_drag_vs_base"] <= 0.02)
            payload["passes_drag_3pct"] = bool(payload["ann_return_drag_vs_base"] <= 0.03)
            payload["passes_dd_1pct"] = bool(payload["max_dd_reduction_vs_base"] >= 0.01)
            payload["passes_dd_1p5pct"] = bool(payload["max_dd_reduction_vs_base"] >= 0.015)
            payload["passes_bad_day_positive"] = bool(payload["bad_day_improvement_bps"] > 0.0)
            payload["passes_cvar_positive"] = bool(payload["cvar05_reduction_vs_base_bps"] > 0.0)
            payload["strict_pass"] = bool(
                payload["passes_drag_2pct"]
                and payload["passes_dd_1pct"]
                and payload["passes_bad_day_positive"]
                and payload["passes_cvar_positive"]
            )
            grid_rows.append(payload)

    grid_summary = pd.DataFrame(grid_rows)
    robustness = _robustness_summary(grid_summary)
    top_configs = _top_configs(grid_summary)
    default_configs = _default_threshold_rows(grid_summary)

    grid_path = output_root / "phase7x_threshold_grid_summary.csv"
    robustness_path = output_root / "phase7x_policy_robustness.csv"
    top_path = output_root / "phase7x_top_configs.csv"
    default_path = output_root / "phase7x_default_threshold_rows.csv"
    rollup_path = output_root / "phase7x_rollup.json"
    memo_path = output_root / "phase7x_insurance_threshold_robustness_memo_zh.md"

    grid_summary.to_csv(grid_path, index=False)
    robustness.to_csv(robustness_path, index=False)
    top_configs.to_csv(top_path, index=False)
    default_configs.to_csv(default_path, index=False)

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
        "policy_names": [policy.name for policy in policies],
        "grid_config_count": int(grid_summary["grid_id"].nunique()),
        "grid_row_count": int(len(grid_summary)),
        "best_strict_policy": _best_strict_policy(robustness),
        "paths": {
            "threshold_grid_summary": str(grid_path),
            "policy_robustness": str(robustness_path),
            "top_configs": str(top_path),
            "default_threshold_rows": str(default_path),
            "memo": str(memo_path),
        },
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_memo(
        memo_path=memo_path,
        rollup=rollup,
        robustness=robustness,
        top_configs=top_configs,
        default_configs=default_configs,
    )
    return rollup


def _threshold_grid(
    *,
    drawdown_watch_grid: Sequence[float],
    drawdown_stress_grid: Sequence[float],
    roll20_stress_grid: Sequence[float],
    vol63_stress_grid: Sequence[float],
    spy_drawdown_stress_grid: Sequence[float],
    spy_roll20_stress_grid: Sequence[float],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for values in itertools.product(
        drawdown_watch_grid,
        drawdown_stress_grid,
        roll20_stress_grid,
        vol63_stress_grid,
        spy_drawdown_stress_grid,
        spy_roll20_stress_grid,
    ):
        drawdown_watch, drawdown_stress, roll20_stress, vol63_stress, spy_drawdown_stress, spy_roll20_stress = values
        if drawdown_watch <= drawdown_stress:
            continue
        rows.append(
            {
                "drawdown_watch": float(drawdown_watch),
                "drawdown_stress": float(drawdown_stress),
                "roll20_stress": float(roll20_stress),
                "vol63_stress": float(vol63_stress),
                "spy_drawdown_stress": float(spy_drawdown_stress),
                "spy_roll20_stress": float(spy_roll20_stress),
            }
        )
    return rows


def _state_rates(state: pd.DataFrame) -> dict[str, float]:
    risk_state = state["risk_state"].astype(str)
    return {
        "calm_rate": float(risk_state.eq("calm").mean()),
        "watch_rate": float(risk_state.eq("watch").mean()),
        "stress_rate": float(risk_state.eq("stress").mean()),
        "mean_risk_score": float(state["risk_score"].mean()),
    }


def _robustness_summary(grid_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy, group in grid_summary.groupby("policy", sort=False):
        row: dict[str, object] = {
            "policy": policy,
            "configs": int(len(group)),
            "strict_pass_rate": float(group["strict_pass"].mean()),
            "drag2_dd1_pass_rate": float((group["passes_drag_2pct"] & group["passes_dd_1pct"]).mean()),
            "drag3_dd1p5_pass_rate": float((group["passes_drag_3pct"] & group["passes_dd_1p5pct"]).mean()),
            "positive_dd_rate": float((group["max_dd_reduction_vs_base"] > 0.0).mean()),
            "positive_bad_day_rate": float((group["bad_day_improvement_bps"] > 0.0).mean()),
            "ann_return_median": float(group["ann_return"].median()),
            "ann_return_drag_median": float(group["ann_return_drag_vs_base"].median()),
            "ann_return_drag_p90": float(group["ann_return_drag_vs_base"].quantile(0.90)),
            "sharpe_median": float(group["sharpe"].median()),
            "max_dd_reduction_p10": float(group["max_dd_reduction_vs_base"].quantile(0.10)),
            "max_dd_reduction_median": float(group["max_dd_reduction_vs_base"].median()),
            "max_dd_reduction_p90": float(group["max_dd_reduction_vs_base"].quantile(0.90)),
            "bad_day_improvement_median_bps": float(group["bad_day_improvement_bps"].median()),
            "cvar05_reduction_median_bps": float(group["cvar05_reduction_vs_base_bps"].median()),
            "mean_overlay_weight_median": float(group["mean_overlay_weight"].median()),
            "active_rate_median": float(group["active_rate"].median()),
            "stress_rate_median": float(group["stress_rate"].median()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["strict_pass_rate", "max_dd_reduction_median", "ann_return_drag_median"],
        ascending=[False, False, True],
    )


def _top_configs(grid_summary: pd.DataFrame, *, per_policy: int = 10) -> pd.DataFrame:
    scored = grid_summary.copy()
    scored["robust_score"] = (
        scored["max_dd_reduction_vs_base"]
        + scored["worst_20d_reduction_vs_base"]
        + scored["cvar05_reduction_vs_base_bps"] / 10000.0
        - scored["ann_return_drag_vs_base"].clip(lower=0.0)
        + scored["bad_day_improvement_bps"] / 10000.0
    )
    chunks: list[pd.DataFrame] = []
    for _, group in scored.groupby("policy", sort=False):
        chunks.append(
            group.sort_values(
                ["strict_pass", "robust_score", "max_dd_reduction_vs_base"],
                ascending=[False, False, False],
            ).head(per_policy)
        )
    return pd.concat(chunks, ignore_index=True)


def _default_threshold_rows(grid_summary: pd.DataFrame) -> pd.DataFrame:
    mask = (
        np.isclose(grid_summary["drawdown_watch"], -0.03)
        & np.isclose(grid_summary["drawdown_stress"], -0.06)
        & np.isclose(grid_summary["roll20_stress"], -0.03)
        & np.isclose(grid_summary["vol63_stress"], 0.14)
        & np.isclose(grid_summary["spy_drawdown_stress"], -0.08)
        & np.isclose(grid_summary["spy_roll20_stress"], -0.06)
    )
    return grid_summary.loc[mask].copy()


def _best_strict_policy(robustness: pd.DataFrame) -> dict[str, object]:
    if robustness.empty:
        return {}
    row = robustness.iloc[0]
    return row.to_dict()


def _write_memo(
    *,
    memo_path: Path,
    rollup: Mapping[str, object],
    robustness: pd.DataFrame,
    top_configs: pd.DataFrame,
    default_configs: pd.DataFrame,
) -> None:
    top_view = top_configs.loc[
        :,
        [
            "policy",
            "grid_id",
            "drawdown_watch",
            "drawdown_stress",
            "roll20_stress",
            "vol63_stress",
            "spy_drawdown_stress",
            "spy_roll20_stress",
            "stress_rate",
            "ann_return",
            "sharpe",
            "max_dd",
            "ann_return_drag_vs_base",
            "max_dd_reduction_vs_base",
            "bad_day_improvement_bps",
            "strict_pass",
        ],
    ].head(20)
    default_view = default_configs.loc[
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
            "strict_pass",
        ],
    ]
    lines = [
        "# Phase7X 保险阈值稳健性检查",
        "",
        "Validation-only。未使用 test lockbox。",
        "",
        "## Setup",
        "",
        f"- shared_core curve: `{rollup['shared_core_curve']}`",
        f"- window: `{rollup['start']}` to `{rollup['end']}`, `{rollup['days']}` days",
        f"- grid configs: `{rollup['grid_config_count']}`",
        f"- policies: `{', '.join(str(name) for name in rollup['policy_names'])}`",
        f"- overlay cost: `{rollup['overlay_cost_bps_per_traded_notional']}` bps per traded notional",
        "",
        "Pass 定义：`ann_return_drag <= 2%`、`max_dd_improvement >= 1%`、bad-day improvement 为正、CVaR improvement 为正。",
        "",
        "## Policy Robustness",
        "",
        _markdown_table(robustness),
        "",
        "## Default Threshold Rows",
        "",
        _markdown_table(default_view),
        "",
        "## Top Configs",
        "",
        _markdown_table(top_view),
        "",
        "## Interpretation",
        "",
        "- 如果一个政策在相邻阈值网格中的 strict pass rate 高，说明保险结论不是默认阈值偶然调出来的。",
        "- `funded_gld_5_10_20` 和 `funded_gld_0_10_20` 的区别，是是否愿意支付少量 always-on 黄金保险费。",
        "- financed 版本若 pass rate 很低，即使个别阈值收益好，也不应作为第一层保险。",
    ]
    memo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_policies(value: str) -> tuple[InsurancePolicy, ...]:
    requested = tuple(part.strip() for part in str(value).split(",") if part.strip())
    by_name = {policy.name: policy for policy in POLICIES}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Unknown policies: {missing}")
    return tuple(by_name[name] for name in requested)


def _parse_grid(value: str) -> tuple[float, ...]:
    parsed = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    if not parsed:
        raise ValueError("Grid cannot be empty.")
    return parsed


def _format_grid(values: Sequence[float]) -> str:
    return ",".join(str(value) for value in values)


def _paths_or_default(values: Sequence[str] | None, defaults: Sequence[Path]) -> tuple[Path, ...]:
    if values:
        return tuple(Path(value) for value in values)
    return tuple(defaults)


if __name__ == "__main__":
    raise SystemExit(main())
