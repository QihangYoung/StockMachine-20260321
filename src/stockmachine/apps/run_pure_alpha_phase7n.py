"""Phase7N multi-factor KG regime allocation experiment.

Phase7N upgrades the reversal-only KG gate into a factor-regime allocation
matrix. It estimates walk-forward node->factor payoff reliability for the five
shared-capital factors, converts each factor's regime score into a conservative
multiplier, normalizes factor weights, and reruns the Phase7K locked-turnover
shared-capital construction.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase3 import (
    DEFAULT_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
    DEFAULT_BENCHMARK_DAILY_GLOBS,
    DEFAULT_DAILY_GLOBS,
    DEFAULT_PHASE1_MEMBERSHIP,
    DEFAULT_PHASE2_BETA_PANEL,
)
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4s import DEFAULT_CIK_MAPPING, DEFAULT_SEC_SUBMISSIONS_DIR
from stockmachine.apps.run_pure_alpha_phase4z import _load_open_to_open_returns
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
    _t_stat,
)
from stockmachine.apps.run_pure_alpha_phase7c import REPAIRED_EDGE_SPECS, RELIABLE_CORE_SOURCES
from stockmachine.apps.run_pure_alpha_phase7f import DEFAULT_HOLDING_HORIZONS, _newey_west_t_stat
from stockmachine.apps.run_pure_alpha_phase7g import (
    DEFAULT_FUNDAMENTAL_PANEL as PHASE7G_DEFAULT_FUNDAMENTAL_PANEL,
)
from stockmachine.apps.run_pure_alpha_phase7i import (
    DEFAULT_CANDIDATE_POOL_PER_SIDE,
    DEFAULT_MAX_SINGLE_NAME_SIDE_WEIGHT,
    DEFAULT_MIN_NONZERO_NAMES,
    DEFAULT_PORTFOLIOS,
    DEFAULT_SCORE_WEIGHT,
    DEFAULT_SECTOR_PENALTY,
    DEFAULT_TURNOVER_PENALTY,
    FACTOR_COLUMNS,
    PortfolioConfig,
    _add_factor_scores,
    _apply_composite_score,
    _factor_exposure_summary,
    _load_multifactor_panel,
    _metrics,
    _score_coverage,
)
from stockmachine.apps.run_pure_alpha_phase7j import (
    DEFAULT_BENCHMARK_SYMBOL,
    DEFAULT_VALIDATION_PRICE_END,
    _add_path_metrics,
    _portfolio_metrics,
    _strict_daily_rebalance_path,
    _turnover_summary,
)
from stockmachine.apps.run_pure_alpha_phase7k import (
    DEFAULT_COST_BPS_PER_SIDE,
    DEFAULT_FACTOR_MIN_HOLDS,
    BudgetedPortfolioConfig,
    _construct_locked_one_session,
    _expand_portfolios,
    _lot_summary,
    _parse_factor_min_holds,
    _parse_float_grid,
    _parse_int_grid,
)
from stockmachine.apps.run_pure_alpha_phase7l import DEFAULT_GATE_STATES_PATH


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7n_multifactor_kg_allocation_20260518"
DEFAULT_PHASE7K_ROOT = RESEARCH_ROOT / "phase7k_locked_turnover_shared_capital_20260518_tb0p15"
DEFAULT_BASELINE_METRICS_PATH = DEFAULT_PHASE7K_ROOT / "phase7k_strict_daily_metrics.csv"
DEFAULT_TURNOVER_BUDGET_GRID = (0.15,)
DEFAULT_FACTOR_HORIZONS: Mapping[str, int] = {
    "reversal_score": 10,
    "momentum_score": 20,
    "small_size_score": 20,
    "low_beta_score": 20,
    "cash_quality_score": 20,
}
DEFAULT_RELIABILITY_WINDOW = 252
DEFAULT_RELIABILITY_MIN_HISTORY = 120
DEFAULT_MULTIPLIER_LOW = 0.90
DEFAULT_MULTIPLIER_HIGH = 1.10
DEFAULT_ADAPTIVE_FACTORS = FACTOR_COLUMNS
NODE_SETS = ("all", "reliable_core")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7N multi-factor KG regime allocation.")
    parser.add_argument("--membership-path", default=str(DEFAULT_PHASE1_MEMBERSHIP))
    parser.add_argument("--beta-panel-path", default=str(DEFAULT_PHASE2_BETA_PANEL))
    parser.add_argument("--size-panel-path", default=str(DEFAULT_SIZE_PANEL))
    parser.add_argument("--non-price-panel-path", default=str(DEFAULT_NON_PRICE_PANEL))
    parser.add_argument("--fundamental-panel-path", default=str(PHASE7G_DEFAULT_FUNDAMENTAL_PANEL or DEFAULT_FUNDAMENTAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--daily-glob", action="append", dest="daily_globs")
    parser.add_argument("--adj-factor-glob", action="append", dest="adj_factor_globs")
    parser.add_argument("--benchmark-daily-glob", action="append", dest="benchmark_daily_globs")
    parser.add_argument("--benchmark-adj-factor-glob", action="append", dest="benchmark_adj_factor_globs")
    parser.add_argument("--graph-states-path", default=str(DEFAULT_GATE_STATES_PATH))
    parser.add_argument("--baseline-metrics-path", default=str(DEFAULT_BASELINE_METRICS_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--benchmark-symbol", default=DEFAULT_BENCHMARK_SYMBOL)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--eval-start", default=DEFAULT_EVAL_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--validation-price-end", default=DEFAULT_VALIDATION_PRICE_END)
    parser.add_argument("--top-bottom-quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument(
        "--holding-horizons",
        default=",".join(str(value) for value in DEFAULT_HOLDING_HORIZONS),
    )
    parser.add_argument(
        "--factor-min-holds",
        default=",".join(f"{key}:{value}" for key, value in DEFAULT_FACTOR_MIN_HOLDS.items()),
    )
    parser.add_argument(
        "--turnover-budget-grid",
        default=",".join(str(value) for value in DEFAULT_TURNOVER_BUDGET_GRID),
    )
    parser.add_argument("--node-set", choices=NODE_SETS, default="all")
    parser.add_argument(
        "--adaptive-factors",
        default=",".join(DEFAULT_ADAPTIVE_FACTORS),
        help="Comma-separated factors allowed to receive KG multipliers. Use 'all' for all factors.",
    )
    parser.add_argument("--reliability-window", type=int, default=DEFAULT_RELIABILITY_WINDOW)
    parser.add_argument("--reliability-min-history", type=int, default=DEFAULT_RELIABILITY_MIN_HISTORY)
    parser.add_argument("--multiplier-low", type=float, default=DEFAULT_MULTIPLIER_LOW)
    parser.add_argument("--multiplier-high", type=float, default=DEFAULT_MULTIPLIER_HIGH)
    parser.add_argument("--candidate-pool-per-side", type=int, default=DEFAULT_CANDIDATE_POOL_PER_SIDE)
    parser.add_argument("--max-single-name-side-weight", type=float, default=DEFAULT_MAX_SINGLE_NAME_SIDE_WEIGHT)
    parser.add_argument("--min-nonzero-names", type=int, default=DEFAULT_MIN_NONZERO_NAMES)
    parser.add_argument("--score-weight", type=float, default=DEFAULT_SCORE_WEIGHT)
    parser.add_argument("--sector-penalty", type=float, default=DEFAULT_SECTOR_PENALTY)
    parser.add_argument("--turnover-penalty", type=float, default=DEFAULT_TURNOVER_PENALTY)
    parser.add_argument("--cost-bps-per-side", type=float, default=DEFAULT_COST_BPS_PER_SIDE)
    args = parser.parse_args(argv)

    rollup = build_phase7n_multifactor_kg_allocation(
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
        benchmark_adj_factor_globs=tuple(args.benchmark_adj_factor_globs or DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS),
        graph_states_path=args.graph_states_path,
        baseline_metrics_path=args.baseline_metrics_path,
        output_root=args.output_root,
        benchmark_symbol=args.benchmark_symbol,
        variant=args.variant,
        start=args.start,
        eval_start=args.eval_start,
        end=args.end,
        validation_price_end=args.validation_price_end,
        top_bottom_quantile=args.top_bottom_quantile,
        holding_horizons=_parse_int_grid(args.holding_horizons),
        factor_min_holds=_parse_factor_min_holds(args.factor_min_holds),
        turnover_budget_grid=_parse_float_grid(args.turnover_budget_grid),
        adaptive_factors=_parse_factor_list(args.adaptive_factors),
        node_set=args.node_set,
        reliability_window=args.reliability_window,
        reliability_min_history=args.reliability_min_history,
        multiplier_low=args.multiplier_low,
        multiplier_high=args.multiplier_high,
        candidate_pool_per_side=args.candidate_pool_per_side,
        max_single_name_side_weight=args.max_single_name_side_weight,
        min_nonzero_names=args.min_nonzero_names,
        score_weight=args.score_weight,
        sector_penalty=args.sector_penalty,
        turnover_penalty=args.turnover_penalty,
        cost_bps_per_side=args.cost_bps_per_side,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


def build_phase7n_multifactor_kg_allocation(
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
    graph_states_path: str | Path = DEFAULT_GATE_STATES_PATH,
    baseline_metrics_path: str | Path = DEFAULT_BASELINE_METRICS_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    variant: str = DEFAULT_VARIANT,
    start: str = DEFAULT_START,
    eval_start: str = DEFAULT_EVAL_START,
    end: str = DEFAULT_END,
    validation_price_end: str = DEFAULT_VALIDATION_PRICE_END,
    top_bottom_quantile: float = DEFAULT_QUANTILE,
    holding_horizons: Sequence[int] = DEFAULT_HOLDING_HORIZONS,
    factor_min_holds: Mapping[str, int] = DEFAULT_FACTOR_MIN_HOLDS,
    turnover_budget_grid: Sequence[float] = DEFAULT_TURNOVER_BUDGET_GRID,
    factor_horizons: Mapping[str, int] = DEFAULT_FACTOR_HORIZONS,
    adaptive_factors: Sequence[str] = DEFAULT_ADAPTIVE_FACTORS,
    node_set: str = "all",
    reliability_window: int = DEFAULT_RELIABILITY_WINDOW,
    reliability_min_history: int = DEFAULT_RELIABILITY_MIN_HISTORY,
    multiplier_low: float = DEFAULT_MULTIPLIER_LOW,
    multiplier_high: float = DEFAULT_MULTIPLIER_HIGH,
    candidate_pool_per_side: int = DEFAULT_CANDIDATE_POOL_PER_SIDE,
    max_single_name_side_weight: float = DEFAULT_MAX_SINGLE_NAME_SIDE_WEIGHT,
    min_nonzero_names: int = DEFAULT_MIN_NONZERO_NAMES,
    score_weight: float = DEFAULT_SCORE_WEIGHT,
    sector_penalty: float = DEFAULT_SECTOR_PENALTY,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
    cost_bps_per_side: float = DEFAULT_COST_BPS_PER_SIDE,
    portfolios: Sequence[PortfolioConfig] = DEFAULT_PORTFOLIOS,
) -> dict[str, Any]:
    missing_horizons = sorted(set(factor_horizons.values()).difference(set(holding_horizons)))
    if missing_horizons:
        raise ValueError(f"holding_horizons must include factor target horizons: {missing_horizons}")
    adaptive_factor_set = _validate_adaptive_factors(adaptive_factors)
    if multiplier_low <= 0.0 or multiplier_high <= 0.0 or multiplier_low > multiplier_high:
        raise ValueError("invalid multiplier range")
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_multifactor_panel(
        membership_path=membership_path,
        beta_panel_path=beta_panel_path,
        size_panel_path=size_panel_path,
        non_price_panel_path=non_price_panel_path,
        fundamental_panel_path=fundamental_panel_path,
        cik_mapping_path=cik_mapping_path,
        sec_submissions_dir=sec_submissions_dir,
        daily_globs=daily_globs,
        adj_factor_globs=adj_factor_globs,
        benchmark_daily_globs=benchmark_daily_globs,
        benchmark_adj_factor_globs=benchmark_adj_factor_globs,
        benchmark_symbol=benchmark_symbol,
        variant=variant,
        start=start,
        end=end,
        holding_horizons=holding_horizons,
    )
    panel = _add_factor_scores(panel)
    graph = _load_graph_states(graph_states_path, node_set=node_set)
    factor_payoff = _factor_payoff_panel(
        panel,
        factor_horizons=factor_horizons,
        top_bottom_quantile=top_bottom_quantile,
    )
    factor_signal, node_factor_reliability = _factor_regime_signals(
        graph,
        factor_payoff=factor_payoff,
        factor_horizons=factor_horizons,
        reliability_window=reliability_window,
        reliability_min_history=reliability_min_history,
        multiplier_low=multiplier_low,
        multiplier_high=multiplier_high,
    )
    factor_payoff_metrics = _factor_payoff_metrics(
        factor_payoff[
            factor_payoff["session_date"].ge(pd.Timestamp(eval_start))
            & factor_payoff["session_date"].le(pd.Timestamp(end))
        ]
    )
    multiplier_summary = _multiplier_summary(factor_signal, eval_start=eval_start, end=end)

    budgeted_portfolios = _expand_portfolios(
        portfolios,
        turnover_budget_grid=turnover_budget_grid,
        turnover_penalty=turnover_penalty,
    )
    suffix = _policy_suffix(
        node_set=node_set,
        low=multiplier_low,
        high=multiplier_high,
        adaptive_factors=adaptive_factor_set,
    )
    kg_portfolios = tuple(
        BudgetedPortfolioConfig(
            portfolio=f"{config.portfolio}_{suffix}",
            base_portfolio=config.base_portfolio,
            factor_weights=dict(config.factor_weights),
            turnover_budget=float(config.turnover_budget),
            turnover_penalty=float(config.turnover_penalty),
        )
        for config in budgeted_portfolios
    )
    positions, diagnostics, skipped = _construct_factor_kg_positions(
        panel,
        factor_signal=factor_signal,
        portfolios=kg_portfolios,
        eval_start=eval_start,
        end=end,
        holding_horizons=holding_horizons,
        factor_min_holds=factor_min_holds,
        adaptive_factors=adaptive_factor_set,
        candidate_pool_per_side=candidate_pool_per_side,
        max_single_name_side_weight=max_single_name_side_weight,
        min_nonzero_names=min_nonzero_names,
        score_weight=score_weight,
        sector_penalty=sector_penalty,
        cost_bps_per_side=cost_bps_per_side,
    )
    forward_metrics = _metrics(diagnostics, skipped=skipped, holding_horizons=holding_horizons)
    factor_exposure = _factor_exposure_summary(diagnostics)
    score_coverage = _score_coverage(panel, eval_start=eval_start, end=end)

    symbols = tuple(sorted(positions["symbol"].astype(str).unique()))
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
    calendar = (
        benchmark_returns[benchmark_returns["symbol"].eq(benchmark_symbol)]["session_date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    path_positions = positions.copy()
    path_positions["session_date"] = pd.to_datetime(path_positions["session_date"])
    daily = _strict_daily_rebalance_path(
        positions=path_positions,
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        calendar=calendar,
        cost_bps_per_side=cost_bps_per_side,
    )
    curve = _add_path_metrics(daily)
    strict_metrics = _portfolio_metrics(curve)
    turnover = _turnover_summary(curve)
    lot_summary = _lot_summary(diagnostics)
    comparison = _baseline_comparison(strict_metrics, baseline_metrics_path, suffix=suffix)

    paths = {
        "panel_sample": output_dir / "phase7n_feature_target_panel_sample.csv",
        "factor_payoff": output_dir / "phase7n_factor_payoff_panel.csv",
        "factor_payoff_metrics": output_dir / "phase7n_factor_payoff_metrics.csv",
        "factor_signal": output_dir / "phase7n_factor_regime_signal.csv",
        "node_factor_reliability": output_dir / "phase7n_node_factor_reliability.csv",
        "multiplier_summary": output_dir / "phase7n_multiplier_summary.csv",
        "positions": output_dir / "phase7n_factor_kg_positions.csv.gz",
        "diagnostics": output_dir / "phase7n_daily_diagnostics.csv",
        "skipped": output_dir / "phase7n_skipped_sessions.csv",
        "forward_metrics": output_dir / "phase7n_forward_label_metrics.csv",
        "factor_exposure": output_dir / "phase7n_factor_exposure_summary.csv",
        "score_coverage": output_dir / "phase7n_score_coverage.csv",
        "curve": output_dir / "phase7n_strict_daily_curve.csv",
        "strict_metrics": output_dir / "phase7n_strict_daily_metrics.csv",
        "turnover": output_dir / "phase7n_turnover_summary.csv",
        "lot_summary": output_dir / "phase7n_lot_summary.csv",
        "comparison": output_dir / "phase7n_vs_phase7k_net_comparison.csv",
        "memo": output_dir / "phase7n_multifactor_kg_memo.md",
        "rollup": output_dir / "phase7n_rollup.json",
    }
    panel.head(5000).to_csv(paths["panel_sample"], index=False)
    factor_payoff.to_csv(paths["factor_payoff"], index=False)
    factor_payoff_metrics.to_csv(paths["factor_payoff_metrics"], index=False)
    factor_signal.to_csv(paths["factor_signal"], index=False)
    node_factor_reliability.to_csv(paths["node_factor_reliability"], index=False)
    multiplier_summary.to_csv(paths["multiplier_summary"], index=False)
    positions.to_csv(paths["positions"], index=False, compression="gzip")
    diagnostics.to_csv(paths["diagnostics"], index=False)
    skipped.to_csv(paths["skipped"], index=False)
    forward_metrics.to_csv(paths["forward_metrics"], index=False)
    factor_exposure.to_csv(paths["factor_exposure"], index=False)
    score_coverage.to_csv(paths["score_coverage"], index=False)
    curve.to_csv(paths["curve"], index=False)
    strict_metrics.to_csv(paths["strict_metrics"], index=False)
    turnover.to_csv(paths["turnover"], index=False)
    lot_summary.to_csv(paths["lot_summary"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    paths["memo"].write_text(
        _memo(
            graph_states_path=Path(graph_states_path),
            baseline_metrics_path=Path(baseline_metrics_path),
            node_set=node_set,
            adaptive_factors=adaptive_factor_set,
            factor_horizons=factor_horizons,
            multiplier_low=multiplier_low,
            multiplier_high=multiplier_high,
            eval_start=eval_start,
            end=end,
            cost_bps_per_side=cost_bps_per_side,
            strict_metrics=strict_metrics,
            comparison=comparison,
            factor_payoff_metrics=factor_payoff_metrics,
            multiplier_summary=multiplier_summary,
            node_factor_reliability=node_factor_reliability,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "phase": "phase7n_multifactor_kg_allocation",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "variant": variant,
        "start": start,
        "eval_start": eval_start,
        "end": end,
        "validation_price_end": validation_price_end,
        "graph_states_path": Path(graph_states_path).as_posix(),
        "baseline_metrics_path": Path(baseline_metrics_path).as_posix(),
        "node_set": node_set,
        "adaptive_factors": list(adaptive_factor_set),
        "factor_horizons": {key: int(value) for key, value in factor_horizons.items()},
        "reliability_window": int(reliability_window),
        "reliability_min_history": int(reliability_min_history),
        "multiplier_low": float(multiplier_low),
        "multiplier_high": float(multiplier_high),
        "turnover_budget_grid": [float(value) for value in turnover_budget_grid],
        "factor_min_holds": dict(factor_min_holds),
        "cost_bps_per_side": float(cost_bps_per_side),
        "rows": {
            "panel": int(len(panel)),
            "factor_payoff": int(len(factor_payoff)),
            "factor_signal": int(len(factor_signal)),
            "positions": int(len(positions)),
            "diagnostics": int(len(diagnostics)),
            "skipped": int(len(skipped)),
            "curve": int(len(curve)),
            "strict_metrics": int(len(strict_metrics)),
        },
        "outputs": {key: value.as_posix() for key, value in paths.items() if key != "rollup"},
        "method": "walk_forward_node_factor_reliability_to_factor_weight_multipliers",
        "label_contract": {
            "factor_payoff": "top-minus-bottom beta-residual forward return by factor-specific horizon",
            "reliability": "rolling corr uses node and factor payoff shifted by factor horizon",
            "allocation": "base factor weights multiplied by factor multipliers and normalized",
        },
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    paths["rollup"].write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_graph_states(path: str | Path, *, node_set: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["session_date"])
    nodes = _node_specs(node_set)
    required = {"session_date", *(f"z_{source}" for source, _ in nodes)}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"graph states missing required columns: {missing}")
    keep = ["session_date", *(f"z_{source}" for source, _ in nodes)]
    out = frame[keep].copy().sort_values("session_date").drop_duplicates("session_date")
    for source, _ in nodes:
        out[f"z_{source}"] = pd.to_numeric(out[f"z_{source}"], errors="coerce")
    return out.reset_index(drop=True)


def _node_specs(node_set: str) -> tuple[tuple[str, float], ...]:
    if node_set == "all":
        specs = REPAIRED_EDGE_SPECS
    elif node_set == "reliable_core":
        specs = tuple(edge for edge in REPAIRED_EDGE_SPECS if edge.source in RELIABLE_CORE_SOURCES)
    else:
        raise ValueError(f"unsupported node_set: {node_set}")
    return tuple((edge.source, float(edge.weight)) for edge in specs)


def _factor_payoff_panel(
    panel: pd.DataFrame,
    *,
    factor_horizons: Mapping[str, int],
    top_bottom_quantile: float,
    min_names: int = 100,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = panel.copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    for session_date, group in frame.groupby("session_date", sort=True):
        for factor, horizon in factor_horizons.items():
            target_column = f"forward_beta_residual_return_h{horizon}"
            if factor not in group.columns or target_column not in group.columns:
                continue
            subset = group[[factor, target_column]].copy()
            subset[factor] = pd.to_numeric(subset[factor], errors="coerce")
            subset[target_column] = pd.to_numeric(subset[target_column], errors="coerce")
            subset = subset.dropna(subset=[factor, target_column])
            if len(subset) < min_names:
                continue
            low = subset[factor].quantile(top_bottom_quantile)
            high = subset[factor].quantile(1.0 - top_bottom_quantile)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                continue
            long = subset[subset[factor] >= high]
            short = subset[subset[factor] <= low]
            if long.empty or short.empty:
                continue
            payoff = float(long[target_column].mean() - short[target_column].mean())
            rows.append(
                {
                    "session_date": session_date,
                    "factor": factor,
                    "horizon": int(horizon),
                    "names": int(len(subset)),
                    "long_names": int(len(long)),
                    "short_names": int(len(short)),
                    "long_score_mean": float(long[factor].mean()),
                    "short_score_mean": float(short[factor].mean()),
                    "long_target_mean": float(long[target_column].mean()),
                    "short_target_mean": float(short[target_column].mean()),
                    "payoff": payoff,
                    "payoff_bps": payoff * 10000.0,
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["session_date", "factor"]).reset_index(drop=True)


def _factor_regime_signals(
    graph: pd.DataFrame,
    *,
    factor_payoff: pd.DataFrame,
    factor_horizons: Mapping[str, int],
    reliability_window: int,
    reliability_min_history: int,
    multiplier_low: float,
    multiplier_high: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = tuple((column[2:], 1.0) for column in graph.columns if column.startswith("z_"))
    node_weights = {source: _edge_weight(source) for source, _ in nodes}
    frame = graph.copy().sort_values("session_date").reset_index(drop=True)
    payoff_wide = factor_payoff.pivot(index="session_date", columns="factor", values="payoff_bps").reset_index()
    frame = frame.merge(payoff_wide, on="session_date", how="left")
    signal_rows: list[pd.DataFrame] = []
    reliability_rows: list[dict[str, Any]] = []
    weight_scale = float(np.sqrt(sum(weight**2 for weight in node_weights.values()))) or 1.0

    for factor, horizon in factor_horizons.items():
        payoff = pd.to_numeric(frame.get(factor), errors="coerce")
        known_payoff = payoff.shift(int(horizon))
        score = pd.Series(0.0, index=frame.index, dtype=float)
        abs_score = pd.Series(0.0, index=frame.index, dtype=float)
        for source, weight in node_weights.items():
            z_col = f"z_{source}"
            z = pd.to_numeric(frame[z_col], errors="coerce")
            known_z = z.shift(int(horizon))
            corr = (
                known_z.rolling(reliability_window, min_periods=reliability_min_history)
                .corr(known_payoff)
                .replace([np.inf, -np.inf], np.nan)
                .clip(-1.0, 1.0)
            )
            contribution = z.fillna(0.0) * corr.fillna(0.0) * float(weight)
            score = score + contribution
            abs_score = abs_score + contribution.abs()
            eval_corr = corr.dropna()
            reliability_rows.append(
                {
                    "factor": factor,
                    "horizon": int(horizon),
                    "node": source,
                    "node_weight": float(weight),
                    "corr_observations": int(len(eval_corr)),
                    "mean_corr": float(eval_corr.mean()) if len(eval_corr) else np.nan,
                    "median_corr": float(eval_corr.median()) if len(eval_corr) else np.nan,
                    "mean_abs_corr": float(eval_corr.abs().mean()) if len(eval_corr) else np.nan,
                    "positive_corr_rate": float((eval_corr > 0.0).mean()) if len(eval_corr) else np.nan,
                    "mean_contribution": float(contribution.mean()),
                    "mean_abs_contribution": float(contribution.abs().mean()),
                    "test_window_used": False,
                }
            )

        regime_score = score / weight_scale
        historical = regime_score.shift(1)
        rolling = historical.rolling(reliability_window, min_periods=reliability_min_history)
        q33 = rolling.quantile(1.0 / 3.0)
        q67 = rolling.quantile(2.0 / 3.0)
        ready = regime_score.notna() & q33.notna() & q67.notna()
        multiplier = pd.Series(1.0, index=frame.index, dtype=float)
        multiplier = multiplier.where(~(ready & regime_score.le(q33)), float(multiplier_low))
        multiplier = multiplier.where(~(ready & regime_score.ge(q67)), float(multiplier_high))
        signal_rows.append(
            pd.DataFrame(
                {
                    "session_date": frame["session_date"],
                    "factor": factor,
                    "horizon": int(horizon),
                    "factor_payoff_bps": payoff,
                    "factor_regime_score": regime_score,
                    "factor_regime_abs_score": abs_score / weight_scale,
                    "factor_regime_q33": q33,
                    "factor_regime_q67": q67,
                    "factor_regime_ready": ready,
                    "factor_multiplier": multiplier,
                    "test_window_used": False,
                }
            )
        )
    signals = pd.concat(signal_rows, ignore_index=True).sort_values(["session_date", "factor"]).reset_index(drop=True)
    reliability = pd.DataFrame(reliability_rows).sort_values(["factor", "mean_abs_corr"], ascending=[True, False])
    return signals, reliability


def _edge_weight(source: str) -> float:
    for edge in REPAIRED_EDGE_SPECS:
        if edge.source == source:
            return float(edge.weight)
    return 1.0


def _construct_factor_kg_positions(
    panel: pd.DataFrame,
    *,
    factor_signal: pd.DataFrame,
    portfolios: Sequence[BudgetedPortfolioConfig],
    eval_start: str,
    end: str,
    holding_horizons: Sequence[int],
    factor_min_holds: Mapping[str, int],
    adaptive_factors: Sequence[str],
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    min_nonzero_names: int,
    score_weight: float,
    sector_penalty: float,
    cost_bps_per_side: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    adaptive_factor_set = set(adaptive_factors)
    frame = panel[panel["session_date"].ge(eval_start) & panel["session_date"].le(end)].copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    sessions = list(frame.groupby("session_date", sort=True))
    signal_lookup = _factor_signal_lookup(factor_signal)
    for config in portfolios:
        ledger: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
        for session_idx, (session_date, group) in enumerate(sessions):
            signal_row = signal_lookup.get(pd.Timestamp(session_date), {})
            multipliers = {
                factor: (
                    float(signal_row.get(f"{factor}_multiplier", 1.0))
                    if factor in adaptive_factor_set
                    else 1.0
                )
                for factor in FACTOR_COLUMNS
            }
            effective_weights = _apply_factor_multipliers(config.factor_weights, multipliers)
            effective_config = BudgetedPortfolioConfig(
                portfolio=config.portfolio,
                base_portfolio=config.base_portfolio,
                factor_weights=effective_weights,
                turnover_budget=float(config.turnover_budget),
                turnover_penalty=float(config.turnover_penalty),
            )
            scored = _apply_composite_score(group, effective_config.factor_weights)
            book, diagnostic, skip, ledger = _construct_locked_one_session(
                scored,
                session_date=str(session_date.date()),
                session_idx=session_idx,
                config=effective_config,
                holding_horizons=holding_horizons,
                factor_min_holds=factor_min_holds,
                candidate_pool_per_side=candidate_pool_per_side,
                max_single_name_side_weight=max_single_name_side_weight,
                min_nonzero_names=min_nonzero_names,
                score_weight=score_weight,
                sector_penalty=sector_penalty,
                cost_bps_per_side=cost_bps_per_side,
                previous_ledger=ledger,
            )
            positions.extend(book)
            if diagnostic is not None:
                diagnostic.update(
                    {
                        "base_factor_weights_json": json.dumps(dict(config.factor_weights), sort_keys=True),
                        "effective_factor_weights_json": json.dumps(dict(effective_weights), sort_keys=True),
                        "factor_multipliers_json": json.dumps(dict(multipliers), sort_keys=True),
                        "mean_factor_multiplier": float(np.mean(list(multipliers.values()))),
                        "min_factor_multiplier": float(np.min(list(multipliers.values()))),
                        "max_factor_multiplier": float(np.max(list(multipliers.values()))),
                    }
                )
                for factor in FACTOR_COLUMNS:
                    diagnostic[f"{factor}_multiplier"] = float(multipliers.get(factor, 1.0))
                    diagnostic[f"{factor}_regime_score"] = _safe_float(
                        signal_row.get(f"{factor}_regime_score", np.nan)
                    )
                diagnostics.append(diagnostic)
            if skip is not None:
                skipped.append(skip)
    return pd.DataFrame(positions), pd.DataFrame(diagnostics), pd.DataFrame(skipped)


def _factor_signal_lookup(factor_signal: pd.DataFrame) -> dict[pd.Timestamp, dict[str, float]]:
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for session_date, group in factor_signal.groupby("session_date", sort=True):
        item: dict[str, float] = {}
        for _, row in group.iterrows():
            factor = str(row["factor"])
            item[f"{factor}_multiplier"] = _safe_float(row.get("factor_multiplier", 1.0))
            item[f"{factor}_regime_score"] = _safe_float(row.get("factor_regime_score", np.nan))
        rows[pd.Timestamp(session_date)] = item
    return rows


def _apply_factor_multipliers(
    base_weights: Mapping[str, float],
    multipliers: Mapping[str, float],
) -> dict[str, float]:
    base = {factor: float(weight) for factor, weight in base_weights.items()}
    total = sum(max(weight, 0.0) for weight in base.values())
    scaled = {
        factor: max(weight, 0.0) * max(float(multipliers.get(factor, 1.0)), 0.0)
        for factor, weight in base.items()
    }
    scaled_total = sum(scaled.values())
    if total <= 0.0 or scaled_total <= 0.0:
        return base
    return {factor: value * total / scaled_total for factor, value in scaled.items()}


def _factor_payoff_metrics(payoff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (factor, horizon), group in payoff.groupby(["factor", "horizon"], sort=True):
        series = pd.to_numeric(group["payoff"], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append(
            {
                "factor": factor,
                "horizon": int(horizon),
                "sessions": int(len(series)),
                "mean_payoff_bps": float(series.mean() * 10000.0),
                "median_payoff_bps": float(series.median() * 10000.0),
                "hit_rate": float((series > 0.0).mean()),
                "t_stat": _t_stat(series),
                "newey_west_lag": int(max(0, int(horizon) - 1)),
                "newey_west_t_stat": _newey_west_t_stat(series, lag=max(0, int(horizon) - 1)),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "mean_payoff_bps"], ascending=[True, False])


def _multiplier_summary(factor_signal: pd.DataFrame, *, eval_start: str, end: str) -> pd.DataFrame:
    frame = factor_signal[
        factor_signal["session_date"].ge(pd.Timestamp(eval_start))
        & factor_signal["session_date"].le(pd.Timestamp(end))
    ].copy()
    rows = []
    for factor, group in frame.groupby("factor", sort=True):
        multiplier = pd.to_numeric(group["factor_multiplier"], errors="coerce")
        ready = group["factor_regime_ready"].astype(bool)
        rows.append(
            {
                "factor": factor,
                "sessions": int(len(group)),
                "ready_sessions": int(ready.sum()),
                "ready_rate": float(ready.mean()),
                "mean_multiplier": float(multiplier.mean()),
                "min_multiplier_rate": float((multiplier < 1.0).mean()),
                "neutral_multiplier_rate": float((multiplier == 1.0).mean()),
                "max_multiplier_rate": float((multiplier > 1.0).mean()),
                "mean_regime_score": float(pd.to_numeric(group["factor_regime_score"], errors="coerce").mean()),
                "test_window_used": bool(group["test_window_used"].any()),
            }
        )
    return pd.DataFrame(rows)


def _baseline_comparison(metrics: pd.DataFrame, baseline_metrics_path: str | Path, *, suffix: str) -> pd.DataFrame:
    path = Path(baseline_metrics_path)
    if not path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(path)
    baseline = baseline[baseline["return_kind"].eq("net")].copy()
    overlay = metrics[metrics["return_kind"].eq("net")].copy()
    if baseline.empty or overlay.empty:
        return pd.DataFrame()
    overlay["base_portfolio"] = overlay["portfolio"].str.replace(f"_{suffix}", "", regex=False)
    baseline = baseline.rename(columns={"portfolio": "base_portfolio"})
    columns = [
        "final_equity",
        "annualized_return",
        "annualized_vol",
        "sharpe_no_rf",
        "max_drawdown",
        "mean_daily_return_bps",
        "mean_turnover",
        "mean_cost_bps",
        "mean_gross_exposure",
        "realized_beta_net",
    ]
    available = [column for column in columns if column in overlay.columns and column in baseline.columns]
    merged = overlay[["portfolio", "base_portfolio", *available]].merge(
        baseline[["base_portfolio", *available]],
        on="base_portfolio",
        suffixes=("_kg", "_baseline"),
    )
    for column in available:
        merged[f"delta_{column}"] = merged[f"{column}_kg"] - merged[f"{column}_baseline"]
    return merged


def _policy_suffix(*, node_set: str, low: float, high: float, adaptive_factors: Sequence[str]) -> str:
    adaptive_set = set(adaptive_factors)
    if adaptive_set == set(FACTOR_COLUMNS):
        factor_label = "allf"
    elif adaptive_set == {"small_size_score", "low_beta_score", "cash_quality_score"}:
        factor_label = "slowf"
    else:
        factor_label = "customf" + str(len(adaptive_set))
    return f"kg5f_{node_set}_{factor_label}_{_float_label(low)}_{_float_label(high)}"


def _validate_adaptive_factors(adaptive_factors: Sequence[str]) -> tuple[str, ...]:
    valid = set(FACTOR_COLUMNS)
    parsed = tuple(dict.fromkeys(str(factor).strip() for factor in adaptive_factors if str(factor).strip()))
    if not parsed:
        raise ValueError("adaptive_factors cannot be empty")
    missing = sorted(set(parsed).difference(valid))
    if missing:
        raise ValueError(f"unknown adaptive factors: {missing}")
    return parsed


def _parse_factor_list(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return tuple(FACTOR_COLUMNS)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _float_label(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _memo(
    *,
    graph_states_path: Path,
    baseline_metrics_path: Path,
    node_set: str,
    adaptive_factors: Sequence[str],
    factor_horizons: Mapping[str, int],
    multiplier_low: float,
    multiplier_high: float,
    eval_start: str,
    end: str,
    cost_bps_per_side: float,
    strict_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    factor_payoff_metrics: pd.DataFrame,
    multiplier_summary: pd.DataFrame,
    node_factor_reliability: pd.DataFrame,
) -> str:
    net = strict_metrics[strict_metrics["return_kind"].eq("net")].copy()
    metric_columns = [
        column
        for column in (
            "portfolio",
            "final_equity",
            "annualized_return",
            "annualized_vol",
            "sharpe_no_rf",
            "max_drawdown",
            "mean_turnover",
            "mean_cost_bps",
            "realized_beta_net",
        )
        if column in net.columns
    ]
    compare_columns = [
        column
        for column in (
            "portfolio",
            "base_portfolio",
            "final_equity_kg",
            "final_equity_baseline",
            "delta_final_equity",
            "annualized_return_kg",
            "annualized_return_baseline",
            "delta_annualized_return",
            "sharpe_no_rf_kg",
            "sharpe_no_rf_baseline",
            "delta_sharpe_no_rf",
            "max_drawdown_kg",
            "max_drawdown_baseline",
            "delta_max_drawdown",
        )
        if column in comparison.columns
    ]
    reliability_view = node_factor_reliability.sort_values(
        ["factor", "mean_abs_corr"],
        ascending=[True, False],
    ).groupby("factor", group_keys=False).head(4)
    lines = [
        "# Phase7N Multi-Factor KG Allocation",
        "",
        f"Generated: {_utc_now()}",
        "",
        "Validation-only. The test lockbox is not used.",
        "",
        "## Setup",
        "",
        f"- graph states: `{graph_states_path.as_posix()}`",
        f"- baseline metrics: `{baseline_metrics_path.as_posix()}`",
        f"- node set: `{node_set}`",
        f"- adaptive factors: `{list(adaptive_factors)}`",
        f"- factor horizons: `{dict(factor_horizons)}`",
        f"- factor multiplier range: `{multiplier_low}` to `{multiplier_high}`",
        f"- evaluation window: `{eval_start}` through `{end}`",
        f"- cost bps per side: `{cost_bps_per_side}`",
        "",
        "## Strict Daily Net Metrics",
        "",
        _markdown_table(net[metric_columns] if metric_columns else net),
        "",
        "## Versus Phase7K Baseline",
        "",
        _markdown_table(comparison[compare_columns] if compare_columns else comparison),
        "",
        "## Factor Payoff Labels",
        "",
        _markdown_table(factor_payoff_metrics),
        "",
        "## Multiplier Summary",
        "",
        _markdown_table(multiplier_summary),
        "",
        "## Top Node-Factor Reliability",
        "",
        _markdown_table(reliability_view),
        "",
        "## Notes",
        "",
        "- Node-factor reliability uses only labels lagged by each factor's payoff horizon.",
        "- The allocation acts on factor signal weights, not on gross exposure.",
        "- Factor weights are normalized back to the original total factor-weight budget each day.",
        "",
    ]
    return "\n".join(lines)


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
