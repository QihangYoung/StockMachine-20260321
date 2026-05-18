"""Phase7P common-regime-risk capped shared-core portfolios.

Phase7P keeps the Phase7K shared-capital signal and locked-lot mechanics, but
adds a soft cap on exposure to a common factor-payoff axis. The first axis is
the diagnostic PC1 from Phase7O's five-factor payoff panel, used here as a
validation-only common-regime risk control.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog

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
    DEFAULT_SIZE_PANEL,
    DEFAULT_START,
    DEFAULT_VARIANT,
    _markdown_table,
)
from stockmachine.apps.run_pure_alpha_phase7f import DEFAULT_HOLDING_HORIZONS
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
    _group_exposure_matrix,
    _load_multifactor_panel,
    _metrics,
    _score_coverage,
    _zscore_array,
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
    DEFAULT_TURNOVER_BUDGET_GRID,
    EPS,
    BudgetedPortfolioConfig,
    _candidate_union_locked,
    _carry_previous,
    _diagnostic_row_locked,
    _expand_portfolios,
    _is_empty_book,
    _ledger_weights,
    _locked_weights,
    _lot_summary,
    _parse_factor_min_holds,
    _parse_float_grid,
    _parse_int_grid,
    _position_rows,
    _prune_ledger,
    _sanitize_locked_lower,
    _side_turnover,
    _skip_row,
    _target_dict,
    _update_side_ledger,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7p_common_regime_risk_cap_20260518"
DEFAULT_BASELINE_METRICS_PATH = (
    RESEARCH_ROOT
    / "phase7k_locked_turnover_shared_capital_20260518_tb0p15"
    / "phase7k_strict_daily_metrics.csv"
)
COMMON_AXIS_COLUMN = "common_regime_pc1_score"
DEFAULT_COMMON_CAP_GRID = (1.0, 1.25)
DEFAULT_COMMON_PENALTY = 5.0
DEFAULT_COMMON_AXIS_WEIGHTS: Mapping[str, float] = {
    "cash_quality_score": -0.16047168171918963,
    "low_beta_score": -0.6350121521889823,
    "momentum_score": -0.30140278767631523,
    "reversal_score": 0.23359509912476195,
    "small_size_score": 0.6523787973136239,
}


@dataclass(frozen=True)
class CommonCappedPortfolioConfig:
    portfolio: str
    base_portfolio: str
    factor_weights: Mapping[str, float]
    turnover_budget: float
    turnover_penalty: float
    common_cap: float
    common_penalty: float


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7P common-regime-risk capped portfolios.")
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
    parser.add_argument("--baseline-metrics-path", default=str(DEFAULT_BASELINE_METRICS_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--benchmark-symbol", default=DEFAULT_BENCHMARK_SYMBOL)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--eval-start", default=DEFAULT_EVAL_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--validation-price-end", default=DEFAULT_VALIDATION_PRICE_END)
    parser.add_argument(
        "--holding-horizons",
        default=",".join(str(value) for value in DEFAULT_HOLDING_HORIZONS),
    )
    parser.add_argument(
        "--turnover-budget-grid",
        default=",".join(str(value) for value in DEFAULT_TURNOVER_BUDGET_GRID),
    )
    parser.add_argument(
        "--factor-min-holds",
        default=",".join(f"{key}:{value}" for key, value in DEFAULT_FACTOR_MIN_HOLDS.items()),
    )
    parser.add_argument(
        "--common-cap-grid",
        default=",".join(str(value) for value in DEFAULT_COMMON_CAP_GRID),
    )
    parser.add_argument("--common-penalty", type=float, default=DEFAULT_COMMON_PENALTY)
    parser.add_argument("--candidate-pool-per-side", type=int, default=DEFAULT_CANDIDATE_POOL_PER_SIDE)
    parser.add_argument("--max-single-name-side-weight", type=float, default=DEFAULT_MAX_SINGLE_NAME_SIDE_WEIGHT)
    parser.add_argument("--min-nonzero-names", type=int, default=DEFAULT_MIN_NONZERO_NAMES)
    parser.add_argument("--score-weight", type=float, default=DEFAULT_SCORE_WEIGHT)
    parser.add_argument("--sector-penalty", type=float, default=DEFAULT_SECTOR_PENALTY)
    parser.add_argument("--turnover-penalty", type=float, default=DEFAULT_TURNOVER_PENALTY)
    parser.add_argument("--cost-bps-per-side", type=float, default=DEFAULT_COST_BPS_PER_SIDE)
    args = parser.parse_args(argv)

    rollup = build_phase7p_common_regime_risk_cap(
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
        baseline_metrics_path=args.baseline_metrics_path,
        output_root=args.output_root,
        benchmark_symbol=args.benchmark_symbol,
        variant=args.variant,
        start=args.start,
        eval_start=args.eval_start,
        end=args.end,
        validation_price_end=args.validation_price_end,
        holding_horizons=_parse_int_grid(args.holding_horizons),
        turnover_budget_grid=_parse_float_grid(args.turnover_budget_grid),
        factor_min_holds=_parse_factor_min_holds(args.factor_min_holds),
        common_cap_grid=_parse_float_grid(args.common_cap_grid),
        common_penalty=args.common_penalty,
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


def build_phase7p_common_regime_risk_cap(
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
    baseline_metrics_path: str | Path = DEFAULT_BASELINE_METRICS_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    variant: str = DEFAULT_VARIANT,
    start: str = DEFAULT_START,
    eval_start: str = DEFAULT_EVAL_START,
    end: str = DEFAULT_END,
    validation_price_end: str = DEFAULT_VALIDATION_PRICE_END,
    holding_horizons: Sequence[int] = DEFAULT_HOLDING_HORIZONS,
    turnover_budget_grid: Sequence[float] = DEFAULT_TURNOVER_BUDGET_GRID,
    factor_min_holds: Mapping[str, int] = DEFAULT_FACTOR_MIN_HOLDS,
    common_cap_grid: Sequence[float] = DEFAULT_COMMON_CAP_GRID,
    common_penalty: float = DEFAULT_COMMON_PENALTY,
    candidate_pool_per_side: int = DEFAULT_CANDIDATE_POOL_PER_SIDE,
    max_single_name_side_weight: float = DEFAULT_MAX_SINGLE_NAME_SIDE_WEIGHT,
    min_nonzero_names: int = DEFAULT_MIN_NONZERO_NAMES,
    score_weight: float = DEFAULT_SCORE_WEIGHT,
    sector_penalty: float = DEFAULT_SECTOR_PENALTY,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
    cost_bps_per_side: float = DEFAULT_COST_BPS_PER_SIDE,
    portfolios: Sequence[PortfolioConfig] = DEFAULT_PORTFOLIOS,
) -> dict[str, Any]:
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
    panel = _add_common_axis(_add_factor_scores(panel), DEFAULT_COMMON_AXIS_WEIGHTS)
    capped_portfolios = _expand_common_capped_portfolios(
        portfolios,
        turnover_budget_grid=turnover_budget_grid,
        turnover_penalty=turnover_penalty,
        common_cap_grid=common_cap_grid,
        common_penalty=common_penalty,
    )
    positions, diagnostics, skipped = _construct_common_capped_positions(
        panel,
        portfolios=capped_portfolios,
        eval_start=eval_start,
        end=end,
        holding_horizons=holding_horizons,
        factor_min_holds=factor_min_holds,
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
    common_summary = _common_exposure_summary(diagnostics)

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
    comparison = _baseline_comparison(strict_metrics, baseline_metrics_path)

    paths = {
        "panel_sample": output_dir / "phase7p_feature_target_panel_sample.csv",
        "positions": output_dir / "phase7p_common_capped_positions.csv.gz",
        "diagnostics": output_dir / "phase7p_daily_diagnostics.csv",
        "skipped": output_dir / "phase7p_skipped_sessions.csv",
        "forward_metrics": output_dir / "phase7p_forward_label_metrics.csv",
        "factor_exposure": output_dir / "phase7p_factor_exposure_summary.csv",
        "score_coverage": output_dir / "phase7p_score_coverage.csv",
        "curve": output_dir / "phase7p_strict_daily_curve.csv",
        "strict_metrics": output_dir / "phase7p_strict_daily_metrics.csv",
        "turnover": output_dir / "phase7p_turnover_summary.csv",
        "lot_summary": output_dir / "phase7p_lot_summary.csv",
        "common_summary": output_dir / "phase7p_common_exposure_summary.csv",
        "comparison": output_dir / "phase7p_vs_phase7k_net_comparison.csv",
        "memo": output_dir / "phase7p_common_regime_risk_cap_memo.md",
        "rollup": output_dir / "phase7p_rollup.json",
    }
    panel.head(5000).to_csv(paths["panel_sample"], index=False)
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
    common_summary.to_csv(paths["common_summary"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    paths["memo"].write_text(
        _memo(
            baseline_metrics_path=Path(baseline_metrics_path),
            common_axis_weights=DEFAULT_COMMON_AXIS_WEIGHTS,
            common_cap_grid=common_cap_grid,
            common_penalty=common_penalty,
            eval_start=eval_start,
            end=end,
            cost_bps_per_side=cost_bps_per_side,
            strict_metrics=strict_metrics,
            comparison=comparison,
            common_summary=common_summary,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "phase": "phase7p_common_regime_risk_cap",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "variant": variant,
        "start": start,
        "eval_start": eval_start,
        "end": end,
        "validation_price_end": validation_price_end,
        "baseline_metrics_path": Path(baseline_metrics_path).as_posix(),
        "common_axis_column": COMMON_AXIS_COLUMN,
        "common_axis_weights": dict(DEFAULT_COMMON_AXIS_WEIGHTS),
        "common_cap_grid": [float(value) for value in common_cap_grid],
        "common_penalty": float(common_penalty),
        "turnover_budget_grid": [float(value) for value in turnover_budget_grid],
        "factor_min_holds": dict(factor_min_holds),
        "cost_bps_per_side": float(cost_bps_per_side),
        "rows": {
            "panel": int(len(panel)),
            "positions": int(len(positions)),
            "diagnostics": int(len(diagnostics)),
            "skipped": int(len(skipped)),
            "curve": int(len(curve)),
            "strict_metrics": int(len(strict_metrics)),
        },
        "outputs": {key: value.as_posix() for key, value in paths.items() if key != "rollup"},
        "method": "phase7k_locked_turnover_with_soft_common_pc1_exposure_cap",
        "limitations": [
            "The common axis uses Phase7O diagnostic PC1 loadings and is validation-only.",
            "The cap is soft via a slack penalty to avoid conflicts with locked lots.",
            "This is a risk-control test, not a return forecast model.",
        ],
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    paths["rollup"].write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _add_common_axis(panel: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    frame = panel.copy()
    score = np.zeros(len(frame), dtype=float)
    for factor, weight in weights.items():
        score += float(weight) * pd.to_numeric(frame[factor], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    frame[COMMON_AXIS_COLUMN] = score
    return frame


def _expand_common_capped_portfolios(
    portfolios: Sequence[PortfolioConfig],
    *,
    turnover_budget_grid: Sequence[float],
    turnover_penalty: float,
    common_cap_grid: Sequence[float],
    common_penalty: float,
) -> tuple[CommonCappedPortfolioConfig, ...]:
    budgeted = _expand_portfolios(
        portfolios,
        turnover_budget_grid=turnover_budget_grid,
        turnover_penalty=turnover_penalty,
    )
    rows: list[CommonCappedPortfolioConfig] = []
    for config in budgeted:
        for cap in common_cap_grid:
            rows.append(
                CommonCappedPortfolioConfig(
                    portfolio=f"{config.portfolio}_pc1cap{_float_label(cap)}",
                    base_portfolio=config.base_portfolio,
                    factor_weights=dict(config.factor_weights),
                    turnover_budget=float(config.turnover_budget),
                    turnover_penalty=float(config.turnover_penalty),
                    common_cap=float(cap),
                    common_penalty=float(common_penalty),
                )
            )
    return tuple(rows)


def _construct_common_capped_positions(
    panel: pd.DataFrame,
    *,
    portfolios: Sequence[CommonCappedPortfolioConfig],
    eval_start: str,
    end: str,
    holding_horizons: Sequence[int],
    factor_min_holds: Mapping[str, int],
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
    frame = panel[panel["session_date"].ge(eval_start) & panel["session_date"].le(end)].copy()
    sessions = list(frame.groupby("session_date", sort=True))
    for config in portfolios:
        ledger: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
        for session_idx, (session_date, group) in enumerate(sessions):
            scored = _apply_composite_score(group, config.factor_weights)
            book, diagnostic, skip, ledger = _construct_common_capped_one_session(
                scored,
                session_date=str(session_date),
                session_idx=session_idx,
                config=config,
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
                diagnostics.append(diagnostic)
            if skip is not None:
                skipped.append(skip)
    return pd.DataFrame(positions), pd.DataFrame(diagnostics), pd.DataFrame(skipped)


def _construct_common_capped_one_session(
    group: pd.DataFrame,
    *,
    session_date: str,
    session_idx: int,
    config: CommonCappedPortfolioConfig,
    holding_horizons: Sequence[int],
    factor_min_holds: Mapping[str, int],
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    min_nonzero_names: int,
    score_weight: float,
    sector_penalty: float,
    cost_bps_per_side: float,
    previous_ledger: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, dict[str, list[dict[str, Any]]]]:
    needed = ["symbol", "composite_score", "beta", "sic2_sector", COMMON_AXIS_COLUMN, *FACTOR_COLUMNS]
    base = group.dropna(subset=needed).drop_duplicates("symbol").copy()
    if len(base) < 2 * min_nonzero_names:
        return [], None, _skip_row(session_date, config, len(base), "insufficient_base"), previous_ledger

    available_symbols = set(base["symbol"].astype(str))
    ledger, dropped_lot_summary = _prune_ledger(previous_ledger, available_symbols, session_idx)
    previous_weights = {side: _ledger_weights(ledger[side]) for side in ("long", "short")}
    locked_weights = {side: _locked_weights(ledger[side], session_idx) for side in ("long", "short")}
    locked_symbols = {side: set(locked_weights[side]) for side in ("long", "short")}

    longs = _candidate_union_locked(
        base,
        side="long",
        previous_weights=previous_weights["long"],
        locked_symbols=locked_symbols["long"],
        blocked_symbols=locked_symbols["short"],
        candidate_pool_per_side=candidate_pool_per_side,
    )
    shorts = _candidate_union_locked(
        base,
        side="short",
        previous_weights=previous_weights["short"],
        locked_symbols=locked_symbols["short"],
        blocked_symbols=locked_symbols["long"],
        candidate_pool_per_side=candidate_pool_per_side,
    )
    if len(longs) < min_nonzero_names or len(shorts) < min_nonzero_names:
        carried = _carry_previous(
            base=base,
            session_date=session_date,
            session_idx=session_idx,
            config=config,
            holding_horizons=holding_horizons,
            previous_weights=previous_weights,
            ledger=ledger,
            cost_bps_per_side=cost_bps_per_side,
            reason="insufficient_candidates",
        )
        if carried is not None:
            return carried
        return [], None, _skip_row(session_date, config, len(base), "insufficient_candidates"), ledger

    effective_budget = 2.0 if _is_empty_book(previous_weights) else float(config.turnover_budget)
    try:
        long_weights, short_weights = _optimize_common_capped_locked(
            longs,
            shorts,
            max_weight=max_single_name_side_weight,
            score_weight=score_weight,
            sector_penalty=sector_penalty,
            turnover_penalty=float(config.turnover_penalty),
            previous_long_weights=previous_weights["long"],
            previous_short_weights=previous_weights["short"],
            locked_long_weights=locked_weights["long"],
            locked_short_weights=locked_weights["short"],
            turnover_budget=effective_budget,
            common_cap=float(config.common_cap),
            common_penalty=float(config.common_penalty),
        )
        carry_reason = ""
    except ValueError as exc:
        carried = _carry_previous(
            base=base,
            session_date=session_date,
            session_idx=session_idx,
            config=config,
            holding_horizons=holding_horizons,
            previous_weights=previous_weights,
            ledger=ledger,
            cost_bps_per_side=cost_bps_per_side,
            reason=f"optimizer_failed:{exc}",
        )
        if carried is not None:
            return carried
        return [], None, _skip_row(session_date, config, len(base), f"optimizer_failed:{exc}"), ledger

    if (long_weights > EPS).sum() < min_nonzero_names:
        return [], None, _skip_row(session_date, config, len(base), "insufficient_long_nonzero"), ledger
    if (short_weights > EPS).sum() < min_nonzero_names:
        return [], None, _skip_row(session_date, config, len(base), "insufficient_short_nonzero"), ledger

    turnover = _side_turnover(previous_weights["long"], longs["symbol"], long_weights)
    turnover += _side_turnover(previous_weights["short"], shorts["symbol"], short_weights)
    book = [
        *_position_rows(longs, long_weights, session_date=session_date, portfolio=config.portfolio, side="long"),
        *_position_rows(shorts, short_weights, session_date=session_date, portfolio=config.portfolio, side="short"),
    ]
    long_target = _target_dict(longs["symbol"], long_weights)
    short_target = _target_dict(shorts["symbol"], short_weights)
    updated_ledger = {
        "long": _update_side_ledger(
            previous_lots=ledger["long"],
            target_weights=long_target,
            session_idx=session_idx,
            side="long",
            base=base,
            factor_weights=config.factor_weights,
            factor_min_holds=factor_min_holds,
        ),
        "short": _update_side_ledger(
            previous_lots=ledger["short"],
            target_weights=short_target,
            session_idx=session_idx,
            side="short",
            base=base,
            factor_weights=config.factor_weights,
            factor_min_holds=factor_min_holds,
        ),
    }
    diagnostic = _diagnostic_row_locked(
        longs=longs,
        shorts=shorts,
        long_weights=long_weights,
        short_weights=short_weights,
        session_date=session_date,
        session_idx=session_idx,
        config=config,
        holding_horizons=holding_horizons,
        turnover=turnover,
        effective_turnover_budget=effective_budget,
        cost_bps_per_side=cost_bps_per_side,
        ledger=updated_ledger,
        locked_weights=locked_weights,
        dropped_lot_summary=dropped_lot_summary,
        carry_reason=carry_reason,
    )
    common_exposure = _weighted_common_exposure(longs, shorts, long_weights, short_weights)
    diagnostic.update(
        {
            "common_axis": COMMON_AXIS_COLUMN,
            "common_cap": float(config.common_cap),
            "common_penalty": float(config.common_penalty),
            "common_pc1_exposure": float(common_exposure),
            "common_pc1_abs_exposure": float(abs(common_exposure)),
            "common_pc1_excess": float(max(0.0, abs(common_exposure) - float(config.common_cap))),
            "long_common_pc1": float(np.dot(long_weights, longs[COMMON_AXIS_COLUMN])),
            "short_common_pc1": float(np.dot(short_weights, shorts[COMMON_AXIS_COLUMN])),
        }
    )
    return book, diagnostic, None, updated_ledger


def _optimize_common_capped_locked(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    *,
    max_weight: float,
    score_weight: float,
    sector_penalty: float,
    turnover_penalty: float,
    previous_long_weights: Mapping[str, float],
    previous_short_weights: Mapping[str, float],
    locked_long_weights: Mapping[str, float],
    locked_short_weights: Mapping[str, float],
    turnover_budget: float,
    common_cap: float,
    common_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_long = len(longs)
    n_short = len(shorts)
    n = n_long + n_short
    long_lower = _sanitize_locked_lower(
        np.array([float(locked_long_weights.get(symbol, 0.0)) for symbol in longs["symbol"]], dtype=float),
        max_weight=max_weight,
    )
    short_lower = _sanitize_locked_lower(
        np.array([float(locked_short_weights.get(symbol, 0.0)) for symbol in shorts["symbol"]], dtype=float),
        max_weight=max_weight,
    )
    lower = np.concatenate([long_lower, short_lower])
    if float(long_lower.sum()) > 1.0 + 1e-8 or float(short_lower.sum()) > 1.0 + 1e-8:
        raise ValueError("locked_weight_exceeds_side_budget")
    if np.any(lower > max_weight + 1e-8):
        raise ValueError("locked_weight_exceeds_name_cap")
    if n_long * max_weight < 1.0 - 1e-12 or n_short * max_weight < 1.0 - 1e-12:
        raise ValueError("insufficient_weight_capacity")

    long_score = _zscore_array(longs["composite_score"].to_numpy(dtype=float))
    short_score = _zscore_array((-shorts["composite_score"]).to_numpy(dtype=float))
    score = np.concatenate([long_score, short_score])
    prev = np.concatenate(
        [
            np.array([float(previous_long_weights.get(symbol, 0.0)) for symbol in longs["symbol"]], dtype=float),
            np.array([float(previous_short_weights.get(symbol, 0.0)) for symbol in shorts["symbol"]], dtype=float),
        ]
    )

    aeq = []
    beq = []
    row = np.zeros(n)
    row[:n_long] = 1.0
    aeq.append(row)
    beq.append(1.0)
    row = np.zeros(n)
    row[n_long:] = 1.0
    aeq.append(row)
    beq.append(1.0)
    row = np.zeros(n)
    row[:n_long] = longs["beta"].to_numpy(dtype=float)
    row[n_long:] = -shorts["beta"].to_numpy(dtype=float)
    aeq.append(row)
    beq.append(0.0)
    a_eq = np.vstack(aeq)
    b_eq = np.array(beq, dtype=float)

    sector_matrix = _group_exposure_matrix(longs, shorts, "sic2_sector")
    m_sector = sector_matrix.shape[0]
    extra_cols = n + m_sector + 1
    c = np.concatenate(
        [
            -score_weight * score,
            np.full(n, turnover_penalty, dtype=float),
            np.full(m_sector, sector_penalty, dtype=float),
            np.array([float(common_penalty)], dtype=float),
        ]
    )
    bounds: list[tuple[float, float | None]] = (
        [(float(lo), max_weight) for lo in lower]
        + [(0.0, None)] * n
        + [(0.0, None)] * m_sector
        + [(0.0, None)]
    )
    a_eq = np.column_stack([a_eq, np.zeros((a_eq.shape[0], extra_cols))])
    zero_sector = np.zeros((n, m_sector))
    zero_common_n = np.zeros((n, 1))
    turnover_top = np.column_stack([np.eye(n), -np.eye(n), zero_sector, zero_common_n])
    turnover_bottom = np.column_stack([-np.eye(n), -np.eye(n), zero_sector, zero_common_n])
    turnover_cap = np.concatenate([np.zeros(n), np.ones(n), np.zeros(m_sector), [0.0]])[None, :]
    sector_top = np.column_stack([sector_matrix, np.zeros((m_sector, n)), -np.eye(m_sector), np.zeros((m_sector, 1))])
    sector_bottom = np.column_stack([-sector_matrix, np.zeros((m_sector, n)), -np.eye(m_sector), np.zeros((m_sector, 1))])
    common_vector = np.concatenate(
        [
            longs[COMMON_AXIS_COLUMN].to_numpy(dtype=float),
            -shorts[COMMON_AXIS_COLUMN].to_numpy(dtype=float),
        ]
    )
    common_top = np.concatenate([common_vector, np.zeros(n), np.zeros(m_sector), [-1.0]])[None, :]
    common_bottom = np.concatenate([-common_vector, np.zeros(n), np.zeros(m_sector), [-1.0]])[None, :]
    result = linprog(
        c,
        A_ub=np.vstack([turnover_top, turnover_bottom, turnover_cap, sector_top, sector_bottom, common_top, common_bottom]),
        b_ub=np.concatenate([prev, -prev, [float(turnover_budget)], np.zeros(m_sector), np.zeros(m_sector), [float(common_cap), float(common_cap)]]),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise ValueError(str(result.message).replace(" ", "_"))
    x = np.clip(result.x[:n], lower, max_weight)
    long_weights = x[:n_long]
    short_weights = x[n_long:]
    if abs(long_weights.sum() - 1.0) > 1e-6 or abs(short_weights.sum() - 1.0) > 1e-6:
        raise ValueError("side_sum_constraint_breach")
    net_beta = float(np.dot(long_weights, longs["beta"]) - np.dot(short_weights, shorts["beta"]))
    if abs(net_beta) > 1e-5:
        raise ValueError("beta_constraint_breach")
    return long_weights, short_weights


def _weighted_common_exposure(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    long_weights: np.ndarray,
    short_weights: np.ndarray,
) -> float:
    return float(
        np.dot(long_weights, longs[COMMON_AXIS_COLUMN])
        - np.dot(short_weights, shorts[COMMON_AXIS_COLUMN])
    )


def _common_exposure_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    rows = []
    for portfolio, group in diagnostics.groupby("portfolio", sort=True):
        exposure = pd.to_numeric(group["common_pc1_exposure"], errors="coerce")
        excess = pd.to_numeric(group["common_pc1_excess"], errors="coerce")
        rows.append(
            {
                "portfolio": portfolio,
                "sessions": int(len(group)),
                "common_cap": float(group["common_cap"].iloc[0]),
                "common_penalty": float(group["common_penalty"].iloc[0]),
                "mean_common_pc1_exposure": float(exposure.mean()),
                "mean_abs_common_pc1_exposure": float(exposure.abs().mean()),
                "median_abs_common_pc1_exposure": float(exposure.abs().median()),
                "p90_abs_common_pc1_exposure": float(exposure.abs().quantile(0.90)),
                "max_abs_common_pc1_exposure": float(exposure.abs().max()),
                "mean_common_pc1_excess": float(excess.mean()),
                "excess_rate": float((excess > 1e-8).mean()),
                "test_window_used": bool(group["test_window_used"].any()),
            }
        )
    return pd.DataFrame(rows)


def _baseline_comparison(metrics: pd.DataFrame, baseline_metrics_path: str | Path) -> pd.DataFrame:
    path = Path(baseline_metrics_path)
    if not path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(path)
    baseline = baseline[baseline["return_kind"].eq("net")].copy()
    capped = metrics[metrics["return_kind"].eq("net")].copy()
    if baseline.empty or capped.empty:
        return pd.DataFrame()
    capped["base_portfolio"] = capped["portfolio"].str.replace(r"_pc1cap\d+p\d+", "", regex=True)
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
    available = [column for column in columns if column in capped.columns and column in baseline.columns]
    merged = capped[["portfolio", "base_portfolio", *available]].merge(
        baseline[["base_portfolio", *available]],
        on="base_portfolio",
        suffixes=("_capped", "_baseline"),
    )
    for column in available:
        merged[f"delta_{column}"] = merged[f"{column}_capped"] - merged[f"{column}_baseline"]
    return merged


def _memo(
    *,
    baseline_metrics_path: Path,
    common_axis_weights: Mapping[str, float],
    common_cap_grid: Sequence[float],
    common_penalty: float,
    eval_start: str,
    end: str,
    cost_bps_per_side: float,
    strict_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    common_summary: pd.DataFrame,
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
            "final_equity_capped",
            "final_equity_baseline",
            "delta_final_equity",
            "annualized_return_capped",
            "annualized_return_baseline",
            "delta_annualized_return",
            "sharpe_no_rf_capped",
            "sharpe_no_rf_baseline",
            "delta_sharpe_no_rf",
            "max_drawdown_capped",
            "max_drawdown_baseline",
            "delta_max_drawdown",
        )
        if column in comparison.columns
    ]
    lines = [
        "# Phase7P Common-Regime-Risk Cap",
        "",
        f"Generated: {_utc_now()}",
        "",
        "Validation-only. The test lockbox is not used.",
        "",
        "## Setup",
        "",
        f"- baseline metrics: `{baseline_metrics_path.as_posix()}`",
        f"- common axis weights: `{dict(common_axis_weights)}`",
        f"- common cap grid: `{list(common_cap_grid)}`",
        f"- common slack penalty: `{common_penalty}`",
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
        "## Common Exposure Summary",
        "",
        _markdown_table(common_summary),
        "",
        "## Notes",
        "",
        "- The common cap is soft; excess exposure is allowed but penalized in the optimizer.",
        "- The common axis is diagnostic PC1 from Phase7O, so this is not a production-ready parameter choice.",
        "- The goal is to test whether common-regime risk can be reduced without destroying shared-core alpha.",
        "",
    ]
    return "\n".join(lines)


def _float_label(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
