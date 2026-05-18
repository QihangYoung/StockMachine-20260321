"""Phase7K locked-lot turnover controlled shared-capital portfolios.

Phase7K keeps the Phase7I shared-capital idea, but adds a first implementation
of factor-reason holding clocks and hard daily turnover budgets. It is
validation-only and does not use the test lockbox.
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
    DEFAULT_QUANTILE,
    DEFAULT_SIZE_PANEL,
    DEFAULT_START,
    DEFAULT_VARIANT,
    _markdown_table,
)
from stockmachine.apps.run_pure_alpha_phase7f import DEFAULT_HOLDING_HORIZONS
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
    _position_rows,
    _score_coverage,
    _sector_diagnostics,
    _weighted_spread,
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
from stockmachine.apps.run_pure_alpha_phase7g import (
    DEFAULT_FUNDAMENTAL_PANEL as PHASE7G_DEFAULT_FUNDAMENTAL_PANEL,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7k_locked_turnover_shared_capital_20260518"
DEFAULT_COST_BPS_PER_SIDE = 4.0
DEFAULT_TURNOVER_BUDGET_GRID = (0.10, 0.15, 0.20)
DEFAULT_FACTOR_MIN_HOLDS: dict[str, int] = {
    "reversal_score": 5,
    "momentum_score": 10,
    "small_size_score": 20,
    "low_beta_score": 20,
    "cash_quality_score": 20,
}
EPS = 1e-10


@dataclass(frozen=True)
class BudgetedPortfolioConfig:
    portfolio: str
    base_portfolio: str
    factor_weights: Mapping[str, float]
    turnover_budget: float
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7K locked-lot turnover controlled portfolios.")
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
        help="Comma-separated factor:min_hold map, e.g. reversal_score:5,momentum_score:10.",
    )
    parser.add_argument("--candidate-pool-per-side", type=int, default=DEFAULT_CANDIDATE_POOL_PER_SIDE)
    parser.add_argument("--max-single-name-side-weight", type=float, default=DEFAULT_MAX_SINGLE_NAME_SIDE_WEIGHT)
    parser.add_argument("--min-nonzero-names", type=int, default=DEFAULT_MIN_NONZERO_NAMES)
    parser.add_argument("--score-weight", type=float, default=DEFAULT_SCORE_WEIGHT)
    parser.add_argument("--sector-penalty", type=float, default=DEFAULT_SECTOR_PENALTY)
    parser.add_argument("--turnover-penalty", type=float, default=DEFAULT_TURNOVER_PENALTY)
    parser.add_argument("--cost-bps-per-side", type=float, default=DEFAULT_COST_BPS_PER_SIDE)
    args = parser.parse_args(argv)

    rollup = build_phase7k_locked_turnover_shared_capital(
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


def build_phase7k_locked_turnover_shared_capital(
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
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    variant: str = DEFAULT_VARIANT,
    start: str = DEFAULT_START,
    eval_start: str = DEFAULT_EVAL_START,
    end: str = DEFAULT_END,
    validation_price_end: str = DEFAULT_VALIDATION_PRICE_END,
    holding_horizons: Sequence[int] = DEFAULT_HOLDING_HORIZONS,
    turnover_budget_grid: Sequence[float] = DEFAULT_TURNOVER_BUDGET_GRID,
    factor_min_holds: Mapping[str, int] = DEFAULT_FACTOR_MIN_HOLDS,
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
    panel = _add_factor_scores(panel)
    budgeted_portfolios = _expand_portfolios(
        portfolios,
        turnover_budget_grid=turnover_budget_grid,
        turnover_penalty=turnover_penalty,
    )
    positions, diagnostics, skipped = _construct_locked_positions(
        panel,
        portfolios=budgeted_portfolios,
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

    paths = {
        "panel_sample": output_dir / "phase7k_feature_target_panel_sample.csv",
        "positions": output_dir / "phase7k_locked_positions.csv.gz",
        "diagnostics": output_dir / "phase7k_daily_diagnostics.csv",
        "skipped": output_dir / "phase7k_skipped_sessions.csv",
        "forward_metrics": output_dir / "phase7k_forward_label_metrics.csv",
        "factor_exposure": output_dir / "phase7k_factor_exposure_summary.csv",
        "score_coverage": output_dir / "phase7k_score_coverage.csv",
        "curve": output_dir / "phase7k_strict_daily_curve.csv",
        "strict_metrics": output_dir / "phase7k_strict_daily_metrics.csv",
        "turnover": output_dir / "phase7k_turnover_summary.csv",
        "lot_summary": output_dir / "phase7k_lot_summary.csv",
        "memo": output_dir / "phase7k_locked_turnover_memo.md",
        "rollup": output_dir / "phase7k_rollup.json",
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
    paths["memo"].write_text(
        _memo(
            variant=variant,
            start=start,
            eval_start=eval_start,
            end=end,
            validation_price_end=validation_price_end,
            holding_horizons=holding_horizons,
            turnover_budget_grid=turnover_budget_grid,
            factor_min_holds=factor_min_holds,
            portfolios=budgeted_portfolios,
            candidate_pool_per_side=candidate_pool_per_side,
            max_single_name_side_weight=max_single_name_side_weight,
            score_weight=score_weight,
            sector_penalty=sector_penalty,
            cost_bps_per_side=cost_bps_per_side,
            positions=positions,
            diagnostics=diagnostics,
            skipped=skipped,
            strict_metrics=strict_metrics,
            turnover=turnover,
            lot_summary=lot_summary,
            factor_exposure=factor_exposure,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "phase": "phase7k_locked_turnover_shared_capital",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "variant": variant,
        "start": start,
        "eval_start": eval_start,
        "end": end,
        "validation_price_end": validation_price_end,
        "holding_horizons": [int(value) for value in holding_horizons],
        "turnover_budget_grid": [float(value) for value in turnover_budget_grid],
        "factor_min_holds": dict(factor_min_holds),
        "candidate_pool_per_side": int(candidate_pool_per_side),
        "max_single_name_side_weight": float(max_single_name_side_weight),
        "min_nonzero_names": int(min_nonzero_names),
        "score_weight": float(score_weight),
        "sector_penalty": float(sector_penalty),
        "turnover_penalty": float(turnover_penalty),
        "cost_bps_per_side": float(cost_bps_per_side),
        "portfolios": [
            {
                "portfolio": config.portfolio,
                "base_portfolio": config.base_portfolio,
                "factor_weights": dict(config.factor_weights),
                "turnover_budget": float(config.turnover_budget),
                "turnover_penalty": float(config.turnover_penalty),
            }
            for config in budgeted_portfolios
        ],
        "rows": {
            "panel": int(len(panel)),
            "positions": int(len(positions)),
            "diagnostics": int(len(diagnostics)),
            "skipped": int(len(skipped)),
            "curve": int(len(curve)),
            "strict_metrics": int(len(strict_metrics)),
        },
        "outputs": {key: value.as_posix() for key, value in paths.items() if key != "rollup"},
        "method": "reason_level_min_hold_hard_daily_turnover_budget_strict_open_to_open_path",
        "label_contract": {
            "signal_session": "T",
            "rebalance": "adjusted open at T+1",
            "pnl": "open-to-open return from T+1 to T+2, stored on T+2 return_date",
            "cost": "sum_abs_target_weight_change * cost_bps_per_side",
        },
        "limitations": [
            "Reason lots are an implementation ledger, not separate funded sleeves.",
            "Expired lots can remain in the book without refreshing their lock unless weight is increased.",
            "Costs are turnover-based only; no borrow, financing, spread, market impact, or capacity model.",
        ],
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    paths["rollup"].write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _construct_locked_positions(
    panel: pd.DataFrame,
    *,
    portfolios: Sequence[BudgetedPortfolioConfig],
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
            book, diagnostic, skip, ledger = _construct_locked_one_session(
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


def _construct_locked_one_session(
    group: pd.DataFrame,
    *,
    session_date: str,
    session_idx: int,
    config: BudgetedPortfolioConfig,
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
    needed = ["symbol", "composite_score", "beta", "sic2_sector", *FACTOR_COLUMNS]
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
        long_weights, short_weights = _optimize_joint_weights_locked(
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
    return book, diagnostic, None, updated_ledger


def _optimize_joint_weights_locked(
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
    c = np.concatenate(
        [
            -score_weight * score,
            np.full(n, turnover_penalty, dtype=float),
            np.full(m_sector, sector_penalty, dtype=float),
        ]
    )
    bounds: list[tuple[float, float | None]] = (
        [(float(lo), max_weight) for lo in lower]
        + [(0.0, None)] * n
        + [(0.0, None)] * m_sector
    )
    a_eq = np.column_stack([a_eq, np.zeros((a_eq.shape[0], n + m_sector))])
    turnover_top = np.column_stack([np.eye(n), -np.eye(n), np.zeros((n, m_sector))])
    turnover_bottom = np.column_stack([-np.eye(n), -np.eye(n), np.zeros((n, m_sector))])
    turnover_cap = np.concatenate([np.zeros(n), np.ones(n), np.zeros(m_sector)])[None, :]
    sector_top = np.column_stack([sector_matrix, np.zeros((m_sector, n)), -np.eye(m_sector)])
    sector_bottom = np.column_stack([-sector_matrix, np.zeros((m_sector, n)), -np.eye(m_sector)])
    result = linprog(
        c,
        A_ub=np.vstack([turnover_top, turnover_bottom, turnover_cap, sector_top, sector_bottom]),
        b_ub=np.concatenate([prev, -prev, [float(turnover_budget)], np.zeros(m_sector), np.zeros(m_sector)]),
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


def _candidate_union_locked(
    group: pd.DataFrame,
    *,
    side: str,
    previous_weights: Mapping[str, float],
    locked_symbols: set[str],
    blocked_symbols: set[str],
    candidate_pool_per_side: int,
) -> pd.DataFrame:
    available = group[~group["symbol"].astype(str).isin(blocked_symbols)].copy()
    if side == "long":
        ranked = available.sort_values(["composite_score", "symbol"], ascending=[False, True])
    elif side == "short":
        ranked = available.sort_values(["composite_score", "symbol"], ascending=[True, True])
    else:
        raise ValueError(f"unknown_side:{side}")
    previous_symbols = set(previous_weights)
    locked = available[available["symbol"].astype(str).isin(locked_symbols)]
    incumbents = available[available["symbol"].astype(str).isin(previous_symbols)]
    top = ranked.head(candidate_pool_per_side)
    union = pd.concat([locked, incumbents, top], ignore_index=True).drop_duplicates("symbol", keep="first")
    return union.reset_index(drop=True)


def _sanitize_locked_lower(values: np.ndarray, *, max_weight: float) -> np.ndarray:
    lower = np.asarray(values, dtype=float).copy()
    lower[np.abs(lower) <= EPS] = 0.0
    if np.any(lower > max_weight) and float(lower.max()) <= max_weight + 1e-7:
        lower = np.minimum(lower, max_weight)
    total = float(lower.sum())
    if total > 1.0 and total <= 1.0 + 1e-7:
        lower *= (1.0 - 1e-9) / total
    return lower


def _carry_previous(
    *,
    base: pd.DataFrame,
    session_date: str,
    session_idx: int,
    config: BudgetedPortfolioConfig,
    holding_horizons: Sequence[int],
    previous_weights: Mapping[str, Mapping[str, float]],
    ledger: dict[str, list[dict[str, Any]]],
    cost_bps_per_side: float,
    reason: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, dict[str, list[dict[str, Any]]]] | None:
    if _is_empty_book(previous_weights):
        return None
    longs = _weights_frame(base, previous_weights["long"])
    shorts = _weights_frame(base, previous_weights["short"])
    if longs.empty or shorts.empty:
        return None
    long_weights = longs["target_weight"].to_numpy(dtype=float)
    short_weights = shorts["target_weight"].to_numpy(dtype=float)
    if abs(long_weights.sum() - 1.0) > 1e-6 or abs(short_weights.sum() - 1.0) > 1e-6:
        return None
    longs = longs.drop(columns=["target_weight"])
    shorts = shorts.drop(columns=["target_weight"])
    book = [
        *_position_rows(longs, long_weights, session_date=session_date, portfolio=config.portfolio, side="long"),
        *_position_rows(shorts, short_weights, session_date=session_date, portfolio=config.portfolio, side="short"),
    ]
    locked_weights = {side: _locked_weights(ledger[side], session_idx) for side in ("long", "short")}
    diagnostic = _diagnostic_row_locked(
        longs=longs,
        shorts=shorts,
        long_weights=long_weights,
        short_weights=short_weights,
        session_date=session_date,
        session_idx=session_idx,
        config=config,
        holding_horizons=holding_horizons,
        turnover=0.0,
        effective_turnover_budget=float(config.turnover_budget),
        cost_bps_per_side=cost_bps_per_side,
        ledger=ledger,
        locked_weights=locked_weights,
        dropped_lot_summary={"dropped_lots": 0, "dropped_weight": 0.0, "dropped_locked_weight": 0.0},
        carry_reason=reason,
    )
    return book, diagnostic, None, ledger


def _weights_frame(base: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    frame = base[base["symbol"].astype(str).isin(set(weights))].copy()
    frame["target_weight"] = frame["symbol"].astype(str).map(lambda symbol: float(weights.get(symbol, 0.0)))
    return frame[frame["target_weight"].gt(EPS)].sort_values("symbol").reset_index(drop=True)


def _diagnostic_row_locked(
    *,
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    long_weights: np.ndarray,
    short_weights: np.ndarray,
    session_date: str,
    session_idx: int,
    config: BudgetedPortfolioConfig,
    holding_horizons: Sequence[int],
    turnover: float,
    effective_turnover_budget: float,
    cost_bps_per_side: float,
    ledger: Mapping[str, list[dict[str, Any]]],
    locked_weights: Mapping[str, Mapping[str, float]],
    dropped_lot_summary: Mapping[str, float | int],
    carry_reason: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "session_date": session_date,
        "session_idx": int(session_idx),
        "portfolio": config.portfolio,
        "base_portfolio": config.base_portfolio,
        "factor_weights_json": json.dumps(dict(config.factor_weights), sort_keys=True),
        "turnover_budget": float(config.turnover_budget),
        "effective_turnover_budget": float(effective_turnover_budget),
        "turnover_penalty": float(config.turnover_penalty),
        "target_turnover": float(turnover),
        "budget_used_fraction": float(turnover / effective_turnover_budget) if effective_turnover_budget > 0 else np.nan,
        "cost_bps": float(turnover * cost_bps_per_side),
        "long_names": int((long_weights > EPS).sum()),
        "short_names": int((short_weights > EPS).sum()),
        "long_beta": float(np.dot(long_weights, longs["beta"])),
        "short_beta": float(np.dot(short_weights, shorts["beta"])),
        "net_beta": float(np.dot(long_weights, longs["beta"]) - np.dot(short_weights, shorts["beta"])),
        "long_composite_score": float(np.dot(long_weights, longs["composite_score"])),
        "short_composite_score": float(np.dot(short_weights, shorts["composite_score"])),
        "score_spread": float(np.dot(long_weights, longs["composite_score"]) - np.dot(short_weights, shorts["composite_score"])),
        "gross_exposure": 2.0,
        "net_exposure": 0.0,
        "locked_long_weight": float(sum(locked_weights["long"].values())),
        "locked_short_weight": float(sum(locked_weights["short"].values())),
        "locked_total_weight": float(sum(locked_weights["long"].values()) + sum(locked_weights["short"].values())),
        "active_lots": int(len(ledger["long"]) + len(ledger["short"])),
        "locked_lots": int(_locked_lot_count(ledger, session_idx)),
        "dropped_lots": int(dropped_lot_summary.get("dropped_lots", 0)),
        "dropped_weight": float(dropped_lot_summary.get("dropped_weight", 0.0)),
        "dropped_locked_weight": float(dropped_lot_summary.get("dropped_locked_weight", 0.0)),
        "carry_reason": carry_reason,
        "test_window_used": False,
    }
    for column in FACTOR_COLUMNS:
        row[f"{column}_exposure"] = float(np.dot(long_weights, longs[column]) - np.dot(short_weights, shorts[column]))
        row[f"long_{column}"] = float(np.dot(long_weights, longs[column]))
        row[f"short_{column}"] = float(np.dot(short_weights, shorts[column]))
    row.update(_sector_diagnostics(longs, shorts, long_weights, short_weights))
    for holding in holding_horizons:
        target_column = f"forward_beta_residual_return_h{holding}"
        payoff = _weighted_spread(longs, shorts, long_weights, short_weights, target_column)
        row[f"gross_payoff_h{holding}_bps"] = payoff * 10000.0 if np.isfinite(payoff) else np.nan
        row[f"net_payoff_h{holding}_bps"] = payoff * 10000.0 - row["cost_bps"] if np.isfinite(payoff) else np.nan
    return row


def _update_side_ledger(
    *,
    previous_lots: Sequence[Mapping[str, Any]],
    target_weights: Mapping[str, float],
    session_idx: int,
    side: str,
    base: pd.DataFrame,
    factor_weights: Mapping[str, float],
    factor_min_holds: Mapping[str, int],
) -> list[dict[str, Any]]:
    rows = {str(row.symbol): row for row in base.itertuples(index=False)}
    lots_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for lot in previous_lots:
        lots_by_symbol.setdefault(str(lot["symbol"]), []).append(dict(lot))
    updated: list[dict[str, Any]] = []
    for symbol, target in sorted(target_weights.items()):
        remaining = float(target)
        if remaining <= EPS:
            continue
        existing = lots_by_symbol.get(symbol, [])
        locked = [lot for lot in existing if _is_lot_locked(lot, session_idx)]
        expired = [lot for lot in existing if not _is_lot_locked(lot, session_idx)]
        for lot in locked:
            weight = float(lot["weight"])
            if weight <= EPS:
                continue
            updated.append(_copy_lot(lot, weight))
            remaining -= weight
        for lot in expired:
            if remaining <= EPS:
                break
            keep = min(float(lot["weight"]), remaining)
            if keep > EPS:
                updated.append(_copy_lot(lot, keep))
                remaining -= keep
        if remaining > EPS and symbol in rows:
            shares = _factor_support_shares(rows[symbol], side=side, factor_weights=factor_weights)
            for factor, share in shares.items():
                weight = remaining * float(share)
                if weight <= EPS:
                    continue
                updated.append(
                    {
                        "symbol": symbol,
                        "factor": factor,
                        "weight": float(weight),
                        "birth_idx": int(session_idx),
                        "min_hold": int(factor_min_holds.get(factor, 10)),
                    }
                )
    return [lot for lot in updated if float(lot["weight"]) > EPS]


def _factor_support_shares(row: Any, *, side: str, factor_weights: Mapping[str, float]) -> dict[str, float]:
    supports: dict[str, float] = {}
    for factor in FACTOR_COLUMNS:
        factor_weight = float(factor_weights.get(factor, 0.0))
        if factor_weight <= 0.0:
            continue
        score = float(getattr(row, factor))
        directional_score = score if side == "long" else -score
        support = max(0.0, factor_weight * directional_score)
        if support > 0.0:
            supports[factor] = support
    total = float(sum(supports.values()))
    if total <= EPS:
        supports = {factor: float(weight) for factor, weight in factor_weights.items() if float(weight) > 0.0}
        total = float(sum(supports.values()))
    if total <= EPS:
        return {}
    return {factor: value / total for factor, value in supports.items()}


def _prune_ledger(
    ledger: Mapping[str, Sequence[Mapping[str, Any]]],
    available_symbols: set[str],
    session_idx: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float | int]]:
    pruned: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
    dropped_lots = 0
    dropped_weight = 0.0
    dropped_locked_weight = 0.0
    for side in ("long", "short"):
        for lot in ledger[side]:
            weight = float(lot["weight"])
            if str(lot["symbol"]) in available_symbols:
                pruned[side].append(dict(lot))
            else:
                dropped_lots += 1
                dropped_weight += weight
                if _is_lot_locked(lot, session_idx):
                    dropped_locked_weight += weight
    return pruned, {
        "dropped_lots": int(dropped_lots),
        "dropped_weight": float(dropped_weight),
        "dropped_locked_weight": float(dropped_locked_weight),
    }


def _ledger_weights(lots: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for lot in lots:
        symbol = str(lot["symbol"])
        weights[symbol] = weights.get(symbol, 0.0) + float(lot["weight"])
    return {symbol: weight for symbol, weight in weights.items() if weight > EPS}


def _locked_weights(lots: Sequence[Mapping[str, Any]], session_idx: int) -> dict[str, float]:
    weights: dict[str, float] = {}
    for lot in lots:
        if not _is_lot_locked(lot, session_idx):
            continue
        symbol = str(lot["symbol"])
        weights[symbol] = weights.get(symbol, 0.0) + float(lot["weight"])
    return {symbol: weight for symbol, weight in weights.items() if weight > EPS}


def _is_lot_locked(lot: Mapping[str, Any], session_idx: int) -> bool:
    return int(session_idx) - int(lot["birth_idx"]) < int(lot["min_hold"])


def _copy_lot(lot: Mapping[str, Any], weight: float) -> dict[str, Any]:
    return {
        "symbol": str(lot["symbol"]),
        "factor": str(lot["factor"]),
        "weight": float(weight),
        "birth_idx": int(lot["birth_idx"]),
        "min_hold": int(lot["min_hold"]),
    }


def _locked_lot_count(ledger: Mapping[str, Sequence[Mapping[str, Any]]], session_idx: int) -> int:
    return int(sum(1 for side in ("long", "short") for lot in ledger[side] if _is_lot_locked(lot, session_idx)))


def _target_dict(symbols: pd.Series, weights: np.ndarray) -> dict[str, float]:
    return {
        str(symbol): float(weight)
        for symbol, weight in zip(symbols.astype(str), weights)
        if float(weight) > EPS
    }


def _side_turnover(previous: Mapping[str, float], symbols: pd.Series, weights: np.ndarray) -> float:
    new = _target_dict(symbols, weights)
    universe = set(previous) | set(new)
    return float(sum(abs(float(new.get(symbol, 0.0)) - float(previous.get(symbol, 0.0))) for symbol in universe))


def _is_empty_book(previous_weights: Mapping[str, Mapping[str, float]]) -> bool:
    return not previous_weights["long"] and not previous_weights["short"]


def _skip_row(session_date: str, config: BudgetedPortfolioConfig, candidates: int, reason: str) -> dict[str, Any]:
    return {
        "session_date": str(session_date),
        "portfolio": config.portfolio,
        "base_portfolio": config.base_portfolio,
        "turnover_budget": float(config.turnover_budget),
        "candidates": int(candidates),
        "skip_reason": reason,
        "test_window_used": False,
    }


def _lot_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    rows = []
    for portfolio, group in diagnostics.groupby("portfolio", sort=True):
        rows.append(
            {
                "portfolio": portfolio,
                "base_portfolio": str(group["base_portfolio"].iloc[0]),
                "turnover_budget": float(group["turnover_budget"].iloc[0]),
                "mean_locked_total_weight": float(group["locked_total_weight"].mean()),
                "median_locked_total_weight": float(group["locked_total_weight"].median()),
                "mean_locked_lots": float(group["locked_lots"].mean()),
                "mean_active_lots": float(group["active_lots"].mean()),
                "carry_session_rate": float(group["carry_reason"].astype(str).ne("").mean()),
                "mean_budget_used_fraction": float(group["budget_used_fraction"].replace([np.inf, -np.inf], np.nan).mean()),
                "mean_dropped_weight": float(group["dropped_weight"].mean()),
                "mean_dropped_locked_weight": float(group["dropped_locked_weight"].mean()),
                "test_window_used": bool(group["test_window_used"].any()),
            }
        )
    return pd.DataFrame(rows)


def _expand_portfolios(
    portfolios: Sequence[PortfolioConfig],
    *,
    turnover_budget_grid: Sequence[float],
    turnover_penalty: float,
) -> tuple[BudgetedPortfolioConfig, ...]:
    expanded: list[BudgetedPortfolioConfig] = []
    for budget in turnover_budget_grid:
        for config in portfolios:
            expanded.append(
                BudgetedPortfolioConfig(
                    portfolio=f"{config.portfolio}_{_budget_label(budget)}",
                    base_portfolio=config.portfolio,
                    factor_weights=dict(config.factor_weights),
                    turnover_budget=float(budget),
                    turnover_penalty=float(turnover_penalty),
                )
            )
    return tuple(expanded)


def _budget_label(value: float) -> str:
    return f"tb{value:.2f}".replace(".", "p")


def _memo(
    *,
    variant: str,
    start: str,
    eval_start: str,
    end: str,
    validation_price_end: str,
    holding_horizons: Sequence[int],
    turnover_budget_grid: Sequence[float],
    factor_min_holds: Mapping[str, int],
    portfolios: Sequence[BudgetedPortfolioConfig],
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    score_weight: float,
    sector_penalty: float,
    cost_bps_per_side: float,
    positions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    skipped: pd.DataFrame,
    strict_metrics: pd.DataFrame,
    turnover: pd.DataFrame,
    lot_summary: pd.DataFrame,
    factor_exposure: pd.DataFrame,
) -> str:
    portfolio_rows = pd.DataFrame(
        [
            {
                "portfolio": config.portfolio,
                "base_portfolio": config.base_portfolio,
                "turnover_budget": float(config.turnover_budget),
                "factor_weights": json.dumps(dict(config.factor_weights), sort_keys=True),
            }
            for config in portfolios
        ]
    )
    net_metrics = strict_metrics[strict_metrics["return_kind"].eq("net")].sort_values(
        ["base_portfolio" if "base_portfolio" in strict_metrics.columns else "portfolio", "sharpe_no_rf"],
        ascending=[True, False],
    )
    if not net_metrics.empty and "base_portfolio" not in net_metrics.columns:
        net_metrics = net_metrics.merge(
            portfolio_rows[["portfolio", "base_portfolio", "turnover_budget"]],
            on="portfolio",
            how="left",
        )
    selected_columns = [
        column
        for column in (
            "portfolio",
            "base_portfolio",
            "turnover_budget",
            "final_equity",
            "annualized_return",
            "annualized_vol",
            "sharpe_no_rf",
            "max_drawdown",
            "mean_turnover",
            "mean_cost_bps",
            "realized_beta_net",
        )
        if column in net_metrics.columns
    ]
    lines = [
        "# Phase7K Locked-Turnover Shared-Capital Results",
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
        f"- validation price end: `{validation_price_end}`",
        f"- holding horizons: `{list(holding_horizons)}`",
        f"- turnover budget grid: `{list(turnover_budget_grid)}`",
        f"- factor min holds: `{dict(factor_min_holds)}`",
        f"- candidate pool per side: `{candidate_pool_per_side}`",
        f"- max single-name side weight: `{max_single_name_side_weight}`",
        f"- score weight: `{score_weight}`",
        f"- SIC2 sector soft penalty: `{sector_penalty}`",
        f"- cost bps per side: `{cost_bps_per_side}`",
        f"- position rows: `{len(positions)}`",
        f"- diagnostics rows: `{len(diagnostics)}`",
        f"- skipped rows: `{len(skipped)}`",
        "",
        "## Portfolios",
        "",
        _markdown_table(portfolio_rows),
        "",
        "## Strict Daily Net Metrics",
        "",
        _markdown_table(net_metrics[selected_columns] if selected_columns else net_metrics),
        "",
        "## Turnover",
        "",
        _markdown_table(turnover),
        "",
        "## Lot Summary",
        "",
        _markdown_table(lot_summary),
        "",
        "## Factor Exposure",
        "",
        _markdown_table(factor_exposure),
        "",
        "## Notes",
        "",
        "- Signal is generated every session, but locked factor-reason lots cannot be reduced before their minimum hold expires.",
        "- Expired lots remain tradable; if kept unchanged, they do not receive a new lock unless weight is increased.",
        "- The turnover budget is hard inside the optimizer after the initial fully-invested build day.",
        "- Strict path uses signal date T, rebalance at T+1 adjusted open, and T+1 to T+2 open-to-open PnL.",
        "",
    ]
    return "\n".join(lines)


def _parse_int_grid(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("grid cannot be empty")
    if any(part <= 0 for part in parsed):
        raise ValueError("holding horizons must be positive")
    return parsed


def _parse_float_grid(value: str) -> tuple[float, ...]:
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("grid cannot be empty")
    if any(part <= 0.0 for part in parsed):
        raise ValueError("turnover budgets must be positive")
    return parsed


def _parse_factor_min_holds(value: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid factor min hold: {item}")
        factor, hold = item.split(":", 1)
        factor = factor.strip()
        if factor not in FACTOR_COLUMNS:
            raise ValueError(f"unknown factor min hold column: {factor}")
        parsed[factor] = int(hold.strip())
    missing = set(FACTOR_COLUMNS).difference(parsed)
    for factor in missing:
        parsed[factor] = DEFAULT_FACTOR_MIN_HOLDS.get(factor, 10)
    if any(value <= 0 for value in parsed.values()):
        raise ValueError("factor min holds must be positive")
    return parsed


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
