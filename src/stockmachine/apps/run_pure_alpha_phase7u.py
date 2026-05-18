"""Phase7U strict asymmetric-universe shared-core test.

Phase7T showed that universe choice is a strong structural prior. Phase7U tests
the most plausible next prior directly: keep the broad top1000-like opportunity
set on the long side, but restrict the short side to the more liquid ADV30m core.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps import run_pure_alpha_phase7k as p7k
from stockmachine.apps import run_pure_alpha_phase7i as p7i
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4z import _load_open_to_open_returns
from stockmachine.apps.run_pure_alpha_phase7j import (
    _add_path_metrics,
    _portfolio_metrics,
    _strict_daily_rebalance_path,
    _turnover_summary,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7u_asymmetric_universe_strict_20260518_top1000_short_adv30_tb0p15"
DEFAULT_LONG_VARIANT = "top1000_clean_core_beta_full"
DEFAULT_SHORT_VARIANT = "adv30m_clean_core_beta_full"
DEFAULT_TURNOVER_BUDGET = 0.15
DEFAULT_PORTFOLIO_NAME = "asym_top1000_short_adv30_shared_core_lambda_0p005"
SHARED_CORE_WEIGHTS: dict[str, float] = {
    "reversal_score": 0.25,
    "momentum_score": 0.10,
    "small_size_score": 0.30,
    "low_beta_score": 0.20,
    "cash_quality_score": 0.15,
}
EPS = 1e-10


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7U strict asymmetric-universe test.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--long-variant", default=DEFAULT_LONG_VARIANT)
    parser.add_argument("--short-variant", default=DEFAULT_SHORT_VARIANT)
    parser.add_argument("--turnover-budget", type=float, default=DEFAULT_TURNOVER_BUDGET)
    args = parser.parse_args(argv)

    rollup = run_phase7u_asymmetric_universe(
        output_root=Path(args.output_root),
        long_variant=str(args.long_variant),
        short_variant=str(args.short_variant),
        turnover_budget=float(args.turnover_budget),
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


def run_phase7u_asymmetric_universe(
    *,
    output_root: Path,
    long_variant: str,
    short_variant: str,
    turnover_budget: float,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    config = p7k.BudgetedPortfolioConfig(
        portfolio=f"{DEFAULT_PORTFOLIO_NAME}_{_budget_label(turnover_budget)}",
        base_portfolio=DEFAULT_PORTFOLIO_NAME,
        factor_weights=SHARED_CORE_WEIGHTS,
        turnover_budget=float(turnover_budget),
        turnover_penalty=float(p7k.DEFAULT_TURNOVER_PENALTY),
    )

    long_panel = _load_scored_panel(variant=long_variant)
    short_panel = _load_scored_panel(variant=short_variant)
    positions, diagnostics, skipped = _construct_asymmetric_positions(
        long_panel,
        short_panel,
        config=config,
        eval_start=p7k.DEFAULT_EVAL_START,
        end=p7k.DEFAULT_END,
        holding_horizons=p7k.DEFAULT_HOLDING_HORIZONS,
        factor_min_holds=p7k.DEFAULT_FACTOR_MIN_HOLDS,
        candidate_pool_per_side=p7k.DEFAULT_CANDIDATE_POOL_PER_SIDE,
        max_single_name_side_weight=p7k.DEFAULT_MAX_SINGLE_NAME_SIDE_WEIGHT,
        min_nonzero_names=p7k.DEFAULT_MIN_NONZERO_NAMES,
        score_weight=p7k.DEFAULT_SCORE_WEIGHT,
        sector_penalty=p7k.DEFAULT_SECTOR_PENALTY,
        cost_bps_per_side=p7k.DEFAULT_COST_BPS_PER_SIDE,
    )

    forward_metrics = p7i._metrics(diagnostics, skipped=skipped, holding_horizons=p7k.DEFAULT_HOLDING_HORIZONS)
    factor_exposure = p7i._factor_exposure_summary(diagnostics)
    score_coverage = pd.concat(
        [
            p7i._score_coverage(long_panel, eval_start=p7k.DEFAULT_EVAL_START, end=p7k.DEFAULT_END).assign(side="long", variant=long_variant),
            p7i._score_coverage(short_panel, eval_start=p7k.DEFAULT_EVAL_START, end=p7k.DEFAULT_END).assign(side="short", variant=short_variant),
        ],
        ignore_index=True,
    )

    symbols = tuple(sorted(positions["symbol"].astype(str).unique()))
    stock_returns = _load_open_to_open_returns(
        daily_globs=p7k.DEFAULT_DAILY_GLOBS,
        adj_factor_globs=p7k.DEFAULT_ADJ_FACTOR_GLOBS,
        symbols=symbols,
        end_date=p7k.DEFAULT_VALIDATION_PRICE_END,
    )
    benchmark_returns = _load_open_to_open_returns(
        daily_globs=p7k.DEFAULT_BENCHMARK_DAILY_GLOBS,
        adj_factor_globs=p7k.DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
        symbols=(p7k.DEFAULT_BENCHMARK_SYMBOL,),
        end_date=p7k.DEFAULT_VALIDATION_PRICE_END,
    ).rename(columns={"oto_return": "benchmark_oto_return"})
    calendar = (
        benchmark_returns[benchmark_returns["symbol"].eq(p7k.DEFAULT_BENCHMARK_SYMBOL)]["session_date"]
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
        cost_bps_per_side=p7k.DEFAULT_COST_BPS_PER_SIDE,
    )
    curve = _add_path_metrics(daily)
    strict_metrics = _portfolio_metrics(curve)
    turnover = _turnover_summary(curve)
    lot_summary = p7k._lot_summary(diagnostics)

    paths = {
        "long_panel_sample": output_root / "phase7u_long_panel_sample.csv",
        "short_panel_sample": output_root / "phase7u_short_panel_sample.csv",
        "positions": output_root / "phase7u_asymmetric_positions.csv.gz",
        "diagnostics": output_root / "phase7u_daily_diagnostics.csv",
        "skipped": output_root / "phase7u_skipped_sessions.csv",
        "forward_metrics": output_root / "phase7u_forward_label_metrics.csv",
        "factor_exposure": output_root / "phase7u_factor_exposure_summary.csv",
        "score_coverage": output_root / "phase7u_score_coverage_by_side.csv",
        "curve": output_root / "phase7u_strict_daily_curve.csv",
        "strict_metrics": output_root / "phase7u_strict_daily_metrics.csv",
        "turnover": output_root / "phase7u_turnover_summary.csv",
        "lot_summary": output_root / "phase7u_lot_summary.csv",
        "memo": output_root / "phase7u_asymmetric_universe_memo.md",
        "rollup": output_root / "phase7u_rollup.json",
    }
    long_panel.head(5000).to_csv(paths["long_panel_sample"], index=False)
    short_panel.head(5000).to_csv(paths["short_panel_sample"], index=False)
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
            long_variant=long_variant,
            short_variant=short_variant,
            config=config,
            strict_metrics=strict_metrics,
            turnover=turnover,
            factor_exposure=factor_exposure,
            score_coverage=score_coverage,
            diagnostics=diagnostics,
        ),
        encoding="utf-8",
    )
    rollup = {
        "created_at_utc": _utc_now(),
        "phase": "phase7u_asymmetric_universe_strict",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "long_variant": long_variant,
        "short_variant": short_variant,
        "turnover_budget": turnover_budget,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    paths["rollup"].write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_scored_panel(*, variant: str) -> pd.DataFrame:
    panel = p7i._load_multifactor_panel(
        membership_path=p7k.DEFAULT_PHASE1_MEMBERSHIP,
        beta_panel_path=p7k.DEFAULT_PHASE2_BETA_PANEL,
        size_panel_path=p7k.DEFAULT_SIZE_PANEL,
        non_price_panel_path=p7k.DEFAULT_NON_PRICE_PANEL,
        fundamental_panel_path=p7k.PHASE7G_DEFAULT_FUNDAMENTAL_PANEL or p7k.DEFAULT_FUNDAMENTAL_PANEL,
        cik_mapping_path=p7k.DEFAULT_CIK_MAPPING,
        sec_submissions_dir=p7k.DEFAULT_SEC_SUBMISSIONS_DIR,
        daily_globs=p7k.DEFAULT_DAILY_GLOBS,
        adj_factor_globs=p7k.DEFAULT_ADJ_FACTOR_GLOBS,
        benchmark_daily_globs=p7k.DEFAULT_BENCHMARK_DAILY_GLOBS,
        benchmark_adj_factor_globs=p7k.DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS,
        benchmark_symbol=p7k.DEFAULT_BENCHMARK_SYMBOL,
        variant=variant,
        start=p7k.DEFAULT_START,
        end=p7k.DEFAULT_END,
        holding_horizons=p7k.DEFAULT_HOLDING_HORIZONS,
    )
    return p7i._add_factor_scores(panel)


def _construct_asymmetric_positions(
    long_panel: pd.DataFrame,
    short_panel: pd.DataFrame,
    *,
    config: p7k.BudgetedPortfolioConfig,
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
    long_frame = long_panel[long_panel["session_date"].ge(eval_start) & long_panel["session_date"].le(end)].copy()
    short_frame = short_panel[short_panel["session_date"].ge(eval_start) & short_panel["session_date"].le(end)].copy()
    long_groups = {str(date): group for date, group in long_frame.groupby("session_date", sort=True)}
    short_groups = {str(date): group for date, group in short_frame.groupby("session_date", sort=True)}
    sessions = [date for date in sorted(long_groups) if date in short_groups]

    positions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    ledger: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
    for session_idx, session_date in enumerate(sessions):
        long_scored = p7i._apply_composite_score(long_groups[session_date], config.factor_weights)
        short_scored = p7i._apply_composite_score(short_groups[session_date], config.factor_weights)
        book, diagnostic, skip, ledger = _construct_asymmetric_one_session(
            long_scored,
            short_scored,
            session_date=session_date,
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


def _construct_asymmetric_one_session(
    long_group: pd.DataFrame,
    short_group: pd.DataFrame,
    *,
    session_date: str,
    session_idx: int,
    config: p7k.BudgetedPortfolioConfig,
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
    needed = ["symbol", "composite_score", "beta", "sic2_sector", *p7i.FACTOR_COLUMNS]
    long_base = long_group.dropna(subset=needed).drop_duplicates("symbol").copy()
    short_base = short_group.dropna(subset=needed).drop_duplicates("symbol").copy()
    if len(long_base) < min_nonzero_names or len(short_base) < min_nonzero_names:
        return [], None, p7k._skip_row(session_date, config, min(len(long_base), len(short_base)), "insufficient_base"), previous_ledger

    ledger, dropped_lot_summary = _prune_asymmetric_ledger(previous_ledger, long_base, short_base, session_idx)
    previous_weights = {side: p7k._ledger_weights(ledger[side]) for side in ("long", "short")}
    locked_weights = {side: p7k._locked_weights(ledger[side], session_idx) for side in ("long", "short")}
    locked_symbols = {side: set(locked_weights[side]) for side in ("long", "short")}

    longs = p7k._candidate_union_locked(
        long_base,
        side="long",
        previous_weights=previous_weights["long"],
        locked_symbols=locked_symbols["long"],
        blocked_symbols=locked_symbols["short"],
        candidate_pool_per_side=candidate_pool_per_side,
    )
    shorts = p7k._candidate_union_locked(
        short_base,
        side="short",
        previous_weights=previous_weights["short"],
        locked_symbols=locked_symbols["short"],
        blocked_symbols=locked_symbols["long"],
        candidate_pool_per_side=candidate_pool_per_side,
    )
    if len(longs) < min_nonzero_names or len(shorts) < min_nonzero_names:
        carried = _carry_previous_asymmetric(
            long_base=long_base,
            short_base=short_base,
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
        return [], None, p7k._skip_row(session_date, config, min(len(long_base), len(short_base)), "insufficient_candidates"), ledger

    effective_budget = 2.0 if p7k._is_empty_book(previous_weights) else float(config.turnover_budget)
    try:
        long_weights, short_weights = p7k._optimize_joint_weights_locked(
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
        carried = _carry_previous_asymmetric(
            long_base=long_base,
            short_base=short_base,
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
        return [], None, p7k._skip_row(session_date, config, min(len(long_base), len(short_base)), f"optimizer_failed:{exc}"), ledger

    turnover = p7k._side_turnover(previous_weights["long"], longs["symbol"], long_weights)
    turnover += p7k._side_turnover(previous_weights["short"], shorts["symbol"], short_weights)
    book = [
        *p7i._position_rows(longs, long_weights, session_date=session_date, portfolio=config.portfolio, side="long"),
        *p7i._position_rows(shorts, short_weights, session_date=session_date, portfolio=config.portfolio, side="short"),
    ]
    long_target = p7k._target_dict(longs["symbol"], long_weights)
    short_target = p7k._target_dict(shorts["symbol"], short_weights)
    updated_ledger = {
        "long": p7k._update_side_ledger(
            previous_lots=ledger["long"],
            target_weights=long_target,
            session_idx=session_idx,
            side="long",
            base=long_base,
            factor_weights=config.factor_weights,
            factor_min_holds=factor_min_holds,
        ),
        "short": p7k._update_side_ledger(
            previous_lots=ledger["short"],
            target_weights=short_target,
            session_idx=session_idx,
            side="short",
            base=short_base,
            factor_weights=config.factor_weights,
            factor_min_holds=factor_min_holds,
        ),
    }
    diagnostic = p7k._diagnostic_row_locked(
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


def _carry_previous_asymmetric(
    *,
    long_base: pd.DataFrame,
    short_base: pd.DataFrame,
    session_date: str,
    session_idx: int,
    config: p7k.BudgetedPortfolioConfig,
    holding_horizons: Sequence[int],
    previous_weights: Mapping[str, Mapping[str, float]],
    ledger: dict[str, list[dict[str, Any]]],
    cost_bps_per_side: float,
    reason: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, dict[str, list[dict[str, Any]]]] | None:
    if p7k._is_empty_book(previous_weights):
        return None
    longs = p7k._weights_frame(long_base, previous_weights["long"])
    shorts = p7k._weights_frame(short_base, previous_weights["short"])
    if longs.empty or shorts.empty:
        return None
    long_weights = longs["target_weight"].to_numpy(dtype=float)
    short_weights = shorts["target_weight"].to_numpy(dtype=float)
    if abs(long_weights.sum() - 1.0) > 1e-6 or abs(short_weights.sum() - 1.0) > 1e-6:
        return None
    longs = longs.drop(columns=["target_weight"])
    shorts = shorts.drop(columns=["target_weight"])
    book = [
        *p7i._position_rows(longs, long_weights, session_date=session_date, portfolio=config.portfolio, side="long"),
        *p7i._position_rows(shorts, short_weights, session_date=session_date, portfolio=config.portfolio, side="short"),
    ]
    locked_weights = {side: p7k._locked_weights(ledger[side], session_idx) for side in ("long", "short")}
    diagnostic = p7k._diagnostic_row_locked(
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


def _prune_asymmetric_ledger(
    ledger: Mapping[str, Sequence[Mapping[str, Any]]],
    long_base: pd.DataFrame,
    short_base: pd.DataFrame,
    session_idx: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float | int]]:
    available = {
        "long": set(long_base["symbol"].astype(str)),
        "short": set(short_base["symbol"].astype(str)),
    }
    pruned: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
    dropped_lots = 0
    dropped_weight = 0.0
    dropped_locked_weight = 0.0
    for side in ("long", "short"):
        for lot in ledger[side]:
            weight = float(lot["weight"])
            if str(lot["symbol"]) in available[side]:
                pruned[side].append(dict(lot))
            else:
                dropped_lots += 1
                dropped_weight += weight
                if p7k._is_lot_locked(lot, session_idx):
                    dropped_locked_weight += weight
    return pruned, {
        "dropped_lots": dropped_lots,
        "dropped_weight": dropped_weight,
        "dropped_locked_weight": dropped_locked_weight,
    }


def _memo(
    *,
    long_variant: str,
    short_variant: str,
    config: p7k.BudgetedPortfolioConfig,
    strict_metrics: pd.DataFrame,
    turnover: pd.DataFrame,
    factor_exposure: pd.DataFrame,
    score_coverage: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    net = strict_metrics[strict_metrics["return_kind"].eq("net")].copy()
    exposure = {
        "mean_abs_constructed_net_beta": float(diagnostics["net_beta"].abs().mean()),
        "p95_abs_constructed_net_beta": float(diagnostics["net_beta"].abs().quantile(0.95)),
        "mean_sector_abs_exposure_sum": float(diagnostics["sector_abs_exposure_sum"].mean()),
        "mean_long_names": float(diagnostics["long_names"].mean()),
        "mean_short_names": float(diagnostics["short_names"].mean()),
    }
    lines = [
        "# Phase7U Strict Asymmetric Universe Test",
        "",
        f"Generated: {_utc_now()}",
        "",
        "Validation-only. The test lockbox is not used.",
        "",
        "## Setup",
        "",
        f"- long variant: `{long_variant}`",
        f"- short variant: `{short_variant}`",
        f"- portfolio: `{config.portfolio}`",
        f"- factor weights: `{json.dumps(dict(config.factor_weights), sort_keys=True)}`",
        f"- turnover budget: `{config.turnover_budget}`",
        "",
        "## Net Strict Metrics",
        "",
        _markdown_table(net),
        "",
        "## Turnover",
        "",
        _markdown_table(turnover),
        "",
        "## Factor Exposure",
        "",
        _markdown_table(factor_exposure),
        "",
        "## Score Coverage By Side",
        "",
        _markdown_table(score_coverage),
        "",
        "## Exposure Checks",
        "",
        _markdown_table(pd.DataFrame([exposure])),
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)


def _budget_label(value: float) -> str:
    return f"tb{value:.2f}".replace(".", "p")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
