"""Phase7L reversal regime-gated shared-capital portfolios.

Phase7L applies the Phase7C reliable-core reversal regime gate to the Phase7K
locked-turnover shared-capital construction. When the gate is active, reversal
weight is partially reduced and the released weight is redistributed equally to
the other active factors. This is validation-only and does not use the test
lockbox.
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


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7l_reversal_gate_shared_capital_20260518"
DEFAULT_GATE_STATES_PATH = (
    RESEARCH_ROOT
    / "phase7c_repaired_graph_reversal_payoff_20260517_wf252min"
    / "phase7c_daily_graph_states.csv"
)
DEFAULT_GATE_SCORE = "reliable_core_score"
DEFAULT_GATE_QUANTILE_COLUMN = "reliable_core_score_wf_q33"
DEFAULT_GATE_UPPER_QUANTILE_COLUMN = "reliable_core_score_wf_q67"
DEFAULT_GATE_MODE = "bottom_down"
DEFAULT_REVERSAL_MULTIPLIER = 0.5
DEFAULT_TOP_REVERSAL_MULTIPLIER = 1.25
DEFAULT_TURNOVER_BUDGET_GRID = (0.15,)
REVERSAL_FACTOR = "reversal_score"
GATE_MODES = ("bottom_down", "top_boost", "tercile_step", "continuous_clip")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7L reversal-gated shared-capital portfolios.")
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
    parser.add_argument("--gate-states-path", default=str(DEFAULT_GATE_STATES_PATH))
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
    parser.add_argument("--gate-mode", choices=GATE_MODES, default=DEFAULT_GATE_MODE)
    parser.add_argument("--reversal-multiplier", type=float, default=DEFAULT_REVERSAL_MULTIPLIER)
    parser.add_argument("--top-reversal-multiplier", type=float, default=DEFAULT_TOP_REVERSAL_MULTIPLIER)
    parser.add_argument("--candidate-pool-per-side", type=int, default=DEFAULT_CANDIDATE_POOL_PER_SIDE)
    parser.add_argument("--max-single-name-side-weight", type=float, default=DEFAULT_MAX_SINGLE_NAME_SIDE_WEIGHT)
    parser.add_argument("--min-nonzero-names", type=int, default=DEFAULT_MIN_NONZERO_NAMES)
    parser.add_argument("--score-weight", type=float, default=DEFAULT_SCORE_WEIGHT)
    parser.add_argument("--sector-penalty", type=float, default=DEFAULT_SECTOR_PENALTY)
    parser.add_argument("--turnover-penalty", type=float, default=DEFAULT_TURNOVER_PENALTY)
    parser.add_argument("--cost-bps-per-side", type=float, default=DEFAULT_COST_BPS_PER_SIDE)
    args = parser.parse_args(argv)

    rollup = build_phase7l_reversal_gate_shared_capital(
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
        gate_states_path=args.gate_states_path,
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
        gate_mode=args.gate_mode,
        reversal_multiplier=args.reversal_multiplier,
        top_reversal_multiplier=args.top_reversal_multiplier,
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


def build_phase7l_reversal_gate_shared_capital(
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
    gate_states_path: str | Path = DEFAULT_GATE_STATES_PATH,
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
    gate_mode: str = DEFAULT_GATE_MODE,
    reversal_multiplier: float = DEFAULT_REVERSAL_MULTIPLIER,
    top_reversal_multiplier: float = DEFAULT_TOP_REVERSAL_MULTIPLIER,
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
    gate = _load_reversal_gate(gate_states_path)
    base_portfolios = _expand_portfolios(
        portfolios,
        turnover_budget_grid=turnover_budget_grid,
        turnover_penalty=turnover_penalty,
    )
    gate_label = _gate_policy_label(
        gate_mode=gate_mode,
        bottom_multiplier=reversal_multiplier,
        top_multiplier=top_reversal_multiplier,
    )
    gated_portfolios = tuple(
        BudgetedPortfolioConfig(
            portfolio=f"{config.portfolio}_{gate_label}",
            base_portfolio=config.base_portfolio,
            factor_weights=dict(config.factor_weights),
            turnover_budget=float(config.turnover_budget),
            turnover_penalty=float(config.turnover_penalty),
        )
        for config in base_portfolios
    )
    positions, diagnostics, skipped = _construct_gated_positions(
        panel,
        gate=gate,
        portfolios=gated_portfolios,
        eval_start=eval_start,
        end=end,
        holding_horizons=holding_horizons,
        factor_min_holds=factor_min_holds,
        gate_mode=gate_mode,
        reversal_multiplier=reversal_multiplier,
        top_reversal_multiplier=top_reversal_multiplier,
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
    gate_summary = _gate_summary(diagnostics)

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
        "panel_sample": output_dir / "phase7l_feature_target_panel_sample.csv",
        "positions": output_dir / "phase7l_gated_positions.csv.gz",
        "diagnostics": output_dir / "phase7l_daily_diagnostics.csv",
        "skipped": output_dir / "phase7l_skipped_sessions.csv",
        "forward_metrics": output_dir / "phase7l_forward_label_metrics.csv",
        "factor_exposure": output_dir / "phase7l_factor_exposure_summary.csv",
        "score_coverage": output_dir / "phase7l_score_coverage.csv",
        "curve": output_dir / "phase7l_strict_daily_curve.csv",
        "strict_metrics": output_dir / "phase7l_strict_daily_metrics.csv",
        "turnover": output_dir / "phase7l_turnover_summary.csv",
        "lot_summary": output_dir / "phase7l_lot_summary.csv",
        "gate_summary": output_dir / "phase7l_gate_summary.csv",
        "memo": output_dir / "phase7l_reversal_gate_memo.md",
        "rollup": output_dir / "phase7l_rollup.json",
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
    gate_summary.to_csv(paths["gate_summary"], index=False)
    paths["memo"].write_text(
        _memo(
            gate_states_path=Path(gate_states_path),
            gate_mode=gate_mode,
            reversal_multiplier=reversal_multiplier,
            top_reversal_multiplier=top_reversal_multiplier,
            eval_start=eval_start,
            end=end,
            cost_bps_per_side=cost_bps_per_side,
            strict_metrics=strict_metrics,
            turnover=turnover,
            gate_summary=gate_summary,
            factor_exposure=factor_exposure,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "phase": "phase7l_reversal_gate_shared_capital",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "variant": variant,
        "start": start,
        "eval_start": eval_start,
        "end": end,
        "validation_price_end": validation_price_end,
        "gate_states_path": Path(gate_states_path).as_posix(),
        "gate_score": DEFAULT_GATE_SCORE,
        "gate_mode": gate_mode,
        "gate_rule": _gate_rule_text(gate_mode),
        "bottom_reversal_multiplier_when_gated": float(reversal_multiplier),
        "top_reversal_multiplier_when_gated": float(top_reversal_multiplier),
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
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    paths["rollup"].write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _construct_gated_positions(
    panel: pd.DataFrame,
    *,
    gate: pd.DataFrame,
    portfolios: Sequence[BudgetedPortfolioConfig],
    eval_start: str,
    end: str,
    holding_horizons: Sequence[int],
    factor_min_holds: Mapping[str, int],
    gate_mode: str,
    reversal_multiplier: float,
    top_reversal_multiplier: float,
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
    gate_lookup = gate.set_index("session_date").to_dict(orient="index")
    for config in portfolios:
        ledger: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
        for session_idx, (session_date, group) in enumerate(sessions):
            gate_row = gate_lookup.get(pd.Timestamp(session_date), {})
            gate_ready = bool(gate_row.get("gate_ready", False))
            gate_bottom_active = bool(gate_row.get("reversal_gate_bottom_active", False))
            gate_top_active = bool(gate_row.get("reversal_gate_top_active", False))
            multiplier = _gate_multiplier(
                gate_row,
                gate_mode=gate_mode,
                bottom_multiplier=reversal_multiplier,
                top_multiplier=top_reversal_multiplier,
            )
            gate_active = bool(np.isfinite(multiplier) and abs(multiplier - 1.0) > 1e-12)
            effective_weights = _gated_factor_weights(
                config.factor_weights,
                multiplier=multiplier,
            )
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
                session_date=str(session_date),
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
                        "gate_ready": gate_ready,
                        "reversal_gate_active": gate_active and REVERSAL_FACTOR in config.factor_weights,
                        "reversal_gate_raw_active": gate_active,
                        "reversal_gate_bottom_active": gate_bottom_active,
                        "reversal_gate_top_active": gate_top_active,
                        "gate_mode": gate_mode,
                        "reversal_multiplier": float(multiplier if REVERSAL_FACTOR in config.factor_weights else 1.0),
                        "bottom_reversal_multiplier": float(reversal_multiplier),
                        "top_reversal_multiplier": float(top_reversal_multiplier),
                        "base_factor_weights_json": json.dumps(dict(config.factor_weights), sort_keys=True),
                        "effective_factor_weights_json": json.dumps(dict(effective_weights), sort_keys=True),
                        "gate_score": _safe_float(gate_row.get(DEFAULT_GATE_SCORE, np.nan)),
                        "gate_q33": _safe_float(gate_row.get(DEFAULT_GATE_QUANTILE_COLUMN, np.nan)),
                        "gate_q67": _safe_float(gate_row.get(DEFAULT_GATE_UPPER_QUANTILE_COLUMN, np.nan)),
                    }
                )
                diagnostics.append(diagnostic)
            if skip is not None:
                skipped.append(skip)
    return pd.DataFrame(positions), pd.DataFrame(diagnostics), pd.DataFrame(skipped)


def _load_reversal_gate(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["session_date"])
    required = {
        "session_date",
        DEFAULT_GATE_SCORE,
        DEFAULT_GATE_QUANTILE_COLUMN,
        DEFAULT_GATE_UPPER_QUANTILE_COLUMN,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"gate states missing required columns: {missing}")
    score = pd.to_numeric(frame[DEFAULT_GATE_SCORE], errors="coerce")
    q33 = pd.to_numeric(frame[DEFAULT_GATE_QUANTILE_COLUMN], errors="coerce")
    q67 = pd.to_numeric(frame[DEFAULT_GATE_UPPER_QUANTILE_COLUMN], errors="coerce")
    frame["gate_ready"] = score.notna() & q33.notna() & q67.notna()
    frame["reversal_gate_bottom_active"] = frame["gate_ready"] & score.le(q33)
    frame["reversal_gate_top_active"] = frame["gate_ready"] & score.ge(q67)
    frame["reversal_gate_active"] = frame["reversal_gate_bottom_active"]
    return frame[
        [
            "session_date",
            DEFAULT_GATE_SCORE,
            DEFAULT_GATE_QUANTILE_COLUMN,
            DEFAULT_GATE_UPPER_QUANTILE_COLUMN,
            "gate_ready",
            "reversal_gate_active",
            "reversal_gate_bottom_active",
            "reversal_gate_top_active",
        ]
    ]


def _gate_multiplier(
    gate_row: Mapping[str, Any],
    *,
    gate_mode: str,
    bottom_multiplier: float,
    top_multiplier: float,
) -> float:
    if not bool(gate_row.get("gate_ready", False)):
        return 1.0
    bottom = max(float(bottom_multiplier), 0.0)
    top = max(float(top_multiplier), 0.0)
    if gate_mode == "bottom_down":
        return bottom if bool(gate_row.get("reversal_gate_bottom_active", False)) else 1.0
    if gate_mode == "top_boost":
        return top if bool(gate_row.get("reversal_gate_top_active", False)) else 1.0
    if gate_mode == "tercile_step":
        if bool(gate_row.get("reversal_gate_bottom_active", False)):
            return bottom
        if bool(gate_row.get("reversal_gate_top_active", False)):
            return top
        return 1.0
    if gate_mode == "continuous_clip":
        score = _safe_float(gate_row.get(DEFAULT_GATE_SCORE, np.nan))
        q33 = _safe_float(gate_row.get(DEFAULT_GATE_QUANTILE_COLUMN, np.nan))
        q67 = _safe_float(gate_row.get(DEFAULT_GATE_UPPER_QUANTILE_COLUMN, np.nan))
        if not np.isfinite(score) or not np.isfinite(q33) or not np.isfinite(q67) or q67 <= q33:
            return 1.0
        mid = 0.5 * (q33 + q67)
        half_width = 0.5 * (q67 - q33)
        clipped = float(np.clip((score - mid) / half_width, -1.0, 1.0))
        if clipped >= 0.0:
            return 1.0 + clipped * (top - 1.0)
        return 1.0 - (-clipped) * (1.0 - bottom)
    raise ValueError(f"unsupported gate mode: {gate_mode}")


def _gated_factor_weights(
    factor_weights: Mapping[str, float],
    *,
    multiplier: float,
) -> dict[str, float]:
    weights = {factor: float(weight) for factor, weight in factor_weights.items()}
    if REVERSAL_FACTOR not in weights:
        return weights
    multiplier = max(float(multiplier), 0.0)
    if not np.isfinite(multiplier) or abs(multiplier - 1.0) <= 1e-12:
        return weights
    original = float(weights[REVERSAL_FACTOR])
    recipients = [factor for factor in weights if factor != REVERSAL_FACTOR and float(weights[factor]) > 0.0]
    if not recipients:
        return weights
    scaled = original * multiplier
    delta = scaled - original
    if delta < 0.0:
        released = -delta
        weights[REVERSAL_FACTOR] = scaled
        add = released / len(recipients)
        for factor in recipients:
            weights[factor] = float(weights[factor]) + add
    elif delta > 0.0:
        available = sum(max(float(weights[factor]), 0.0) for factor in recipients)
        extra = min(delta, available)
        if extra <= 0.0:
            return weights
        weights[REVERSAL_FACTOR] = original + extra
        for factor in recipients:
            share = max(float(weights[factor]), 0.0) / available if available > 0.0 else 0.0
            weights[factor] = max(float(weights[factor]) - extra * share, 0.0)
    return weights


def _gate_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    rows = []
    for portfolio, group in diagnostics.groupby("portfolio", sort=True):
        ready = group["gate_ready"].astype(bool)
        active = group["reversal_gate_active"].astype(bool)
        bottom_active = group["reversal_gate_bottom_active"].astype(bool) if "reversal_gate_bottom_active" in group else active
        top_active = group["reversal_gate_top_active"].astype(bool) if "reversal_gate_top_active" in group else pd.Series(False, index=group.index)
        rows.append(
            {
                "portfolio": portfolio,
                "sessions": int(len(group)),
                "gate_ready_sessions": int(ready.sum()),
                "gate_active_sessions": int(active.sum()),
                "gate_bottom_sessions": int(bottom_active.sum()),
                "gate_top_sessions": int(top_active.sum()),
                "gate_ready_rate": float(ready.mean()),
                "gate_active_rate_all": float(active.mean()),
                "gate_active_rate_ready": float(active.sum() / ready.sum()) if ready.sum() else np.nan,
                "gate_bottom_rate_ready": float(bottom_active.sum() / ready.sum()) if ready.sum() else np.nan,
                "gate_top_rate_ready": float(top_active.sum() / ready.sum()) if ready.sum() else np.nan,
                "mean_reversal_multiplier": float(pd.to_numeric(group["reversal_multiplier"], errors="coerce").mean()),
                "mean_gate_score": float(pd.to_numeric(group["gate_score"], errors="coerce").mean()),
                "mean_gate_q33": float(pd.to_numeric(group["gate_q33"], errors="coerce").mean()),
                "mean_gate_q67": float(pd.to_numeric(group["gate_q67"], errors="coerce").mean()),
                "test_window_used": bool(group["test_window_used"].any()),
            }
        )
    return pd.DataFrame(rows)


def _memo(
    *,
    gate_states_path: Path,
    gate_mode: str,
    reversal_multiplier: float,
    top_reversal_multiplier: float,
    eval_start: str,
    end: str,
    cost_bps_per_side: float,
    strict_metrics: pd.DataFrame,
    turnover: pd.DataFrame,
    gate_summary: pd.DataFrame,
    factor_exposure: pd.DataFrame,
) -> str:
    net = strict_metrics[strict_metrics["return_kind"].eq("net")].copy()
    columns = [
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
    lines = [
        "# Phase7L Reversal-Gated Shared-Capital Results",
        "",
        f"Generated: {_utc_now()}",
        "",
        "Validation-only. The test lockbox is not used.",
        "",
        "## Setup",
        "",
        f"- gate states: `{gate_states_path.as_posix()}`",
        f"- gate mode: `{gate_mode}`",
        f"- gate rule: {_gate_rule_text(gate_mode)}",
        f"- bottom reversal multiplier: `{reversal_multiplier}`",
        f"- top reversal multiplier: `{top_reversal_multiplier}`",
        "- reduced reversal score weight is redistributed equally to other active factors",
        "- boosted reversal score weight is drawn proportionally from other active factors",
        f"- evaluation window: `{eval_start}` through `{end}`",
        f"- cost bps per side: `{cost_bps_per_side}`",
        "",
        "## Strict Daily Net Metrics",
        "",
        _markdown_table(net[columns] if columns else net),
        "",
        "## Gate Summary",
        "",
        _markdown_table(gate_summary),
        "",
        "## Turnover",
        "",
        _markdown_table(turnover),
        "",
        "## Factor Exposure",
        "",
        _markdown_table(factor_exposure),
        "",
        "## Notes",
        "",
        "- Missing or not-ready gate dates keep the original reversal weight.",
        "- This test does not reverse the reversal signal; it only reallocates part of the reversal score weight.",
        "",
    ]
    return "\n".join(lines)


def _multiplier_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _gate_policy_label(*, gate_mode: str, bottom_multiplier: float, top_multiplier: float) -> str:
    if gate_mode == "bottom_down":
        return f"rev_gate{_multiplier_label(bottom_multiplier)}"
    if gate_mode == "top_boost":
        return f"rev_top{_multiplier_label(top_multiplier)}"
    if gate_mode == "tercile_step":
        return f"rev_step{_multiplier_label(bottom_multiplier)}_{_multiplier_label(top_multiplier)}"
    if gate_mode == "continuous_clip":
        return f"rev_linear{_multiplier_label(bottom_multiplier)}_{_multiplier_label(top_multiplier)}"
    raise ValueError(f"unsupported gate mode: {gate_mode}")


def _gate_rule_text(gate_mode: str) -> str:
    if gate_mode == "bottom_down":
        return "ready and reliable_core_score <= reliable_core_score_wf_q33"
    if gate_mode == "top_boost":
        return "ready and reliable_core_score >= reliable_core_score_wf_q67"
    if gate_mode == "tercile_step":
        return "bottom tercile downweight, top tercile boost, middle unchanged"
    if gate_mode == "continuous_clip":
        return "linear multiplier clipped between q33 and q67, with middle at 1.0"
    raise ValueError(f"unsupported gate mode: {gate_mode}")


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
