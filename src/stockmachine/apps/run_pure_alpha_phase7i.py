"""Phase7I shared-capital multifactor optimizer MVP.

Phase7I turns the Phase7F/H factor evidence into a first shared-capital
optimizer: one long-short book, several factor ledgers, and turnover-aware LP
construction. This is validation-only and does not use the test lockbox.
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
    _load_adjusted_prices,
)
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_CIK_MAPPING,
    DEFAULT_SEC_SUBMISSIONS_DIR,
    _load_sec_sic_map,
)
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
from stockmachine.apps.run_pure_alpha_phase7f import (
    DEFAULT_HOLDING_HORIZONS,
    _benchmark_feature_frame,
    _load_beta,
    _load_membership,
)
from stockmachine.apps.run_pure_alpha_phase7g import (
    FUNDAMENTAL_COLUMNS,
    NON_PRICE_COLUMNS,
    SIZE_COLUMNS,
    _load_optional_panel,
    _load_size_panel,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase7i_shared_capital_multifactor_20260517"
DEFAULT_FORMATION_LOOKBACKS = (60, 120)
DEFAULT_SKIP_WINDOWS = (20,)
DEFAULT_MAX_SINGLE_NAME_SIDE_WEIGHT = 1.0 / 30.0
DEFAULT_MIN_NONZERO_NAMES = 20
DEFAULT_CANDIDATE_POOL_PER_SIDE = 120
DEFAULT_SCORE_WEIGHT = 0.01
DEFAULT_SECTOR_PENALTY = 25.0
DEFAULT_TURNOVER_PENALTY = 0.005
DEFAULT_COST_BPS_PER_SIDE = 2.0
FACTOR_COLUMNS = (
    "reversal_score",
    "momentum_score",
    "small_size_score",
    "low_beta_score",
    "cash_quality_score",
)


@dataclass(frozen=True)
class PortfolioConfig:
    portfolio: str
    factor_weights: Mapping[str, float]
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY


DEFAULT_PORTFOLIOS: tuple[PortfolioConfig, ...] = (
    PortfolioConfig(
        "shared_core_lambda_0p005",
        {
            "reversal_score": 0.25,
            "momentum_score": 0.10,
            "small_size_score": 0.30,
            "low_beta_score": 0.20,
            "cash_quality_score": 0.15,
        },
    ),
    PortfolioConfig(
        "shared_no_momentum_lambda_0p005",
        {
            "reversal_score": 0.30,
            "small_size_score": 0.35,
            "low_beta_score": 0.20,
            "cash_quality_score": 0.15,
        },
    ),
    PortfolioConfig(
        "shared_slow_core_lambda_0p005",
        {
            "small_size_score": 0.45,
            "low_beta_score": 0.30,
            "cash_quality_score": 0.25,
        },
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase7I shared-capital multifactor MVP.")
    parser.add_argument("--membership-path", default=str(DEFAULT_PHASE1_MEMBERSHIP))
    parser.add_argument("--beta-panel-path", default=str(DEFAULT_PHASE2_BETA_PANEL))
    parser.add_argument("--size-panel-path", default=str(DEFAULT_SIZE_PANEL))
    parser.add_argument("--non-price-panel-path", default=str(DEFAULT_NON_PRICE_PANEL))
    parser.add_argument("--fundamental-panel-path", default=str(DEFAULT_FUNDAMENTAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--daily-glob", action="append", dest="daily_globs")
    parser.add_argument("--adj-factor-glob", action="append", dest="adj_factor_globs")
    parser.add_argument("--benchmark-daily-glob", action="append", dest="benchmark_daily_globs")
    parser.add_argument(
        "--benchmark-adj-factor-glob",
        action="append",
        dest="benchmark_adj_factor_globs",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--benchmark-symbol", default="SPY")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--eval-start", default=DEFAULT_EVAL_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument(
        "--holding-horizons",
        default=",".join(str(value) for value in DEFAULT_HOLDING_HORIZONS),
    )
    parser.add_argument("--candidate-pool-per-side", type=int, default=DEFAULT_CANDIDATE_POOL_PER_SIDE)
    parser.add_argument("--max-single-name-side-weight", type=float, default=DEFAULT_MAX_SINGLE_NAME_SIDE_WEIGHT)
    parser.add_argument("--min-nonzero-names", type=int, default=DEFAULT_MIN_NONZERO_NAMES)
    parser.add_argument("--score-weight", type=float, default=DEFAULT_SCORE_WEIGHT)
    parser.add_argument("--sector-penalty", type=float, default=DEFAULT_SECTOR_PENALTY)
    parser.add_argument("--cost-bps-per-side", type=float, default=DEFAULT_COST_BPS_PER_SIDE)
    args = parser.parse_args(argv)

    rollup = build_phase7i_shared_capital_multifactor(
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
        benchmark_adj_factor_globs=tuple(
            args.benchmark_adj_factor_globs or DEFAULT_BENCHMARK_ADJ_FACTOR_GLOBS
        ),
        output_root=args.output_root,
        benchmark_symbol=args.benchmark_symbol,
        variant=args.variant,
        start=args.start,
        eval_start=args.eval_start,
        end=args.end,
        holding_horizons=_parse_int_grid(args.holding_horizons),
        candidate_pool_per_side=args.candidate_pool_per_side,
        max_single_name_side_weight=args.max_single_name_side_weight,
        min_nonzero_names=args.min_nonzero_names,
        score_weight=args.score_weight,
        sector_penalty=args.sector_penalty,
        cost_bps_per_side=args.cost_bps_per_side,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


def build_phase7i_shared_capital_multifactor(
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
    benchmark_symbol: str = "SPY",
    variant: str = DEFAULT_VARIANT,
    start: str = DEFAULT_START,
    eval_start: str = DEFAULT_EVAL_START,
    end: str = DEFAULT_END,
    holding_horizons: Sequence[int] = DEFAULT_HOLDING_HORIZONS,
    candidate_pool_per_side: int = DEFAULT_CANDIDATE_POOL_PER_SIDE,
    max_single_name_side_weight: float = DEFAULT_MAX_SINGLE_NAME_SIDE_WEIGHT,
    min_nonzero_names: int = DEFAULT_MIN_NONZERO_NAMES,
    score_weight: float = DEFAULT_SCORE_WEIGHT,
    sector_penalty: float = DEFAULT_SECTOR_PENALTY,
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

    positions, diagnostics, skipped = _construct_shared_capital_positions(
        panel,
        portfolios=portfolios,
        eval_start=eval_start,
        end=end,
        holding_horizons=holding_horizons,
        candidate_pool_per_side=candidate_pool_per_side,
        max_single_name_side_weight=max_single_name_side_weight,
        min_nonzero_names=min_nonzero_names,
        score_weight=score_weight,
        sector_penalty=sector_penalty,
        cost_bps_per_side=cost_bps_per_side,
    )
    metrics = _metrics(diagnostics, skipped=skipped, holding_horizons=holding_horizons)
    factor_exposure = _factor_exposure_summary(diagnostics)
    score_coverage = _score_coverage(panel, eval_start=eval_start, end=end)

    panel_sample_path = output_dir / "phase7i_feature_target_panel_sample.csv"
    positions_path = output_dir / "phase7i_shared_capital_positions.csv.gz"
    diagnostics_path = output_dir / "phase7i_daily_diagnostics.csv"
    skipped_path = output_dir / "phase7i_skipped_sessions.csv"
    metrics_path = output_dir / "phase7i_metrics.csv"
    factor_exposure_path = output_dir / "phase7i_factor_exposure_summary.csv"
    score_coverage_path = output_dir / "phase7i_score_coverage.csv"
    memo_path = output_dir / "phase7i_shared_capital_multifactor_memo.md"
    rollup_path = output_dir / "phase7i_rollup.json"

    panel.head(5000).to_csv(panel_sample_path, index=False)
    positions.to_csv(positions_path, index=False, compression="gzip")
    diagnostics.to_csv(diagnostics_path, index=False)
    skipped.to_csv(skipped_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    factor_exposure.to_csv(factor_exposure_path, index=False)
    score_coverage.to_csv(score_coverage_path, index=False)
    memo_path.write_text(
        _memo(
            variant=variant,
            start=start,
            eval_start=eval_start,
            end=end,
            holding_horizons=holding_horizons,
            portfolios=portfolios,
            candidate_pool_per_side=candidate_pool_per_side,
            max_single_name_side_weight=max_single_name_side_weight,
            score_weight=score_weight,
            sector_penalty=sector_penalty,
            cost_bps_per_side=cost_bps_per_side,
            panel=panel,
            positions=positions,
            diagnostics=diagnostics,
            skipped=skipped,
            metrics=metrics,
            factor_exposure=factor_exposure,
            score_coverage=score_coverage,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "phase": "phase7i_shared_capital_multifactor",
        "scope": "validation_only",
        "test_lockbox_used": False,
        "variant": variant,
        "start": start,
        "eval_start": eval_start,
        "end": end,
        "holding_horizons": [int(value) for value in holding_horizons],
        "candidate_pool_per_side": int(candidate_pool_per_side),
        "max_single_name_side_weight": float(max_single_name_side_weight),
        "min_nonzero_names": int(min_nonzero_names),
        "score_weight": float(score_weight),
        "sector_penalty": float(sector_penalty),
        "cost_bps_per_side": float(cost_bps_per_side),
        "portfolios": [
            {
                "portfolio": config.portfolio,
                "factor_weights": dict(config.factor_weights),
                "turnover_penalty": float(config.turnover_penalty),
            }
            for config in portfolios
        ],
        "rows": {
            "panel": int(len(panel)),
            "positions": int(len(positions)),
            "diagnostics": int(len(diagnostics)),
            "skipped": int(len(skipped)),
        },
        "outputs": {
            "feature_target_panel_sample": panel_sample_path.as_posix(),
            "positions": positions_path.as_posix(),
            "daily_diagnostics": diagnostics_path.as_posix(),
            "skipped_sessions": skipped_path.as_posix(),
            "metrics": metrics_path.as_posix(),
            "factor_exposure_summary": factor_exposure_path.as_posix(),
            "score_coverage": score_coverage_path.as_posix(),
            "memo": memo_path.as_posix(),
        },
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_multifactor_panel(
    *,
    membership_path: str | Path,
    beta_panel_path: str | Path,
    size_panel_path: str | Path,
    non_price_panel_path: str | Path,
    fundamental_panel_path: str | Path,
    cik_mapping_path: str | Path,
    sec_submissions_dir: str | Path,
    daily_globs: Sequence[str | Path],
    adj_factor_globs: Sequence[str | Path],
    benchmark_daily_globs: Sequence[str | Path],
    benchmark_adj_factor_globs: Sequence[str | Path],
    benchmark_symbol: str,
    variant: str,
    start: str,
    end: str,
    holding_horizons: Sequence[int],
) -> pd.DataFrame:
    membership = _load_membership(membership_path, variant=variant, start=start, end=end)
    beta = _load_beta(beta_panel_path)
    size = _load_size_panel(size_panel_path, variant=variant, start=start, end=end)
    non_price = _load_optional_panel(non_price_panel_path, NON_PRICE_COLUMNS, start=start, end=end)
    fundamentals = _load_optional_panel(
        fundamental_panel_path,
        FUNDAMENTAL_COLUMNS,
        start=start,
        end=end,
    )
    sic = _load_sec_sic_map(cik_mapping_path, sec_submissions_dir)
    symbols = tuple(sorted(membership["symbol"].astype(str).unique()))
    stock_prices = _load_adjusted_prices(
        daily_globs=daily_globs,
        adj_factor_globs=adj_factor_globs,
        symbols=symbols,
        end_date=end,
    )
    benchmark_prices = _load_adjusted_prices(
        daily_globs=benchmark_daily_globs,
        adj_factor_globs=benchmark_adj_factor_globs,
        symbols=(benchmark_symbol,),
        end_date=end,
    )
    stock_features = _stock_feature_frame(
        stock_prices,
        holding_horizons=holding_horizons,
    )
    benchmark_features = _benchmark_feature_frame(
        benchmark_prices,
        benchmark_symbol=benchmark_symbol,
        holding_horizons=holding_horizons,
    )
    panel = (
        membership.merge(stock_features, on=["session_date", "symbol"], how="left")
        .merge(beta, on=["session_date", "symbol"], how="left")
        .merge(size, on=["session_date", "symbol"], how="left")
        .merge(non_price, on=["session_date", "symbol"], how="left")
        .merge(fundamentals, on=["session_date", "symbol"], how="left")
        .merge(sic[["symbol", "sic2_sector", "sic4_industry"]], on="symbol", how="left")
        .merge(benchmark_features, on="session_date", how="left")
    )
    panel["sic2_sector"] = panel["sic2_sector"].fillna("UNKNOWN").astype(str)
    panel["sic4_industry"] = panel["sic4_industry"].fillna("UNKNOWN").astype(str)
    for holding in holding_horizons:
        panel[f"forward_beta_residual_return_h{holding}"] = (
            panel[f"forward_return_h{holding}"]
            - panel["beta"] * panel[f"benchmark_forward_return_h{holding}"]
        )
    return panel.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _stock_feature_frame(
    prices: pd.DataFrame,
    *,
    holding_horizons: Sequence[int],
) -> pd.DataFrame:
    frame = prices.sort_values(["symbol", "session_date"]).copy()
    grouped = frame.groupby("symbol", group_keys=False)
    frame["return_5d"] = grouped["adjusted_close"].pct_change(5)
    frame["momentum_l60_s20"] = grouped["adjusted_close"].shift(20) / grouped["adjusted_close"].shift(80) - 1.0
    frame["momentum_l120_s20"] = grouped["adjusted_close"].shift(20) / grouped["adjusted_close"].shift(140) - 1.0
    frame["next_adjusted_open"] = grouped["adjusted_open"].shift(-1)
    for holding in holding_horizons:
        exit_open = grouped["adjusted_open"].shift(-(holding + 1))
        frame[f"forward_return_h{holding}"] = exit_open / frame["next_adjusted_open"] - 1.0
    keep = [
        "session_date",
        "symbol",
        "return_5d",
        "momentum_l60_s20",
        "momentum_l120_s20",
    ]
    keep.extend(f"forward_return_h{holding}" for holding in holding_horizons)
    return frame[keep]


def _add_factor_scores(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    raw_map = {
        "reversal_score": -pd.to_numeric(frame["return_5d"], errors="coerce"),
        "momentum_score": pd.to_numeric(frame["momentum_l120_s20"], errors="coerce"),
        "small_size_score": -pd.to_numeric(frame["market_cap_log"], errors="coerce"),
        "low_beta_score": -pd.to_numeric(frame["beta"], errors="coerce"),
        "cash_quality_score": pd.to_numeric(frame["cash_to_assets"], errors="coerce"),
    }
    for column, raw in raw_map.items():
        raw_column = f"{column}_raw"
        frame[raw_column] = raw.replace([np.inf, -np.inf], np.nan)
        frame[column] = _sector_zscore(frame, raw_column).fillna(0.0)
    return frame


def _sector_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    grouped = values.groupby([frame["session_date"], frame["sic2_sector"]])
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    z = (values - mean) / std.where(std > 1e-12)
    fallback = _date_zscore(frame, column)
    return z.where(np.isfinite(z), fallback).clip(-3.0, 3.0)


def _date_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    grouped = values.groupby(frame["session_date"])
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    return ((values - mean) / std.where(std > 1e-12)).clip(-3.0, 3.0)


def _construct_shared_capital_positions(
    panel: pd.DataFrame,
    *,
    portfolios: Sequence[PortfolioConfig],
    eval_start: str,
    end: str,
    holding_horizons: Sequence[int],
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
    previous_by_portfolio: dict[str, dict[str, dict[str, float]]] = {
        config.portfolio: {"long": {}, "short": {}} for config in portfolios
    }
    for config in portfolios:
        previous = previous_by_portfolio[config.portfolio]
        for session_date, group in frame.groupby("session_date", sort=True):
            scored = _apply_composite_score(group, config.factor_weights)
            book, diagnostic, skip, updated = _construct_one_session(
                scored,
                session_date=str(session_date),
                config=config,
                holding_horizons=holding_horizons,
                candidate_pool_per_side=candidate_pool_per_side,
                max_single_name_side_weight=max_single_name_side_weight,
                min_nonzero_names=min_nonzero_names,
                score_weight=score_weight,
                sector_penalty=sector_penalty,
                cost_bps_per_side=cost_bps_per_side,
                previous_weights=previous,
            )
            positions.extend(book)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
            if skip is not None:
                skipped.append(skip)
            previous = updated
        previous_by_portfolio[config.portfolio] = previous
    return pd.DataFrame(positions), pd.DataFrame(diagnostics), pd.DataFrame(skipped)


def _apply_composite_score(group: pd.DataFrame, factor_weights: Mapping[str, float]) -> pd.DataFrame:
    frame = group.copy()
    score = np.zeros(len(frame), dtype=float)
    total_weight = 0.0
    for column, weight in factor_weights.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        score += float(weight) * values
        total_weight += abs(float(weight))
    if total_weight <= 0.0:
        raise ValueError("factor weights must contain at least one non-zero score")
    frame["composite_score_raw"] = score / total_weight
    frame["composite_score"] = _single_date_zscore(frame["composite_score_raw"]).fillna(0.0)
    return frame


def _single_date_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean()
    std = values.std()
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return ((values - mean) / std).clip(-3.0, 3.0)


def _construct_one_session(
    group: pd.DataFrame,
    *,
    session_date: str,
    config: PortfolioConfig,
    holding_horizons: Sequence[int],
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    min_nonzero_names: int,
    score_weight: float,
    sector_penalty: float,
    cost_bps_per_side: float,
    previous_weights: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, dict[str, dict[str, float]]]:
    needed = ["symbol", "composite_score", "beta", "sic2_sector", *FACTOR_COLUMNS]
    base = group.dropna(subset=needed).drop_duplicates("symbol").copy()
    if len(base) < 2 * min_nonzero_names:
        return [], None, _skip_row(session_date, config.portfolio, len(base), "insufficient_base"), previous_weights
    longs = _candidate_union(
        base,
        side="long",
        previous_weights=previous_weights["long"],
        blocked_symbols=set(previous_weights["short"]),
        candidate_pool_per_side=candidate_pool_per_side,
    )
    shorts = _candidate_union(
        base,
        side="short",
        previous_weights=previous_weights["short"],
        blocked_symbols=set(previous_weights["long"]),
        candidate_pool_per_side=candidate_pool_per_side,
    )
    if len(longs) < min_nonzero_names or len(shorts) < min_nonzero_names:
        return [], None, _skip_row(session_date, config.portfolio, len(base), "insufficient_candidates"), previous_weights
    try:
        long_weights, short_weights = _optimize_joint_weights(
            longs,
            shorts,
            max_weight=max_single_name_side_weight,
            score_weight=score_weight,
            sector_penalty=sector_penalty,
            turnover_penalty=float(config.turnover_penalty),
            previous_long_weights=previous_weights["long"],
            previous_short_weights=previous_weights["short"],
        )
    except ValueError as exc:
        return [], None, _skip_row(session_date, config.portfolio, len(base), f"optimizer_failed:{exc}"), previous_weights
    if (long_weights > 1e-10).sum() < min_nonzero_names:
        return [], None, _skip_row(session_date, config.portfolio, len(base), "insufficient_long_nonzero"), previous_weights
    if (short_weights > 1e-10).sum() < min_nonzero_names:
        return [], None, _skip_row(session_date, config.portfolio, len(base), "insufficient_short_nonzero"), previous_weights

    prev_long_vec = np.array(
        [float(previous_weights["long"].get(symbol, 0.0)) for symbol in longs["symbol"]],
        dtype=float,
    )
    prev_short_vec = np.array(
        [float(previous_weights["short"].get(symbol, 0.0)) for symbol in shorts["symbol"]],
        dtype=float,
    )
    turnover = float(np.abs(long_weights - prev_long_vec).sum() + np.abs(short_weights - prev_short_vec).sum())
    book = [
        *_position_rows(longs, long_weights, session_date=session_date, portfolio=config.portfolio, side="long"),
        *_position_rows(shorts, short_weights, session_date=session_date, portfolio=config.portfolio, side="short"),
    ]
    diagnostic = _diagnostic_row(
        longs=longs,
        shorts=shorts,
        long_weights=long_weights,
        short_weights=short_weights,
        session_date=session_date,
        config=config,
        holding_horizons=holding_horizons,
        turnover=turnover,
        cost_bps_per_side=cost_bps_per_side,
    )
    updated = {
        "long": {
            str(symbol): float(weight)
            for symbol, weight in zip(longs["symbol"], long_weights)
            if float(weight) > 1e-10
        },
        "short": {
            str(symbol): float(weight)
            for symbol, weight in zip(shorts["symbol"], short_weights)
            if float(weight) > 1e-10
        },
    }
    return book, diagnostic, None, updated


def _candidate_union(
    group: pd.DataFrame,
    *,
    side: str,
    previous_weights: dict[str, float],
    blocked_symbols: set[str],
    candidate_pool_per_side: int,
) -> pd.DataFrame:
    previous_symbols = set(previous_weights)
    if side == "long":
        ranked = group[~group["symbol"].astype(str).isin(blocked_symbols)].sort_values(
            ["composite_score", "symbol"],
            ascending=[False, True],
        )
    elif side == "short":
        ranked = group[~group["symbol"].astype(str).isin(blocked_symbols)].sort_values(
            ["composite_score", "symbol"],
            ascending=[True, True],
        )
    else:
        raise ValueError(f"unknown_side:{side}")
    top = ranked.head(candidate_pool_per_side)
    incumbents = group[group["symbol"].astype(str).isin(previous_symbols)].copy()
    union = pd.concat([top, incumbents], ignore_index=True).drop_duplicates("symbol", keep="first")
    return union.reset_index(drop=True)


def _optimize_joint_weights(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    *,
    max_weight: float,
    score_weight: float,
    sector_penalty: float,
    turnover_penalty: float,
    previous_long_weights: dict[str, float],
    previous_short_weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    n_long = len(longs)
    n_short = len(shorts)
    n = n_long + n_short
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
        [(0.0, max_weight)] * n + [(0.0, None)] * n + [(0.0, None)] * m_sector
    )
    a_eq = np.column_stack([a_eq, np.zeros((a_eq.shape[0], n + m_sector))])

    turnover_top = np.column_stack([np.eye(n), -np.eye(n), np.zeros((n, m_sector))])
    turnover_bottom = np.column_stack([-np.eye(n), -np.eye(n), np.zeros((n, m_sector))])
    sector_top = np.column_stack([sector_matrix, np.zeros((m_sector, n)), -np.eye(m_sector)])
    sector_bottom = np.column_stack([-sector_matrix, np.zeros((m_sector, n)), -np.eye(m_sector)])
    result = linprog(
        c,
        A_ub=np.vstack([turnover_top, turnover_bottom, sector_top, sector_bottom]),
        b_ub=np.concatenate([prev, -prev, np.zeros(m_sector), np.zeros(m_sector)]),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise ValueError(str(result.message).replace(" ", "_"))
    x = np.clip(result.x[:n], 0.0, max_weight)
    long_weights = x[:n_long]
    short_weights = x[n_long:]
    if abs(long_weights.sum() - 1.0) > 1e-6 or abs(short_weights.sum() - 1.0) > 1e-6:
        raise ValueError("side_sum_constraint_breach")
    net_beta = float(np.dot(long_weights, longs["beta"]) - np.dot(short_weights, shorts["beta"]))
    if abs(net_beta) > 1e-5:
        raise ValueError("beta_constraint_breach")
    return long_weights, short_weights


def _group_exposure_matrix(longs: pd.DataFrame, shorts: pd.DataFrame, group_column: str) -> np.ndarray:
    groups = sorted(set(longs[group_column].astype(str)).union(set(shorts[group_column].astype(str))))
    n_long = len(longs)
    n_short = len(shorts)
    matrix = np.zeros((len(groups), n_long + n_short), dtype=float)
    long_values = longs[group_column].astype(str).to_numpy()
    short_values = shorts[group_column].astype(str).to_numpy()
    for i, group in enumerate(groups):
        matrix[i, :n_long] = (long_values == group).astype(float)
        matrix[i, n_long:] = -((short_values == group).astype(float))
    return matrix


def _position_rows(
    candidates: pd.DataFrame,
    weights: np.ndarray,
    *,
    session_date: str,
    portfolio: str,
    side: str,
) -> list[dict[str, Any]]:
    sign = 1.0 if side == "long" else -1.0
    rows = []
    for row, weight in zip(candidates.itertuples(index=False), weights):
        weight = float(weight)
        if weight <= 1e-10:
            continue
        item = {
            "session_date": session_date,
            "portfolio": portfolio,
            "side": side,
            "symbol": str(row.symbol),
            "side_weight": weight,
            "signed_weight": sign * weight,
            "beta": float(row.beta),
            "sic2_sector": str(row.sic2_sector),
            "sic4_industry": str(row.sic4_industry),
            "composite_score": float(row.composite_score),
            "test_window_used": False,
        }
        for column in FACTOR_COLUMNS:
            item[column] = float(getattr(row, column))
        rows.append(item)
    return rows


def _diagnostic_row(
    *,
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    long_weights: np.ndarray,
    short_weights: np.ndarray,
    session_date: str,
    config: PortfolioConfig,
    holding_horizons: Sequence[int],
    turnover: float,
    cost_bps_per_side: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "session_date": session_date,
        "portfolio": config.portfolio,
        "factor_weights_json": json.dumps(dict(config.factor_weights), sort_keys=True),
        "turnover_penalty": float(config.turnover_penalty),
        "target_turnover": float(turnover),
        "cost_bps": float(turnover * cost_bps_per_side),
        "long_names": int((long_weights > 1e-10).sum()),
        "short_names": int((short_weights > 1e-10).sum()),
        "long_beta": float(np.dot(long_weights, longs["beta"])),
        "short_beta": float(np.dot(short_weights, shorts["beta"])),
        "net_beta": float(np.dot(long_weights, longs["beta"]) - np.dot(short_weights, shorts["beta"])),
        "long_composite_score": float(np.dot(long_weights, longs["composite_score"])),
        "short_composite_score": float(np.dot(short_weights, shorts["composite_score"])),
        "score_spread": float(
            np.dot(long_weights, longs["composite_score"])
            - np.dot(short_weights, shorts["composite_score"])
        ),
        "gross_exposure": 2.0,
        "net_exposure": 0.0,
        "test_window_used": False,
    }
    for column in FACTOR_COLUMNS:
        row[f"{column}_exposure"] = float(
            np.dot(long_weights, longs[column]) - np.dot(short_weights, shorts[column])
        )
        row[f"long_{column}"] = float(np.dot(long_weights, longs[column]))
        row[f"short_{column}"] = float(np.dot(short_weights, shorts[column]))
    row.update(_sector_diagnostics(longs, shorts, long_weights, short_weights))
    for holding in holding_horizons:
        target_column = f"forward_beta_residual_return_h{holding}"
        payoff = _weighted_spread(longs, shorts, long_weights, short_weights, target_column)
        row[f"gross_payoff_h{holding}_bps"] = payoff * 10000.0 if np.isfinite(payoff) else np.nan
        row[f"net_payoff_h{holding}_bps"] = (
            payoff * 10000.0 - row["cost_bps"] if np.isfinite(payoff) else np.nan
        )
    return row


def _weighted_spread(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    long_weights: np.ndarray,
    short_weights: np.ndarray,
    column: str,
) -> float:
    long_values = pd.to_numeric(longs[column], errors="coerce").to_numpy(dtype=float)
    short_values = pd.to_numeric(shorts[column], errors="coerce").to_numpy(dtype=float)
    active_long = long_weights > 1e-10
    active_short = short_weights > 1e-10
    if not np.isfinite(long_values[active_long]).all() or not np.isfinite(short_values[active_short]).all():
        return np.nan
    return float(np.dot(long_weights, long_values) - np.dot(short_weights, short_values))


def _sector_diagnostics(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    long_weights: np.ndarray,
    short_weights: np.ndarray,
) -> dict[str, Any]:
    sectors = sorted(set(longs["sic2_sector"].astype(str)).union(set(shorts["sic2_sector"].astype(str))))
    exposures = []
    for sector in sectors:
        long_exposure = float(long_weights[longs["sic2_sector"].astype(str).to_numpy() == sector].sum())
        short_exposure = float(short_weights[shorts["sic2_sector"].astype(str).to_numpy() == sector].sum())
        exposures.append(long_exposure - short_exposure)
    values = np.array(exposures, dtype=float)
    return {
        "sector_abs_exposure_sum": float(np.abs(values).sum()) if len(values) else 0.0,
        "sector_max_abs_exposure": float(np.abs(values).max()) if len(values) else 0.0,
        "long_sectors": int((pd.Series(longs["sic2_sector"].astype(str)).groupby(longs["sic2_sector"].astype(str)).size() > 0).sum()),
        "short_sectors": int((pd.Series(shorts["sic2_sector"].astype(str)).groupby(shorts["sic2_sector"].astype(str)).size() > 0).sum()),
    }


def _skip_row(session_date: str, portfolio: str, candidates: int, reason: str) -> dict[str, Any]:
    return {
        "session_date": str(session_date),
        "portfolio": portfolio,
        "candidates": int(candidates),
        "skip_reason": reason,
        "test_window_used": False,
    }


def _metrics(
    diagnostics: pd.DataFrame,
    *,
    skipped: pd.DataFrame,
    holding_horizons: Sequence[int],
) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    rows = []
    total_sessions_by_portfolio = (
        diagnostics.groupby("portfolio")["session_date"].nunique()
        + skipped.groupby("portfolio")["session_date"].nunique()
        if not skipped.empty
        else diagnostics.groupby("portfolio")["session_date"].nunique()
    )
    for portfolio, group in diagnostics.groupby("portfolio", sort=True):
        base = {
            "portfolio": portfolio,
            "constructed_sessions": int(group["session_date"].nunique()),
            "total_sessions_with_skips": int(total_sessions_by_portfolio.get(portfolio, group["session_date"].nunique())),
            "construction_rate": float(
                group["session_date"].nunique()
                / max(1, total_sessions_by_portfolio.get(portfolio, group["session_date"].nunique()))
            ),
            "mean_target_turnover": float(group["target_turnover"].mean()),
            "median_target_turnover": float(group["target_turnover"].median()),
            "mean_cost_bps": float(group["cost_bps"].mean()),
            "mean_abs_net_beta": float(group["net_beta"].abs().mean()),
            "mean_sector_abs_exposure_sum": float(group["sector_abs_exposure_sum"].mean()),
            "mean_sector_max_abs_exposure": float(group["sector_max_abs_exposure"].mean()),
            "mean_long_names": float(group["long_names"].mean()),
            "mean_short_names": float(group["short_names"].mean()),
            "test_window_used": False,
        }
        for holding in holding_horizons:
            for kind in ("gross", "net"):
                column = f"{kind}_payoff_h{holding}_bps"
                series = pd.to_numeric(group[column], errors="coerce").dropna()
                item = dict(base)
                item.update(
                    {
                        "payoff_kind": kind,
                        "holding": int(holding),
                        "payoff_sessions": int(len(series)),
                        "mean_payoff_bps": float(series.mean()) if len(series) else np.nan,
                        "median_payoff_bps": float(series.median()) if len(series) else np.nan,
                        "hit_rate": float((series > 0.0).mean()) if len(series) else np.nan,
                        "t_stat": _t_stat(series / 10000.0) if len(series) else np.nan,
                        "newey_west_lag": int(max(0, holding - 1)),
                        "newey_west_t_stat": _newey_west_t_stat(series / 10000.0, lag=max(0, holding - 1))
                        if len(series)
                        else np.nan,
                    }
                )
                rows.append(item)
    return pd.DataFrame(rows).sort_values(
        ["holding", "payoff_kind", "mean_payoff_bps"],
        ascending=[True, True, False],
    )


def _factor_exposure_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    rows = []
    for portfolio, group in diagnostics.groupby("portfolio", sort=True):
        for column in FACTOR_COLUMNS:
            exposure = pd.to_numeric(group[f"{column}_exposure"], errors="coerce").dropna()
            rows.append(
                {
                    "portfolio": portfolio,
                    "factor": column,
                    "mean_exposure": float(exposure.mean()),
                    "median_exposure": float(exposure.median()),
                    "positive_exposure_rate": float((exposure > 0.0).mean()),
                    "p10_exposure": float(exposure.quantile(0.10)),
                    "p90_exposure": float(exposure.quantile(0.90)),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows)


def _score_coverage(panel: pd.DataFrame, *, eval_start: str, end: str) -> pd.DataFrame:
    frame = panel[panel["session_date"].ge(eval_start) & panel["session_date"].le(end)].copy()
    rows = []
    for column in FACTOR_COLUMNS:
        raw_column = f"{column}_raw"
        raw = pd.to_numeric(frame[raw_column], errors="coerce")
        score = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "factor": column,
                "raw_coverage": float(raw.notna().mean()),
                "score_coverage": float(score.notna().mean()),
                "raw_nonzero_rate": float((raw.fillna(0.0) != 0.0).mean()),
                "score_nonzero_rate": float((score.fillna(0.0) != 0.0).mean()),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows)


def _zscore_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros_like(values)
    return np.clip((values - mean) / std, -3.0, 3.0)


def _newey_west_t_stat(series: pd.Series, *, lag: int) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 2:
        return np.nan
    mean = float(values.mean())
    centered = values - mean
    lag = min(max(int(lag), 0), n - 1)
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    for step in range(1, lag + 1):
        weight = 1.0 - step / (lag + 1.0)
        gamma = float(np.dot(centered[step:], centered[:-step]) / n)
        long_run_variance += 2.0 * weight * gamma
    if not np.isfinite(long_run_variance) or long_run_variance <= 0.0:
        return np.nan
    standard_error = float(np.sqrt(long_run_variance / n))
    if standard_error <= 0.0:
        return np.nan
    return mean / standard_error


def _memo(
    *,
    variant: str,
    start: str,
    eval_start: str,
    end: str,
    holding_horizons: Sequence[int],
    portfolios: Sequence[PortfolioConfig],
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    score_weight: float,
    sector_penalty: float,
    cost_bps_per_side: float,
    panel: pd.DataFrame,
    positions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    skipped: pd.DataFrame,
    metrics: pd.DataFrame,
    factor_exposure: pd.DataFrame,
    score_coverage: pd.DataFrame,
) -> str:
    portfolio_rows = pd.DataFrame(
        [
            {
                "portfolio": config.portfolio,
                "factor_weights": json.dumps(dict(config.factor_weights), sort_keys=True),
                "turnover_penalty": float(config.turnover_penalty),
            }
            for config in portfolios
        ]
    )
    top_metrics = metrics[
        metrics["payoff_kind"].eq("net") & metrics["holding"].isin([10, 20, 60])
    ].sort_values(["holding", "mean_payoff_bps"], ascending=[True, False])
    lines = [
        "# Phase7I Shared-Capital Multifactor Optimizer Results",
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
        f"- holding horizons: `{list(holding_horizons)}`",
        f"- candidate pool per side: `{candidate_pool_per_side}`",
        f"- max single-name side weight: `{max_single_name_side_weight}`",
        f"- score weight: `{score_weight}`",
        f"- SIC2 sector soft penalty: `{sector_penalty}`",
        f"- cost bps per side: `{cost_bps_per_side}`",
        f"- panel rows: `{len(panel)}`",
        f"- position rows: `{len(positions)}`",
        f"- diagnostics rows: `{len(diagnostics)}`",
        f"- skipped rows: `{len(skipped)}`",
        "",
        "## Portfolios",
        "",
        _markdown_table(portfolio_rows),
        "",
        "## Net Payoff Metrics",
        "",
        _markdown_table(top_metrics),
        "",
        "## Factor Exposure Ledger",
        "",
        _markdown_table(factor_exposure),
        "",
        "## Score Coverage",
        "",
        _markdown_table(score_coverage),
        "",
        "## Notes",
        "",
        "- This is a target-portfolio forward-label diagnostic, not a strict daily execution path.",
        "- A single stock can receive a higher composite score by carrying several positive factor exposures at once.",
        "- Costs are approximated as `target_turnover * cost_bps_per_side` per rebalance.",
        "- The optimizer enforces hard long/short side sums and hard beta neutrality, with SIC2 sector exposure handled by slack penalty.",
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
