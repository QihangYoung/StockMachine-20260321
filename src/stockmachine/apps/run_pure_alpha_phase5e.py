from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from stockmachine.apps.run_pure_alpha_phase4aa import _add_former_winner_scores
from stockmachine.apps.run_pure_alpha_phase4ab import _add_hybrid_scores
from stockmachine.apps.run_pure_alpha_phase4ag import PLOT_COLORS as PHASE4AG_PLOT_COLORS
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_CIK_MAPPING,
    DEFAULT_H10_SIGNAL_PANEL,
    DEFAULT_LONG_SCORE,
    DEFAULT_LONG_VARIANT,
    DEFAULT_SEC_SUBMISSIONS_DIR,
    DEFAULT_SHORT_VARIANT,
    RESIDUAL_TARGETS,
    TARGET_COLUMN,
    _add_residual_targets,
    _diagnostic_row,
    _load_panel,
    _load_sec_sic_map,
    _position_rows,
    _skip_row,
    _zscore,
)
from stockmachine.apps.run_pure_alpha_phase4z import (
    DEFAULT_PHASE3_H10_ROLLUP,
    build_phase4z_strict_horizon_daily_paths,
)
from stockmachine.apps.run_pure_alpha_phase5a import (
    DEFAULT_BASELINE_PORTFOLIO,
    DEFAULT_H10_COMPARE_CURVE_PATH,
    DEFAULT_TARGET_POSITIONS_PATH,
    _benchmark_lookup,
    _load_h10_compare_curve,
)
from stockmachine.apps.run_pure_alpha_phase5b import (
    _build_h10_turnover_curve,
    _fmt_float_like,
    _fmt_pct_like,
    _reference_metrics_from_comparison,
    _text_table,
    _utc_now,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5e_turnover_aware_optimizer_20260427"
DEFAULT_STRICT_ROOT = DEFAULT_OUTPUT_ROOT / "strict_h10_rebuild"
DEFAULT_PORTFOLIO = "sic2_soft_neutral"
DEFAULT_TARGET_PORTFOLIO = "sic2_soft_neutral__short_hybrid_soft_fw_overlay"
DEFAULT_SHORT_SELECTOR = "short_hybrid_soft_fw_overlay"
DEFAULT_HOLDING_PERIOD_SESSIONS = 10
TURNOVER_PENALTIES: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1)
PLOT_COLORS = {
    "baseline": PHASE4AG_PLOT_COLORS["sic2_soft_neutral"],
    "short_hybrid": PHASE4AG_PLOT_COLORS["sic2_soft_neutral__short_hybrid_soft_fw_overlay"],
    "turnover_aware": "#c2410c",
    "spy_raw": "#6b7280",
}


def build_phase5e_turnover_aware_optimizer_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    strict_root: str | Path = DEFAULT_STRICT_ROOT,
    phase3_rollup_path: str | Path = DEFAULT_PHASE3_H10_ROLLUP,
    h10_compare_curve_path: str | Path = DEFAULT_H10_COMPARE_CURVE_PATH,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    long_score: str = DEFAULT_LONG_SCORE,
    short_selector: str = DEFAULT_SHORT_SELECTOR,
    candidate_pool_per_side: int = 80,
    max_single_name_side_weight: float = 1.0 / 30.0,
    min_nonzero_names: int = 20,
    score_weight: float = 0.01,
    soft_group_penalty: float = 25.0,
    turnover_penalties: Sequence[float] = TURNOVER_PENALTIES,
    min_regression_rows: int = 80,
    min_dummy_count: int = 5,
    holding_period_sessions: int = DEFAULT_HOLDING_PERIOD_SESSIONS,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    industry_map = _load_sec_sic_map(cik_mapping_path, sec_submissions_dir)
    panel = _load_panel(signal_panel_path, long_variant, short_variant)
    panel = _add_former_winner_scores(panel)
    panel = _add_hybrid_scores(panel)
    panel = panel.merge(industry_map, on="symbol", how="left")
    panel["sic2_sector"] = panel["sic2_sector"].fillna("UNKNOWN")
    panel["sic4_industry"] = panel["sic4_industry"].fillna("UNKNOWN")
    panel["sic_description"] = panel["sic_description"].fillna("UNKNOWN")
    panel = _add_residual_targets(
        panel,
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )

    positions, diagnostics, skipped = _construct_turnover_aware_positions(
        panel,
        long_variant=long_variant,
        short_variant=short_variant,
        long_score=long_score,
        short_selector=short_selector,
        candidate_pool_per_side=candidate_pool_per_side,
        max_single_name_side_weight=max_single_name_side_weight,
        min_nonzero_names=min_nonzero_names,
        score_weight=score_weight,
        soft_group_penalty=soft_group_penalty,
        turnover_penalties=tuple(float(v) for v in turnover_penalties),
    )

    positions_path = output_dir / "phase5e_turnover_aware_positions_validation.csv.gz"
    diagnostics_path = output_dir / "phase5e_turnover_aware_daily_validation.csv"
    skipped_path = output_dir / "phase5e_turnover_aware_skipped_validation.csv"
    positions.to_csv(positions_path, index=False, compression="gzip")
    diagnostics.to_csv(diagnostics_path, index=False)
    skipped.to_csv(skipped_path, index=False)

    strict_rollup = build_phase4z_strict_horizon_daily_paths(
        positions_path=positions_path,
        output_root=strict_root,
        phase3_rollup_path=phase3_rollup_path,
        existing_curve_path=None,
        holding_period_sessions=holding_period_sessions,
        lead_portfolio=_portfolio_label(float(tuple(turnover_penalties)[0])),
    )

    strict_curve_path = Path(strict_root) / "phase4z_strict_h10_daily_curve.csv"
    strict_curve = pd.read_csv(strict_curve_path, parse_dates=["return_date"])
    h10_compare_curve = _load_h10_compare_curve(
        h10_compare_curve_path,
        baseline_portfolio=DEFAULT_BASELINE_PORTFOLIO,
        target_portfolio=DEFAULT_TARGET_PORTFOLIO,
    )

    target_turnover = _target_turnover_summary(positions)
    aggregate_turnover = _aggregate_turnover_summary(
        positions_path=positions_path,
        benchmark_returns=h10_compare_curve[["return_date", "benchmark_oto_return"]].rename(
            columns={"return_date": "session_date"}
        ),
        holding_period_sessions=holding_period_sessions,
    )
    strict_metrics = _strict_metric_summary(strict_curve)
    sweep = target_turnover.merge(aggregate_turnover, on="portfolio", how="left").merge(
        strict_metrics,
        on="portfolio",
        how="left",
    )
    sweep = sweep.merge(_diagnostic_summary(diagnostics), on="portfolio", how="left")
    sweep["penalty"] = sweep["portfolio"].map(_penalty_from_portfolio)

    lead_aggregate_turnover = float(
        _build_h10_turnover_curve(
            positions_path=DEFAULT_TARGET_POSITIONS_PATH,
            portfolio=DEFAULT_TARGET_PORTFOLIO,
            benchmark_calendar=h10_compare_curve["return_date"],
            holding_period_sessions=holding_period_sessions,
        )["h10_turnover"].iloc[holding_period_sessions:].mean()
    )
    sweep["aggregate_turnover_vs_lead"] = sweep["aggregate_turnover_mean"] / lead_aggregate_turnover
    eligible = sweep[sweep["penalty"] > 0].copy()
    if eligible.empty:
        raise ValueError("No turnover-aware portfolios were built.")
    best = eligible.sort_values(
        ["strict_sharpe_no_rf", "strict_annualized_return"],
        ascending=[False, False],
    ).iloc[0]
    best_portfolio = str(best["portfolio"])

    comparison = _build_comparison_curve(
        h10_compare_curve=h10_compare_curve,
        strict_curve=strict_curve,
        best_portfolio=best_portfolio,
    )
    metrics = _build_metrics(
        comparison=comparison,
        best_row=best.to_dict(),
        lead_aggregate_turnover=lead_aggregate_turnover,
    )

    sweep_path = output_dir / "phase5e_turnover_aware_sweep.csv"
    comparison_curve_path = output_dir / "phase5e_comparison_curve.csv"
    metrics_path = output_dir / "phase5e_metrics.csv"
    plot_path = output_dir / "phase5e_comparison_plot.png"
    memo_path = output_dir / "phase5e_turnover_aware_optimizer_memo.md"
    rollup_path = output_dir / "phase5e_rollup.json"

    sweep.to_csv(sweep_path, index=False)
    comparison.to_csv(comparison_curve_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    _plot_comparison(comparison, output_path=plot_path)
    memo_path.write_text(
        _memo(
            metrics=metrics,
            sweep=sweep,
            best_portfolio=best_portfolio,
            plot_path=plot_path,
            lead_aggregate_turnover=lead_aggregate_turnover,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "phase3_rollup_path": Path(phase3_rollup_path).as_posix(),
        "positions_artifact": positions_path.as_posix(),
        "diagnostics_artifact": diagnostics_path.as_posix(),
        "skipped_artifact": skipped_path.as_posix(),
        "strict_rollup": strict_rollup,
        "sweep_artifact": sweep_path.as_posix(),
        "comparison_curve_artifact": comparison_curve_path.as_posix(),
        "metrics_artifact": metrics_path.as_posix(),
        "plot_artifact": plot_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lead_aggregate_turnover": float(lead_aggregate_turnover),
        "best_portfolio": best_portfolio,
        "turnover_penalties": [float(v) for v in turnover_penalties],
        "method": "turnover_penalty_inside_daily_joint_optimizer",
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _construct_turnover_aware_positions(
    panel: pd.DataFrame,
    *,
    long_variant: str,
    short_variant: str,
    long_score: str,
    short_selector: str,
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    min_nonzero_names: int,
    score_weight: float,
    soft_group_penalty: float,
    turnover_penalties: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    groups = {
        (variant, session_date): group
        for (variant, session_date), group in panel.groupby(["variant", "session_date"], sort=True)
    }
    sessions = sorted(panel["session_date"].unique())
    previous_by_penalty: dict[float, dict[str, dict[str, float]]] = {
        float(penalty): {"long": {}, "short": {}} for penalty in turnover_penalties
    }

    for penalty in turnover_penalties:
        portfolio = _portfolio_label(float(penalty))
        previous_side = previous_by_penalty[float(penalty)]
        for session_date in sessions:
            long_group = groups.get((long_variant, session_date))
            short_group = groups.get((short_variant, session_date))
            if long_group is None or short_group is None:
                skipped.append(_skip_row(session_date, portfolio, 0, 0, "missing_group"))
                continue
            book, diagnostic, skip, prev_side = _construct_one_session_turnover_aware(
                long_group,
                short_group,
                session_date=session_date,
                portfolio=portfolio,
                long_variant=long_variant,
                short_variant=short_variant,
                long_score=long_score,
                short_selector=short_selector,
                soft_group="sic2_sector",
                candidate_pool_per_side=candidate_pool_per_side,
                max_single_name_side_weight=max_single_name_side_weight,
                min_nonzero_names=min_nonzero_names,
                score_weight=score_weight,
                soft_group_penalty=soft_group_penalty,
                turnover_penalty=float(penalty),
                previous_weights=previous_side,
            )
            positions.extend(book)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
            if skip is not None:
                skipped.append(skip)
            previous_side = prev_side
            previous_by_penalty[float(penalty)] = prev_side
    return pd.DataFrame(positions), pd.DataFrame(diagnostics), pd.DataFrame(skipped)


def _construct_one_session_turnover_aware(
    long_group: pd.DataFrame,
    short_group: pd.DataFrame,
    *,
    session_date: str,
    portfolio: str,
    long_variant: str,
    short_variant: str,
    long_score: str,
    short_selector: str,
    soft_group: str | None,
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    min_nonzero_names: int,
    score_weight: float,
    soft_group_penalty: float,
    turnover_penalty: float,
    previous_weights: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, dict[str, dict[str, float]]]:
    needed = [
        long_score,
        short_selector,
        "beta",
        "forward_return_5d",
        "benchmark_forward_return_5d",
        TARGET_COLUMN,
        *RESIDUAL_TARGETS,
        "sic2_sector",
        "sic4_industry",
    ]
    long_base = long_group.dropna(subset=[long_score, *needed[2:]]).drop_duplicates("symbol").copy()
    short_base = short_group.dropna(subset=[short_selector, *needed[2:]]).drop_duplicates("symbol").copy()
    if len(long_base) < min_nonzero_names or len(short_base) < min_nonzero_names:
        return [], None, _skip_row(session_date, portfolio, len(long_base), len(short_base), "insufficient_candidates"), previous_weights

    longs = _candidate_union(
        long_base,
        score_column=long_score,
        previous_weights=previous_weights["long"],
        candidate_pool_per_side=candidate_pool_per_side,
    )
    shorts = _candidate_union(
        short_base,
        score_column=short_selector,
        previous_weights=previous_weights["short"],
        candidate_pool_per_side=candidate_pool_per_side,
    )
    try:
        long_weights, short_weights = _optimize_turnover_aware_joint_weights(
            longs,
            shorts,
            long_score=long_score,
            short_selector=short_selector,
            soft_group=soft_group,
            max_weight=max_single_name_side_weight,
            score_weight=score_weight,
            soft_group_penalty=soft_group_penalty,
            turnover_penalty=turnover_penalty,
            previous_long_weights=previous_weights["long"],
            previous_short_weights=previous_weights["short"],
        )
    except ValueError as exc:
        return [], None, _skip_row(session_date, portfolio, len(longs), len(shorts), f"optimizer_failed:{exc}"), previous_weights

    if (long_weights > 1e-10).sum() < min_nonzero_names:
        return [], None, _skip_row(session_date, portfolio, len(longs), len(shorts), "insufficient_long_nonzero"), previous_weights
    if (short_weights > 1e-10).sum() < min_nonzero_names:
        return [], None, _skip_row(session_date, portfolio, len(longs), len(shorts), "insufficient_short_nonzero"), previous_weights

    book = [
        *_position_rows(
            longs,
            long_weights,
            session_date=session_date,
            portfolio=portfolio,
            long_variant=long_variant,
            short_variant=short_variant,
            side="long",
            score_column=long_score,
        ),
        *_position_rows(
            shorts,
            short_weights,
            session_date=session_date,
            portfolio=portfolio,
            long_variant=long_variant,
            short_variant=short_variant,
            side="short",
            score_column=short_selector,
        ),
    ]
    diagnostic = _diagnostic_row(
        longs,
        shorts,
        long_weights,
        short_weights,
        session_date=session_date,
        portfolio=portfolio,
        long_variant=long_variant,
        short_variant=short_variant,
        hard_group=None,
        soft_group=soft_group,
        long_score=long_score,
        short_selector=short_selector,
    )
    prev_long_vec = np.array([float(previous_weights["long"].get(symbol, 0.0)) for symbol in longs["symbol"]], dtype=float)
    prev_short_vec = np.array([float(previous_weights["short"].get(symbol, 0.0)) for symbol in shorts["symbol"]], dtype=float)
    diagnostic["turnover_penalty"] = float(turnover_penalty)
    diagnostic["target_turnover"] = float(np.abs(long_weights - prev_long_vec).sum() + np.abs(short_weights - prev_short_vec).sum())
    diagnostic["target_new_long_names"] = int(((long_weights > 1e-10) & (prev_long_vec <= 1e-10)).sum())
    diagnostic["target_new_short_names"] = int(((short_weights > 1e-10) & (prev_short_vec <= 1e-10)).sum())
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
    score_column: str,
    previous_weights: dict[str, float],
    candidate_pool_per_side: int,
) -> pd.DataFrame:
    top = group.sort_values([score_column, "symbol"], ascending=[False, True]).head(candidate_pool_per_side)
    if not previous_weights:
        return top.reset_index(drop=True)
    incumbents = group[group["symbol"].astype(str).isin(set(previous_weights))].copy()
    union = pd.concat([top, incumbents], ignore_index=True).drop_duplicates("symbol", keep="first")
    return union.sort_values([score_column, "symbol"], ascending=[False, True]).reset_index(drop=True)


def _optimize_turnover_aware_joint_weights(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    *,
    long_score: str,
    short_selector: str,
    soft_group: str | None,
    max_weight: float,
    score_weight: float,
    soft_group_penalty: float,
    turnover_penalty: float,
    previous_long_weights: dict[str, float],
    previous_short_weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    n_long = len(longs)
    n_short = len(shorts)
    n = n_long + n_short
    if n_long * max_weight < 1.0 - 1e-12 or n_short * max_weight < 1.0 - 1e-12:
        raise ValueError("insufficient_weight_capacity")
    score = np.concatenate(
        [
            _zscore(longs[long_score].to_numpy(dtype=float)),
            _zscore(shorts[short_selector].to_numpy(dtype=float)),
        ]
    )
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

    soft_matrix = (
        _group_exposure_matrix(longs, shorts, soft_group)
        if soft_group is not None
        else np.zeros((0, n), dtype=float)
    )
    m_soft = soft_matrix.shape[0]
    c = np.concatenate(
        [
            -score_weight * score,
            np.full(n, turnover_penalty, dtype=float),
            np.full(m_soft, soft_group_penalty, dtype=float),
        ]
    )
    bounds: list[tuple[float, float | None]] = (
        [(0.0, max_weight)] * n + [(0.0, None)] * n + [(0.0, None)] * m_soft
    )
    a_eq = np.column_stack([a_eq, np.zeros((a_eq.shape[0], n + m_soft))])

    turnover_top = np.column_stack([np.eye(n), -np.eye(n), np.zeros((n, m_soft))])
    turnover_bottom = np.column_stack([-np.eye(n), -np.eye(n), np.zeros((n, m_soft))])
    a_ub = [turnover_top, turnover_bottom]
    b_ub = [prev, -prev]
    if m_soft:
        soft_top = np.column_stack([soft_matrix, np.zeros((m_soft, n)), -np.eye(m_soft)])
        soft_bottom = np.column_stack([-soft_matrix, np.zeros((m_soft, n)), -np.eye(m_soft)])
        a_ub.extend([soft_top, soft_bottom])
        b_ub.extend([np.zeros(m_soft), np.zeros(m_soft)])
    result = linprog(
        c,
        A_ub=np.vstack(a_ub),
        b_ub=np.concatenate(b_ub),
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


def _portfolio_label(penalty: float) -> str:
    return f"turnover_aware_lambda_{str(penalty).replace('.', 'p')}"


def _penalty_from_portfolio(portfolio: str) -> float:
    suffix = str(portfolio).split("turnover_aware_lambda_", 1)[1]
    return float(suffix.replace("p", "."))


def _target_turnover_summary(positions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for portfolio, group in positions.groupby("portfolio", sort=True):
        net = (
            group.groupby(["session_date", "symbol"], as_index=False)["signed_weight"]
            .sum()
            .sort_values(["session_date", "symbol"])
        )
        weights_by_date = {
            pd.Timestamp(session_date): {
                str(row.symbol): float(row.signed_weight)
                for row in sub.itertuples(index=False)
                if abs(float(row.signed_weight)) > 1e-12
            }
            for session_date, sub in net.groupby("session_date", sort=True)
        }
        previous: dict[str, float] = {}
        turnovers: list[float] = []
        positions_count: list[int] = []
        for session_date in sorted(weights_by_date):
            weights = weights_by_date[session_date]
            turnovers.append(_signed_turnover(previous, weights))
            positions_count.append(len(weights))
            previous = weights
        rows.append(
            {
                "portfolio": portfolio,
                "target_turnover_mean": float(np.mean(turnovers)),
                "target_turnover_median": float(np.median(turnovers)),
                "target_positions_mean": float(np.mean(positions_count)),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_turnover_summary(
    *,
    positions_path: str | Path,
    benchmark_returns: pd.DataFrame,
    holding_period_sessions: int,
) -> pd.DataFrame:
    calendar = pd.to_datetime(benchmark_returns["session_date"]).drop_duplicates().sort_values().reset_index(drop=True)
    frame = pd.read_csv(positions_path)
    portfolios = sorted(frame["portfolio"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for portfolio in portfolios:
        turnover_curve = _build_h10_turnover_curve(
            positions_path=positions_path,
            portfolio=portfolio,
            benchmark_calendar=calendar,
            holding_period_sessions=holding_period_sessions,
        )
        warm = (
            turnover_curve.iloc[holding_period_sessions:].copy()
            if len(turnover_curve) > holding_period_sessions
            else turnover_curve.copy()
        )
        rows.append(
            {
                "portfolio": portfolio,
                "aggregate_turnover_mean": float(warm["h10_turnover"].mean()),
                "aggregate_turnover_median": float(warm["h10_turnover"].median()),
                "aggregate_positions_mean": float(warm["h10_positions"].mean()),
                "aggregate_gross_mean": float(warm["h10_gross_exposure"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _strict_metric_summary(strict_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for portfolio, group in strict_curve.groupby("portfolio", sort=True):
        returns = group["gross_return"].astype(float)
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        rolling_60 = equity / equity.shift(60) - 1.0
        benchmark = group["benchmark_oto_return"].astype(float)
        rows.append(
            {
                "portfolio": portfolio,
                "strict_final_equity": float(equity.iloc[-1]),
                "strict_annualized_return": float(equity.iloc[-1] ** (252.0 / len(equity)) - 1.0),
                "strict_annualized_vol": float(returns.std(ddof=1) * np.sqrt(252.0)),
                "strict_sharpe_no_rf": float(
                    returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)
                    if returns.std(ddof=1) > 0
                    else np.nan
                ),
                "strict_max_drawdown": float(drawdown.min()),
                "strict_rolling_60_positive_rate": float((rolling_60 > 0).mean()),
                "strict_corr_to_spy": float(returns.corr(benchmark)),
            }
        )
    return pd.DataFrame(rows)


def _diagnostic_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame(
            columns=[
                "portfolio",
                "mean_abs_net_beta",
                "mean_sic2_max_abs_exposure",
                "mean_long_count",
                "mean_short_count",
            ]
        )
    summary = (
        diagnostics.groupby("portfolio", as_index=False)
        .agg(
            mean_abs_net_beta=("net_beta", lambda s: float(np.abs(s).mean())),
            mean_sic2_max_abs_exposure=("sic2_max_abs_exposure", "mean"),
            mean_long_count=("long_count", "mean"),
            mean_short_count=("short_count", "mean"),
        )
        .sort_values("portfolio")
        .reset_index(drop=True)
    )
    return summary


def _signed_turnover(previous: dict[str, float], new: dict[str, float]) -> float:
    symbols = set(previous) | set(new)
    return float(sum(abs(new.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols))


def _build_comparison_curve(
    *,
    h10_compare_curve: pd.DataFrame,
    strict_curve: pd.DataFrame,
    best_portfolio: str,
) -> pd.DataFrame:
    best = strict_curve[strict_curve["portfolio"].eq(best_portfolio)].copy()
    best = best.rename(
        columns={
            "gross_return": "turnover_aware__return",
            "equity": "turnover_aware__equity",
            "drawdown": "turnover_aware__drawdown",
            "rolling_60_return": "turnover_aware__rolling_60_return",
        }
    )
    out = h10_compare_curve.merge(
        best[["return_date", "turnover_aware__return", "turnover_aware__equity", "turnover_aware__drawdown", "turnover_aware__rolling_60_return"]],
        on="return_date",
        how="inner",
    )
    out["test_window_used"] = False
    return out.sort_values("return_date").reset_index(drop=True)


def _build_metrics(
    *,
    comparison: pd.DataFrame,
    best_row: dict[str, Any],
    lead_aggregate_turnover: float,
) -> pd.DataFrame:
    reference = _reference_metrics_from_comparison(
        comparison=comparison,
        target_portfolio=DEFAULT_TARGET_PORTFOLIO,
        h10_mean_turnover=lead_aggregate_turnover,
        h10_mean_gross=float("nan"),
        h10_mean_positions=float("nan"),
    )
    turnover_aware = pd.DataFrame(
        [
            {
                "series": "turnover aware optimizer",
                "portfolio": str(best_row["portfolio"]),
                "start": str(pd.to_datetime(comparison["return_date"].min()).date()),
                "end": str(pd.to_datetime(comparison["return_date"].max()).date()),
                "final_equity": float(best_row["strict_final_equity"]),
                "annualized_return": float(best_row["strict_annualized_return"]),
                "annualized_vol": float(best_row["strict_annualized_vol"]),
                "sharpe_no_rf": float(best_row["strict_sharpe_no_rf"]),
                "max_drawdown": float(best_row["strict_max_drawdown"]),
                "rolling_60_positive_rate": float(best_row["strict_rolling_60_positive_rate"]),
                "corr_to_spy": float(best_row["strict_corr_to_spy"]),
                "mean_turnover": float(best_row["aggregate_turnover_mean"]),
                "mean_gross_exposure": float(best_row["aggregate_gross_mean"]),
                "mean_positions": float(best_row["aggregate_positions_mean"]),
                "target_turnover_mean": float(best_row["target_turnover_mean"]),
                "turnover_budget_mean": float(lead_aggregate_turnover),
                "turnover_vs_h10_ratio": float(best_row["aggregate_turnover_vs_lead"]),
                "aggregate_turnover_vs_lead": float(best_row["aggregate_turnover_vs_lead"]),
                "mean_daily_return_bps": float(
                    comparison["turnover_aware__return"].astype(float).mean() * 10000.0
                ),
                "mean_abs_net_beta": float(best_row["mean_abs_net_beta"]),
                "test_window_used": False,
                "penalty": float(best_row["penalty"]),
            }
        ]
    )
    return pd.concat([reference, turnover_aware], ignore_index=True, sort=False)


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
        ("baseline", "sic2_soft_neutral__equity", "sic2_soft_neutral__drawdown", "sic2_soft_neutral__rolling_60_return"),
        ("short_hybrid", "sic2_soft_neutral__short_hybrid_soft_fw_overlay__equity", "sic2_soft_neutral__short_hybrid_soft_fw_overlay__drawdown", "sic2_soft_neutral__short_hybrid_soft_fw_overlay__rolling_60_return"),
        ("turnover_aware", "turnover_aware__equity", "turnover_aware__drawdown", "turnover_aware__rolling_60_return"),
    ]
    for label, equity_col, drawdown_col, rolling_col in series_specs:
        color = PLOT_COLORS[label]
        ax1.plot(dates, curve[equity_col], label=label, color=color, linewidth=2.0)
        ax2.plot(dates, curve[drawdown_col] * 100.0, color=color, linewidth=1.8)
        ax3.plot(dates, curve[rolling_col] * 100.0, color=color, linewidth=1.8)
    ax1.plot(dates, curve["benchmark_equity"], label="spy_raw", color=PLOT_COLORS["spy_raw"], linestyle="--", linewidth=1.8)
    ax2.plot(dates, curve["benchmark_drawdown"] * 100.0, color=PLOT_COLORS["spy_raw"], linestyle="--", linewidth=1.6)
    ax3.plot(dates, curve["benchmark_rolling_60_return"] * 100.0, color=PLOT_COLORS["spy_raw"], linestyle="--", linewidth=1.6)
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax1.set_ylabel("Growth of $1")
    ax2.set_ylabel("Drawdown %")
    ax3.set_ylabel("Rolling 60-session %")
    ax3.set_xlabel("Date")
    ax1.set_title("Phase5E Turnover-Aware Daily Optimizer")
    for ax in axes:
        ax.grid(alpha=0.2)
    ax1.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _memo(
    *,
    metrics: pd.DataFrame,
    sweep: pd.DataFrame,
    best_portfolio: str,
    plot_path: Path,
    lead_aggregate_turnover: float,
) -> str:
    display = metrics.copy()
    for column in ("annualized_return", "annualized_vol", "max_drawdown", "rolling_60_positive_rate"):
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct_like)
    for column in ("corr_to_spy", "sharpe_no_rf", "mean_turnover", "target_turnover_mean", "aggregate_turnover_vs_lead", "mean_daily_return_bps", "mean_abs_net_beta", "penalty"):
        if column in display.columns:
            display[column] = display[column].map(_fmt_float_like)

    shortlist = sweep[
        [
            "portfolio",
            "penalty",
            "target_turnover_mean",
            "aggregate_turnover_mean",
            "aggregate_turnover_vs_lead",
            "strict_annualized_return",
            "strict_annualized_vol",
            "strict_sharpe_no_rf",
            "strict_max_drawdown",
        ]
    ].copy().sort_values("penalty")
    for column in ("strict_annualized_return", "strict_annualized_vol", "strict_max_drawdown"):
        shortlist[column] = shortlist[column].map(_fmt_pct_like)
    for column in ("penalty", "target_turnover_mean", "aggregate_turnover_mean", "aggregate_turnover_vs_lead", "strict_sharpe_no_rf"):
        shortlist[column] = shortlist[column].map(_fmt_float_like)

    return "\n".join(
        [
            "# Phase5E Turnover-Aware Daily Optimizer",
            "",
            f"Generated: {_utc_now()}",
            "",
            f"Lead aggregate turnover baseline = {lead_aggregate_turnover:.6f}",
            "",
            f"Best turnover-aware portfolio: {best_portfolio}",
            "",
            "## Metrics",
            "",
            _text_table(display),
            "",
            "## Sweep",
            "",
            _text_table(shortlist),
            "",
            "## Plot",
            "",
            f"![Phase5E comparison]({plot_path.as_posix()})",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase5E turnover-aware daily optimizer.")
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--strict-root", default=str(DEFAULT_STRICT_ROOT))
    parser.add_argument("--phase3-rollup-path", default=str(DEFAULT_PHASE3_H10_ROLLUP))
    parser.add_argument("--h10-compare-curve-path", default=str(DEFAULT_H10_COMPARE_CURVE_PATH))
    args = parser.parse_args(argv)

    build_phase5e_turnover_aware_optimizer_artifacts(
        signal_panel_path=args.signal_panel_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        output_root=args.output_root,
        strict_root=args.strict_root,
        phase3_rollup_path=args.phase3_rollup_path,
        h10_compare_curve_path=args.h10_compare_curve_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
