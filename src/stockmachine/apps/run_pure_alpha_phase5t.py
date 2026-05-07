from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from stockmachine.apps.run_pure_alpha_phase4aa import _add_former_winner_scores
from stockmachine.apps.run_pure_alpha_phase4ab import _add_hybrid_scores
from stockmachine.apps.run_pure_alpha_phase4af import _add_falling_knife_overlay_scores
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
from stockmachine.apps.run_pure_alpha_phase5b import _fmt_pct_like, _text_table, _utc_now
from stockmachine.apps.run_pure_alpha_phase5e import (
    DEFAULT_HOLDING_PERIOD_SESSIONS,
    _aggregate_turnover_summary,
    _candidate_union,
    _diagnostic_summary,
    _group_exposure_matrix,
    _strict_metric_summary,
    _target_turnover_summary,
)
from stockmachine.apps.run_pure_alpha_phase5f import DEFAULT_SHORT_OVERLAY
from stockmachine.apps.run_pure_alpha_phase5r import (
    DEFAULT_TURNOVER_PENALTY,
    _add_price_only_falling_knife_scores,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5t_price_only_long_risk_budget_20260507"
DEFAULT_STRICT_ROOT = DEFAULT_OUTPUT_ROOT / "strict_h10_rebuild"
DEFAULT_RISK_CAP_PENALTY = 50.0
CURRENT_SOTA_SERIES = "current sota"

WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("2015_peak_to_trough", "2015-06-04", "2015-08-21"),
    ("2016_jan_feb", "2016-01-05", "2016-02-09"),
    ("2017_spring_short_fail", "2017-03-20", "2017-05-31"),
    ("2018_aug_sep_short_fail", "2018-08-08", "2018-09-12"),
    ("2019_may", "2019-05-01", "2019-06-05"),
    ("2019_aug", "2019-08-01", "2019-08-31"),
)

RISK_CAPS_BY_PORTFOLIO: dict[str, tuple[dict[str, Any], ...]] = {
    "current_sota": (),
    "fk90_cap50": (
        {"name": "fk90", "column": "price_falling_knife_score", "threshold": 0.90, "cap": 0.50},
    ),
    "fk90_cap35": (
        {"name": "fk90", "column": "price_falling_knife_score", "threshold": 0.90, "cap": 0.35},
    ),
    "stillfall90_cap35": (
        {"name": "stillfall90", "column": "price_still_falling_score", "threshold": 0.90, "cap": 0.35},
    ),
    "beta90_cap35": (
        {"name": "beta90", "column": "long_beta_rank_pct", "threshold": 0.90, "cap": 0.35},
    ),
    "fk90_beta90_cap35": (
        {"name": "fk90", "column": "price_falling_knife_score", "threshold": 0.90, "cap": 0.35},
        {"name": "beta90", "column": "long_beta_rank_pct", "threshold": 0.90, "cap": 0.35},
    ),
}

COMBO_SPECS: tuple[dict[str, Any], ...] = (
    {
        "portfolio": "current_sota",
        "series": CURRENT_SOTA_SERIES,
        "long_score": DEFAULT_LONG_SCORE,
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "fk90_cap50",
        "series": "long fk90 cap50",
        "long_score": DEFAULT_LONG_SCORE,
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "fk90_cap35",
        "series": "long fk90 cap35",
        "long_score": DEFAULT_LONG_SCORE,
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "stillfall90_cap35",
        "series": "long stillfall90 cap35",
        "long_score": DEFAULT_LONG_SCORE,
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "beta90_cap35",
        "series": "long beta90 cap35",
        "long_score": DEFAULT_LONG_SCORE,
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "fk90_beta90_cap35",
        "series": "long fk90+beta90 cap35",
        "long_score": DEFAULT_LONG_SCORE,
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
)

PLOT_COLORS = {
    CURRENT_SOTA_SERIES: "#2563eb",
    "long fk90 cap50": "#0891b2",
    "long fk90 cap35": "#155e75",
    "long stillfall90 cap35": "#c2410c",
    "long beta90 cap35": "#7c3aed",
    "long fk90+beta90 cap35": "#be123c",
    "SPY raw": "#6b7280",
}


def build_phase5t_price_only_long_risk_budget_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    strict_root: str | Path = DEFAULT_STRICT_ROOT,
    phase3_rollup_path: str | Path = DEFAULT_PHASE3_H10_ROLLUP,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    candidate_pool_per_side: int = 80,
    max_single_name_side_weight: float = 1.0 / 30.0,
    min_nonzero_names: int = 20,
    score_weight: float = 0.01,
    soft_group_penalty: float = 25.0,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
    risk_cap_penalty: float = DEFAULT_RISK_CAP_PENALTY,
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
    panel = _add_falling_knife_overlay_scores(panel)
    panel = panel.merge(industry_map, on="symbol", how="left")
    panel["sic2_sector"] = panel["sic2_sector"].fillna("UNKNOWN")
    panel["sic4_industry"] = panel["sic4_industry"].fillna("UNKNOWN")
    panel["sic_description"] = panel["sic_description"].fillna("UNKNOWN")
    panel = _add_price_only_falling_knife_scores(panel, long_variant=long_variant)
    panel = _add_long_beta_rank(panel)
    panel = _add_residual_targets(
        panel,
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )

    positions, diagnostics, skipped = _construct_combo_positions(
        panel,
        long_variant=long_variant,
        short_variant=short_variant,
        candidate_pool_per_side=candidate_pool_per_side,
        max_single_name_side_weight=max_single_name_side_weight,
        min_nonzero_names=min_nonzero_names,
        score_weight=score_weight,
        soft_group_penalty=soft_group_penalty,
        turnover_penalty=turnover_penalty,
        risk_cap_penalty=risk_cap_penalty,
    )

    positions_path = output_dir / "phase5t_positions_validation.csv.gz"
    diagnostics_path = output_dir / "phase5t_daily_validation.csv"
    skipped_path = output_dir / "phase5t_skipped_validation.csv"
    positions.to_csv(positions_path, index=False, compression="gzip")
    diagnostics.to_csv(diagnostics_path, index=False)
    skipped.to_csv(skipped_path, index=False)

    strict_rollup = build_phase4z_strict_horizon_daily_paths(
        positions_path=positions_path,
        output_root=strict_root,
        phase3_rollup_path=phase3_rollup_path,
        existing_curve_path=None,
        holding_period_sessions=holding_period_sessions,
        lead_portfolio="current_sota",
    )
    strict_curve_path = Path(strict_root) / "phase4z_strict_h10_daily_curve.csv"
    strict_curve = pd.read_csv(strict_curve_path, parse_dates=["return_date"])
    benchmark_calendar = strict_curve["return_date"].drop_duplicates().sort_values().to_frame(
        name="session_date"
    )
    target_turnover = _target_turnover_summary(positions)
    aggregate_turnover = _aggregate_turnover_summary(
        positions_path=positions_path,
        benchmark_returns=benchmark_calendar,
        holding_period_sessions=holding_period_sessions,
    )
    strict_metrics = _strict_metric_summary(strict_curve)
    diagnostic_summary = _diagnostic_summary(diagnostics)
    risk_exposure = _risk_exposure_summary(positions=positions, panel=panel)
    diagnostic_extra = _diagnostic_extra_summary(diagnostics)
    sweep = (
        strict_metrics.merge(target_turnover, on="portfolio", how="left")
        .merge(aggregate_turnover, on="portfolio", how="left")
        .merge(diagnostic_summary, on="portfolio", how="left")
        .merge(diagnostic_extra, on="portfolio", how="left")
        .merge(risk_exposure, on="portfolio", how="left")
    )
    sweep["series"] = sweep["portfolio"].map(_portfolio_to_series)
    sweep["long_score"] = sweep["portfolio"].map(_portfolio_to_long_score)
    sweep["short_selector"] = sweep["portfolio"].map(_portfolio_to_short_selector)
    sweep["risk_caps"] = sweep["portfolio"].map(_portfolio_to_risk_caps_label)
    sweep["turnover_penalty"] = float(turnover_penalty)
    sweep["risk_cap_penalty"] = float(risk_cap_penalty)

    metrics = _build_metrics(sweep=sweep, strict_curve=strict_curve)
    windows = _window_return_summary(strict_curve)
    comparison_curve = _build_comparison_curve(strict_curve)

    sweep_path = output_dir / "phase5t_sweep.csv"
    metrics_path = output_dir / "phase5t_metrics.csv"
    windows_path = output_dir / "phase5t_window_returns.csv"
    risk_exposure_path = output_dir / "phase5t_risk_exposure_summary.csv"
    comparison_curve_path = output_dir / "phase5t_comparison_curve.csv"
    plot_path = output_dir / "phase5t_comparison_plot.png"
    memo_path = output_dir / "phase5t_price_only_long_risk_budget_memo.md"
    rollup_path = output_dir / "phase5t_rollup.json"

    sweep.to_csv(sweep_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    windows.to_csv(windows_path, index=False)
    risk_exposure.to_csv(risk_exposure_path, index=False)
    comparison_curve.to_csv(comparison_curve_path, index=False)
    _plot_comparison(comparison_curve, output_path=plot_path)
    memo_path.write_text(
        _memo(
            metrics=metrics,
            windows=windows,
            risk_exposure=risk_exposure,
            skipped=skipped,
            plot_path=plot_path,
            risk_cap_penalty=risk_cap_penalty,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "cik_mapping_path": Path(cik_mapping_path).as_posix(),
        "sec_submissions_dir": Path(sec_submissions_dir).as_posix(),
        "long_variant": long_variant,
        "short_variant": short_variant,
        "turnover_penalty": float(turnover_penalty),
        "risk_cap_penalty": float(risk_cap_penalty),
        "strict_rollup": strict_rollup,
        "risk_caps_by_portfolio": RISK_CAPS_BY_PORTFOLIO,
        "artifacts": {
            "positions": positions_path.as_posix(),
            "diagnostics": diagnostics_path.as_posix(),
            "skipped": skipped_path.as_posix(),
            "sweep": sweep_path.as_posix(),
            "metrics": metrics_path.as_posix(),
            "window_returns": windows_path.as_posix(),
            "risk_exposure_summary": risk_exposure_path.as_posix(),
            "comparison_curve": comparison_curve_path.as_posix(),
            "plot": plot_path.as_posix(),
            "memo": memo_path.as_posix(),
            "strict_curve": strict_curve_path.as_posix(),
        },
        "method": "price_only_long_uncertainty_risk_budget_overlay",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "No non-price factor panels are loaded or used.",
            "SEC-derived SIC labels are used only as neutralization and sector-relative price-state grouping keys.",
            "Risk budgets are soft constraints with explicit slack penalties, so infeasible days can still be solved.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _add_long_beta_rank(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    frame["long_beta_rank_pct"] = (
        frame.groupby(["variant", "session_date"], sort=False)["beta"]
        .rank(method="average", pct=True)
        .fillna(0.0)
    )
    return frame


def _construct_combo_positions(
    panel: pd.DataFrame,
    *,
    long_variant: str,
    short_variant: str,
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    min_nonzero_names: int,
    score_weight: float,
    soft_group_penalty: float,
    turnover_penalty: float,
    risk_cap_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    groups = {
        (variant, session_date): group
        for (variant, session_date), group in panel.groupby(["variant", "session_date"], sort=True)
    }
    sessions = sorted(panel["session_date"].unique())
    previous_by_portfolio: dict[str, dict[str, dict[str, float]]] = {
        str(spec["portfolio"]): {"long": {}, "short": {}} for spec in COMBO_SPECS
    }
    for spec in COMBO_SPECS:
        portfolio = str(spec["portfolio"])
        previous_side = previous_by_portfolio[portfolio]
        for session_date in sessions:
            long_group = groups.get((long_variant, session_date))
            short_group = groups.get((short_variant, session_date))
            if long_group is None or short_group is None:
                skipped.append(_skip_row(session_date, portfolio, 0, 0, "missing_group"))
                continue
            book, diagnostic, skip, updated = _construct_one_session_with_risk_caps(
                long_group,
                short_group,
                session_date=session_date,
                portfolio=portfolio,
                long_variant=long_variant,
                short_variant=short_variant,
                long_score=str(spec["long_score"]),
                short_selector=str(spec["short_selector"]),
                risk_caps=RISK_CAPS_BY_PORTFOLIO.get(portfolio, ()),
                soft_group="sic2_sector",
                candidate_pool_per_side=candidate_pool_per_side,
                max_single_name_side_weight=max_single_name_side_weight,
                min_nonzero_names=min_nonzero_names,
                score_weight=score_weight,
                soft_group_penalty=soft_group_penalty,
                turnover_penalty=turnover_penalty,
                risk_cap_penalty=risk_cap_penalty,
                previous_weights=previous_side,
            )
            positions.extend(book)
            if diagnostic is not None:
                diagnostic["series"] = str(spec["series"])
                diagnostics.append(diagnostic)
            if skip is not None:
                skip["series"] = str(spec["series"])
                skipped.append(skip)
            previous_side = updated
        previous_by_portfolio[portfolio] = previous_side
    return pd.DataFrame(positions), pd.DataFrame(diagnostics), pd.DataFrame(skipped)


def _construct_one_session_with_risk_caps(
    long_group: pd.DataFrame,
    short_group: pd.DataFrame,
    *,
    session_date: str,
    portfolio: str,
    long_variant: str,
    short_variant: str,
    long_score: str,
    short_selector: str,
    risk_caps: tuple[dict[str, Any], ...],
    soft_group: str | None,
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    min_nonzero_names: int,
    score_weight: float,
    soft_group_penalty: float,
    turnover_penalty: float,
    risk_cap_penalty: float,
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
        "sic_description",
    ]
    needed_long = [*needed, *(str(cap["column"]) for cap in risk_caps)]
    long_base = long_group.dropna(subset=needed_long).drop_duplicates("symbol").copy()
    short_base = short_group.dropna(subset=needed).drop_duplicates("symbol").copy()
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
        long_weights, short_weights, cap_slacks = _optimize_turnover_aware_joint_weights_with_caps(
            longs,
            shorts,
            long_score=long_score,
            short_selector=short_selector,
            soft_group=soft_group,
            risk_caps=risk_caps,
            max_weight=max_single_name_side_weight,
            score_weight=score_weight,
            soft_group_penalty=soft_group_penalty,
            turnover_penalty=turnover_penalty,
            risk_cap_penalty=risk_cap_penalty,
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
    diagnostic["risk_cap_penalty"] = float(risk_cap_penalty)
    diagnostic["target_turnover"] = float(np.abs(long_weights - prev_long_vec).sum() + np.abs(short_weights - prev_short_vec).sum())
    diagnostic["target_new_long_names"] = int(((long_weights > 1e-10) & (prev_long_vec <= 1e-10)).sum())
    diagnostic["target_new_short_names"] = int(((short_weights > 1e-10) & (prev_short_vec <= 1e-10)).sum())
    for cap, slack in zip(risk_caps, cap_slacks):
        name = str(cap["name"])
        mask = longs[str(cap["column"])].astype(float).ge(float(cap["threshold"])).to_numpy(dtype=float)
        exposure = float(np.dot(long_weights, mask))
        diagnostic[f"long_risk_{name}_exposure"] = exposure
        diagnostic[f"long_risk_{name}_cap"] = float(cap["cap"])
        diagnostic[f"long_risk_{name}_slack"] = float(slack)
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


def _optimize_turnover_aware_joint_weights_with_caps(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    *,
    long_score: str,
    short_selector: str,
    soft_group: str | None,
    risk_caps: tuple[dict[str, Any], ...],
    max_weight: float,
    score_weight: float,
    soft_group_penalty: float,
    turnover_penalty: float,
    risk_cap_penalty: float,
    previous_long_weights: dict[str, float],
    previous_short_weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    a_eq_x = np.vstack(aeq)
    b_eq = np.array(beq, dtype=float)

    soft_matrix = (
        _group_exposure_matrix(longs, shorts, soft_group)
        if soft_group is not None
        else np.zeros((0, n), dtype=float)
    )
    m_soft = soft_matrix.shape[0]
    m_risk = len(risk_caps)
    c = np.concatenate(
        [
            -score_weight * score,
            np.full(n, turnover_penalty, dtype=float),
            np.full(m_soft, soft_group_penalty, dtype=float),
            np.full(m_risk, risk_cap_penalty, dtype=float),
        ]
    )
    bounds: list[tuple[float, float | None]] = (
        [(0.0, max_weight)] * n
        + [(0.0, None)] * n
        + [(0.0, None)] * m_soft
        + [(0.0, None)] * m_risk
    )
    total_extra = n + m_soft + m_risk
    a_eq = np.column_stack([a_eq_x, np.zeros((a_eq_x.shape[0], total_extra))])

    zeros_soft_risk = np.zeros((n, m_soft + m_risk))
    turnover_top = np.column_stack([np.eye(n), -np.eye(n), zeros_soft_risk])
    turnover_bottom = np.column_stack([-np.eye(n), -np.eye(n), zeros_soft_risk])
    a_ub = [turnover_top, turnover_bottom]
    b_ub = [prev, -prev]
    if m_soft:
        soft_top = np.column_stack(
            [soft_matrix, np.zeros((m_soft, n)), -np.eye(m_soft), np.zeros((m_soft, m_risk))]
        )
        soft_bottom = np.column_stack(
            [-soft_matrix, np.zeros((m_soft, n)), -np.eye(m_soft), np.zeros((m_soft, m_risk))]
        )
        a_ub.extend([soft_top, soft_bottom])
        b_ub.extend([np.zeros(m_soft), np.zeros(m_soft)])
    if m_risk:
        risk_rows = []
        risk_rhs = []
        for idx, cap in enumerate(risk_caps):
            mask = np.zeros(n, dtype=float)
            mask[:n_long] = longs[str(cap["column"])].astype(float).ge(float(cap["threshold"])).to_numpy(dtype=float)
            row = np.concatenate(
                [
                    mask,
                    np.zeros(n, dtype=float),
                    np.zeros(m_soft, dtype=float),
                    -np.eye(m_risk)[idx],
                ]
            )
            risk_rows.append(row)
            risk_rhs.append(float(cap["cap"]))
        a_ub.append(np.vstack(risk_rows))
        b_ub.append(np.array(risk_rhs, dtype=float))

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
    risk_start = n + n + m_soft
    cap_slacks = np.array(result.x[risk_start : risk_start + m_risk], dtype=float) if m_risk else np.zeros(0)
    if abs(long_weights.sum() - 1.0) > 1e-6 or abs(short_weights.sum() - 1.0) > 1e-6:
        raise ValueError("side_sum_constraint_breach")
    net_beta = float(np.dot(long_weights, longs["beta"]) - np.dot(short_weights, shorts["beta"]))
    if abs(net_beta) > 1e-5:
        raise ValueError("beta_constraint_breach")
    return long_weights, short_weights, cap_slacks


def _diagnostic_extra_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame(columns=["portfolio"])
    agg: dict[str, Any] = {
        "mean_long_style_factor_sic2_residual": ("long_style_factor_sic2_residual", "mean"),
        "mean_spread_style_factor_sic2_residual": ("spread_style_factor_sic2_residual", "mean"),
        "mean_target_turnover": ("target_turnover", "mean"),
    }
    for column in diagnostics.columns:
        if column.startswith("long_risk_") and (column.endswith("_exposure") or column.endswith("_slack")):
            agg[f"mean_{column}"] = (column, "mean")
    return diagnostics.groupby("portfolio", as_index=False).agg(**agg)


def _risk_exposure_summary(*, positions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    features = [
        "price_falling_knife_score",
        "price_still_falling_score",
        "price_stabilization_score",
        "long_beta_rank_pct",
    ]
    factors = panel[["session_date", "symbol", *features]].drop_duplicates(["session_date", "symbol"])
    longs = positions[positions["side"].eq("long")].merge(factors, on=["session_date", "symbol"], how="left")
    rows: list[dict[str, Any]] = []
    for (portfolio, session_date), group in longs.groupby(["portfolio", "session_date"], sort=True):
        weights = group["side_weight"].astype(float)
        denom = float(weights.sum())
        if denom <= 0:
            continue
        rows.append(
            {
                "portfolio": portfolio,
                "session_date": session_date,
                "weighted_price_falling_knife_score": _weighted_mean(group, weights, "price_falling_knife_score"),
                "weighted_price_still_falling_score": _weighted_mean(group, weights, "price_still_falling_score"),
                "weighted_price_stabilization_score": _weighted_mean(group, weights, "price_stabilization_score"),
                "weighted_long_beta_rank_pct": _weighted_mean(group, weights, "long_beta_rank_pct"),
                "long_fk90_weight": float(weights[group["price_falling_knife_score"].ge(0.90)].sum()),
                "long_stillfall90_weight": float(weights[group["price_still_falling_score"].ge(0.90)].sum()),
                "long_beta90_weight": float(weights[group["long_beta_rank_pct"].ge(0.90)].sum()),
                "test_window_used": False,
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return pd.DataFrame(columns=["portfolio"])
    return (
        daily.groupby("portfolio", as_index=False)
        .agg(
            mean_weighted_price_falling_knife_score=("weighted_price_falling_knife_score", "mean"),
            mean_weighted_price_still_falling_score=("weighted_price_still_falling_score", "mean"),
            mean_weighted_price_stabilization_score=("weighted_price_stabilization_score", "mean"),
            mean_weighted_long_beta_rank_pct=("weighted_long_beta_rank_pct", "mean"),
            mean_long_fk90_weight=("long_fk90_weight", "mean"),
            mean_long_stillfall90_weight=("long_stillfall90_weight", "mean"),
            mean_long_beta90_weight=("long_beta90_weight", "mean"),
        )
        .sort_values("portfolio")
        .reset_index(drop=True)
    )


def _weighted_mean(group: pd.DataFrame, weights: pd.Series, column: str) -> float:
    values = pd.to_numeric(group[column], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return np.nan
    w = weights[valid].astype(float)
    denom = float(w.sum())
    if denom <= 0:
        return np.nan
    return float(np.dot(w, values[valid].astype(float)) / denom)


def _build_metrics(*, sweep: pd.DataFrame, strict_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in COMBO_SPECS:
        portfolio = str(spec["portfolio"])
        row = sweep[sweep["portfolio"].eq(portfolio)]
        if row.empty:
            continue
        item = row.iloc[0]
        curve_subset = strict_curve[strict_curve["portfolio"].eq(portfolio)]
        rows.append(
            {
                "series": str(spec["series"]),
                "portfolio": portfolio,
                "risk_caps": _portfolio_to_risk_caps_label(portfolio),
                "annualized_return": float(item["strict_annualized_return"]),
                "annualized_vol": float(item["strict_annualized_vol"]),
                "sharpe_no_rf": float(item["strict_sharpe_no_rf"]),
                "max_drawdown": float(item["strict_max_drawdown"]),
                "rolling_60_positive_rate": float(item["strict_rolling_60_positive_rate"]),
                "corr_to_spy": float(item["strict_corr_to_spy"]),
                "final_equity": float(item["strict_final_equity"]),
                "mean_daily_return_bps": float(curve_subset["gross_return"].mean() * 10000.0),
                "target_turnover_mean": float(item["target_turnover_mean"]),
                "aggregate_turnover_mean": float(item["aggregate_turnover_mean"]),
                "aggregate_positions_mean": float(item["aggregate_positions_mean"]),
                "mean_abs_net_beta": float(item["mean_abs_net_beta"]),
                "mean_long_style_factor_sic2_residual": float(
                    item.get("mean_long_style_factor_sic2_residual", np.nan)
                ),
                "mean_spread_style_factor_sic2_residual": float(
                    item.get("mean_spread_style_factor_sic2_residual", np.nan)
                ),
                "mean_long_fk90_weight": float(item.get("mean_long_fk90_weight", np.nan)),
                "mean_long_stillfall90_weight": float(item.get("mean_long_stillfall90_weight", np.nan)),
                "mean_long_beta90_weight": float(item.get("mean_long_beta90_weight", np.nan)),
                "test_window_used": False,
            }
        )
    rows.append(_spy_metric_row(strict_curve))
    order = {str(spec["series"]): i for i, spec in enumerate(COMBO_SPECS)}
    order["SPY raw"] = len(order)
    frame = pd.DataFrame(rows)
    frame["sort_key"] = frame["series"].map(order)
    return frame.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)


def _spy_metric_row(strict_curve: pd.DataFrame) -> dict[str, Any]:
    benchmark = strict_curve.sort_values(["return_date", "portfolio"]).drop_duplicates("return_date")
    spy_returns = benchmark["benchmark_oto_return"].astype(float)
    spy_equity = (1.0 + spy_returns).cumprod()
    spy_drawdown = spy_equity / spy_equity.cummax() - 1.0
    spy_rolling_60 = spy_equity / spy_equity.shift(60) - 1.0
    return {
        "series": "SPY raw",
        "portfolio": "SPY",
        "risk_caps": "",
        "annualized_return": float(spy_equity.iloc[-1] ** (252.0 / len(spy_equity)) - 1.0),
        "annualized_vol": float(spy_returns.std(ddof=1) * np.sqrt(252.0)),
        "sharpe_no_rf": float(
            spy_returns.mean() / spy_returns.std(ddof=1) * np.sqrt(252.0)
            if spy_returns.std(ddof=1) > 0
            else np.nan
        ),
        "max_drawdown": float(spy_drawdown.min()),
        "rolling_60_positive_rate": float((spy_rolling_60 > 0).mean()),
        "corr_to_spy": 1.0,
        "final_equity": float(spy_equity.iloc[-1]),
        "mean_daily_return_bps": float(spy_returns.mean() * 10000.0),
        "target_turnover_mean": np.nan,
        "aggregate_turnover_mean": np.nan,
        "aggregate_positions_mean": np.nan,
        "mean_abs_net_beta": np.nan,
        "mean_long_style_factor_sic2_residual": np.nan,
        "mean_spread_style_factor_sic2_residual": np.nan,
        "mean_long_fk90_weight": np.nan,
        "mean_long_stillfall90_weight": np.nan,
        "mean_long_beta90_weight": np.nan,
        "test_window_used": False,
    }


def _window_return_summary(strict_curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for portfolio, group in strict_curve.groupby("portfolio", sort=True):
        frame = group.sort_values("return_date").copy()
        for window, start, end in WINDOWS:
            subset = frame[
                (frame["return_date"] >= pd.Timestamp(start))
                & (frame["return_date"] <= pd.Timestamp(end))
            ]
            if subset.empty:
                continue
            rows.append(
                {
                    "portfolio": portfolio,
                    "series": _portfolio_to_series(portfolio),
                    "window": window,
                    "start": start,
                    "end": end,
                    "compound_return": float((1.0 + subset["gross_return"].astype(float)).prod() - 1.0),
                    "hit_rate_daily": float((subset["gross_return"].astype(float) > 0).mean()),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["window", "portfolio"]).reset_index(drop=True)


def _build_comparison_curve(strict_curve: pd.DataFrame) -> pd.DataFrame:
    benchmark = (
        strict_curve.sort_values(["return_date", "portfolio"])
        .drop_duplicates("return_date")[
            ["return_date", "benchmark_oto_return", "benchmark_equity", "benchmark_drawdown"]
        ]
        .copy()
    )
    benchmark["benchmark_rolling_60_return"] = (
        benchmark["benchmark_equity"].astype(float)
        / benchmark["benchmark_equity"].astype(float).shift(60)
        - 1.0
    )
    out = benchmark
    for spec in COMBO_SPECS:
        portfolio = str(spec["portfolio"])
        series = _series_key(str(spec["series"]))
        subset = strict_curve[strict_curve["portfolio"].eq(portfolio)].copy()
        subset = subset.rename(
            columns={
                "gross_return": f"{series}__return",
                "equity": f"{series}__equity",
                "drawdown": f"{series}__drawdown",
                "rolling_60_return": f"{series}__rolling_60_return",
            }
        )
        out = out.merge(
            subset[
                [
                    "return_date",
                    f"{series}__return",
                    f"{series}__equity",
                    f"{series}__drawdown",
                    f"{series}__rolling_60_return",
                ]
            ],
            on="return_date",
            how="left",
        )
    out["test_window_used"] = False
    return out.sort_values("return_date").reset_index(drop=True)


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
    for spec in COMBO_SPECS:
        label = str(spec["series"])
        key = _series_key(label)
        color = PLOT_COLORS[label]
        ax1.plot(dates, curve[f"{key}__equity"], label=label, color=color, linewidth=1.6)
        ax2.plot(dates, curve[f"{key}__drawdown"] * 100.0, color=color, linewidth=1.3)
        ax3.plot(dates, curve[f"{key}__rolling_60_return"] * 100.0, color=color, linewidth=1.3)
    ax1.plot(
        dates,
        curve["benchmark_equity"],
        label="SPY raw",
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.4,
    )
    ax2.plot(
        dates,
        curve["benchmark_drawdown"] * 100.0,
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.2,
    )
    ax3.plot(
        dates,
        curve["benchmark_rolling_60_return"] * 100.0,
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.2,
    )
    ax1.set_title("Phase5T Price-Only Long Risk-Budget Overlay")
    ax1.set_ylabel("Growth of $1")
    ax2.set_ylabel("Drawdown %")
    ax3.set_ylabel("Rolling 60-session %")
    ax3.set_xlabel("Date")
    for ax in axes:
        ax.axhline(0.0 if ax is not ax1 else 1.0, color="#374151", linestyle="--", linewidth=0.9)
        ax.grid(True, alpha=0.22)
    ax1.legend(loc="best")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _memo(
    *,
    metrics: pd.DataFrame,
    windows: pd.DataFrame,
    risk_exposure: pd.DataFrame,
    skipped: pd.DataFrame,
    plot_path: Path,
    risk_cap_penalty: float,
) -> str:
    metric_cols = [
        "series",
        "portfolio",
        "risk_caps",
        "annualized_return",
        "annualized_vol",
        "sharpe_no_rf",
        "max_drawdown",
        "rolling_60_positive_rate",
        "aggregate_turnover_mean",
        "mean_long_fk90_weight",
        "mean_long_stillfall90_weight",
        "mean_long_beta90_weight",
    ]
    window_cols = ["series", "window", "compound_return", "hit_rate_daily"]
    risk_cols = [
        "portfolio",
        "mean_weighted_price_falling_knife_score",
        "mean_weighted_price_still_falling_score",
        "mean_weighted_price_stabilization_score",
        "mean_weighted_long_beta_rank_pct",
        "mean_long_fk90_weight",
        "mean_long_stillfall90_weight",
        "mean_long_beta90_weight",
    ]
    skipped_summary = (
        skipped.groupby(["portfolio", "reason"], as_index=False).size().rename(columns={"size": "rows"})
        if not skipped.empty
        else pd.DataFrame(columns=["portfolio", "reason", "rows"])
    )
    lines = [
        "# Phase5T Price-Only Long Risk-Budget Overlay",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        f"- Short side fixed to `{DEFAULT_SHORT_OVERLAY}`.",
        f"- Long selector fixed to `{DEFAULT_LONG_SCORE}`; this is an optimizer-level risk-budget overlay.",
        f"- Risk cap slack penalty: `{risk_cap_penalty:.2f}`.",
        "- No non-price factor panels are loaded or used.",
        "- Validation window only; no test-window performance is computed.",
        "",
        "## Metrics",
        "",
        _text_table(_format_table(metrics[metric_cols] if not metrics.empty else metrics)),
        "",
        "## Stress Windows",
        "",
        _text_table(_format_table(windows[window_cols] if not windows.empty else windows)),
        "",
        "## Long Risk Exposure",
        "",
        _text_table(_format_table(risk_exposure[risk_cols] if not risk_exposure.empty else risk_exposure)),
        "",
        "## Skipped Rows",
        "",
        _text_table(skipped_summary) if not skipped_summary.empty else "No skipped rows.",
        "",
        "## Plot",
        "",
        f"![Phase5T comparison]({plot_path.as_posix()})",
        "",
        "## Read",
        "",
        "- This tests whether high price weakness is better treated as uncertainty/risk budget than as a direct negative alpha signal.",
        "- If drawdown improves while annualized return is mostly preserved, risk budgeting is a better direction than score veto.",
        "- If both return and drawdown worsen, price-only still lacks the state variable needed for long-leg repair.",
    ]
    return "\n".join(lines)


def _format_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if column in {
            "annualized_return",
            "annualized_vol",
            "max_drawdown",
            "rolling_60_positive_rate",
            "corr_to_spy",
            "compound_return",
            "hit_rate_daily",
        }:
            out[column] = out[column].map(_fmt_pct_like)
        elif column.startswith("mean_long_") or column.startswith("mean_weighted_"):
            out[column] = out[column].map(lambda value: _fmt_number(value, digits=3))
        elif pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: _fmt_number(value, digits=3))
    return out


def _fmt_number(value: Any, *, digits: int) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _portfolio_to_series(portfolio: str) -> str:
    mapping = {str(spec["portfolio"]): str(spec["series"]) for spec in COMBO_SPECS}
    return mapping.get(str(portfolio), str(portfolio))


def _portfolio_to_long_score(portfolio: str) -> str:
    mapping = {str(spec["portfolio"]): str(spec["long_score"]) for spec in COMBO_SPECS}
    return mapping.get(str(portfolio), "")


def _portfolio_to_short_selector(portfolio: str) -> str:
    mapping = {str(spec["portfolio"]): str(spec["short_selector"]) for spec in COMBO_SPECS}
    return mapping.get(str(portfolio), "")


def _portfolio_to_risk_caps_label(portfolio: str) -> str:
    caps = RISK_CAPS_BY_PORTFOLIO.get(str(portfolio), ())
    if not caps:
        return ""
    return ";".join(
        f"{cap['name']}:{cap['column']}>={float(cap['threshold']):.2f}<={float(cap['cap']):.2f}"
        for cap in caps
    )


def _series_key(series: str) -> str:
    return (
        str(series)
        .replace("+", "plus")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "p")
        .lower()
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase5T price-only long-side risk-budget overlays."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--strict-root", default=str(DEFAULT_STRICT_ROOT))
    parser.add_argument("--phase3-rollup-path", default=str(DEFAULT_PHASE3_H10_ROLLUP))
    parser.add_argument("--risk-cap-penalty", type=float, default=DEFAULT_RISK_CAP_PENALTY)
    args = parser.parse_args(argv)
    result = build_phase5t_price_only_long_risk_budget_artifacts(
        signal_panel_path=args.signal_panel_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        output_root=args.output_root,
        strict_root=args.strict_root,
        phase3_rollup_path=args.phase3_rollup_path,
        risk_cap_penalty=args.risk_cap_penalty,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
