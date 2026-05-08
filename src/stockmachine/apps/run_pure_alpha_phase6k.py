"""Validation-only explicit size-neutral optimizer experiments.

Phase6J built the MVP point-in-time market-cap panel. Phase6K consumes that
panel and adds an explicit `market_cap_log_z` neutrality term to the current
ADV-floor product candidate construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT, TARGET_COLUMN
from stockmachine.apps.run_pure_alpha_phase4s import (
    DEFAULT_LONG_SCORE,
    DEFAULT_LONG_VARIANT,
    DEFAULT_SHORT_VARIANT,
    RESIDUAL_TARGETS,
    _diagnostic_row,
    _group_exposure_matrix,
    _position_rows,
    _skip_row,
    _zscore,
)
from stockmachine.apps.run_pure_alpha_phase4z import build_phase4z_strict_horizon_daily_paths
from stockmachine.apps.run_pure_alpha_phase5e import (
    DEFAULT_HOLDING_PERIOD_SESSIONS,
    _aggregate_turnover_summary,
    _candidate_union,
    _strict_metric_summary,
    _target_turnover_summary,
)
from stockmachine.apps.run_pure_alpha_phase5f import (
    DEFAULT_SHORT_OVERLAY,
    DEFAULT_TURNOVER_PENALTY,
)
from stockmachine.apps.run_pure_alpha_phase6e import _load_feature_panel


OUTDIR = RESEARCH_ROOT / "phase6k_size_neutral_optimizer_20260508"
STRICT_ROOT = OUTDIR / "strict_h10_rebuild"
POSITIONS_PATH = OUTDIR / "phase6k_positions_validation.csv.gz"
DAILY_PATH = OUTDIR / "phase6k_daily_validation.csv"
SKIPPED_PATH = OUTDIR / "phase6k_skipped_validation.csv"
SIZE_PANEL_PATH = (
    RESEARCH_ROOT
    / "phase6j_true_size_mvp_20260508"
    / "phase6j_true_size_panel_validation.csv.gz"
)
PHASE6E_METRICS_PATH = (
    RESEARCH_ROOT
    / "phase6e_adv_floor_hardening_20260507"
    / "phase6e_adv_floor_metrics.csv"
)
PHASE6E_CURVE_PATH = (
    RESEARCH_ROOT
    / "phase6e_adv_floor_hardening_20260507"
    / "phase6e_adv_floor_curve.csv"
)
ADV_FLOOR_USD = 1_000_000.0
SIZE_EXPOSURE_COLUMN = "market_cap_log_z"
SOFT_SIZE_PENALTIES = (0.0, 5.0, 25.0, 100.0)


@dataclass(frozen=True)
class SizeNeutralSpec:
    portfolio: str
    series: str
    size_mode: str
    size_penalty: float


def main() -> None:
    rollup = build_phase6k_size_neutral_artifacts()
    print(json.dumps(rollup, indent=2, ensure_ascii=True))


def build_phase6k_size_neutral_artifacts() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    specs = [
        SizeNeutralSpec(
            portfolio=f"size_soft_p{_label_number(penalty)}",
            series=f"size soft penalty {penalty:g}",
            size_mode="soft",
            size_penalty=float(penalty),
        )
        for penalty in SOFT_SIZE_PENALTIES
    ]
    specs.append(
        SizeNeutralSpec(
            portfolio="size_hard_neutral",
            series="size hard neutral",
            size_mode="hard",
            size_penalty=np.nan,
        )
    )

    positions, daily, skipped = _build_positions(specs)
    positions.to_csv(POSITIONS_PATH, index=False, compression="gzip")
    daily.to_csv(DAILY_PATH, index=False)
    skipped.to_csv(SKIPPED_PATH, index=False)

    strict_rollup = build_phase4z_strict_horizon_daily_paths(
        positions_path=POSITIONS_PATH,
        output_root=STRICT_ROOT,
        existing_curve_path=None,
        holding_period_sessions=DEFAULT_HOLDING_PERIOD_SESSIONS,
        lead_portfolio="size_hard_neutral",
    )
    strict_curve_path = STRICT_ROOT / "phase4z_strict_h10_daily_curve.csv"
    strict_curve = pd.read_csv(strict_curve_path, parse_dates=["return_date"])
    metrics = _build_metrics(strict_curve, positions, specs)
    size_exposure = _build_size_exposure(positions, daily)
    curves = _build_curve_export(strict_curve)
    comparison = _build_baseline_comparison(metrics)

    metrics_path = OUTDIR / "phase6k_size_neutral_metrics.csv"
    exposure_path = OUTDIR / "phase6k_size_exposure_validation.csv"
    curve_path = OUTDIR / "phase6k_size_neutral_curve.csv"
    comparison_path = OUTDIR / "phase6k_size_neutral_comparison.csv"
    plot_path = OUTDIR / "phase6k_size_neutral_plot.png"
    memo_path = OUTDIR / "phase6k_size_neutral_memo.md"
    rollup_path = OUTDIR / "phase6k_rollup.json"

    metrics.to_csv(metrics_path, index=False)
    size_exposure.to_csv(exposure_path, index=False)
    curves.to_csv(curve_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    _plot(curves, plot_path)
    memo_path.write_text(
        _memo(metrics=metrics, comparison=comparison, size_exposure=size_exposure, plot_path=plot_path),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": OUTDIR.as_posix(),
        "positions_artifact": POSITIONS_PATH.as_posix(),
        "daily_artifact": DAILY_PATH.as_posix(),
        "skipped_artifact": SKIPPED_PATH.as_posix(),
        "metrics_artifact": metrics_path.as_posix(),
        "size_exposure_artifact": exposure_path.as_posix(),
        "curve_artifact": curve_path.as_posix(),
        "comparison_artifact": comparison_path.as_posix(),
        "plot_artifact": plot_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "strict_rollup": strict_rollup,
        "size_panel_path": SIZE_PANEL_PATH.as_posix(),
        "adv_floor_usd": ADV_FLOOR_USD,
        "size_exposure_column": SIZE_EXPOSURE_COLUMN,
        "soft_size_penalties": list(SOFT_SIZE_PENALTIES),
        "method": "validation_only_explicit_market_cap_log_z_neutral_optimizer",
        "lockbox_status": "validation_only_no_test_window_performance",
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _build_positions(
    specs: list[SizeNeutralSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = _load_feature_panel()
    size_panel = pd.read_csv(
        SIZE_PANEL_PATH,
        usecols=[
            "variant",
            "session_date",
            "symbol",
            "market_cap",
            "market_cap_log",
            "market_cap_log_z",
            "market_cap_uses_spot_shares",
            "market_cap_price_source",
        ],
    )
    size_panel["session_date"] = size_panel["session_date"].astype(str)
    size_panel["symbol"] = size_panel["symbol"].astype(str)
    panel["session_date"] = panel["session_date"].astype(str)
    panel = panel.merge(size_panel, on=["variant", "session_date", "symbol"], how="left")
    panel = panel[panel["trailing_median_dollar_volume_20"].fillna(0.0) >= ADV_FLOOR_USD].copy()

    all_positions: list[dict[str, Any]] = []
    all_daily: list[dict[str, Any]] = []
    all_skipped: list[dict[str, Any]] = []
    groups = {
        (variant, session_date): group
        for (variant, session_date), group in panel.groupby(["variant", "session_date"], sort=True)
    }
    sessions = sorted(panel["session_date"].unique())

    for spec in specs:
        previous_side: dict[str, dict[str, float]] = {"long": {}, "short": {}}
        for session_date in sessions:
            long_group = groups.get((DEFAULT_LONG_VARIANT, session_date))
            short_group = groups.get((DEFAULT_SHORT_VARIANT, session_date))
            if long_group is None or short_group is None:
                all_skipped.append(_skip_row(session_date, spec.portfolio, 0, 0, "missing_group"))
                continue
            book, diagnostic, skip, updated = _construct_one_session_size_neutral(
                long_group,
                short_group,
                session_date=session_date,
                spec=spec,
                previous_weights=previous_side,
            )
            all_positions.extend(book)
            if diagnostic is not None:
                diagnostic["series"] = spec.series
                diagnostic["size_mode"] = spec.size_mode
                diagnostic["size_penalty"] = spec.size_penalty
                diagnostic["adv_floor_usd"] = ADV_FLOOR_USD
                all_daily.append(diagnostic)
            if skip is not None:
                skip["series"] = spec.series
                skip["size_mode"] = spec.size_mode
                skip["size_penalty"] = spec.size_penalty
                skip["adv_floor_usd"] = ADV_FLOOR_USD
                all_skipped.append(skip)
            previous_side = updated

    return (
        pd.DataFrame(all_positions),
        pd.DataFrame(all_daily),
        pd.DataFrame(all_skipped),
    )


def _construct_one_session_size_neutral(
    long_group: pd.DataFrame,
    short_group: pd.DataFrame,
    *,
    session_date: str,
    spec: SizeNeutralSpec,
    previous_weights: dict[str, dict[str, float]],
    candidate_pool_per_side: int = 80,
    max_single_name_side_weight: float = 1.0 / 30.0,
    min_nonzero_names: int = 20,
    score_weight: float = 0.01,
    soft_group_penalty: float = 25.0,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, dict[str, dict[str, float]]]:
    needed = [
        DEFAULT_LONG_SCORE,
        DEFAULT_SHORT_OVERLAY,
        "beta",
        "forward_return_5d",
        "benchmark_forward_return_5d",
        TARGET_COLUMN,
        *RESIDUAL_TARGETS,
        "sic2_sector",
        "sic4_industry",
        SIZE_EXPOSURE_COLUMN,
    ]
    long_base = long_group.dropna(subset=[DEFAULT_LONG_SCORE, *needed[2:]]).drop_duplicates("symbol").copy()
    short_base = short_group.dropna(subset=[DEFAULT_SHORT_OVERLAY, *needed[2:]]).drop_duplicates("symbol").copy()
    if len(long_base) < min_nonzero_names or len(short_base) < min_nonzero_names:
        return [], None, _skip_row(session_date, spec.portfolio, len(long_base), len(short_base), "insufficient_candidates"), previous_weights

    longs = _candidate_union(
        long_base,
        score_column=DEFAULT_LONG_SCORE,
        previous_weights=previous_weights["long"],
        candidate_pool_per_side=candidate_pool_per_side,
    )
    shorts = _candidate_union(
        short_base,
        score_column=DEFAULT_SHORT_OVERLAY,
        previous_weights=previous_weights["short"],
        candidate_pool_per_side=candidate_pool_per_side,
    )
    try:
        long_weights, short_weights = _optimize_size_neutral_joint_weights(
            longs,
            shorts,
            size_mode=spec.size_mode,
            size_penalty=spec.size_penalty,
            max_weight=max_single_name_side_weight,
            score_weight=score_weight,
            soft_group_penalty=soft_group_penalty,
            turnover_penalty=turnover_penalty,
            previous_long_weights=previous_weights["long"],
            previous_short_weights=previous_weights["short"],
        )
    except ValueError as exc:
        return [], None, _skip_row(session_date, spec.portfolio, len(longs), len(shorts), f"optimizer_failed:{exc}"), previous_weights

    if (long_weights > 1e-10).sum() < min_nonzero_names:
        return [], None, _skip_row(session_date, spec.portfolio, len(longs), len(shorts), "insufficient_long_nonzero"), previous_weights
    if (short_weights > 1e-10).sum() < min_nonzero_names:
        return [], None, _skip_row(session_date, spec.portfolio, len(longs), len(shorts), "insufficient_short_nonzero"), previous_weights

    book = [
        *_position_rows_with_size(
            longs,
            long_weights,
            session_date=session_date,
            portfolio=spec.portfolio,
            side="long",
            score_column=DEFAULT_LONG_SCORE,
        ),
        *_position_rows_with_size(
            shorts,
            short_weights,
            session_date=session_date,
            portfolio=spec.portfolio,
            side="short",
            score_column=DEFAULT_SHORT_OVERLAY,
        ),
    ]
    diagnostic = _diagnostic_row(
        longs,
        shorts,
        long_weights,
        short_weights,
        session_date=session_date,
        portfolio=spec.portfolio,
        long_variant=DEFAULT_LONG_VARIANT,
        short_variant=DEFAULT_SHORT_VARIANT,
        hard_group=None,
        soft_group="sic2_sector",
        long_score=DEFAULT_LONG_SCORE,
        short_selector=DEFAULT_SHORT_OVERLAY,
    )
    prev_long_vec = np.array([float(previous_weights["long"].get(symbol, 0.0)) for symbol in longs["symbol"]], dtype=float)
    prev_short_vec = np.array([float(previous_weights["short"].get(symbol, 0.0)) for symbol in shorts["symbol"]], dtype=float)
    diagnostic["turnover_penalty"] = float(turnover_penalty)
    diagnostic["target_turnover"] = float(np.abs(long_weights - prev_long_vec).sum() + np.abs(short_weights - prev_short_vec).sum())
    diagnostic["target_new_long_names"] = int(((long_weights > 1e-10) & (prev_long_vec <= 1e-10)).sum())
    diagnostic["target_new_short_names"] = int(((short_weights > 1e-10) & (prev_short_vec <= 1e-10)).sum())
    diagnostic.update(_size_diag(longs, shorts, long_weights, short_weights))
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


def _optimize_size_neutral_joint_weights(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    *,
    size_mode: str,
    size_penalty: float,
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
            _zscore(longs[DEFAULT_LONG_SCORE].to_numpy(dtype=float)),
            _zscore(shorts[DEFAULT_SHORT_OVERLAY].to_numpy(dtype=float)),
        ]
    )
    prev = np.concatenate(
        [
            np.array([float(previous_long_weights.get(symbol, 0.0)) for symbol in longs["symbol"]], dtype=float),
            np.array([float(previous_short_weights.get(symbol, 0.0)) for symbol in shorts["symbol"]], dtype=float),
        ]
    )
    size_row = _size_exposure_row(longs, shorts)

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
    if size_mode == "hard":
        aeq.append(size_row)
        beq.append(0.0)
    elif size_mode != "soft":
        raise ValueError(f"unknown_size_mode:{size_mode}")

    soft_matrix = _group_exposure_matrix(longs, shorts, "sic2_sector")
    m_soft = soft_matrix.shape[0]
    m_size = 1 if size_mode == "soft" else 0
    c = np.concatenate(
        [
            -score_weight * score,
            np.full(n, turnover_penalty, dtype=float),
            np.full(m_soft, soft_group_penalty, dtype=float),
            np.full(m_size, float(size_penalty), dtype=float),
        ]
    )
    bounds: list[tuple[float, float | None]] = (
        [(0.0, max_weight)] * n
        + [(0.0, None)] * n
        + [(0.0, None)] * m_soft
        + [(0.0, None)] * m_size
    )
    a_eq = np.vstack(aeq)
    b_eq = np.array(beq, dtype=float)
    a_eq = np.column_stack([a_eq, np.zeros((a_eq.shape[0], n + m_soft + m_size))])

    turnover_top = np.column_stack([np.eye(n), -np.eye(n), np.zeros((n, m_soft + m_size))])
    turnover_bottom = np.column_stack([-np.eye(n), -np.eye(n), np.zeros((n, m_soft + m_size))])
    a_ub = [turnover_top, turnover_bottom]
    b_ub = [prev, -prev]

    soft_top = np.column_stack([soft_matrix, np.zeros((m_soft, n)), -np.eye(m_soft), np.zeros((m_soft, m_size))])
    soft_bottom = np.column_stack([-soft_matrix, np.zeros((m_soft, n)), -np.eye(m_soft), np.zeros((m_soft, m_size))])
    a_ub.extend([soft_top, soft_bottom])
    b_ub.extend([np.zeros(m_soft), np.zeros(m_soft)])

    if m_size:
        size_top = np.concatenate([size_row, np.zeros(n + m_soft), np.array([-1.0])])[None, :]
        size_bottom = np.concatenate([-size_row, np.zeros(n + m_soft), np.array([-1.0])])[None, :]
        a_ub.extend([size_top, size_bottom])
        b_ub.extend([np.array([0.0]), np.array([0.0])])

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
    net_size = float(np.dot(size_row, x))
    if size_mode == "hard" and abs(net_size) > 1e-5:
        raise ValueError("size_constraint_breach")
    return long_weights, short_weights


def _position_rows_with_size(
    candidates: pd.DataFrame,
    weights: np.ndarray,
    *,
    session_date: str,
    portfolio: str,
    side: str,
    score_column: str,
) -> list[dict[str, Any]]:
    rows = _position_rows(
        candidates,
        weights,
        session_date=session_date,
        portfolio=portfolio,
        long_variant=DEFAULT_LONG_VARIANT,
        short_variant=DEFAULT_SHORT_VARIANT,
        side=side,
        score_column=score_column,
    )
    candidate_lookup = candidates.set_index("symbol")
    for row in rows:
        source = candidate_lookup.loc[row["symbol"]]
        row["market_cap"] = float(source["market_cap"])
        row["market_cap_log"] = float(source["market_cap_log"])
        row["market_cap_log_z"] = float(source["market_cap_log_z"])
        row["market_cap_uses_spot_shares"] = bool(source["market_cap_uses_spot_shares"])
        row["market_cap_price_source"] = str(source["market_cap_price_source"])
    return rows


def _size_exposure_row(longs: pd.DataFrame, shorts: pd.DataFrame) -> np.ndarray:
    return np.concatenate(
        [
            longs[SIZE_EXPOSURE_COLUMN].to_numpy(dtype=float),
            -shorts[SIZE_EXPOSURE_COLUMN].to_numpy(dtype=float),
        ]
    )


def _size_diag(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    long_weights: np.ndarray,
    short_weights: np.ndarray,
) -> dict[str, Any]:
    long_size = float(np.dot(long_weights, longs[SIZE_EXPOSURE_COLUMN].to_numpy(dtype=float)))
    short_size = float(np.dot(short_weights, shorts[SIZE_EXPOSURE_COLUMN].to_numpy(dtype=float)))
    return {
        "long_market_cap_log_z": long_size,
        "short_market_cap_log_z": short_size,
        "net_market_cap_log_z": long_size - short_size,
        "abs_net_market_cap_log_z": abs(long_size - short_size),
        "spot_share_weight_rate": float(
            (
                np.dot(long_weights, longs["market_cap_uses_spot_shares"].astype(float))
                + np.dot(short_weights, shorts["market_cap_uses_spot_shares"].astype(float))
            )
            / 2.0
        ),
    }


def _build_metrics(
    strict_curve: pd.DataFrame,
    positions: pd.DataFrame,
    specs: list[SizeNeutralSpec],
) -> pd.DataFrame:
    strict_metrics = _strict_metric_summary(strict_curve)
    target_turnover = _target_turnover_summary(positions)
    benchmark_calendar = strict_curve["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
    aggregate_turnover = _aggregate_turnover_summary(
        positions_path=POSITIONS_PATH,
        benchmark_returns=benchmark_calendar.to_frame(name="session_date"),
        holding_period_sessions=DEFAULT_HOLDING_PERIOD_SESSIONS,
    )
    spec_frame = pd.DataFrame([spec.__dict__ for spec in specs])
    size_summary = (
        positions.groupby("portfolio", sort=False)
        .apply(_position_size_summary, include_groups=False)
        .reset_index()
    )
    return (
        spec_frame.merge(strict_metrics, on="portfolio", how="left")
        .merge(target_turnover, on="portfolio", how="left")
        .merge(aggregate_turnover, on="portfolio", how="left")
        .merge(size_summary, on="portfolio", how="left")
        .reset_index(drop=True)
    )


def _position_size_summary(group: pd.DataFrame) -> pd.Series:
    daily = (
        group.assign(size_contribution=group["signed_weight"].astype(float) * group["market_cap_log_z"].astype(float))
        .groupby("session_date")["size_contribution"]
        .sum()
    )
    return pd.Series(
        {
            "mean_net_market_cap_log_z": float(daily.mean()),
            "mean_abs_net_market_cap_log_z": float(daily.abs().mean()),
            "p90_abs_net_market_cap_log_z": float(daily.abs().quantile(0.90)),
            "max_abs_net_market_cap_log_z": float(daily.abs().max()),
            "spot_share_position_rate": float(group["market_cap_uses_spot_shares"].astype(bool).mean()),
        }
    )


def _build_size_exposure(positions: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, group in positions.groupby("portfolio", sort=False):
        target = (
            group.assign(size_contribution=group["signed_weight"].astype(float) * group["market_cap_log_z"].astype(float))
            .groupby("session_date")["size_contribution"]
            .sum()
        )
        rows.append(
            {
                "portfolio": portfolio,
                "scope": "target",
                "sessions": int(target.count()),
                "mean_net_market_cap_log_z": float(target.mean()),
                "mean_abs_net_market_cap_log_z": float(target.abs().mean()),
                "p90_abs_net_market_cap_log_z": float(target.abs().quantile(0.90)),
                "max_abs_net_market_cap_log_z": float(target.abs().max()),
            }
        )
    diag_cols = [
        "portfolio",
        "session_date",
        "series",
        "size_mode",
        "size_penalty",
        "long_market_cap_log_z",
        "short_market_cap_log_z",
        "net_market_cap_log_z",
        "abs_net_market_cap_log_z",
        "spot_share_weight_rate",
    ]
    details = daily[diag_cols].copy() if not daily.empty else pd.DataFrame(columns=diag_cols)
    return pd.concat([pd.DataFrame(rows), details], ignore_index=True, sort=False)


def _build_curve_export(strict_curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, sub in strict_curve.groupby("portfolio", sort=False):
        frame = sub.sort_values("return_date").copy()
        frame["equity"] = (1.0 + frame["gross_return"].astype(float)).cumprod()
        frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_baseline_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if PHASE6E_METRICS_PATH.exists():
        base = pd.read_csv(PHASE6E_METRICS_PATH)
        base = base[base["portfolio"].astype(str).eq("adv_floor_1m")].copy()
        if not base.empty:
            item = base.iloc[0]
            rows.append(
                {
                    "portfolio": "adv_floor_1m",
                    "series": "current product candidate",
                    "strict_annualized_return": float(item["strict_annualized_return"]),
                    "strict_annualized_vol": float(item["strict_annualized_vol"]),
                    "strict_sharpe_no_rf": float(item["strict_sharpe_no_rf"]),
                    "strict_max_drawdown": float(item["strict_max_drawdown"]),
                    "aggregate_turnover_mean": float(item["aggregate_turnover_mean"]),
                    "mean_abs_net_market_cap_log_z": 0.22560187956747646,
                    "source": "phase6e_baseline_plus_phase6j_size_audit",
                }
            )
    for _, item in metrics.iterrows():
        rows.append(
            {
                "portfolio": item["portfolio"],
                "series": item["series"],
                "strict_annualized_return": float(item["strict_annualized_return"]),
                "strict_annualized_vol": float(item["strict_annualized_vol"]),
                "strict_sharpe_no_rf": float(item["strict_sharpe_no_rf"]),
                "strict_max_drawdown": float(item["strict_max_drawdown"]),
                "aggregate_turnover_mean": float(item["aggregate_turnover_mean"]),
                "mean_abs_net_market_cap_log_z": float(item["mean_abs_net_market_cap_log_z"]),
                "source": "phase6k",
            }
        )
    return pd.DataFrame(rows)


def _plot(curves: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = {
        "size_soft_p0": "#64748b",
        "size_soft_p5": "#2563eb",
        "size_soft_p25": "#0f766e",
        "size_soft_p100": "#c2410c",
        "size_hard_neutral": "#111827",
    }
    for portfolio, group in curves.groupby("portfolio", sort=False):
        axes[0].plot(group["return_date"], group["equity"], label=portfolio, color=colors.get(portfolio))
        axes[1].plot(group["return_date"], group["drawdown"] * 100, label=portfolio, color=colors.get(portfolio))
    axes[0].set_title("Phase6K Explicit Size-Neutral Optimizer (Validation Only)")
    axes[0].set_ylabel("Growth of $1")
    axes[1].set_ylabel("Drawdown %")
    axes[1].set_xlabel("Date")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _memo(
    *,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    size_exposure: pd.DataFrame,
    plot_path: Path,
) -> str:
    view_cols = [
        "portfolio",
        "series",
        "strict_annualized_return",
        "strict_annualized_vol",
        "strict_sharpe_no_rf",
        "strict_max_drawdown",
        "aggregate_turnover_mean",
        "mean_abs_net_market_cap_log_z",
    ]
    lines = [
        "# Phase6K Explicit Size-Neutral Optimizer",
        "",
        "Scope: validation-window only. Test lockbox remains closed.",
        "",
        "This experiment adds an explicit `market_cap_log_z` neutrality term to the current ADV-floor product-candidate construction.",
        "",
        "## Comparison",
        "",
        "```text",
        comparison[view_cols].to_string(index=False),
        "```",
        "",
        "## Size Exposure",
        "",
        "```text",
        size_exposure[size_exposure["scope"].astype(str).eq("target")].to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        "- `size_hard_neutral` is the cleanest hygiene version because it enforces target-level market-cap-log exposure equal to zero.",
        "- Soft penalty variants show whether size neutrality meaningfully harms path quality before we promote the hard constraint.",
        "- If hard size neutral preserves most economics, small-cap tilt was mostly incidental. If it collapses, the old SOTA was leaning on size style risk.",
        "",
        f"Plot: `{plot_path.as_posix()}`",
    ]
    return "\n".join(lines) + "\n"


def _label_number(value: float) -> str:
    text = f"{value:g}".replace(".", "p")
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    main()
