"""Validation-only dynamic price-style allocation experiment.

Phase6T intentionally treats the price-only strategy as dynamic style
allocation rather than pure alpha. The first version uses the same adv30m
universe on both sides and one shared score:

    score_i,t = b_i,t' * posterior_style_payoff_t

The test lockbox is not used.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT, TARGET_COLUMN
from stockmachine.apps.run_pure_alpha_phase4s import (
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
from stockmachine.apps.run_pure_alpha_phase5f import DEFAULT_TURNOVER_PENALTY
from stockmachine.apps.run_pure_alpha_phase6e import _load_feature_panel


STYLE_SET_NAME = os.environ.get("PHASE6T_STYLE_SET", "k4_core").strip() or "k4_core"
OUTDIR_BY_STYLE_SET = {
    "k4_core": "phase6t_dynamic_price_style_allocation_20260511",
    "k1_reversal": "phase6t_dynamic_price_style_allocation_k1_reversal_20260512",
}
if STYLE_SET_NAME not in OUTDIR_BY_STYLE_SET:
    raise ValueError(f"unknown PHASE6T_STYLE_SET={STYLE_SET_NAME!r}; expected one of {sorted(OUTDIR_BY_STYLE_SET)}")

OUTDIR = RESEARCH_ROOT / OUTDIR_BY_STYLE_SET[STYLE_SET_NAME]
STRICT_ROOT = OUTDIR / "strict_h10_rebuild"
POSITIONS_PATH = OUTDIR / "phase6t_positions_validation.csv.gz"
DAILY_PATH = OUTDIR / "phase6t_daily_validation.csv"
SKIPPED_PATH = OUTDIR / "phase6t_skipped_validation.csv"
SIZE_PANEL_PATH = (
    RESEARCH_ROOT
    / "phase6j_true_size_mvp_20260508"
    / "phase6j_true_size_panel_validation.csv.gz"
)
SOTA_CURVE_PATH = (
    RESEARCH_ROOT
    / "phase6k_size_neutral_optimizer_20260508"
    / "phase6k_size_neutral_curve.csv"
)

UNIVERSE_VARIANT = DEFAULT_SHORT_VARIANT
# The h10 beta-full signal panel starts on 2014-08-05. The dynamic payoff
# posterior needs 60 matured h10 observations before it is allowed to trade.
EVAL_START = "2014-11-11"
EVAL_END = "2019-12-31"
ADV_FLOOR_USD = 1_000_000.0
SIZE_EXPOSURE_COLUMN = "market_cap_log_z"
LONG_SCORE_COLUMN = "dynamic_style_score"
SHORT_SCORE_COLUMN = "dynamic_short_score"
MIN_MATURE_OBSERVATIONS = 60
TOP_BOTTOM_QUANTILE = 0.20


@dataclass(frozen=True)
class StyleSpec:
    name: str
    source_column: str
    sign: float
    description: str


@dataclass(frozen=True)
class PosteriorSpec:
    portfolio: str
    series: str
    method: str
    half_life: int | None = None
    phi: float | None = None
    q_ratio: float | None = None
    posterior_penalty: float = 0.0


ALL_STYLE_SPECS = {
    "reversal_5d": StyleSpec(
        "reversal_5d",
        "reversal_5d",
        1.0,
        "High score means recent 5-session loser / short-term reversal exposure.",
    ),
    "anti_momentum_20d": StyleSpec(
        "anti_momentum_20d",
        "momentum_20d",
        -1.0,
        "High score means recent 20-session loser / intermediate anti-momentum exposure.",
    ),
    "anti_beta_residual_momentum_20d": StyleSpec(
        "anti_beta_residual_momentum_20d",
        "beta_residual_momentum_20d",
        -1.0,
        "High score means beta-residual 20-session loser.",
    ),
    "anti_vol_adjusted_momentum_20d": StyleSpec(
        "anti_vol_adjusted_momentum_20d",
        "vol_adjusted_momentum_20d",
        -1.0,
        "High score means volatility-adjusted 20-session loser.",
    ),
}
STYLE_SET_SPECS = {
    "k4_core": (
        ALL_STYLE_SPECS["reversal_5d"],
        ALL_STYLE_SPECS["anti_momentum_20d"],
        ALL_STYLE_SPECS["anti_beta_residual_momentum_20d"],
        ALL_STYLE_SPECS["anti_vol_adjusted_momentum_20d"],
    ),
    "k1_reversal": (ALL_STYLE_SPECS["reversal_5d"],),
}
STYLE_SPECS = STYLE_SET_SPECS[STYLE_SET_NAME]

POSTERIOR_SPECS = (
    PosteriorSpec("static_mean", "static expanding mean", "static"),
    PosteriorSpec("ewma_hl20", "EWMA half-life 20", "ewma", half_life=20),
    PosteriorSpec("ewma_hl60", "EWMA half-life 60", "ewma", half_life=60),
    PosteriorSpec("ewma_hl120", "EWMA half-life 120", "ewma", half_life=120),
    PosteriorSpec(
        "kalman_phi099_q0004",
        "Kalman phi 0.99 q_ratio 0.0004",
        "kalman",
        phi=0.99,
        q_ratio=0.0004,
        posterior_penalty=0.0,
    ),
    PosteriorSpec(
        "kalman_phi099_q0004_shrink025",
        "Kalman phi 0.99 q_ratio 0.0004 shrink 0.25",
        "kalman",
        phi=0.99,
        q_ratio=0.0004,
        posterior_penalty=0.25,
    ),
)


def main() -> None:
    rollup = build_phase6t_dynamic_price_style_allocation()
    print(json.dumps(rollup, indent=2, ensure_ascii=True))


def build_phase6t_dynamic_price_style_allocation() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = _load_adv30m_panel()
    payoff_panel = _build_style_payoff_panel(panel)
    forecasts = _build_posterior_forecasts(payoff_panel, sorted(panel["session_date"].unique()))
    positions, daily, skipped = _build_positions(panel, forecasts)

    payoff_path = OUTDIR / "phase6t_style_payoff_panel.csv"
    forecast_path = OUTDIR / "phase6t_style_posterior_forecasts.csv"
    positions.to_csv(POSITIONS_PATH, index=False, compression="gzip")
    daily.to_csv(DAILY_PATH, index=False)
    skipped.to_csv(SKIPPED_PATH, index=False)
    payoff_panel.to_csv(payoff_path, index=False)
    forecasts.to_csv(forecast_path, index=False)

    strict_rollup = build_phase4z_strict_horizon_daily_paths(
        positions_path=POSITIONS_PATH,
        output_root=STRICT_ROOT,
        existing_curve_path=None,
        holding_period_sessions=DEFAULT_HOLDING_PERIOD_SESSIONS,
        lead_portfolio="kalman_phi099_q0004",
    )
    strict_curve = pd.read_csv(
        STRICT_ROOT / "phase4z_strict_h10_daily_curve.csv",
        parse_dates=["return_date"],
    )
    metrics = _build_metrics(strict_curve, positions, daily)
    curve = _build_curve_export(strict_curve)
    curve_metrics = _build_curve_metrics(curve)
    style_exposure = _build_style_exposure_summary(positions)
    posterior_summary = _build_posterior_summary(forecasts)

    metrics_path = OUTDIR / "phase6t_metrics.csv"
    curve_path = OUTDIR / "phase6t_curve.csv"
    curve_metrics_path = OUTDIR / "phase6t_curve_metrics.csv"
    exposure_path = OUTDIR / "phase6t_style_exposure_summary.csv"
    posterior_summary_path = OUTDIR / "phase6t_posterior_summary.csv"
    plot_path = OUTDIR / "phase6t_dynamic_style_allocation_plot.png"
    memo_path = OUTDIR / "phase6t_dynamic_style_allocation_memo.md"
    rollup_path = OUTDIR / "phase6t_rollup.json"

    metrics.to_csv(metrics_path, index=False)
    curve.to_csv(curve_path, index=False)
    curve_metrics.to_csv(curve_metrics_path, index=False)
    style_exposure.to_csv(exposure_path, index=False)
    posterior_summary.to_csv(posterior_summary_path, index=False)
    _plot(curve, plot_path)
    memo_path.write_text(
        _memo(
            metrics=metrics,
            curve_metrics=curve_metrics,
            style_exposure=style_exposure,
            posterior_summary=posterior_summary,
            plot_path=plot_path,
        ),
        encoding="utf-8",
    )

    rollup = {
        "phase": "phase6t_dynamic_price_style_allocation",
        "created_at_utc": _utc_now(),
        "scope": "validation_only",
        "test_lockbox_used": False,
        "style_set_name": STYLE_SET_NAME,
        "universe_variant": UNIVERSE_VARIANT,
        "long_variant": UNIVERSE_VARIANT,
        "short_variant": UNIVERSE_VARIANT,
        "same_score_function_both_sides": True,
        "eval_start": EVAL_START,
        "eval_end": EVAL_END,
        "style_specs": [spec.__dict__ for spec in STYLE_SPECS],
        "posterior_specs": [spec.__dict__ for spec in POSTERIOR_SPECS],
        "rows": {
            "panel": int(len(panel)),
            "payoff_panel": int(len(payoff_panel)),
            "forecasts": int(len(forecasts)),
            "positions": int(len(positions)),
            "daily": int(len(daily)),
            "skipped": int(len(skipped)),
        },
        "outputs": {
            "positions": str(POSITIONS_PATH),
            "daily": str(DAILY_PATH),
            "skipped": str(SKIPPED_PATH),
            "payoff_panel": str(payoff_path),
            "posterior_forecasts": str(forecast_path),
            "metrics": str(metrics_path),
            "curve": str(curve_path),
            "curve_metrics": str(curve_metrics_path),
            "style_exposure": str(exposure_path),
            "posterior_summary": str(posterior_summary_path),
            "plot": str(plot_path),
            "memo": str(memo_path),
        },
        "strict_rollup": strict_rollup,
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_adv30m_panel() -> pd.DataFrame:
    panel = _load_feature_panel()
    panel["session_date"] = panel["session_date"].astype(str)
    panel["variant"] = panel["variant"].astype(str)
    panel["symbol"] = panel["symbol"].astype(str)
    panel = panel[panel["variant"].eq(UNIVERSE_VARIANT)].copy()

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
    size_panel["variant"] = size_panel["variant"].astype(str)
    size_panel["symbol"] = size_panel["symbol"].astype(str)
    panel = panel.merge(size_panel, on=["variant", "session_date", "symbol"], how="left")
    panel = panel[panel["trailing_median_dollar_volume_20"].fillna(0.0) >= ADV_FLOOR_USD].copy()

    numeric_columns = {
        "beta",
        "forward_return_5d",
        "benchmark_forward_return_5d",
        TARGET_COLUMN,
        "trailing_median_dollar_volume_20",
        SIZE_EXPOSURE_COLUMN,
        "market_cap",
        "market_cap_log",
        *(spec.source_column for spec in STYLE_SPECS),
        *RESIDUAL_TARGETS,
    }
    for column in numeric_columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

    for spec in STYLE_SPECS:
        raw_column = _style_raw_column(spec)
        z_column = _style_z_column(spec)
        panel[raw_column] = float(spec.sign) * panel[spec.source_column]
        panel[z_column] = panel.groupby("session_date", sort=False)[raw_column].transform(_zscore)

    required = [
        "session_date",
        "symbol",
        "beta",
        "forward_return_5d",
        "benchmark_forward_return_5d",
        TARGET_COLUMN,
        "sic2_sector",
        "sic4_industry",
        SIZE_EXPOSURE_COLUMN,
        *(spec.source_column for spec in STYLE_SPECS),
        *(_style_z_column(spec) for spec in STYLE_SPECS),
        *RESIDUAL_TARGETS,
    ]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    return panel.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _build_style_payoff_panel(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session_date, group in panel.groupby("session_date", sort=True):
        for spec in STYLE_SPECS:
            score_column = _style_z_column(spec)
            frame = group[[score_column, TARGET_COLUMN]].dropna().copy()
            if len(frame) < 100:
                continue
            low = frame[score_column].quantile(TOP_BOTTOM_QUANTILE)
            high = frame[score_column].quantile(1.0 - TOP_BOTTOM_QUANTILE)
            top = frame[frame[score_column] >= high][TARGET_COLUMN]
            bottom = frame[frame[score_column] <= low][TARGET_COLUMN]
            if top.empty or bottom.empty:
                continue
            rows.append(
                {
                    "session_date": session_date,
                    "style": spec.name,
                    "score_column": score_column,
                    "names": int(len(frame)),
                    "top_count": int(len(top)),
                    "bottom_count": int(len(bottom)),
                    "payoff": float(top.mean() - bottom.mean()),
                    "payoff_bps": float((top.mean() - bottom.mean()) * 10000.0),
                    "top_mean": float(top.mean()),
                    "bottom_mean": float(bottom.mean()),
                }
            )
    return pd.DataFrame(rows)


def _build_posterior_forecasts(payoff_panel: pd.DataFrame, sessions: list[str]) -> pd.DataFrame:
    payoff = payoff_panel.pivot(index="session_date", columns="style", values="payoff")
    rows: list[dict[str, Any]] = []
    states = {
        spec.portfolio: {style.name: _initial_state() for style in STYLE_SPECS}
        for spec in POSTERIOR_SPECS
    }
    sessions = sorted(sessions)
    for idx, session_date in enumerate(sessions):
        mature_idx = idx - DEFAULT_HOLDING_PERIOD_SESSIONS
        if mature_idx >= 0:
            mature_date = sessions[mature_idx]
            if mature_date in payoff.index:
                observations = payoff.loc[mature_date]
                for posterior_spec in POSTERIOR_SPECS:
                    for style_spec in STYLE_SPECS:
                        value = observations.get(style_spec.name, np.nan)
                        if np.isfinite(value):
                            _update_state(
                                states[posterior_spec.portfolio][style_spec.name],
                                float(value),
                                posterior_spec,
                            )
        for posterior_spec in POSTERIOR_SPECS:
            row: dict[str, Any] = {
                "session_date": session_date,
                "portfolio": posterior_spec.portfolio,
                "series": posterior_spec.series,
                "method": posterior_spec.method,
                "posterior_penalty": float(posterior_spec.posterior_penalty),
                "min_style_observations": min(
                    states[posterior_spec.portfolio][style.name]["count"]
                    for style in STYLE_SPECS
                ),
                "test_window_used": False,
            }
            for style_spec in STYLE_SPECS:
                state = states[posterior_spec.portfolio][style_spec.name]
                forecast, variance = _state_forecast(state, posterior_spec)
                row[f"forecast_{style_spec.name}"] = forecast
                row[f"variance_{style_spec.name}"] = variance
                row[f"observations_{style_spec.name}"] = int(state["count"])
            rows.append(row)
    return pd.DataFrame(rows)


def _initial_state() -> dict[str, Any]:
    return {
        "count": 0,
        "obs_mean": 0.0,
        "obs_m2": 0.0,
        "ewma": np.nan,
        "kalman_mean": np.nan,
        "kalman_var": np.nan,
    }


def _update_state(state: dict[str, Any], value: float, spec: PosteriorSpec) -> None:
    _update_observation_moments(state, value)
    if spec.method == "ewma":
        alpha = 1.0 - 0.5 ** (1.0 / float(spec.half_life or 60))
        if not np.isfinite(state["ewma"]):
            state["ewma"] = value
        else:
            state["ewma"] = alpha * value + (1.0 - alpha) * float(state["ewma"])
    elif spec.method == "kalman":
        _update_kalman_state(
            state,
            value,
            phi=float(spec.phi or 1.0),
            q_ratio=float(spec.q_ratio or 0.0004),
        )


def _update_observation_moments(state: dict[str, Any], value: float) -> None:
    count = int(state["count"]) + 1
    delta = value - float(state["obs_mean"])
    mean = float(state["obs_mean"]) + delta / count
    delta2 = value - mean
    state["count"] = count
    state["obs_mean"] = mean
    state["obs_m2"] = float(state["obs_m2"]) + delta * delta2


def _update_kalman_state(state: dict[str, Any], value: float, *, phi: float, q_ratio: float) -> None:
    r = _observation_variance(state)
    q = max(q_ratio * r, 1e-10)
    if not np.isfinite(state["kalman_mean"]):
        state["kalman_mean"] = float(state["obs_mean"])
        state["kalman_var"] = r
        return
    pred_mean = phi * float(state["kalman_mean"])
    pred_var = phi * phi * float(state["kalman_var"]) + q
    gain = pred_var / max(pred_var + r, 1e-12)
    state["kalman_mean"] = pred_mean + gain * (value - pred_mean)
    state["kalman_var"] = max((1.0 - gain) * pred_var, 1e-12)


def _state_forecast(state: dict[str, Any], spec: PosteriorSpec) -> tuple[float, float]:
    if int(state["count"]) < MIN_MATURE_OBSERVATIONS:
        return 0.0, _observation_variance(state)
    if spec.method == "static":
        return float(state["obs_mean"]), _observation_variance(state) / max(int(state["count"]), 1)
    if spec.method == "ewma":
        value = float(state["ewma"]) if np.isfinite(state["ewma"]) else float(state["obs_mean"])
        return value, _observation_variance(state)
    if spec.method == "kalman":
        mean = float(state["kalman_mean"]) if np.isfinite(state["kalman_mean"]) else float(state["obs_mean"])
        variance = float(state["kalman_var"]) if np.isfinite(state["kalman_var"]) else _observation_variance(state)
        if spec.posterior_penalty > 0:
            mean = _shrink_toward_zero(mean, float(spec.posterior_penalty) * np.sqrt(max(variance, 0.0)))
        return mean, variance
    raise ValueError(f"unknown posterior method: {spec.method}")


def _observation_variance(state: dict[str, Any]) -> float:
    count = int(state["count"])
    if count <= 2:
        return 1e-4
    return max(float(state["obs_m2"]) / (count - 1), 1e-8)


def _shrink_toward_zero(value: float, amount: float) -> float:
    if value > 0:
        return max(value - amount, 0.0)
    if value < 0:
        return min(value + amount, 0.0)
    return 0.0


def _build_positions(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    candidate_pool_per_side: int = 80,
    max_single_name_side_weight: float = 1.0 / 30.0,
    min_nonzero_names: int = 20,
    score_weight: float = 0.01,
    soft_group_penalty: float = 25.0,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eval_panel = panel[(panel["session_date"] >= EVAL_START) & (panel["session_date"] <= EVAL_END)].copy()
    eval_sessions = sorted(eval_panel["session_date"].unique())
    groups = {
        session_date: group
        for session_date, group in eval_panel.groupby("session_date", sort=True)
    }
    forecast_map = {
        (row["portfolio"], row["session_date"]): row
        for _, row in forecasts.iterrows()
    }
    positions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    previous_by_portfolio = {
        spec.portfolio: {"long": {}, "short": {}}
        for spec in POSTERIOR_SPECS
    }

    for spec in POSTERIOR_SPECS:
        previous_side = previous_by_portfolio[spec.portfolio]
        for session_date in eval_sessions:
            group = groups.get(session_date)
            forecast = forecast_map.get((spec.portfolio, session_date))
            if group is None or forecast is None:
                skipped.append(_skip_row(session_date, spec.portfolio, 0, 0, "missing_group_or_forecast"))
                continue
            scored = _add_dynamic_score(group, forecast)
            book, diagnostic, skip, updated = _construct_one_session_dynamic(
                scored,
                session_date=session_date,
                portfolio=spec.portfolio,
                series=spec.series,
                previous_weights=previous_side,
                candidate_pool_per_side=candidate_pool_per_side,
                max_single_name_side_weight=max_single_name_side_weight,
                min_nonzero_names=min_nonzero_names,
                score_weight=score_weight,
                soft_group_penalty=soft_group_penalty,
                turnover_penalty=turnover_penalty,
            )
            positions.extend(book)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
            if skip is not None:
                skip["series"] = spec.series
                skipped.append(skip)
            previous_side = updated
            previous_by_portfolio[spec.portfolio] = updated

    return pd.DataFrame(positions), pd.DataFrame(diagnostics), pd.DataFrame(skipped)


def _add_dynamic_score(group: pd.DataFrame, forecast: pd.Series) -> pd.DataFrame:
    out = group.copy()
    score = np.zeros(len(out), dtype=float)
    for spec in STYLE_SPECS:
        score += out[_style_z_column(spec)].to_numpy(dtype=float) * float(
            forecast[f"forecast_{spec.name}"]
        )
    out[LONG_SCORE_COLUMN] = score
    out[SHORT_SCORE_COLUMN] = -score
    return out


def _construct_one_session_dynamic(
    group: pd.DataFrame,
    *,
    session_date: str,
    portfolio: str,
    series: str,
    previous_weights: dict[str, dict[str, float]],
    candidate_pool_per_side: int,
    max_single_name_side_weight: float,
    min_nonzero_names: int,
    score_weight: float,
    soft_group_penalty: float,
    turnover_penalty: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, dict[str, dict[str, float]]]:
    needed = [
        LONG_SCORE_COLUMN,
        SHORT_SCORE_COLUMN,
        "beta",
        "forward_return_5d",
        "benchmark_forward_return_5d",
        TARGET_COLUMN,
        *RESIDUAL_TARGETS,
        "sic2_sector",
        "sic4_industry",
        SIZE_EXPOSURE_COLUMN,
    ]
    base = group.dropna(subset=needed).drop_duplicates("symbol").copy()
    if len(base) < 2 * min_nonzero_names:
        return [], None, _skip_row(session_date, portfolio, len(base), len(base), "insufficient_candidates"), previous_weights

    longs = _candidate_union(
        base,
        score_column=LONG_SCORE_COLUMN,
        previous_weights=previous_weights["long"],
        candidate_pool_per_side=candidate_pool_per_side,
    )
    shorts = _candidate_union(
        base,
        score_column=SHORT_SCORE_COLUMN,
        previous_weights=previous_weights["short"],
        candidate_pool_per_side=candidate_pool_per_side,
    )
    longs, shorts = _drop_cross_side_overlaps(longs, shorts)
    if len(longs) < min_nonzero_names or len(shorts) < min_nonzero_names:
        return [], None, _skip_row(session_date, portfolio, len(longs), len(shorts), "insufficient_disjoint_candidates"), previous_weights

    try:
        long_weights, short_weights = _optimize_dynamic_joint_weights(
            longs,
            shorts,
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
        *_position_rows_with_dynamic_fields(
            longs,
            long_weights,
            session_date=session_date,
            portfolio=portfolio,
            side="long",
            score_column=LONG_SCORE_COLUMN,
        ),
        *_position_rows_with_dynamic_fields(
            shorts,
            short_weights,
            session_date=session_date,
            portfolio=portfolio,
            side="short",
            score_column=SHORT_SCORE_COLUMN,
        ),
    ]
    diagnostic = _diagnostic_row(
        longs,
        shorts,
        long_weights,
        short_weights,
        session_date=session_date,
        portfolio=portfolio,
        long_variant=UNIVERSE_VARIANT,
        short_variant=UNIVERSE_VARIANT,
        hard_group=None,
        soft_group="sic2_sector",
        long_score=LONG_SCORE_COLUMN,
        short_selector=SHORT_SCORE_COLUMN,
    )
    prev_long_vec = np.array([float(previous_weights["long"].get(symbol, 0.0)) for symbol in longs["symbol"]], dtype=float)
    prev_short_vec = np.array([float(previous_weights["short"].get(symbol, 0.0)) for symbol in shorts["symbol"]], dtype=float)
    diagnostic["series"] = series
    diagnostic["turnover_penalty"] = float(turnover_penalty)
    diagnostic["target_turnover"] = float(np.abs(long_weights - prev_long_vec).sum() + np.abs(short_weights - prev_short_vec).sum())
    diagnostic["target_new_long_names"] = int(((long_weights > 1e-10) & (prev_long_vec <= 1e-10)).sum())
    diagnostic["target_new_short_names"] = int(((short_weights > 1e-10) & (prev_short_vec <= 1e-10)).sum())
    diagnostic.update(_size_diag(longs, shorts, long_weights, short_weights))
    diagnostic.update(_style_diag(longs, shorts, long_weights, short_weights))
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


def _drop_cross_side_overlaps(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap = set(longs["symbol"].astype(str)).intersection(set(shorts["symbol"].astype(str)))
    if not overlap:
        return longs.reset_index(drop=True), shorts.reset_index(drop=True)
    long_drop: set[str] = set()
    short_drop: set[str] = set()
    score_lookup = pd.concat([longs, shorts], ignore_index=True).drop_duplicates("symbol").set_index("symbol")
    for symbol in overlap:
        score = float(score_lookup.loc[symbol, LONG_SCORE_COLUMN])
        if score >= 0:
            short_drop.add(symbol)
        else:
            long_drop.add(symbol)
    return (
        longs[~longs["symbol"].astype(str).isin(long_drop)].reset_index(drop=True),
        shorts[~shorts["symbol"].astype(str).isin(short_drop)].reset_index(drop=True),
    )


def _optimize_dynamic_joint_weights(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    *,
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
            _zscore(longs[LONG_SCORE_COLUMN].to_numpy(dtype=float)),
            _zscore(shorts[SHORT_SCORE_COLUMN].to_numpy(dtype=float)),
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
    aeq.append(size_row)
    beq.append(0.0)

    soft_matrix = _group_exposure_matrix(longs, shorts, "sic2_sector")
    m_soft = soft_matrix.shape[0]
    c = np.concatenate(
        [
            -score_weight * score,
            np.full(n, turnover_penalty, dtype=float),
            np.full(m_soft, soft_group_penalty, dtype=float),
        ]
    )
    bounds: list[tuple[float, float | None]] = (
        [(0.0, max_weight)] * n
        + [(0.0, None)] * n
        + [(0.0, None)] * m_soft
    )
    a_eq = np.vstack(aeq)
    b_eq = np.array(beq, dtype=float)
    a_eq = np.column_stack([a_eq, np.zeros((a_eq.shape[0], n + m_soft))])

    turnover_top = np.column_stack([np.eye(n), -np.eye(n), np.zeros((n, m_soft))])
    turnover_bottom = np.column_stack([-np.eye(n), -np.eye(n), np.zeros((n, m_soft))])
    soft_top = np.column_stack([soft_matrix, np.zeros((m_soft, n)), -np.eye(m_soft)])
    soft_bottom = np.column_stack([-soft_matrix, np.zeros((m_soft, n)), -np.eye(m_soft)])

    result = linprog(
        c,
        A_ub=np.vstack([turnover_top, turnover_bottom, soft_top, soft_bottom]),
        b_ub=np.concatenate([prev, -prev, np.zeros(m_soft), np.zeros(m_soft)]),
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
    if abs(net_size) > 1e-5:
        raise ValueError("size_constraint_breach")
    return long_weights, short_weights


def _position_rows_with_dynamic_fields(
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
        long_variant=UNIVERSE_VARIANT,
        short_variant=UNIVERSE_VARIANT,
        side=side,
        score_column=score_column,
    )
    lookup = candidates.set_index("symbol")
    for row in rows:
        source = lookup.loc[row["symbol"]]
        row["market_cap"] = float(source["market_cap"])
        row["market_cap_log"] = float(source["market_cap_log"])
        row["market_cap_log_z"] = float(source["market_cap_log_z"])
        row["market_cap_uses_spot_shares"] = bool(source["market_cap_uses_spot_shares"])
        row["market_cap_price_source"] = str(source["market_cap_price_source"])
        row[LONG_SCORE_COLUMN] = float(source[LONG_SCORE_COLUMN])
        row[SHORT_SCORE_COLUMN] = float(source[SHORT_SCORE_COLUMN])
        for spec in STYLE_SPECS:
            row[_style_z_column(spec)] = float(source[_style_z_column(spec)])
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
    }


def _style_diag(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    long_weights: np.ndarray,
    short_weights: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for spec in STYLE_SPECS:
        column = _style_z_column(spec)
        long_value = float(np.dot(long_weights, longs[column].to_numpy(dtype=float)))
        short_value = float(np.dot(short_weights, shorts[column].to_numpy(dtype=float)))
        out[f"long_{column}"] = long_value
        out[f"short_{column}"] = short_value
        out[f"net_{column}"] = long_value - short_value
    return out


def _build_metrics(
    strict_curve: pd.DataFrame,
    positions: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    strict_metrics = _strict_metric_summary(strict_curve)
    target_turnover = _target_turnover_summary(positions)
    benchmark_calendar = strict_curve["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
    aggregate_turnover = _aggregate_turnover_summary(
        positions_path=POSITIONS_PATH,
        benchmark_returns=benchmark_calendar.to_frame(name="session_date"),
        holding_period_sessions=DEFAULT_HOLDING_PERIOD_SESSIONS,
    )
    spec_frame = pd.DataFrame([spec.__dict__ for spec in POSTERIOR_SPECS])
    diagnostics = _diagnostic_summary(daily)
    return (
        spec_frame.merge(strict_metrics, on="portfolio", how="left")
        .merge(target_turnover, on="portfolio", how="left")
        .merge(aggregate_turnover, on="portfolio", how="left")
        .merge(diagnostics, on="portfolio", how="left")
        .sort_values(["strict_sharpe_no_rf", "strict_annualized_return"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _diagnostic_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, group in daily.groupby("portfolio", sort=False):
        rows.append(
            {
                "portfolio": portfolio,
                "daily_rows": int(len(group)),
                "mean_target_turnover": float(group["target_turnover"].mean()),
                "mean_abs_net_market_cap_log_z": float(group["abs_net_market_cap_log_z"].mean()),
                "mean_sic2_l1_exposure": float(group["sic2_l1_exposure"].mean()),
                "mean_spread_beta_residual": float(group["spread_beta_residual"].mean()),
                "mean_spread_style_factor_sic2_residual": float(group["spread_style_factor_sic2_residual"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_style_exposure_summary(positions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if positions.empty:
        return pd.DataFrame()
    for portfolio, group in positions.groupby("portfolio", sort=False):
        daily = group.groupby("session_date", sort=True)
        for spec in STYLE_SPECS:
            column = _style_z_column(spec)
            exposure = daily.apply(
                lambda frame: float(
                    np.dot(frame["signed_weight"].to_numpy(dtype=float), frame[column].to_numpy(dtype=float))
                ),
                include_groups=False,
            )
            rows.append(
                {
                    "portfolio": portfolio,
                    "style": spec.name,
                    "mean_exposure": float(exposure.mean()),
                    "mean_abs_exposure": float(exposure.abs().mean()),
                    "p90_abs_exposure": float(exposure.abs().quantile(0.90)),
                    "max_abs_exposure": float(exposure.abs().max()),
                }
            )
    return pd.DataFrame(rows)


def _build_posterior_summary(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eval_forecasts = forecasts[
        (forecasts["session_date"] >= EVAL_START) & (forecasts["session_date"] <= EVAL_END)
    ].copy()
    for portfolio, group in eval_forecasts.groupby("portfolio", sort=False):
        for spec in STYLE_SPECS:
            values = pd.to_numeric(group[f"forecast_{spec.name}"], errors="coerce")
            rows.append(
                {
                    "portfolio": portfolio,
                    "style": spec.name,
                    "mean_forecast_bps": float(values.mean() * 10000.0),
                    "mean_abs_forecast_bps": float(values.abs().mean() * 10000.0),
                    "positive_forecast_rate": float((values > 0).mean()),
                    "negative_forecast_rate": float((values < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _build_curve_export(strict_curve: pd.DataFrame) -> pd.DataFrame:
    curve = strict_curve.copy()
    curve["source"] = "phase6t"
    if SOTA_CURVE_PATH.exists():
        sota = pd.read_csv(SOTA_CURVE_PATH, parse_dates=["return_date"])
        sota = sota[
            sota["portfolio"].astype(str).eq("size_hard_neutral")
            & (sota["return_date"] >= pd.Timestamp(EVAL_START))
            & (sota["return_date"] <= pd.Timestamp(EVAL_END))
        ].copy()
        if not sota.empty:
            first_equity = float(sota["equity"].iloc[0])
            sota["equity"] = sota["equity"] / first_equity
            sota["drawdown"] = sota["equity"] / sota["equity"].cummax() - 1.0
            sota["portfolio"] = "sota_size_hard_neutral"
            sota["source"] = "phase6k_sota"
            keep = [column for column in curve.columns if column in sota.columns]
            curve = pd.concat([curve, sota[keep]], ignore_index=True, sort=False)
    return curve.sort_values(["portfolio", "return_date"]).reset_index(drop=True)


def _build_curve_metrics(curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if curve.empty:
        return pd.DataFrame()
    for portfolio, group in curve.groupby("portfolio", sort=True):
        group = group.sort_values("return_date")
        returns = pd.to_numeric(group["gross_return"], errors="coerce").fillna(0.0)
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        rolling_60 = equity / equity.shift(60) - 1.0
        benchmark = pd.to_numeric(group.get("benchmark_oto_return"), errors="coerce")
        rows.append(
            {
                "portfolio": portfolio,
                "source": str(group["source"].iloc[0]) if "source" in group else "",
                "final_equity": float(equity.iloc[-1]),
                "annualized_return": float(equity.iloc[-1] ** (252.0 / len(equity)) - 1.0),
                "annualized_vol": float(returns.std(ddof=1) * np.sqrt(252.0)),
                "sharpe_no_rf": float(
                    returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)
                    if returns.std(ddof=1) > 0
                    else np.nan
                ),
                "max_drawdown": float(drawdown.min()),
                "rolling_60_positive_rate": float((rolling_60 > 0).mean()),
                "corr_to_spy": float(returns.corr(benchmark)) if benchmark.notna().any() else np.nan,
                "rows": int(len(group)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["sharpe_no_rf", "annualized_return"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _plot(curve: pd.DataFrame, output_path: Path) -> None:
    if curve.empty:
        return
    plot = curve.copy()
    preferred = [
        "sota_size_hard_neutral",
        "ewma_hl120",
        "static_mean",
        "ewma_hl60",
        "kalman_phi099_q0004",
        "kalman_phi099_q0004_shrink025",
    ]
    plot = plot[plot["portfolio"].isin(preferred)].copy()
    if plot.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(17, 13), sharex=True)
    colors = {
        "sota_size_hard_neutral": "#111827",
        "ewma_hl120": "#16a34a",
        "static_mean": "#64748b",
        "ewma_hl60": "#c2410c",
        "kalman_phi099_q0004": "#0f766e",
        "kalman_phi099_q0004_shrink025": "#2563eb",
    }
    for portfolio, group in plot.groupby("portfolio", sort=False):
        group = group.sort_values("return_date")
        label = portfolio
        color = colors.get(portfolio)
        linewidth = 2.8 if portfolio.startswith("sota") else 2.0
        axes[0].plot(group["return_date"], group["equity"], label=label, color=color, linewidth=linewidth)
        axes[1].plot(group["return_date"], group["drawdown"] * 100.0, label=label, color=color, linewidth=linewidth)
        if "rolling_60_return" in group.columns:
            axes[2].plot(group["return_date"], group["rolling_60_return"] * 100.0, label=label, color=color, linewidth=linewidth)
    axes[0].set_title("Phase6T Dynamic Price-Style Allocation vs SOTA (Validation Only)")
    axes[0].set_ylabel("Growth of $1")
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("%")
    axes[2].set_title("Rolling 60-Session Return")
    axes[2].set_ylabel("%")
    axes[2].set_xlabel("Date")
    for ax in axes:
        ax.axhline(0.0 if ax is not axes[0] else 1.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.grid(True, linewidth=0.5, alpha=0.35)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _memo(
    *,
    metrics: pd.DataFrame,
    curve_metrics: pd.DataFrame,
    style_exposure: pd.DataFrame,
    posterior_summary: pd.DataFrame,
    plot_path: Path,
) -> str:
    lines = [
        "# Phase6T Dynamic Price-Style Allocation Memo",
        "",
        "Validation-only. Test lockbox is not used.",
        "",
        "## Design",
        "",
        "- Long universe and short universe are both `adv30m_clean_core_beta_full`.",
        "- Long and short books use the same dynamic score: `b_i,t' * posterior_style_payoff_t`.",
        "- Long book buys high-score names; short book shorts low-score names.",
        "- The strategy intentionally allows selected price-style exposure; this is not a pure-alpha construction.",
        "- Hard controls: dollar neutral, beta matched, true size neutral. Soft control: SIC2 exposure.",
        "",
        "## Comparable Curve Metrics",
        "",
        _markdown_table(curve_metrics),
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
        "",
        "## Style Exposure Summary",
        "",
        _markdown_table(style_exposure),
        "",
        "## Posterior Forecast Summary",
        "",
        _markdown_table(posterior_summary),
        "",
        f"Plot: `{plot_path.as_posix()}`",
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6g}")
        else:
            display[column] = display[column].astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str)]
    return "\n".join([header, separator, *rows])


def _style_raw_column(spec: StyleSpec) -> str:
    return f"dynamic_style_raw_{spec.name}"


def _style_z_column(spec: StyleSpec) -> str:
    return f"dynamic_style_z_{spec.name}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
