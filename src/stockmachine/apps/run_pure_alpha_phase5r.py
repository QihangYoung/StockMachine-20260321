from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    _add_residual_targets,
    _load_panel,
    _load_sec_sic_map,
    _skip_row,
)
from stockmachine.apps.run_pure_alpha_phase4z import (
    DEFAULT_PHASE3_H10_ROLLUP,
    build_phase4z_strict_horizon_daily_paths,
)
from stockmachine.apps.run_pure_alpha_phase5b import _fmt_float_like, _fmt_pct_like, _text_table, _utc_now
from stockmachine.apps.run_pure_alpha_phase5e import (
    DEFAULT_HOLDING_PERIOD_SESSIONS,
    _aggregate_turnover_summary,
    _construct_one_session_turnover_aware,
    _diagnostic_summary,
    _strict_metric_summary,
    _target_turnover_summary,
)
from stockmachine.apps.run_pure_alpha_phase5f import DEFAULT_SHORT_OVERLAY


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5r_price_only_stabilization_overlay_20260506"
DEFAULT_STRICT_ROOT = DEFAULT_OUTPUT_ROOT / "strict_h10_rebuild"
DEFAULT_TURNOVER_PENALTY = 0.005
CURRENT_SOTA_SERIES = "current sota"

PRICE_FK_COLUMNS = (
    "price_fk_sector_mom20_down",
    "price_fk_sector_mom60_down",
    "price_fk_sector_relative_mom20_down",
    "price_fk_sector_relative_mom60_down",
    "price_fk_residual_mom20_down",
    "price_fk_vol_adjusted_down",
    "price_fk_no_stabilization",
)

COMBO_SPECS: tuple[dict[str, str], ...] = (
    {
        "portfolio": "current_sota",
        "series": CURRENT_SOTA_SERIES,
        "long_score": DEFAULT_LONG_SCORE,
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "price_fk_soft_w075",
        "series": "price fk soft w075",
        "long_score": "long_price_fk_soft_w075",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "price_fk_soft_w100",
        "series": "price fk soft w100",
        "long_score": "long_price_fk_soft_w100",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "price_fk_stillfall_w075",
        "series": "price fk stillfall w075",
        "long_score": "long_price_fk_stillfall_w075",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "price_fk_stillfall_w100",
        "series": "price fk stillfall w100",
        "long_score": "long_price_fk_stillfall_w100",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "price_fk_stab_relief_w075",
        "series": "price fk stab relief w075",
        "long_score": "long_price_fk_stab_relief_w075",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "price_fk_stab_relief_w100",
        "series": "price fk stab relief w100",
        "long_score": "long_price_fk_stab_relief_w100",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
)

PLOT_COLORS = {
    CURRENT_SOTA_SERIES: "#2563eb",
    "price fk soft w075": "#0891b2",
    "price fk soft w100": "#155e75",
    "price fk stillfall w075": "#c2410c",
    "price fk stillfall w100": "#9a3412",
    "price fk stab relief w075": "#7c3aed",
    "price fk stab relief w100": "#9333ea",
    "SPY raw": "#6b7280",
}

WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("2015_peak_to_trough", "2015-06-04", "2015-08-21"),
    ("2016_jan_feb", "2016-01-05", "2016-02-09"),
    ("2017_spring_short_fail", "2017-03-20", "2017-05-31"),
    ("2018_aug_sep_short_fail", "2018-08-08", "2018-09-12"),
    ("2019_may", "2019-05-01", "2019-06-05"),
    ("2019_aug", "2019-08-01", "2019-08-31"),
)


def build_phase5r_price_only_stabilization_overlay_artifacts(
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
    activation = _activation_summary(panel, long_variant=long_variant, candidate_count=candidate_pool_per_side)
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
    )

    positions_path = output_dir / "phase5r_positions_validation.csv.gz"
    diagnostics_path = output_dir / "phase5r_daily_validation.csv"
    skipped_path = output_dir / "phase5r_skipped_validation.csv"
    activation_path = output_dir / "phase5r_price_state_activation_summary.csv"
    positions.to_csv(positions_path, index=False, compression="gzip")
    diagnostics.to_csv(diagnostics_path, index=False)
    skipped.to_csv(skipped_path, index=False)
    activation.to_csv(activation_path, index=False)

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
    exposure = _position_price_state_exposure_summary(positions=positions, panel=panel)
    sweep = (
        strict_metrics.merge(target_turnover, on="portfolio", how="left")
        .merge(aggregate_turnover, on="portfolio", how="left")
        .merge(diagnostic_summary, on="portfolio", how="left")
        .merge(exposure, on="portfolio", how="left")
    )
    sweep["series"] = sweep["portfolio"].map(_portfolio_to_series)
    sweep["long_score"] = sweep["portfolio"].map(_portfolio_to_long_score)
    sweep["short_selector"] = sweep["portfolio"].map(_portfolio_to_short_selector)
    sweep["turnover_penalty"] = float(turnover_penalty)

    metrics = _build_metrics(sweep=sweep, strict_curve=strict_curve, turnover_penalty=turnover_penalty)
    windows = _window_return_summary(strict_curve)
    comparison_curve = _build_comparison_curve(strict_curve)

    sweep_path = output_dir / "phase5r_sweep.csv"
    metrics_path = output_dir / "phase5r_metrics.csv"
    windows_path = output_dir / "phase5r_window_returns.csv"
    exposure_path = output_dir / "phase5r_position_price_state_exposure.csv"
    comparison_curve_path = output_dir / "phase5r_comparison_curve.csv"
    plot_path = output_dir / "phase5r_comparison_plot.png"
    memo_path = output_dir / "phase5r_price_only_stabilization_overlay_memo.md"
    rollup_path = output_dir / "phase5r_rollup.json"

    sweep.to_csv(sweep_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    windows.to_csv(windows_path, index=False)
    exposure.to_csv(exposure_path, index=False)
    comparison_curve.to_csv(comparison_curve_path, index=False)
    _plot_comparison(comparison_curve, output_path=plot_path)
    memo_path.write_text(
        _memo(
            metrics=metrics,
            activation=activation,
            exposure=exposure,
            windows=windows,
            plot_path=plot_path,
            turnover_penalty=turnover_penalty,
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
        "strict_rollup": strict_rollup,
        "artifacts": {
            "positions": positions_path.as_posix(),
            "diagnostics": diagnostics_path.as_posix(),
            "skipped": skipped_path.as_posix(),
            "activation": activation_path.as_posix(),
            "sweep": sweep_path.as_posix(),
            "metrics": metrics_path.as_posix(),
            "window_returns": windows_path.as_posix(),
            "position_price_state_exposure": exposure_path.as_posix(),
            "comparison_curve": comparison_curve_path.as_posix(),
            "plot": plot_path.as_posix(),
            "memo": memo_path.as_posix(),
            "strict_curve": strict_curve_path.as_posix(),
        },
        "method": "price_only_falling_knife_penalty_with_stabilization_relief",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "No non-price inputs are loaded or used.",
            "SEC-derived SIC labels are used only as neutralization and sector-relative price-state grouping keys.",
            "No test-window performance is computed.",
            "Stabilization is proxied using existing price fields: recent 5-session return and sector-relative recent return.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _add_price_only_falling_knife_scores(panel: pd.DataFrame, *, long_variant: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for (variant, session_date), group in panel.groupby(["variant", "session_date"], sort=False):
        part = group.copy()
        baseline = pd.to_numeric(part[DEFAULT_LONG_SCORE], errors="coerce")
        is_long_variant = str(variant) == str(long_variant)
        sector_keys = part["sic2_sector"].astype(str)
        sector_mom20 = part.groupby(sector_keys)["momentum_20d_z"].transform("median")
        sector_mom60 = part.groupby(sector_keys)["momentum_60d_z"].transform("median")
        sector_ret5 = part.groupby(sector_keys)["return_5d_z"].transform("median")

        part["price_fk_sector_mom20_down"] = (-sector_mom20).clip(lower=0.0)
        part["price_fk_sector_mom60_down"] = (-sector_mom60).clip(lower=0.0)
        part["price_fk_sector_relative_mom20_down"] = -(
            part["momentum_20d_z"] - sector_mom20
        )
        part["price_fk_sector_relative_mom20_down"] = part[
            "price_fk_sector_relative_mom20_down"
        ].clip(lower=0.0)
        part["price_fk_sector_relative_mom60_down"] = -(
            part["momentum_60d_z"] - sector_mom60
        )
        part["price_fk_sector_relative_mom60_down"] = part[
            "price_fk_sector_relative_mom60_down"
        ].clip(lower=0.0)
        part["price_fk_residual_mom20_down"] = (
            -part["beta_residual_momentum_20d_z"]
        ).clip(lower=0.0)
        part["price_fk_vol_adjusted_down"] = (
            -part["vol_adjusted_momentum_20d_z"]
        ).clip(lower=0.0)
        part["price_fk_no_stabilization"] = (-part["return_5d_z"]).clip(lower=0.0)
        raw = part[list(PRICE_FK_COLUMNS)].mean(axis=1)
        part["price_falling_knife_raw"] = raw
        part["price_falling_knife_score"] = raw.rank(method="average", pct=True).fillna(0.0)

        still_falling_raw = _mean_available(
            [
                (-part["return_5d_z"]).clip(lower=0.0),
                (-(part["return_5d_z"] - sector_ret5)).clip(lower=0.0),
                (-part["beta_residual_momentum_20d_z"]).clip(lower=0.0),
            ]
        )
        stabilization_raw = _mean_available(
            [
                part["return_5d_z"].clip(lower=0.0),
                (part["return_5d_z"] - sector_ret5).clip(lower=0.0),
            ]
        )
        part["price_still_falling_score"] = still_falling_raw.rank(
            method="average",
            pct=True,
        ).fillna(0.0)
        part["price_stabilization_score"] = stabilization_raw.rank(
            method="average",
            pct=True,
        ).fillna(0.0)
        part["price_fk_stillfall_penalty"] = part["price_falling_knife_score"] * (
            0.35 + 0.65 * part["price_still_falling_score"]
        )
        part["price_fk_stab_relief_penalty"] = part["price_fk_stillfall_penalty"] * (
            1.0 - 0.55 * part["price_stabilization_score"]
        ).clip(lower=0.10, upper=1.0)

        if is_long_variant:
            base_z = _zscore_array(baseline.to_numpy(dtype=float))
            part["long_price_fk_soft_w075"] = _zscore_array(
                base_z - 0.75 * part["price_falling_knife_score"].to_numpy(dtype=float)
            )
            part["long_price_fk_soft_w100"] = _zscore_array(
                base_z - 1.00 * part["price_falling_knife_score"].to_numpy(dtype=float)
            )
            part["long_price_fk_stillfall_w075"] = _zscore_array(
                base_z - 0.75 * part["price_fk_stillfall_penalty"].to_numpy(dtype=float)
            )
            part["long_price_fk_stillfall_w100"] = _zscore_array(
                base_z - 1.00 * part["price_fk_stillfall_penalty"].to_numpy(dtype=float)
            )
            part["long_price_fk_stab_relief_w075"] = _zscore_array(
                base_z - 0.75 * part["price_fk_stab_relief_penalty"].to_numpy(dtype=float)
            )
            part["long_price_fk_stab_relief_w100"] = _zscore_array(
                base_z - 1.00 * part["price_fk_stab_relief_penalty"].to_numpy(dtype=float)
            )
        else:
            for column in _long_price_score_columns():
                part[column] = np.nan
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else panel


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
            book, diagnostic, skip, updated = _construct_one_session_turnover_aware(
                long_group,
                short_group,
                session_date=session_date,
                portfolio=portfolio,
                long_variant=long_variant,
                short_variant=short_variant,
                long_score=str(spec["long_score"]),
                short_selector=str(spec["short_selector"]),
                soft_group="sic2_sector",
                candidate_pool_per_side=candidate_pool_per_side,
                max_single_name_side_weight=max_single_name_side_weight,
                min_nonzero_names=min_nonzero_names,
                score_weight=score_weight,
                soft_group_penalty=soft_group_penalty,
                turnover_penalty=turnover_penalty,
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


def _activation_summary(
    panel: pd.DataFrame,
    *,
    long_variant: str,
    candidate_count: int,
) -> pd.DataFrame:
    long_panel = panel[panel["variant"].astype(str).eq(long_variant)].copy()
    scopes = {
        "long_universe": long_panel,
        f"baseline_top{candidate_count}": _top_by_session(
            long_panel,
            score_column=DEFAULT_LONG_SCORE,
            count=candidate_count,
        ),
    }
    rows: list[dict[str, Any]] = []
    for scope, frame in scopes.items():
        for window, start, end in (("full", "", ""), *WINDOWS):
            subset = frame.copy()
            if window != "full":
                subset = subset[(subset["session_date"] >= start) & (subset["session_date"] <= end)].copy()
            rows.append(
                {
                    "scope": scope,
                    "window": window,
                    "start": start,
                    "end": end,
                    "rows": int(len(subset)),
                    "sessions": int(subset["session_date"].nunique()) if len(subset) else 0,
                    "mean_price_falling_knife_score": _safe_mean(
                        subset.get("price_falling_knife_score", pd.Series(dtype=float))
                    ),
                    "mean_price_still_falling_score": _safe_mean(
                        subset.get("price_still_falling_score", pd.Series(dtype=float))
                    ),
                    "mean_price_stabilization_score": _safe_mean(
                        subset.get("price_stabilization_score", pd.Series(dtype=float))
                    ),
                    "mean_stillfall_penalty": _safe_mean(
                        subset.get("price_fk_stillfall_penalty", pd.Series(dtype=float))
                    ),
                    "mean_stab_relief_penalty": _safe_mean(
                        subset.get("price_fk_stab_relief_penalty", pd.Series(dtype=float))
                    ),
                    "share_price_falling_knife_gt075": _safe_mean_bool(
                        subset.get("price_falling_knife_score", pd.Series(dtype=float)).ge(0.75)
                    ),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows)


def _position_price_state_exposure_summary(
    *,
    positions: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    score_columns = [
        "price_falling_knife_score",
        "price_still_falling_score",
        "price_stabilization_score",
        "price_fk_stillfall_penalty",
        "price_fk_stab_relief_penalty",
    ]
    factors = panel[["session_date", "symbol", *score_columns]].drop_duplicates(
        ["session_date", "symbol"]
    )
    longs = positions[positions["side"].eq("long")].copy()
    merged = longs.merge(factors, on=["session_date", "symbol"], how="left")
    for column in score_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    daily_rows: list[dict[str, Any]] = []
    for (portfolio, session_date), group in merged.groupby(["portfolio", "session_date"], sort=True):
        weights = group["side_weight"].astype(float)
        denom = float(weights.sum())
        if denom <= 0:
            continue
        row: dict[str, Any] = {"portfolio": str(portfolio), "session_date": session_date}
        for column in score_columns:
            row[f"weighted_{column}"] = float(np.dot(weights, group[column]) / denom)
        daily_rows.append(row)
    if not daily_rows:
        return pd.DataFrame()
    daily = pd.DataFrame(daily_rows)
    rows: list[dict[str, Any]] = []
    for portfolio, group in daily.groupby("portfolio", sort=True):
        row = {"portfolio": portfolio}
        for column in score_columns:
            row[f"mean_selected_long_{column}"] = float(group[f"weighted_{column}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _build_metrics(
    *,
    sweep: pd.DataFrame,
    strict_curve: pd.DataFrame,
    turnover_penalty: float,
) -> pd.DataFrame:
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
                "long_score": str(spec["long_score"]),
                "short_selector": str(spec["short_selector"]),
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
                "aggregate_gross_mean": float(item["aggregate_gross_mean"]),
                "mean_abs_net_beta": float(item["mean_abs_net_beta"]),
                "mean_selected_long_price_falling_knife_score": float(
                    item.get("mean_selected_long_price_falling_knife_score", np.nan)
                ),
                "mean_selected_long_price_still_falling_score": float(
                    item.get("mean_selected_long_price_still_falling_score", np.nan)
                ),
                "mean_selected_long_price_stabilization_score": float(
                    item.get("mean_selected_long_price_stabilization_score", np.nan)
                ),
                "mean_selected_long_price_fk_stillfall_penalty": float(
                    item.get("mean_selected_long_price_fk_stillfall_penalty", np.nan)
                ),
                "mean_selected_long_price_fk_stab_relief_penalty": float(
                    item.get("mean_selected_long_price_fk_stab_relief_penalty", np.nan)
                ),
                "turnover_penalty": float(turnover_penalty),
                "test_window_used": False,
            }
        )
    benchmark = strict_curve.sort_values(["return_date", "portfolio"]).drop_duplicates("return_date")
    spy_returns = benchmark["benchmark_oto_return"].astype(float)
    spy_equity = (1.0 + spy_returns).cumprod()
    spy_drawdown = spy_equity / spy_equity.cummax() - 1.0
    spy_rolling_60 = spy_equity / spy_equity.shift(60) - 1.0
    rows.append(
        {
            "series": "SPY raw",
            "portfolio": "SPY",
            "long_score": "",
            "short_selector": "",
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
            "aggregate_gross_mean": np.nan,
            "mean_abs_net_beta": np.nan,
            "mean_selected_long_price_falling_knife_score": np.nan,
            "mean_selected_long_price_still_falling_score": np.nan,
            "mean_selected_long_price_stabilization_score": np.nan,
            "mean_selected_long_price_fk_stillfall_penalty": np.nan,
            "mean_selected_long_price_fk_stab_relief_penalty": np.nan,
            "turnover_penalty": np.nan,
            "test_window_used": False,
        }
    )
    order = {str(spec["series"]): i for i, spec in enumerate(COMBO_SPECS)}
    order["SPY raw"] = len(order)
    frame = pd.DataFrame(rows)
    frame["sort_key"] = frame["series"].map(order)
    return frame.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)


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
        ax1.plot(dates, curve[f"{key}__equity"], label=label, color=color, linewidth=1.7)
        ax2.plot(dates, curve[f"{key}__drawdown"] * 100.0, color=color, linewidth=1.4)
        ax3.plot(dates, curve[f"{key}__rolling_60_return"] * 100.0, color=color, linewidth=1.4)
    ax1.plot(
        dates,
        curve["benchmark_equity"],
        label="SPY raw",
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.5,
    )
    ax2.plot(
        dates,
        curve["benchmark_drawdown"] * 100.0,
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.3,
    )
    ax3.plot(
        dates,
        curve["benchmark_rolling_60_return"] * 100.0,
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.3,
    )
    ax1.set_title("Phase5R Price-Only Stabilization-Aware Long Overlay")
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
    activation: pd.DataFrame,
    exposure: pd.DataFrame,
    windows: pd.DataFrame,
    plot_path: Path,
    turnover_penalty: float,
) -> str:
    metric_cols = [
        "series",
        "portfolio",
        "annualized_return",
        "annualized_vol",
        "sharpe_no_rf",
        "max_drawdown",
        "rolling_60_positive_rate",
        "corr_to_spy",
        "aggregate_turnover_mean",
        "mean_selected_long_price_falling_knife_score",
        "mean_selected_long_price_fk_stillfall_penalty",
        "mean_selected_long_price_fk_stab_relief_penalty",
    ]
    window_cols = ["series", "window", "compound_return", "hit_rate_daily"]
    exposure_cols = [
        "portfolio",
        "mean_selected_long_price_falling_knife_score",
        "mean_selected_long_price_still_falling_score",
        "mean_selected_long_price_stabilization_score",
        "mean_selected_long_price_fk_stillfall_penalty",
        "mean_selected_long_price_fk_stab_relief_penalty",
    ]
    lines = [
        "# Phase5R Price-Only Stabilization-Aware Long Overlay",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        f"- Short side fixed to `{DEFAULT_SHORT_OVERLAY}`.",
        f"- Flexible schedule fixed to turnover-aware optimizer with `lambda = {turnover_penalty:.4f}`.",
        "- No non-price inputs are loaded or used.",
        "- SIC labels are retained only for soft neutralization and sector-relative price-state grouping.",
        "- Stabilization is proxied using recent 5-session return and sector-relative recent return.",
        "- Validation window only; no test-window performance is computed.",
        "",
        "## Metrics",
        "",
        _text_table(_display_frame(metrics[metric_cols])),
        "",
        "## Price-State Coverage",
        "",
        _text_table(_display_frame(activation)),
        "",
        "## Selected Long Price-State Exposure",
        "",
        _text_table(_display_frame(exposure[exposure_cols] if not exposure.empty else exposure)),
        "",
        "## Stress Windows",
        "",
        _text_table(_display_frame(windows[window_cols])),
        "",
        "## Plot",
        "",
        f"![Phase5R comparison]({plot_path.as_posix()})",
        "",
        "## Read",
        "",
        "- This pass isolates price/state. Non-price fragility is intentionally absent.",
        "- `stillfall` variants penalize falling-knife names more when recent price action is still weak.",
        "- `stab relief` variants reduce that penalty when recent price action or sector-relative price action has stabilized.",
    ]
    return "\n".join(lines) + "\n"


def _top_by_session(frame: pd.DataFrame, *, score_column: str, count: int) -> pd.DataFrame:
    rows = []
    for _, group in frame.groupby("session_date", sort=True):
        rows.append(group.sort_values([score_column, "symbol"], ascending=[False, True]).head(count))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=frame.columns)


def _portfolio_to_series(portfolio: str) -> str:
    for spec in COMBO_SPECS:
        if spec["portfolio"] == portfolio:
            return str(spec["series"])
    return str(portfolio)


def _portfolio_to_long_score(portfolio: str) -> str:
    for spec in COMBO_SPECS:
        if spec["portfolio"] == portfolio:
            return str(spec["long_score"])
    return ""


def _portfolio_to_short_selector(portfolio: str) -> str:
    for spec in COMBO_SPECS:
        if spec["portfolio"] == portfolio:
            return str(spec["short_selector"])
    return ""


def _series_key(series: str) -> str:
    return str(series).replace(" ", "_").replace("-", "_")


def _long_price_score_columns() -> tuple[str, ...]:
    return (
        "long_price_fk_soft_w075",
        "long_price_fk_soft_w100",
        "long_price_fk_stillfall_w075",
        "long_price_fk_stillfall_w100",
        "long_price_fk_stab_relief_w075",
        "long_price_fk_stab_relief_w100",
    )


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    pct_columns = {
        "annualized_return",
        "annualized_vol",
        "max_drawdown",
        "rolling_60_positive_rate",
        "corr_to_spy",
        "compound_return",
        "hit_rate_daily",
        "share_price_falling_knife_gt075",
    }
    for column in result.columns:
        if column in pct_columns:
            result[column] = result[column].map(_fmt_pct_like)
        elif pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].map(_fmt_float_like)
    return result


def _mean_available(series_list: Sequence[pd.Series]) -> pd.Series:
    return pd.concat(series_list, axis=1).mean(axis=1)


def _zscore_array(values: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    std = float(np.nanstd(clean))
    if std <= 1e-12 or not np.isfinite(std):
        return np.zeros_like(clean)
    mean = float(np.nanmean(clean))
    return (clean - mean) / std


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float(pd.to_numeric(series, errors="coerce").mean())


def _safe_mean_bool(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float(series.astype("boolean").fillna(False).astype(float).mean())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase5R price-only stabilization overlay.")
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--strict-root", default=str(DEFAULT_STRICT_ROOT))
    parser.add_argument("--phase3-rollup-path", default=str(DEFAULT_PHASE3_H10_ROLLUP))
    args = parser.parse_args(argv)
    rollup = build_phase5r_price_only_stabilization_overlay_artifacts(
        signal_panel_path=args.signal_panel_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        output_root=args.output_root,
        strict_root=args.strict_root,
        phase3_rollup_path=args.phase3_rollup_path,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
