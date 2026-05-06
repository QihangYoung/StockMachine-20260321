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


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5o_non_price_long_veto_on_sota_20260505"
DEFAULT_STRICT_ROOT = DEFAULT_OUTPUT_ROOT / "strict_h10_rebuild"
DEFAULT_LONG_NON_PRICE_PANEL = (
    RESEARCH_ROOT
    / "non_price_phase1_top1000_two_factor_build_20260428"
    / "non_price_two_factor_panel.csv.gz"
)
DEFAULT_LONG_FUNDAMENTAL_PANEL = (
    RESEARCH_ROOT
    / "non_price_phase4_companyfacts_fundamentals_20260505"
    / "companyfacts_fundamental_panel.csv.gz"
)
DEFAULT_TURNOVER_PENALTY = 0.005
DEFAULT_TRIGGER_THRESHOLD = 0.75
DEFAULT_FUNDAMENTAL_TRIGGER_THRESHOLD = 0.75
DEFAULT_FUNDAMENTAL_SWEEP_THRESHOLDS = (0.65, 0.70, 0.75, 0.80, 0.85)
DEFAULT_RECENT_FILING_DAYS = 20
DEFAULT_SOFT_PENALTY_WEIGHT = 0.75
CURRENT_SOTA_SERIES = "current sota"

LONG_RAW_COLUMNS = (
    "filing_red_flag_events_20d",
    "red_8k_events_60d",
    "periodic_delay_events_252d",
    "days_since_last_filing_red_flag",
    "insider_sell_events_20d",
    "insider_sell_intensity_60d",
    "insider_net_buy_score",
)

COMBO_SPECS: tuple[dict[str, str], ...] = (
    {
        "portfolio": "current_sota",
        "series": CURRENT_SOTA_SERIES,
        "long_score": DEFAULT_LONG_SCORE,
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_filing_veto_t075",
        "series": "np filing veto t075",
        "long_score": "long_np_filing_veto_t075",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_insider_sell_veto_t075",
        "series": "np insider sell veto t075",
        "long_score": "long_np_insider_sell_veto_t075",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_fund_fragility_veto_t075",
        "series": "np fund fragility veto t075",
        "long_score": "long_np_fund_fragility_veto_t075",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_filing_and_fund_veto_t075",
        "series": "np filing and fund veto t075",
        "long_score": "long_np_filing_and_fund_veto_t075",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_filing_or_fund_veto_t075",
        "series": "np filing or fund veto t075",
        "long_score": "long_np_filing_or_fund_veto_t075",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_filing_or_fund_veto_t065",
        "series": "np filing or fund veto t065",
        "long_score": "long_np_filing_or_fund_veto_t065",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_filing_or_fund_veto_t070",
        "series": "np filing or fund veto t070",
        "long_score": "long_np_filing_or_fund_veto_t070",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_filing_or_fund_veto_t080",
        "series": "np filing or fund veto t080",
        "long_score": "long_np_filing_or_fund_veto_t080",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_filing_or_fund_veto_t085",
        "series": "np filing or fund veto t085",
        "long_score": "long_np_filing_or_fund_veto_t085",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_fof_low_mom20_veto_t075",
        "series": "np fof low mom20 veto t075",
        "long_score": "long_np_fof_low_mom20_veto_t075",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_fof_low_mom60_veto_t075",
        "series": "np fof low mom60 veto t075",
        "long_score": "long_np_fof_low_mom60_veto_t075",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_fof_low_beta_veto_t075",
        "series": "np fof low beta veto t075",
        "long_score": "long_np_fof_low_beta_veto_t075",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_fof_low_mom60_low_beta_veto_t075",
        "series": "np fof low mom60 low beta veto t075",
        "long_score": "long_np_fof_low_mom60_low_beta_veto_t075",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_any_veto_t075",
        "series": "np any veto t075",
        "long_score": "long_np_any_promising_veto_t075",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_top70_any_veto_t075",
        "series": "np top70 any veto t075",
        "long_score": "long_np_top70_any_promising_veto_t075",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_soft_promising",
        "series": "np soft promising",
        "long_score": "long_np_promising_soft_overlay",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "np_soft_promising_plus_fund",
        "series": "np soft promising plus fund",
        "long_score": "long_np_promising_plus_fund_soft_overlay",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
)

PLOT_COLORS = {
    CURRENT_SOTA_SERIES: "#2563eb",
    "np filing veto t075": "#0f766e",
    "np insider sell veto t075": "#c2410c",
    "np fund fragility veto t075": "#9333ea",
    "np filing and fund veto t075": "#16a34a",
    "np filing or fund veto t075": "#dc2626",
    "np filing or fund veto t065": "#f97316",
    "np filing or fund veto t070": "#ef4444",
    "np filing or fund veto t080": "#b91c1c",
    "np filing or fund veto t085": "#7f1d1d",
    "np fof low mom20 veto t075": "#f59e0b",
    "np fof low mom60 veto t075": "#d97706",
    "np fof low beta veto t075": "#65a30d",
    "np fof low mom60 low beta veto t075": "#4d7c0f",
    "np any veto t075": "#7c3aed",
    "np top70 any veto t075": "#b45309",
    "np soft promising": "#0891b2",
    "np soft promising plus fund": "#0e7490",
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


def build_phase5o_non_price_long_veto_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    long_non_price_panel_path: str | Path = DEFAULT_LONG_NON_PRICE_PANEL,
    long_fundamental_panel_path: str | Path = DEFAULT_LONG_FUNDAMENTAL_PANEL,
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
    trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD,
    fundamental_trigger_threshold: float = DEFAULT_FUNDAMENTAL_TRIGGER_THRESHOLD,
    recent_filing_days: int = DEFAULT_RECENT_FILING_DAYS,
    soft_penalty_weight: float = DEFAULT_SOFT_PENALTY_WEIGHT,
) -> dict[str, Any]:
    """Backtest non-price long veto factors on current SOTA."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    industry_map = _load_sec_sic_map(cik_mapping_path, sec_submissions_dir)
    panel = _load_panel(signal_panel_path, long_variant, short_variant)
    panel = _add_former_winner_scores(panel)
    panel = _add_hybrid_scores(panel)
    panel = _add_non_price_long_scores(
        panel,
        long_non_price_panel_path=long_non_price_panel_path,
        long_fundamental_panel_path=long_fundamental_panel_path,
        long_variant=long_variant,
        trigger_threshold=trigger_threshold,
        fundamental_trigger_threshold=fundamental_trigger_threshold,
        recent_filing_days=recent_filing_days,
        soft_penalty_weight=soft_penalty_weight,
    )
    activation = _activation_summary(
        panel,
        long_variant=long_variant,
        candidate_count=candidate_pool_per_side,
    )

    panel = panel.merge(industry_map, on="symbol", how="left")
    panel["sic2_sector"] = panel["sic2_sector"].fillna("UNKNOWN")
    panel["sic4_industry"] = panel["sic4_industry"].fillna("UNKNOWN")
    panel["sic_description"] = panel["sic_description"].fillna("UNKNOWN")
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

    positions_path = output_dir / "phase5o_positions_validation.csv.gz"
    diagnostics_path = output_dir / "phase5o_daily_validation.csv"
    skipped_path = output_dir / "phase5o_skipped_validation.csv"
    activation_path = output_dir / "phase5o_non_price_activation_summary.csv"
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

    benchmark_calendar = (
        strict_curve["return_date"].drop_duplicates().sort_values().reset_index(drop=True)
    )
    target_turnover = _target_turnover_summary(positions)
    aggregate_turnover = _aggregate_turnover_summary(
        positions_path=positions_path,
        benchmark_returns=benchmark_calendar.to_frame(name="session_date"),
        holding_period_sessions=holding_period_sessions,
    )
    strict_metrics = _strict_metric_summary(strict_curve)
    diagnostic_summary = _diagnostic_summary(diagnostics)
    exposure = _position_non_price_exposure_summary(positions=positions, panel=panel)

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

    sweep_path = output_dir / "phase5o_sweep.csv"
    metrics_path = output_dir / "phase5o_metrics.csv"
    windows_path = output_dir / "phase5o_window_returns.csv"
    exposure_path = output_dir / "phase5o_position_non_price_exposure.csv"
    comparison_curve_path = output_dir / "phase5o_comparison_curve.csv"
    plot_path = output_dir / "phase5o_comparison_plot.png"
    memo_path = output_dir / "phase5o_non_price_long_veto_memo.md"
    rollup_path = output_dir / "phase5o_rollup.json"

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
            trigger_threshold=trigger_threshold,
            fundamental_trigger_threshold=fundamental_trigger_threshold,
            turnover_penalty=turnover_penalty,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "long_non_price_panel_path": Path(long_non_price_panel_path).as_posix(),
        "long_fundamental_panel_path": Path(long_fundamental_panel_path).as_posix(),
        "cik_mapping_path": Path(cik_mapping_path).as_posix(),
        "sec_submissions_dir": Path(sec_submissions_dir).as_posix(),
        "long_variant": long_variant,
        "short_variant": short_variant,
        "candidate_pool_per_side": int(candidate_pool_per_side),
        "trigger_threshold": float(trigger_threshold),
        "fundamental_trigger_threshold": float(fundamental_trigger_threshold),
        "recent_filing_days": int(recent_filing_days),
        "soft_penalty_weight": float(soft_penalty_weight),
        "turnover_penalty": float(turnover_penalty),
        "positions_rows": int(len(positions)),
        "diagnostic_rows": int(len(diagnostics)),
        "skipped_rows": int(len(skipped)),
        "strict_rollup": strict_rollup,
        "artifacts": {
            "positions": positions_path.as_posix(),
            "diagnostics": diagnostics_path.as_posix(),
            "skipped": skipped_path.as_posix(),
            "activation": activation_path.as_posix(),
            "sweep": sweep_path.as_posix(),
            "metrics": metrics_path.as_posix(),
            "window_returns": windows_path.as_posix(),
            "position_non_price_exposure": exposure_path.as_posix(),
            "comparison_curve": comparison_curve_path.as_posix(),
            "plot": plot_path.as_posix(),
            "memo": memo_path.as_posix(),
            "strict_curve": strict_curve_path.as_posix(),
        },
        "method": "current_sota_validation_backtest_with_phase5n_and_companyfacts_long_vetoes",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "This is a validation-only alpha integration smoke test; no test-window performance is computed.",
            "Phase5N filing/Form4 factors and Phase4 SEC companyfacts fundamental fragility factors are used.",
            "The short selector is fixed to the current SOTA short overlay.",
            "The non-price features inherit SEC filing, Form 4, and companyfacts point-in-time alignment assumptions from their source phases.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _add_non_price_long_scores(
    panel: pd.DataFrame,
    *,
    long_non_price_panel_path: str | Path,
    long_fundamental_panel_path: str | Path | None = None,
    long_variant: str,
    trigger_threshold: float,
    fundamental_trigger_threshold: float = DEFAULT_FUNDAMENTAL_TRIGGER_THRESHOLD,
    recent_filing_days: int,
    soft_penalty_weight: float,
) -> pd.DataFrame:
    non_price = _load_long_non_price_percentiles(Path(long_non_price_panel_path))
    frame = panel.merge(non_price, on=["session_date", "symbol"], how="left")
    if long_fundamental_panel_path is not None:
        fundamentals = _load_long_fundamental_scores(Path(long_fundamental_panel_path))
        frame = frame.merge(fundamentals, on=["session_date", "symbol"], how="left")

    raw_fill = {
        "filing_red_flag_events_20d": 0.0,
        "red_8k_events_60d": 0.0,
        "periodic_delay_events_252d": 0.0,
        "days_since_last_filing_red_flag": 9999.0,
        "insider_sell_events_20d": 0.0,
        "insider_sell_intensity_60d": 0.0,
        "insider_net_buy_score": 0.0,
    }
    for column, fill_value in raw_fill.items():
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(fill_value)
    pct_columns = [
        "filing_red_flag_events_20d_pct",
        "red_8k_events_60d_pct",
        "periodic_delay_events_252d_pct",
        "days_since_last_filing_red_flag_inverse_pct",
        "insider_sell_events_20d_pct",
        "insider_sell_intensity_60d_pct",
        "insider_net_buy_score_inverse_pct",
    ]
    for column in pct_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in ("momentum_20d", "momentum_60d", "beta"):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[f"{column}_pct_local"] = frame.groupby(["variant", "session_date"])[column].rank(
            method="average",
            pct=True,
        )
    fundamental_score_columns = (
        "long_fundamental_leverage_pressure_score",
        "long_fundamental_profit_stress_score",
        "long_fundamental_fragility_score",
    )
    for column in fundamental_score_columns:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "has_long_fundamental_fragility_score" not in frame.columns:
        frame["has_long_fundamental_fragility_score"] = False
    frame["has_long_fundamental_fragility_score"] = (
        frame["has_long_fundamental_fragility_score"].astype("boolean").fillna(False).astype(bool)
    )

    frame["recent_filing_red_flag"] = (
        frame["days_since_last_filing_red_flag"].le(float(recent_filing_days))
        & frame["filing_red_flag_events_20d"].gt(0.0)
    ).astype(float)
    frame["long_filing_distress_score"] = _mean_available(
        [
            frame["filing_red_flag_events_20d_pct"],
            frame["red_8k_events_60d_pct"],
            frame["periodic_delay_events_252d_pct"],
            frame["days_since_last_filing_red_flag_inverse_pct"],
            frame["recent_filing_red_flag"],
        ]
    )
    frame["long_insider_sell_pressure_score"] = _mean_available(
        [
            frame["insider_sell_events_20d_pct"],
            frame["insider_sell_intensity_60d_pct"],
            frame["insider_net_buy_score_inverse_pct"],
        ]
    )
    frame["long_np_promising_score"] = _mean_available(
        [
            frame["long_filing_distress_score"],
            frame["long_insider_sell_pressure_score"],
        ]
    )
    frame["long_np_plus_fund_score"] = _mean_available(
        [
            frame["long_np_promising_score"],
            frame["long_fundamental_fragility_score"],
        ]
    )
    fundamental_sweep_thresholds = _fundamental_sweep_thresholds(fundamental_trigger_threshold)
    frame["trigger_long_filing_distress"] = frame["long_filing_distress_score"].ge(
        trigger_threshold
    )
    frame["trigger_long_insider_sell_pressure"] = frame["long_insider_sell_pressure_score"].ge(
        trigger_threshold
    )
    frame["trigger_long_np_any_promising"] = (
        frame["trigger_long_filing_distress"] | frame["trigger_long_insider_sell_pressure"]
    )
    for threshold in fundamental_sweep_thresholds:
        suffix = _threshold_suffix(threshold)
        trigger = (
            frame["has_long_fundamental_fragility_score"]
            & frame["long_fundamental_fragility_score"].ge(threshold)
        )
        frame[f"trigger_long_fundamental_fragility_{suffix}"] = trigger
        frame[f"trigger_long_filing_and_fundamental_{suffix}"] = (
            frame["trigger_long_filing_distress"] & trigger
        )
        frame[f"trigger_long_filing_or_fundamental_{suffix}"] = (
            frame["trigger_long_filing_distress"] | trigger
        )
    default_suffix = _threshold_suffix(fundamental_trigger_threshold)
    frame["trigger_long_fundamental_fragility"] = frame[
        f"trigger_long_fundamental_fragility_{default_suffix}"
    ]
    frame["trigger_long_filing_and_fundamental"] = frame[
        f"trigger_long_filing_and_fundamental_{default_suffix}"
    ]
    frame["trigger_long_filing_or_fundamental"] = frame[
        f"trigger_long_filing_or_fundamental_{default_suffix}"
    ]
    fundamental_veto_columns = [
        f"long_np_fund_fragility_veto_{_threshold_suffix(threshold)}"
        for threshold in fundamental_sweep_thresholds
    ] + [
        f"long_np_filing_and_fund_veto_{_threshold_suffix(threshold)}"
        for threshold in fundamental_sweep_thresholds
    ] + [
        f"long_np_filing_or_fund_veto_{_threshold_suffix(threshold)}"
        for threshold in fundamental_sweep_thresholds
    ]

    frames = []
    for (variant, session_date), group in frame.groupby(["variant", "session_date"], sort=False):
        part = group.copy()
        baseline = part[DEFAULT_LONG_SCORE].astype(float)
        is_long_variant = str(variant) == str(long_variant)
        if is_long_variant:
            top70_score = baseline >= baseline.quantile(0.70)
            part["long_np_filing_veto_t075"] = baseline.where(
                ~part["trigger_long_filing_distress"].astype(bool)
            )
            part["long_np_insider_sell_veto_t075"] = baseline.where(
                ~part["trigger_long_insider_sell_pressure"].astype(bool)
            )
            for threshold in fundamental_sweep_thresholds:
                suffix = _threshold_suffix(threshold)
                part[f"long_np_fund_fragility_veto_{suffix}"] = baseline.where(
                    ~part[f"trigger_long_fundamental_fragility_{suffix}"].astype(bool)
                )
                part[f"long_np_filing_and_fund_veto_{suffix}"] = baseline.where(
                    ~part[f"trigger_long_filing_and_fundamental_{suffix}"].astype(bool)
                )
                part[f"long_np_filing_or_fund_veto_{suffix}"] = baseline.where(
                    ~part[f"trigger_long_filing_or_fundamental_{suffix}"].astype(bool)
                )
            default_fof_trigger = part[
                f"trigger_long_filing_or_fundamental_{default_suffix}"
            ].astype(bool)
            low_mom20 = part["momentum_20d_pct_local"].le(0.50)
            low_mom60 = part["momentum_60d_pct_local"].le(0.50)
            low_beta = part["beta"].lt(1.20)
            part["long_np_fof_low_mom20_veto_t075"] = baseline.where(
                ~(default_fof_trigger & low_mom20)
            )
            part["long_np_fof_low_mom60_veto_t075"] = baseline.where(
                ~(default_fof_trigger & low_mom60)
            )
            part["long_np_fof_low_beta_veto_t075"] = baseline.where(
                ~(default_fof_trigger & low_beta)
            )
            part["long_np_fof_low_mom60_low_beta_veto_t075"] = baseline.where(
                ~(default_fof_trigger & low_mom60 & low_beta)
            )
            part["long_np_any_promising_veto_t075"] = baseline.where(
                ~part["trigger_long_np_any_promising"].astype(bool)
            )
            part["long_np_top70_any_promising_veto_t075"] = baseline.where(
                ~(top70_score & part["trigger_long_np_any_promising"].astype(bool))
            )
            soft_raw = _zscore_array(baseline.to_numpy(dtype=float)) - soft_penalty_weight * part[
                "long_np_promising_score"
            ].to_numpy(dtype=float)
            part["long_np_promising_soft_overlay"] = _zscore_array(soft_raw)
            soft_plus_fund_raw = _zscore_array(
                baseline.to_numpy(dtype=float)
            ) - soft_penalty_weight * part["long_np_plus_fund_score"].to_numpy(dtype=float)
            part["long_np_promising_plus_fund_soft_overlay"] = _zscore_array(soft_plus_fund_raw)
        else:
            for column in (
                "long_np_filing_veto_t075",
                "long_np_insider_sell_veto_t075",
                "long_np_any_promising_veto_t075",
                "long_np_top70_any_promising_veto_t075",
                "long_np_promising_soft_overlay",
                "long_np_promising_plus_fund_soft_overlay",
                "long_np_fof_low_mom20_veto_t075",
                "long_np_fof_low_mom60_veto_t075",
                "long_np_fof_low_beta_veto_t075",
                "long_np_fof_low_mom60_low_beta_veto_t075",
                *fundamental_veto_columns,
            ):
                part[column] = np.nan
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else frame


def _load_long_non_price_percentiles(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Non-price panel not found: {path}")
    usecols = ["session_date", "symbol", *LONG_RAW_COLUMNS]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    for column in LONG_RAW_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        frame[f"{column}_pct"] = frame.groupby("session_date")[column].rank(
            method="average",
            pct=True,
        )
    for column in ("days_since_last_filing_red_flag", "insider_net_buy_score"):
        inverse_name = f"{column}_inverse_raw"
        pct_name = f"{column}_inverse_pct"
        frame[inverse_name] = -frame[column]
        frame[pct_name] = frame.groupby("session_date")[inverse_name].rank(
            method="average",
            pct=True,
        )
    return frame


def _load_long_fundamental_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fundamental panel not found: {path}")
    usecols = [
        "session_date",
        "symbol",
        "fundamental_leverage_pressure_score",
        "fundamental_profit_stress_score",
        "fundamental_fragility_score",
    ]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame.rename(
        columns={
            "fundamental_leverage_pressure_score": "long_fundamental_leverage_pressure_score",
            "fundamental_profit_stress_score": "long_fundamental_profit_stress_score",
            "fundamental_fragility_score": "long_fundamental_fragility_score",
        }
    )
    score_columns = [
        "long_fundamental_leverage_pressure_score",
        "long_fundamental_profit_stress_score",
        "long_fundamental_fragility_score",
    ]
    for column in score_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").clip(0.0, 1.0)
    frame["has_long_fundamental_fragility_score"] = frame[
        "long_fundamental_fragility_score"
    ].notna()
    return frame.drop_duplicates(["session_date", "symbol"], keep="last").reset_index(drop=True)


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
    rows: list[dict[str, Any]] = []
    scopes = {
        "long_universe": long_panel,
        f"baseline_top{candidate_count}": _top_by_session(
            long_panel,
            score_column=DEFAULT_LONG_SCORE,
            count=candidate_count,
        ),
    }
    for scope, frame in scopes.items():
        for window, start, end in (("full", "", ""), *WINDOWS):
            subset = frame.copy()
            if window != "full":
                subset = subset[
                    (subset["session_date"] >= start) & (subset["session_date"] <= end)
                ].copy()
            rows.append(
                {
                    "scope": scope,
                    "window": window,
                    "start": start,
                    "end": end,
                    "rows": int(len(subset)),
                    "sessions": int(subset["session_date"].nunique()) if len(subset) else 0,
                    "filing_trigger_share": _safe_mean_bool(
                        subset.get("trigger_long_filing_distress", pd.Series(dtype=bool))
                    ),
                    "insider_sell_trigger_share": _safe_mean_bool(
                        subset.get("trigger_long_insider_sell_pressure", pd.Series(dtype=bool))
                    ),
                    "fundamental_fragility_trigger_share": _safe_mean_bool(
                        subset.get("trigger_long_fundamental_fragility", pd.Series(dtype=bool))
                    ),
                    "filing_and_fundamental_trigger_share": _safe_mean_bool(
                        subset.get("trigger_long_filing_and_fundamental", pd.Series(dtype=bool))
                    ),
                    "filing_or_fundamental_trigger_share": _safe_mean_bool(
                        subset.get("trigger_long_filing_or_fundamental", pd.Series(dtype=bool))
                    ),
                    "any_promising_trigger_share": _safe_mean_bool(
                        subset.get("trigger_long_np_any_promising", pd.Series(dtype=bool))
                    ),
                    "mean_filing_distress_score": _safe_mean(
                        subset.get("long_filing_distress_score", pd.Series(dtype=float))
                    ),
                    "mean_insider_sell_pressure_score": _safe_mean(
                        subset.get("long_insider_sell_pressure_score", pd.Series(dtype=float))
                    ),
                    "mean_fundamental_fragility_score": _safe_mean(
                        subset.get("long_fundamental_fragility_score", pd.Series(dtype=float))
                    ),
                    "mean_np_promising_score": _safe_mean(
                        subset.get("long_np_promising_score", pd.Series(dtype=float))
                    ),
                    "mean_np_plus_fund_score": _safe_mean(
                        subset.get("long_np_plus_fund_score", pd.Series(dtype=float))
                    ),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows)


def _position_non_price_exposure_summary(
    *,
    positions: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame()
    score_columns = [
        "long_filing_distress_score",
        "long_insider_sell_pressure_score",
        "long_fundamental_fragility_score",
        "long_np_promising_score",
        "long_np_plus_fund_score",
    ]
    trigger_columns = [
        "trigger_long_filing_distress",
        "trigger_long_insider_sell_pressure",
        "trigger_long_fundamental_fragility",
        "trigger_long_filing_and_fundamental",
        "trigger_long_filing_or_fundamental",
        "trigger_long_np_any_promising",
    ]
    factor_source = panel.copy()
    for column in score_columns:
        if column not in factor_source.columns:
            factor_source[column] = 0.0
    for column in trigger_columns:
        if column not in factor_source.columns:
            factor_source[column] = False
    factors = factor_source[
        ["session_date", "symbol", *score_columns, *trigger_columns]
    ].drop_duplicates(["session_date", "symbol"])
    long_positions = positions[positions["side"].eq("long")].copy()
    merged = long_positions.merge(factors, on=["session_date", "symbol"], how="left")
    for column in score_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    for column in trigger_columns:
        merged[column] = merged[column].fillna(False).astype(bool).astype(float)

    daily_rows: list[dict[str, Any]] = []
    for (portfolio, session_date), group in merged.groupby(["portfolio", "session_date"], sort=True):
        weights = group["side_weight"].astype(float)
        denom = float(weights.sum())
        if denom <= 0:
            continue
        daily_rows.append(
            {
                "portfolio": str(portfolio),
                "session_date": session_date,
                "filing_trigger_weight_share": float(
                    np.dot(weights, group["trigger_long_filing_distress"]) / denom
                ),
                "insider_sell_trigger_weight_share": float(
                    np.dot(weights, group["trigger_long_insider_sell_pressure"]) / denom
                ),
                "fundamental_fragility_trigger_weight_share": float(
                    np.dot(weights, group["trigger_long_fundamental_fragility"]) / denom
                ),
                "filing_and_fundamental_trigger_weight_share": float(
                    np.dot(weights, group["trigger_long_filing_and_fundamental"]) / denom
                ),
                "filing_or_fundamental_trigger_weight_share": float(
                    np.dot(weights, group["trigger_long_filing_or_fundamental"]) / denom
                ),
                "any_promising_trigger_weight_share": float(
                    np.dot(weights, group["trigger_long_np_any_promising"]) / denom
                ),
                "weighted_filing_distress_score": float(
                    np.dot(weights, group["long_filing_distress_score"]) / denom
                ),
                "weighted_insider_sell_pressure_score": float(
                    np.dot(weights, group["long_insider_sell_pressure_score"]) / denom
                ),
                "weighted_fundamental_fragility_score": float(
                    np.dot(weights, group["long_fundamental_fragility_score"]) / denom
                ),
                "weighted_np_promising_score": float(
                    np.dot(weights, group["long_np_promising_score"]) / denom
                ),
                "weighted_np_plus_fund_score": float(
                    np.dot(weights, group["long_np_plus_fund_score"]) / denom
                ),
            }
        )
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return pd.DataFrame()
    return (
        daily.groupby("portfolio", as_index=False)
        .agg(
            mean_selected_long_filing_trigger_weight_share=(
                "filing_trigger_weight_share",
                "mean",
            ),
            mean_selected_long_insider_sell_trigger_weight_share=(
                "insider_sell_trigger_weight_share",
                "mean",
            ),
            mean_selected_long_fundamental_fragility_trigger_weight_share=(
                "fundamental_fragility_trigger_weight_share",
                "mean",
            ),
            mean_selected_long_filing_and_fund_trigger_weight_share=(
                "filing_and_fundamental_trigger_weight_share",
                "mean",
            ),
            mean_selected_long_filing_or_fund_trigger_weight_share=(
                "filing_or_fundamental_trigger_weight_share",
                "mean",
            ),
            mean_selected_long_any_trigger_weight_share=(
                "any_promising_trigger_weight_share",
                "mean",
            ),
            mean_selected_long_filing_distress_score=(
                "weighted_filing_distress_score",
                "mean",
            ),
            mean_selected_long_insider_sell_pressure_score=(
                "weighted_insider_sell_pressure_score",
                "mean",
            ),
            mean_selected_long_fundamental_fragility_score=(
                "weighted_fundamental_fragility_score",
                "mean",
            ),
            mean_selected_long_np_promising_score=("weighted_np_promising_score", "mean"),
            mean_selected_long_np_plus_fund_score=("weighted_np_plus_fund_score", "mean"),
        )
        .sort_values("portfolio")
        .reset_index(drop=True)
    )


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


def _series_key(series: str) -> str:
    return str(series).replace(" ", "_").replace("-", "_")


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


def _build_metrics(
    *,
    sweep: pd.DataFrame,
    strict_curve: pd.DataFrame,
    turnover_penalty: float,
) -> pd.DataFrame:
    ordered = [_portfolio_to_series(str(spec["portfolio"])) for spec in COMBO_SPECS]
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
                "mean_long_count": float(item["mean_long_count"]),
                "mean_short_count": float(item["mean_short_count"]),
                "mean_selected_long_any_trigger_weight_share": float(
                    item.get("mean_selected_long_any_trigger_weight_share", np.nan)
                ),
                "mean_selected_long_filing_trigger_weight_share": float(
                    item.get("mean_selected_long_filing_trigger_weight_share", np.nan)
                ),
                "mean_selected_long_insider_sell_trigger_weight_share": float(
                    item.get("mean_selected_long_insider_sell_trigger_weight_share", np.nan)
                ),
                "mean_selected_long_fundamental_fragility_trigger_weight_share": float(
                    item.get(
                        "mean_selected_long_fundamental_fragility_trigger_weight_share",
                        np.nan,
                    )
                ),
                "mean_selected_long_filing_and_fund_trigger_weight_share": float(
                    item.get("mean_selected_long_filing_and_fund_trigger_weight_share", np.nan)
                ),
                "mean_selected_long_filing_or_fund_trigger_weight_share": float(
                    item.get("mean_selected_long_filing_or_fund_trigger_weight_share", np.nan)
                ),
                "mean_selected_long_np_promising_score": float(
                    item.get("mean_selected_long_np_promising_score", np.nan)
                ),
                "mean_selected_long_fundamental_fragility_score": float(
                    item.get("mean_selected_long_fundamental_fragility_score", np.nan)
                ),
                "mean_selected_long_np_plus_fund_score": float(
                    item.get("mean_selected_long_np_plus_fund_score", np.nan)
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
            "mean_long_count": np.nan,
            "mean_short_count": np.nan,
            "mean_selected_long_any_trigger_weight_share": np.nan,
            "mean_selected_long_filing_trigger_weight_share": np.nan,
            "mean_selected_long_insider_sell_trigger_weight_share": np.nan,
            "mean_selected_long_fundamental_fragility_trigger_weight_share": np.nan,
            "mean_selected_long_filing_and_fund_trigger_weight_share": np.nan,
            "mean_selected_long_filing_or_fund_trigger_weight_share": np.nan,
            "mean_selected_long_np_promising_score": np.nan,
            "mean_selected_long_fundamental_fragility_score": np.nan,
            "mean_selected_long_np_plus_fund_score": np.nan,
            "turnover_penalty": np.nan,
            "test_window_used": False,
        }
    )
    frame = pd.DataFrame(rows)
    frame["sort_key"] = frame["series"].map(
        {name: i for i, name in enumerate([*ordered, "SPY raw"])}
    )
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
            compound = float((1.0 + subset["gross_return"].astype(float)).prod() - 1.0)
            rows.append(
                {
                    "portfolio": portfolio,
                    "series": _portfolio_to_series(portfolio),
                    "window": window,
                    "start": start,
                    "end": end,
                    "compound_return": compound,
                    "hit_rate_daily": float((subset["gross_return"].astype(float) > 0).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["window", "portfolio"]).reset_index(drop=True)


def _build_comparison_curve(strict_curve: pd.DataFrame) -> pd.DataFrame:
    benchmark = (
        strict_curve.sort_values(["return_date", "portfolio"])
        .drop_duplicates("return_date")[
            [
                "return_date",
                "benchmark_oto_return",
                "benchmark_equity",
                "benchmark_drawdown",
            ]
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
        ax1.plot(dates, curve[f"{key}__equity"], label=label, color=color, linewidth=1.8)
        ax2.plot(dates, curve[f"{key}__drawdown"] * 100.0, color=color, linewidth=1.5)
        ax3.plot(
            dates,
            curve[f"{key}__rolling_60_return"] * 100.0,
            color=color,
            linewidth=1.5,
        )

    ax1.plot(
        dates,
        curve["benchmark_equity"],
        label="SPY raw",
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.7,
    )
    ax2.plot(
        dates,
        curve["benchmark_drawdown"] * 100.0,
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.4,
    )
    ax3.plot(
        dates,
        curve["benchmark_rolling_60_return"] * 100.0,
        color=PLOT_COLORS["SPY raw"],
        linestyle="--",
        linewidth=1.4,
    )
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax1.set_ylabel("Growth of $1")
    ax2.set_ylabel("Drawdown %")
    ax3.set_ylabel("Rolling 60d return %")
    ax1.legend(loc="upper left", ncols=2)
    ax1.set_title("Phase5O Non-Price Long Veto Variants vs Current SOTA")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _memo(
    *,
    metrics: pd.DataFrame,
    activation: pd.DataFrame,
    exposure: pd.DataFrame,
    windows: pd.DataFrame,
    plot_path: Path,
    trigger_threshold: float,
    fundamental_trigger_threshold: float,
    turnover_penalty: float,
) -> str:
    display_metrics = metrics.copy()
    for column in (
        "annualized_return",
        "annualized_vol",
        "max_drawdown",
        "rolling_60_positive_rate",
        "mean_selected_long_any_trigger_weight_share",
        "mean_selected_long_filing_trigger_weight_share",
        "mean_selected_long_insider_sell_trigger_weight_share",
        "mean_selected_long_fundamental_fragility_trigger_weight_share",
        "mean_selected_long_filing_and_fund_trigger_weight_share",
        "mean_selected_long_filing_or_fund_trigger_weight_share",
    ):
        if column in display_metrics.columns:
            display_metrics[column] = display_metrics[column].map(_fmt_pct_like)
    for column in (
        "sharpe_no_rf",
        "corr_to_spy",
        "mean_daily_return_bps",
        "target_turnover_mean",
        "aggregate_turnover_mean",
        "aggregate_positions_mean",
        "aggregate_gross_mean",
        "mean_abs_net_beta",
        "mean_long_count",
        "mean_short_count",
        "mean_selected_long_np_promising_score",
        "mean_selected_long_fundamental_fragility_score",
        "mean_selected_long_np_plus_fund_score",
        "turnover_penalty",
    ):
        if column in display_metrics.columns:
            display_metrics[column] = display_metrics[column].map(_fmt_float_like)

    display_activation = activation.copy()
    for column in (
        "filing_trigger_share",
        "insider_sell_trigger_share",
        "fundamental_fragility_trigger_share",
        "filing_and_fundamental_trigger_share",
        "filing_or_fundamental_trigger_share",
        "any_promising_trigger_share",
    ):
        if column in display_activation.columns:
            display_activation[column] = display_activation[column].map(_fmt_pct_like)
    for column in (
        "mean_filing_distress_score",
        "mean_insider_sell_pressure_score",
        "mean_fundamental_fragility_score",
        "mean_np_promising_score",
        "mean_np_plus_fund_score",
    ):
        if column in display_activation.columns:
            display_activation[column] = display_activation[column].map(_fmt_float_like)

    display_exposure = exposure.copy()
    for column in (
        "mean_selected_long_filing_trigger_weight_share",
        "mean_selected_long_insider_sell_trigger_weight_share",
        "mean_selected_long_fundamental_fragility_trigger_weight_share",
        "mean_selected_long_filing_and_fund_trigger_weight_share",
        "mean_selected_long_filing_or_fund_trigger_weight_share",
        "mean_selected_long_any_trigger_weight_share",
    ):
        if column in display_exposure.columns:
            display_exposure[column] = display_exposure[column].map(_fmt_pct_like)
    for column in (
        "mean_selected_long_filing_distress_score",
        "mean_selected_long_insider_sell_pressure_score",
        "mean_selected_long_fundamental_fragility_score",
        "mean_selected_long_np_promising_score",
        "mean_selected_long_np_plus_fund_score",
    ):
        if column in display_exposure.columns:
            display_exposure[column] = display_exposure[column].map(_fmt_float_like)

    display_windows = windows.copy()
    if not display_windows.empty:
        display_windows["compound_return"] = display_windows["compound_return"].map(_fmt_pct_like)
        display_windows["hit_rate_daily"] = display_windows["hit_rate_daily"].map(_fmt_pct_like)

    activation_short = display_activation[
        display_activation["window"].isin(["full", "2015_peak_to_trough", "2016_jan_feb", "2019_may", "2019_aug"])
    ].copy()

    lines = [
        "# Phase5O Non-Price Long Veto on Current SOTA",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        f"- Trigger threshold: `{trigger_threshold:.2f}`.",
        f"- Fundamental fragility trigger threshold: `{fundamental_trigger_threshold:.2f}`.",
        f"- Short side is fixed to `{DEFAULT_SHORT_OVERLAY}`.",
        f"- Flexible schedule stays fixed: turnover-aware daily optimizer with `lambda = {turnover_penalty:.4f}`.",
        "- Tested long-side variants: filing distress, insider sell pressure, companyfacts fundamental fragility, filing/fundamental intersections, combined vetoes, and soft penalties.",
        "- Validation window only; no test-window performance is computed.",
        "",
        "## Metrics",
        "",
        _text_table(display_metrics),
        "",
        "## Non-Price Trigger Coverage",
        "",
        _text_table(activation_short),
        "",
        "## Selected Long Exposure",
        "",
        _text_table(display_exposure),
        "",
        "## Stress Windows",
        "",
        _text_table(display_windows),
        "",
        "## Plot",
        "",
        f"![Phase5O comparison]({plot_path.as_posix()})",
        "",
        "## Read",
        "",
        "- This pass answers whether the promising Phase5N long-side factors survive a real current-SOTA portfolio construction loop.",
        "- A useful variant should reduce selected long exposure to high non-price distress without simply buying lower volatility at the cost of all full-sample return.",
        "- The result is a handoff candidate for alpha governance, not a production promotion.",
        "",
    ]
    return "\n".join(lines)


def _mean_available(series_list: Sequence[pd.Series]) -> pd.Series:
    return pd.concat(series_list, axis=1).mean(axis=1, skipna=True)


def _fundamental_sweep_thresholds(extra_threshold: float) -> tuple[float, ...]:
    values = [*DEFAULT_FUNDAMENTAL_SWEEP_THRESHOLDS, float(extra_threshold)]
    return tuple(sorted({round(value, 4) for value in values}))


def _threshold_suffix(threshold: float) -> str:
    return f"t{int(round(float(threshold) * 100.0)):03d}"


def _zscore_array(values: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    std = float(np.nanstd(clean))
    if std <= 1e-12 or not np.isfinite(std):
        return np.zeros(len(clean), dtype=float)
    mean = float(np.nanmean(clean))
    return (clean - mean) / std


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(pd.to_numeric(series, errors="coerce").mean())


def _safe_mean_bool(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.fillna(False).astype(bool).mean())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase5O non-price long veto variants on current SOTA."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--long-non-price-panel-path", default=str(DEFAULT_LONG_NON_PRICE_PANEL))
    parser.add_argument(
        "--long-fundamental-panel-path",
        default=str(DEFAULT_LONG_FUNDAMENTAL_PANEL),
    )
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--strict-root", default=str(DEFAULT_STRICT_ROOT))
    parser.add_argument("--phase3-rollup-path", default=str(DEFAULT_PHASE3_H10_ROLLUP))
    parser.add_argument("--turnover-penalty", type=float, default=DEFAULT_TURNOVER_PENALTY)
    parser.add_argument("--trigger-threshold", type=float, default=DEFAULT_TRIGGER_THRESHOLD)
    parser.add_argument(
        "--fundamental-trigger-threshold",
        type=float,
        default=DEFAULT_FUNDAMENTAL_TRIGGER_THRESHOLD,
    )
    parser.add_argument("--recent-filing-days", type=int, default=DEFAULT_RECENT_FILING_DAYS)
    parser.add_argument("--soft-penalty-weight", type=float, default=DEFAULT_SOFT_PENALTY_WEIGHT)
    args = parser.parse_args(argv)

    result = build_phase5o_non_price_long_veto_artifacts(
        signal_panel_path=args.signal_panel_path,
        long_non_price_panel_path=args.long_non_price_panel_path,
        long_fundamental_panel_path=args.long_fundamental_panel_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        output_root=args.output_root,
        strict_root=args.strict_root,
        phase3_rollup_path=args.phase3_rollup_path,
        turnover_penalty=args.turnover_penalty,
        trigger_threshold=args.trigger_threshold,
        fundamental_trigger_threshold=args.fundamental_trigger_threshold,
        recent_filing_days=args.recent_filing_days,
        soft_penalty_weight=args.soft_penalty_weight,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
