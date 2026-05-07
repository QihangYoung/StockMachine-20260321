from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
)
from stockmachine.apps.run_pure_alpha_phase4z import (
    DEFAULT_PHASE3_H10_ROLLUP,
    build_phase4z_strict_horizon_daily_paths,
)
from stockmachine.apps.run_pure_alpha_phase5b import _fmt_pct_like, _text_table, _utc_now
from stockmachine.apps.run_pure_alpha_phase5e import (
    DEFAULT_HOLDING_PERIOD_SESSIONS,
    _aggregate_turnover_summary,
    _construct_one_session_turnover_aware,
    _diagnostic_summary,
    _strict_metric_summary,
    _target_turnover_summary,
)
from stockmachine.apps.run_pure_alpha_phase5f import DEFAULT_SHORT_OVERLAY
from stockmachine.apps.run_pure_alpha_phase5r import (
    DEFAULT_TURNOVER_PENALTY,
    _add_price_only_falling_knife_scores,
    _zscore_array,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5u_price_only_tail_direction_classifier_20260507"
DEFAULT_STRICT_ROOT = DEFAULT_OUTPUT_ROOT / "strict_h10_rebuild"
CURRENT_SOTA_SERIES = "current sota"
PRIMARY_TARGET = "style_factor_sic2_residual"
LEFT_TAIL_THRESHOLD = -0.02
RIGHT_TAIL_THRESHOLD = 0.02
DEFAULT_MIN_TRAIN_SESSIONS = 252
DEFAULT_REFIT_FREQUENCY_SESSIONS = 42
DEFAULT_MIN_TRAIN_ROWS = 12000
DEFAULT_CANDIDATE_COUNT = 80

WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("2015_peak_to_trough", "2015-06-04", "2015-08-21"),
    ("2016_jan_feb", "2016-01-05", "2016-02-09"),
    ("2017_spring_short_fail", "2017-03-20", "2017-05-31"),
    ("2018_aug_sep_short_fail", "2018-08-08", "2018-09-12"),
    ("2019_may", "2019-05-01", "2019-06-05"),
    ("2019_aug", "2019-08-01", "2019-08-31"),
)

MODEL_FEATURES = (
    "return_5d_z",
    "momentum_20d_z",
    "momentum_60d_z",
    "beta_residual_momentum_20d_z",
    "vol_adjusted_momentum_20d_z",
    "beta",
    "long_beta_rank_pct",
    "liquidity_rank_z",
    "price_fk_sector_relative_mom20_down",
    "price_fk_sector_relative_mom60_down",
    "price_fk_no_stabilization",
    "price_falling_knife_score",
    "price_still_falling_score",
    "price_stabilization_score",
    "price_fk_stillfall_penalty",
    "price_fk_stab_relief_penalty",
    "price_fk_x_stillfall",
    "price_fk_x_stabilization",
    "price_fk_x_beta_rank",
    "stillfall_minus_stabilization",
    "price_fk_score_sq",
    "stabilization_score_sq",
    "beta_rank_sq",
)

COMBO_SPECS: tuple[dict[str, str], ...] = (
    {
        "portfolio": "current_sota",
        "series": CURRENT_SOTA_SERIES,
        "long_score": DEFAULT_LONG_SCORE,
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "tail_edge_w025",
        "series": "tail edge w025",
        "long_score": "long_tail_edge_w025",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "tail_edge_w050",
        "series": "tail edge w050",
        "long_score": "long_tail_edge_w050",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "left_penalty_w050",
        "series": "left penalty w050",
        "long_score": "long_left_penalty_w050",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
    {
        "portfolio": "right_boost_w050",
        "series": "right boost w050",
        "long_score": "long_right_boost_w050",
        "short_selector": DEFAULT_SHORT_OVERLAY,
    },
)

PLOT_COLORS = {
    CURRENT_SOTA_SERIES: "#2563eb",
    "tail edge w025": "#0891b2",
    "tail edge w050": "#155e75",
    "left penalty w050": "#c2410c",
    "right boost w050": "#7c3aed",
    "SPY raw": "#6b7280",
}


def build_phase5u_price_only_tail_direction_classifier_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    strict_root: str | Path = DEFAULT_STRICT_ROOT,
    phase3_rollup_path: str | Path = DEFAULT_PHASE3_H10_ROLLUP,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    candidate_pool_per_side: int = DEFAULT_CANDIDATE_COUNT,
    max_single_name_side_weight: float = 1.0 / 30.0,
    min_nonzero_names: int = 20,
    score_weight: float = 0.01,
    soft_group_penalty: float = 25.0,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
    min_train_sessions: int = DEFAULT_MIN_TRAIN_SESSIONS,
    refit_frequency_sessions: int = DEFAULT_REFIT_FREQUENCY_SESSIONS,
    min_train_rows: int = DEFAULT_MIN_TRAIN_ROWS,
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
    panel = _add_tail_model_features(panel)
    panel = _add_residual_targets(
        panel,
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )
    panel, model_log = _add_walk_forward_tail_predictions(
        panel,
        long_variant=long_variant,
        candidate_count=candidate_pool_per_side,
        min_train_sessions=min_train_sessions,
        refit_frequency_sessions=refit_frequency_sessions,
        min_train_rows=min_train_rows,
    )
    panel = _add_tail_direction_long_scores(panel, long_variant=long_variant)
    classifier_eval = _classifier_evaluation(panel, long_variant=long_variant, candidate_count=candidate_pool_per_side)
    classifier_deciles = _classifier_decile_summary(
        panel,
        long_variant=long_variant,
        candidate_count=candidate_pool_per_side,
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

    positions_path = output_dir / "phase5u_positions_validation.csv.gz"
    diagnostics_path = output_dir / "phase5u_daily_validation.csv"
    skipped_path = output_dir / "phase5u_skipped_validation.csv"
    model_log_path = output_dir / "phase5u_walk_forward_model_log.csv"
    classifier_eval_path = output_dir / "phase5u_classifier_eval.csv"
    classifier_deciles_path = output_dir / "phase5u_classifier_deciles.csv"
    positions.to_csv(positions_path, index=False, compression="gzip")
    diagnostics.to_csv(diagnostics_path, index=False)
    skipped.to_csv(skipped_path, index=False)
    model_log.to_csv(model_log_path, index=False)
    classifier_eval.to_csv(classifier_eval_path, index=False)
    classifier_deciles.to_csv(classifier_deciles_path, index=False)

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
    tail_exposure = _tail_exposure_summary(positions=positions, panel=panel)
    sweep = (
        strict_metrics.merge(target_turnover, on="portfolio", how="left")
        .merge(aggregate_turnover, on="portfolio", how="left")
        .merge(diagnostic_summary, on="portfolio", how="left")
        .merge(tail_exposure, on="portfolio", how="left")
    )
    sweep["series"] = sweep["portfolio"].map(_portfolio_to_series)
    sweep["long_score"] = sweep["portfolio"].map(_portfolio_to_long_score)
    sweep["short_selector"] = sweep["portfolio"].map(_portfolio_to_short_selector)
    sweep["turnover_penalty"] = float(turnover_penalty)

    metrics = _build_metrics(sweep=sweep, strict_curve=strict_curve)
    windows = _window_return_summary(strict_curve)
    comparison_curve = _build_comparison_curve(strict_curve)

    sweep_path = output_dir / "phase5u_sweep.csv"
    metrics_path = output_dir / "phase5u_metrics.csv"
    windows_path = output_dir / "phase5u_window_returns.csv"
    tail_exposure_path = output_dir / "phase5u_tail_exposure_summary.csv"
    comparison_curve_path = output_dir / "phase5u_comparison_curve.csv"
    plot_path = output_dir / "phase5u_comparison_plot.png"
    memo_path = output_dir / "phase5u_price_only_tail_direction_classifier_memo.md"
    rollup_path = output_dir / "phase5u_rollup.json"

    sweep.to_csv(sweep_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    windows.to_csv(windows_path, index=False)
    tail_exposure.to_csv(tail_exposure_path, index=False)
    comparison_curve.to_csv(comparison_curve_path, index=False)
    _plot_comparison(comparison_curve, output_path=plot_path)
    memo_path.write_text(
        _memo(
            metrics=metrics,
            windows=windows,
            classifier_eval=classifier_eval,
            classifier_deciles=classifier_deciles,
            tail_exposure=tail_exposure,
            model_log=model_log,
            skipped=skipped,
            plot_path=plot_path,
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
        "primary_target": PRIMARY_TARGET,
        "left_tail_threshold": float(LEFT_TAIL_THRESHOLD),
        "right_tail_threshold": float(RIGHT_TAIL_THRESHOLD),
        "candidate_count": int(candidate_pool_per_side),
        "min_train_sessions": int(min_train_sessions),
        "refit_frequency_sessions": int(refit_frequency_sessions),
        "min_train_rows": int(min_train_rows),
        "strict_rollup": strict_rollup,
        "artifacts": {
            "positions": positions_path.as_posix(),
            "diagnostics": diagnostics_path.as_posix(),
            "skipped": skipped_path.as_posix(),
            "walk_forward_model_log": model_log_path.as_posix(),
            "classifier_eval": classifier_eval_path.as_posix(),
            "classifier_deciles": classifier_deciles_path.as_posix(),
            "sweep": sweep_path.as_posix(),
            "metrics": metrics_path.as_posix(),
            "window_returns": windows_path.as_posix(),
            "tail_exposure_summary": tail_exposure_path.as_posix(),
            "comparison_curve": comparison_curve_path.as_posix(),
            "plot": plot_path.as_posix(),
            "memo": memo_path.as_posix(),
            "strict_curve": strict_curve_path.as_posix(),
        },
        "method": "price_only_walk_forward_tail_direction_classifier",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "No non-price factor panels are loaded or used.",
            "Classifier predictions are walk-forward: each session uses only earlier validation sessions.",
            "Early warmup sessions before enough history receive neutral classifier scores.",
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


def _add_tail_model_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    frame["price_fk_x_stillfall"] = (
        frame["price_falling_knife_score"].astype(float)
        * frame["price_still_falling_score"].astype(float)
    )
    frame["price_fk_x_stabilization"] = (
        frame["price_falling_knife_score"].astype(float)
        * frame["price_stabilization_score"].astype(float)
    )
    frame["price_fk_x_beta_rank"] = (
        frame["price_falling_knife_score"].astype(float) * frame["long_beta_rank_pct"].astype(float)
    )
    frame["stillfall_minus_stabilization"] = (
        frame["price_still_falling_score"].astype(float)
        - frame["price_stabilization_score"].astype(float)
    )
    frame["price_fk_score_sq"] = frame["price_falling_knife_score"].astype(float) ** 2
    frame["stabilization_score_sq"] = frame["price_stabilization_score"].astype(float) ** 2
    frame["beta_rank_sq"] = frame["long_beta_rank_pct"].astype(float) ** 2
    return frame


def _add_walk_forward_tail_predictions(
    panel: pd.DataFrame,
    *,
    long_variant: str,
    candidate_count: int,
    min_train_sessions: int,
    refit_frequency_sessions: int,
    min_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = panel.copy()
    long_panel = frame[frame["variant"].astype(str).eq(long_variant)].copy()
    candidate_panel = _top_by_session(long_panel, score_column=DEFAULT_LONG_SCORE, count=candidate_count)
    candidate_panel = candidate_panel.dropna(subset=[PRIMARY_TARGET]).copy()
    candidate_panel["left_tail_label"] = candidate_panel[PRIMARY_TARGET].lt(LEFT_TAIL_THRESHOLD).astype(int)
    candidate_panel["right_tail_label"] = candidate_panel[PRIMARY_TARGET].gt(RIGHT_TAIL_THRESHOLD).astype(int)
    sessions = sorted(long_panel["session_date"].astype(str).unique())
    pred_frames: list[pd.DataFrame] = []
    log_rows: list[dict[str, Any]] = []
    left_model: Any | None = None
    right_model: Any | None = None
    trained = False
    train_rows = 0
    left_rate = 0.30
    right_rate = 0.30
    for idx, session_date in enumerate(sessions):
        should_refit = (
            idx >= min_train_sessions
            and (idx == min_train_sessions or (idx - min_train_sessions) % refit_frequency_sessions == 0)
        )
        if should_refit:
            train = candidate_panel[candidate_panel["session_date"].astype(str) < str(session_date)].copy()
            train = train.dropna(subset=[PRIMARY_TARGET])
            train_rows = int(len(train))
            left_rate = float(train["left_tail_label"].mean()) if train_rows else 0.30
            right_rate = float(train["right_tail_label"].mean()) if train_rows else 0.30
            left_ok = train_rows >= min_train_rows and train["left_tail_label"].nunique() == 2
            right_ok = train_rows >= min_train_rows and train["right_tail_label"].nunique() == 2
            if left_ok and right_ok:
                left_model = _fit_logistic_tail_model(train, target_column="left_tail_label")
                right_model = _fit_logistic_tail_model(train, target_column="right_tail_label")
                trained = True
            log_rows.append(
                {
                    "session_date": session_date,
                    "session_index": idx,
                    "train_rows": train_rows,
                    "left_tail_rate": left_rate,
                    "right_tail_rate": right_rate,
                    "model_trained": bool(trained),
                    "test_window_used": False,
                }
            )
        subset = long_panel[long_panel["session_date"].astype(str).eq(str(session_date))].copy()
        if trained and left_model is not None and right_model is not None:
            x_pred = subset[list(MODEL_FEATURES)]
            subset["tail_p_left"] = left_model.predict_proba(x_pred)[:, 1]
            subset["tail_p_right"] = right_model.predict_proba(x_pred)[:, 1]
            subset["tail_model_trained"] = True
            subset["tail_train_rows"] = train_rows
        else:
            subset["tail_p_left"] = left_rate
            subset["tail_p_right"] = right_rate
            subset["tail_model_trained"] = False
            subset["tail_train_rows"] = train_rows
        subset["tail_edge_raw"] = subset["tail_p_right"].astype(float) - subset["tail_p_left"].astype(float)
        pred_frames.append(
            subset[
                [
                    "session_date",
                    "symbol",
                    "tail_p_left",
                    "tail_p_right",
                    "tail_edge_raw",
                    "tail_model_trained",
                    "tail_train_rows",
                ]
            ]
        )
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    frame = frame.merge(predictions, on=["session_date", "symbol"], how="left")
    for column, default in (
        ("tail_p_left", left_rate),
        ("tail_p_right", right_rate),
        ("tail_edge_raw", 0.0),
        ("tail_train_rows", 0),
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default)
    frame["tail_model_trained"] = frame["tail_model_trained"].fillna(False).astype(bool)
    return frame, pd.DataFrame(log_rows)


def _fit_logistic_tail_model(train: pd.DataFrame, *, target_column: str) -> Any:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=0.25,
            penalty="l2",
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        ),
    )
    model.fit(train[list(MODEL_FEATURES)], train[target_column].astype(int))
    return model


def _add_tail_direction_long_scores(panel: pd.DataFrame, *, long_variant: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for (variant, session_date), group in panel.groupby(["variant", "session_date"], sort=False):
        part = group.copy()
        is_long_variant = str(variant) == str(long_variant)
        if is_long_variant:
            base_z = _zscore_array(pd.to_numeric(part[DEFAULT_LONG_SCORE], errors="coerce").to_numpy(dtype=float))
            edge_z = _zscore_array(part["tail_edge_raw"].to_numpy(dtype=float))
            p_left_z = _zscore_array(part["tail_p_left"].to_numpy(dtype=float))
            p_right_z = _zscore_array(part["tail_p_right"].to_numpy(dtype=float))
            part["long_tail_edge_w025"] = _zscore_array(base_z + 0.25 * edge_z)
            part["long_tail_edge_w050"] = _zscore_array(base_z + 0.50 * edge_z)
            part["long_left_penalty_w050"] = _zscore_array(base_z - 0.50 * p_left_z)
            part["long_right_boost_w050"] = _zscore_array(base_z + 0.50 * p_right_z)
        else:
            for column in _tail_long_score_columns():
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


def _classifier_evaluation(
    panel: pd.DataFrame,
    *,
    long_variant: str,
    candidate_count: int,
) -> pd.DataFrame:
    candidates = _top_by_session(
        panel[panel["variant"].astype(str).eq(long_variant)].copy(),
        score_column=DEFAULT_LONG_SCORE,
        count=candidate_count,
    )
    candidates = candidates[candidates["tail_model_trained"].astype(bool)].dropna(subset=[PRIMARY_TARGET]).copy()
    rows: list[dict[str, Any]] = []
    for window, start, end in (("full", "", ""), *WINDOWS):
        subset = candidates.copy()
        if window != "full":
            subset = subset[(subset["session_date"] >= start) & (subset["session_date"] <= end)].copy()
        if len(subset) < 100:
            continue
        left = subset[PRIMARY_TARGET].lt(LEFT_TAIL_THRESHOLD).astype(int)
        right = subset[PRIMARY_TARGET].gt(RIGHT_TAIL_THRESHOLD).astype(int)
        rows.append(
            {
                "window": window,
                "start": start,
                "end": end,
                "rows": int(len(subset)),
                "sessions": int(subset["session_date"].nunique()),
                "left_tail_rate": float(left.mean()),
                "right_tail_rate": float(right.mean()),
                "left_auc": _safe_auc(left, subset["tail_p_left"]),
                "right_auc": _safe_auc(right, subset["tail_p_right"]),
                "edge_spearman_to_target": float(
                    subset["tail_edge_raw"].corr(subset[PRIMARY_TARGET], method="spearman")
                ),
                "median_target_bps": float(subset[PRIMARY_TARGET].median() * 10000.0),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows)


def _classifier_decile_summary(
    panel: pd.DataFrame,
    *,
    long_variant: str,
    candidate_count: int,
) -> pd.DataFrame:
    candidates = _top_by_session(
        panel[panel["variant"].astype(str).eq(long_variant)].copy(),
        score_column=DEFAULT_LONG_SCORE,
        count=candidate_count,
    )
    candidates = candidates[candidates["tail_model_trained"].astype(bool)].dropna(subset=[PRIMARY_TARGET]).copy()
    rows: list[dict[str, Any]] = []
    for score_column in ("tail_p_left", "tail_p_right", "tail_edge_raw"):
        valid = candidates.dropna(subset=[score_column, PRIMARY_TARGET]).copy()
        valid["score_pct"] = valid.groupby("session_date")[score_column].rank(method="average", pct=True)
        valid["score_decile"] = np.ceil(valid["score_pct"] * 10.0).clip(1, 10).astype(int)
        for decile, group in valid.groupby("score_decile", sort=True):
            target = group[PRIMARY_TARGET].astype(float)
            rows.append(
                {
                    "score_column": score_column,
                    "score_decile": int(decile),
                    "rows": int(len(group)),
                    "target_mean_bps": float(target.mean() * 10000.0),
                    "target_median_bps": float(target.median() * 10000.0),
                    "left_tail_rate": float(target.lt(LEFT_TAIL_THRESHOLD).mean()),
                    "right_tail_rate": float(target.gt(RIGHT_TAIL_THRESHOLD).mean()),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["score_column", "score_decile"]).reset_index(drop=True)


def _tail_exposure_summary(*, positions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    features = ["tail_p_left", "tail_p_right", "tail_edge_raw", "tail_model_trained"]
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
                "weighted_tail_p_left": _weighted_mean(group, weights, "tail_p_left"),
                "weighted_tail_p_right": _weighted_mean(group, weights, "tail_p_right"),
                "weighted_tail_edge_raw": _weighted_mean(group, weights, "tail_edge_raw"),
                "trained_share": float(group["tail_model_trained"].astype(bool).mean()),
                "test_window_used": False,
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return pd.DataFrame(columns=["portfolio"])
    return (
        daily.groupby("portfolio", as_index=False)
        .agg(
            mean_weighted_tail_p_left=("weighted_tail_p_left", "mean"),
            mean_weighted_tail_p_right=("weighted_tail_p_right", "mean"),
            mean_weighted_tail_edge_raw=("weighted_tail_edge_raw", "mean"),
            mean_trained_share=("trained_share", "mean"),
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
                "long_score": str(spec["long_score"]),
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
                "mean_weighted_tail_p_left": float(item.get("mean_weighted_tail_p_left", np.nan)),
                "mean_weighted_tail_p_right": float(item.get("mean_weighted_tail_p_right", np.nan)),
                "mean_weighted_tail_edge_raw": float(item.get("mean_weighted_tail_edge_raw", np.nan)),
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
        "long_score": "",
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
        "mean_weighted_tail_p_left": np.nan,
        "mean_weighted_tail_p_right": np.nan,
        "mean_weighted_tail_edge_raw": np.nan,
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
    ax1.set_title("Phase5U Price-Only Walk-Forward Tail Direction Classifier")
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
    classifier_eval: pd.DataFrame,
    classifier_deciles: pd.DataFrame,
    tail_exposure: pd.DataFrame,
    model_log: pd.DataFrame,
    skipped: pd.DataFrame,
    plot_path: Path,
) -> str:
    metric_cols = [
        "series",
        "portfolio",
        "annualized_return",
        "annualized_vol",
        "sharpe_no_rf",
        "max_drawdown",
        "rolling_60_positive_rate",
        "aggregate_turnover_mean",
        "mean_weighted_tail_p_left",
        "mean_weighted_tail_p_right",
        "mean_weighted_tail_edge_raw",
    ]
    eval_cols = [
        "window",
        "rows",
        "left_tail_rate",
        "right_tail_rate",
        "left_auc",
        "right_auc",
        "edge_spearman_to_target",
        "median_target_bps",
    ]
    decile_cols = [
        "score_column",
        "score_decile",
        "target_median_bps",
        "left_tail_rate",
        "right_tail_rate",
    ]
    window_cols = ["series", "window", "compound_return", "hit_rate_daily"]
    exposure_cols = [
        "portfolio",
        "mean_weighted_tail_p_left",
        "mean_weighted_tail_p_right",
        "mean_weighted_tail_edge_raw",
        "mean_trained_share",
    ]
    model_summary = pd.DataFrame(
        [
            {
                "refits": int(len(model_log)),
                "trained_refits": int(model_log["model_trained"].sum()) if not model_log.empty else 0,
                "first_trained_session": (
                    str(model_log.loc[model_log["model_trained"], "session_date"].iloc[0])
                    if not model_log.empty and model_log["model_trained"].any()
                    else ""
                ),
                "last_train_rows": int(model_log["train_rows"].iloc[-1]) if not model_log.empty else 0,
            }
        ]
    )
    skipped_summary = (
        skipped.groupby(["portfolio", "reason"], as_index=False).size().rename(columns={"size": "rows"})
        if not skipped.empty
        else pd.DataFrame(columns=["portfolio", "reason", "rows"])
    )
    lines = [
        "# Phase5U Price-Only Tail Direction Classifier",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        f"- Short side fixed to `{DEFAULT_SHORT_OVERLAY}`.",
        f"- Base long selector fixed to `{DEFAULT_LONG_SCORE}`; classifier is a second-stage overlay.",
        f"- Target: left tail `{PRIMARY_TARGET} < {LEFT_TAIL_THRESHOLD:.2%}`, right tail `{PRIMARY_TARGET} > {RIGHT_TAIL_THRESHOLD:.2%}`.",
        "- Walk-forward only: each session uses only earlier validation sessions.",
        "- No non-price factor panels are loaded or used.",
        "- Validation window only; no test-window performance is computed.",
        "",
        "## Model Summary",
        "",
        _text_table(model_summary),
        "",
        "## Strategy Metrics",
        "",
        _text_table(_format_table(metrics[metric_cols] if not metrics.empty else metrics)),
        "",
        "## Classifier OOS Evaluation",
        "",
        _text_table(_format_table(classifier_eval[eval_cols] if not classifier_eval.empty else classifier_eval)),
        "",
        "## Classifier Deciles",
        "",
        _text_table(_format_table(classifier_deciles[decile_cols] if not classifier_deciles.empty else classifier_deciles)),
        "",
        "## Stress Windows",
        "",
        _text_table(_format_table(windows[window_cols] if not windows.empty else windows)),
        "",
        "## Tail Exposure",
        "",
        _text_table(_format_table(tail_exposure[exposure_cols] if not tail_exposure.empty else tail_exposure)),
        "",
        "## Skipped Rows",
        "",
        _text_table(skipped_summary) if not skipped_summary.empty else "No skipped rows.",
        "",
        "## Plot",
        "",
        f"![Phase5U comparison]({plot_path.as_posix()})",
        "",
        "## Read",
        "",
        "- This is the first model-based tail-direction test after price-only veto/cap failed to separate rebound winners from falling knives.",
        "- A useful classifier should show OOS AUC above random and improve the portfolio without relying on a single stress window.",
        "- If classifier AUC is weak, price-only features likely do not contain the missing state variable.",
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
            "compound_return",
            "hit_rate_daily",
            "left_tail_rate",
            "right_tail_rate",
        }:
            out[column] = out[column].map(_fmt_pct_like)
        elif column.endswith("_bps"):
            out[column] = out[column].map(lambda value: _fmt_number(value, digits=1))
        elif pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: _fmt_number(value, digits=3))
    return out


def _fmt_number(value: Any, *, digits: int) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _top_by_session(frame: pd.DataFrame, *, score_column: str, count: int) -> pd.DataFrame:
    rows = []
    for _, group in frame.groupby("session_date", sort=False):
        rows.append(group.sort_values([score_column, "symbol"], ascending=[False, True]).head(count))
    return pd.concat(rows, ignore_index=True) if rows else frame.iloc[0:0].copy()


def _safe_auc(labels: pd.Series, score: pd.Series) -> float:
    y = labels.astype(int)
    if y.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y, score.astype(float)))


def _tail_long_score_columns() -> tuple[str, ...]:
    return tuple(str(spec["long_score"]) for spec in COMBO_SPECS if str(spec["portfolio"]) != "current_sota")


def _portfolio_to_series(portfolio: str) -> str:
    mapping = {str(spec["portfolio"]): str(spec["series"]) for spec in COMBO_SPECS}
    return mapping.get(str(portfolio), str(portfolio))


def _portfolio_to_long_score(portfolio: str) -> str:
    mapping = {str(spec["portfolio"]): str(spec["long_score"]) for spec in COMBO_SPECS}
    return mapping.get(str(portfolio), "")


def _portfolio_to_short_selector(portfolio: str) -> str:
    mapping = {str(spec["portfolio"]): str(spec["short_selector"]) for spec in COMBO_SPECS}
    return mapping.get(str(portfolio), "")


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
        description="Run Phase5U price-only walk-forward tail-direction classifier overlays."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--strict-root", default=str(DEFAULT_STRICT_ROOT))
    parser.add_argument("--phase3-rollup-path", default=str(DEFAULT_PHASE3_H10_ROLLUP))
    parser.add_argument("--min-train-sessions", type=int, default=DEFAULT_MIN_TRAIN_SESSIONS)
    parser.add_argument("--refit-frequency-sessions", type=int, default=DEFAULT_REFIT_FREQUENCY_SESSIONS)
    parser.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    args = parser.parse_args(argv)
    result = build_phase5u_price_only_tail_direction_classifier_artifacts(
        signal_panel_path=args.signal_panel_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        output_root=args.output_root,
        strict_root=args.strict_root,
        phase3_rollup_path=args.phase3_rollup_path,
        min_train_sessions=args.min_train_sessions,
        refit_frequency_sessions=args.refit_frequency_sessions,
        min_train_rows=args.min_train_rows,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
