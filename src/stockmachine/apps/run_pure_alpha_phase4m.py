from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT, TARGET_COLUMN
from stockmachine.apps.run_pure_alpha_phase4f import (
    DEFAULT_PHASE3_SIGNAL_PANEL,
    _load_posterior_panel,
)
from stockmachine.apps.run_pure_alpha_phase4i import (
    TARGET_BETA_RESIDUAL,
    TARGET_CS_DEMEANED,
    TARGET_RISK_NEUTRAL,
    _add_residual_targets,
)
from stockmachine.apps.run_pure_alpha_phase4j import _add_selector_scores
from stockmachine.apps.run_pure_alpha_phase4k import (
    _construct_books as _construct_asymmetric_books,
    _summary as _portfolio_summary,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4m_ridge_short_selector_20260418"
DEFAULT_LONG_VARIANTS = ("top1000_clean_core_beta_full",)
DEFAULT_SHORT_VARIANTS = (
    "adv30m_clean_core_beta_full",
    "top1000_clean_core_beta_full",
)
DEFAULT_RESIDUAL_TARGETS = (TARGET_CS_DEMEANED,)
RIDGE_FEATURES = (
    "momentum_20d_z",
    "beta_residual_momentum_20d_z",
    "vol_adjusted_momentum_20d_z",
    "exhausted_winner_20_5",
    "residual_overextension_20_5",
    "return_5d_z",
    "beta_z",
    "fragile_winner_proxy",
)
RIDGE_SELECTOR = "ridge_short_score"
BASELINE_SELECTOR = "short_core_plus_overextension"


def build_phase4m_ridge_short_selector_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_PHASE3_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    long_variants: Sequence[str] = DEFAULT_LONG_VARIANTS,
    short_variants: Sequence[str] = DEFAULT_SHORT_VARIANTS,
    residual_targets: Sequence[str] = DEFAULT_RESIDUAL_TARGETS,
    features: Sequence[str] = RIDGE_FEATURES,
    ridge_alpha: float = 10.0,
    candidate_count: int = 30,
    initial_train_sessions: int = 252,
    label_embargo_sessions: int = 5,
    block_sessions: int = 126,
    max_train_rows: int = 200_000,
    min_regression_rows: int = 80,
    min_train_rows: int = 1_000,
    long_score: str = "reversal_5d",
    max_names_per_side: int = 30,
    max_single_name_side_weight: float = 0.05,
    beta_match_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Train a rolling Ridge short selector and compare it with transparent overextension."""

    _validate_settings(
        long_variants=long_variants,
        short_variants=short_variants,
        residual_targets=residual_targets,
        features=features,
        ridge_alpha=ridge_alpha,
        candidate_count=candidate_count,
        initial_train_sessions=initial_train_sessions,
        label_embargo_sessions=label_embargo_sessions,
        block_sessions=block_sessions,
        max_train_rows=max_train_rows,
        min_regression_rows=min_regression_rows,
        min_train_rows=min_train_rows,
    )
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_variants = tuple(dict.fromkeys([*long_variants, *short_variants]))
    panel = _load_ridge_panel(
        signal_panel_path,
        variant_names=all_variants,
        min_regression_rows=min_regression_rows,
    )
    predictions, folds, coefficients = _rolling_ridge_predictions(
        panel,
        variant_names=short_variants,
        residual_targets=residual_targets,
        features=features,
        ridge_alpha=ridge_alpha,
        initial_train_sessions=initial_train_sessions,
        label_embargo_sessions=label_embargo_sessions,
        block_sessions=block_sessions,
        max_train_rows=max_train_rows,
        min_train_rows=min_train_rows,
    )
    selector_daily = _selector_daily(
        predictions,
        residual_targets=residual_targets,
        candidate_count=candidate_count,
    )
    selector_summary = _selector_summary(selector_daily)
    portfolio_positions, portfolio_daily, portfolio_skipped, portfolio_summary = (
        _portfolio_diagnostics(
            panel,
            predictions,
            long_variants=long_variants,
            short_variants=short_variants,
            residual_targets=residual_targets,
            ridge_alpha=ridge_alpha,
            long_score=long_score,
            max_names_per_side=max_names_per_side,
            max_single_name_side_weight=max_single_name_side_weight,
            beta_match_tolerance=beta_match_tolerance,
        )
    )
    portfolio_common = _common_portfolio_comparison(portfolio_daily)
    paths = _write_outputs(
        output_dir,
        predictions=predictions,
        folds=folds,
        coefficients=coefficients,
        selector_daily=selector_daily,
        selector_summary=selector_summary,
        portfolio_positions=portfolio_positions,
        portfolio_daily=portfolio_daily,
        portfolio_skipped=portfolio_skipped,
        portfolio_summary=portfolio_summary,
        portfolio_common=portfolio_common,
    )
    memo_path = output_dir / "phase4m_ridge_short_selector_memo.md"
    memo_path.write_text(
        _memo(
            selector_summary,
            portfolio_summary,
            portfolio_common,
            features=features,
            residual_targets=residual_targets,
            ridge_alpha=ridge_alpha,
            candidate_count=candidate_count,
        ),
        encoding="utf-8",
    )
    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "long_variants": list(long_variants),
        "short_variants": list(short_variants),
        "residual_targets": list(residual_targets),
        "features": list(features),
        "ridge_alpha": float(ridge_alpha),
        "candidate_count": int(candidate_count),
        "initial_train_sessions": int(initial_train_sessions),
        "label_embargo_sessions": int(label_embargo_sessions),
        "block_sessions": int(block_sessions),
        "max_train_rows": int(max_train_rows),
        "min_train_rows": int(min_train_rows),
        "panel_rows_loaded": int(len(panel)),
        "prediction_rows": int(len(predictions)),
        "selector_daily_rows": int(len(selector_daily)),
        "selector_summary_rows": int(len(selector_summary)),
        "portfolio_daily_rows": int(len(portfolio_daily)),
        "portfolio_summary_rows": int(len(portfolio_summary)),
        "portfolio_common_rows": int(len(portfolio_common)),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "rolling_ridge_short_selector_with_embargo",
        "primary_evaluation": "standalone_short_selector_performance",
        "secondary_evaluation": "beta_matched_portfolio_diagnostics",
        "promotion_rule": (
            "Do not promote a short selector unless it beats the transparent "
            "baseline on standalone selector performance; portfolio diagnostics "
            "are secondary at this stage."
        ),
        **paths,
    }
    (output_dir / "phase4m_ridge_short_selector_rollup.json").write_text(
        json.dumps(rollup, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return rollup


def _validate_settings(
    *,
    long_variants: Sequence[str],
    short_variants: Sequence[str],
    residual_targets: Sequence[str],
    features: Sequence[str],
    ridge_alpha: float,
    candidate_count: int,
    initial_train_sessions: int,
    label_embargo_sessions: int,
    block_sessions: int,
    max_train_rows: int,
    min_regression_rows: int,
    min_train_rows: int,
) -> None:
    if not long_variants:
        raise ValueError("At least one long variant is required.")
    if not short_variants:
        raise ValueError("At least one short variant is required.")
    if not residual_targets:
        raise ValueError("At least one residual target is required.")
    if not features:
        raise ValueError("At least one feature is required.")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative.")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive.")
    if initial_train_sessions <= 0:
        raise ValueError("initial_train_sessions must be positive.")
    if label_embargo_sessions < 0:
        raise ValueError("label_embargo_sessions must be non-negative.")
    if block_sessions <= 0:
        raise ValueError("block_sessions must be positive.")
    if max_train_rows <= 0:
        raise ValueError("max_train_rows must be positive.")
    if min_regression_rows <= 0:
        raise ValueError("min_regression_rows must be positive.")
    if min_train_rows <= 0:
        raise ValueError("min_train_rows must be positive.")


def _load_ridge_panel(
    signal_panel_path: str | Path,
    *,
    variant_names: Sequence[str],
    min_regression_rows: int,
) -> pd.DataFrame:
    panel = _load_posterior_panel(
        signal_panel_path,
        variant_names=variant_names,
        target_column=TARGET_COLUMN,
    )
    panel = _add_residual_targets(panel, min_regression_rows=min_regression_rows)
    scored = [_add_selector_scores(group) for _, group in panel.groupby(["variant", "session_date"])]
    if not scored:
        return panel
    return pd.concat(scored, ignore_index=True).sort_values(
        ["variant", "session_date", "symbol"]
    )


def _rolling_ridge_predictions(
    panel: pd.DataFrame,
    *,
    variant_names: Sequence[str],
    residual_targets: Sequence[str],
    features: Sequence[str],
    ridge_alpha: float,
    initial_train_sessions: int,
    label_embargo_sessions: int,
    block_sessions: int,
    max_train_rows: int,
    min_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for variant in variant_names:
        variant_panel = panel.loc[panel["variant"] == variant].copy()
        sessions = sorted(variant_panel["session_date"].unique())
        start = initial_train_sessions + label_embargo_sessions
        fold_id = 0
        while start < len(sessions):
            train_sessions = sessions[: start - label_embargo_sessions]
            predict_sessions = sessions[start : start + block_sessions]
            if not predict_sessions:
                break
            for target_name in residual_targets:
                train, predict = _train_predict_frames(
                    variant_panel,
                    target_name=target_name,
                    features=features,
                    train_sessions=train_sessions,
                    predict_sessions=predict_sessions,
                    max_train_rows=max_train_rows,
                )
                if len(train) < min_train_rows or predict.empty:
                    continue
                model, train_means, train_stds = _fit_ridge(
                    train,
                    target_name=target_name,
                    features=features,
                    ridge_alpha=ridge_alpha,
                )
                prediction = predict.copy()
                x_predict = _standardize(prediction[list(features)], train_means, train_stds)
                prediction["residual_target"] = target_name
                prediction["ridge_alpha"] = float(ridge_alpha)
                prediction["model_fold_id"] = int(fold_id)
                prediction["ridge_predicted_residual"] = model.predict(
                    x_predict.to_numpy(dtype=float)
                )
                prediction[RIDGE_SELECTOR] = -prediction["ridge_predicted_residual"]
                prediction["test_window_used"] = False
                prediction_frames.append(prediction[_prediction_columns()])
                fold_rows.append(
                    {
                        "variant": variant,
                        "residual_target": target_name,
                        "ridge_alpha": float(ridge_alpha),
                        "model_fold_id": int(fold_id),
                        "train_start_session": train_sessions[0],
                        "train_end_session": train_sessions[-1],
                        "predict_start_session": predict_sessions[0],
                        "predict_end_session": predict_sessions[-1],
                        "train_sessions": int(len(train_sessions)),
                        "predict_sessions": int(len(predict_sessions)),
                        "train_rows": int(len(train)),
                        "predict_rows": int(len(predict)),
                        "label_embargo_sessions": int(label_embargo_sessions),
                        "intercept": float(model.intercept_),
                        "test_window_used": False,
                    }
                )
                for feature, coefficient in zip(features, model.coef_):
                    coefficient_rows.append(
                        {
                            "variant": variant,
                            "residual_target": target_name,
                            "ridge_alpha": float(ridge_alpha),
                            "model_fold_id": int(fold_id),
                            "feature": feature,
                            "coefficient": float(coefficient),
                            "test_window_used": False,
                        }
                    )
            start += block_sessions
            fold_id += 1
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame(columns=_prediction_columns())
    )
    folds = pd.DataFrame(fold_rows, columns=_fold_columns())
    coefficients = _coefficient_summary(pd.DataFrame(coefficient_rows))
    return predictions, folds, coefficients


def _train_predict_frames(
    variant_panel: pd.DataFrame,
    *,
    target_name: str,
    features: Sequence[str],
    train_sessions: Sequence[str],
    predict_sessions: Sequence[str],
    max_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(dict.fromkeys([
        "session_date",
        "variant",
        "symbol",
        *features,
        target_name,
        TARGET_BETA_RESIDUAL,
        TARGET_CS_DEMEANED,
        TARGET_RISK_NEUTRAL,
        "forward_return_5d",
        BASELINE_SELECTOR,
    ]))
    train = (
        variant_panel.loc[variant_panel["session_date"].isin(set(train_sessions)), columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(["session_date", "symbol"])
    )
    if len(train) > max_train_rows:
        train = train.tail(max_train_rows)
    predict = (
        variant_panel.loc[variant_panel["session_date"].isin(set(predict_sessions)), columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(["session_date", "symbol"])
    )
    return train, predict


def _fit_ridge(
    train: pd.DataFrame,
    *,
    target_name: str,
    features: Sequence[str],
    ridge_alpha: float,
) -> tuple[Ridge, pd.Series, pd.Series]:
    x_train = train[list(features)].astype(float)
    train_means = x_train.mean(axis=0)
    train_stds = x_train.std(axis=0, ddof=0).replace(0.0, 1.0)
    x_standard = _standardize(x_train, train_means, train_stds)
    y_train = train[target_name].astype(float)
    model = Ridge(alpha=ridge_alpha)
    model.fit(x_standard.to_numpy(dtype=float), y_train.to_numpy(dtype=float))
    return model, train_means, train_stds


def _standardize(frame: pd.DataFrame, means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    return (frame.astype(float) - means) / stds


def _selector_daily(
    predictions: pd.DataFrame,
    *,
    residual_targets: Sequence[str],
    candidate_count: int,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=_selector_daily_columns())
    rows: list[dict[str, Any]] = []
    for (variant, target_name, ridge_alpha, session_date), group in predictions.groupby(
        ["variant", "residual_target", "ridge_alpha", "session_date"],
        sort=True,
    ):
        if target_name not in residual_targets or len(group) < candidate_count:
            continue
        oracle = group.nsmallest(candidate_count, target_name)
        oracle_symbols = set(oracle["symbol"])
        for selector_name in (RIDGE_SELECTOR, BASELINE_SELECTOR):
            selected = group.dropna(subset=[selector_name]).sort_values(
                [selector_name, "symbol"],
                ascending=[False, True],
            ).head(candidate_count)
            if len(selected) < candidate_count:
                continue
            all_scores = group[[selector_name, target_name]].dropna()
            rank_ic = all_scores[selector_name].corr(all_scores[target_name], method="spearman")
            rows.append(
                {
                    "session_date": session_date,
                    "variant": variant,
                    "residual_target": target_name,
                    "ridge_alpha": float(ridge_alpha),
                    "selector": selector_name,
                    "eligible_names": int(len(group)),
                    "selected_names": int(len(selected)),
                    "selected_mean_target_return": float(selected[target_name].mean()),
                    "short_contribution_target": float(-selected[target_name].mean()),
                    "selected_negative_target_share": float((selected[target_name] < 0).mean()),
                    "short_contribution_beta_residual": float(
                        -selected[TARGET_BETA_RESIDUAL].mean()
                    ),
                    "short_contribution_cs_demeaned_residual": float(
                        -selected[TARGET_CS_DEMEANED].mean()
                    ),
                    "short_contribution_risk_neutral_residual": float(
                        -selected[TARGET_RISK_NEUTRAL].mean()
                    ),
                    "short_contribution_return": float(-selected["forward_return_5d"].mean()),
                    "oracle_short_contribution_target": float(-oracle[target_name].mean()),
                    "oracle_overlap_rate": float(
                        len(set(selected["symbol"]).intersection(oracle_symbols))
                        / candidate_count
                    ),
                    "rank_ic_selector_vs_target": float(rank_ic),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows, columns=_selector_daily_columns())


def _selector_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=_selector_summary_columns())
    rows = []
    for (variant, target_name, ridge_alpha, selector), group in daily.groupby(
        ["variant", "residual_target", "ridge_alpha", "selector"],
        sort=True,
    ):
        contribution = group["short_contribution_target"]
        rows.append(
            {
                "variant": variant,
                "residual_target": target_name,
                "ridge_alpha": float(ridge_alpha),
                "selector": selector,
                "sessions": int(len(group)),
                "mean_short_contribution_target": float(contribution.mean()),
                "target_hit_rate": float((contribution > 0).mean()),
                "mean_selected_negative_target_share": float(
                    group["selected_negative_target_share"].mean()
                ),
                "mean_short_contribution_beta_residual": float(
                    group["short_contribution_beta_residual"].mean()
                ),
                "mean_short_contribution_cs_demeaned_residual": float(
                    group["short_contribution_cs_demeaned_residual"].mean()
                ),
                "mean_short_contribution_risk_neutral_residual": float(
                    group["short_contribution_risk_neutral_residual"].mean()
                ),
                "mean_short_contribution_return": float(
                    group["short_contribution_return"].mean()
                ),
                "mean_oracle_short_contribution_target": float(
                    group["oracle_short_contribution_target"].mean()
                ),
                "mean_oracle_overlap_rate": float(group["oracle_overlap_rate"].mean()),
                "mean_rank_ic_selector_vs_target": float(
                    group["rank_ic_selector_vs_target"].mean()
                ),
                "rank_ic_negative_share": float(
                    (group["rank_ic_selector_vs_target"] < 0).mean()
                ),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows, columns=_selector_summary_columns()).sort_values(
        ["residual_target", "variant", "mean_short_contribution_target"],
        ascending=[True, True, False],
    )


def _portfolio_diagnostics(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    long_variants: Sequence[str],
    short_variants: Sequence[str],
    residual_targets: Sequence[str],
    ridge_alpha: float,
    long_score: str,
    max_names_per_side: int,
    max_single_name_side_weight: float,
    beta_match_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    position_frames = []
    daily_frames = []
    skipped_frames = []
    summary_frames = []
    for target_name in residual_targets:
        target_predictions = predictions[
            (predictions["residual_target"] == target_name)
            & (predictions["ridge_alpha"] == float(ridge_alpha))
        ].copy()
        if target_predictions.empty:
            continue
        sessions = set(target_predictions["session_date"])
        score_column = _portfolio_score_column(target_name, ridge_alpha)
        baseline_column = _portfolio_baseline_column(target_name, ridge_alpha)
        score_panel = panel.loc[panel["session_date"].isin(sessions)].copy()
        score_panel = score_panel.merge(
            target_predictions[["variant", "session_date", "symbol", RIDGE_SELECTOR]],
            on=["variant", "session_date", "symbol"],
            how="left",
        )
        score_panel[score_column] = score_panel[RIDGE_SELECTOR]
        score_panel[baseline_column] = score_panel[BASELINE_SELECTOR]
        for selector_kind, selector_column in (
            ("ridge", score_column),
            ("baseline", baseline_column),
        ):
            positions, daily, skipped = _construct_asymmetric_books(
                score_panel,
                long_variants=long_variants,
                short_variants=short_variants,
                long_score=long_score,
                short_selector=selector_column,
                required_nonzero_names=int(np.ceil(1.0 / max_single_name_side_weight)),
                target_names_per_side=min(max_names_per_side, 20),
                max_names_per_side=max_names_per_side,
                max_single_name_side_weight=max_single_name_side_weight,
                beta_match_tolerance=beta_match_tolerance,
            )
            summary = _portfolio_summary(daily, skipped)
            for frame in (positions, daily, skipped, summary):
                if not frame.empty:
                    frame["residual_target"] = target_name
                    frame["ridge_alpha"] = float(ridge_alpha)
                    frame["selector_kind"] = selector_kind
            position_frames.append(positions)
            daily_frames.append(daily)
            skipped_frames.append(skipped)
            summary_frames.append(summary)
    return (
        pd.concat(position_frames, ignore_index=True) if position_frames else pd.DataFrame(),
        pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(),
        pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(),
        pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(),
    )


def _common_portfolio_comparison(portfolio_daily: pd.DataFrame) -> pd.DataFrame:
    if portfolio_daily.empty:
        return pd.DataFrame(columns=_common_portfolio_columns())
    rows = []
    keys = ["pair", "residual_target", "ridge_alpha"]
    for key, group in portfolio_daily.groupby(keys, sort=True):
        kinds = set(group["selector_kind"])
        if not {"ridge", "baseline"}.issubset(kinds):
            continue
        ridge = group.loc[group["selector_kind"] == "ridge"]
        baseline = group.loc[group["selector_kind"] == "baseline"]
        common_sessions = sorted(set(ridge["session_date"]).intersection(baseline["session_date"]))
        if not common_sessions:
            continue
        common = group.loc[group["session_date"].isin(common_sessions)]
        metrics = {}
        for selector_kind, selector_frame in common.groupby("selector_kind", sort=True):
            for column in (
                "spread_return",
                "spread_cs_demeaned_residual",
                "short_cs_demeaned_residual_contribution",
                "long_cs_demeaned_residual",
                "mean_abs_net_beta_proxy",
            ):
                if column == "mean_abs_net_beta_proxy":
                    metrics[f"{selector_kind}_mean_abs_net_beta"] = float(
                        selector_frame["net_beta"].abs().mean()
                    )
                else:
                    metrics[f"{selector_kind}_{column}"] = float(selector_frame[column].mean())
        row = {
            "pair": key[0],
            "residual_target": key[1],
            "ridge_alpha": float(key[2]),
            "common_sessions": int(len(common_sessions)),
            **metrics,
            "test_window_used": False,
        }
        row["ridge_minus_baseline_spread_cs_demeaned_residual"] = (
            row.get("ridge_spread_cs_demeaned_residual", np.nan)
            - row.get("baseline_spread_cs_demeaned_residual", np.nan)
        )
        row["ridge_minus_baseline_short_cs_contribution"] = (
            row.get("ridge_short_cs_demeaned_residual_contribution", np.nan)
            - row.get("baseline_short_cs_demeaned_residual_contribution", np.nan)
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=_common_portfolio_columns()).sort_values(
        "ridge_minus_baseline_spread_cs_demeaned_residual",
        ascending=False,
    )


def _portfolio_score_column(target_name: str, ridge_alpha: float) -> str:
    return f"ridge_short_score_{_target_alias(target_name)}_alpha_{_alpha_alias(ridge_alpha)}"


def _portfolio_baseline_column(target_name: str, ridge_alpha: float) -> str:
    return f"baseline_overextension_{_target_alias(target_name)}_alpha_{_alpha_alias(ridge_alpha)}"


def _target_alias(target_name: str) -> str:
    aliases = {
        TARGET_BETA_RESIDUAL: "beta",
        TARGET_CS_DEMEANED: "cs",
        TARGET_RISK_NEUTRAL: "risk",
    }
    return aliases.get(target_name, target_name.replace("_", "")[:16])


def _alpha_alias(ridge_alpha: float) -> str:
    return str(ridge_alpha).replace(".", "p").replace("-", "m")


def _write_outputs(
    output_dir: Path,
    *,
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    coefficients: pd.DataFrame,
    selector_daily: pd.DataFrame,
    selector_summary: pd.DataFrame,
    portfolio_positions: pd.DataFrame,
    portfolio_daily: pd.DataFrame,
    portfolio_skipped: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    portfolio_common: pd.DataFrame,
) -> dict[str, str]:
    paths = {
        "predictions_artifact": output_dir / "phase4m_ridge_predictions_validation.csv.gz",
        "fold_manifest_artifact": output_dir / "phase4m_ridge_fold_manifest_validation.csv",
        "coefficient_artifact": output_dir / "phase4m_ridge_coefficients_validation.csv",
        "selector_daily_artifact": output_dir / "phase4m_ridge_selector_daily_validation.csv",
        "selector_summary_artifact": output_dir / "phase4m_ridge_selector_summary_validation.csv",
        "portfolio_positions_artifact": output_dir
        / "phase4m_ridge_portfolio_positions_validation.csv.gz",
        "portfolio_daily_artifact": output_dir / "phase4m_ridge_portfolio_daily_validation.csv",
        "portfolio_skipped_artifact": output_dir
        / "phase4m_ridge_portfolio_skipped_validation.csv",
        "portfolio_summary_artifact": output_dir
        / "phase4m_ridge_portfolio_summary_validation.csv",
        "portfolio_common_artifact": output_dir
        / "phase4m_ridge_portfolio_common_session_validation.csv",
    }
    predictions.to_csv(paths["predictions_artifact"], index=False, compression="gzip")
    folds.to_csv(paths["fold_manifest_artifact"], index=False)
    coefficients.to_csv(paths["coefficient_artifact"], index=False)
    selector_daily.to_csv(paths["selector_daily_artifact"], index=False)
    selector_summary.to_csv(paths["selector_summary_artifact"], index=False)
    portfolio_positions.to_csv(
        paths["portfolio_positions_artifact"],
        index=False,
        compression="gzip",
    )
    portfolio_daily.to_csv(paths["portfolio_daily_artifact"], index=False)
    portfolio_skipped.to_csv(paths["portfolio_skipped_artifact"], index=False)
    portfolio_summary.to_csv(paths["portfolio_summary_artifact"], index=False)
    portfolio_common.to_csv(paths["portfolio_common_artifact"], index=False)
    return {key: path.as_posix() for key, path in paths.items()}


def _memo(
    selector_summary: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    portfolio_common: pd.DataFrame,
    *,
    features: Sequence[str],
    residual_targets: Sequence[str],
    ridge_alpha: float,
    candidate_count: int,
) -> str:
    lines = [
        "# Pure Alpha Phase 4M Ridge Short Selector Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "Validation-only rolling Ridge diagnostic for short-side residual-loser selection.",
        "",
        "Primary decision standard: standalone top-N short selector performance. "
        "Beta-matched portfolio diagnostics are secondary at this stage because "
        "construction filters can change the evaluated session set.",
        "",
        "## Model",
        "",
        f"- residual targets: `{', '.join(residual_targets)}`",
        f"- Ridge alpha: `{ridge_alpha}`",
        f"- selected names per selector diagnostic: `{candidate_count}`",
        f"- features: `{', '.join(features)}`",
        "- score: `ridge_short_score = - predicted residual return`",
        "- baseline: `short_core_plus_overextension` on the same prediction sessions",
        "",
        "## Selector Summary",
        "",
        "| Variant | Target | Selector | Contribution | Hit Rate | RankIC |",
        "|---|---|---|---:|---:|---:|",
    ]
    for _, row in selector_summary.head(16).iterrows():
        lines.append(
            f"| {row['variant']} | {row['residual_target']} | {row['selector']} | "
            f"{row['mean_short_contribution_target']} | {row['target_hit_rate']} | "
            f"{row['mean_rank_ic_selector_vs_target']} |"
        )
    lines.extend(
        [
            "",
            "## Portfolio Summary",
            "",
            "Secondary diagnostic only; do not promote Ridge from portfolio results if "
            "it loses the standalone selector test.",
            "",
            "| Pair | Selector | CS Resid Spread | Short CS Contrib | Abs Net Beta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    if portfolio_summary.empty:
        lines.append("| n/a | n/a | n/a | n/a | n/a |")
    else:
        for _, row in portfolio_summary.head(16).iterrows():
            lines.append(
                f"| {row['pair']} | {row['short_selector']} | "
                f"{row['mean_spread_cs_demeaned_residual']} | "
                f"{row['mean_short_cs_demeaned_residual_contribution']} | "
                f"{row['mean_abs_net_beta']} |"
            )
    lines.extend(
        [
            "",
            "## Common-Session Portfolio Check",
            "",
            "This controls for overlapping constructed sessions but still reflects "
            "beta-matching feasibility, not pure selector skill.",
            "",
            "| Pair | Common Sessions | Ridge - Baseline CS Spread | Ridge - Baseline Short CS |",
            "|---|---:|---:|---:|",
        ]
    )
    if portfolio_common.empty:
        lines.append("| n/a | 0 | n/a | n/a |")
    else:
        for _, row in portfolio_common.head(12).iterrows():
            lines.append(
                f"| {row['pair']} | {row['common_sessions']} | "
                f"{row['ridge_minus_baseline_spread_cs_demeaned_residual']} | "
                f"{row['ridge_minus_baseline_short_cs_contribution']} |"
            )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Training uses only prior validation sessions with an h5 label embargo.",
            "- Ridge is used instead of naked OLS to stabilize correlated feature weights.",
            "- This is still validation-only research; no test-window performance is computed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _coefficient_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return pd.DataFrame(columns=_coefficient_columns())
    summary = (
        coefficients.groupby(["variant", "residual_target", "ridge_alpha", "feature"], sort=True)
        .agg(
            folds=("model_fold_id", "nunique"),
            mean_coefficient=("coefficient", "mean"),
            median_coefficient=("coefficient", "median"),
            mean_abs_coefficient=("coefficient", lambda x: float(np.abs(x).mean())),
            positive_share=("coefficient", lambda x: float((x > 0).mean())),
        )
        .reset_index()
        .sort_values(["variant", "residual_target", "mean_abs_coefficient"], ascending=[True, True, False])
    )
    summary["test_window_used"] = False
    return summary[_coefficient_columns()]


def _prediction_columns() -> list[str]:
    return [
        "session_date",
        "variant",
        "symbol",
        "residual_target",
        "ridge_alpha",
        "model_fold_id",
        "ridge_predicted_residual",
        RIDGE_SELECTOR,
        BASELINE_SELECTOR,
        TARGET_BETA_RESIDUAL,
        TARGET_CS_DEMEANED,
        TARGET_RISK_NEUTRAL,
        "forward_return_5d",
        "test_window_used",
    ]


def _fold_columns() -> list[str]:
    return [
        "variant",
        "residual_target",
        "ridge_alpha",
        "model_fold_id",
        "train_start_session",
        "train_end_session",
        "predict_start_session",
        "predict_end_session",
        "train_sessions",
        "predict_sessions",
        "train_rows",
        "predict_rows",
        "label_embargo_sessions",
        "intercept",
        "test_window_used",
    ]


def _coefficient_columns() -> list[str]:
    return [
        "variant",
        "residual_target",
        "ridge_alpha",
        "feature",
        "folds",
        "mean_coefficient",
        "median_coefficient",
        "mean_abs_coefficient",
        "positive_share",
        "test_window_used",
    ]


def _selector_daily_columns() -> list[str]:
    return [
        "session_date",
        "variant",
        "residual_target",
        "ridge_alpha",
        "selector",
        "eligible_names",
        "selected_names",
        "selected_mean_target_return",
        "short_contribution_target",
        "selected_negative_target_share",
        "short_contribution_beta_residual",
        "short_contribution_cs_demeaned_residual",
        "short_contribution_risk_neutral_residual",
        "short_contribution_return",
        "oracle_short_contribution_target",
        "oracle_overlap_rate",
        "rank_ic_selector_vs_target",
        "test_window_used",
    ]


def _selector_summary_columns() -> list[str]:
    return [
        "variant",
        "residual_target",
        "ridge_alpha",
        "selector",
        "sessions",
        "mean_short_contribution_target",
        "target_hit_rate",
        "mean_selected_negative_target_share",
        "mean_short_contribution_beta_residual",
        "mean_short_contribution_cs_demeaned_residual",
        "mean_short_contribution_risk_neutral_residual",
        "mean_short_contribution_return",
        "mean_oracle_short_contribution_target",
        "mean_oracle_overlap_rate",
        "mean_rank_ic_selector_vs_target",
        "rank_ic_negative_share",
        "test_window_used",
    ]


def _common_portfolio_columns() -> list[str]:
    return [
        "pair",
        "residual_target",
        "ridge_alpha",
        "common_sessions",
        "baseline_spread_return",
        "baseline_spread_cs_demeaned_residual",
        "baseline_short_cs_demeaned_residual_contribution",
        "baseline_long_cs_demeaned_residual",
        "baseline_mean_abs_net_beta",
        "ridge_spread_return",
        "ridge_spread_cs_demeaned_residual",
        "ridge_short_cs_demeaned_residual_contribution",
        "ridge_long_cs_demeaned_residual",
        "ridge_mean_abs_net_beta",
        "ridge_minus_baseline_spread_cs_demeaned_residual",
        "ridge_minus_baseline_short_cs_contribution",
        "test_window_used",
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pure-alpha Phase 4M Ridge short selector.")
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_PHASE3_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--long-variant", action="append", dest="long_variants")
    parser.add_argument("--short-variant", action="append", dest="short_variants")
    parser.add_argument("--residual-target", action="append", dest="residual_targets")
    parser.add_argument("--feature", action="append", dest="features")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--initial-train-sessions", type=int, default=252)
    parser.add_argument("--label-embargo-sessions", type=int, default=5)
    parser.add_argument("--block-sessions", type=int, default=126)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--min-regression-rows", type=int, default=80)
    parser.add_argument("--min-train-rows", type=int, default=1_000)
    args = parser.parse_args(argv)
    result = build_phase4m_ridge_short_selector_artifacts(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        long_variants=tuple(args.long_variants or DEFAULT_LONG_VARIANTS),
        short_variants=tuple(args.short_variants or DEFAULT_SHORT_VARIANTS),
        residual_targets=tuple(args.residual_targets or DEFAULT_RESIDUAL_TARGETS),
        features=tuple(args.features or RIDGE_FEATURES),
        ridge_alpha=args.ridge_alpha,
        candidate_count=args.candidate_count,
        initial_train_sessions=args.initial_train_sessions,
        label_embargo_sessions=args.label_embargo_sessions,
        block_sessions=args.block_sessions,
        max_train_rows=args.max_train_rows,
        min_regression_rows=args.min_regression_rows,
        min_train_rows=args.min_train_rows,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
