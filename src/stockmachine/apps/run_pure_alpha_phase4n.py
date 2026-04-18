from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4i import (
    TARGET_BETA_RESIDUAL,
    TARGET_CS_DEMEANED,
    TARGET_RISK_NEUTRAL,
)
from stockmachine.apps.run_pure_alpha_phase4m import (
    BASELINE_SELECTOR,
    DEFAULT_PHASE3_SIGNAL_PANEL,
    RIDGE_FEATURES,
    _load_ridge_panel,
    _standardize,
    _train_predict_frames,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4n_pca_ridge_standalone_short_20260418"
DEFAULT_SHORT_VARIANT = "adv30m_clean_core_beta_full"
DEFAULT_RESIDUAL_TARGET = TARGET_CS_DEMEANED
DEFAULT_COMPONENT_COUNTS = (3,)
PCA_RIDGE_SELECTOR = "pca_ridge_short_score"


def build_phase4n_pca_ridge_standalone_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_PHASE3_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    residual_target: str = DEFAULT_RESIDUAL_TARGET,
    features: Sequence[str] = RIDGE_FEATURES,
    pca_components: Sequence[int] = DEFAULT_COMPONENT_COUNTS,
    ridge_alpha: float = 10.0,
    candidate_count: int = 30,
    initial_train_sessions: int = 252,
    label_embargo_sessions: int = 5,
    block_sessions: int = 126,
    max_train_rows: int = 200_000,
    min_regression_rows: int = 80,
    min_train_rows: int = 1_000,
) -> dict[str, Any]:
    """Run a validation-only standalone PCA-Ridge short-selector diagnostic."""

    _validate_settings(
        features=features,
        pca_components=pca_components,
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
    panel = _load_ridge_panel(
        signal_panel_path,
        variant_names=(short_variant,),
        min_regression_rows=min_regression_rows,
    )
    predictions, folds, loadings = _rolling_pca_ridge_predictions(
        panel,
        short_variant=short_variant,
        residual_target=residual_target,
        features=features,
        pca_components=tuple(sorted(set(pca_components))),
        ridge_alpha=ridge_alpha,
        initial_train_sessions=initial_train_sessions,
        label_embargo_sessions=label_embargo_sessions,
        block_sessions=block_sessions,
        max_train_rows=max_train_rows,
        min_train_rows=min_train_rows,
    )
    selector_daily = _selector_daily(
        predictions,
        residual_target=residual_target,
        candidate_count=candidate_count,
    )
    selector_summary = _selector_summary(selector_daily)
    paths = _write_outputs(
        output_dir,
        predictions=predictions,
        folds=folds,
        loadings=loadings,
        selector_daily=selector_daily,
        selector_summary=selector_summary,
    )
    memo_path = output_dir / "phase4n_pca_ridge_standalone_short_memo.md"
    memo_path.write_text(
        _memo(
            selector_summary,
            loadings,
            short_variant=short_variant,
            residual_target=residual_target,
            features=features,
            pca_components=pca_components,
            ridge_alpha=ridge_alpha,
            candidate_count=candidate_count,
        ),
        encoding="utf-8",
    )
    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "short_variant": short_variant,
        "residual_target": residual_target,
        "features": list(features),
        "pca_components": list(tuple(sorted(set(pca_components)))),
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
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "rolling_pca_ridge_standalone_short_selector_with_embargo",
        "primary_evaluation": "standalone_short_selector_performance",
        "portfolio_diagnostics_computed": False,
        **paths,
    }
    (output_dir / "phase4n_pca_ridge_standalone_short_rollup.json").write_text(
        json.dumps(rollup, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return rollup


def _validate_settings(
    *,
    features: Sequence[str],
    pca_components: Sequence[int],
    ridge_alpha: float,
    candidate_count: int,
    initial_train_sessions: int,
    label_embargo_sessions: int,
    block_sessions: int,
    max_train_rows: int,
    min_regression_rows: int,
    min_train_rows: int,
) -> None:
    if not features:
        raise ValueError("At least one feature is required.")
    if not pca_components:
        raise ValueError("At least one PCA component count is required.")
    if any(component <= 0 for component in pca_components):
        raise ValueError("PCA component counts must be positive.")
    if max(pca_components) > len(features):
        raise ValueError("PCA component count cannot exceed feature count.")
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


def _rolling_pca_ridge_predictions(
    panel: pd.DataFrame,
    *,
    short_variant: str,
    residual_target: str,
    features: Sequence[str],
    pca_components: Sequence[int],
    ridge_alpha: float,
    initial_train_sessions: int,
    label_embargo_sessions: int,
    block_sessions: int,
    max_train_rows: int,
    min_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []
    variant_panel = panel.loc[panel["variant"] == short_variant].copy()
    sessions = sorted(variant_panel["session_date"].unique())
    start = initial_train_sessions + label_embargo_sessions
    fold_id = 0
    max_components = max(pca_components)
    while start < len(sessions):
        train_sessions = sessions[: start - label_embargo_sessions]
        predict_sessions = sessions[start : start + block_sessions]
        if not predict_sessions:
            break
        train, predict = _train_predict_frames(
            variant_panel,
            target_name=residual_target,
            features=features,
            train_sessions=train_sessions,
            predict_sessions=predict_sessions,
            max_train_rows=max_train_rows,
        )
        if len(train) < min_train_rows or predict.empty:
            start += block_sessions
            fold_id += 1
            continue
        train_means = train[list(features)].astype(float).mean(axis=0)
        train_stds = train[list(features)].astype(float).std(axis=0, ddof=0).replace(0.0, 1.0)
        x_train = _standardize(train[list(features)], train_means, train_stds)
        x_predict = _standardize(predict[list(features)], train_means, train_stds)
        pca = PCA(n_components=max_components, svd_solver="full")
        train_scores = pca.fit_transform(x_train.to_numpy(dtype=float))
        predict_scores = pca.transform(x_predict.to_numpy(dtype=float))
        target = train[residual_target].astype(float).to_numpy(dtype=float)
        for feature_index, feature in enumerate(features):
            for component_index in range(max_components):
                weight = float(pca.components_[component_index, feature_index])
                loading = float(weight * np.sqrt(pca.explained_variance_[component_index]))
                loading_rows.append(
                    {
                        "variant": short_variant,
                        "residual_target": residual_target,
                        "ridge_alpha": float(ridge_alpha),
                        "model_fold_id": int(fold_id),
                        "feature": feature,
                        "component": f"PC{component_index + 1}",
                        "component_weight": weight,
                        "loading": loading,
                        "explained_variance_ratio": float(
                            pca.explained_variance_ratio_[component_index]
                        ),
                        "test_window_used": False,
                    }
                )
        for component_count in pca_components:
            model = Ridge(alpha=ridge_alpha)
            model.fit(train_scores[:, :component_count], target)
            prediction = predict.copy()
            prediction["residual_target"] = residual_target
            prediction["ridge_alpha"] = float(ridge_alpha)
            prediction["pca_components"] = int(component_count)
            prediction["model_fold_id"] = int(fold_id)
            prediction["pca_predicted_residual"] = model.predict(
                predict_scores[:, :component_count]
            )
            prediction[PCA_RIDGE_SELECTOR] = -prediction["pca_predicted_residual"]
            prediction["test_window_used"] = False
            prediction_frames.append(prediction[_prediction_columns()])
            fold_rows.append(
                {
                    "variant": short_variant,
                    "residual_target": residual_target,
                    "ridge_alpha": float(ridge_alpha),
                    "pca_components": int(component_count),
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
        start += block_sessions
        fold_id += 1
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame(columns=_prediction_columns())
    )
    folds = pd.DataFrame(fold_rows, columns=_fold_columns())
    loadings = pd.DataFrame(loading_rows, columns=_loading_columns())
    return predictions, folds, loadings


def _selector_daily(
    predictions: pd.DataFrame,
    *,
    residual_target: str,
    candidate_count: int,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=_selector_daily_columns())
    rows: list[dict[str, Any]] = []
    for (variant, target_name, ridge_alpha, pca_components, session_date), group in predictions.groupby(
        ["variant", "residual_target", "ridge_alpha", "pca_components", "session_date"],
        sort=True,
    ):
        if target_name != residual_target or len(group) < candidate_count:
            continue
        rows.extend(
            _session_selector_rows(
                group,
                selector=PCA_RIDGE_SELECTOR,
                selector_label=f"pca_ridge_pc{int(pca_components)}",
                pca_components=int(pca_components),
                candidate_count=candidate_count,
                variant=variant,
                target_name=target_name,
                ridge_alpha=float(ridge_alpha),
                session_date=session_date,
            )
        )
    baseline_frame = predictions.drop_duplicates(
        ["variant", "residual_target", "ridge_alpha", "session_date", "symbol"]
    )
    for (variant, target_name, ridge_alpha, session_date), group in baseline_frame.groupby(
        ["variant", "residual_target", "ridge_alpha", "session_date"],
        sort=True,
    ):
        if target_name != residual_target or len(group) < candidate_count:
            continue
        rows.extend(
            _session_selector_rows(
                group,
                selector=BASELINE_SELECTOR,
                selector_label=BASELINE_SELECTOR,
                pca_components=0,
                candidate_count=candidate_count,
                variant=variant,
                target_name=target_name,
                ridge_alpha=float(ridge_alpha),
                session_date=session_date,
            )
        )
    return pd.DataFrame(rows, columns=_selector_daily_columns()).sort_values(
        ["variant", "residual_target", "selector", "session_date"]
    )


def _session_selector_rows(
    group: pd.DataFrame,
    *,
    selector: str,
    selector_label: str,
    pca_components: int,
    candidate_count: int,
    variant: str,
    target_name: str,
    ridge_alpha: float,
    session_date: str,
) -> list[dict[str, Any]]:
    selected = group.dropna(subset=[selector]).sort_values(
        [selector, "symbol"],
        ascending=[False, True],
    ).head(candidate_count)
    if len(selected) < candidate_count:
        return []
    oracle = group.nsmallest(candidate_count, target_name)
    oracle_symbols = set(oracle["symbol"])
    all_scores = group[[selector, target_name]].dropna()
    rank_ic = all_scores[selector].corr(all_scores[target_name], method="spearman")
    return [
        {
            "session_date": session_date,
            "variant": variant,
            "residual_target": target_name,
            "ridge_alpha": float(ridge_alpha),
            "pca_components": int(pca_components),
            "selector": selector_label,
            "eligible_names": int(len(group)),
            "selected_names": int(len(selected)),
            "selected_mean_target_return": float(selected[target_name].mean()),
            "short_contribution_target": float(-selected[target_name].mean()),
            "selected_negative_target_share": float((selected[target_name] < 0).mean()),
            "short_contribution_beta_residual": float(-selected[TARGET_BETA_RESIDUAL].mean()),
            "short_contribution_cs_demeaned_residual": float(-selected[TARGET_CS_DEMEANED].mean()),
            "short_contribution_risk_neutral_residual": float(
                -selected[TARGET_RISK_NEUTRAL].mean()
            ),
            "short_contribution_return": float(-selected["forward_return_5d"].mean()),
            "oracle_short_contribution_target": float(-oracle[target_name].mean()),
            "oracle_overlap_rate": float(
                len(set(selected["symbol"]).intersection(oracle_symbols)) / candidate_count
            ),
            "rank_ic_selector_vs_target": float(rank_ic),
            "test_window_used": False,
        }
    ]


def _selector_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=_selector_summary_columns())
    rows = []
    for (variant, target_name, ridge_alpha, pca_components, selector), group in daily.groupby(
        ["variant", "residual_target", "ridge_alpha", "pca_components", "selector"],
        sort=True,
    ):
        contribution = group["short_contribution_target"]
        rows.append(
            {
                "variant": variant,
                "residual_target": target_name,
                "ridge_alpha": float(ridge_alpha),
                "pca_components": int(pca_components),
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
        ["mean_short_contribution_target", "target_hit_rate"],
        ascending=[False, False],
    )


def _write_outputs(
    output_dir: Path,
    *,
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    loadings: pd.DataFrame,
    selector_daily: pd.DataFrame,
    selector_summary: pd.DataFrame,
) -> dict[str, str]:
    paths = {
        "predictions_artifact": output_dir / "phase4n_pca_ridge_predictions_validation.csv.gz",
        "fold_manifest_artifact": output_dir / "phase4n_pca_ridge_fold_manifest_validation.csv",
        "loadings_artifact": output_dir / "phase4n_pca_ridge_loadings_validation.csv",
        "selector_daily_artifact": output_dir / "phase4n_pca_ridge_selector_daily_validation.csv",
        "selector_summary_artifact": output_dir
        / "phase4n_pca_ridge_selector_summary_validation.csv",
    }
    predictions.to_csv(paths["predictions_artifact"], index=False, compression="gzip")
    folds.to_csv(paths["fold_manifest_artifact"], index=False)
    loadings.to_csv(paths["loadings_artifact"], index=False)
    selector_daily.to_csv(paths["selector_daily_artifact"], index=False)
    selector_summary.to_csv(paths["selector_summary_artifact"], index=False)
    return {key: path.as_posix() for key, path in paths.items()}


def _memo(
    selector_summary: pd.DataFrame,
    loadings: pd.DataFrame,
    *,
    short_variant: str,
    residual_target: str,
    features: Sequence[str],
    pca_components: Sequence[int],
    ridge_alpha: float,
    candidate_count: int,
) -> str:
    lines = [
        "# Pure Alpha Phase 4N PCA-Ridge Standalone Short Selector Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "Validation-only standalone short-selector test for route 1: replace the raw "
        "correlated Ridge inputs with train-window PCA components before Ridge.",
        "",
        "No beta-matched portfolio diagnostics are computed in this pass.",
        "",
        "## Model",
        "",
        f"- short variant: `{short_variant}`",
        f"- residual target: `{residual_target}`",
        f"- PCA component counts: `{', '.join(str(x) for x in pca_components)}`",
        f"- Ridge alpha: `{ridge_alpha}`",
        f"- selected names per daily diagnostic: `{candidate_count}`",
        f"- raw features before PCA: `{', '.join(features)}`",
        "- baseline: `short_core_plus_overextension` on the same prediction sessions",
        "",
        "## Selector Summary",
        "",
        "| Selector | PCs | Contribution | Hit Rate | Negative Share | RankIC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if selector_summary.empty:
        lines.append("| n/a | 0 | n/a | n/a | n/a | n/a |")
    else:
        for _, row in selector_summary.iterrows():
            lines.append(
                f"| {row['selector']} | {row['pca_components']} | "
                f"{row['mean_short_contribution_target']} | {row['target_hit_rate']} | "
                f"{row['mean_selected_negative_target_share']} | "
                f"{row['mean_rank_ic_selector_vs_target']} |"
            )
    if not loadings.empty:
        loading_summary = (
            loadings.groupby(["feature", "component"], sort=True)
            .agg(mean_loading=("loading", "mean"))
            .reset_index()
        )
        lines.extend(
            [
                "",
                "## Mean PCA Loadings",
                "",
                "| Feature | Component | Mean Loading |",
                "|---|---|---:|",
            ]
        )
        for _, row in loading_summary.iterrows():
            lines.append(
                f"| `{row['feature']}` | {row['component']} | {row['mean_loading']} |"
            )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- PCA is fit inside each train window only, then applied to the prediction block.",
            "- Training uses only prior validation sessions with an h5 label embargo.",
            "- This is standalone selector research only; no test-window performance is computed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _prediction_columns() -> list[str]:
    return [
        "session_date",
        "variant",
        "symbol",
        "residual_target",
        "ridge_alpha",
        "pca_components",
        "model_fold_id",
        "pca_predicted_residual",
        PCA_RIDGE_SELECTOR,
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
        "pca_components",
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


def _loading_columns() -> list[str]:
    return [
        "variant",
        "residual_target",
        "ridge_alpha",
        "model_fold_id",
        "feature",
        "component",
        "component_weight",
        "loading",
        "explained_variance_ratio",
        "test_window_used",
    ]


def _selector_daily_columns() -> list[str]:
    return [
        "session_date",
        "variant",
        "residual_target",
        "ridge_alpha",
        "pca_components",
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
        "pca_components",
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run pure-alpha Phase 4N PCA-Ridge standalone short selector."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_PHASE3_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--short-variant", default=DEFAULT_SHORT_VARIANT)
    parser.add_argument("--residual-target", default=DEFAULT_RESIDUAL_TARGET)
    parser.add_argument("--feature", action="append", dest="features")
    parser.add_argument("--pca-components", action="append", type=int, dest="pca_components")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--initial-train-sessions", type=int, default=252)
    parser.add_argument("--label-embargo-sessions", type=int, default=5)
    parser.add_argument("--block-sessions", type=int, default=126)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--min-regression-rows", type=int, default=80)
    parser.add_argument("--min-train-rows", type=int, default=1_000)
    args = parser.parse_args(argv)
    result = build_phase4n_pca_ridge_standalone_artifacts(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        short_variant=args.short_variant,
        residual_target=args.residual_target,
        features=tuple(args.features or RIDGE_FEATURES),
        pca_components=tuple(args.pca_components or DEFAULT_COMPONENT_COUNTS),
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
