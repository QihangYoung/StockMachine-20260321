from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import (
    DEFAULT_VARIANTS,
    MODEL_FEATURES,
    RESEARCH_ROOT,
    TARGET_COLUMN,
)
from stockmachine.apps.run_pure_alpha_phase4e import (
    _target_independence_table,
)
from stockmachine.apps.run_pure_alpha_phase4f import (
    DEFAULT_PHASE3_SIGNAL_PANEL,
    _assign_feature_bins,
    _load_posterior_panel,
)
from stockmachine.apps.run_pure_alpha_phase4g import (
    _baseline_distribution_table,
    _feature_bin_utility_table,
    _feature_utility_summary,
    _robust_stats,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4i_residual_target_study_20260418"
DEFAULT_TARGET_VARIANTS = ("top1000_clean_core_beta_full",)
TARGET_BETA_RESIDUAL = "beta_residual"
TARGET_CS_DEMEANED = "cs_demeaned_beta_residual"
TARGET_RISK_NEUTRAL = "risk_neutral_beta_residual"
TARGET_STYLE_NEUTRAL = "style_neutral_beta_residual"
RESIDUAL_TARGETS = (
    TARGET_BETA_RESIDUAL,
    TARGET_CS_DEMEANED,
    TARGET_RISK_NEUTRAL,
    TARGET_STYLE_NEUTRAL,
)
RISK_NEUTRAL_FEATURES = (
    "beta",
    "lagged_close_log",
    "trailing_median_dollar_volume_20_log",
    "liquidity_rank",
)
STYLE_NEUTRAL_FEATURES = (
    *RISK_NEUTRAL_FEATURES,
    "return_5d",
    "momentum_20d",
    "momentum_60d",
    "beta_residual_momentum_20d",
    "vol_adjusted_momentum_20d",
)


def build_phase4i_residual_target_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_PHASE3_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant_names: Sequence[str] = DEFAULT_TARGET_VARIANTS,
    features: Sequence[str] = MODEL_FEATURES,
    n_bins: int = 10,
    max_independence_rows: int = 200_000,
    permutation_count: int = 5,
    loser_quantile: float = 0.20,
    winner_quantile: float = 0.20,
    right_tail_tolerance: float = 0.02,
    min_regression_rows: int = 80,
    random_state: int = 260321,
) -> dict[str, Any]:
    """Compare beta-only and simple multi-factor residual targets."""

    _validate_settings(
        variant_names=variant_names,
        features=features,
        n_bins=n_bins,
        max_independence_rows=max_independence_rows,
        permutation_count=permutation_count,
        loser_quantile=loser_quantile,
        winner_quantile=winner_quantile,
        right_tail_tolerance=right_tail_tolerance,
        min_regression_rows=min_regression_rows,
    )
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_posterior_panel(
        signal_panel_path,
        variant_names=variant_names,
        target_column=TARGET_COLUMN,
    )
    panel = _add_residual_targets(panel, min_regression_rows=min_regression_rows)

    prior_summary = _target_prior_summary(
        panel,
        target_names=RESIDUAL_TARGETS,
        loser_quantile=loser_quantile,
        winner_quantile=winner_quantile,
    )
    daily_prior = _target_daily_prior(panel, target_names=RESIDUAL_TARGETS)
    independence = _feature_independence_by_target(
        panel,
        target_names=RESIDUAL_TARGETS,
        features=features,
        n_bins=n_bins,
        max_rows=max_independence_rows,
        permutation_count=permutation_count,
        random_state=random_state,
    )
    robust_utility = _robust_feature_utility_by_target(
        panel,
        target_names=RESIDUAL_TARGETS,
        features=features,
        n_bins=n_bins,
        loser_quantile=loser_quantile,
        winner_quantile=winner_quantile,
        right_tail_tolerance=right_tail_tolerance,
    )

    prior_path = output_dir / "phase4i_residual_target_prior_summary_validation.csv"
    daily_path = output_dir / "phase4i_residual_target_daily_prior_validation.csv"
    independence_path = output_dir / "phase4i_feature_independence_by_target_validation.csv"
    utility_path = output_dir / "phase4i_robust_feature_utility_by_target_validation.csv"
    memo_path = output_dir / "phase4i_target_comparison_memo.md"
    rollup_path = output_dir / "phase4i_residual_target_rollup.json"

    prior_summary.to_csv(prior_path, index=False)
    daily_prior.to_csv(daily_path, index=False)
    independence.to_csv(independence_path, index=False)
    robust_utility.to_csv(utility_path, index=False)
    memo_path.write_text(
        _target_comparison_memo(
            prior_summary,
            robust_utility,
            min_regression_rows=min_regression_rows,
            n_bins=n_bins,
            loser_quantile=loser_quantile,
            winner_quantile=winner_quantile,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "variant_names": list(variant_names),
        "features": list(features),
        "residual_targets": list(RESIDUAL_TARGETS),
        "risk_neutral_features": list(RISK_NEUTRAL_FEATURES),
        "style_neutral_features": list(STYLE_NEUTRAL_FEATURES),
        "n_bins": int(n_bins),
        "max_independence_rows": int(max_independence_rows),
        "permutation_count": int(permutation_count),
        "loser_quantile": float(loser_quantile),
        "winner_quantile": float(winner_quantile),
        "right_tail_tolerance": float(right_tail_tolerance),
        "min_regression_rows": int(min_regression_rows),
        "panel_rows_loaded": int(len(panel)),
        "prior_summary_rows": int(len(prior_summary)),
        "daily_prior_rows": int(len(daily_prior)),
        "feature_independence_rows": int(len(independence)),
        "robust_feature_utility_rows": int(len(robust_utility)),
        "prior_summary_artifact": prior_path.as_posix(),
        "daily_prior_artifact": daily_path.as_posix(),
        "feature_independence_artifact": independence_path.as_posix(),
        "robust_feature_utility_artifact": utility_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "cross_sectional_residual_target_comparison",
        "limitations": [
            "Sector-neutral target is not computed because the current Phase 3 panel has no sector field.",
            "Style-neutral residuals intentionally remove some candidate alpha features and should be interpreted as transfer diagnostics.",
            "Residual neutralization is cross-sectional within validation sessions only.",
            "No test-window performance is computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _validate_settings(
    *,
    variant_names: Sequence[str],
    features: Sequence[str],
    n_bins: int,
    max_independence_rows: int,
    permutation_count: int,
    loser_quantile: float,
    winner_quantile: float,
    right_tail_tolerance: float,
    min_regression_rows: int,
) -> None:
    if not variant_names:
        raise ValueError("At least one variant is required.")
    if not features:
        raise ValueError("At least one feature is required.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if max_independence_rows <= 0:
        raise ValueError("max_independence_rows must be positive.")
    if permutation_count <= 0:
        raise ValueError("permutation_count must be positive.")
    if not 0 < loser_quantile < 1:
        raise ValueError("loser_quantile must be between 0 and 1.")
    if not 0 < winner_quantile < 1:
        raise ValueError("winner_quantile must be between 0 and 1.")
    if loser_quantile + winner_quantile >= 1:
        raise ValueError("loser_quantile + winner_quantile must be below 1.")
    if right_tail_tolerance < 0:
        raise ValueError("right_tail_tolerance must be non-negative.")
    if min_regression_rows <= 0:
        raise ValueError("min_regression_rows must be positive.")


def _add_residual_targets(panel: pd.DataFrame, *, min_regression_rows: int) -> pd.DataFrame:
    frame = panel.copy()
    frame[TARGET_BETA_RESIDUAL] = frame[TARGET_COLUMN]
    frame[TARGET_CS_DEMEANED] = frame[TARGET_COLUMN] - frame.groupby(
        ["variant", "session_date"]
    )[TARGET_COLUMN].transform("mean")
    frame[TARGET_RISK_NEUTRAL] = _daily_residualize(
        frame,
        target_column=TARGET_COLUMN,
        neutralizer_features=RISK_NEUTRAL_FEATURES,
        min_regression_rows=min_regression_rows,
    )
    frame[TARGET_STYLE_NEUTRAL] = _daily_residualize(
        frame,
        target_column=TARGET_COLUMN,
        neutralizer_features=STYLE_NEUTRAL_FEATURES,
        min_regression_rows=min_regression_rows,
    )
    return frame


def _daily_residualize(
    frame: pd.DataFrame,
    *,
    target_column: str,
    neutralizer_features: Sequence[str],
    min_regression_rows: int,
) -> pd.Series:
    residuals = pd.Series(np.nan, index=frame.index, dtype=float)
    columns = [target_column, *neutralizer_features]
    for _, group in frame.groupby(["variant", "session_date"], sort=False):
        valid = group[columns].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < min_regression_rows:
            continue
        y = valid[target_column].to_numpy(dtype=float)
        x = valid[list(neutralizer_features)].astype(float)
        x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0).replace(0.0, np.nan)
        x = x.dropna(axis=1)
        if x.empty:
            fitted = np.full_like(y, y.mean(), dtype=float)
        else:
            design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
            if len(valid) <= design.shape[1]:
                continue
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            fitted = design @ coefficients
        residuals.loc[valid.index] = y - fitted
    return residuals


def _target_prior_summary(
    panel: pd.DataFrame,
    *,
    target_names: Sequence[str],
    loser_quantile: float,
    winner_quantile: float,
) -> pd.DataFrame:
    rows = []
    for variant, group in panel.groupby("variant", sort=True):
        for target_name in target_names:
            target = group[target_name].dropna()
            if target.empty:
                continue
            bottom_cutoff = float(target.quantile(loser_quantile))
            top_cutoff = float(target.quantile(1.0 - winner_quantile))
            stats = _robust_stats(
                target,
                loser_quantile=loser_quantile,
                winner_quantile=winner_quantile,
                bottom_cutoff=bottom_cutoff,
                top_cutoff=top_cutoff,
            )
            rows.append(
                {
                    "variant": variant,
                    "residual_target": target_name,
                    "rows": int(len(target)),
                    "sessions": int(group.loc[target.index, "session_date"].nunique()),
                    **stats,
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows, columns=_prior_columns())


def _target_daily_prior(
    panel: pd.DataFrame,
    *,
    target_names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for variant, group in panel.groupby("variant", sort=True):
        for target_name in target_names:
            daily = (
                group.dropna(subset=[target_name])
                .groupby("session_date", sort=True)
                .agg(
                    residual_mean=(target_name, "mean"),
                    residual_median=(target_name, "median"),
                    benchmark_forward_return_5d=("benchmark_forward_return_5d", "first"),
                    rows=("symbol", "size"),
                )
                .reset_index()
            )
            for _, row in daily.iterrows():
                rows.append(
                    {
                        "variant": variant,
                        "residual_target": target_name,
                        "session_date": row["session_date"],
                        "rows": int(row["rows"]),
                        "residual_mean": float(row["residual_mean"]),
                        "residual_median": float(row["residual_median"]),
                        "benchmark_forward_return_5d": float(
                            row["benchmark_forward_return_5d"]
                        ),
                        "test_window_used": False,
                    }
                )
    return pd.DataFrame(rows)


def _feature_independence_by_target(
    panel: pd.DataFrame,
    *,
    target_names: Sequence[str],
    features: Sequence[str],
    n_bins: int,
    max_rows: int,
    permutation_count: int,
    random_state: int,
) -> pd.DataFrame:
    frames = []
    for variant, group in panel.groupby("variant", sort=True):
        for target_index, target_name in enumerate(target_names):
            sample = (
                group[["session_date", "variant", "symbol", *features, target_name]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(sample) > max_rows:
                sample = sample.sample(n=max_rows, random_state=random_state)
            table = _target_independence_table(
                sample,
                variant=variant,
                features=features,
                target_column=target_name,
                n_bins=n_bins,
                permutation_count=permutation_count,
                random_state=random_state + target_index * 1000,
            )
            table.insert(1, "residual_target", target_name)
            frames.append(table)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _robust_feature_utility_by_target(
    panel: pd.DataFrame,
    *,
    target_names: Sequence[str],
    features: Sequence[str],
    n_bins: int,
    loser_quantile: float,
    winner_quantile: float,
    right_tail_tolerance: float,
) -> pd.DataFrame:
    frames = []
    for variant, group in panel.groupby("variant", sort=True):
        for target_name in target_names:
            target_group = group.dropna(subset=[target_name]).copy()
            if target_group.empty:
                continue
            bottom_cutoff = float(target_group[target_name].quantile(loser_quantile))
            top_cutoff = float(target_group[target_name].quantile(1.0 - winner_quantile))
            baseline = _baseline_distribution_table(
                target_group,
                variant=variant,
                target_column=target_name,
                loser_quantile=loser_quantile,
                winner_quantile=winner_quantile,
                bottom_cutoff=bottom_cutoff,
                top_cutoff=top_cutoff,
            )
            bin_frames = []
            for feature in features:
                binned = _assign_feature_bins(target_group, feature=feature, n_bins=n_bins)
                if binned.empty:
                    continue
                bin_frames.append(
                    _feature_bin_utility_table(
                        binned,
                        variant=variant,
                        feature=feature,
                        target_column=target_name,
                        loser_quantile=loser_quantile,
                        winner_quantile=winner_quantile,
                        bottom_cutoff=bottom_cutoff,
                        top_cutoff=top_cutoff,
                    )
                )
            if not bin_frames:
                continue
            bins = pd.concat(bin_frames, ignore_index=True)
            summary = _feature_utility_summary(
                bins,
                baseline,
                right_tail_tolerance=right_tail_tolerance,
            )
            summary.insert(1, "residual_target", target_name)
            frames.append(summary)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _target_comparison_memo(
    prior_summary: pd.DataFrame,
    robust_utility: pd.DataFrame,
    *,
    min_regression_rows: int,
    n_bins: int,
    loser_quantile: float,
    winner_quantile: float,
) -> str:
    lines = [
        "# Pure Alpha Phase 4I Residual Target Comparison Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "This validation-only diagnostic compares the current beta-only residual "
        "target with simple cross-sectional residual targets.",
        "",
        "## Targets",
        "",
        "- `beta_residual`: current `forward_beta_residual_return_5d`;",
        "- `cs_demeaned_beta_residual`: current beta residual minus the daily "
        "cross-sectional mean;",
        "- `risk_neutral_beta_residual`: beta residual neutralized by beta, price, "
        "dollar-volume, and liquidity rank;",
        "- `style_neutral_beta_residual`: beta residual neutralized by risk "
        "features plus return, momentum, residual momentum, and volatility-adjusted momentum;",
        "- sector-neutral residual is not computed because the current panel has no sector field.",
        "",
        "## Method",
        "",
        f"- feature bins: `{n_bins}`;",
        f"- residual-loser cutoff: bottom `{loser_quantile}`;",
        f"- residual-winner cutoff: top `{winner_quantile}`;",
        f"- minimum daily regression rows: `{min_regression_rows}`;",
        "- Phase 4E-style feature independence and Phase 4G-style robust utility "
        "are rerun for each target;",
        "- no test-window performance is computed.",
        "",
        "## Target Priors",
        "",
        "| Variant | Target | Rows | Median | Trimmed Mean | Winsor Mean | Negative Share |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in prior_summary.iterrows():
        lines.append(
            f"| {row['variant']} | {row['residual_target']} | {row['rows']} | "
            f"{row['target_median']} | {row['target_trimmed_mean_10_90']} | "
            f"{row['target_winsorized_mean_5_95']} | "
            f"{row['negative_residual_share']} |"
        )
    lines.extend(
        [
            "",
            "## Top Robust Features By Target",
            "",
            "| Target | Feature | Label | Bin | Trimmed Short | Median Short | Tail Balance |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    if not robust_utility.empty:
        top = (
            robust_utility.sort_values(
                ["residual_target", "robust_rank_score"], ascending=[True, False]
            )
            .groupby("residual_target", sort=True)
            .head(8)
        )
        for _, row in top.iterrows():
            lines.append(
                f"| {row['residual_target']} | {row['feature']} | "
                f"{row['robust_usefulness_label']} | "
                f"{row['selected_feature_quantile_bin']} | "
                f"{row['all_short_contribution_trimmed_mean_10_90']} | "
                f"{row['all_short_contribution_median']} | {row['all_tail_balance']} |"
            )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Style-neutral residuals remove some candidate alpha fields, so feature "
            "collapse under that target is a transfer diagnostic rather than proof of no alpha.",
            "- Sector residuals require timestamp-safe sector metadata and are deferred.",
            "- Test lockbox remains closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _prior_columns() -> list[str]:
    return [
        "variant",
        "residual_target",
        "rows",
        "sessions",
        "target_mean",
        "target_median",
        "target_trimmed_mean_10_90",
        "target_winsorized_mean_5_95",
        "target_q10",
        "target_q25",
        "target_q75",
        "target_q90",
        "negative_residual_share",
        "bottom_loser_quantile",
        "bottom_loser_cutoff",
        "bottom_loser_share",
        "top_winner_quantile",
        "top_winner_cutoff",
        "top_winner_share",
        "tail_balance",
        "short_contribution_mean",
        "short_contribution_median",
        "short_contribution_trimmed_mean_10_90",
        "short_contribution_winsorized_mean_5_95",
        "test_window_used",
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run pure-alpha Phase 4I residual-target diagnostics."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_PHASE3_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--feature", action="append", dest="features")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--max-independence-rows", type=int, default=200_000)
    parser.add_argument("--permutation-count", type=int, default=5)
    parser.add_argument("--loser-quantile", type=float, default=0.20)
    parser.add_argument("--winner-quantile", type=float, default=0.20)
    parser.add_argument("--right-tail-tolerance", type=float, default=0.02)
    parser.add_argument("--min-regression-rows", type=int, default=80)
    parser.add_argument("--random-state", type=int, default=260321)
    args = parser.parse_args(argv)

    result = build_phase4i_residual_target_artifacts(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        variant_names=tuple(args.variants or DEFAULT_TARGET_VARIANTS),
        features=tuple(args.features or MODEL_FEATURES),
        n_bins=args.n_bins,
        max_independence_rows=args.max_independence_rows,
        permutation_count=args.permutation_count,
        loser_quantile=args.loser_quantile,
        winner_quantile=args.winner_quantile,
        right_tail_tolerance=args.right_tail_tolerance,
        min_regression_rows=args.min_regression_rows,
        random_state=args.random_state,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
