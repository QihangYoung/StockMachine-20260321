from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import (
    DEFAULT_PHASE3_SIGNAL_PANEL,
    DEFAULT_VARIANTS,
    MODEL_FEATURES,
    RESEARCH_ROOT,
    TARGET_COLUMN,
)
from stockmachine.apps.run_pure_alpha_phase4f import (
    BENCHMARK_FORWARD_COLUMN,
    FORWARD_RETURN_COLUMN,
    _assign_feature_bins,
    _load_posterior_panel,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4g_robust_feature_utility_20260418"
DEFAULT_FEATURES = MODEL_FEATURES


def build_phase4g_robust_feature_utility_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_PHASE3_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant_names: Sequence[str] = DEFAULT_VARIANTS,
    features: Sequence[str] = DEFAULT_FEATURES,
    target_column: str = TARGET_COLUMN,
    n_bins: int = 10,
    loser_quantile: float = 0.20,
    winner_quantile: float = 0.20,
    right_tail_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Rank features by robust conditional residual-return utility."""

    _validate_settings(
        variant_names=variant_names,
        features=features,
        n_bins=n_bins,
        loser_quantile=loser_quantile,
        winner_quantile=winner_quantile,
        right_tail_tolerance=right_tail_tolerance,
    )
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_posterior_panel(
        signal_panel_path,
        variant_names=variant_names,
        target_column=target_column,
    )
    bin_frames: list[pd.DataFrame] = []
    baseline_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    for variant in variant_names:
        variant_panel = panel[panel["variant"] == variant].copy()
        complete = (
            variant_panel[
                [
                    "session_date",
                    "variant",
                    "symbol",
                    *features,
                    target_column,
                    FORWARD_RETURN_COLUMN,
                    BENCHMARK_FORWARD_COLUMN,
                ]
            ]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        manifest_rows.append(
            {
                "variant": variant,
                "loaded_rows": int(len(variant_panel)),
                "complete_case_rows": int(len(complete)),
                "sample_start_session": _first_or_none(complete["session_date"])
                if not complete.empty
                else None,
                "sample_end_session": _last_or_none(complete["session_date"])
                if not complete.empty
                else None,
                "test_window_used": False,
            }
        )
        if complete.empty:
            continue
        bottom_cutoff = float(complete[target_column].quantile(loser_quantile))
        top_cutoff = float(complete[target_column].quantile(1.0 - winner_quantile))
        baseline_frames.append(
            _baseline_distribution_table(
                complete,
                variant=variant,
                target_column=target_column,
                loser_quantile=loser_quantile,
                winner_quantile=winner_quantile,
                bottom_cutoff=bottom_cutoff,
                top_cutoff=top_cutoff,
            )
        )
        for feature in features:
            binned = _assign_feature_bins(complete, feature=feature, n_bins=n_bins)
            if binned.empty:
                continue
            bin_frames.append(
                _feature_bin_utility_table(
                    binned,
                    variant=variant,
                    feature=feature,
                    target_column=target_column,
                    loser_quantile=loser_quantile,
                    winner_quantile=winner_quantile,
                    bottom_cutoff=bottom_cutoff,
                    top_cutoff=top_cutoff,
                )
            )

    bins = _concat_or_empty(bin_frames, _bin_columns())
    baseline = _concat_or_empty(baseline_frames, _baseline_columns())
    summary = _feature_utility_summary(
        bins,
        baseline,
        right_tail_tolerance=right_tail_tolerance,
    )
    manifest = pd.DataFrame(manifest_rows)

    bins_path = output_dir / "phase4g_robust_feature_bins_validation.csv"
    baseline_path = output_dir / "phase4g_robust_feature_baseline_validation.csv"
    summary_path = output_dir / "phase4g_robust_feature_summary_validation.csv"
    manifest_path = output_dir / "phase4g_robust_feature_sample_manifest_validation.csv"
    memo_path = output_dir / "phase4g_robust_feature_utility_memo.md"
    rollup_path = output_dir / "phase4g_robust_feature_utility_rollup.json"

    bins.to_csv(bins_path, index=False)
    baseline.to_csv(baseline_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    memo_path.write_text(
        _utility_memo(
            summary,
            baseline,
            manifest,
            n_bins=n_bins,
            loser_quantile=loser_quantile,
            winner_quantile=winner_quantile,
            right_tail_tolerance=right_tail_tolerance,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "variant_names": list(variant_names),
        "features": list(features),
        "target_column": target_column,
        "n_bins": int(n_bins),
        "loser_quantile": float(loser_quantile),
        "winner_quantile": float(winner_quantile),
        "right_tail_tolerance": float(right_tail_tolerance),
        "feature_panel_rows_loaded": int(len(panel)),
        "robust_feature_bin_rows": int(len(bins)),
        "baseline_rows": int(len(baseline)),
        "summary_rows": int(len(summary)),
        "sample_manifest_rows": int(len(manifest)),
        "robust_feature_bins_artifact": bins_path.as_posix(),
        "baseline_artifact": baseline_path.as_posix(),
        "summary_artifact": summary_path.as_posix(),
        "sample_manifest_artifact": manifest_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "nonlinear_feature_bin_robust_posterior_utility",
        "limitations": [
            "This is a robust feature diagnostic, not a beta-matched portfolio.",
            "Feature bins are formed on the validation sample only.",
            "Regime splits use future benchmark return for diagnosis only.",
            "No transaction costs, borrow costs, or test-window performance are computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _validate_settings(
    *,
    variant_names: Sequence[str],
    features: Sequence[str],
    n_bins: int,
    loser_quantile: float,
    winner_quantile: float,
    right_tail_tolerance: float,
) -> None:
    if not variant_names:
        raise ValueError("At least one variant is required.")
    if not features:
        raise ValueError("At least one feature is required.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if not 0 < loser_quantile < 1:
        raise ValueError("loser_quantile must be between 0 and 1.")
    if not 0 < winner_quantile < 1:
        raise ValueError("winner_quantile must be between 0 and 1.")
    if loser_quantile + winner_quantile >= 1:
        raise ValueError("loser_quantile + winner_quantile must be below 1.")
    if right_tail_tolerance < 0:
        raise ValueError("right_tail_tolerance must be non-negative.")


def _baseline_distribution_table(
    frame: pd.DataFrame,
    *,
    variant: str,
    target_column: str,
    loser_quantile: float,
    winner_quantile: float,
    bottom_cutoff: float,
    top_cutoff: float,
) -> pd.DataFrame:
    rows = []
    for regime, regime_frame in _regime_slices(frame):
        if regime_frame.empty:
            continue
        stats = _robust_stats(
            regime_frame[target_column],
            loser_quantile=loser_quantile,
            winner_quantile=winner_quantile,
            bottom_cutoff=bottom_cutoff,
            top_cutoff=top_cutoff,
        )
        rows.append(
            {
                "variant": variant,
                "market_regime": regime,
                "rows": int(len(regime_frame)),
                **stats,
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows, columns=_baseline_columns())


def _feature_bin_utility_table(
    frame: pd.DataFrame,
    *,
    variant: str,
    feature: str,
    target_column: str,
    loser_quantile: float,
    winner_quantile: float,
    bottom_cutoff: float,
    top_cutoff: float,
) -> pd.DataFrame:
    rows = []
    for regime, regime_frame in _regime_slices(frame):
        if regime_frame.empty:
            continue
        for feature_bin, group in regime_frame.groupby("feature_quantile_bin", sort=True):
            stats = _robust_stats(
                group[target_column],
                loser_quantile=loser_quantile,
                winner_quantile=winner_quantile,
                bottom_cutoff=bottom_cutoff,
                top_cutoff=top_cutoff,
            )
            rows.append(
                {
                    "variant": variant,
                    "feature": feature,
                    "market_regime": regime,
                    "feature_quantile_bin": int(feature_bin),
                    "feature_bin_count": int(group["feature_bin_count"].max()),
                    "feature_bin_side": _bin_side(
                        int(feature_bin), int(group["feature_bin_count"].max())
                    ),
                    "rows": int(len(group)),
                    "feature_min": float(group[feature].min()),
                    "feature_mean": float(group[feature].mean()),
                    "feature_max": float(group[feature].max()),
                    **stats,
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows, columns=_bin_columns())


def _robust_stats(
    target: pd.Series,
    *,
    loser_quantile: float,
    winner_quantile: float,
    bottom_cutoff: float,
    top_cutoff: float,
) -> dict[str, float]:
    values = pd.to_numeric(target, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {column: np.nan for column in _robust_stat_columns()}
    mean = float(values.mean())
    median = float(values.median())
    trimmed = _trimmed_mean(values, lower=0.10, upper=0.90)
    winsorized = _winsorized_mean(values, lower=0.05, upper=0.95)
    bottom_share = float((values <= bottom_cutoff).mean())
    top_share = float((values >= top_cutoff).mean())
    return {
        "target_mean": mean,
        "target_median": median,
        "target_trimmed_mean_10_90": trimmed,
        "target_winsorized_mean_5_95": winsorized,
        "target_q10": float(values.quantile(0.10)),
        "target_q25": float(values.quantile(0.25)),
        "target_q75": float(values.quantile(0.75)),
        "target_q90": float(values.quantile(0.90)),
        "negative_residual_share": float((values < 0).mean()),
        "bottom_loser_quantile": float(loser_quantile),
        "bottom_loser_cutoff": float(bottom_cutoff),
        "bottom_loser_share": bottom_share,
        "top_winner_quantile": float(winner_quantile),
        "top_winner_cutoff": float(top_cutoff),
        "top_winner_share": top_share,
        "tail_balance": float(bottom_share - top_share),
        "short_contribution_mean": float(-mean),
        "short_contribution_median": float(-median),
        "short_contribution_trimmed_mean_10_90": float(-trimmed),
        "short_contribution_winsorized_mean_5_95": float(-winsorized),
    }


def _trimmed_mean(values: pd.Series, *, lower: float, upper: float) -> float:
    lower_cut = values.quantile(lower)
    upper_cut = values.quantile(upper)
    trimmed = values[(values >= lower_cut) & (values <= upper_cut)]
    if trimmed.empty:
        return float(values.mean())
    return float(trimmed.mean())


def _winsorized_mean(values: pd.Series, *, lower: float, upper: float) -> float:
    lower_cut = values.quantile(lower)
    upper_cut = values.quantile(upper)
    return float(values.clip(lower=lower_cut, upper=upper_cut).mean())


def _regime_slices(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    benchmark = frame[BENCHMARK_FORWARD_COLUMN]
    return [
        ("all", frame),
        ("market_up_5d", frame[benchmark > 0]),
        ("market_down_5d", frame[benchmark < 0]),
    ]


def _feature_utility_summary(
    bins: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    right_tail_tolerance: float,
) -> pd.DataFrame:
    rows = []
    if bins.empty:
        return pd.DataFrame(columns=_summary_columns())
    baseline_lookup = {
        (row["variant"], row["market_regime"]): row for _, row in baseline.iterrows()
    }
    for (variant, feature), group in bins.groupby(["variant", "feature"], sort=True):
        all_group = group[group["market_regime"] == "all"]
        if all_group.empty:
            continue
        selected = all_group.sort_values(
            [
                "target_trimmed_mean_10_90",
                "target_median",
                "top_winner_share",
                "bottom_loser_share",
            ],
            ascending=[True, True, True, False],
        ).iloc[0]
        selected_bin = int(selected["feature_quantile_bin"])
        row: dict[str, object] = {
            "variant": variant,
            "feature": feature,
            "selected_feature_quantile_bin": selected_bin,
            "selected_feature_bin_side": selected["feature_bin_side"],
            "selected_rows_all": int(selected["rows"]),
            "test_window_used": False,
        }
        for regime in ("all", "market_up_5d", "market_down_5d"):
            regime_row = group[
                (group["market_regime"] == regime)
                & (group["feature_quantile_bin"] == selected_bin)
            ]
            baseline_row = baseline_lookup.get((variant, regime))
            _add_regime_metrics(row, regime, regime_row, baseline_row)
        row["regime_trimmed_short_positive_count"] = int(
            sum(
                1
                for regime in ("all", "market_up_5d", "market_down_5d")
                if _positive(row.get(f"{regime}_short_contribution_trimmed_mean_10_90"))
            )
        )
        row["regime_trimmed_edge_positive_count"] = int(
            sum(
                1
                for regime in ("all", "market_up_5d", "market_down_5d")
                if _positive(row.get(f"{regime}_trimmed_edge_vs_universe"))
            )
        )
        row["right_tail_controlled_all"] = bool(
            row["all_top_winner_excess_vs_universe"] <= right_tail_tolerance
        )
        row["robust_rank_score"] = _robust_rank_score(row)
        row["robust_usefulness_label"] = _robust_usefulness_label(row)
        rows.append(row)
    summary = pd.DataFrame(rows, columns=_summary_columns())
    if summary.empty:
        return summary
    return summary.sort_values(
        [
            "robust_usefulness_label",
            "robust_rank_score",
            "all_short_contribution_trimmed_mean_10_90",
            "all_tail_balance",
        ],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def _add_regime_metrics(
    row: dict[str, object],
    regime: str,
    regime_row: pd.DataFrame,
    baseline_row: pd.Series | None,
) -> None:
    prefix = regime
    if regime_row.empty or baseline_row is None:
        for column in _regime_summary_metric_names():
            row[f"{prefix}_{column}"] = np.nan
        return
    selected = regime_row.iloc[0]
    row[f"{prefix}_rows"] = int(selected["rows"])
    for column in _selected_metric_names():
        row[f"{prefix}_{column}"] = selected[column]
    row[f"{prefix}_median_edge_vs_universe"] = (
        baseline_row["target_median"] - selected["target_median"]
    )
    row[f"{prefix}_trimmed_edge_vs_universe"] = (
        baseline_row["target_trimmed_mean_10_90"] - selected["target_trimmed_mean_10_90"]
    )
    row[f"{prefix}_winsorized_edge_vs_universe"] = (
        baseline_row["target_winsorized_mean_5_95"]
        - selected["target_winsorized_mean_5_95"]
    )
    row[f"{prefix}_tail_balance_edge_vs_universe"] = (
        selected["tail_balance"] - baseline_row["tail_balance"]
    )
    row[f"{prefix}_top_winner_excess_vs_universe"] = (
        selected["top_winner_share"] - baseline_row["top_winner_share"]
    )


def _positive(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(float(value) > 0)


def _robust_rank_score(row: dict[str, object]) -> float:
    return float(
        row["all_short_contribution_trimmed_mean_10_90"]
        + row["all_short_contribution_median"]
        + row["all_short_contribution_winsorized_mean_5_95"]
        + 0.01 * row["all_tail_balance"]
        - 0.01 * max(0.0, row["all_top_winner_excess_vs_universe"])
    )


def _robust_usefulness_label(row: dict[str, object]) -> str:
    core_positive = (
        row["all_short_contribution_median"] > 0
        and row["all_short_contribution_trimmed_mean_10_90"] > 0
        and row["all_short_contribution_winsorized_mean_5_95"] > 0
    )
    relative_edge = (
        row["all_median_edge_vs_universe"] > 0
        and row["all_trimmed_edge_vs_universe"] > 0
        and row["all_winsorized_edge_vs_universe"] > 0
    )
    tail_good = row["all_tail_balance"] > 0
    right_ok = bool(row["right_tail_controlled_all"])
    stable = row["regime_trimmed_edge_positive_count"] >= 2
    if core_positive and relative_edge and tail_good and right_ok and stable:
        return "1_robust_short_candidate"
    if core_positive and relative_edge and tail_good:
        return "2_promising_with_tail_or_regime_risk"
    if relative_edge and tail_good:
        return "3_relative_edge_not_short_positive"
    if tail_good and row["all_bottom_loser_share"] > 0.23:
        return "4_tail_only_or_volatility_candidate"
    return "5_weak_or_unhelpful"


def _utility_memo(
    summary: pd.DataFrame,
    baseline: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    n_bins: int,
    loser_quantile: float,
    winner_quantile: float,
    right_tail_tolerance: float,
) -> str:
    lines = [
        "# Pure Alpha Phase 4G Robust Feature Utility Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "This is a validation-only diagnostic that re-ranks features using robust "
        "posterior metrics instead of relying on mean return or left-tail hit "
        "rate alone.",
        "",
        "## Method",
        "",
        f"- feature quantile bins: `{n_bins}`;",
        f"- left-tail residual-loser cutoff: bottom `{loser_quantile}`;",
        f"- right-tail residual-winner cutoff: top `{winner_quantile}`;",
        f"- right-tail tolerance versus universe: `{right_tail_tolerance}`;",
        "- primary robust metrics: median, 10/90 trimmed mean, 5/95 winsorized "
        "mean, left-tail share, right-tail share, and tail balance;",
        "- selected bin per feature: the validation bin with the lowest 10/90 "
        "trimmed residual mean;",
        "- no test-window performance is computed.",
        "",
        "## Sample",
        "",
        "| Variant | Loaded Rows | Complete-Case Rows | Start | End |",
        "|---|---:|---:|---|---|",
    ]
    for _, row in manifest.iterrows():
        lines.append(
            f"| {row['variant']} | {row['loaded_rows']} | {row['complete_case_rows']} | "
            f"{row['sample_start_session']} | {row['sample_end_session']} |"
        )
    lines.extend(
        [
            "",
            "## Universe Baseline",
            "",
            "| Variant | Regime | Median | Trimmed Mean | Winsorized Mean | Bottom Share | Top Share | Tail Balance |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in baseline.iterrows():
        lines.append(
            f"| {row['variant']} | {row['market_regime']} | {row['target_median']} | "
            f"{row['target_trimmed_mean_10_90']} | "
            f"{row['target_winsorized_mean_5_95']} | {row['bottom_loser_share']} | "
            f"{row['top_winner_share']} | {row['tail_balance']} |"
        )
    lines.extend(
        [
            "",
            "## Top Robust Feature Bins",
            "",
            "| Feature | Bin | Label | Trimmed Short | Median Short | Winsor Short | Tail Balance | Top Winner Share | Regime Edge Count |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.head(20).iterrows():
        lines.append(
            f"| {row['feature']} | {row['selected_feature_quantile_bin']} | "
            f"{row['robust_usefulness_label']} | "
            f"{row['all_short_contribution_trimmed_mean_10_90']} | "
            f"{row['all_short_contribution_median']} | "
            f"{row['all_short_contribution_winsorized_mean_5_95']} | "
            f"{row['all_tail_balance']} | {row['all_top_winner_share']} | "
            f"{row['regime_trimmed_edge_positive_count']} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- A robust feature bin is not yet a beta-matched, cost-aware portfolio.",
            "- The selected bin is a diagnostic best bin on validation data, not a "
            "frozen trading rule.",
            "- Features whose edge comes only from left-tail hit rate are explicitly "
            "labeled as tail-only or volatility candidates.",
            "- Test lockbox remains closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _bin_side(feature_bin: int, bin_count: int) -> str:
    if feature_bin == 1:
        return "low_feature_tail"
    if feature_bin == bin_count:
        return "high_feature_tail"
    return "middle"


def _concat_or_empty(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    return pd.concat(non_empty, ignore_index=True)[columns]


def _robust_stat_columns() -> list[str]:
    return [
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
    ]


def _baseline_columns() -> list[str]:
    return [
        "variant",
        "market_regime",
        "rows",
        *_robust_stat_columns(),
        "test_window_used",
    ]


def _bin_columns() -> list[str]:
    return [
        "variant",
        "feature",
        "market_regime",
        "feature_quantile_bin",
        "feature_bin_count",
        "feature_bin_side",
        "rows",
        "feature_min",
        "feature_mean",
        "feature_max",
        *_robust_stat_columns(),
        "test_window_used",
    ]


def _selected_metric_names() -> list[str]:
    return [
        "rows",
        "target_mean",
        "target_median",
        "target_trimmed_mean_10_90",
        "target_winsorized_mean_5_95",
        "negative_residual_share",
        "bottom_loser_share",
        "top_winner_share",
        "tail_balance",
        "short_contribution_mean",
        "short_contribution_median",
        "short_contribution_trimmed_mean_10_90",
        "short_contribution_winsorized_mean_5_95",
    ]


def _regime_summary_metric_names() -> list[str]:
    return [
        *_selected_metric_names(),
        "median_edge_vs_universe",
        "trimmed_edge_vs_universe",
        "winsorized_edge_vs_universe",
        "tail_balance_edge_vs_universe",
        "top_winner_excess_vs_universe",
    ]


def _summary_columns() -> list[str]:
    columns = [
        "variant",
        "feature",
        "selected_feature_quantile_bin",
        "selected_feature_bin_side",
        "selected_rows_all",
    ]
    for regime in ("all", "market_up_5d", "market_down_5d"):
        columns.extend([f"{regime}_{name}" for name in _regime_summary_metric_names()])
    columns.extend(
        [
            "regime_trimmed_short_positive_count",
            "regime_trimmed_edge_positive_count",
            "right_tail_controlled_all",
            "robust_rank_score",
            "robust_usefulness_label",
            "test_window_used",
        ]
    )
    return columns


def _first_or_none(series: pd.Series) -> str | None:
    if len(series) == 0:
        return None
    return str(series.min())


def _last_or_none(series: pd.Series) -> str | None:
    if len(series) == 0:
        return None
    return str(series.max())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run pure-alpha Phase 4G robust feature utility diagnostics."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_PHASE3_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--feature", action="append", dest="features")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--loser-quantile", type=float, default=0.20)
    parser.add_argument("--winner-quantile", type=float, default=0.20)
    parser.add_argument("--right-tail-tolerance", type=float, default=0.02)
    args = parser.parse_args(argv)

    result = build_phase4g_robust_feature_utility_artifacts(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        variant_names=tuple(args.variants or DEFAULT_VARIANTS),
        features=tuple(args.features or DEFAULT_FEATURES),
        n_bins=args.n_bins,
        loser_quantile=args.loser_quantile,
        winner_quantile=args.winner_quantile,
        right_tail_tolerance=args.right_tail_tolerance,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
