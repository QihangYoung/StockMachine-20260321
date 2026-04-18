from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import (
    MODEL_FEATURES,
    RESEARCH_ROOT,
    TARGET_COLUMN,
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


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4h_universe_prior_strength_20260418"
DEFAULT_VARIANTS = (
    "top500_clean_core_beta_full",
    "top1000_clean_core_beta_full",
    "adv50m_clean_core_beta_full",
    "adv30m_clean_core_beta_full",
    "adv20m_clean_core_beta_full",
    "adv10m_clean_core_beta_full",
)
DEFAULT_BUCKET_FEATURES = (
    "liquidity_rank",
    "beta",
    "lagged_close_log",
    "trailing_median_dollar_volume_20_log",
)


def build_phase4h_universe_prior_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_PHASE3_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant_names: Sequence[str] = DEFAULT_VARIANTS,
    features: Sequence[str] = MODEL_FEATURES,
    target_column: str = TARGET_COLUMN,
    n_bins: int = 10,
    loser_quantile: float = 0.20,
    winner_quantile: float = 0.20,
    right_tail_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Compare residual priors and robust short-feature utility by universe."""

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
    panel["forward_market_relative_return_5d"] = (
        panel["forward_return_5d"] - panel["benchmark_forward_return_5d"]
    )
    bottom_cutoffs = panel.groupby("variant")[target_column].quantile(loser_quantile)
    top_cutoffs = panel.groupby("variant")[target_column].quantile(1.0 - winner_quantile)

    prior_summary = _universe_prior_summary(
        panel,
        target_column=target_column,
        bottom_cutoffs=bottom_cutoffs,
        top_cutoffs=top_cutoffs,
        loser_quantile=loser_quantile,
        winner_quantile=winner_quantile,
    )
    by_year = _universe_prior_by_year(
        panel,
        target_column=target_column,
        bottom_cutoffs=bottom_cutoffs,
        top_cutoffs=top_cutoffs,
        loser_quantile=loser_quantile,
        winner_quantile=winner_quantile,
    )
    by_regime = _universe_prior_by_regime(
        panel,
        target_column=target_column,
        bottom_cutoffs=bottom_cutoffs,
        top_cutoffs=top_cutoffs,
        loser_quantile=loser_quantile,
        winner_quantile=winner_quantile,
    )
    by_bucket = _universe_prior_by_bucket(
        panel,
        target_column=target_column,
        bottom_cutoffs=bottom_cutoffs,
        top_cutoffs=top_cutoffs,
        loser_quantile=loser_quantile,
        winner_quantile=winner_quantile,
        n_bins=n_bins,
    )
    feature_utility = _feature_utility_by_universe(
        panel,
        features=features,
        target_column=target_column,
        loser_quantile=loser_quantile,
        winner_quantile=winner_quantile,
        n_bins=n_bins,
        right_tail_tolerance=right_tail_tolerance,
    )

    summary_path = output_dir / "phase4h_universe_prior_summary_validation.csv"
    year_path = output_dir / "phase4h_universe_prior_by_year_validation.csv"
    regime_path = output_dir / "phase4h_universe_prior_by_regime_validation.csv"
    bucket_path = output_dir / "phase4h_universe_prior_by_bucket_validation.csv"
    utility_path = output_dir / "phase4h_feature_utility_by_universe_validation.csv"
    memo_path = output_dir / "phase4h_universe_prior_memo.md"
    rollup_path = output_dir / "phase4h_universe_prior_rollup.json"

    prior_summary.to_csv(summary_path, index=False)
    by_year.to_csv(year_path, index=False)
    by_regime.to_csv(regime_path, index=False)
    by_bucket.to_csv(bucket_path, index=False)
    feature_utility.to_csv(utility_path, index=False)
    memo_path.write_text(
        _universe_prior_memo(
            prior_summary,
            feature_utility,
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
        "target_column": target_column,
        "n_bins": int(n_bins),
        "loser_quantile": float(loser_quantile),
        "winner_quantile": float(winner_quantile),
        "right_tail_tolerance": float(right_tail_tolerance),
        "panel_rows_loaded": int(len(panel)),
        "prior_summary_rows": int(len(prior_summary)),
        "prior_by_year_rows": int(len(by_year)),
        "prior_by_regime_rows": int(len(by_regime)),
        "prior_by_bucket_rows": int(len(by_bucket)),
        "feature_utility_rows": int(len(feature_utility)),
        "prior_summary_artifact": summary_path.as_posix(),
        "prior_by_year_artifact": year_path.as_posix(),
        "prior_by_regime_artifact": regime_path.as_posix(),
        "prior_by_bucket_artifact": bucket_path.as_posix(),
        "feature_utility_artifact": utility_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "universe_prior_and_robust_feature_utility_comparison",
        "limitations": [
            "Universe variants are limited to the current Phase 3 signal panel.",
            "Current validation universes are still provisional and not final survivorship-bias-free membership.",
            "Feature utility is diagnostic only and is not a beta-matched portfolio.",
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


def _universe_prior_summary(
    panel: pd.DataFrame,
    *,
    target_column: str,
    bottom_cutoffs: pd.Series,
    top_cutoffs: pd.Series,
    loser_quantile: float,
    winner_quantile: float,
) -> pd.DataFrame:
    rows = []
    for variant, group in panel.groupby("variant", sort=True):
        rows.append(
            _prior_row(
                group,
                variant=variant,
                target_column=target_column,
                bottom_cutoff=float(bottom_cutoffs.loc[variant]),
                top_cutoff=float(top_cutoffs.loc[variant]),
                loser_quantile=loser_quantile,
                winner_quantile=winner_quantile,
            )
        )
    return pd.DataFrame(rows, columns=_summary_columns())


def _universe_prior_by_year(
    panel: pd.DataFrame,
    *,
    target_column: str,
    bottom_cutoffs: pd.Series,
    top_cutoffs: pd.Series,
    loser_quantile: float,
    winner_quantile: float,
) -> pd.DataFrame:
    frame = panel.copy()
    frame["year"] = pd.to_datetime(frame["session_date"]).dt.year
    rows = []
    for (variant, year), group in frame.groupby(["variant", "year"], sort=True):
        row = _prior_row(
            group,
            variant=variant,
            target_column=target_column,
            bottom_cutoff=float(bottom_cutoffs.loc[variant]),
            top_cutoff=float(top_cutoffs.loc[variant]),
            loser_quantile=loser_quantile,
            winner_quantile=winner_quantile,
        )
        row["year"] = int(year)
        rows.append(row)
    return pd.DataFrame(rows, columns=["variant", "year", *_prior_metric_columns()])


def _universe_prior_by_regime(
    panel: pd.DataFrame,
    *,
    target_column: str,
    bottom_cutoffs: pd.Series,
    top_cutoffs: pd.Series,
    loser_quantile: float,
    winner_quantile: float,
) -> pd.DataFrame:
    rows = []
    for variant, group in panel.groupby("variant", sort=True):
        for regime, regime_group in _regime_slices(group):
            if regime_group.empty:
                continue
            row = _prior_row(
                regime_group,
                variant=variant,
                target_column=target_column,
                bottom_cutoff=float(bottom_cutoffs.loc[variant]),
                top_cutoff=float(top_cutoffs.loc[variant]),
                loser_quantile=loser_quantile,
                winner_quantile=winner_quantile,
            )
            row["market_regime"] = regime
            rows.append(row)
    return pd.DataFrame(rows, columns=["variant", "market_regime", *_prior_metric_columns()])


def _universe_prior_by_bucket(
    panel: pd.DataFrame,
    *,
    target_column: str,
    bottom_cutoffs: pd.Series,
    top_cutoffs: pd.Series,
    loser_quantile: float,
    winner_quantile: float,
    n_bins: int,
) -> pd.DataFrame:
    rows = []
    for variant, group in panel.groupby("variant", sort=True):
        for bucket_feature in DEFAULT_BUCKET_FEATURES:
            binned = _assign_feature_bins(group, feature=bucket_feature, n_bins=n_bins)
            if binned.empty:
                continue
            for bucket, bucket_group in binned.groupby("feature_quantile_bin", sort=True):
                row = _prior_row(
                    bucket_group,
                    variant=variant,
                    target_column=target_column,
                    bottom_cutoff=float(bottom_cutoffs.loc[variant]),
                    top_cutoff=float(top_cutoffs.loc[variant]),
                    loser_quantile=loser_quantile,
                    winner_quantile=winner_quantile,
                )
                row["bucket_feature"] = bucket_feature
                row["bucket"] = int(bucket)
                row["bucket_count"] = int(bucket_group["feature_bin_count"].max())
                row["bucket_feature_min"] = float(bucket_group[bucket_feature].min())
                row["bucket_feature_mean"] = float(bucket_group[bucket_feature].mean())
                row["bucket_feature_max"] = float(bucket_group[bucket_feature].max())
                rows.append(row)
    return pd.DataFrame(rows, columns=_bucket_columns())


def _feature_utility_by_universe(
    panel: pd.DataFrame,
    *,
    features: Sequence[str],
    target_column: str,
    loser_quantile: float,
    winner_quantile: float,
    n_bins: int,
    right_tail_tolerance: float,
) -> pd.DataFrame:
    summaries = []
    for variant, group in panel.groupby("variant", sort=True):
        bottom_cutoff = float(group[target_column].quantile(loser_quantile))
        top_cutoff = float(group[target_column].quantile(1.0 - winner_quantile))
        baseline = _baseline_distribution_table(
            group,
            variant=variant,
            target_column=target_column,
            loser_quantile=loser_quantile,
            winner_quantile=winner_quantile,
            bottom_cutoff=bottom_cutoff,
            top_cutoff=top_cutoff,
        )
        bin_frames = []
        for feature in features:
            binned = _assign_feature_bins(group, feature=feature, n_bins=n_bins)
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
        if not bin_frames:
            continue
        bins = pd.concat(bin_frames, ignore_index=True)
        summaries.append(
            _feature_utility_summary(
                bins,
                baseline,
                right_tail_tolerance=right_tail_tolerance,
            )
        )
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True)


def _prior_row(
    group: pd.DataFrame,
    *,
    variant: str,
    target_column: str,
    bottom_cutoff: float,
    top_cutoff: float,
    loser_quantile: float,
    winner_quantile: float,
) -> dict[str, object]:
    stats = _robust_stats(
        group[target_column],
        loser_quantile=loser_quantile,
        winner_quantile=winner_quantile,
        bottom_cutoff=bottom_cutoff,
        top_cutoff=top_cutoff,
    )
    daily = (
        group.groupby("session_date", sort=True)
        .agg(
            daily_residual_mean=(target_column, "mean"),
            daily_residual_median=(target_column, "median"),
            benchmark_forward_return_5d=("benchmark_forward_return_5d", "first"),
            eligible_names=("symbol", "size"),
            median_dollar_volume=("trailing_median_dollar_volume_20", "median"),
            median_beta=("beta", "median"),
        )
        .reset_index()
    )
    return {
        "variant": variant,
        "rows": int(len(group)),
        "sessions": int(group["session_date"].nunique()),
        "symbols": int(group["symbol"].nunique()),
        "mean_daily_names": float(daily["eligible_names"].mean()),
        "median_daily_names": float(daily["eligible_names"].median()),
        "median_trailing_dollar_volume_20": float(
            group["trailing_median_dollar_volume_20"].median()
        ),
        "median_beta": float(group["beta"].median()),
        "mean_forward_return_5d": float(group["forward_return_5d"].mean()),
        "mean_benchmark_forward_return_5d": float(
            group["benchmark_forward_return_5d"].mean()
        ),
        "mean_forward_market_relative_return_5d": float(
            group["forward_market_relative_return_5d"].mean()
        ),
        **stats,
        "daily_residual_mean_mean": float(daily["daily_residual_mean"].mean()),
        "daily_residual_median_mean": float(daily["daily_residual_median"].mean()),
        "daily_residual_mean_positive_share": float(
            (daily["daily_residual_mean"] > 0).mean()
        ),
        "daily_residual_median_positive_share": float(
            (daily["daily_residual_median"] > 0).mean()
        ),
        "daily_residual_mean_vs_benchmark_corr": float(
            daily["daily_residual_mean"].corr(daily["benchmark_forward_return_5d"])
        ),
        "test_window_used": False,
    }


def _regime_slices(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    benchmark = frame["benchmark_forward_return_5d"]
    return [
        ("all", frame),
        ("market_up_5d", frame[benchmark > 0]),
        ("market_down_5d", frame[benchmark < 0]),
    ]


def _universe_prior_memo(
    prior_summary: pd.DataFrame,
    feature_utility: pd.DataFrame,
    *,
    n_bins: int,
    loser_quantile: float,
    winner_quantile: float,
) -> str:
    lines = [
        "# Pure Alpha Phase 4H Universe Prior Strength Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "This validation-only diagnostic compares whether the current universe "
        "choice creates a positive residual-return prior that makes the short "
        "book structurally harder.",
        "",
        "## Method",
        "",
        f"- feature bins for utility rerank: `{n_bins}`;",
        f"- residual-loser cutoff: bottom `{loser_quantile}`;",
        f"- residual-winner cutoff: top `{winner_quantile}`;",
        "- universe robust priors are measured with median, trimmed mean, "
        "winsorized mean, and tail balance;",
        "- Phase 4G robust utility is rerun by universe variant;",
        "- no test-window performance is computed.",
        "",
        "## Universe Prior Summary",
        "",
        "| Variant | Rows | Sessions | Median | Trimmed Mean | Winsor Mean | Negative Share | Daily Mean Positive Share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in prior_summary.iterrows():
        lines.append(
            f"| {row['variant']} | {row['rows']} | {row['sessions']} | "
            f"{row['target_median']} | {row['target_trimmed_mean_10_90']} | "
            f"{row['target_winsorized_mean_5_95']} | "
            f"{row['negative_residual_share']} | "
            f"{row['daily_residual_mean_positive_share']} |"
        )
    lines.extend(
        [
            "",
            "## Top Robust Feature Per Universe",
            "",
            "| Variant | Feature | Label | Bin | Trimmed Short | Median Short | Tail Balance |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    if not feature_utility.empty:
        top = feature_utility.sort_values(
            ["variant", "robust_rank_score"], ascending=[True, False]
        ).groupby("variant", sort=True).head(5)
        for _, row in top.iterrows():
            lines.append(
                f"| {row['variant']} | {row['feature']} | "
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
            "- This is not final universe selection.",
            "- Broader universe candidates must still pass liquidity, capacity, borrow, "
            "and data-quality checks.",
            "- Current membership remains provisional and not final survivorship-bias-free data.",
            "- Test lockbox remains closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _prior_metric_columns() -> list[str]:
    return [
        "rows",
        "sessions",
        "symbols",
        "mean_daily_names",
        "median_daily_names",
        "median_trailing_dollar_volume_20",
        "median_beta",
        "mean_forward_return_5d",
        "mean_benchmark_forward_return_5d",
        "mean_forward_market_relative_return_5d",
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
        "daily_residual_mean_mean",
        "daily_residual_median_mean",
        "daily_residual_mean_positive_share",
        "daily_residual_median_positive_share",
        "daily_residual_mean_vs_benchmark_corr",
        "test_window_used",
    ]


def _summary_columns() -> list[str]:
    return ["variant", *_prior_metric_columns()]


def _bucket_columns() -> list[str]:
    return [
        "variant",
        "bucket_feature",
        "bucket",
        "bucket_count",
        "bucket_feature_min",
        "bucket_feature_mean",
        "bucket_feature_max",
        *_prior_metric_columns(),
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run pure-alpha Phase 4H universe-prior strength diagnostics."
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

    result = build_phase4h_universe_prior_artifacts(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        variant_names=tuple(args.variants or DEFAULT_VARIANTS),
        features=tuple(args.features or MODEL_FEATURES),
        n_bins=args.n_bins,
        loser_quantile=args.loser_quantile,
        winner_quantile=args.winner_quantile,
        right_tail_tolerance=args.right_tail_tolerance,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
