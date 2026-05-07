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
)
from stockmachine.apps.run_pure_alpha_phase5b import _fmt_pct_like, _text_table, _utc_now
from stockmachine.apps.run_pure_alpha_phase5r import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_PHASE5R_ROOT,
    _add_price_only_falling_knife_scores,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase5s_price_only_long_leg_diagnostics_20260506"
DEFAULT_PHASE5R_POSITIONS = DEFAULT_PHASE5R_ROOT / "phase5r_positions_validation.csv.gz"
DEFAULT_CANDIDATE_COUNT = 80
CURRENT_PORTFOLIO = "current_sota"
PRIMARY_TARGET = "style_factor_sic2_residual"
LEFT_TAIL_THRESHOLD = -0.02
RIGHT_TAIL_THRESHOLD = 0.02

WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("full", "", ""),
    ("2015_peak_to_trough", "2015-06-04", "2015-08-21"),
    ("2016_jan_feb", "2016-01-05", "2016-02-09"),
    ("2017_spring_short_fail", "2017-03-20", "2017-05-31"),
    ("2018_aug_sep_short_fail", "2018-08-08", "2018-09-12"),
    ("2019_may", "2019-05-01", "2019-06-05"),
    ("2019_aug", "2019-08-01", "2019-08-31"),
)

TARGET_COLUMNS = (
    "forward_return_5d",
    "forward_beta_residual_return_5d",
    "sic2_neutral_beta_residual",
    "risk_factor_sic2_residual",
    "style_factor_sic2_residual",
    "style_factor_sic4_residual",
)

FEATURE_COLUMNS = (
    "reversal_5d_z",
    "momentum_20d_z",
    "momentum_60d_z",
    "beta_residual_momentum_20d_z",
    "vol_adjusted_momentum_20d_z",
    "beta",
    "liquidity_rank_score",
    "price_fk_sector_mom20_down",
    "price_fk_sector_mom60_down",
    "price_fk_sector_relative_mom20_down",
    "price_fk_sector_relative_mom60_down",
    "price_fk_residual_mom20_down",
    "price_fk_vol_adjusted_down",
    "price_fk_no_stabilization",
    "price_falling_knife_score",
    "price_still_falling_score",
    "price_stabilization_score",
    "price_fk_stillfall_penalty",
    "price_fk_stab_relief_penalty",
    "long_price_fk_soft_w100",
    "long_price_fk_stillfall_w100",
    "long_price_fk_stab_relief_w075",
)

PLOT_FEATURES = (
    "reversal_5d_z",
    "momentum_20d_z",
    "momentum_60d_z",
    "beta_residual_momentum_20d_z",
    "price_falling_knife_score",
    "price_still_falling_score",
    "price_stabilization_score",
    "price_fk_stab_relief_penalty",
)


def build_phase5s_price_only_long_leg_diagnostics_artifacts(
    *,
    positions_path: str | Path = DEFAULT_PHASE5R_POSITIONS,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    cik_mapping_path: str | Path = DEFAULT_CIK_MAPPING,
    sec_submissions_dir: str | Path = DEFAULT_SEC_SUBMISSIONS_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    long_variant: str = DEFAULT_LONG_VARIANT,
    short_variant: str = DEFAULT_SHORT_VARIANT,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    min_regression_rows: int = 80,
    min_dummy_count: int = 5,
) -> dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = pd.read_csv(positions_path, low_memory=False)
    if "test_window_used" in positions.columns and positions["test_window_used"].astype(bool).any():
        raise ValueError("Positions include test-window rows; refusing to run diagnostics.")
    selected_longs = _selected_current_sota_longs(positions)
    factor_panel = _build_price_only_factor_panel(
        signal_panel_path=signal_panel_path,
        cik_mapping_path=cik_mapping_path,
        sec_submissions_dir=sec_submissions_dir,
        long_variant=long_variant,
        short_variant=short_variant,
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )
    candidate_top = _top_by_session(
        factor_panel,
        score_column=DEFAULT_LONG_SCORE,
        count=int(candidate_count),
    )
    selected_enriched = _merge_selected_features(selected_longs, factor_panel)
    selected_bucket = _selected_outcome_bucket_summary(selected_enriched)
    window_profile = _selected_window_profile(selected_enriched)
    candidate_deciles = _candidate_feature_decile_summary(candidate_top)
    feature_shape = _feature_shape_summary(candidate_deciles)
    sector_stress = _stress_sector_concentration(selected_enriched)

    selected_path = output_dir / "phase5s_selected_long_enriched.csv.gz"
    selected_bucket_path = output_dir / "phase5s_selected_long_outcome_buckets.csv"
    window_profile_path = output_dir / "phase5s_selected_long_window_profile.csv"
    candidate_deciles_path = output_dir / "phase5s_candidate_feature_deciles.csv"
    feature_shape_path = output_dir / "phase5s_feature_shape_summary.csv"
    sector_stress_path = output_dir / "phase5s_stress_sector_loser_concentration.csv"
    plot_path = output_dir / "phase5s_feature_decile_plot.png"
    memo_path = output_dir / "phase5s_price_only_long_leg_diagnostics_memo.md"
    rollup_path = output_dir / "phase5s_rollup.json"

    selected_enriched.to_csv(selected_path, index=False, compression="gzip")
    selected_bucket.to_csv(selected_bucket_path, index=False)
    window_profile.to_csv(window_profile_path, index=False)
    candidate_deciles.to_csv(candidate_deciles_path, index=False)
    feature_shape.to_csv(feature_shape_path, index=False)
    sector_stress.to_csv(sector_stress_path, index=False)
    _plot_feature_deciles(candidate_deciles, output_path=plot_path)
    memo_path.write_text(
        _memo(
            selected_bucket=selected_bucket,
            window_profile=window_profile,
            feature_shape=feature_shape,
            sector_stress=sector_stress,
            plot_path=plot_path,
            candidate_count=candidate_count,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "positions_path": Path(positions_path).as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "long_variant": long_variant,
        "short_variant": short_variant,
        "portfolio": CURRENT_PORTFOLIO,
        "primary_target": PRIMARY_TARGET,
        "candidate_count": int(candidate_count),
        "left_tail_threshold": float(LEFT_TAIL_THRESHOLD),
        "right_tail_threshold": float(RIGHT_TAIL_THRESHOLD),
        "artifacts": {
            "selected_long_enriched": selected_path.as_posix(),
            "selected_long_outcome_buckets": selected_bucket_path.as_posix(),
            "selected_long_window_profile": window_profile_path.as_posix(),
            "candidate_feature_deciles": candidate_deciles_path.as_posix(),
            "feature_shape_summary": feature_shape_path.as_posix(),
            "stress_sector_loser_concentration": sector_stress_path.as_posix(),
            "feature_decile_plot": plot_path.as_posix(),
            "memo": memo_path.as_posix(),
        },
        "method": "price_only_long_leg_rebound_winner_vs_falling_knife_diagnostics",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "No non-price factor panels are loaded or used.",
            "SEC-derived SIC labels are used only as neutralization and sector-relative price-state grouping keys.",
            "Legacy forward-return columns are named *_5d, but the input panel and positions come from the h10 validation contract.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _build_price_only_factor_panel(
    *,
    signal_panel_path: str | Path,
    cik_mapping_path: str | Path,
    sec_submissions_dir: str | Path,
    long_variant: str,
    short_variant: str,
    min_regression_rows: int,
    min_dummy_count: int,
) -> pd.DataFrame:
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
    panel = _add_residual_targets(
        panel,
        min_regression_rows=min_regression_rows,
        min_dummy_count=min_dummy_count,
    )
    long_panel = panel[panel["variant"].astype(str).eq(long_variant)].copy()
    columns = [
        "session_date",
        "symbol",
        "sic2_sector",
        "sic4_industry",
        DEFAULT_LONG_SCORE,
        *TARGET_COLUMNS,
        *FEATURE_COLUMNS,
    ]
    existing = [column for column in columns if column in long_panel.columns]
    return long_panel[existing].drop_duplicates(["session_date", "symbol"]).reset_index(drop=True)


def _selected_current_sota_longs(positions: pd.DataFrame) -> pd.DataFrame:
    selected = positions[
        positions["portfolio"].astype(str).eq(CURRENT_PORTFOLIO)
        & positions["side"].astype(str).eq("long")
    ].copy()
    if selected.empty:
        raise ValueError(f"No selected long rows found for {CURRENT_PORTFOLIO}.")
    selected["session_date"] = selected["session_date"].astype(str)
    selected["symbol"] = selected["symbol"].astype(str)
    selected["side_weight"] = pd.to_numeric(selected["side_weight"], errors="coerce").fillna(0.0)
    return selected


def _merge_selected_features(selected_longs: pd.DataFrame, factor_panel: pd.DataFrame) -> pd.DataFrame:
    factor_features = factor_panel.drop(columns=list(TARGET_COLUMNS), errors="ignore")
    overlap = [
        column
        for column in factor_features.columns
        if column in selected_longs.columns and column not in {"session_date", "symbol"}
    ]
    factor_features = factor_features.drop(columns=overlap, errors="ignore")
    merged = selected_longs.merge(factor_features, on=["session_date", "symbol"], how="left")
    for column in [*TARGET_COLUMNS, *FEATURE_COLUMNS]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged = merged.dropna(subset=[PRIMARY_TARGET]).copy()
    merged["target_bps"] = merged[PRIMARY_TARGET].astype(float) * 10000.0
    merged["left_tail"] = merged[PRIMARY_TARGET].lt(LEFT_TAIL_THRESHOLD)
    merged["right_tail"] = merged[PRIMARY_TARGET].gt(RIGHT_TAIL_THRESHOLD)
    merged["outcome_quintile"] = _quintile_labels(merged[PRIMARY_TARGET])
    merged["test_window_used"] = False
    return merged.reset_index(drop=True)


def _selected_outcome_bucket_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket, group in selected.groupby("outcome_quintile", sort=True):
        row: dict[str, Any] = _weighted_target_stats(group)
        row.update(
            {
                "outcome_quintile": bucket,
                "rows": int(len(group)),
                "sessions": int(group["session_date"].nunique()),
                "mean_side_weight": _safe_mean(group["side_weight"]),
                "test_window_used": False,
            }
        )
        row.update(_weighted_feature_stats(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _selected_window_profile(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window, start, end in WINDOWS:
        subset = selected.copy()
        if window != "full":
            subset = subset[(subset["session_date"] >= start) & (subset["session_date"] <= end)]
        if subset.empty:
            continue
        row = _weighted_target_stats(subset)
        row.update(
            {
                "window": window,
                "start": start,
                "end": end,
                "rows": int(len(subset)),
                "sessions": int(subset["session_date"].nunique()),
                "test_window_used": False,
            }
        )
        row.update(_weighted_feature_stats(subset))
        rows.append(row)
    return pd.DataFrame(rows)


def _candidate_feature_decile_summary(candidate_top: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = candidate_top.dropna(subset=[PRIMARY_TARGET]).copy()
    for feature in FEATURE_COLUMNS:
        if feature not in frame.columns:
            continue
        valid = frame.dropna(subset=[feature, PRIMARY_TARGET]).copy()
        if valid.empty:
            continue
        valid["feature_pct"] = valid.groupby("session_date")[feature].rank(
            method="average",
            pct=True,
        )
        valid["feature_decile"] = np.ceil(valid["feature_pct"] * 10.0).clip(1, 10).astype(int)
        for decile, group in valid.groupby("feature_decile", sort=True):
            values = group[PRIMARY_TARGET].astype(float)
            rows.append(
                {
                    "feature": feature,
                    "feature_decile": int(decile),
                    "rows": int(len(group)),
                    "sessions": int(group["session_date"].nunique()),
                    "target_mean_bps": float(values.mean() * 10000.0),
                    "target_median_bps": float(values.median() * 10000.0),
                    "target_trimmed_mean_bps": float(_trimmed_mean(values.to_numpy()) * 10000.0),
                    "target_iqr_bps": float((values.quantile(0.75) - values.quantile(0.25)) * 10000.0),
                    "left_tail_rate": float(values.lt(LEFT_TAIL_THRESHOLD).mean()),
                    "right_tail_rate": float(values.gt(RIGHT_TAIL_THRESHOLD).mean()),
                    "feature_median": float(group[feature].median()),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "feature_decile"]).reset_index(drop=True)


def _feature_shape_summary(candidate_deciles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature, group in candidate_deciles.groupby("feature", sort=True):
        ordered = group.sort_values("feature_decile")
        medians = ordered["target_median_bps"].astype(float)
        trimmed = ordered["target_trimmed_mean_bps"].astype(float)
        left_tail = ordered["left_tail_rate"].astype(float)
        deciles = ordered["feature_decile"].astype(int)
        if ordered.empty:
            continue
        worst_idx = medians.idxmin()
        best_idx = medians.idxmax()
        top = ordered[ordered["feature_decile"].eq(10)]
        bottom = ordered[ordered["feature_decile"].eq(1)]
        corr = float(deciles.corr(medians, method="spearman")) if len(ordered) > 2 else np.nan
        row = {
            "feature": feature,
            "spearman_decile_vs_median": corr,
            "best_decile": int(ordered.loc[best_idx, "feature_decile"]),
            "best_median_bps": float(ordered.loc[best_idx, "target_median_bps"]),
            "worst_decile": int(ordered.loc[worst_idx, "feature_decile"]),
            "worst_median_bps": float(ordered.loc[worst_idx, "target_median_bps"]),
            "worst_is_middle": bool(int(ordered.loc[worst_idx, "feature_decile"]) in {4, 5, 6, 7}),
            "median_range_bps": float(medians.max() - medians.min()),
            "trimmed_mean_range_bps": float(trimmed.max() - trimmed.min()),
            "left_tail_range_pct": float(left_tail.max() - left_tail.min()),
            "top_minus_bottom_median_bps": float(top["target_median_bps"].iloc[0] - bottom["target_median_bps"].iloc[0])
            if not top.empty and not bottom.empty
            else np.nan,
            "top_minus_bottom_left_tail_pct": float(top["left_tail_rate"].iloc[0] - bottom["left_tail_rate"].iloc[0])
            if not top.empty and not bottom.empty
            else np.nan,
            "test_window_used": False,
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["abs_spearman"] = frame["spearman_decile_vs_median"].abs()
    return frame.sort_values(
        ["median_range_bps", "abs_spearman"],
        ascending=[False, False],
    ).drop(columns="abs_spearman").reset_index(drop=True)


def _stress_sector_concentration(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window, start, end in WINDOWS:
        if window == "full":
            continue
        subset = selected[(selected["session_date"] >= start) & (selected["session_date"] <= end)].copy()
        if subset.empty:
            continue
        subset["weighted_target"] = subset["side_weight"].astype(float) * subset[PRIMARY_TARGET].astype(float)
        for sector, group in subset.groupby("sic2_sector", sort=True):
            if len(group) < 10:
                continue
            rows.append(
                {
                    "window": window,
                    "start": start,
                    "end": end,
                    "sic2_sector": sector,
                    "rows": int(len(group)),
                    "sessions": int(group["session_date"].nunique()),
                    "weighted_target_sum_bps": float(group["weighted_target"].sum() * 10000.0),
                    "weighted_target_mean_bps": _weighted_mean_bps(group, PRIMARY_TARGET),
                    "left_tail_rate": float(group[PRIMARY_TARGET].lt(LEFT_TAIL_THRESHOLD).mean()),
                    "right_tail_rate": float(group[PRIMARY_TARGET].gt(RIGHT_TAIL_THRESHOLD).mean()),
                    "weighted_price_falling_knife_score": _weighted_feature_mean(group, "price_falling_knife_score"),
                    "weighted_price_stabilization_score": _weighted_feature_mean(group, "price_stabilization_score"),
                    "test_window_used": False,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["window", "weighted_target_sum_bps"]).reset_index(drop=True)


def _plot_feature_deciles(candidate_deciles: pd.DataFrame, *, output_path: str | Path) -> None:
    features = [feature for feature in PLOT_FEATURES if feature in set(candidate_deciles["feature"])]
    if not features:
        return
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    axes_flat = axes.ravel()
    for ax, feature in zip(axes_flat, features):
        subset = candidate_deciles[candidate_deciles["feature"].eq(feature)].sort_values("feature_decile")
        ax.plot(
            subset["feature_decile"],
            subset["target_median_bps"],
            marker="o",
            linewidth=1.8,
            label="median",
        )
        ax.plot(
            subset["feature_decile"],
            subset["target_trimmed_mean_bps"],
            marker="s",
            linewidth=1.2,
            alpha=0.8,
            label="trimmed mean",
        )
        ax.axhline(0.0, color="#374151", linestyle="--", linewidth=0.8)
        ax.set_title(feature)
        ax.set_xlabel("within-session feature decile")
        ax.set_ylabel(f"{PRIMARY_TARGET} bps")
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(features) :]:
        ax.axis("off")
    axes_flat[0].legend(loc="best")
    fig.suptitle("Phase5S price-only candidate top80 feature deciles")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _memo(
    *,
    selected_bucket: pd.DataFrame,
    window_profile: pd.DataFrame,
    feature_shape: pd.DataFrame,
    sector_stress: pd.DataFrame,
    plot_path: Path,
    candidate_count: int,
) -> str:
    bucket_cols = [
        "outcome_quintile",
        "rows",
        "weighted_target_bps",
        "median_target_bps",
        "left_tail_rate",
        "right_tail_rate",
        "weighted_price_falling_knife_score",
        "weighted_price_still_falling_score",
        "weighted_price_stabilization_score",
        "weighted_momentum_20d_z",
        "weighted_momentum_60d_z",
    ]
    window_cols = [
        "window",
        "rows",
        "weighted_target_bps",
        "median_target_bps",
        "left_tail_rate",
        "right_tail_rate",
        "weighted_price_falling_knife_score",
        "weighted_price_stabilization_score",
    ]
    feature_cols = [
        "feature",
        "spearman_decile_vs_median",
        "best_decile",
        "best_median_bps",
        "worst_decile",
        "worst_median_bps",
        "worst_is_middle",
        "median_range_bps",
        "top_minus_bottom_median_bps",
        "top_minus_bottom_left_tail_pct",
    ]
    sector_cols = [
        "window",
        "sic2_sector",
        "rows",
        "weighted_target_sum_bps",
        "weighted_target_mean_bps",
        "left_tail_rate",
        "weighted_price_falling_knife_score",
    ]
    lines = [
        "# Phase5S Price-Only Long-Leg Diagnostics",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        f"- Portfolio: `{CURRENT_PORTFOLIO}`, long side only.",
        f"- Candidate pool: baseline top{candidate_count} by `{DEFAULT_LONG_SCORE}`.",
        f"- Primary target: `{PRIMARY_TARGET}` from the h10 validation contract.",
        "- No non-price factor panels are loaded or used.",
        "- SIC labels are retained only for residualization and sector-relative price-state grouping.",
        "- Validation window only; no test-window performance is computed.",
        "",
        "## Selected Long Outcome Buckets",
        "",
        _text_table(_format_table(selected_bucket[bucket_cols] if not selected_bucket.empty else selected_bucket)),
        "",
        "## Stress Window Profile",
        "",
        _text_table(_format_table(window_profile[window_cols] if not window_profile.empty else window_profile)),
        "",
        "## Feature Shape Summary",
        "",
        _text_table(_format_table(feature_shape[feature_cols].head(16) if not feature_shape.empty else feature_shape)),
        "",
        "## Stress Sector Loser Concentration",
        "",
        _text_table(_format_table(sector_stress[sector_cols].head(20) if not sector_stress.empty else sector_stress)),
        "",
        "## Plot",
        "",
        f"![Phase5S feature deciles]({plot_path.as_posix()})",
        "",
        "## Read",
        "",
        "- This is a diagnostic pass, not a strategy change.",
        "- The key test is whether price-state features separate loser buckets from rebound-winner buckets by median and trimmed mean, not just by left-tail events.",
        "- If a feature's worst decile sits in the middle, the relation is non-linear and a hard monotonic penalty is likely too blunt.",
    ]
    return "\n".join(lines)


def _weighted_target_stats(group: pd.DataFrame) -> dict[str, float]:
    values = group[PRIMARY_TARGET].astype(float)
    return {
        "weighted_target_bps": _weighted_mean_bps(group, PRIMARY_TARGET),
        "mean_target_bps": float(values.mean() * 10000.0),
        "median_target_bps": float(values.median() * 10000.0),
        "trimmed_mean_target_bps": float(_trimmed_mean(values.to_numpy()) * 10000.0),
        "left_tail_rate": float(values.lt(LEFT_TAIL_THRESHOLD).mean()),
        "right_tail_rate": float(values.gt(RIGHT_TAIL_THRESHOLD).mean()),
    }


def _weighted_feature_stats(group: pd.DataFrame) -> dict[str, float]:
    row: dict[str, float] = {}
    for feature in FEATURE_COLUMNS:
        if feature not in group.columns:
            continue
        row[f"weighted_{feature}"] = _weighted_feature_mean(group, feature)
    return row


def _weighted_feature_mean(group: pd.DataFrame, feature: str) -> float:
    valid = group.dropna(subset=[feature, "side_weight"]).copy()
    if valid.empty:
        return np.nan
    weights = valid["side_weight"].astype(float).clip(lower=0.0)
    denom = float(weights.sum())
    if denom <= 0:
        return np.nan
    return float(np.dot(weights, valid[feature].astype(float)) / denom)


def _weighted_mean_bps(group: pd.DataFrame, column: str) -> float:
    value = _weighted_feature_mean(group, column)
    return float(value * 10000.0) if np.isfinite(value) else np.nan


def _top_by_session(frame: pd.DataFrame, *, score_column: str, count: int) -> pd.DataFrame:
    rows = []
    for _, group in frame.groupby("session_date", sort=False):
        rows.append(group.sort_values([score_column, "symbol"], ascending=[False, True]).head(count))
    return pd.concat(rows, ignore_index=True) if rows else frame.iloc[0:0].copy()


def _quintile_labels(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="first")
    labels = ["Q1_falling_knife_loser", "Q2_weak", "Q3_middle", "Q4_good", "Q5_rebound_winner"]
    return pd.qcut(ranks, q=5, labels=labels, duplicates="drop").astype(str)


def _trimmed_mean(values: np.ndarray, trim: float = 0.10) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    if finite.size < 10:
        return float(np.mean(finite))
    low = np.quantile(finite, trim)
    high = np.quantile(finite, 1.0 - trim)
    trimmed = finite[(finite >= low) & (finite <= high)]
    return float(np.mean(trimmed)) if trimmed.size else float(np.mean(finite))


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else np.nan


def _format_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if column.endswith("_rate") or column.endswith("_pct"):
            out[column] = out[column].map(_fmt_pct_like)
        elif column.endswith("_bps"):
            out[column] = out[column].map(lambda value: _fmt_number(value, digits=1))
        elif column.startswith("weighted_") or column.startswith("mean_"):
            out[column] = out[column].map(lambda value: _fmt_number(value, digits=3))
        elif column in {"spearman_decile_vs_median", "median_range_bps"}:
            out[column] = out[column].map(lambda value: _fmt_number(value, digits=3))
    return out


def _fmt_number(value: Any, *, digits: int) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose current SOTA long leg falling-knife vs rebound-winner behavior with price-only features."
    )
    parser.add_argument("--positions-path", default=str(DEFAULT_PHASE5R_POSITIONS))
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--cik-mapping-path", default=str(DEFAULT_CIK_MAPPING))
    parser.add_argument("--sec-submissions-dir", default=str(DEFAULT_SEC_SUBMISSIONS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    args = parser.parse_args(argv)
    result = build_phase5s_price_only_long_leg_diagnostics_artifacts(
        positions_path=args.positions_path,
        signal_panel_path=args.signal_panel_path,
        cik_mapping_path=args.cik_mapping_path,
        sec_submissions_dir=args.sec_submissions_dir,
        output_root=args.output_root,
        candidate_count=args.candidate_count,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
