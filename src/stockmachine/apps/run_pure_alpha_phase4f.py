from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import (
    BASE_COLUMNS,
    DEFAULT_PHASE3_SIGNAL_PANEL,
    DEFAULT_VARIANTS,
    MODEL_FEATURES,
    RESEARCH_ROOT,
    TARGET_COLUMN,
    _add_model_features,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4f_feature_posterior_shapes_20260418"
BENCHMARK_FORWARD_COLUMN = "benchmark_forward_return_5d"
FORWARD_RETURN_COLUMN = "forward_return_5d"
DEFAULT_FEATURES = MODEL_FEATURES


def build_phase4f_feature_posterior_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_PHASE3_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant_names: Sequence[str] = DEFAULT_VARIANTS,
    features: Sequence[str] = DEFAULT_FEATURES,
    target_column: str = TARGET_COLUMN,
    n_bins: int = 10,
    loser_quantile: float = 0.20,
) -> dict[str, Any]:
    """Describe conditional target distributions by feature quantile bucket."""

    _validate_settings(
        variant_names=variant_names,
        features=features,
        n_bins=n_bins,
        loser_quantile=loser_quantile,
    )
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_posterior_panel(
        signal_panel_path,
        variant_names=variant_names,
        target_column=target_column,
    )
    bin_frames = []
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
        loser_cutoff = float(complete[target_column].quantile(loser_quantile))
        for feature in features:
            binned = _assign_feature_bins(complete, feature=feature, n_bins=n_bins)
            if binned.empty:
                continue
            bin_frames.append(
                _feature_bin_distribution_table(
                    binned,
                    variant=variant,
                    feature=feature,
                    target_column=target_column,
                    loser_quantile=loser_quantile,
                    loser_cutoff=loser_cutoff,
                )
            )

    bins = _concat_or_empty(bin_frames, _bin_columns())
    extremes = _extreme_bin_comparison_table(bins)
    manifest = pd.DataFrame(manifest_rows)

    bins_path = output_dir / "phase4f_feature_posterior_bins_validation.csv"
    extremes_path = output_dir / "phase4f_feature_posterior_extremes_validation.csv"
    manifest_path = output_dir / "phase4f_feature_posterior_sample_manifest_validation.csv"
    memo_path = output_dir / "phase4f_feature_posterior_memo.md"
    rollup_path = output_dir / "phase4f_feature_posterior_rollup.json"

    bins.to_csv(bins_path, index=False)
    extremes.to_csv(extremes_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    memo_path.write_text(
        _posterior_memo(
            bins,
            extremes,
            manifest,
            n_bins=n_bins,
            loser_quantile=loser_quantile,
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
        "feature_panel_rows_loaded": int(len(panel)),
        "posterior_bin_rows": int(len(bins)),
        "extreme_comparison_rows": int(len(extremes)),
        "sample_manifest_rows": int(len(manifest)),
        "posterior_bins_artifact": bins_path.as_posix(),
        "extreme_comparison_artifact": extremes_path.as_posix(),
        "sample_manifest_artifact": manifest_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "feature_quantile_conditional_target_distribution",
        "limitations": [
            "This is a conditional distribution diagnostic, not a portfolio constructor.",
            "Feature bins are formed on the validation sample only.",
            "Regime splits use the future 5-session benchmark return for diagnosis only.",
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
) -> None:
    if not variant_names:
        raise ValueError("At least one variant is required.")
    if not features:
        raise ValueError("At least one feature is required.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if not 0 < loser_quantile < 1:
        raise ValueError("loser_quantile must be between 0 and 1.")


def _load_posterior_panel(
    signal_panel_path: str | Path,
    *,
    variant_names: Sequence[str],
    target_column: str,
) -> pd.DataFrame:
    usecols = list(dict.fromkeys([*BASE_COLUMNS, BENCHMARK_FORWARD_COLUMN]))
    panel = pd.read_csv(signal_panel_path, usecols=usecols, low_memory=False)
    panel["session_date"] = pd.to_datetime(panel["session_date"]).dt.date.astype(str)
    panel["variant"] = panel["variant"].astype(str)
    panel["symbol"] = panel["symbol"].astype(str)
    panel = panel[panel["variant"].isin(set(variant_names))].copy()
    for column in usecols:
        if column not in ("session_date", "variant", "symbol"):
            panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.dropna(
        subset=[
            "beta",
            target_column,
            FORWARD_RETURN_COLUMN,
            BENCHMARK_FORWARD_COLUMN,
        ]
    )
    panel = _add_model_features(panel)
    return panel.sort_values(["variant", "session_date", "symbol"]).reset_index(drop=True)

def _assign_feature_bins(
    frame: pd.DataFrame,
    *,
    feature: str,
    n_bins: int,
) -> pd.DataFrame:
    values = pd.to_numeric(frame[feature], errors="coerce")
    unique_count = int(values.nunique(dropna=True))
    if unique_count < 2:
        return pd.DataFrame(columns=[*frame.columns, "feature_quantile_bin", "feature_bin_count"])
    bins = min(n_bins, unique_count)
    codes = pd.qcut(values, q=bins, labels=False, duplicates="drop")
    binned = frame.copy()
    binned["feature_quantile_bin"] = pd.to_numeric(codes, errors="coerce") + 1
    binned = binned.dropna(subset=["feature_quantile_bin"]).copy()
    binned["feature_quantile_bin"] = binned["feature_quantile_bin"].astype(int)
    binned["feature_bin_count"] = int(binned["feature_quantile_bin"].max())
    return binned


def _feature_bin_distribution_table(
    frame: pd.DataFrame,
    *,
    variant: str,
    feature: str,
    target_column: str,
    loser_quantile: float,
    loser_cutoff: float,
) -> pd.DataFrame:
    rows = []
    for regime, regime_frame in _regime_slices(frame):
        if regime_frame.empty:
            continue
        for feature_bin, group in regime_frame.groupby("feature_quantile_bin", sort=True):
            target = group[target_column]
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
                    "target_mean": float(target.mean()),
                    "target_median": float(target.median()),
                    "target_std": float(target.std(ddof=0)),
                    "target_q10": float(target.quantile(0.10)),
                    "target_q25": float(target.quantile(0.25)),
                    "target_q75": float(target.quantile(0.75)),
                    "target_q90": float(target.quantile(0.90)),
                    "negative_residual_share": float((target < 0).mean()),
                    "bottom_loser_quantile": float(loser_quantile),
                    "bottom_loser_cutoff": float(loser_cutoff),
                    "bottom_loser_share": float((target <= loser_cutoff).mean()),
                    "mean_forward_return_5d": float(group[FORWARD_RETURN_COLUMN].mean()),
                    "mean_benchmark_forward_return_5d": float(
                        group[BENCHMARK_FORWARD_COLUMN].mean()
                    ),
                    "short_contribution_beta_residual_return": float(-target.mean()),
                    "short_contribution_return": float(-group[FORWARD_RETURN_COLUMN].mean()),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows, columns=_bin_columns())


def _regime_slices(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    benchmark = frame[BENCHMARK_FORWARD_COLUMN]
    slices = [
        ("all", frame),
        ("market_up_5d", frame[benchmark > 0]),
        ("market_down_5d", frame[benchmark < 0]),
    ]
    flat = frame[benchmark == 0]
    if not flat.empty:
        slices.append(("market_flat_5d", flat))
    return slices


def _bin_side(feature_bin: int, bin_count: int) -> str:
    if feature_bin == 1:
        return "low_feature_tail"
    if feature_bin == bin_count:
        return "high_feature_tail"
    return "middle"


def _extreme_bin_comparison_table(bins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if bins.empty:
        return pd.DataFrame(columns=_extreme_columns())
    group_keys = ["variant", "feature", "market_regime"]
    for (variant, feature, regime), group in bins.groupby(group_keys, sort=True):
        low_bin = int(group["feature_quantile_bin"].min())
        high_bin = int(group["feature_quantile_bin"].max())
        low = group[group["feature_quantile_bin"] == low_bin].iloc[0]
        high = group[group["feature_quantile_bin"] == high_bin].iloc[0]
        high_minus_low_target_mean = high["target_mean"] - low["target_mean"]
        high_minus_low_loser_share = high["bottom_loser_share"] - low["bottom_loser_share"]
        rows.append(
            {
                "variant": variant,
                "feature": feature,
                "market_regime": regime,
                "low_bin": low_bin,
                "high_bin": high_bin,
                "low_rows": int(low["rows"]),
                "high_rows": int(high["rows"]),
                "low_feature_mean": float(low["feature_mean"]),
                "high_feature_mean": float(high["feature_mean"]),
                "low_target_mean": float(low["target_mean"]),
                "high_target_mean": float(high["target_mean"]),
                "high_minus_low_target_mean": float(high_minus_low_target_mean),
                "low_short_contribution_beta_residual_return": float(
                    low["short_contribution_beta_residual_return"]
                ),
                "high_short_contribution_beta_residual_return": float(
                    high["short_contribution_beta_residual_return"]
                ),
                "high_minus_low_short_contribution": float(-high_minus_low_target_mean),
                "low_negative_residual_share": float(low["negative_residual_share"]),
                "high_negative_residual_share": float(high["negative_residual_share"]),
                "high_minus_low_negative_residual_share": float(
                    high["negative_residual_share"] - low["negative_residual_share"]
                ),
                "low_bottom_loser_share": float(low["bottom_loser_share"]),
                "high_bottom_loser_share": float(high["bottom_loser_share"]),
                "high_minus_low_bottom_loser_share": float(high_minus_low_loser_share),
                "high_bin_more_loser_by_mean": bool(high["target_mean"] < low["target_mean"]),
                "high_bin_more_loser_by_bottom_share": bool(high_minus_low_loser_share > 0),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows, columns=_extreme_columns()).sort_values(
        ["variant", "feature", "market_regime"]
    )


def _posterior_memo(
    bins: pd.DataFrame,
    extremes: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    n_bins: int,
    loser_quantile: float,
) -> str:
    lines = [
        "# Pure Alpha Phase 4F Feature Posterior Shape Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "This is a validation-only diagnostic that translates feature-target "
        "non-independence into conditional target distributions. It asks whether "
        "knowing a feature bucket materially changes the posterior distribution "
        "of future 5-session beta-residual return.",
        "",
        "## Method",
        "",
        f"- feature quantile bins: `{n_bins}`;",
        f"- residual-loser tail cutoff: bottom `{loser_quantile}` of validation targets;",
        "- regimes: `all`, `market_up_5d`, and `market_down_5d`; ",
        "- outputs include conditional mean, quantiles, negative residual share, "
        "bottom-loser share, and hypothetical short contribution for each bin;",
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
            "## Beta Tail Check",
            "",
            "| Feature | Regime | Low Mean Residual | High Mean Residual | "
            "High-Low Short Contribution | Low Loser Share | High Loser Share |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    beta_rows = extremes[extremes["feature"].isin(["beta", "beta_z"])]
    for _, row in beta_rows.iterrows():
        lines.append(
            f"| {row['feature']} | {row['market_regime']} | {row['low_target_mean']} | "
            f"{row['high_target_mean']} | {row['high_minus_low_short_contribution']} | "
            f"{row['low_bottom_loser_share']} | {row['high_bottom_loser_share']} |"
        )

    lines.extend(
        [
            "",
            "## Largest High-Minus-Low Short Contributions",
            "",
            "| Feature | Regime | High-Low Short Contribution | High Loser Share | Low Loser Share |",
            "|---|---|---:|---:|---:|",
        ]
    )
    top = extremes.sort_values(
        "high_minus_low_short_contribution", ascending=False
    ).head(20)
    for _, row in top.iterrows():
        lines.append(
            f"| {row['feature']} | {row['market_regime']} | "
            f"{row['high_minus_low_short_contribution']} | "
            f"{row['high_bottom_loser_share']} | {row['low_bottom_loser_share']} |"
        )

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- A shifted posterior distribution is not automatically a deployable "
            "selector.",
            "- Regime rows use future benchmark return only to diagnose what happened; "
            "they are not tradable regime labels.",
            "- Extreme-bin diagnostics are unweighted and not beta-matched portfolios.",
            "- Test lockbox remains closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _concat_or_empty(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    return pd.concat(non_empty, ignore_index=True)[columns]


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
        "target_mean",
        "target_median",
        "target_std",
        "target_q10",
        "target_q25",
        "target_q75",
        "target_q90",
        "negative_residual_share",
        "bottom_loser_quantile",
        "bottom_loser_cutoff",
        "bottom_loser_share",
        "mean_forward_return_5d",
        "mean_benchmark_forward_return_5d",
        "short_contribution_beta_residual_return",
        "short_contribution_return",
        "test_window_used",
    ]


def _extreme_columns() -> list[str]:
    return [
        "variant",
        "feature",
        "market_regime",
        "low_bin",
        "high_bin",
        "low_rows",
        "high_rows",
        "low_feature_mean",
        "high_feature_mean",
        "low_target_mean",
        "high_target_mean",
        "high_minus_low_target_mean",
        "low_short_contribution_beta_residual_return",
        "high_short_contribution_beta_residual_return",
        "high_minus_low_short_contribution",
        "low_negative_residual_share",
        "high_negative_residual_share",
        "high_minus_low_negative_residual_share",
        "low_bottom_loser_share",
        "high_bottom_loser_share",
        "high_minus_low_bottom_loser_share",
        "high_bin_more_loser_by_mean",
        "high_bin_more_loser_by_bottom_share",
        "test_window_used",
    ]


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
        description="Run pure-alpha Phase 4F feature posterior diagnostics."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_PHASE3_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--feature", action="append", dest="features")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--loser-quantile", type=float, default=0.20)
    args = parser.parse_args(argv)

    result = build_phase4f_feature_posterior_artifacts(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        variant_names=tuple(args.variants or DEFAULT_VARIANTS),
        features=tuple(args.features or DEFAULT_FEATURES),
        n_bins=args.n_bins,
        loser_quantile=args.loser_quantile,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
