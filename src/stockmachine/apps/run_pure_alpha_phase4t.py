from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT, TARGET_COLUMN
from stockmachine.apps.run_pure_alpha_phase4e import (
    _feature_pair_independence_table,
    _target_independence_table,
)
from stockmachine.apps.run_pure_alpha_phase4f import (
    BENCHMARK_FORWARD_COLUMN,
    FORWARD_RETURN_COLUMN,
    _assign_feature_bins,
)
from stockmachine.apps.run_pure_alpha_phase4g import (
    _baseline_columns,
    _baseline_distribution_table,
    _bin_columns,
    _concat_or_empty,
    _feature_bin_utility_table,
    _feature_utility_summary,
)


DEFAULT_SIGNAL_PANEL = (
    RESEARCH_ROOT
    / "phase3_baseline_signals_h10_probe_20260418"
    / "phase3_baseline_signal_panel_validation.csv.gz"
)
DEFAULT_NON_PRICE_PANEL = (
    RESEARCH_ROOT
    / "non_price_phase1_adv30m_two_factor_build_20260419"
    / "non_price_two_factor_panel.csv.gz"
)
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4t_non_price_short_factor_residual_utility_v2_20260419"
DEFAULT_VARIANT = "adv30m_clean_core_beta_full"
DEFAULT_MAX_NMI_ROWS = 200_000

FACTOR_SPECS = (
    {
        "feature": "filing_red_flag_short_score",
        "source_column": "filing_red_flag_score",
        "orientation": 1.0,
        "intended_short_tail": "high",
        "intended_long_tail": "",
        "short_interpretation": "higher SEC filing red-flag score should identify worse residual-return candidates",
    },
    {
        "feature": "insider_net_buy_score",
        "source_column": "insider_net_buy_score",
        "orientation": 1.0,
        "intended_short_tail": "low",
        "intended_long_tail": "high",
        "short_interpretation": "low insider net-buy score is only a weak short candidate; high score is primarily a long signal",
    },
)


def build_phase4t_non_price_short_factor_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_SIGNAL_PANEL,
    non_price_panel_path: str | Path = DEFAULT_NON_PRICE_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant: str = DEFAULT_VARIANT,
    target_column: str = TARGET_COLUMN,
    n_bins: int = 10,
    loser_quantile: float = 0.20,
    winner_quantile: float = 0.20,
    right_tail_tolerance: float = 0.02,
    max_nmi_rows: int = DEFAULT_MAX_NMI_ROWS,
    permutation_count: int = 5,
    random_state: int = 260321,
) -> dict[str, Any]:
    """Evaluate two Non-Price short factors versus future residual return."""

    _validate_settings(
        n_bins=n_bins,
        loser_quantile=loser_quantile,
        winner_quantile=winner_quantile,
        max_nmi_rows=max_nmi_rows,
        permutation_count=permutation_count,
    )
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel, coverage = _load_and_merge_panel(
        signal_panel_path=signal_panel_path,
        non_price_panel_path=non_price_panel_path,
        variant=variant,
        target_column=target_column,
    )
    features = tuple(spec["feature"] for spec in FACTOR_SPECS)
    complete = (
        panel[
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
        .copy()
    )

    baseline_frames: list[pd.DataFrame] = []
    bin_frames: list[pd.DataFrame] = []
    manifest = pd.DataFrame(
        [
            {
                "variant": variant,
                "loaded_signal_rows": int(coverage["signal_rows"]),
                "loaded_non_price_rows": int(coverage["non_price_rows"]),
                "merged_rows": int(len(panel)),
                "complete_case_rows": int(len(complete)),
                "sample_start_session": _first_or_none(complete["session_date"]),
                "sample_end_session": _last_or_none(complete["session_date"]),
                "test_window_used": False,
            }
        ]
    )
    if not complete.empty:
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

    baseline = _concat_or_empty(baseline_frames, _baseline_columns())
    bins = _concat_or_empty(bin_frames, _bin_columns())
    summary = _feature_utility_summary(
        bins,
        baseline,
        right_tail_tolerance=right_tail_tolerance,
    )
    monotonic = _monotonicity_table(bins, factor_specs=FACTOR_SPECS)
    nmi_sample = _nmi_sample(
        complete,
        features=features,
        target_column=target_column,
        max_rows=max_nmi_rows,
        random_state=random_state,
    )
    target_nmi = _target_independence_table(
        nmi_sample,
        variant=variant,
        features=features,
        target_column=target_column,
        n_bins=n_bins,
        permutation_count=permutation_count,
        random_state=random_state,
    )
    pair_nmi = _feature_pair_independence_table(
        nmi_sample,
        variant=variant,
        features=features,
        n_bins=n_bins,
        permutation_count=permutation_count,
        random_state=random_state,
    )

    factor_spec = pd.DataFrame(FACTOR_SPECS)
    factor_spec_path = output_dir / "phase4t_non_price_factor_specs.csv"
    manifest_path = output_dir / "phase4t_non_price_sample_manifest_validation.csv"
    baseline_path = output_dir / "phase4t_non_price_baseline_validation.csv"
    bins_path = output_dir / "phase4t_non_price_bins_validation.csv"
    summary_path = output_dir / "phase4t_non_price_summary_validation.csv"
    monotonic_path = output_dir / "phase4t_non_price_monotonicity_validation.csv"
    target_nmi_path = output_dir / "phase4t_non_price_target_nmi_validation.csv"
    pair_nmi_path = output_dir / "phase4t_non_price_pair_nmi_validation.csv"
    memo_path = output_dir / "phase4t_non_price_short_factor_memo.md"
    rollup_path = output_dir / "phase4t_non_price_rollup.json"

    factor_spec.to_csv(factor_spec_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    baseline.to_csv(baseline_path, index=False)
    bins.to_csv(bins_path, index=False)
    summary.to_csv(summary_path, index=False)
    monotonic.to_csv(monotonic_path, index=False)
    target_nmi.to_csv(target_nmi_path, index=False)
    pair_nmi.to_csv(pair_nmi_path, index=False)
    memo_path.write_text(
        _memo(
            summary=summary,
            monotonic=monotonic,
            target_nmi=target_nmi,
            pair_nmi=pair_nmi,
            baseline=baseline,
            manifest=manifest,
            n_bins=n_bins,
            loser_quantile=loser_quantile,
            winner_quantile=winner_quantile,
            max_nmi_rows=max_nmi_rows,
            permutation_count=permutation_count,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "non_price_panel_path": Path(non_price_panel_path).as_posix(),
        "variant": variant,
        "target_column": target_column,
        "features": list(features),
        "factor_specs": list(FACTOR_SPECS),
        "n_bins": int(n_bins),
        "loser_quantile": float(loser_quantile),
        "winner_quantile": float(winner_quantile),
        "right_tail_tolerance": float(right_tail_tolerance),
        "max_nmi_rows": int(max_nmi_rows),
        "permutation_count": int(permutation_count),
        "random_state": int(random_state),
        "signal_rows": int(coverage["signal_rows"]),
        "non_price_rows": int(coverage["non_price_rows"]),
        "merged_rows": int(len(panel)),
        "complete_case_rows": int(len(complete)),
        "summary_rows": int(len(summary)),
        "bin_rows": int(len(bins)),
        "target_nmi_rows": int(len(target_nmi)),
        "pair_nmi_rows": int(len(pair_nmi)),
        "factor_spec_artifact": factor_spec_path.as_posix(),
        "manifest_artifact": manifest_path.as_posix(),
        "baseline_artifact": baseline_path.as_posix(),
        "bins_artifact": bins_path.as_posix(),
        "summary_artifact": summary_path.as_posix(),
        "monotonicity_artifact": monotonic_path.as_posix(),
        "target_nmi_artifact": target_nmi_path.as_posix(),
        "pair_nmi_artifact": pair_nmi_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "non_price_short_factor_robust_posterior_and_nmi_diagnostics",
        "limitations": [
            "The h10 signal panel keeps legacy 5d column names; values are from the h10 probe artifact.",
            "Feature bins are formed on the validation sample only.",
            "This is an unweighted factor diagnostic, not a beta-matched portfolio.",
            "Regime splits use future benchmark return for diagnosis only.",
            "No test-window performance is computed.",
        ],
    }
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _validate_settings(
    *,
    n_bins: int,
    loser_quantile: float,
    winner_quantile: float,
    max_nmi_rows: int,
    permutation_count: int,
) -> None:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if not 0 < loser_quantile < 1:
        raise ValueError("loser_quantile must be between 0 and 1.")
    if not 0 < winner_quantile < 1:
        raise ValueError("winner_quantile must be between 0 and 1.")
    if loser_quantile + winner_quantile >= 1:
        raise ValueError("loser_quantile + winner_quantile must be below 1.")
    if max_nmi_rows <= 0:
        raise ValueError("max_nmi_rows must be positive.")
    if permutation_count <= 0:
        raise ValueError("permutation_count must be positive.")


def _load_and_merge_panel(
    *,
    signal_panel_path: str | Path,
    non_price_panel_path: str | Path,
    variant: str,
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    signal_cols = [
        "session_date",
        "variant",
        "symbol",
        target_column,
        FORWARD_RETURN_COLUMN,
        BENCHMARK_FORWARD_COLUMN,
    ]
    signal = pd.read_csv(signal_panel_path, usecols=signal_cols, low_memory=False)
    signal["session_date"] = pd.to_datetime(signal["session_date"]).dt.date.astype(str)
    signal["variant"] = signal["variant"].astype(str)
    signal["symbol"] = signal["symbol"].astype(str)
    signal = signal[signal["variant"] == variant].copy()
    for column in signal_cols:
        if column not in ("session_date", "variant", "symbol"):
            signal[column] = pd.to_numeric(signal[column], errors="coerce")

    source_columns = list(dict.fromkeys(spec["source_column"] for spec in FACTOR_SPECS))
    non_price_cols = ["session_date", "symbol", *source_columns]
    non_price = pd.read_csv(non_price_panel_path, usecols=non_price_cols, low_memory=False)
    non_price["session_date"] = pd.to_datetime(non_price["session_date"]).dt.date.astype(str)
    non_price["symbol"] = non_price["symbol"].astype(str)
    for column in source_columns:
        non_price[column] = pd.to_numeric(non_price[column], errors="coerce")

    merged = signal.merge(non_price, on=["session_date", "symbol"], how="inner")
    for spec in FACTOR_SPECS:
        merged[spec["feature"]] = (
            float(spec["orientation"]) * merged[spec["source_column"]].astype(float)
        )
    return merged.sort_values(["session_date", "symbol"]).reset_index(drop=True), {
        "signal_rows": int(len(signal)),
        "non_price_rows": int(len(non_price)),
    }


def _nmi_sample(
    panel: pd.DataFrame,
    *,
    features: Sequence[str],
    target_column: str,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    columns = ["session_date", "variant", "symbol", *features, target_column]
    sample = panel[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=random_state)
    return sample.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _monotonicity_table(
    bins: pd.DataFrame,
    *,
    factor_specs: Sequence[dict[str, object]],
) -> pd.DataFrame:
    rows = []
    all_bins = bins[bins["market_regime"] == "all"].copy()
    spec_by_feature = {str(spec["feature"]): spec for spec in factor_specs}
    for (variant, feature), group in all_bins.groupby(["variant", "feature"], sort=True):
        group = group.sort_values("feature_quantile_bin")
        if group.empty:
            continue
        low = group.iloc[0]
        high = group.iloc[-1]
        intended_short_tail = str(
            spec_by_feature.get(str(feature), {}).get("intended_short_tail", "high")
        )
        oriented = _oriented_short_metrics(
            intended_short_tail=intended_short_tail,
            low_trimmed=float(low["target_trimmed_mean_10_90"]),
            high_trimmed=float(high["target_trimmed_mean_10_90"]),
            low_median=float(low["target_median"]),
            high_median=float(high["target_median"]),
            low_loser_share=float(low["bottom_loser_share"]),
            high_loser_share=float(high["bottom_loser_share"]),
            low_winner_share=float(low["top_winner_share"]),
            high_winner_share=float(high["top_winner_share"]),
        )
        rows.append(
            {
                "variant": variant,
                "feature": feature,
                "intended_short_tail": intended_short_tail,
                "bin_count": int(group["feature_bin_count"].max()),
                "rows": int(group["rows"].sum()),
                "spearman_bin_vs_target_trimmed_mean": float(
                    group["feature_quantile_bin"].corr(
                        group["target_trimmed_mean_10_90"], method="spearman"
                    )
                ),
                "spearman_bin_vs_target_median": float(
                    group["feature_quantile_bin"].corr(group["target_median"], method="spearman")
                ),
                "low_bin": int(low["feature_quantile_bin"]),
                "high_bin": int(high["feature_quantile_bin"]),
                "low_target_trimmed_mean_10_90": float(low["target_trimmed_mean_10_90"]),
                "high_target_trimmed_mean_10_90": float(high["target_trimmed_mean_10_90"]),
                "high_minus_low_short_contribution_trimmed": float(
                    low["target_trimmed_mean_10_90"] - high["target_trimmed_mean_10_90"]
                ),
                "low_target_median": float(low["target_median"]),
                "high_target_median": float(high["target_median"]),
                "high_minus_low_short_contribution_median": float(
                    low["target_median"] - high["target_median"]
                ),
                "low_bottom_loser_share": float(low["bottom_loser_share"]),
                "high_bottom_loser_share": float(high["bottom_loser_share"]),
                "high_minus_low_bottom_loser_share": float(
                    high["bottom_loser_share"] - low["bottom_loser_share"]
                ),
                "low_top_winner_share": float(low["top_winner_share"]),
                "high_top_winner_share": float(high["top_winner_share"]),
                "high_minus_low_top_winner_share": float(
                    high["top_winner_share"] - low["top_winner_share"]
                ),
                **oriented,
                "test_window_used": False,
            }
        )
    columns = [
        "variant",
        "feature",
        "intended_short_tail",
        "bin_count",
        "rows",
        "spearman_bin_vs_target_trimmed_mean",
        "spearman_bin_vs_target_median",
        "low_bin",
        "high_bin",
        "low_target_trimmed_mean_10_90",
        "high_target_trimmed_mean_10_90",
        "high_minus_low_short_contribution_trimmed",
        "low_target_median",
        "high_target_median",
        "high_minus_low_short_contribution_median",
        "low_bottom_loser_share",
        "high_bottom_loser_share",
        "high_minus_low_bottom_loser_share",
        "low_top_winner_share",
        "high_top_winner_share",
        "high_minus_low_top_winner_share",
        "oriented_short_contribution_trimmed",
        "oriented_short_contribution_median",
        "oriented_short_loser_share_edge",
        "oriented_short_winner_share_edge",
        "oriented_short_label",
        "test_window_used",
    ]
    return pd.DataFrame(rows, columns=columns)


def _oriented_short_metrics(
    *,
    intended_short_tail: str,
    low_trimmed: float,
    high_trimmed: float,
    low_median: float,
    high_median: float,
    low_loser_share: float,
    high_loser_share: float,
    low_winner_share: float,
    high_winner_share: float,
) -> dict[str, float | str]:
    if intended_short_tail == "low":
        trimmed_edge = high_trimmed - low_trimmed
        median_edge = high_median - low_median
        loser_edge = low_loser_share - high_loser_share
        winner_edge = low_winner_share - high_winner_share
    else:
        trimmed_edge = low_trimmed - high_trimmed
        median_edge = low_median - high_median
        loser_edge = high_loser_share - low_loser_share
        winner_edge = high_winner_share - low_winner_share
    if trimmed_edge > 0 and median_edge > 0 and loser_edge > 0:
        label = "intended_short_direction_supported"
    elif trimmed_edge < 0 and median_edge < 0 and loser_edge < 0:
        label = "intended_short_direction_rejected"
    else:
        label = "mixed_or_non_monotonic"
    return {
        "oriented_short_contribution_trimmed": float(trimmed_edge),
        "oriented_short_contribution_median": float(median_edge),
        "oriented_short_loser_share_edge": float(loser_edge),
        "oriented_short_winner_share_edge": float(winner_edge),
        "oriented_short_label": label,
    }


def _memo(
    *,
    summary: pd.DataFrame,
    monotonic: pd.DataFrame,
    target_nmi: pd.DataFrame,
    pair_nmi: pd.DataFrame,
    baseline: pd.DataFrame,
    manifest: pd.DataFrame,
    n_bins: int,
    loser_quantile: float,
    winner_quantile: float,
    max_nmi_rows: int,
    permutation_count: int,
) -> str:
    lines = [
        "# Phase4T Non-Price Short Factor Residual Utility Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "Validation-only evaluation of the two new Non-Price factors on the adv30m short universe. "
        "The target is future beta-residual return from the h10 signal panel, whose legacy column "
        "name is still `forward_beta_residual_return_5d`.",
        "",
        "## Method",
        "",
        f"- feature quantile bins: `{n_bins}`;",
        f"- residual-loser cutoff: bottom `{loser_quantile}`;",
        f"- residual-winner cutoff: top `{winner_quantile}`;",
        "- robust utility metrics follow Phase4G: median, 10/90 trimmed mean, 5/95 winsorized mean, "
        "left-tail share, right-tail share, and tail balance;",
        f"- nonlinear dependence follows Phase4E: quantile NMI with `{permutation_count}` permutations "
        f"on up to `{max_nmi_rows}` rows;",
        "- no test-window performance is computed.",
        "",
        "## Factor Orientation",
        "",
        "| Feature | Source | Orientation | Short Tail | Long Tail | Interpretation |",
        "|---|---|---:|---|---|---|",
    ]
    for spec in FACTOR_SPECS:
        lines.append(
            f"| {spec['feature']} | {spec['source_column']} | {spec['orientation']} | "
            f"{spec['intended_short_tail']} | {spec['intended_long_tail']} | "
            f"{spec['short_interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Sample",
            "",
            "| Variant | Signal Rows | Non-Price Rows | Merged Rows | Complete Rows | Start | End |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for _, row in manifest.iterrows():
        lines.append(
            f"| {row['variant']} | {row['loaded_signal_rows']} | {row['loaded_non_price_rows']} | "
            f"{row['merged_rows']} | {row['complete_case_rows']} | "
            f"{row['sample_start_session']} | {row['sample_end_session']} |"
        )
    lines.extend(
        [
            "",
            "## Universe Baseline",
            "",
            "| Regime | Median | Trimmed Mean | Winsorized Mean | Bottom Share | Top Share |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in baseline.iterrows():
        lines.append(
            f"| {row['market_regime']} | {row['target_median']} | "
            f"{row['target_trimmed_mean_10_90']} | {row['target_winsorized_mean_5_95']} | "
            f"{row['bottom_loser_share']} | {row['top_winner_share']} |"
        )
    lines.extend(
        [
            "",
            "## Robust Short Utility",
            "",
            "| Feature | Selected Bin | Label | Trimmed Short | Median Short | Tail Balance | Top Winner Share | Regime Edge Count |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['feature']} | {row['selected_feature_quantile_bin']} | "
            f"{row['robust_usefulness_label']} | "
            f"{row['all_short_contribution_trimmed_mean_10_90']} | "
            f"{row['all_short_contribution_median']} | {row['all_tail_balance']} | "
            f"{row['all_top_winner_share']} | {row['regime_trimmed_edge_positive_count']} |"
        )
    lines.extend(
        [
            "",
            "## Intended Direction Check",
            "",
            "| Feature | Short Tail | Direction Label | Oriented Short Trimmed | Oriented Loser Share Edge | Spearman Bin vs Trimmed Target |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for _, row in monotonic.iterrows():
        lines.append(
            f"| {row['feature']} | {row['intended_short_tail']} | {row['oriented_short_label']} | "
            f"{row['oriented_short_contribution_trimmed']} | "
            f"{row['oriented_short_loser_share_edge']} | "
            f"{row['spearman_bin_vs_target_trimmed_mean']} |"
        )
    lines.extend(
        [
            "",
            "## Nonlinear Non-Independence",
            "",
            "| Feature | Excess NMI vs Target | Label | Rows |",
            "|---|---:|---|---:|",
        ]
    )
    for _, row in target_nmi.iterrows():
        lines.append(
            f"| {row['feature']} | {row['excess_normalized_mutual_information']} | "
            f"{row['independence_label']} | {row['rows']} |"
        )
    if not pair_nmi.empty:
        lines.extend(
            [
                "",
                "| Feature Left | Feature Right | Excess NMI | Label |",
                "|---|---|---:|---|",
            ]
        )
        for _, row in pair_nmi.iterrows():
            lines.append(
                f"| {row['feature_left']} | {row['feature_right']} | "
                f"{row['excess_normalized_mutual_information']} | {row['independence_label']} |"
            )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- These are unweighted factor diagnostics, not beta-matched portfolio returns.",
            "- The selected bin is a validation diagnostic, not a frozen production rule.",
            "- Non-Price factors may carry filing/insider reporting delay assumptions inherited from their source build.",
            "- Test lockbox remains closed.",
        ]
    )
    return "\n".join(lines) + "\n"


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
        description="Run Phase4T Non-Price short factor residual utility diagnostics."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_SIGNAL_PANEL))
    parser.add_argument("--non-price-panel-path", default=str(DEFAULT_NON_PRICE_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--loser-quantile", type=float, default=0.20)
    parser.add_argument("--winner-quantile", type=float, default=0.20)
    parser.add_argument("--right-tail-tolerance", type=float, default=0.02)
    parser.add_argument("--max-nmi-rows", type=int, default=DEFAULT_MAX_NMI_ROWS)
    parser.add_argument("--permutation-count", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=260321)
    args = parser.parse_args(argv)

    result = build_phase4t_non_price_short_factor_artifacts(
        signal_panel_path=args.signal_panel_path,
        non_price_panel_path=args.non_price_panel_path,
        output_root=args.output_root,
        variant=args.variant,
        target_column=args.target_column,
        n_bins=args.n_bins,
        loser_quantile=args.loser_quantile,
        winner_quantile=args.winner_quantile,
        right_tail_tolerance=args.right_tail_tolerance,
        max_nmi_rows=args.max_nmi_rows,
        permutation_count=args.permutation_count,
        random_state=args.random_state,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
