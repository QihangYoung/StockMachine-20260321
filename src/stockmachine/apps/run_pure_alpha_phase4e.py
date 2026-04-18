from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from stockmachine.apps.run_pure_alpha_phase4d import (
    DEFAULT_PHASE3_SIGNAL_PANEL,
    DEFAULT_VARIANTS,
    MODEL_FEATURES,
    RESEARCH_ROOT,
    TARGET_COLUMN,
    _load_feature_panel,
)


DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase4e_feature_independence_20260418"
DEFAULT_MAX_ROWS = 200_000


def build_phase4e_feature_independence_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_PHASE3_SIGNAL_PANEL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant_names: Sequence[str] = DEFAULT_VARIANTS,
    features: Sequence[str] = MODEL_FEATURES,
    target_column: str = TARGET_COLUMN,
    n_bins: int = 10,
    max_rows: int = DEFAULT_MAX_ROWS,
    permutation_count: int = 5,
    cluster_threshold: float = 0.05,
    random_state: int = 260321,
) -> dict[str, Any]:
    """Measure nonlinear independence diagnostics for selector features."""

    _validate_settings(
        variant_names=variant_names,
        features=features,
        n_bins=n_bins,
        max_rows=max_rows,
        permutation_count=permutation_count,
        cluster_threshold=cluster_threshold,
    )
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_feature_panel(signal_panel_path, variant_names=variant_names)
    target_rows: list[pd.DataFrame] = []
    pair_rows: list[pd.DataFrame] = []
    sample_manifest: list[dict[str, object]] = []
    for variant in variant_names:
        variant_panel = panel[panel["variant"] == variant].copy()
        sample = _complete_case_sample(
            variant_panel,
            features=features,
            target_column=target_column,
            max_rows=max_rows,
            random_state=random_state,
        )
        sample_manifest.append(
            {
                "variant": variant,
                "loaded_rows": int(len(variant_panel)),
                "complete_case_rows": int(
                    variant_panel[list(features) + [target_column]].dropna().shape[0]
                ),
                "sampled_rows": int(len(sample)),
                "sample_start_session": _first_or_none(sample["session_date"])
                if not sample.empty
                else None,
                "sample_end_session": _last_or_none(sample["session_date"])
                if not sample.empty
                else None,
                "test_window_used": False,
            }
        )
        target_rows.append(
            _target_independence_table(
                sample,
                variant=variant,
                features=features,
                target_column=target_column,
                n_bins=n_bins,
                permutation_count=permutation_count,
                random_state=random_state,
            )
        )
        pair_rows.append(
            _feature_pair_independence_table(
                sample,
                variant=variant,
                features=features,
                n_bins=n_bins,
                permutation_count=permutation_count,
                random_state=random_state,
            )
        )

    target_table = _concat_or_empty(target_rows, _target_columns())
    pair_table = _concat_or_empty(pair_rows, _pair_columns())
    clusters = _feature_dependence_clusters(pair_table, threshold=cluster_threshold)
    manifest = pd.DataFrame(sample_manifest)

    target_path = output_dir / "phase4e_feature_target_independence_validation.csv"
    pair_path = output_dir / "phase4e_feature_pair_independence_validation.csv"
    cluster_path = output_dir / "phase4e_feature_dependence_clusters_validation.csv"
    manifest_path = output_dir / "phase4e_feature_independence_sample_manifest_validation.csv"
    memo_path = output_dir / "phase4e_feature_independence_memo.md"
    rollup_path = output_dir / "phase4e_feature_independence_rollup.json"

    target_table.to_csv(target_path, index=False)
    pair_table.to_csv(pair_path, index=False)
    clusters.to_csv(cluster_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    memo_path.write_text(
        _feature_independence_memo(
            target_table,
            pair_table,
            clusters,
            manifest,
            n_bins=n_bins,
            max_rows=max_rows,
            permutation_count=permutation_count,
            cluster_threshold=cluster_threshold,
        ),
        encoding="utf-8",
    )

    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "variant_names": list(variant_names),
        "target_column": target_column,
        "features": list(features),
        "n_bins": int(n_bins),
        "max_rows": int(max_rows),
        "permutation_count": int(permutation_count),
        "cluster_threshold": float(cluster_threshold),
        "random_state": int(random_state),
        "feature_panel_rows_loaded": int(len(panel)),
        "target_independence_rows": int(len(target_table)),
        "feature_pair_independence_rows": int(len(pair_table)),
        "feature_dependence_cluster_rows": int(len(clusters)),
        "sample_manifest_rows": int(len(manifest)),
        "target_independence_artifact": target_path.as_posix(),
        "feature_pair_independence_artifact": pair_path.as_posix(),
        "feature_dependence_clusters_artifact": cluster_path.as_posix(),
        "sample_manifest_artifact": manifest_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "quantile_discretized_normalized_mutual_information_with_permutation_baseline",
        "limitations": [
            "This measures nonlinear dependence, not tradable predictive power.",
            "Continuous variables are quantile-discretized before NMI estimation.",
            "Permutation baselines reduce finite-sample bias but do not prove true statistical independence.",
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
    max_rows: int,
    permutation_count: int,
    cluster_threshold: float,
) -> None:
    if not variant_names:
        raise ValueError("At least one variant is required.")
    if not features:
        raise ValueError("At least one feature is required.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")
    if permutation_count <= 0:
        raise ValueError("permutation_count must be positive.")
    if cluster_threshold < 0:
        raise ValueError("cluster_threshold must be non-negative.")


def _complete_case_sample(
    panel: pd.DataFrame,
    *,
    features: Sequence[str],
    target_column: str,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    columns = ["session_date", "variant", "symbol", *features, target_column]
    sample = panel.loc[:, columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=random_state)
    return sample.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _target_independence_table(
    sample: pd.DataFrame,
    *,
    variant: str,
    features: Sequence[str],
    target_column: str,
    n_bins: int,
    permutation_count: int,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    target_codes = _quantile_codes(sample[target_column], n_bins=n_bins)
    for index, feature in enumerate(features):
        feature_codes = _quantile_codes(sample[feature], n_bins=n_bins)
        stats = _dependence_stats(
            feature_codes,
            target_codes,
            permutation_count=permutation_count,
            random_state=random_state + index,
        )
        rows.append(
            {
                "variant": variant,
                "feature": feature,
                "target": target_column,
                "rows": int(stats["rows"]),
                "n_bins": int(n_bins),
                "feature_bins": int(stats["x_bins"]),
                "target_bins": int(stats["y_bins"]),
                "normalized_mutual_information": stats[
                    "normalized_mutual_information"
                ],
                "permutation_mean_nmi": stats["permutation_mean_nmi"],
                "excess_normalized_mutual_information": stats[
                    "excess_normalized_mutual_information"
                ],
                "permutation_p_value": stats["permutation_p_value"],
                "independence_label": _independence_label(
                    stats["excess_normalized_mutual_information"]
                ),
                "test_window_used": False,
            }
        )
    table = pd.DataFrame(rows, columns=_target_columns())
    if table.empty:
        return table
    table = table.sort_values(
        ["variant", "excess_normalized_mutual_information", "feature"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    table["dependence_rank"] = table.groupby("variant").cumcount() + 1
    return table[_target_columns()]


def _feature_pair_independence_table(
    sample: pd.DataFrame,
    *,
    variant: str,
    features: Sequence[str],
    n_bins: int,
    permutation_count: int,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    discretized = {feature: _quantile_codes(sample[feature], n_bins=n_bins) for feature in features}
    pair_index = 0
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            stats = _dependence_stats(
                discretized[left],
                discretized[right],
                permutation_count=permutation_count,
                random_state=random_state + 10_000 + pair_index,
            )
            rows.append(
                {
                    "variant": variant,
                    "feature_left": left,
                    "feature_right": right,
                    "rows": int(stats["rows"]),
                    "n_bins": int(n_bins),
                    "left_bins": int(stats["x_bins"]),
                    "right_bins": int(stats["y_bins"]),
                    "normalized_mutual_information": stats[
                        "normalized_mutual_information"
                    ],
                    "permutation_mean_nmi": stats["permutation_mean_nmi"],
                    "excess_normalized_mutual_information": stats[
                        "excess_normalized_mutual_information"
                    ],
                    "permutation_p_value": stats["permutation_p_value"],
                    "independence_label": _independence_label(
                        stats["excess_normalized_mutual_information"]
                    ),
                    "test_window_used": False,
                }
            )
            pair_index += 1
    table = pd.DataFrame(rows, columns=_pair_columns())
    if table.empty:
        return table
    table = table.sort_values(
        ["variant", "excess_normalized_mutual_information", "feature_left", "feature_right"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    table["dependence_rank"] = table.groupby("variant").cumcount() + 1
    return table[_pair_columns()]


def _quantile_codes(series: pd.Series, *, n_bins: int) -> np.ndarray | None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    unique_count = int(values.nunique(dropna=True))
    if unique_count < 2:
        return None
    bins = min(n_bins, unique_count)
    codes = pd.qcut(values, q=bins, labels=False, duplicates="drop")
    if pd.isna(codes).any() or codes.nunique(dropna=True) < 2:
        return None
    return codes.astype(int).to_numpy()


def _dependence_stats(
    x_codes: np.ndarray | None,
    y_codes: np.ndarray | None,
    *,
    permutation_count: int,
    random_state: int,
) -> dict[str, float | int]:
    if x_codes is None or y_codes is None:
        return _empty_dependence_stats()
    if len(x_codes) != len(y_codes) or len(x_codes) == 0:
        return _empty_dependence_stats()
    observed = float(
        normalized_mutual_info_score(x_codes, y_codes, average_method="arithmetic")
    )
    rng = np.random.default_rng(random_state)
    permutation_scores = []
    for _ in range(permutation_count):
        shuffled = rng.permutation(y_codes)
        permutation_scores.append(
            float(
                normalized_mutual_info_score(
                    x_codes, shuffled, average_method="arithmetic"
                )
            )
        )
    permutation_array = np.asarray(permutation_scores, dtype=float)
    permutation_mean = float(permutation_array.mean()) if len(permutation_array) else 0.0
    excess = observed - permutation_mean
    p_value = float((1 + np.sum(permutation_array >= observed)) / (1 + permutation_count))
    return {
        "rows": int(len(x_codes)),
        "x_bins": int(pd.Series(x_codes).nunique()),
        "y_bins": int(pd.Series(y_codes).nunique()),
        "normalized_mutual_information": observed,
        "permutation_mean_nmi": permutation_mean,
        "excess_normalized_mutual_information": excess,
        "permutation_p_value": p_value,
    }


def _empty_dependence_stats() -> dict[str, float | int]:
    return {
        "rows": 0,
        "x_bins": 0,
        "y_bins": 0,
        "normalized_mutual_information": np.nan,
        "permutation_mean_nmi": np.nan,
        "excess_normalized_mutual_information": np.nan,
        "permutation_p_value": np.nan,
    }


def _independence_label(excess_nmi: float | int | None) -> str:
    if excess_nmi is None or pd.isna(excess_nmi):
        return "not_estimable"
    if excess_nmi <= 0.001:
        return "near_independent_after_permutation"
    if excess_nmi <= 0.01:
        return "weak_dependence"
    if excess_nmi <= 0.05:
        return "moderate_dependence"
    return "strong_dependence"


def _feature_dependence_clusters(
    pair_table: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for variant, group in pair_table.groupby("variant", sort=True):
        strong = group[group["excess_normalized_mutual_information"] >= threshold]
        components = _connected_components(strong)
        for cluster_id, features in enumerate(components, start=1):
            feature_set = set(features)
            pairs = strong[
                strong["feature_left"].isin(feature_set)
                & strong["feature_right"].isin(feature_set)
            ]
            rows.append(
                {
                    "variant": variant,
                    "cluster_id": cluster_id,
                    "features": "|".join(sorted(feature_set)),
                    "feature_count": len(feature_set),
                    "pair_count": int(len(pairs)),
                    "max_excess_normalized_mutual_information": float(
                        pairs["excess_normalized_mutual_information"].max()
                    ),
                    "mean_excess_normalized_mutual_information": float(
                        pairs["excess_normalized_mutual_information"].mean()
                    ),
                    "threshold": float(threshold),
                    "test_window_used": False,
                }
            )
    return pd.DataFrame(rows, columns=_cluster_columns())


def _connected_components(pair_table: pd.DataFrame) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {}
    for _, row in pair_table.iterrows():
        left = str(row["feature_left"])
        right = str(row["feature_right"])
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    seen: set[str] = set()
    components: list[list[str]] = []
    for node in sorted(adjacency):
        if node in seen:
            continue
        stack = [node]
        component: list[str] = []
        seen.add(node)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _feature_independence_memo(
    target_table: pd.DataFrame,
    pair_table: pd.DataFrame,
    clusters: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    n_bins: int,
    max_rows: int,
    permutation_count: int,
    cluster_threshold: float,
) -> str:
    lines = [
        "# Pure Alpha Phase 4E Feature Independence Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "This is a validation-only nonlinear independence diagnostic for the "
        "current Phase 4D feature set. It does not use Pearson or Spearman "
        "correlation and does not inspect the test lockbox.",
        "",
        "## Method",
        "",
        "- metric: quantile-discretized normalized mutual information;",
        f"- quantile bins: `{n_bins}`;",
        f"- maximum sampled rows per variant: `{max_rows}`;",
        f"- permutation baseline count: `{permutation_count}`;",
        f"- feature-cluster threshold: `{cluster_threshold}` excess NMI;",
        "- interpretation: excess NMI near `0` means approximately independent "
        "after subtracting finite-sample permutation bias.",
        "",
        "## Sample",
        "",
        "| Variant | Loaded Rows | Complete-Case Rows | Sampled Rows | Start | End |",
        "|---|---:|---:|---:|---|---|",
    ]
    for _, row in manifest.iterrows():
        lines.append(
            f"| {row['variant']} | {row['loaded_rows']} | {row['complete_case_rows']} | "
            f"{row['sampled_rows']} | {row['sample_start_session']} | "
            f"{row['sample_end_session']} |"
        )
    lines.extend(
        [
            "",
            "## Feature Versus Future 5D Residual Return",
            "",
            "| Rank | Feature | Excess NMI | Observed NMI | Permutation Mean | Label |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for _, row in target_table.head(20).iterrows():
        lines.append(
            f"| {row['dependence_rank']} | {row['feature']} | "
            f"{row['excess_normalized_mutual_information']} | "
            f"{row['normalized_mutual_information']} | {row['permutation_mean_nmi']} | "
            f"{row['independence_label']} |"
        )
    lines.extend(
        [
            "",
            "## Feature Versus Feature",
            "",
            "| Rank | Feature Left | Feature Right | Excess NMI | Observed NMI | Label |",
            "|---:|---|---|---:|---:|---|",
        ]
    )
    for _, row in pair_table.head(30).iterrows():
        lines.append(
            f"| {row['dependence_rank']} | {row['feature_left']} | "
            f"{row['feature_right']} | {row['excess_normalized_mutual_information']} | "
            f"{row['normalized_mutual_information']} | {row['independence_label']} |"
        )
    lines.extend(
        [
            "",
            "## Strong Feature Clusters",
            "",
            "| Variant | Cluster | Feature Count | Pair Count | Max Excess NMI | Features |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in clusters.iterrows():
        lines.append(
            f"| {row['variant']} | {row['cluster_id']} | {row['feature_count']} | "
            f"{row['pair_count']} | {row['max_excess_normalized_mutual_information']} | "
            f"{row['features']} |"
        )
    if clusters.empty:
        lines.append("| n/a | n/a | 0 | 0 | n/a | n/a |")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Mutual information detects nonlinear dependence, but it is not a "
            "portfolio return estimate.",
            "- High feature-feature dependence flags redundancy, not necessarily a bad "
            "feature.",
            "- Low feature-target dependence does not prove there is no conditional "
            "alpha after interactions.",
            "- Test lockbox remains closed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _concat_or_empty(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    return pd.concat(non_empty, ignore_index=True)[columns]


def _target_columns() -> list[str]:
    return [
        "variant",
        "dependence_rank",
        "feature",
        "target",
        "rows",
        "n_bins",
        "feature_bins",
        "target_bins",
        "normalized_mutual_information",
        "permutation_mean_nmi",
        "excess_normalized_mutual_information",
        "permutation_p_value",
        "independence_label",
        "test_window_used",
    ]


def _pair_columns() -> list[str]:
    return [
        "variant",
        "dependence_rank",
        "feature_left",
        "feature_right",
        "rows",
        "n_bins",
        "left_bins",
        "right_bins",
        "normalized_mutual_information",
        "permutation_mean_nmi",
        "excess_normalized_mutual_information",
        "permutation_p_value",
        "independence_label",
        "test_window_used",
    ]


def _cluster_columns() -> list[str]:
    return [
        "variant",
        "cluster_id",
        "features",
        "feature_count",
        "pair_count",
        "max_excess_normalized_mutual_information",
        "mean_excess_normalized_mutual_information",
        "threshold",
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
        description="Run pure-alpha Phase 4E feature independence diagnostics."
    )
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_PHASE3_SIGNAL_PANEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--permutation-count", type=int, default=5)
    parser.add_argument("--cluster-threshold", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=260321)
    args = parser.parse_args(argv)

    result = build_phase4e_feature_independence_artifacts(
        signal_panel_path=args.signal_panel_path,
        output_root=args.output_root,
        variant_names=tuple(args.variants or DEFAULT_VARIANTS),
        n_bins=args.n_bins,
        max_rows=args.max_rows,
        permutation_count=args.permutation_count,
        cluster_threshold=args.cluster_threshold,
        random_state=args.random_state,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
