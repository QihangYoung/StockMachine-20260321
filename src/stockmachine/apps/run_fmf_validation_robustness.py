from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from stockmachine.apps import run_robustness_suite
from stockmachine.research.strict_reports import write_csv_artifact, write_json_artifact


DEFAULT_BASELINE_ROOT = Path("artifacts/fmf_validation_baseline_validation_only_20260409")
DEFAULT_BROAD_GRID_ROOT = Path("artifacts/fmf_validation_c2_grid_search_20260409")
DEFAULT_NARROW_GRID_ROOT = Path("artifacts/fmf_validation_c2_grid_search_narrow_20260409")
DEFAULT_OUTPUT_ROOT = Path("artifacts/fmf_validation_robustness_20260409")
DEFAULT_STRATEGY_PROJECT = "multi_asset_fmf_validation"
DEFAULT_HORIZON = 1
DEFAULT_TOP_N_CANDIDATES = 5

_BASELINE_STRATEGY_NAMES: tuple[str, ...] = (
    "static_equal_weight_non_cash",
    "rolling_erc_core",
    "rolling_c2_v0_seed_core",
)
_PARAMETER_COLUMNS: tuple[str, ...] = (
    "equity_total",
    "credit",
    "duration",
    "inflation_hedge",
    "trend",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run validation-only robustness analysis for the FMF rebuilt multi-asset line."
    )
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--broad-grid-root", default=str(DEFAULT_BROAD_GRID_ROOT))
    parser.add_argument("--narrow-grid-root", default=str(DEFAULT_NARROW_GRID_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--top-n-candidates", type=int, default=DEFAULT_TOP_N_CANDIDATES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    baseline_root = Path(args.baseline_root)
    broad_grid_root = Path(args.broad_grid_root)
    narrow_grid_root = Path(args.narrow_grid_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    _validate_baseline_source_root(baseline_root)
    _validate_grid_source_root(broad_grid_root)
    _validate_grid_source_root(narrow_grid_root)

    manifests_root = output_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    candidate_manifest = build_candidate_manifest(
        baseline_root=baseline_root,
        narrow_grid_root=narrow_grid_root,
        top_n_candidates=int(args.top_n_candidates),
    )
    candidate_manifest_path = manifests_root / "candidate_manifest.csv"
    write_csv_artifact(candidate_manifest_path, candidate_manifest)

    broad_surface = build_grid_search_surface(broad_grid_root)
    narrow_surface = build_grid_search_surface(narrow_grid_root)
    broad_surface_path = manifests_root / "broad_grid_surface.csv"
    narrow_surface_path = manifests_root / "narrow_grid_surface.csv"
    write_csv_artifact(broad_surface_path, broad_surface)
    write_csv_artifact(narrow_surface_path, narrow_surface)

    selection_bias_manifest = pd.DataFrame(
        [
            {
                "run_label": "fmf_validation_broad_search",
                "surface_path": str(broad_surface_path.resolve()),
                "metric_column": "sharpe",
                "higher_is_better": True,
                "family_column": "equity_total",
            },
            {
                "run_label": "fmf_validation_narrow_search",
                "surface_path": str(narrow_surface_path.resolve()),
                "metric_column": "sharpe",
                "higher_is_better": True,
                "family_column": "equity_total",
            },
        ]
    )
    selection_bias_manifest_path = manifests_root / "selection_bias_manifest.csv"
    write_csv_artifact(selection_bias_manifest_path, selection_bias_manifest)

    parameter_stability_manifest = pd.DataFrame(
        [
            {
                "run_label": "fmf_validation_broad_search",
                "surface_path": str(broad_surface_path.resolve()),
                "metric_column": "sharpe",
                "parameter_columns": ",".join(_PARAMETER_COLUMNS),
                "higher_is_better": True,
                "neighborhood_radius": 1,
            },
            {
                "run_label": "fmf_validation_narrow_search",
                "surface_path": str(narrow_surface_path.resolve()),
                "metric_column": "sharpe",
                "parameter_columns": ",".join(_PARAMETER_COLUMNS),
                "higher_is_better": True,
                "neighborhood_radius": 1,
            },
        ]
    )
    parameter_stability_manifest_path = manifests_root / "parameter_stability_manifest.csv"
    write_csv_artifact(parameter_stability_manifest_path, parameter_stability_manifest)

    suite_output_root = output_root / "robustness_suite"
    exit_code = run_robustness_suite.main(
        [
            "--candidate-manifest",
            str(candidate_manifest_path),
            "--strategy-project",
            DEFAULT_STRATEGY_PROJECT,
            "--horizon",
            str(DEFAULT_HORIZON),
            "--output-root",
            str(suite_output_root),
            "--selection-bias-manifest",
            str(selection_bias_manifest_path),
            "--parameter-stability-manifest",
            str(parameter_stability_manifest_path),
            "--include-parameter-stability",
        ]
    )

    write_json_artifact(
        output_root / "driver_meta.json",
        {
            "ok": exit_code == 0,
            "strategy_project": DEFAULT_STRATEGY_PROJECT,
            "horizon": DEFAULT_HORIZON,
            "baseline_root": str(baseline_root),
            "broad_grid_root": str(broad_grid_root),
            "narrow_grid_root": str(narrow_grid_root),
            "suite_output_root": str(suite_output_root),
            "top_n_candidates": int(args.top_n_candidates),
            "manifests": {
                "candidate_manifest": str(candidate_manifest_path),
                "selection_bias_manifest": str(selection_bias_manifest_path),
                "parameter_stability_manifest": str(parameter_stability_manifest_path),
                "broad_surface": str(broad_surface_path),
                "narrow_surface": str(narrow_surface_path),
            },
            "selected_candidates": candidate_manifest["report_name"].tolist(),
        },
    )
    return int(exit_code)


def build_candidate_manifest(
    *,
    baseline_root: Path,
    narrow_grid_root: Path,
    top_n_candidates: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for strategy_name in _BASELINE_STRATEGY_NAMES:
        rows.append(
            {
                "report_name": strategy_name,
                "model": strategy_name,
                "source": "baseline_validation_only",
                "config_id": strategy_name,
                "records_path": str((baseline_root / strategy_name / "backtest_records.csv").resolve()),
            }
        )

    candidate_summary = pd.read_csv(
        narrow_grid_root / "summary_metrics_c2_candidates_validation_common_window.csv"
    )
    shortlisted = (
        candidate_summary.loc[~candidate_summary["strategy_name"].isin(_BASELINE_STRATEGY_NAMES)]
        .head(int(top_n_candidates))
        .copy()
    )
    for row in shortlisted.to_dict(orient="records"):
        strategy_name = str(row["strategy_name"])
        rows.append(
            {
                "report_name": strategy_name,
                "model": strategy_name,
                "source": "narrow_validation_grid",
                "config_id": strategy_name,
                "records_path": str((narrow_grid_root / strategy_name / "backtest_records.csv").resolve()),
            }
        )

    return pd.DataFrame(rows)


def build_grid_search_surface(grid_root: Path) -> pd.DataFrame:
    policy_grid = pd.read_csv(grid_root / "policy_grid.csv")
    candidate_summary = pd.read_csv(
        grid_root / "summary_metrics_c2_candidates_validation_common_window.csv"
    )
    surface = candidate_summary.merge(
        policy_grid,
        on="strategy_name",
        how="left",
        validate="one_to_one",
    )
    missing = [column for column in _PARAMETER_COLUMNS if column not in surface.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Grid search surface is missing required parameter columns: {joined}")
    return surface


def _validate_baseline_source_root(root: Path) -> None:
    run_meta_path = root / "run_meta.json"
    if not run_meta_path.exists():
        raise FileNotFoundError(f"Missing run_meta.json under baseline root: {run_meta_path}")
    payload = json.loads(run_meta_path.read_text(encoding="utf-8"))
    lockbox = dict(payload.get("lockbox_policy") or {})
    if not bool(lockbox.get("test_window_locked")):
        raise ValueError(f"Baseline root is not lockbox-safe: {root}")
    if bool(lockbox.get("include_test_window")):
        raise ValueError(f"Baseline root has already exposed the test window: {root}")
    if (root / "summary_metrics_test_window.csv").exists():
        raise ValueError(f"Baseline root contains exposed test-window metrics: {root}")


def _validate_grid_source_root(root: Path) -> None:
    sweep_meta_path = root / "sweep_meta.json"
    if not sweep_meta_path.exists():
        raise FileNotFoundError(f"Missing sweep_meta.json under grid root: {sweep_meta_path}")
    payload = json.loads(sweep_meta_path.read_text(encoding="utf-8"))
    lockbox = dict(payload.get("lockbox_policy") or {})
    if not bool(lockbox.get("test_window_locked")):
        raise ValueError(f"Grid root is not lockbox-safe: {root}")
    if bool(lockbox.get("test_window_exposed")):
        raise ValueError(f"Grid root has already exposed the test window: {root}")
    if (root / "summary_metrics_test_window.csv").exists():
        raise ValueError(f"Grid root contains exposed test-window metrics: {root}")


if __name__ == "__main__":
    raise SystemExit(main())
