from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stockmachine.apps import run_robustness_suite as app
from stockmachine.domain.project_paths import build_strategy_project_paths


def _write_records(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "signal_date": "2025-01-02",
                "entry_date": "2025-01-03",
                "exit_date": "2025-01-10",
                "gross_return": 0.02,
                "net_return": 0.019,
                "benchmark_return": 0.01,
                "turnover": 0.4,
                "cost_bps": 4.0,
                "positions": 6,
            },
            {
                "signal_date": "2025-01-09",
                "entry_date": "2025-01-10",
                "exit_date": "2025-01-17",
                "gross_return": -0.01,
                "net_return": -0.011,
                "benchmark_return": -0.004,
                "turnover": 0.2,
                "cost_bps": 2.0,
                "positions": 6,
            },
        ]
    ).to_csv(path, index=False)


def test_run_robustness_suite_writes_expected_outputs(tmp_path, capsys) -> None:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    records_path = candidate_dir / "records.csv"
    _write_records(records_path)

    manifest_path = tmp_path / "candidate_manifest.csv"
    pd.DataFrame(
        [
            {
                "report_name": "ridge_v3_A",
                "model": "ridge",
                "source": "feature_v3",
                "config_id": "A_topk6_mt0p3_mwc0p02",
                "records_path": str(records_path),
            }
        ]
    ).to_csv(manifest_path, index=False)

    output_root = tmp_path / "robustness_output"
    exit_code = app.main(
        [
            "--candidate-manifest",
            str(manifest_path),
            "--strategy-project",
            "us_equities_h1",
            "--horizon",
            "1",
            "--output-root",
            str(output_root),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["analyzers"]["parameter_stability"] is False
    assert (output_root / "robustness_overview.csv").exists()
    assert (output_root / "per_model" / "ridge_v3_A" / "time_stability_yearly.csv").exists()
    assert (output_root / "per_model" / "ridge_v3_A" / "tail_dependence_summary.csv").exists()
    assert (output_root / "run_meta.json").exists()


def test_run_robustness_suite_defaults_to_project_scoped_root(tmp_path, capsys) -> None:
    workspace = build_strategy_project_paths("us_equities_h5", artifact_root=tmp_path / "artifacts")
    candidate_dir = tmp_path / "manifests"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    records_path = candidate_dir / "h5_records.csv"
    _write_records(records_path)

    manifest_path = candidate_dir / "candidate_manifest.csv"
    pd.DataFrame(
        [
            {
                "report_name": "ridge_h5",
                "model": "ridge",
                "records_path": str(records_path),
            }
        ]
    ).to_csv(manifest_path, index=False)

    exit_code = app.main(
        [
            "--candidate-manifest",
            str(manifest_path),
            "--strategy-project",
            "us_equities_h5",
            "--horizon",
            "5",
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    expected_root = workspace.research_root / "robustness_suite"
    assert exit_code == 0
    assert payload["output_root"] == str(expected_root)
    assert payload["strategy_project"] == "us_equities_h5"
    assert (expected_root / "robustness_overview.csv").exists()
