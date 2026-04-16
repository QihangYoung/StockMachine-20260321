from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stockmachine.apps import run_fmf_validation_subperiod_review as app


def _write_records(path: Path, *, start_date: str, end_date: str) -> None:
    pd.DataFrame(
        [
            {
                "signal_date": start_date,
                "entry_date": start_date,
                "exit_date": start_date,
                "gross_return": 0.010,
                "net_return": 0.010,
                "benchmark_return": 0.005,
                "turnover": 0.10,
                "cost_bps": 0.0,
                "positions": 7,
            },
            {
                "signal_date": "2015-12-31",
                "entry_date": "2015-12-31",
                "exit_date": "2015-12-31",
                "gross_return": -0.005,
                "net_return": -0.005,
                "benchmark_return": -0.002,
                "turnover": 0.10,
                "cost_bps": 0.0,
                "positions": 7,
            },
            {
                "signal_date": "2017-12-29",
                "entry_date": "2017-12-29",
                "exit_date": "2017-12-29",
                "gross_return": 0.008,
                "net_return": 0.008,
                "benchmark_return": 0.004,
                "turnover": 0.10,
                "cost_bps": 0.0,
                "positions": 7,
            },
            {
                "signal_date": end_date,
                "entry_date": end_date,
                "exit_date": end_date,
                "gross_return": 0.006,
                "net_return": 0.006,
                "benchmark_return": 0.003,
                "turnover": 0.10,
                "cost_bps": 0.0,
                "positions": 7,
            },
        ]
    ).to_csv(path, index=False)


def test_run_fmf_validation_subperiod_review_writes_outputs(tmp_path) -> None:
    records_a = tmp_path / "a.csv"
    records_b = tmp_path / "b.csv"
    _write_records(records_a, start_date="2014-08-05", end_date="2019-12-31")
    _write_records(records_b, start_date="2014-10-01", end_date="2019-12-31")

    manifest = pd.DataFrame(
        [
            {
                "report_name": "candidate_a",
                "source": "baseline",
                "config_id": "candidate_a",
                "shortlist_role": "lead",
                "records_path": str(records_a),
            },
            {
                "report_name": "candidate_b",
                "source": "baseline",
                "config_id": "candidate_b",
                "shortlist_role": "neighbor",
                "records_path": str(records_b),
            },
        ]
    )
    manifest_path = tmp_path / "shortlist.csv"
    manifest.to_csv(manifest_path, index=False)

    output_root = tmp_path / "out"
    exit_code = app.main(
        [
            "--shortlist-manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    subperiod_summary = pd.read_csv(output_root / "subperiod_summary.csv")
    stability_overview = pd.read_csv(output_root / "stability_overview.csv")
    run_meta = json.loads((output_root / "run_meta.json").read_text(encoding="utf-8"))

    assert set(subperiod_summary["subperiod_label"]) == {
        "block_2014_2015",
        "block_2016_2017",
        "block_2018_2019",
    }
    assert len(stability_overview) == 2
    assert run_meta["common_window"]["start_date"] == "2014-10-01"
    assert run_meta["common_window"]["end_date"] == "2019-12-31"


def test_run_fmf_validation_subperiod_review_rejects_lockbox_leak(tmp_path) -> None:
    records_a = tmp_path / "a.csv"
    _write_records(records_a, start_date="2014-08-05", end_date="2020-01-02")

    manifest = pd.DataFrame(
        [
            {
                "report_name": "candidate_a",
                "source": "baseline",
                "config_id": "candidate_a",
                "shortlist_role": "lead",
                "records_path": str(records_a),
            }
        ]
    )
    manifest_path = tmp_path / "shortlist.csv"
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="validation lockbox boundary"):
        app.main(
            [
                "--shortlist-manifest",
                str(manifest_path),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
