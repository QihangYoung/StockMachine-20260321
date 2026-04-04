from __future__ import annotations

import json

import pandas as pd

from stockmachine.research.robustness_reports import (
    write_robustness_gate_artifact,
    write_robustness_overview_artifact,
)


def test_write_robustness_overview_artifact_persists_csv(tmp_path) -> None:
    path = tmp_path / "robustness_overview.csv"
    frame = write_robustness_overview_artifact(
        path,
        [{"model": "ridge", "pass": True}],
    )

    assert path.exists()
    assert frame["model"].iloc[0] == "ridge"


def test_write_robustness_gate_artifact_persists_json(tmp_path) -> None:
    path = tmp_path / "robustness_gate.json"
    write_robustness_gate_artifact(path, {"strategy_project": "us_equities_h1", "status": "warning"})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "warning"
