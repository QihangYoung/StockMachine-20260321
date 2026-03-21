from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from stockmachine.apps import paper_smoke


def _make_artifact_dir(root: Path) -> Path:
    artifact_dir = root / "paper-smoke-2026-03-22-hist_gbm"
    artifact_dir.mkdir(parents=True)
    for file_name in ("backtest_summary.csv", "backtest_records.csv", "predictions.csv"):
        (artifact_dir / file_name).write_text("dummy\n", encoding="utf-8")
    return artifact_dir


def test_paper_smoke_autolinks_artifacts_and_builds_commands(tmp_path, monkeypatch, capsys) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_dir = _make_artifact_dir(artifact_root)
    calls: dict[str, object] = {}

    def _run_command(args):
        calls["run_args"] = args
        return {
            "command": "run",
            "ok": True,
            "summary": {"decision": "executed"},
            "preflight": {"allowed": True},
            "run": {"report": {"run_id": "run-smoke"}},
            "post_run": {"ok": True, "reconciliation": {"found": True}},
            "error": None,
        }

    monkeypatch.setattr(paper_smoke.paper_daily, "run_command", _run_command)

    exit_code = paper_smoke.main(
        [
            "--session-date",
            "2026-03-22",
            "--run-name",
            "paper-smoke",
            "--artifact-root",
            str(artifact_root),
            "--ledger-path",
            str(tmp_path / "ledger.sqlite3"),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["ok"] is True
    assert output["artifact_link"]["artifact_dir"] == str(artifact_dir)
    assert output["artifact_link"]["exists"] is True
    assert output["run_id"] == "run-smoke"
    assert output["recommended_commands"]["paper_reconcile"][0] == "python"
    assert "--run-id" in output["recommended_commands"]["paper_reconcile"]
    assert str(artifact_dir) in output["recommended_commands"]["paper_reconcile"]
    assert calls["run_args"].artifact_dir == str(artifact_dir)
    assert calls["run_args"].execute is False
    assert output["summary"]["decision"] == "executed"


def test_paper_smoke_reports_unresolved_artifacts_without_failing(tmp_path, monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    def _run_command(args):
        calls["run_args"] = args
        return {
            "command": "run",
            "ok": True,
            "summary": {"decision": "executed"},
            "preflight": {"allowed": True},
            "run": {"report": {"run_id": "run-smoke"}},
            "post_run": None,
            "error": None,
        }

    monkeypatch.setattr(paper_smoke.paper_daily, "run_command", _run_command)

    exit_code = paper_smoke.main(
        [
            "--session-date",
            "2026-03-22",
            "--run-name",
            "paper-smoke",
            "--artifact-root",
            str(tmp_path / "missing"),
            "--ledger-path",
            str(tmp_path / "ledger.sqlite3"),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["artifact_link"]["exists"] is False
    assert output["artifact_link"]["source"] == "unresolved"
    assert calls["run_args"].artifact_dir is None
