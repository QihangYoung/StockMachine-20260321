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


def test_paper_smoke_strategy_profile_sets_defaults_and_recommended_command(tmp_path, monkeypatch, capsys) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / "us-ridge-daily-2026-03-22-ridge"
    artifact_dir.mkdir(parents=True)
    for file_name in ("backtest_summary.csv", "backtest_records.csv", "predictions.csv"):
        (artifact_dir / file_name).write_text("dummy\n", encoding="utf-8")

    def _run_command(args):
        return {
            "command": "run",
            "ok": True,
            "summary": {"decision": "executed"},
            "preflight": {"allowed": True},
            "run": {"report": {"run_id": "run-smoke"}},
            "post_run": {"ok": True},
            "error": None,
        }

    monkeypatch.setattr(paper_smoke.paper_daily, "run_command", _run_command)

    exit_code = paper_smoke.main(
        [
            "--strategy-profile",
            "us_ridge_daily",
            "--artifact-root",
            str(artifact_root),
            "--ledger-path",
            str(tmp_path / "ledger.sqlite3"),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["strategy_profile"] == "us_ridge_daily"
    assert output["model"] == "ridge"
    assert output["run_name"] == "us-ridge-daily"
    assert output["artifact_link"]["artifact_dir"] == str(artifact_dir)
    assert "--strategy-profile" in output["recommended_commands"]["paper_daily"]
    assert "us_ridge_daily" in output["recommended_commands"]["paper_daily"]


def test_paper_smoke_parse_args_reads_sys_argv_when_not_explicit(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["paper_smoke", "--strategy-profile", "us_hist_gbm_ridge_mean_daily"],
    )

    args = paper_smoke.parse_args()

    assert args.strategy_profile == "us_hist_gbm_ridge_mean_daily"
    assert args.model == "ensemble_hist_gbm_ridge_mean"
    assert args.run_name == "us-hist-gbm-ridge-mean-daily"


def test_paper_smoke_rank_profile_sets_defaults(tmp_path, monkeypatch, capsys) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / "us_equities_silver_chain_2025_ensemble_hist_gbm_ridge_rank"
    artifact_dir.mkdir(parents=True)
    for file_name in ("backtest_summary.csv", "backtest_records.csv", "predictions.csv"):
        (artifact_dir / file_name).write_text("dummy\n", encoding="utf-8")

    def _run_command(args):
        return {
            "command": "run",
            "ok": True,
            "summary": {"decision": "executed"},
            "preflight": {"allowed": True},
            "run": {"report": {"run_id": "run-smoke-rank"}},
            "post_run": {"ok": True},
            "error": None,
        }

    monkeypatch.setattr(paper_smoke.paper_daily, "run_command", _run_command)

    exit_code = paper_smoke.main(
        [
            "--strategy-profile",
            "us_hist_gbm_ridge_rank_daily",
            "--artifact-root",
            str(artifact_root),
            "--ledger-path",
            str(tmp_path / "ledger.sqlite3"),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["strategy_profile"] == "us_hist_gbm_ridge_rank_daily"
    assert output["model"] == "ensemble_hist_gbm_ridge_rank"
    assert output["run_name"] == "us-hist-gbm-ridge-rank-daily"


def test_paper_smoke_random_forest_rank_profile_sets_defaults(tmp_path, monkeypatch, capsys) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / "us_equities_silver_chain_2025_ensemble_hist_gbm_random_forest_rank"
    artifact_dir.mkdir(parents=True)
    for file_name in ("backtest_summary.csv", "backtest_records.csv", "predictions.csv"):
        (artifact_dir / file_name).write_text("dummy\n", encoding="utf-8")

    def _run_command(args):
        return {
            "command": "run",
            "ok": True,
            "summary": {"decision": "executed"},
            "preflight": {"allowed": True},
            "run": {"report": {"run_id": "run-smoke-rf-rank"}},
            "post_run": {"ok": True},
            "error": None,
        }

    monkeypatch.setattr(paper_smoke.paper_daily, "run_command", _run_command)

    exit_code = paper_smoke.main(
        [
            "--strategy-profile",
            "us_hist_gbm_random_forest_rank_daily",
            "--artifact-root",
            str(artifact_root),
            "--ledger-path",
            str(tmp_path / "ledger.sqlite3"),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["strategy_profile"] == "us_hist_gbm_random_forest_rank_daily"
    assert output["model"] == "ensemble_hist_gbm_random_forest_rank"
    assert output["run_name"] == "us-hist-gbm-random-forest-rank-daily"
