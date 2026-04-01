from __future__ import annotations

import json
from datetime import date, datetime, timezone

from stockmachine.apps.paper_report import build_arg_parser, dispatch_command, main
from stockmachine.domain.project_paths import build_strategy_project_paths
from stockmachine.state import LocalLedger, OrderRecord, RunManifestRecord, RunRecord


def _seed_digest_ledger(ledger: LocalLedger) -> None:
    ledger.record_run(
        RunRecord(
            run_id="run-001",
            strategy_name="hist_gbm",
            market="US",
            created_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            status="blocked",
            meta={"blocked_reason": "duplicate_run_blocked"},
        )
    )
    ledger.record_run_manifest(
        RunManifestRecord(
            run_id="run-001",
            session_date=date(2026, 3, 22),
            strategy_name="hist_gbm",
            model_name="lightgbm_v1",
            generated_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            client_order_id_prefix="smk",
            dry_run=False,
            data_snapshot={"snapshot_age_days": 3},
            risk_policy={"max_order_notional": 500.0},
            execution_policy={"max_total_orders": 10},
            meta={"source": "tests"},
        )
    )
    ledger.upsert_order(
        OrderRecord(
            order_id="order-open",
            run_id="run-001",
            session_date=date(2026, 3, 22),
            client_order_id="client-open",
            symbol="AAPL",
            side="buy",
            quantity=1,
            order_type="market",
            limit_price=None,
            status="accepted",
            filled_quantity=0,
            avg_fill_price=None,
            submitted_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            broker_payload={},
        )
    )
    ledger.upsert_order(
        OrderRecord(
            order_id="order-rejected",
            run_id="run-001",
            session_date=date(2026, 3, 22),
            client_order_id="client-rejected",
            symbol="MSFT",
            side="buy",
            quantity=1,
            order_type="market",
            limit_price=None,
            status="rejected",
            filled_quantity=0,
            avg_fill_price=None,
            submitted_at_utc=datetime(2026, 3, 22, 1, 1, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 1, 2, tzinfo=timezone.utc),
            broker_payload={},
        )
    )
    ledger.record_run(
        RunRecord(
            run_id="run-002",
            strategy_name="hist_gbm",
            market="US",
            created_at_utc=datetime(2026, 3, 22, 2, 0, tzinfo=timezone.utc),
            status="finished",
            meta={},
        )
    )
    ledger.record_run_manifest(
        RunManifestRecord(
            run_id="run-002",
            session_date=date(2026, 3, 21),
            strategy_name="hist_gbm",
            model_name="lightgbm_v1",
            generated_at_utc=datetime(2026, 3, 22, 2, 0, tzinfo=timezone.utc),
            client_order_id_prefix="smk",
            dry_run=True,
            data_snapshot={"snapshot_age_days": 0},
            risk_policy={"max_order_notional": 500.0},
            execution_policy={"max_total_orders": 10},
            meta={"source": "tests"},
        )
    )


def test_paper_report_parser_accepts_expected_subcommands() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--ledger-path", "ledger.sqlite3", "daily-digest", "--session-date", "2026-03-22"])

    assert args.ledger_path == "ledger.sqlite3"
    assert args.command in {"daily-digest", "daily-summary"}
    assert args.session_date == "2026-03-22"


def test_paper_report_run_index_json(tmp_path, capsys) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    _seed_digest_ledger(ledger)

    main(["--ledger-path", str(tmp_path / "ledger.sqlite3"), "run-index", "--limit", "5"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["command"] == "run-index"
    assert payload["count"] == 2
    assert payload["entries"][0]["run_id"] == "run-002"
    assert payload["entries"][1]["run_id"] == "run-001"


def test_paper_report_daily_digest_health_trend_and_operator_digest_json(tmp_path, capsys) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    _seed_digest_ledger(ledger)

    summary_payload = dispatch_command(
        build_arg_parser().parse_args(
            ["--ledger-path", str(tmp_path / "ledger.sqlite3"), "daily-digest", "--session-date", "2026-03-22"]
        )
    )
    assert summary_payload["command"] == "daily-summary"
    assert summary_payload["session_date"] == "2026-03-22"
    assert summary_payload["run_count"] == 1
    assert summary_payload["alerts"][0]["code"] == "data_stale"

    main(
        [
            "--ledger-path",
            str(tmp_path / "ledger.sqlite3"),
            "health-trend",
            "--limit-runs",
            "5",
            "--limit-sessions",
            "2",
            "--session-date",
            "2026-03-22",
        ]
    )
    trend_payload = json.loads(capsys.readouterr().out)
    assert trend_payload["command"] == "health-trend"
    assert trend_payload["trend"]["open_order_run_count"] == 1
    assert trend_payload["anomaly_summary"]["count"] >= 1

    main(["--ledger-path", str(tmp_path / "ledger.sqlite3"), "anomaly-summary", "--limit-runs", "5"])
    anomaly_payload = json.loads(capsys.readouterr().out)
    assert anomaly_payload["command"] == "anomaly-summary"
    assert anomaly_payload["count"] >= 1

    main(["--ledger-path", str(tmp_path / "ledger.sqlite3"), "operator-digest", "--session-date", "2026-03-22"])
    digest_payload = json.loads(capsys.readouterr().out)

    assert digest_payload["command"] == "operator-digest"
    assert "run_index" in digest_payload
    assert "daily_summary" in digest_payload
    assert "health_trend" in digest_payload
    assert "anomaly_summary" in digest_payload
    assert {alert["code"] for alert in digest_payload["alerts"]} >= {
        "data_stale",
        "duplicate_run_blocked",
        "broker_rejection",
        "open_orders_lingering",
    }


def test_paper_report_strategy_project_resolves_default_ledger(tmp_path, capsys) -> None:
    artifact_root = tmp_path / "artifacts"
    workspace = build_strategy_project_paths("us_equities_h5", artifact_root=artifact_root)
    workspace.paper_root.mkdir(parents=True, exist_ok=True)
    ledger = LocalLedger(workspace.ledger_path)
    ledger.initialize()
    _seed_digest_ledger(ledger)

    main(
        [
            "--strategy-project",
            "us_equities_h5",
            "--artifact-root",
            str(artifact_root),
            "run-index",
            "--limit",
            "5",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["command"] == "run-index"
    assert payload["strategy_workspace"]["ledger_path"] == str(workspace.ledger_path)
