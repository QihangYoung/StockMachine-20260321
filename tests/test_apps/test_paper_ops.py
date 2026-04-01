from __future__ import annotations

import json
from datetime import date, datetime, timezone

from stockmachine.apps.paper_ops import build_arg_parser, dispatch_command, main
from stockmachine.domain.project_paths import build_strategy_project_paths
from stockmachine.state import LocalLedger, OrderDecisionRecord, OrderRecord, RunManifestRecord, RunRecord


def _seed_ledger(ledger: LocalLedger) -> None:
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
    ledger.record_order_decision(
        OrderDecisionRecord(
            decision_id="decision-001",
            run_id="run-001",
            session_date=date(2026, 3, 22),
            client_order_id="client-open",
            symbol="AAPL",
            side="BUY",
            decision_type="risk_gate",
            decision_price=100.0,
            estimated_notional=100.0,
            approved=True,
            reason="approved",
            decision_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            meta={},
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


def test_paper_ops_parser_accepts_expected_subcommands() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--ledger-path", "ledger.sqlite3", "run-summary", "--run-id", "run-001"])

    assert args.ledger_path == "ledger.sqlite3"
    assert args.command == "run-summary"
    assert args.run_id == "run-001"


def test_paper_ops_latest_run_and_open_orders_json(tmp_path, capsys) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    _seed_ledger(ledger)

    payload = dispatch_command(
        build_arg_parser().parse_args(["--ledger-path", str(tmp_path / "ledger.sqlite3"), "latest-run"])
    )
    assert payload["command"] == "latest-run"
    assert payload["run_id"] == "run-001"
    assert payload["summary"]["order_decision_count"] == 1
    assert payload["report"]["meta"]["run_manifest"]["data_snapshot"]["snapshot_age_days"] == 3
    assert {alert["code"] for alert in payload["alerts"]} >= {
        "data_stale",
        "duplicate_run_blocked",
        "broker_rejection",
        "open_orders_lingering",
    }

    main(["--ledger-path", str(tmp_path / "ledger.sqlite3"), "open-orders"])
    raw = capsys.readouterr().out
    open_payload = json.loads(raw)
    assert open_payload["command"] == "open-orders"
    assert open_payload["count"] == 1
    assert open_payload["open_orders"][0]["symbol"] == "AAPL"
    assert {alert["code"] for alert in open_payload["alerts"]} >= {
        "data_stale",
        "duplicate_run_blocked",
        "broker_rejection",
        "open_orders_lingering",
    }


def test_paper_ops_run_summary_json(tmp_path, capsys) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    _seed_ledger(ledger)

    main(["--ledger-path", str(tmp_path / "ledger.sqlite3"), "run-summary", "--run-id", "run-001"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["command"] == "run-summary"
    assert payload["run_id"] == "run-001"
    assert payload["summary"]["order_count"] == 2
    assert payload["summary"]["open_order_count"] == 1
    assert payload["report"]["failures"][0]["reason"] == "duplicate_run_blocked"
    assert any(order["status"] == "rejected" for order in payload["orders"])


def test_paper_ops_strategy_project_resolves_default_ledger(tmp_path, capsys) -> None:
    artifact_root = tmp_path / "artifacts"
    workspace = build_strategy_project_paths("us_equities_h5", artifact_root=artifact_root)
    workspace.paper_root.mkdir(parents=True, exist_ok=True)
    ledger = LocalLedger(workspace.ledger_path)
    ledger.initialize()
    _seed_ledger(ledger)

    main(
        [
            "--strategy-project",
            "us_equities_h5",
            "--artifact-root",
            str(artifact_root),
            "latest-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["command"] == "latest-run"
    assert payload["strategy_workspace"]["ledger_path"] == str(workspace.ledger_path)
