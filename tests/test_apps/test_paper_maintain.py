from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pandas as pd

from stockmachine.apps.paper_maintain import main
from stockmachine.domain.project_paths import build_strategy_project_paths
from stockmachine.monitoring.reports import build_paper_run_manifest
from stockmachine.state import LocalLedger, OrderRecord


def _seed_cli_ledger(tmp_path) -> LocalLedger:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    ledger.record_run_manifest(
        build_paper_run_manifest(
            run_id="run-maint",
            session_date=date(2026, 3, 22),
            strategy_name="demo",
            model_name="hist_gbm",
            dry_run=False,
        ).to_record()
    )
    ledger.upsert_order(
        OrderRecord(
            order_id="order-aapl",
            run_id="run-maint",
            session_date=date(2026, 3, 22),
            client_order_id="client-aapl",
            symbol="AAPL",
            side="buy",
            quantity=10,
            order_type="market",
            limit_price=None,
            status="submitted",
            filled_quantity=0,
            avg_fill_price=None,
            submitted_at_utc=datetime(2026, 3, 22, 13, 50, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 13, 50, tzinfo=timezone.utc),
            broker_payload={},
        )
    )
    ledger.upsert_order(
        OrderRecord(
            order_id="order-msft",
            run_id="run-maint",
            session_date=date(2026, 3, 22),
            client_order_id="client-msft",
            symbol="MSFT",
            side="buy",
            quantity=5,
            order_type="market",
            limit_price=None,
            status="submitted",
            filled_quantity=0,
            avg_fill_price=None,
            submitted_at_utc=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
            broker_payload={},
        )
    )
    return ledger


def test_paper_maintain_cli_outputs_json_plan(tmp_path, capsys) -> None:
    ledger = _seed_cli_ledger(tmp_path)
    broker_orders_path = tmp_path / "broker_orders.json"
    pd.DataFrame(
        [
            {
                "id": "order-aapl",
                "client_order_id": "client-aapl",
                "symbol": "AAPL",
                "side": "buy",
                "status": "accepted",
                "qty": 10,
                "filled_qty": 0,
                "type": "market",
                "submitted_at": "2026-03-22T13:50:00+00:00",
                "updated_at": "2026-03-22T13:51:00+00:00",
            },
            {
                "id": "broker-gme",
                "client_order_id": "client-gme",
                "symbol": "GME",
                "side": "buy",
                "status": "accepted",
                "qty": 1,
                "filled_qty": 0,
                "type": "market",
                "submitted_at": "2026-03-22T12:30:00+00:00",
                "updated_at": "2026-03-22T12:31:00+00:00",
            },
        ]
    ).to_json(broker_orders_path, orient="records")

    exit_code = main(
        [
            "--ledger",
            str(ledger.path),
            "latest-run",
            "--broker-orders-json",
            str(broker_orders_path),
            "--stale-after-minutes",
            "60",
            "--as-of-utc",
            "2026-03-22T14:00:00+00:00",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["run_id"] == "run-maint"
    assert output["counts"]["aligned_open_order_count"] == 1
    assert output["counts"]["stale_ledger_order_count"] == 1
    assert output["counts"]["stale_broker_order_count"] == 1
    assert output["counts"]["cancel_candidate_count"] == 2
    assert output["plan"]["recovery_plan"]["aligned_count"] == 1
    assert output["plan"]["recovery_plan"]["orphan_count"] == 1
    assert output["plan"]["recovery_plan"]["stale_count"] == 1


def test_paper_maintain_strategy_project_resolves_default_ledger(tmp_path, capsys) -> None:
    artifact_root = tmp_path / "artifacts"
    workspace = build_strategy_project_paths("us_equities_h5", artifact_root=artifact_root)
    workspace.paper_root.mkdir(parents=True, exist_ok=True)
    ledger = LocalLedger(workspace.ledger_path)
    ledger.initialize()
    ledger.record_run_manifest(
        build_paper_run_manifest(
            run_id="run-maint",
            session_date=date(2026, 3, 22),
            strategy_name="demo",
            model_name="hist_gbm",
            dry_run=False,
        ).to_record()
    )
    ledger.upsert_order(
        OrderRecord(
            order_id="order-aapl",
            run_id="run-maint",
            session_date=date(2026, 3, 22),
            client_order_id="client-aapl",
            symbol="AAPL",
            side="buy",
            quantity=10,
            order_type="market",
            limit_price=None,
            status="submitted",
            filled_quantity=0,
            avg_fill_price=None,
            submitted_at_utc=datetime(2026, 3, 22, 13, 50, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 13, 50, tzinfo=timezone.utc),
            broker_payload={},
        )
    )
    ledger.close()
    broker_orders_path = tmp_path / "broker_orders.json"
    pd.DataFrame([]).to_json(broker_orders_path, orient="records")

    exit_code = main(
        [
            "--strategy-project",
            "us_equities_h5",
            "--artifact-root",
            str(artifact_root),
            "latest-run",
            "--broker-orders-json",
            str(broker_orders_path),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["strategy_workspace"]["ledger_path"] == str(workspace.ledger_path)
