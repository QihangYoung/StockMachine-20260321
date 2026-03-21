from __future__ import annotations

import json
from datetime import date, datetime, timezone

from stockmachine.live.order_maintenance import (
    OrderMaintenancePolicy,
    apply_order_maintenance_plan,
    build_order_maintenance_summary,
)
from stockmachine.monitoring.reports import build_paper_run_manifest
from stockmachine.state import LocalLedger, OrderRecord


def _seed_maintenance_ledger(tmp_path) -> LocalLedger:
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
    ledger.upsert_order(
        OrderRecord(
            order_id="order-tsla",
            run_id="run-maint",
            session_date=date(2026, 3, 21),
            client_order_id="client-tsla",
            symbol="TSLA",
            side="buy",
            quantity=3,
            order_type="market",
            limit_price=None,
            status="submitted",
            filled_quantity=0,
            avg_fill_price=None,
            submitted_at_utc=datetime(2026, 3, 22, 13, 55, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 13, 55, tzinfo=timezone.utc),
            broker_payload={},
        )
    )
    return ledger


def test_build_order_maintenance_summary_identifies_stale_open_orders(tmp_path) -> None:
    ledger = _seed_maintenance_ledger(tmp_path)
    broker_orders = [
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

    summary = build_order_maintenance_summary(
        ledger,
        broker_orders,
        run_id="run-maint",
        policy=OrderMaintenancePolicy(stale_after_minutes=60),
        as_of_utc=datetime(2026, 3, 22, 14, 0, tzinfo=timezone.utc),
    )

    assert summary.found is True
    assert summary.run_id == "run-maint"
    assert summary.counts["aligned_open_order_count"] == 1
    assert summary.counts["stale_ledger_order_count"] == 2
    assert summary.counts["stale_broker_order_count"] == 1
    assert summary.counts["cancel_candidate_count"] == 3
    assert summary.plan.aligned_open_orders[0].order_id == "order-aapl"
    assert {candidate.order_id for candidate in summary.plan.cancel_candidates} == {
        "order-msft",
        "order-tsla",
        "broker-gme",
    }
    assert "cancel_plan_available" in summary.notes


def test_apply_order_maintenance_plan_supports_optional_execution_seam(tmp_path) -> None:
    ledger = _seed_maintenance_ledger(tmp_path)
    broker_orders = [
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
    summary = build_order_maintenance_summary(
        ledger,
        broker_orders,
        run_id="run-maint",
        policy=OrderMaintenancePolicy(stale_after_minutes=60),
        as_of_utc=datetime(2026, 3, 22, 14, 0, tzinfo=timezone.utc),
    )

    dry_run = apply_order_maintenance_plan(summary.plan, execute=False)
    assert dry_run.requested is False
    assert dry_run.executed is False
    assert dry_run.canceled_order_ids == ()

    class _FakeCanceller:
        def __init__(self) -> None:
            self.canceled: list[str] = []

        def cancel_order(self, order_id: str) -> object:
            self.canceled.append(order_id)
            return {"order_id": order_id, "status": "canceled"}

    canceller = _FakeCanceller()
    execution = apply_order_maintenance_plan(summary.plan, execute=True, canceller=canceller)

    assert execution.requested is True
    assert execution.executed is True
    assert set(execution.canceled_order_ids) == {"order-msft", "order-tsla", "broker-gme"}
    assert canceller.canceled == list(execution.canceled_order_ids)
    assert execution.failed_order_ids == ()


def test_build_order_maintenance_summary_handles_missing_run(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()

    summary = build_order_maintenance_summary(ledger, ())

    assert summary.found is False
    assert summary.plan.cancel_candidates == ()
    assert summary.counts["cancel_candidate_count"] == 0
    assert summary.execution.requested is False

