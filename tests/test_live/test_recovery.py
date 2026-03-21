from __future__ import annotations

from datetime import date, datetime, timezone

from stockmachine.live import PollingOrderReconciler, build_recovery_plan, recover_open_orders
from stockmachine.live.reconciler import BrokerOrderSnapshot
from stockmachine.state import LocalLedger, OrderRecord


def test_build_recovery_plan_classifies_aligned_orphan_and_stale_orders() -> None:
    ledger_orders = [
        OrderRecord(
            order_id="ledger-aapl",
            run_id="run-1",
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
            submitted_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            broker_payload={},
        ),
        OrderRecord(
            order_id="ledger-tsla",
            run_id="run-1",
            session_date=date(2026, 3, 22),
            client_order_id="client-tsla",
            symbol="TSLA",
            side="buy",
            quantity=4,
            order_type="market",
            limit_price=None,
            status="submitted",
            filled_quantity=0,
            avg_fill_price=None,
            submitted_at_utc=datetime(2026, 3, 22, 1, 1, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 1, 1, tzinfo=timezone.utc),
            broker_payload={},
        ),
    ]
    broker_orders = [
        BrokerOrderSnapshot.from_payload(
            {
                "id": "ledger-aapl",
                "client_order_id": "client-aapl",
                "symbol": "AAPL",
                "side": "buy",
                "status": "accepted",
                "qty": 10,
                "filled_qty": 0,
                "type": "market",
                "submitted_at": "2026-03-22T01:00:00+00:00",
                "updated_at": "2026-03-22T01:02:00+00:00",
            }
        ),
        BrokerOrderSnapshot.from_payload(
            {
                "id": "broker-msft",
                "client_order_id": "client-msft",
                "symbol": "MSFT",
                "side": "buy",
                "status": "accepted",
                "qty": 5,
                "filled_qty": 0,
                "type": "market",
                "submitted_at": "2026-03-22T01:00:00+00:00",
                "updated_at": "2026-03-22T01:02:00+00:00",
            }
        ),
    ]

    plan = build_recovery_plan(ledger_orders, broker_orders)

    assert plan.aligned_count == 1
    assert plan.orphan_count == 1
    assert plan.stale_count == 1
    assert plan.aligned_open_orders[0].order_id == "ledger-aapl"
    assert plan.orphan_broker_orders[0].order_id == "broker-msft"
    assert plan.stale_ledger_orders[0].order_id == "ledger-tsla"


def test_recover_open_orders_reconciles_broker_state_into_ledger(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    ledger.upsert_order(
        OrderRecord(
            order_id="ledger-aapl",
            run_id="run-1",
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
            submitted_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            broker_payload={},
        )
    )

    broker_order = BrokerOrderSnapshot.from_payload(
        {
            "id": "ledger-aapl",
            "client_order_id": "client-aapl",
            "symbol": "AAPL",
            "side": "buy",
            "status": "accepted",
            "qty": 10,
            "filled_qty": 0,
            "type": "market",
            "submitted_at": "2026-03-22T01:00:00+00:00",
            "updated_at": "2026-03-22T01:02:00+00:00",
        }
    )

    result = recover_open_orders(ledger, [broker_order])

    order = ledger.get_order("ledger-aapl")
    assert result.plan.aligned_count == 1
    assert result.plan.orphan_count == 0
    assert result.plan.stale_count == 0
    assert result.reconciliation.updated_orders == 1
    assert order is not None
    assert order.status == "accepted"
