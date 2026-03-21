from __future__ import annotations

from datetime import date, datetime, timezone

from stockmachine.live import PollingOrderReconciler
from stockmachine.state import LocalLedger, OrderRecord


def test_polling_reconciler_updates_order_state_and_infers_fill(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    ledger.upsert_order(
        OrderRecord(
            order_id="alpaca-123",
            run_id="run-003",
            session_date=date(2026, 3, 22),
            client_order_id="client-123",
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
            broker_payload={"source": "runner"},
        )
    )

    reconciler = PollingOrderReconciler(ledger)
    result = reconciler.reconcile_orders(
        [
            {
                "id": "alpaca-123",
                "client_order_id": "client-123",
                "symbol": "AAPL",
                "side": "buy",
                "status": "filled",
                "qty": 10,
                "filled_qty": 10,
                "type": "market",
                "submitted_at": "2026-03-22T01:00:00+00:00",
                "updated_at": "2026-03-22T01:05:00+00:00",
                "avg_fill_price": 101.5,
            }
        ]
    )

    order = ledger.get_order("alpaca-123")
    fills = ledger.list_fills(order_id="alpaca-123")

    assert result.created_orders == 0
    assert result.updated_orders == 1
    assert result.fill_events_created == 1
    assert result.changes[0].fill_delta == 10
    assert order is not None
    assert order.status == "filled"
    assert order.filled_quantity == 10
    assert len(fills) == 1
    assert fills[0].quantity == 10
    assert fills[0].price == 101.5


def test_polling_reconciler_creates_missing_order_from_broker_payload(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()

    reconciler = PollingOrderReconciler(ledger)
    result = reconciler.reconcile_orders(
        [
            {
                "id": "alpaca-456",
                "client_order_id": "client-456",
                "symbol": "MSFT",
                "side": "buy",
                "status": "accepted",
                "qty": 4,
                "filled_qty": 0,
                "type": "market",
                "submitted_at": "2026-03-22T01:00:00+00:00",
                "updated_at": "2026-03-22T01:01:00+00:00",
            }
        ]
    )

    order = ledger.get_order("alpaca-456")

    assert result.created_orders == 1
    assert result.updated_orders == 0
    assert order is not None
    assert order.symbol == "MSFT"
    assert order.status == "accepted"
