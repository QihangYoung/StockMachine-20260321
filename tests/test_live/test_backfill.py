from __future__ import annotations

from datetime import date, datetime, timezone

from stockmachine.live import PollingOrderReconciler, backfill_order_statuses
from stockmachine.state import LocalLedger, OrderDecisionRecord, OrderRecord


class _FilledStatusProvider:
    def get_order(self, order_id: str):
        assert order_id == "alpaca-123"
        return {
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


def test_backfill_order_statuses_updates_order_and_creates_fill_audit(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    ledger.record_order_decision(
        OrderDecisionRecord(
            decision_id="decision-1",
            run_id="run-003",
            session_date=date(2026, 3, 22),
            client_order_id="client-123",
            symbol="AAPL",
            side="BUY",
            decision_type="risk_gate",
            decision_price=101.0,
            estimated_notional=1_010.0,
            approved=True,
            reason="approved",
            decision_at_utc=datetime(2026, 3, 22, 0, 59, tzinfo=timezone.utc),
            meta={},
        )
    )
    stale_order = OrderRecord(
        order_id="alpaca-123",
        run_id="run-003",
        session_date=date(2026, 3, 22),
        client_order_id="client-123",
        symbol="AAPL",
        side="buy",
        quantity=10,
        order_type="market",
        limit_price=None,
        status="pending_new",
        filled_quantity=0,
        avg_fill_price=None,
        submitted_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
        updated_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
        broker_payload={},
    )
    ledger.upsert_order(stale_order)

    result = backfill_order_statuses(
        ledger,
        _FilledStatusProvider(),
        candidate_orders=[stale_order],
        reconciler=PollingOrderReconciler(ledger),
    )

    order = ledger.get_order("alpaca-123")
    fills = ledger.list_fills(order_id="alpaca-123")
    fill_audits = ledger.list_fill_audits(order_id="alpaca-123")

    assert result.requested_orders == 1
    assert result.matched_snapshots == 1
    assert result.reconciliation.updated_orders == 1
    assert result.reconciliation.fill_events_created == 1
    assert result.fill_audits_created == 1
    assert order is not None
    assert order.status == "filled"
    assert order.filled_quantity == 10
    assert len(fills) == 1
    assert fills[0].price == 101.5
    assert len(fill_audits) == 1
    assert fill_audits[0].slippage == 0.5
