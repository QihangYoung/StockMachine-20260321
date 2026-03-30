from __future__ import annotations

from argparse import Namespace
from datetime import date, datetime, timezone

from stockmachine.apps import paper_backfill
from stockmachine.state import LocalLedger, OrderDecisionRecord, OrderRecord


class _FilledBroker:
    def get_order(self, order_id: str):
        assert order_id == "broker-order-1"
        return {
            "id": "broker-order-1",
            "client_order_id": "client-order-1",
            "symbol": "AAPL",
            "side": "buy",
            "status": "filled",
            "qty": 5,
            "filled_qty": 5,
            "type": "market",
            "submitted_at": "2026-03-22T01:00:00+00:00",
            "updated_at": "2026-03-22T01:05:00+00:00",
            "avg_fill_price": 101.25,
        }


def test_build_backfill_payload_updates_latest_run(monkeypatch, tmp_path) -> None:
    ledger_path = tmp_path / "paper-ledger.sqlite3"
    ledger = LocalLedger(ledger_path)
    ledger.initialize()
    ledger.record_order_decision(
        OrderDecisionRecord(
            decision_id="decision-1",
            run_id="run-1",
            session_date=date(2026, 3, 22),
            client_order_id="client-order-1",
            symbol="AAPL",
            side="BUY",
            decision_type="risk_gate",
            decision_price=100.0,
            estimated_notional=500.0,
            approved=True,
            reason="approved",
            decision_at_utc=datetime(2026, 3, 22, 0, 59, tzinfo=timezone.utc),
            meta={},
        )
    )
    ledger.upsert_order(
        OrderRecord(
            order_id="broker-order-1",
            run_id="run-1",
            session_date=date(2026, 3, 22),
            client_order_id="client-order-1",
            symbol="AAPL",
            side="buy",
            quantity=5,
            order_type="market",
            limit_price=None,
            status="pending_new",
            filled_quantity=0,
            avg_fill_price=None,
            submitted_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            broker_payload={},
        )
    )
    ledger.close()

    monkeypatch.setattr(
        paper_backfill,
        "AlpacaTradingAdapter",
        type("_BrokerFactory", (), {"from_env": staticmethod(lambda: _FilledBroker())}),
    )

    payload = paper_backfill.build_backfill_payload(
        Namespace(
            ledger_path=str(ledger_path),
            run_id=None,
            command="latest-run",
        )
    )

    reloaded = LocalLedger(ledger_path)
    reloaded.initialize()
    try:
        order = reloaded.get_order("broker-order-1")
        fills = reloaded.list_fills(run_id="run-1")
    finally:
        reloaded.close()

    assert payload["ok"] is True
    assert payload["run_id"] == "run-1"
    assert payload["summary"]["fills_created"] == 1
    assert order is not None
    assert order.status == "filled"
    assert len(fills) == 1
