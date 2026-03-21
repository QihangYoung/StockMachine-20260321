from __future__ import annotations

from datetime import date, datetime, timezone

from stockmachine.monitoring.digest import build_daily_summary_payload, build_operator_digest_payload, build_run_index_payload
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


def test_run_index_payload_orders_history_newest_first(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    _seed_digest_ledger(ledger)

    payload = build_run_index_payload(ledger, limit=5, reference_date=date(2026, 3, 22))

    assert payload["command"] == "run-index"
    assert payload["count"] == 2
    assert payload["entries"][0]["run_id"] == "run-002"
    assert payload["entries"][1]["run_id"] == "run-001"
    assert "data_stale" in payload["entries"][1]["alert_codes"]


def test_daily_summary_and_operator_digest_are_json_friendly(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    _seed_digest_ledger(ledger)

    summary = build_daily_summary_payload(ledger, session_date=date(2026, 3, 22), reference_date=date(2026, 3, 22))
    digest = build_operator_digest_payload(ledger, session_date=date(2026, 3, 22), limit=5, reference_date=date(2026, 3, 22))

    assert summary["command"] == "daily-summary"
    assert summary["session_date"] == "2026-03-22"
    assert summary["run_count"] == 1
    assert summary["run_ids"] == ["run-001"]
    assert summary["order_count"] == 2
    assert summary["open_order_count"] == 1
    assert summary["alerts"][0]["code"] == "data_stale"
    assert digest["command"] == "operator-digest"
    assert "run_index" in digest and "daily_summary" in digest and "health_trend" in digest
    assert "anomaly_summary" in digest
    assert {alert["code"] for alert in digest["alerts"]} >= {
        "data_stale",
        "duplicate_run_blocked",
        "broker_rejection",
        "open_orders_lingering",
    }
