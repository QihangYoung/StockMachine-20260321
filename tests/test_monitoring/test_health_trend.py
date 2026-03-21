from __future__ import annotations

from datetime import date, datetime, timezone

from stockmachine.monitoring.health_trend import build_anomaly_summary_payload, build_health_trend_payload
from stockmachine.state import LocalLedger, OrderRecord, RunManifestRecord, RunRecord


def _seed_health_ledger(ledger: LocalLedger) -> None:
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
    ledger.record_run(
        RunRecord(
            run_id="run-003",
            strategy_name="hist_gbm",
            market="US",
            created_at_utc=datetime(2026, 3, 23, 1, 0, tzinfo=timezone.utc),
            status="finished",
            meta={},
        )
    )
    ledger.record_run_manifest(
        RunManifestRecord(
            run_id="run-003",
            session_date=date(2026, 3, 23),
            strategy_name="hist_gbm",
            model_name="lightgbm_v1",
            generated_at_utc=datetime(2026, 3, 23, 1, 0, tzinfo=timezone.utc),
            client_order_id_prefix="smk",
            dry_run=True,
            data_snapshot={"snapshot_age_days": 0},
            risk_policy={"max_order_notional": 500.0},
            execution_policy={"max_total_orders": 10},
            meta={"source": "tests"},
        )
    )


def test_health_trend_payload_summarizes_recent_anomalies_and_sessions(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    _seed_health_ledger(ledger)

    payload = build_health_trend_payload(
        ledger,
        limit_runs=5,
        limit_sessions=3,
        reference_date=date(2026, 3, 23),
    )

    assert payload["command"] == "health-trend"
    assert payload["window"]["run_limit"] == 5
    assert payload["trend"]["status_counts"]["finished"] == 2
    assert payload["trend"]["alert_counts"]["data_stale"] == 1
    assert payload["trend"]["open_order_run_count"] == 1
    assert payload["sessions"][0]["session_date"] == "2026-03-23"
    assert payload["anomaly_summary"]["count"] >= 1
    assert any(point["key"] == "latest_run_alert_count" for point in payload["trend"]["recent_trend_points"])
    assert payload["alerts"][0]["code"] in {"data_stale", "duplicate_run_blocked", "broker_rejection", "open_orders_lingering"}


def test_anomaly_summary_payload_is_json_friendly(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    _seed_health_ledger(ledger)

    payload = build_anomaly_summary_payload(
        ledger,
        limit_runs=5,
        reference_date=date(2026, 3, 23),
    )

    assert payload["command"] == "anomaly-summary"
    assert payload["count"] >= 1
    assert any(item["run_id"] == "run-001" for item in payload["anomalies"])
