from __future__ import annotations

from datetime import date, datetime, timezone

from stockmachine.monitoring import build_operator_alerts
from stockmachine.monitoring.reports import PaperRunFailure, build_paper_run_manifest, build_paper_run_report
from stockmachine.state import LocalLedger, OrderRecord, RunRecord


def test_build_operator_alerts_covers_core_operator_patterns(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
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
    manifest = build_paper_run_manifest(
        run_id="run-001",
        session_date=date(2026, 3, 22),
        strategy_name="hist_gbm",
        model_name="lightgbm_v1",
        dry_run=False,
        generated_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
        data_snapshot={"snapshot_age_days": 3},
        risk_policy={"max_order_notional": 500.0},
        execution_policy={"max_total_orders": 10},
        meta={"source": "tests"},
    )
    report = build_paper_run_report(
        session_date=date(2026, 3, 22),
        dry_run=False,
        stage="completed_with_warnings",
        counts={"orders": 2},
        failures=(
            PaperRunFailure(
                stage="duplicate_run_guard",
                reason="duplicate_run_blocked",
                details={"run_id": "run-001"},
            ),
        ),
        meta={"run_manifest": manifest.to_dict()},
        run_id="run-001",
        manifest=manifest,
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

    alerts = build_operator_alerts(
        report=report,
        ledger=ledger,
        run_id="run-001",
        reference_date=date(2026, 3, 22),
        stale_after_days=2,
    )

    codes = {alert.code for alert in alerts}
    assert codes == {
        "data_stale",
        "duplicate_run_blocked",
        "broker_rejection",
        "open_orders_lingering",
    }
    assert any(alert.code == "broker_rejection" and alert.severity == "critical" for alert in alerts)
    assert any(alert.code == "open_orders_lingering" and alert.details["open_order_count"] == 1 for alert in alerts)
