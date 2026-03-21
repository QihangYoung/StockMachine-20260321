from __future__ import annotations

from datetime import date, datetime, timezone

from stockmachine.monitoring.healthcheck import build_paper_daily_healthcheck
from stockmachine.state import LocalLedger, OrderRecord, RunManifestRecord, RunRecord


def _seed_run_and_manifest(
    ledger: LocalLedger,
    *,
    run_id: str,
    session_date: date,
    dry_run: bool,
    with_open_order: bool = False,
) -> None:
    ledger.record_run(
        RunRecord(
            run_id=run_id,
            strategy_name="hist_gbm",
            market="US",
            created_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            status="finished",
            meta={"session_date": session_date.isoformat(), "dry_run": dry_run},
        )
    )
    ledger.record_run_manifest(
        RunManifestRecord(
            run_id=run_id,
            session_date=session_date,
            strategy_name="hist_gbm",
            model_name="lightgbm_v1",
            generated_at_utc=datetime(2026, 3, 22, 1, 5, tzinfo=timezone.utc),
            client_order_id_prefix="smk",
            dry_run=dry_run,
            data_snapshot={"market_data": "alpaca"},
            risk_policy={"max_order_notional": 500.0},
            execution_policy={"max_total_orders": 10},
            meta={
                "session_guard": {
                    "resolution": "exact",
                    "requested_session_date": session_date.isoformat(),
                    "effective_session_date": session_date.isoformat(),
                    "session_lag_days": 0,
                    "silver_available_session_count": 1,
                    "silver_first_session_date": session_date.isoformat(),
                    "silver_last_session_date": session_date.isoformat(),
                    "exact_session_match": True,
                }
            },
        )
    )
    if with_open_order:
        ledger.upsert_order(
            OrderRecord(
                order_id=f"order-{run_id}",
                run_id=run_id,
                session_date=session_date,
                client_order_id=f"smk-{session_date:%Y%m%d}-AAPL-BUY-0001-run001",
                symbol="AAPL",
                side="buy",
                quantity=5,
                order_type="market",
                limit_price=None,
                status="accepted",
                filled_quantity=0,
                avg_fill_price=None,
                submitted_at_utc=datetime(2026, 3, 22, 1, 10, tzinfo=timezone.utc),
                updated_at_utc=datetime(2026, 3, 22, 1, 10, tzinfo=timezone.utc),
                broker_payload={},
            )
        )


def test_healthcheck_handles_empty_ledger(tmp_path) -> None:
    result = build_paper_daily_healthcheck(tmp_path / "ledger.sqlite3")

    assert result.healthy is False
    assert result.ledger_readable is True
    assert result.latest_run is None
    assert result.latest_manifest is None
    assert result.open_orders_total == 0
    assert "missing_latest_run" in result.reasons
    assert "missing_latest_manifest" in result.reasons


def test_healthcheck_reports_latest_run_and_manifest(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    _seed_run_and_manifest(ledger, run_id="run-001", session_date=date(2026, 3, 22), dry_run=True)

    result = build_paper_daily_healthcheck(ledger.path)

    assert result.healthy is True
    assert result.latest_run is not None
    assert result.latest_run["run_id"] == "run-001"
    assert result.latest_manifest is not None
    assert result.latest_manifest["model_name"] == "lightgbm_v1"
    assert result.data_freshness_meta["effective_session_date"] == "2026-03-22"
    assert result.data_freshness_meta["resolution"] == "exact"


def test_healthcheck_detects_lingering_open_orders(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    _seed_run_and_manifest(
        ledger,
        run_id="run-002",
        session_date=date(2026, 3, 22),
        dry_run=False,
        with_open_order=True,
    )

    result = build_paper_daily_healthcheck(ledger.path)

    assert result.healthy is False
    assert result.open_orders_total == 1
    assert result.open_orders_for_latest_run == 1
    assert "lingering_open_orders" in result.reasons
