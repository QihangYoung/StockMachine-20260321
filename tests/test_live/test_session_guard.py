from __future__ import annotations

from datetime import date, datetime, timezone

from stockmachine.live import SessionGuard, SessionGuardRequest, evaluate_session_guard
from stockmachine.state import LocalLedger, RunManifestRecord, RunRecord


def _seed_completed_run(
    ledger: LocalLedger,
    *,
    run_id: str,
    strategy_name: str,
    session_date: date,
    dry_run: bool,
) -> None:
    ledger.record_run(
        RunRecord(
            run_id=run_id,
            strategy_name=strategy_name,
            market="us_equities",
            created_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            meta={"session_date": session_date.isoformat(), "dry_run": dry_run},
        )
    )
    ledger.record_run_manifest(
        RunManifestRecord(
            run_id=run_id,
            session_date=session_date,
            strategy_name=strategy_name,
            model_name="lightgbm_v1",
            generated_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            client_order_id_prefix="smk",
            dry_run=dry_run,
            data_snapshot={"source": "alpaca"},
            risk_policy={"max_order_notional": 500.0},
            execution_policy={"max_total_orders": 10},
            meta={"session_date": session_date.isoformat(), "dry_run": dry_run},
        )
    )
    ledger.finish_run(run_id, finished_at_utc=datetime(2026, 3, 22, 2, 0, tzinfo=timezone.utc))


def test_session_guard_blocks_duplicate_completed_run_same_mode(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    _seed_completed_run(
        ledger,
        run_id="run-001",
        strategy_name="hist_gbm",
        session_date=date(2026, 3, 22),
        dry_run=False,
    )

    result = SessionGuard(ledger).evaluate(
        SessionGuardRequest(
            ledger=ledger,
            strategy_name="hist_gbm",
            session_date=date(2026, 3, 22),
            dry_run=False,
            silver_session_dates=[date(2026, 3, 22)],
        )
    )

    assert result.allowed is False
    assert "duplicate_completed_run" in result.reasons
    assert result.duplicate_run_ids == ("run-001",)
    assert result.effective_session_date == date(2026, 3, 22)


def test_session_guard_fails_on_empty_silver_data(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()

    result = evaluate_session_guard(
        SessionGuardRequest(
            ledger=ledger,
            strategy_name="hist_gbm",
            session_date=date(2026, 3, 22),
            dry_run=True,
            silver_session_dates=(),
        )
    )

    assert result.allowed is False
    assert result.reasons == ("missing_silver_data",)
    assert result.effective_session_date is None
    assert result.data_freshness_meta["silver_available_session_count"] == 0


def test_session_guard_falls_back_to_previous_available_session(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()

    result = SessionGuard(ledger).evaluate(
        SessionGuardRequest(
            ledger=ledger,
            strategy_name="hist_gbm",
            session_date=date(2026, 3, 22),
            dry_run=True,
            silver_session_dates=[
                date(2026, 3, 18),
                date(2026, 3, 20),
            ],
        )
    )

    assert result.allowed is True
    assert result.effective_session_date == date(2026, 3, 20)
    assert result.data_freshness_meta["resolution"] == "fallback_previous_available_session"
    assert result.data_freshness_meta["session_lag_days"] == 2


def test_session_guard_separates_dry_run_and_execute_modes(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    _seed_completed_run(
        ledger,
        run_id="run-002",
        strategy_name="hist_gbm",
        session_date=date(2026, 3, 22),
        dry_run=True,
    )

    dry_result = SessionGuard(ledger).evaluate(
        SessionGuardRequest(
            ledger=ledger,
            strategy_name="hist_gbm",
            session_date=date(2026, 3, 22),
            dry_run=True,
            silver_session_dates=[date(2026, 3, 22)],
        )
    )
    execute_result = SessionGuard(ledger).evaluate(
        SessionGuardRequest(
            ledger=ledger,
            strategy_name="hist_gbm",
            session_date=date(2026, 3, 22),
            dry_run=False,
            silver_session_dates=[date(2026, 3, 22)],
        )
    )

    assert dry_result.allowed is False
    assert "duplicate_completed_run" in dry_result.reasons
    assert execute_result.allowed is True
    assert execute_result.reasons == ()
