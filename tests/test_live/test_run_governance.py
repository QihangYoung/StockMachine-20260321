from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from stockmachine.live import run_governance
from stockmachine.state import LocalLedger, RunManifestRecord, RunRecord


def _seed_completed_run(
    ledger: LocalLedger,
    *,
    run_id: str,
    session_date: date,
    dry_run: bool,
) -> None:
    ledger.record_run(
        RunRecord(
            run_id=run_id,
            strategy_name="paper-demo",
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
            strategy_name="paper-demo",
            model_name="hist_gbm",
            generated_at_utc=datetime(2026, 3, 22, 1, 5, tzinfo=timezone.utc),
            client_order_id_prefix="smk",
            dry_run=dry_run,
            data_snapshot={"market_data": "alpaca"},
            risk_policy={"max_order_notional": 500.0},
            execution_policy={"max_total_orders": 10},
            meta={"session_guard": {"resolution": "exact"}},
        )
    )


def test_governance_blocks_kill_switch(tmp_path, monkeypatch) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    _seed_completed_run(ledger, run_id="run-001", session_date=date(2026, 3, 22), dry_run=True)
    kill_switch = tmp_path / "paper_daily.kill"
    kill_switch.write_text('{"active": true, "reason": "maintenance"}', encoding="utf-8")

    monkeypatch.setattr(
        run_governance,
        "load_us_equities_dataset",
        lambda layout: {"daily_bar": pd.DataFrame({"session_date": ["2026-03-22"], "symbol": ["AAPL"]})},
    )

    result = run_governance.evaluate_daily_run_governance(
        run_governance.DailyRunGovernanceRequest(
            ledger_path=ledger.path,
            data_root=tmp_path / "data",
            session_date=date(2026, 3, 22),
            strategy_name="paper-demo",
            dry_run=True,
            kill_switch_path=kill_switch,
        )
    )

    assert result.allowed is False
    assert result.policy_allowed is False
    assert any(reason.startswith("kill_switch_active:") for reason in result.reasons)


def test_governance_blocks_duplicate_completed_run(tmp_path, monkeypatch) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    _seed_completed_run(ledger, run_id="run-002", session_date=date(2026, 3, 22), dry_run=True)

    monkeypatch.setattr(
        run_governance,
        "load_us_equities_dataset",
        lambda layout: {"daily_bar": pd.DataFrame({"session_date": ["2026-03-22"], "symbol": ["AAPL"]})},
    )

    result = run_governance.evaluate_daily_run_governance(
        run_governance.DailyRunGovernanceRequest(
            ledger_path=ledger.path,
            data_root=tmp_path / "data",
            session_date=date(2026, 3, 22),
            strategy_name="paper-demo",
            dry_run=True,
        )
    )

    assert result.allowed is False
    assert any(reason.startswith("session_guard:duplicate_completed_run") for reason in result.reasons)
    assert result.session_guard is not None
    assert "duplicate_run_ids" in result.session_guard["data_freshness_meta"]


def test_governance_allows_bootstrap_run_with_empty_ledger(tmp_path, monkeypatch) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()

    monkeypatch.setattr(
        run_governance,
        "load_us_equities_dataset",
        lambda layout: {"daily_bar": pd.DataFrame({"session_date": ["2026-03-22"], "symbol": ["AAPL"]})},
    )

    result = run_governance.evaluate_daily_run_governance(
        run_governance.DailyRunGovernanceRequest(
            ledger_path=ledger.path,
            data_root=tmp_path / "data",
            session_date=date(2026, 3, 22),
            strategy_name="paper-demo",
            dry_run=True,
        )
    )

    assert result.allowed is True
    assert result.policy_allowed is True
    assert result.reasons == ()
    assert result.data_freshness_meta["healthcheck_warnings"] == [
        "missing_latest_run",
        "missing_latest_manifest",
    ]


def test_governance_override_allows_blocked_run(tmp_path, monkeypatch) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()

    monkeypatch.setattr(
        run_governance,
        "load_us_equities_dataset",
        lambda layout: {"daily_bar": pd.DataFrame({"session_date": ["2026-03-22"], "symbol": ["AAPL"]})},
    )

    result = run_governance.evaluate_daily_run_governance(
        run_governance.DailyRunGovernanceRequest(
            ledger_path=ledger.path,
            data_root=tmp_path / "data",
            session_date=date(2026, 3, 22),
            strategy_name="paper-demo",
            dry_run=True,
            allow_unhealthy=True,
        )
    )

    assert result.allowed is True
    assert result.override_used is False
