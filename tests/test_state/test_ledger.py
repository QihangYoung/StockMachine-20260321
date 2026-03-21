from __future__ import annotations

from datetime import date, datetime, timezone

from stockmachine.state import (
    EquitySnapshotRecord,
    FillAuditRecord,
    FillRecord,
    LocalLedger,
    OrderDecisionRecord,
    OrderRecord,
    RunManifestRecord,
    RunRecord,
    SignalRecord,
    TargetRecord,
)


def test_local_ledger_persists_core_events(tmp_path) -> None:
    ledger_path = tmp_path / "paper-ledger.sqlite3"

    with LocalLedger(ledger_path) as ledger:
        ledger.initialize()
        ledger.record_run(
            RunRecord(
                run_id="run-001",
                strategy_name="hist_gbm",
                market="us_equities",
                created_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
                meta={"mode": "paper"},
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
                data_snapshot={"market_data": "alpaca"},
                risk_policy={"max_order_notional": 500.0},
                execution_policy={"max_total_orders": 10},
                meta={"source": "paper-runner"},
            )
        )
        ledger.record_order_decision(
            OrderDecisionRecord(
                decision_id="decision-001",
                run_id="run-001",
                session_date=date(2026, 3, 22),
                client_order_id="smk-20260322-AAPL-BUY-0001-run001",
                symbol="AAPL",
                side="BUY",
                decision_type="submit",
                decision_price=100.0,
                estimated_notional=1000.0,
                approved=True,
                reason="passed_risk_gate",
                decision_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
                slippage_bps=3.5,
                fee_estimate=0.35,
                meta={"rule": "paper"},
            )
        )
        ledger.append_signal(
            SignalRecord(
                run_id="run-001",
                session_date=date(2026, 3, 22),
                symbol="AAPL",
                side="BUY",
                score=0.91,
                confidence=0.82,
                horizon_bars=5,
                timestamp_utc=datetime(2026, 3, 22, 1, 1, tzinfo=timezone.utc),
                meta={"sector": "Technology"},
            )
        )
        ledger.append_target(
            TargetRecord(
                run_id="run-001",
                session_date=date(2026, 3, 22),
                symbol="AAPL",
                target_weight=0.1,
                max_weight=0.1,
                reason="risk_aware_top_k",
                timestamp_utc=datetime(2026, 3, 22, 1, 2, tzinfo=timezone.utc),
                meta={"adjusted_score": 0.77},
            )
        )
        ledger.upsert_order(
            OrderRecord(
                order_id="order-001",
                run_id="run-001",
                session_date=date(2026, 3, 22),
                client_order_id="client-001",
                symbol="AAPL",
                side="buy",
                quantity=10,
                order_type="market",
                limit_price=None,
                status="submitted",
                filled_quantity=0,
                avg_fill_price=None,
                submitted_at_utc=datetime(2026, 3, 22, 1, 3, tzinfo=timezone.utc),
                updated_at_utc=datetime(2026, 3, 22, 1, 3, tzinfo=timezone.utc),
                broker_payload={"source": "runner"},
            )
        )
        ledger.record_fill(
            FillRecord(
                fill_id="fill-001",
                order_id="order-001",
                run_id="run-001",
                symbol="AAPL",
                side="buy",
                quantity=10,
                price=101.25,
                filled_at_utc=datetime(2026, 3, 22, 1, 4, tzinfo=timezone.utc),
                expected_price=100.0,
                slippage=1.25,
                fee=0.35,
                client_order_id="client-001",
                broker_payload={"source": "broker"},
            )
        )
        ledger.record_fill_audit(
            FillAuditRecord(
                audit_id="audit-001",
                order_id="order-001",
                run_id="run-001",
                session_date=date(2026, 3, 22),
                client_order_id="client-001",
                symbol="AAPL",
                side="buy",
                quantity=10,
                expected_price=100.0,
                price=101.25,
                slippage=1.25,
                fee=0.35,
                filled_at_utc=datetime(2026, 3, 22, 1, 4, tzinfo=timezone.utc),
                meta={"source": "broker"},
            )
        )
        ledger.record_equity_snapshot(
            EquitySnapshotRecord(
                run_id="run-001",
                session_date=date(2026, 3, 22),
                timestamp_utc=datetime(2026, 3, 22, 1, 5, tzinfo=timezone.utc),
                cash=90_000.0,
                equity=100_500.0,
                gross_exposure=10_500.0,
                payload={"source": "broker"},
            )
        )

    with LocalLedger(ledger_path) as ledger:
        ledger.initialize()
        run = ledger.get_run("run-001")
        manifest = ledger.get_run_manifest("run-001")
        order = ledger.get_order("order-001")
        decisions = ledger.list_order_decisions(run_id="run-001")
        fills = ledger.list_fills(order_id="order-001")
        fill_audits = ledger.list_fill_audits(order_id="order-001")
        equity = ledger.list_equity_snapshots(run_id="run-001")

        assert run is not None
        assert run.strategy_name == "hist_gbm"
        assert manifest is not None
        assert manifest.model_name == "lightgbm_v1"
        assert len(decisions) == 1
        assert decisions[0].client_order_id.startswith("smk-")
        assert order is not None
        assert order.client_order_id == "client-001"
        assert len(fills) == 1
        assert fills[0].price == 101.25
        assert len(fill_audits) == 1
        assert fill_audits[0].slippage == 1.25
        assert len(equity) == 1
        assert equity[0].equity == 100_500.0


def test_local_ledger_lists_open_orders(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    ledger.upsert_order(
        OrderRecord(
            order_id="order-open",
            run_id="run-002",
            session_date=date(2026, 3, 22),
            client_order_id="client-open",
            symbol="MSFT",
            side="buy",
            quantity=5,
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
            order_id="order-done",
            run_id="run-002",
            session_date=date(2026, 3, 22),
            client_order_id="client-done",
            symbol="SPY",
            side="buy",
            quantity=3,
            order_type="market",
            limit_price=None,
            status="filled",
            filled_quantity=3,
            avg_fill_price=500.0,
            submitted_at_utc=datetime(2026, 3, 22, 1, 0, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 1, 1, tzinfo=timezone.utc),
            broker_payload={},
        )
    )

    open_orders = ledger.list_open_orders(run_id="run-002")
    assert len(open_orders) == 1
    assert open_orders[0].symbol == "MSFT"
