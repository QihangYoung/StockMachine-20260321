from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from stockmachine.monitoring.reconciliation import (
    build_paper_reconciliation_summary,
    load_expected_snapshot_from_artifact_dir,
)
from stockmachine.monitoring.reports import build_paper_run_manifest
from stockmachine.state import FillAuditRecord, FillRecord, LocalLedger, OrderDecisionRecord, OrderRecord


def _seed_reconciliation_ledger(tmp_path):
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    manifest = build_paper_run_manifest(
        run_id="run-1",
        session_date=date(2026, 3, 22),
        strategy_name="demo",
        model_name="hist_gbm",
        dry_run=False,
    )
    ledger.record_run_manifest(manifest.to_record())
    ledger.record_order_decision(
        OrderDecisionRecord(
            decision_id="dec-1",
            run_id="run-1",
            session_date=date(2026, 3, 22),
            client_order_id="client-aapl",
            symbol="AAPL",
            side="buy",
            decision_type="rebalance",
            decision_price=100.0,
            estimated_notional=1000.0,
            approved=True,
            reason="signal",
            decision_at_utc=datetime(2026, 3, 22, 13, 0, tzinfo=timezone.utc),
        )
    )
    ledger.record_order_decision(
        OrderDecisionRecord(
            decision_id="dec-2",
            run_id="run-1",
            session_date=date(2026, 3, 22),
            client_order_id="client-msft",
            symbol="MSFT",
            side="buy",
            decision_type="rebalance",
            decision_price=200.0,
            estimated_notional=800.0,
            approved=True,
            reason="signal",
            decision_at_utc=datetime(2026, 3, 22, 13, 1, tzinfo=timezone.utc),
        )
    )
    ledger.record_order_decision(
        OrderDecisionRecord(
            decision_id="dec-3",
            run_id="run-1",
            session_date=date(2026, 3, 22),
            client_order_id="client-tsla",
            symbol="TSLA",
            side="sell",
            decision_type="risk",
            decision_price=None,
            estimated_notional=500.0,
            approved=False,
            reason="risk veto",
            decision_at_utc=datetime(2026, 3, 22, 13, 2, tzinfo=timezone.utc),
        )
    )
    ledger.upsert_order(
        OrderRecord(
            order_id="order-aapl",
            run_id="run-1",
            session_date=date(2026, 3, 22),
            client_order_id="client-aapl",
            symbol="AAPL",
            side="buy",
            quantity=10,
            order_type="market",
            limit_price=None,
            status="filled",
            filled_quantity=10,
            avg_fill_price=100.4,
            submitted_at_utc=datetime(2026, 3, 22, 13, 5, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 13, 10, tzinfo=timezone.utc),
            broker_payload={},
        )
    )
    ledger.upsert_order(
        OrderRecord(
            order_id="order-msft",
            run_id="run-1",
            session_date=date(2026, 3, 22),
            client_order_id="client-msft",
            symbol="MSFT",
            side="buy",
            quantity=4,
            order_type="market",
            limit_price=None,
            status="submitted",
            filled_quantity=0,
            avg_fill_price=None,
            submitted_at_utc=datetime(2026, 3, 22, 13, 6, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 13, 6, tzinfo=timezone.utc),
            broker_payload={},
        )
    )
    ledger.upsert_order(
        OrderRecord(
            order_id="order-tsla",
            run_id="run-1",
            session_date=date(2026, 3, 22),
            client_order_id="client-tsla",
            symbol="TSLA",
            side="sell",
            quantity=2,
            order_type="market",
            limit_price=None,
            status="rejected",
            filled_quantity=0,
            avg_fill_price=None,
            submitted_at_utc=datetime(2026, 3, 22, 13, 7, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 13, 7, tzinfo=timezone.utc),
            broker_payload={},
        )
    )
    ledger.record_fill(
        FillRecord(
            fill_id="fill-aapl-1",
            order_id="order-aapl",
            run_id="run-1",
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=100.4,
            filled_at_utc=datetime(2026, 3, 22, 13, 10, tzinfo=timezone.utc),
            broker_payload={},
        )
    )
    ledger.record_fill_audit(
        FillAuditRecord(
            audit_id="audit-aapl-1",
            order_id="order-aapl",
            run_id="run-1",
            session_date=date(2026, 3, 22),
            client_order_id="client-aapl",
            symbol="AAPL",
            side="buy",
            quantity=5,
            expected_price=100.0,
            price=100.2,
            slippage=2.0,
            fee=0.1,
            filled_at_utc=datetime(2026, 3, 22, 13, 10, tzinfo=timezone.utc),
        )
    )
    ledger.record_fill_audit(
        FillAuditRecord(
            audit_id="audit-aapl-2",
            order_id="order-aapl",
            run_id="run-1",
            session_date=date(2026, 3, 22),
            client_order_id="client-aapl",
            symbol="AAPL",
            side="buy",
            quantity=5,
            expected_price=100.0,
            price=100.6,
            slippage=6.0,
            fee=0.1,
            filled_at_utc=datetime(2026, 3, 22, 13, 10, tzinfo=timezone.utc),
        )
    )
    return ledger


def test_build_paper_reconciliation_summary_from_ledger(tmp_path) -> None:
    ledger = _seed_reconciliation_ledger(tmp_path)
    expected_snapshot = {
        "reference": "backtest-artifact",
        "counts": {"decision_count": 3, "submitted_count": 3},
        "symbol_status": {"AAPL": {"filled_count": 1}},
        "meta": {"source": "research"},
    }

    summary = build_paper_reconciliation_summary(
        ledger,
        run_id="run-1",
        expected_snapshot=expected_snapshot,
    )

    assert summary.found is True
    assert summary.run_id == "run-1"
    assert summary.decision_count == 3
    assert summary.approved_count == 2
    assert summary.submitted_count == 3
    assert summary.filled_count == 1
    assert summary.fill_rate == 1 / 3
    assert summary.open_order_count == 1
    assert summary.rejected_count == 1
    assert summary.slippage.count == 2
    assert summary.slippage.mean_bps == 4.0
    assert summary.slippage.median_bps == 4.0
    assert summary.slippage.min_bps == 2.0
    assert summary.slippage.max_bps == 6.0
    assert summary.symbol_status["AAPL"].filled_count == 1
    assert summary.symbol_status["MSFT"].open_order_count == 1
    assert summary.symbol_status["TSLA"].rejected_count == 1
    assert summary.expected_snapshot is not None
    assert summary.expected_snapshot.reference == "backtest-artifact"
    assert summary.manifest["strategy_name"] == "demo"
    assert summary.to_dict()["expected_snapshot"]["reference"] == "backtest-artifact"


def test_build_paper_reconciliation_summary_handles_missing_run(tmp_path) -> None:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()

    summary = build_paper_reconciliation_summary(ledger)

    assert summary.found is False
    assert summary.run_id is None
    assert summary.decision_count == 0
    assert summary.submitted_count == 0
    assert summary.symbol_status == {}
    assert summary.to_dict()["expected_snapshot"] == {}


def test_load_expected_snapshot_from_artifact_dir_and_compare_with_actual(tmp_path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    pd.DataFrame(
        [
            {
                "model": "hist_gbm",
                "sessions": 2,
                "total_return": 0.12,
                "annualized_return": 0.20,
                "annualized_volatility": 0.10,
                "sharpe": 1.2,
                "max_drawdown": -0.05,
                "benchmark_total_return": 0.08,
                "mean_turnover": 1.5,
                "mean_cost_bps": 12.0,
            }
        ]
    ).to_csv(artifact_dir / "backtest_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "signal_date": "2025-01-02",
                "entry_date": "2025-01-03",
                "exit_date": "2025-01-10",
                "gross_return": 0.01,
                "net_return": 0.009,
                "benchmark_return": 0.004,
                "turnover": 1.0,
                "cost_bps": 10.0,
                "positions": 2,
            },
            {
                "signal_date": "2025-01-10",
                "entry_date": "2025-01-13",
                "exit_date": "2025-01-20",
                "gross_return": 0.02,
                "net_return": 0.019,
                "benchmark_return": 0.005,
                "turnover": 1.0,
                "cost_bps": 10.0,
                "positions": 2,
            },
        ]
    ).to_csv(artifact_dir / "backtest_records.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": "2025-01-02",
                "symbol": "AAPL",
                "sector": "Tech",
                "industry": "Hardware",
                "close": 100.0,
                "vol_20": 0.02,
                "median_dollar_volume_20": 1_000_000.0,
                "target": 0.01,
                "future_return": 0.015,
                "benchmark_future_return": 0.005,
                "score": 0.9,
                "confidence": 1.0,
                "model": "hist_gbm",
            },
            {
                "date": "2025-01-02",
                "symbol": "MSFT",
                "sector": "Tech",
                "industry": "Software",
                "close": 200.0,
                "vol_20": 0.02,
                "median_dollar_volume_20": 1_000_000.0,
                "target": 0.01,
                "future_return": 0.015,
                "benchmark_future_return": 0.005,
                "score": 0.8,
                "confidence": 0.9,
                "model": "hist_gbm",
            },
            {
                "date": "2025-01-10",
                "symbol": "AAPL",
                "sector": "Tech",
                "industry": "Hardware",
                "close": 101.0,
                "vol_20": 0.02,
                "median_dollar_volume_20": 1_000_000.0,
                "target": 0.01,
                "future_return": 0.015,
                "benchmark_future_return": 0.005,
                "score": 0.95,
                "confidence": 1.0,
                "model": "hist_gbm",
            },
            {
                "date": "2025-01-10",
                "symbol": "NVDA",
                "sector": "Tech",
                "industry": "Semiconductors",
                "close": 300.0,
                "vol_20": 0.02,
                "median_dollar_volume_20": 1_000_000.0,
                "target": 0.01,
                "future_return": 0.015,
                "benchmark_future_return": 0.005,
                "score": 0.7,
                "confidence": 0.8,
                "model": "hist_gbm",
            },
        ]
    ).to_csv(artifact_dir / "predictions.csv", index=False)

    expected = load_expected_snapshot_from_artifact_dir(artifact_dir)
    assert expected.counts["decision_count"] == 2
    assert expected.counts["selected_rows"] == 2
    assert expected.meta["prediction_date_resolution"] == "latest_available"
    assert expected.meta["selected_prediction_date"] == "2025-01-10"
    assert expected.symbol_status["AAPL"]["submitted_count"] == 1
    assert "MSFT" not in expected.symbol_status
    assert expected.symbol_status["NVDA"]["submitted_count"] == 1

    fallback_expected = load_expected_snapshot_from_artifact_dir(
        artifact_dir,
        target_session_date="2025-01-05",
    )
    assert fallback_expected.meta["prediction_date_resolution"] == "fallback_latest_on_or_before_target"
    assert fallback_expected.meta["selected_prediction_date"] == "2025-01-02"
    assert fallback_expected.counts["decision_count"] == 2
    assert fallback_expected.symbol_status["AAPL"]["submitted_count"] == 1
    assert fallback_expected.symbol_status["MSFT"]["submitted_count"] == 1

    ledger = _seed_reconciliation_ledger(tmp_path)
    summary = build_paper_reconciliation_summary(ledger, run_id="run-1", expected_snapshot=expected)

    comparison = summary.to_dict()["comparison"]
    assert comparison["counts"]["decision_count"]["expected"] == 2
    assert comparison["counts"]["decision_count"]["actual"] == 3
    assert comparison["counts"]["decision_count"]["delta"] == 1
    assert comparison["symbol_status"]["AAPL"]["expected"]["submitted_count"] == 1
    assert comparison["symbol_status"]["AAPL"]["actual"]["submitted_count"] == 1
    assert comparison["symbol_status"]["AAPL"]["delta"]["submitted_count"] == 0
    assert any("artifact_reference" in note for note in summary.notes)
