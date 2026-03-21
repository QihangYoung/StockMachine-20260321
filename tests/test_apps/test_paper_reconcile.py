from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pandas as pd

from stockmachine.apps.paper_reconcile import main
from stockmachine.monitoring.reconciliation import load_expected_snapshot_from_artifact_dir
from stockmachine.monitoring.reports import build_paper_run_manifest
from stockmachine.state import LocalLedger, OrderDecisionRecord, OrderRecord


def _seed_cli_ledger(tmp_path, *, artifact_dir: str | None = None) -> LocalLedger:
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    ledger.initialize()
    ledger.record_run_manifest(
        build_paper_run_manifest(
            run_id="run-cli",
            session_date=date(2026, 3, 22),
            strategy_name="demo",
            model_name="hist_gbm",
            dry_run=False,
            meta={"artifact_dir": artifact_dir} if artifact_dir is not None else {},
        ).to_record()
    )
    ledger.record_order_decision(
        OrderDecisionRecord(
            decision_id="dec-cli-1",
            run_id="run-cli",
            session_date=date(2026, 3, 22),
            client_order_id="client-cli-1",
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
    ledger.upsert_order(
        OrderRecord(
            order_id="order-cli-1",
            run_id="run-cli",
            session_date=date(2026, 3, 22),
            client_order_id="client-cli-1",
            symbol="AAPL",
            side="buy",
            quantity=10,
            order_type="market",
            limit_price=None,
            status="filled",
            filled_quantity=10,
            avg_fill_price=100.2,
            submitted_at_utc=datetime(2026, 3, 22, 13, 1, tzinfo=timezone.utc),
            updated_at_utc=datetime(2026, 3, 22, 13, 2, tzinfo=timezone.utc),
            broker_payload={},
        )
    )
    return ledger


def test_paper_reconcile_cli_outputs_json_for_run_id(tmp_path, capsys) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    pd.DataFrame(
        [
            {
                "model": "hist_gbm",
                "sessions": 1,
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
                "positions": 1,
            }
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
            }
        ]
    ).to_csv(artifact_dir / "predictions.csv", index=False)

    ledger = _seed_cli_ledger(tmp_path)

    exit_code = main(["--ledger", str(ledger.path), "--run-id", "run-cli", "--artifact-dir", str(artifact_dir)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["run_id"] == "run-cli"
    assert output["found"] is True
    assert output["decision_count"] == 1
    assert output["filled_count"] == 1
    assert output["expected_snapshot"]["counts"]["decision_count"] == 1
    assert output["expected_snapshot"]["meta"]["selected_prediction_date"] == "2025-01-02"
    assert output["comparison"]["counts"]["decision_count"]["expected"] == 1
    assert output["comparison"]["counts"]["decision_count"]["actual"] == 1


def test_paper_reconcile_cli_latest_run_uses_latest_manifest(tmp_path, capsys) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    pd.DataFrame(
        [
            {
                "model": "hist_gbm",
                "sessions": 1,
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
                "positions": 1,
            }
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
            }
        ]
    ).to_csv(artifact_dir / "predictions.csv", index=False)

    ledger = _seed_cli_ledger(tmp_path, artifact_dir=str(artifact_dir))

    exit_code = main(["--ledger", str(ledger.path), "latest-run", "--artifact-dir", str(artifact_dir)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["run_id"] == "run-cli"
    assert output["manifest"]["strategy_name"] == "demo"
    assert output["expected_snapshot"]["meta"]["artifact_dir"] == str(artifact_dir)
    assert output["expected_snapshot"]["meta"]["selected_prediction_date"] == "2025-01-02"
