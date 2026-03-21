from __future__ import annotations

from datetime import date, datetime, timezone

from stockmachine.monitoring.reports import (
    PaperRunFailure,
    build_paper_artifact_link,
    build_paper_run_manifest,
    build_paper_run_report,
)


def test_paper_run_report_serializes_counts_and_failures() -> None:
    manifest = build_paper_run_manifest(
        run_id="run-001",
        session_date=date(2026, 3, 21),
        strategy_name="hist_gbm",
        model_name="lightgbm_v1",
        dry_run=True,
        client_order_id_prefix="smk",
        generated_at_utc=datetime(2026, 3, 21, 1, 0, tzinfo=timezone.utc),
        data_snapshot={"market_data": "alpaca"},
        risk_policy={"max_order_notional": 500.0},
        execution_policy={"max_total_orders": 10},
        meta={"run_name": "demo"},
    )
    report = build_paper_run_report(
        session_date=date(2026, 3, 21),
        dry_run=True,
        stage="completed_with_warnings",
        counts={"orders": 3, "signals": 5},
        failures=(
            PaperRunFailure(
                stage="submit_orders",
                reason="missing_order_submitter",
                details={"dry_run": True},
            ),
        ),
        meta={"run_name": "demo"},
        run_id="run-001",
        manifest=manifest,
    )

    payload = report.to_dict()
    assert payload["run_id"] == "run-001"
    assert payload["session_date"] == "2026-03-21"
    assert payload["status"] == "failed"
    assert payload["counts"]["orders"] == 3
    assert payload["failures"][0]["stage"] == "submit_orders"
    assert payload["meta"]["run_name"] == "demo"
    assert payload["meta"]["run_manifest"]["model_name"] == "lightgbm_v1"
    manifest_record = manifest.to_record()
    assert manifest_record.strategy_name == "hist_gbm"


def test_build_paper_artifact_link_discovers_matching_artifact_dir(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    candidate = artifact_root / "paper-smoke-2026-03-22-hist_gbm"
    candidate.mkdir(parents=True)
    for file_name in ("backtest_summary.csv", "backtest_records.csv", "predictions.csv"):
        (candidate / file_name).write_text("dummy\n", encoding="utf-8")

    other = artifact_root / "paper-smoke-2026-03-21-hist_gbm"
    other.mkdir(parents=True)
    for file_name in ("backtest_summary.csv", "backtest_records.csv", "predictions.csv"):
        (other / file_name).write_text("dummy\n", encoding="utf-8")

    link = build_paper_artifact_link(
        artifact_root=artifact_root,
        run_name="paper-smoke",
        session_date=date(2026, 3, 22),
        model_name="hist_gbm",
    )

    assert link.artifact_dir == candidate
    assert link.exists is True
    assert link.source == "discovered"
    assert "backtest_summary" in link.files
    assert str(candidate) in link.candidates
    assert "matched_run_name" in link.notes


def test_build_paper_artifact_link_can_fall_back_to_partial_model_match(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    candidate = artifact_root / "us_equities_silver_chain_2025_hist_gbm_alpaca_adj"
    candidate.mkdir(parents=True)
    for file_name in ("backtest_summary.csv", "backtest_records.csv", "predictions.csv"):
        (candidate / file_name).write_text("dummy\n", encoding="utf-8")

    link = build_paper_artifact_link(
        artifact_root=artifact_root,
        model_name="hist_gbm",
        session_date=date(2026, 3, 22),
    )

    assert link.artifact_dir == candidate
    assert link.source == "discovered"
    assert "matched_model_name" in link.notes


def test_build_paper_artifact_link_uses_manifest_meta(tmp_path) -> None:
    artifact_dir = tmp_path / "resolved-artifact"
    artifact_dir.mkdir()
    for file_name in ("backtest_summary.csv", "backtest_records.csv", "predictions.csv"):
        (artifact_dir / file_name).write_text("dummy\n", encoding="utf-8")

    manifest = build_paper_run_manifest(
        run_id="run-002",
        session_date=date(2026, 3, 22),
        strategy_name="hist_gbm",
        model_name="lightgbm_v1",
        dry_run=True,
        meta={"artifact_dir": str(artifact_dir)},
    )

    link = build_paper_artifact_link(manifest=manifest)

    assert link.artifact_dir == artifact_dir
    assert link.source == "manifest_meta"
    assert link.exists is True
