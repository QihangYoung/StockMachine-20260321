from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from stockmachine.apps.run_us_equities_paper import (
    PaperRunConfig,
    build_alpaca_paper_runner,
    build_demo_runner,
    parse_session_date,
)
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.monitoring.healthcheck import build_paper_daily_healthcheck
from stockmachine.monitoring.digest import build_daily_summary_payload
from stockmachine.monitoring.reconciliation import (
    build_paper_reconciliation_summary,
    load_expected_snapshot_from_artifact_dir,
)
from stockmachine.monitoring.reports import build_paper_artifact_link
from stockmachine.live.run_governance import (
    DEFAULT_KILL_SWITCH_PATH,
    DailyRunGovernanceRequest,
    DailyRunGovernanceResult,
    evaluate_daily_run_governance,
)
from stockmachine.research.us_equities_baseline import OverlayConfig
from stockmachine.state import LocalLedger

PaperDailyPreflightResult = DailyRunGovernanceResult


@dataclass(slots=True, frozen=True)
class PaperDailyRunPayload:
    """JSON-ready result from one daily run invocation."""

    command: str
    ok: bool
    result: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "ok": self.ok,
            "result": dict(self.result),
        }


@dataclass(slots=True, frozen=True)
class PaperDailyOperationPayload:
    """Scheduler-friendly JSON payload for daily run and healthcheck outputs."""

    command: str
    ok: bool
    summary: Mapping[str, Any]
    preflight: Mapping[str, Any] | None = None
    run: Mapping[str, Any] | None = None
    post_run: Mapping[str, Any] | None = None
    error: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "command": self.command,
            "ok": self.ok,
            "summary": dict(self.summary),
        }
        if self.preflight is not None:
            payload["preflight"] = dict(self.preflight)
        if self.run is not None:
            payload["run"] = dict(self.run)
        if self.post_run is not None:
            payload["post_run"] = dict(self.post_run)
        if self.error is not None:
            payload["error"] = dict(self.error)
        return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scheduler-friendly daily operations shell for the paper demo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the daily paper workflow.")
    run_parser.add_argument("--session-date", default=date.today().isoformat())
    run_parser.add_argument("--universe", nargs="*", default=[])
    run_parser.add_argument("--run-name", default="paper-demo")
    run_parser.add_argument("--execute", action="store_true")
    run_parser.add_argument("--demo-mode", action="store_true")
    run_parser.add_argument("--model", default="hist_gbm")
    run_parser.add_argument("--top-k", type=int, default=10)
    run_parser.add_argument("--horizon", type=int, default=5)
    run_parser.add_argument("--data-root", default="data")
    run_parser.add_argument("--ledger-path", default="artifacts/paper_demo/paper_ledger.sqlite3")
    run_parser.add_argument("--min-close", type=float, default=10.0)
    run_parser.add_argument("--min-median-dollar-volume-20", type=float, default=50_000_000.0)
    run_parser.add_argument("--max-vol-20", type=float, default=0.04)
    run_parser.add_argument("--max-positions-per-sector", type=int, default=2)
    run_parser.add_argument("--disable-sector-neutral", action="store_true")
    run_parser.add_argument("--require-market-open", action="store_true")
    run_parser.add_argument("--min-buying-power-buffer", type=float, default=0.0)
    run_parser.add_argument("--max-order-notional", type=float, default=None)
    run_parser.add_argument("--max-total-notional", type=float, default=None)
    run_parser.add_argument("--max-total-orders", type=int, default=None)
    run_parser.add_argument("--execution-equity-cap", type=float, default=None)
    run_parser.add_argument("--post-submit-poll-seconds", type=float, default=15.0)
    run_parser.add_argument("--post-submit-poll-interval-seconds", type=float, default=2.0)
    run_parser.add_argument("--allow-unhealthy", action="store_true", help="Override preflight blockers and run anyway.")
    run_parser.add_argument("--kill-switch-path", default=str(DEFAULT_KILL_SWITCH_PATH))
    run_parser.add_argument("--artifact-root", default="artifacts")
    run_parser.add_argument("--artifact-dir", default=None, help="Optional research/backtest artifact dir for post-run reconciliation.")

    health_parser = subparsers.add_parser("healthcheck", help="Inspect ledger health for daily operations.")
    health_parser.add_argument("--ledger-path", default="artifacts/paper_demo/paper_ledger.sqlite3")

    return parser


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    session_date = parse_session_date(args.session_date)
    artifact_link = build_paper_artifact_link(
        artifact_dir=getattr(args, "artifact_dir", None),
        artifact_root=getattr(args, "artifact_root", "artifacts"),
        model_name=getattr(args, "model", None),
        session_date=session_date,
    )
    resolved_artifact_dir = str(artifact_link.artifact_dir) if artifact_link.artifact_dir is not None else None
    preflight = build_paper_daily_preflight(
        session_date=session_date,
        ledger_path=args.ledger_path,
        data_root=args.data_root,
        run_name=args.run_name,
        dry_run=not args.execute,
        universe=tuple(args.universe),
        model_name=args.model,
        kill_switch_path=getattr(args, "kill_switch_path", DEFAULT_KILL_SWITCH_PATH),
        override_unhealthy=getattr(args, "allow_unhealthy", False),
    )
    if not preflight.allowed and not args.allow_unhealthy:
        payload = PaperDailyOperationPayload(
            command="run",
            ok=False,
            summary=summarize_paper_daily_result(
                session_date=session_date,
                run_name=args.run_name,
                preflight=preflight,
                report=None,
                override_allowed=args.allow_unhealthy,
                stage="preflight_blocked",
            ),
            preflight=preflight.to_dict(),
        )
        return payload.to_dict()

    runner = None
    try:
        if getattr(args, "demo_mode", False):
            runner, config = build_demo_runner(args.universe)
        else:
            overlay_config = OverlayConfig(
                min_close=args.min_close,
                min_median_dollar_volume_20=args.min_median_dollar_volume_20,
                max_vol_20=args.max_vol_20,
                max_positions_per_sector=args.max_positions_per_sector,
                sector_neutral=not args.disable_sector_neutral,
            )
            runner, config = build_alpaca_paper_runner(
                universe=args.universe,
                session_date=session_date,
                model_name=args.model,
                top_k=args.top_k,
                horizon=args.horizon,
                overlay_config=overlay_config,
                layout=StorageLayout(root=Path(args.data_root)),
                ledger_path=args.ledger_path,
                require_market_open=args.require_market_open,
                min_buying_power_buffer=args.min_buying_power_buffer,
                max_order_notional=args.max_order_notional,
                max_total_notional=args.max_total_notional,
                max_total_orders=args.max_total_orders,
            )

        config = replace(
            config,
            session_date=session_date,
            dry_run=not args.execute,
            universe=tuple(args.universe),
            run_name=args.run_name,
            artifact_dir=resolved_artifact_dir,
            execution_equity_cap=args.execution_equity_cap,
            post_submit_poll_seconds=args.post_submit_poll_seconds,
            post_submit_poll_interval_seconds=args.post_submit_poll_interval_seconds,
        )
        report = runner.run(config)
        post_run_payload = _safe_build_post_run_payload(
            ledger_path=args.ledger_path,
            run_id=report.run_id,
            session_date=session_date,
            artifact_dir=resolved_artifact_dir,
        )
        payload = PaperDailyOperationPayload(
            command="run",
            ok=True,
            summary=summarize_paper_daily_result(
                session_date=session_date,
                run_name=args.run_name,
                preflight=preflight,
                report=report,
                override_allowed=args.allow_unhealthy,
                stage="completed",
            ),
            preflight=preflight.to_dict(),
            run={"report": report.to_dict()},
            post_run=post_run_payload,
        )
        if resolved_artifact_dir is not None:
            payload.summary["artifact_dir"] = resolved_artifact_dir
        return payload.to_dict()
    except Exception as exc:
        payload = PaperDailyOperationPayload(
            command="run",
            ok=False,
            summary=summarize_paper_daily_result(
                session_date=session_date,
                run_name=args.run_name,
                preflight=preflight,
                report=None,
                override_allowed=args.allow_unhealthy,
                stage="failed",
            ),
            preflight=preflight.to_dict(),
            error={
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        return payload.to_dict()
    finally:
        if runner is not None:
            runner.close()


def healthcheck_command(args: argparse.Namespace) -> dict[str, Any]:
    result = build_paper_daily_healthcheck(args.ledger_path)
    result_payload = result.to_dict()
    payload = PaperDailyOperationPayload(
        command="healthcheck",
        ok=result.healthy,
        summary={
            "ledger_path": result_payload.get("ledger_path"),
            "healthy": result_payload.get("healthy"),
            "reasons": list(result_payload.get("reasons", [])),
        },
        preflight=result_payload,
    )
    return payload.to_dict()


def build_paper_daily_preflight(
    *,
    session_date: date,
    ledger_path: str | Path,
    data_root: str | Path,
    run_name: str,
    dry_run: bool,
    universe: Sequence[str] = (),
    model_name: str = "hist_gbm",
    kill_switch_path: str | Path | None = None,
    override_unhealthy: bool = False,
) -> PaperDailyPreflightResult:
    return evaluate_daily_run_governance(
        DailyRunGovernanceRequest(
            ledger_path=ledger_path,
            data_root=data_root,
            session_date=session_date,
            strategy_name=run_name,
            dry_run=dry_run,
            universe=universe,
            model_name=model_name,
            kill_switch_path=kill_switch_path,
            allow_unhealthy=override_unhealthy,
        )
    )


def summarize_paper_daily_result(
    *,
    session_date: date,
    run_name: str,
    preflight: PaperDailyPreflightResult,
    report: Mapping[str, Any] | None,
    override_allowed: bool,
    stage: str,
) -> dict[str, Any]:
    decision = "blocked_preflight"
    if report is not None:
        decision = "executed_with_override" if (override_allowed and not preflight.policy_allowed) else "executed"
    elif stage == "failed":
        decision = "failed"
    report_status = None
    if isinstance(report, Mapping):
        report_status = report.get("status")
    elif report is not None:
        report_status = getattr(report, "status", None)
    return {
        "session_date": session_date.isoformat(),
        "run_name": run_name,
        "stage": stage,
        "decision": decision,
        "override_allowed": override_allowed,
        "preflight_allowed": preflight.policy_allowed,
        "preflight_reasons": list(preflight.reasons),
        "effective_session_date": preflight.effective_session_date.isoformat() if preflight.effective_session_date else None,
        "data_freshness_meta": dict(preflight.data_freshness_meta),
        "report_status": report_status,
    }


def _safe_build_post_run_payload(
    *,
    ledger_path: str | Path,
    run_id: str,
    session_date: date,
    artifact_dir: str | None = None,
) -> dict[str, Any]:
    try:
        return _build_post_run_payload(
            ledger_path=ledger_path,
            run_id=run_id,
            session_date=session_date,
            artifact_dir=artifact_dir,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


def _build_post_run_payload(
    *,
    ledger_path: str | Path,
    run_id: str,
    session_date: date,
    artifact_dir: str | None = None,
) -> dict[str, Any]:
    with LocalLedger(ledger_path) as ledger:
        ledger.initialize()
        expected_snapshot = None
        if artifact_dir:
            expected_snapshot = load_expected_snapshot_from_artifact_dir(
                artifact_dir,
                target_session_date=session_date,
            )
        reconciliation = build_paper_reconciliation_summary(
            ledger,
            run_id=run_id,
            expected_snapshot=expected_snapshot,
        ).to_dict()
        daily_summary = build_daily_summary_payload(
            ledger,
            session_date=session_date,
        )
    return {
        "ok": True,
        "reconciliation": _compact_reconciliation_payload(reconciliation),
        "daily_summary": _compact_daily_summary_payload(daily_summary),
    }


def _compact_reconciliation_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_snapshot = payload.get("expected_snapshot", {})
    expected_meta = expected_snapshot.get("meta", {}) if isinstance(expected_snapshot, Mapping) else {}
    comparison = payload.get("comparison", {})
    comparison_counts = comparison.get("counts", {}) if isinstance(comparison, Mapping) else {}
    count_keys = ("decision_count", "submitted_count", "filled_count")
    return {
        "run_id": payload.get("run_id"),
        "found": payload.get("found"),
        "dry_run": payload.get("dry_run"),
        "decision_count": payload.get("decision_count"),
        "approved_count": payload.get("approved_count"),
        "submitted_count": payload.get("submitted_count"),
        "filled_count": payload.get("filled_count"),
        "open_order_count": payload.get("open_order_count"),
        "notes": list(payload.get("notes", [])),
        "expected_reference": expected_snapshot.get("reference") if isinstance(expected_snapshot, Mapping) else None,
        "selected_prediction_date": expected_meta.get("selected_prediction_date"),
        "comparison_counts": {
            key: dict(value)
            for key, value in comparison_counts.items()
            if key in count_keys and isinstance(value, Mapping)
        },
    }


def _compact_daily_summary_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    alerts = payload.get("alerts", [])
    return {
        "session_date": payload.get("session_date"),
        "run_count": payload.get("run_count"),
        "status_counts": dict(payload.get("status_counts", {})),
        "order_count": payload.get("order_count"),
        "open_order_count": payload.get("open_order_count"),
        "fill_count": payload.get("fill_count"),
        "alert_codes": [
            alert.get("code")
            for alert in alerts
            if isinstance(alert, Mapping) and alert.get("code") is not None
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        payload = run_command(args)
    elif args.command == "healthcheck":
        payload = healthcheck_command(args)
    else:  # pragma: no cover - argparse enforces command selection
        parser.error(f"Unsupported command: {args.command}")
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
