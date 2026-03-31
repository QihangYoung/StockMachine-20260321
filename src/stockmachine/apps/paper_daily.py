from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from stockmachine.alpha import list_alpha_expert_names
from stockmachine.apps.paper_profiles import load_strategy_profile
from stockmachine.apps.run_us_equities_paper import (
    PaperRunConfig,
    build_alpaca_paper_runner,
    build_demo_runner,
    parse_session_date,
)
from stockmachine.data.loaders import load_us_equities_dataset
from stockmachine.ingestion.jobs import collect_research_seed
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
    silver_refresh: Mapping[str, Any] | None = None
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
        if self.silver_refresh is not None:
            payload["silver_refresh"] = dict(self.silver_refresh)
        if self.preflight is not None:
            payload["preflight"] = dict(self.preflight)
        if self.run is not None:
            payload["run"] = dict(self.run)
        if self.post_run is not None:
            payload["post_run"] = dict(self.post_run)
        if self.error is not None:
            payload["error"] = dict(self.error)
        return payload


def build_arg_parser(*, run_defaults: Mapping[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scheduler-friendly daily operations shell for the paper demo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the daily paper workflow.")
    run_parser.add_argument("--strategy-profile", default=None, help="Built-in strategy profile name or JSON path.")
    run_parser.add_argument("--session-date", default=date.today().isoformat())
    run_parser.add_argument("--universe", nargs="*", default=[])
    run_parser.add_argument("--run-name", default="paper-demo")
    run_parser.add_argument("--execute", action="store_true")
    run_parser.add_argument("--demo-mode", action="store_true")
    run_parser.add_argument("--model", default="hist_gbm", choices=list_alpha_expert_names())
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
    run_parser.add_argument(
        "--skip-silver-refresh",
        action="store_true",
        help="Skip the incremental silver refresh step before preflight and execution.",
    )
    run_parser.add_argument(
        "--silver-refresh-feed",
        default="iex",
        help="Market-data feed used for incremental silver refresh.",
    )
    run_parser.add_argument(
        "--silver-refresh-adjustment",
        default="raw",
        help="Adjustment mode used for incremental silver refresh.",
    )
    run_parser.add_argument(
        "--silver-refresh-chunk-size",
        type=int,
        default=25,
        help="Chunk size for the incremental silver refresh universe sync.",
    )
    run_parser.add_argument(
        "--include-silver-symbol-master-refresh",
        action="store_true",
        help="Also refresh symbol_master during the incremental silver update.",
    )
    if run_defaults:
        run_parser.set_defaults(**dict(run_defaults))

    health_parser = subparsers.add_parser("healthcheck", help="Inspect ledger health for daily operations.")
    health_parser.add_argument("--ledger-path", default="artifacts/paper_demo/paper_ledger.sqlite3")

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    run_defaults: dict[str, Any] = {}
    if raw_args[:1] == ["run"]:
        profile_ref = _extract_flag_value(raw_args, "--strategy-profile")
        if profile_ref:
            profile = load_strategy_profile(profile_ref)
            run_defaults = profile.to_arg_defaults()
    parser = build_arg_parser(run_defaults=run_defaults)
    return parser.parse_args(raw_args)


def _extract_flag_value(argv: Sequence[str], flag: str) -> str | None:
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return argv[index + 1]
        prefix = f"{flag}="
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    session_date = parse_session_date(args.session_date)
    artifact_link = build_paper_artifact_link(
        artifact_dir=getattr(args, "artifact_dir", None),
        artifact_root=getattr(args, "artifact_root", "artifacts"),
        model_name=getattr(args, "model", None),
        session_date=session_date,
    )
    resolved_artifact_dir = str(artifact_link.artifact_dir) if artifact_link.artifact_dir is not None else None
    silver_refresh_payload = maybe_refresh_silver_before_run(
        session_date=session_date,
        data_root=args.data_root,
        demo_mode=getattr(args, "demo_mode", False),
        skip_refresh=getattr(args, "skip_silver_refresh", False),
        feed=getattr(args, "silver_refresh_feed", "iex"),
        adjustment=getattr(args, "silver_refresh_adjustment", "raw"),
        chunk_size=getattr(args, "silver_refresh_chunk_size", 25),
        include_symbol_master=getattr(args, "include_silver_symbol_master_refresh", False),
    )
    if not bool(silver_refresh_payload.get("ok", False)):
        payload = PaperDailyOperationPayload(
            command="run",
            ok=False,
            summary={
                "session_date": session_date.isoformat(),
                "run_name": args.run_name,
                "stage": "silver_refresh_failed",
                "decision": "failed",
                "report_status": None,
            },
            silver_refresh=silver_refresh_payload,
            error={
                "type": "SilverRefreshError",
                "message": str(silver_refresh_payload.get("reason", "silver_refresh_failed")),
            },
        )
        return payload.to_dict()

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
            silver_refresh=silver_refresh_payload,
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
        report_payload = report.to_dict()
        report_ok = report.status == "success"
        post_run_payload = _safe_build_post_run_payload(
            ledger_path=args.ledger_path,
            run_id=report.run_id,
            session_date=session_date,
            artifact_dir=resolved_artifact_dir,
        )
        payload = PaperDailyOperationPayload(
            command="run",
            ok=report_ok,
            summary=summarize_paper_daily_result(
                session_date=session_date,
                run_name=args.run_name,
                preflight=preflight,
                report=report,
                override_allowed=args.allow_unhealthy,
                stage=str(report_payload.get("stage", "completed")),
            ),
            silver_refresh=silver_refresh_payload,
            preflight=preflight.to_dict(),
            run={"report": report_payload},
            post_run=post_run_payload,
        )
        if resolved_artifact_dir is not None:
            payload.summary["artifact_dir"] = resolved_artifact_dir
        if getattr(args, "strategy_profile", None) is not None:
            payload.summary["strategy_profile"] = str(args.strategy_profile)
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
            silver_refresh=silver_refresh_payload,
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


def maybe_refresh_silver_before_run(
    *,
    session_date: date,
    data_root: str | Path,
    demo_mode: bool,
    skip_refresh: bool,
    feed: str,
    adjustment: str,
    chunk_size: int,
    include_symbol_master: bool,
) -> dict[str, Any]:
    if demo_mode:
        return {
            "ok": True,
            "performed": False,
            "skipped": True,
            "reason": "demo_mode",
            "data_root": str(Path(data_root)),
            "session_date": session_date.isoformat(),
        }
    if skip_refresh:
        return {
            "ok": True,
            "performed": False,
            "skipped": True,
            "reason": "skip_requested",
            "data_root": str(Path(data_root)),
            "session_date": session_date.isoformat(),
        }

    storage = StorageLayout(root=Path(data_root))
    latest_local_session = _latest_local_daily_bar_session_date(storage)
    if latest_local_session is None:
        return {
            "ok": False,
            "performed": False,
            "skipped": True,
            "reason": "missing_initial_silver_seed",
            "data_root": str(storage.root),
            "session_date": session_date.isoformat(),
        }

    expected_latest_session = _expected_latest_completed_session_date(session_date)
    if latest_local_session >= expected_latest_session:
        return {
            "ok": True,
            "performed": False,
            "skipped": True,
            "reason": "already_fresh_enough",
            "data_root": str(storage.root),
            "session_date": session_date.isoformat(),
            "expected_latest_completed_session": expected_latest_session.isoformat(),
            "latest_local_session_before_refresh": latest_local_session.isoformat(),
        }

    refresh_start = latest_local_session + pd.Timedelta(days=1)
    refresh_end = expected_latest_session
    try:
        result = collect_research_seed(
            start_date=refresh_start,
            end_date=refresh_end,
            layout=storage,
            chunk_size=chunk_size,
            feed=feed,
            adjustment=adjustment,
            include_symbol_master=include_symbol_master,
            include_adj_factor=True,
        )
    except Exception as exc:
        return {
            "ok": False,
            "performed": True,
            "skipped": False,
            "reason": "collector_exception",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "data_root": str(storage.root),
            "session_date": session_date.isoformat(),
            "refresh_start_date": refresh_start.isoformat(),
            "refresh_end_date": refresh_end.isoformat(),
            "expected_latest_completed_session": expected_latest_session.isoformat(),
            "latest_local_session_before_refresh": latest_local_session.isoformat(),
        }

    latest_after = _latest_local_daily_bar_session_date(storage)
    if latest_after is None or latest_after < expected_latest_session:
        return {
            "ok": False,
            "performed": True,
            "skipped": False,
            "reason": "refresh_left_silver_stale",
            "data_root": str(storage.root),
            "session_date": session_date.isoformat(),
            "refresh_start_date": refresh_start.isoformat(),
            "refresh_end_date": refresh_end.isoformat(),
            "expected_latest_completed_session": expected_latest_session.isoformat(),
            "latest_local_session_before_refresh": latest_local_session.isoformat(),
            "latest_local_session_after_refresh": latest_after.isoformat() if latest_after is not None else None,
            "collector": {
                "feed": feed,
                "adjustment": adjustment,
                "chunk_size": chunk_size,
                "include_symbol_master": include_symbol_master,
                "include_adj_factor": True,
            },
            "result": result,
        }
    return {
        "ok": True,
        "performed": True,
        "skipped": False,
        "reason": "refresh_completed",
        "data_root": str(storage.root),
        "session_date": session_date.isoformat(),
        "refresh_start_date": refresh_start.isoformat(),
        "refresh_end_date": refresh_end.isoformat(),
        "expected_latest_completed_session": expected_latest_session.isoformat(),
        "latest_local_session_before_refresh": latest_local_session.isoformat(),
        "latest_local_session_after_refresh": latest_after.isoformat() if latest_after is not None else None,
        "collector": {
            "feed": feed,
            "adjustment": adjustment,
            "chunk_size": chunk_size,
            "include_symbol_master": include_symbol_master,
            "include_adj_factor": True,
        },
        "result": result,
    }


def _latest_local_daily_bar_session_date(storage: StorageLayout) -> date | None:
    dataset = load_us_equities_dataset(layout=storage)
    daily_bar = dataset.get("daily_bar")
    if daily_bar is None or daily_bar.empty:
        return None
    session_dates = pd.to_datetime(daily_bar["session_date"], errors="coerce").dropna()
    if session_dates.empty:
        return None
    return session_dates.max().date()


def _expected_latest_completed_session_date(session_date: date) -> date:
    return (pd.Timestamp(session_date) - pd.offsets.BDay(1)).date()


def summarize_paper_daily_result(
    *,
    session_date: date,
    run_name: str,
    preflight: PaperDailyPreflightResult,
    report: Mapping[str, Any] | None,
    override_allowed: bool,
    stage: str,
) -> dict[str, Any]:
    report_status = None
    report_meta: Mapping[str, Any] = {}
    if isinstance(report, Mapping):
        report_status = report.get("status")
        report_meta = report.get("meta") if isinstance(report.get("meta"), Mapping) else {}
    elif report is not None:
        report_status = getattr(report, "status", None)
        candidate_meta = getattr(report, "meta", None)
        if isinstance(candidate_meta, Mapping):
            report_meta = candidate_meta
    decision = "blocked_preflight"
    if report_status and report_status != "success":
        decision = "failed"
    elif report is not None:
        decision = "executed_with_override" if (override_allowed and not preflight.policy_allowed) else "executed"
    elif stage == "failed":
        decision = "failed"
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
    prediction_context = report_meta.get("prediction_context")
    if isinstance(prediction_context, Mapping):
        summary["prediction_context"] = dict(prediction_context)
    return summary
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
    args = parse_args(argv)
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
