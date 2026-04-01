from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from stockmachine.alpha import list_alpha_expert_names
from stockmachine.apps import paper_daily
from stockmachine.apps.paper_profiles import load_strategy_profile
from stockmachine.apps.run_us_equities_paper import parse_session_date
from stockmachine.domain.project_paths import build_strategy_project_paths
from stockmachine.monitoring.reports import build_paper_artifact_link


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a repeatable paper smoke harness and emit JSON.")
    parser.add_argument("--strategy-profile", default=None, help="Built-in strategy profile name or JSON path.")
    parser.add_argument("--session-date", default=date.today().isoformat())
    parser.add_argument("--run-name", default="paper-smoke")
    parser.add_argument("--model", default="hist_gbm", choices=list_alpha_expert_names())
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--ledger-path", default="artifacts/paper_demo/paper_ledger.sqlite3")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--universe", nargs="*", default=())
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--demo-mode", action="store_true")
    parser.add_argument("--min-close", type=float, default=10.0)
    parser.add_argument("--min-median-dollar-volume-20", type=float, default=50_000_000.0)
    parser.add_argument("--max-vol-20", type=float, default=0.04)
    parser.add_argument("--max-positions-per-sector", type=int, default=2)
    parser.add_argument("--disable-sector-neutral", action="store_true")
    parser.add_argument("--require-market-open", action="store_true")
    parser.add_argument("--min-buying-power-buffer", type=float, default=0.0)
    parser.add_argument("--max-order-notional", type=float, default=None)
    parser.add_argument("--max-total-notional", type=float, default=None)
    parser.add_argument("--max-total-orders", type=int, default=None)
    parser.add_argument("--execution-equity-cap", type=float, default=None)
    parser.add_argument("--post-submit-poll-seconds", type=float, default=15.0)
    parser.add_argument("--post-submit-poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--allow-unhealthy", action="store_true")
    parser.add_argument("--kill-switch-path", default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    profile_ref = _extract_flag_value(raw_args, "--strategy-profile")
    parser = build_arg_parser()
    if profile_ref:
        profile = load_strategy_profile(profile_ref)
        parser.set_defaults(**profile.to_arg_defaults())
    return parser.parse_args(raw_args)


def _extract_flag_value(argv: Sequence[str], flag: str) -> str | None:
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return argv[index + 1]
        prefix = f"{flag}="
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def build_smoke_payload(args: argparse.Namespace) -> dict[str, Any]:
    session_date = parse_session_date(args.session_date)
    artifact_link = build_paper_artifact_link(
        artifact_dir=args.artifact_dir,
        artifact_root=args.artifact_root,
        run_name=args.run_name,
        session_date=session_date,
        model_name=args.model,
        strategy_project=getattr(args, "strategy_project", None),
    )
    strategy_lineage = paper_daily._strategy_lineage_from_args(args)
    strategy_workspace = _strategy_workspace_from_args(args)
    run_args = _build_run_namespace(args, session_date=session_date, artifact_link=artifact_link)
    try:
        run_payload = paper_daily.run_command(run_args)
    except Exception as exc:  # pragma: no cover - safety net for operator runs
        return {
            "command": "smoke",
            "ok": False,
            "session_date": session_date.isoformat(),
            "run_name": args.run_name,
            "strategy_profile": getattr(args, "strategy_profile", None),
            "strategy_lineage": strategy_lineage,
            "strategy_workspace": strategy_workspace,
            "model": args.model,
            "dry_run": not args.execute,
            "artifact_link": artifact_link.to_dict(),
            "run": None,
            "preflight": None,
            "post_run": None,
            "summary": {
                "stage": "exception",
                "decision": "failed",
            },
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "recommended_commands": _build_recommended_commands(
                args=args,
                session_date=session_date,
                artifact_link=artifact_link,
                run_id=None,
            ),
            "guardrails": _guardrails_payload(args),
        }

    run_block = run_payload.get("run") if isinstance(run_payload, Mapping) else None
    report = run_block.get("report") if isinstance(run_block, Mapping) else None
    run_id = report.get("run_id") if isinstance(report, Mapping) else None
    return {
        "command": "smoke",
        "ok": bool(run_payload.get("ok")) if isinstance(run_payload, Mapping) else False,
        "session_date": session_date.isoformat(),
        "run_name": args.run_name,
        "strategy_profile": getattr(args, "strategy_profile", None),
        "strategy_lineage": strategy_lineage,
        "strategy_workspace": strategy_workspace,
        "model": args.model,
        "dry_run": not args.execute,
        "artifact_link": artifact_link.to_dict(),
        "summary": run_payload.get("summary") if isinstance(run_payload, Mapping) else None,
        "silver_refresh": run_payload.get("silver_refresh") if isinstance(run_payload, Mapping) else None,
        "preflight": run_payload.get("preflight") if isinstance(run_payload, Mapping) else None,
        "run": run_payload.get("run") if isinstance(run_payload, Mapping) else None,
        "post_run": run_payload.get("post_run") if isinstance(run_payload, Mapping) else None,
        "error": run_payload.get("error") if isinstance(run_payload, Mapping) else None,
        "run_id": run_id,
        "recommended_commands": _build_recommended_commands(
            args=args,
            session_date=session_date,
            artifact_link=artifact_link,
            run_id=run_id,
        ),
        "guardrails": _guardrails_payload(args),
    }


def _build_run_namespace(
    args: argparse.Namespace,
    *,
    session_date: date,
    artifact_link,
) -> argparse.Namespace:
    artifact_dir = str(artifact_link.artifact_dir) if artifact_link.artifact_dir is not None else None
    return argparse.Namespace(
        session_date=session_date.isoformat(),
        universe=list(args.universe),
        run_name=args.run_name,
        execute=args.execute,
        demo_mode=args.demo_mode,
        model=args.model,
        strategy_profile=getattr(args, "strategy_profile", None),
        strategy_family=getattr(args, "strategy_family", None),
        strategy_project=getattr(args, "strategy_project", None),
        strategy_horizon_bucket=getattr(args, "strategy_horizon_bucket", None),
        strategy_track=getattr(args, "strategy_track", None),
        top_k=args.top_k,
        horizon=args.horizon,
        data_root=args.data_root,
        ledger_path=args.ledger_path,
        min_close=args.min_close,
        min_median_dollar_volume_20=args.min_median_dollar_volume_20,
        max_vol_20=args.max_vol_20,
        max_positions_per_sector=args.max_positions_per_sector,
        disable_sector_neutral=args.disable_sector_neutral,
        require_market_open=args.require_market_open,
        min_buying_power_buffer=args.min_buying_power_buffer,
        max_order_notional=args.max_order_notional,
        max_total_notional=args.max_total_notional,
        max_total_orders=args.max_total_orders,
        execution_equity_cap=args.execution_equity_cap,
        post_submit_poll_seconds=args.post_submit_poll_seconds,
        post_submit_poll_interval_seconds=args.post_submit_poll_interval_seconds,
        allow_unhealthy=args.allow_unhealthy,
        artifact_dir=artifact_dir,
        kill_switch_path=args.kill_switch_path,
    )


def _build_recommended_commands(
    *,
    args: argparse.Namespace,
    session_date: date,
    artifact_link,
    run_id: str | None,
) -> dict[str, list[str]]:
    artifact_args = ["--artifact-dir", str(artifact_link.artifact_dir)] if artifact_link.artifact_dir is not None else []
    reconcile_run_selector = ["--run-id", run_id] if run_id is not None else ["latest-run"]
    daily_command = [
        "python",
        "-m",
        "stockmachine.apps.paper_daily",
        "run",
        "--session-date",
        session_date.isoformat(),
        "--run-name",
        args.run_name,
        "--model",
        args.model,
        "--top-k",
        str(args.top_k),
        "--horizon",
        str(args.horizon),
        "--ledger-path",
        str(args.ledger_path),
        "--data-root",
        str(args.data_root),
    ]
    if args.execute:
        daily_command.append("--execute")
    if args.demo_mode:
        daily_command.append("--demo-mode")
    if args.strategy_profile:
        daily_command.extend(["--strategy-profile", str(args.strategy_profile)])
    if args.universe:
        daily_command.extend(["--universe", *list(args.universe)])
    if args.artifact_dir is not None or artifact_link.artifact_dir is not None:
        daily_command.extend(artifact_args)

    reconcile_command = [
        "python",
        "-m",
        "stockmachine.apps.paper_reconcile",
        "--ledger",
        str(args.ledger_path),
        *reconcile_run_selector,
    ]
    reconcile_command.extend(["--artifact-model", args.model, "--top-k", str(args.top_k)])
    reconcile_command.extend(artifact_args)

    maintain_command = [
        "python",
        "-m",
        "stockmachine.apps.paper_maintain",
        "latest-run",
        "--ledger",
        str(args.ledger_path),
    ]
    return {
        "paper_daily": daily_command,
        "paper_reconcile": reconcile_command,
        "paper_maintain": maintain_command,
    }


def _guardrails_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "execute": bool(args.execute),
        "demo_mode": bool(args.demo_mode),
        "require_market_open": bool(args.require_market_open),
        "allow_unhealthy": bool(args.allow_unhealthy),
        "max_order_notional": args.max_order_notional,
        "max_total_notional": args.max_total_notional,
        "max_total_orders": args.max_total_orders,
        "execution_equity_cap": args.execution_equity_cap,
        "min_buying_power_buffer": args.min_buying_power_buffer,
    }


def _strategy_workspace_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    strategy_project = getattr(args, "strategy_project", None)
    if strategy_project in (None, ""):
        return None
    return build_strategy_project_paths(str(strategy_project), artifact_root=args.artifact_root).to_dict()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_smoke_payload(args)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
