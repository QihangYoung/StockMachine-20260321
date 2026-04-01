from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from stockmachine.apps.operator_paths import resolve_operator_ledger_path, resolve_strategy_workspace
from stockmachine.domain.datetime_utils import parse_iso_datetime_like
from stockmachine.live.order_maintenance import (
    OrderMaintenancePolicy,
    build_order_maintenance_summary,
    load_broker_open_orders_from_json,
)
from stockmachine.state import LocalLedger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan stale open-order maintenance for paper trading runs.")
    parser.add_argument(
        "--strategy-project",
        default=None,
        help="Optional strategy project id used to resolve default operator paths.",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts",
        help="Artifact root used when resolving project-scoped default paths.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="Path to the paper-trading SQLite ledger.",
    )
    parser.add_argument("--run-id", help="Use a specific run id.")
    _add_common_args(parser)
    subparsers = parser.add_subparsers(dest="command")
    latest_parser = subparsers.add_parser("latest-run", help="Plan maintenance for the latest run in the ledger.")
    _add_common_args(latest_parser)
    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--broker-orders-json",
        type=Path,
        help="Optional JSON file containing broker open-order snapshots.",
    )
    parser.add_argument(
        "--stale-after-minutes",
        type=int,
        default=60,
        help="Mark open orders as stale when they are older than this threshold.",
    )
    parser.add_argument(
        "--session-date",
        type=str,
        help="Optional session date override in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--as-of-utc",
        type=str,
        help="Optional UTC timestamp override for stale-age calculations.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.ledger = resolve_operator_ledger_path(
        ledger_path=args.ledger,
        strategy_project=getattr(args, "strategy_project", None),
        artifact_root=getattr(args, "artifact_root", "artifacts"),
    )
    ledger = LocalLedger(args.ledger)
    ledger.initialize()
    try:
        run_id = args.run_id
        if args.command == "latest-run":
            run_id = None
        broker_open_orders = ()
        if args.broker_orders_json is not None:
            broker_open_orders = load_broker_open_orders_from_json(args.broker_orders_json)
        policy = OrderMaintenancePolicy(stale_after_minutes=args.stale_after_minutes)
        session_date = datetime.strptime(args.session_date, "%Y-%m-%d").date() if args.session_date else None
        as_of_utc = _parse_as_of_utc(args.as_of_utc)
        summary = build_order_maintenance_summary(
            ledger,
            broker_open_orders,
            run_id=run_id,
            policy=policy,
            session_date=session_date,
            as_of_utc=as_of_utc,
        )
        workspace = resolve_strategy_workspace(
            strategy_project=getattr(args, "strategy_project", None),
            artifact_root=getattr(args, "artifact_root", "artifacts"),
        )
        payload = summary.to_dict()
        if workspace is not None:
            payload["strategy_workspace"] = workspace.to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        ledger.close()


def _parse_as_of_utc(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = parse_iso_datetime_like(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())
