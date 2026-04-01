from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from stockmachine.apps.operator_paths import resolve_operator_ledger_path, resolve_strategy_workspace
from stockmachine.execution.brokers import AlpacaTradingAdapter
from stockmachine.live import backfill_order_statuses
from stockmachine.state import LocalLedger


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill broker order history into the local paper ledger.")
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
        "--ledger-path",
        default=None,
        help="Path to the local paper-demo ledger SQLite file.",
    )
    parser.add_argument("--run-id", default=None, help="Specific run id to backfill. Defaults to the latest run.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("latest-run", help="Backfill the latest run in the ledger.")
    return parser


def build_backfill_payload(args: argparse.Namespace) -> dict[str, Any]:
    broker = AlpacaTradingAdapter.from_env()
    resolved_ledger_path = resolve_operator_ledger_path(
        ledger_path=args.ledger_path,
        strategy_project=getattr(args, "strategy_project", None),
        artifact_root=getattr(args, "artifact_root", "artifacts"),
    )
    workspace = resolve_strategy_workspace(
        strategy_project=getattr(args, "strategy_project", None),
        artifact_root=getattr(args, "artifact_root", "artifacts"),
    )
    with LocalLedger(resolved_ledger_path) as ledger:
        ledger.initialize()
        resolved_run_id = _resolve_run_id(ledger, args)
        if resolved_run_id is None:
            return {
                "command": "backfill",
                "ok": False,
                "ledger_path": str(resolved_ledger_path),
                "run_id": None,
                "error": {
                    "type": "ValueError",
                    "message": "ledger is empty",
                },
            }

        candidate_orders = ledger.list_orders(run_id=resolved_run_id)
        before_fills = len(ledger.list_fills(run_id=resolved_run_id))
        before_audits = len(ledger.list_fill_audits(run_id=resolved_run_id))
        before_open_orders = len(ledger.list_open_orders(run_id=resolved_run_id))

        result = backfill_order_statuses(
            ledger,
            broker,
            candidate_orders=candidate_orders,
        )

        after_fills = len(ledger.list_fills(run_id=resolved_run_id))
        after_audits = len(ledger.list_fill_audits(run_id=resolved_run_id))
        after_open_orders = len(ledger.list_open_orders(run_id=resolved_run_id))

        payload = {
            "command": "backfill",
            "ok": len(result.errors) == 0,
            "ledger_path": str(resolved_ledger_path),
            "run_id": resolved_run_id,
            "summary": {
                "candidate_orders": len(candidate_orders),
                "before_fill_count": before_fills,
                "after_fill_count": after_fills,
                "fills_created": after_fills - before_fills,
                "before_fill_audit_count": before_audits,
                "after_fill_audit_count": after_audits,
                "fill_audits_created": after_audits - before_audits,
                "before_open_order_count": before_open_orders,
                "after_open_order_count": after_open_orders,
                "open_orders_cleared": before_open_orders - after_open_orders,
            },
            "backfill": result.to_dict(),
        }
        if workspace is not None:
            payload["strategy_workspace"] = workspace.to_dict()
        return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = build_backfill_payload(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("ok") else 1


def _resolve_run_id(ledger: LocalLedger, args: argparse.Namespace) -> str | None:
    if args.command == "latest-run":
        return _latest_run_id(ledger)
    if args.run_id:
        return str(args.run_id)
    return _latest_run_id(ledger)


def _latest_run_id(ledger: LocalLedger) -> str | None:
    manifests = ledger.list_run_manifests()
    if manifests:
        return manifests[0].run_id

    orders = ledger.list_orders()
    for order in orders:
        if order.run_id:
            return order.run_id
    return None


if __name__ == "__main__":
    raise SystemExit(main())
