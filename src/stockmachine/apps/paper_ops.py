from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from stockmachine.apps.operator_paths import resolve_operator_ledger_path, resolve_strategy_workspace
from stockmachine.monitoring.alerts import build_operator_alerts
from stockmachine.monitoring.reports import PaperRunFailure, build_paper_run_report
from stockmachine.state import LocalLedger
from stockmachine.state.models import OrderRecord, RunManifestRecord, RunRecord


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the paper-demo ledger and emit JSON.")
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
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("latest-run", help="Show the latest run summary from the ledger.")
    subparsers.add_parser("open-orders", help="Show open orders from the latest run.")

    run_summary = subparsers.add_parser("run-summary", help="Show a single run summary by run id.")
    run_summary.add_argument("--run-id", required=True, help="Run id to inspect.")

    return parser


def dispatch_command(args: argparse.Namespace) -> dict[str, Any]:
    with LocalLedger(args.ledger_path) as ledger:
        ledger.initialize()
        if args.command == "latest-run":
            return build_latest_run_payload(ledger, ledger_path=args.ledger_path)
        if args.command == "open-orders":
            return build_open_orders_payload(ledger, ledger_path=args.ledger_path)
        if args.command == "run-summary":
            return build_run_summary_payload(ledger, run_id=args.run_id, ledger_path=args.ledger_path)
    raise ValueError(f"Unsupported command: {args.command}")


def build_latest_run_payload(ledger: LocalLedger, *, ledger_path: str | Path | None = None) -> dict[str, Any]:
    run_id = _latest_run_id(ledger)
    if run_id is None:
        return {
            "command": "latest-run",
            "ledger_path": str(ledger_path) if ledger_path is not None else None,
            "run_id": None,
            "run": None,
            "manifest": None,
            "summary": {"message": "ledger is empty"},
            "alerts": [],
        }
    return build_run_summary_payload(ledger, run_id=run_id, ledger_path=ledger_path, command="latest-run")


def build_open_orders_payload(ledger: LocalLedger, *, ledger_path: str | Path | None = None) -> dict[str, Any]:
    run_id = _latest_run_id(ledger)
    run = ledger.get_run(run_id) if run_id is not None else None
    manifest = ledger.get_run_manifest(run_id) if run_id is not None else None
    orders = ledger.list_orders(run_id=run_id) if run_id is not None else []
    open_orders = ledger.list_open_orders(run_id=run_id) if run_id is not None else []
    decisions = ledger.list_order_decisions(run_id=run_id) if run_id is not None else []
    fills = ledger.list_fills(run_id=run_id) if run_id is not None else []
    fill_audits = ledger.list_fill_audits(run_id=run_id) if run_id is not None else []
    equity_snapshots = ledger.list_equity_snapshots(run_id=run_id) if run_id is not None else []
    report = _build_report_from_ledger(
        run=run,
        manifest=manifest,
        run_id=run_id or "",
        orders=orders,
        open_orders=open_orders,
        fill_count=len(fills),
    ) if run_id is not None else None
    alerts = build_operator_alerts(report=report, ledger=ledger, run_id=run_id)
    return {
        "command": "open-orders",
        "ledger_path": str(ledger_path) if ledger_path is not None else None,
        "run_id": run_id,
        "count": len(open_orders),
        "open_orders": [_order_to_dict(order) for order in open_orders],
        "report": report.to_dict() if report is not None else None,
        "summary": {
            "order_count": len(orders),
            "open_order_count": len(open_orders),
            "order_decision_count": len(decisions),
            "fill_count": len(fills),
            "fill_audit_count": len(fill_audits),
            "equity_snapshot_count": len(equity_snapshots),
            "status_counts": _status_counts(orders),
        },
        "alerts": [alert.to_dict() for alert in alerts],
    }


def build_run_summary_payload(
    ledger: LocalLedger,
    *,
    run_id: str,
    ledger_path: str | Path | None = None,
    command: str = "run-summary",
) -> dict[str, Any]:
    run = ledger.get_run(run_id)
    manifest = ledger.get_run_manifest(run_id)
    orders = ledger.list_orders(run_id=run_id)
    open_orders = ledger.list_open_orders(run_id=run_id)
    decisions = ledger.list_order_decisions(run_id=run_id)
    fills = ledger.list_fills(run_id=run_id)
    fill_audits = ledger.list_fill_audits(run_id=run_id)
    equity_snapshots = ledger.list_equity_snapshots(run_id=run_id)
    report = _build_report_from_ledger(
        run=run,
        manifest=manifest,
        run_id=run_id,
        orders=orders,
        open_orders=open_orders,
        fill_count=len(fills),
    )
    alerts = build_operator_alerts(report=report, ledger=ledger, run_id=run_id)

    return {
        "command": command,
        "ledger_path": str(ledger_path) if ledger_path is not None else None,
        "run_id": run_id,
        "run": _run_to_dict(run),
        "manifest": _manifest_to_dict(manifest),
        "report": report.to_dict(),
        "summary": {
            "order_count": len(orders),
            "open_order_count": len(open_orders),
            "order_decision_count": len(decisions),
            "fill_count": len(fills),
            "fill_audit_count": len(fill_audits),
            "equity_snapshot_count": len(equity_snapshots),
            "status_counts": _status_counts(orders),
        },
        "orders": [_order_to_dict(order) for order in orders],
        "open_orders": [_order_to_dict(order) for order in open_orders],
        "decisions": [_json_safe(decision) for decision in decisions],
        "fills": [_json_safe(fill) for fill in fills],
        "fill_audits": [_json_safe(audit) for audit in fill_audits],
        "equity_snapshots": [_json_safe(snapshot) for snapshot in equity_snapshots],
        "alerts": [alert.to_dict() for alert in alerts],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.ledger_path = str(
        resolve_operator_ledger_path(
            ledger_path=args.ledger_path,
            strategy_project=getattr(args, "strategy_project", None),
            artifact_root=getattr(args, "artifact_root", "artifacts"),
        )
    )
    payload = dispatch_command(args)
    workspace = resolve_strategy_workspace(
        strategy_project=getattr(args, "strategy_project", None),
        artifact_root=getattr(args, "artifact_root", "artifacts"),
    )
    if workspace is not None:
        payload["strategy_workspace"] = workspace.to_dict()
    print(json.dumps(payload, indent=2, default=_json_default, sort_keys=True))


def _latest_run_id(ledger: LocalLedger) -> str | None:
    manifests = ledger.list_run_manifests()
    if manifests:
        return manifests[0].run_id

    orders = ledger.list_orders()
    for order in orders:
        if order.run_id:
            return order.run_id
    return None


def _build_report_from_ledger(
    *,
    run: RunRecord | None,
    manifest: RunManifestRecord | None,
    run_id: str,
    orders: Sequence[OrderRecord],
    open_orders: Sequence[OrderRecord],
    fill_count: int,
):
    session_date = manifest.session_date if manifest is not None else date.today()
    dry_run = bool(manifest.dry_run) if manifest is not None else False
    failures: list[PaperRunFailure] = []

    if run is not None and run.status.lower() not in {"open", "running", "finished", "success", "completed"}:
        reason = str(run.meta.get("blocked_reason") or run.status)
        failures.append(
            PaperRunFailure(
                stage="run_status",
                reason=reason,
                details={"run_status": run.status, "run_meta": _json_safe(run.meta)},
            )
        )

    rejected_orders = [order for order in orders if order.status.lower() in {"rejected", "reject"}]
    if rejected_orders:
        failures.append(
            PaperRunFailure(
                stage="broker_reconciliation",
                reason="broker_rejection",
                details={
                    "rejected_order_count": len(rejected_orders),
                    "symbols": sorted({order.symbol for order in rejected_orders}),
                },
            )
        )

    if open_orders:
        failures.append(
            PaperRunFailure(
                stage="broker_reconciliation",
                reason="open_orders_lingering",
                details={
                    "open_order_count": len(open_orders),
                    "symbols": sorted({order.symbol for order in open_orders}),
                },
            )
        )

    counts = {
        "orders": len(orders),
        "open_orders": len(open_orders),
        "fill_count": fill_count,
    }
    meta: dict[str, Any] = {"source": "paper_ops"}
    if manifest is not None:
        meta["run_manifest"] = _manifest_to_dict(manifest)
    if run is not None:
        meta["run"] = _run_to_dict(run)

    return build_paper_run_report(
        session_date=session_date,
        dry_run=dry_run,
        stage="ledger_summary",
        counts=counts,
        failures=tuple(failures),
        meta=meta,
        run_id=run_id,
        manifest=_manifest_to_dict(manifest) if manifest is not None else None,
    )


def _run_to_dict(run: RunRecord | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "run_id": run.run_id,
        "strategy_name": run.strategy_name,
        "market": run.market,
        "created_at_utc": run.created_at_utc.astimezone(timezone.utc).isoformat(),
        "status": run.status,
        "meta": _json_safe(run.meta),
    }


def _manifest_to_dict(manifest: RunManifestRecord | None) -> dict[str, Any] | None:
    if manifest is None:
        return None
    return {
        "run_id": manifest.run_id,
        "session_date": manifest.session_date.isoformat(),
        "strategy_name": manifest.strategy_name,
        "model_name": manifest.model_name,
        "generated_at_utc": manifest.generated_at_utc.astimezone(timezone.utc).isoformat(),
        "client_order_id_prefix": manifest.client_order_id_prefix,
        "dry_run": manifest.dry_run,
        "data_snapshot": _json_safe(manifest.data_snapshot),
        "risk_policy": _json_safe(manifest.risk_policy),
        "execution_policy": _json_safe(manifest.execution_policy),
        "meta": _json_safe(manifest.meta),
    }


def _order_to_dict(order: OrderRecord) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "run_id": order.run_id,
        "session_date": order.session_date.isoformat() if order.session_date is not None else None,
        "client_order_id": order.client_order_id,
        "symbol": order.symbol,
        "side": order.side,
        "quantity": order.quantity,
        "order_type": order.order_type,
        "limit_price": order.limit_price,
        "status": order.status,
        "filled_quantity": order.filled_quantity,
        "avg_fill_price": order.avg_fill_price,
        "submitted_at_utc": order.submitted_at_utc.astimezone(timezone.utc).isoformat(),
        "updated_at_utc": order.updated_at_utc.astimezone(timezone.utc).isoformat(),
        "broker_payload": _json_safe(order.broker_payload),
    }


def _status_counts(orders: Sequence[OrderRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for order in orders:
        status = order.status.lower()
        counts[status] = counts.get(status, 0) + 1
    return counts


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return asdict(value)
    return value


def _json_default(value: Any) -> Any:
    return _json_safe(value)


if __name__ == "__main__":
    main()
