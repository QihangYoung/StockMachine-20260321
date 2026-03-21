from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping, Sequence

from stockmachine.monitoring.alerts import OperatorAlert, build_operator_alerts
from stockmachine.monitoring.reports import PaperRunFailure, build_paper_run_report
from stockmachine.state.ledger import LocalLedger
from stockmachine.state.models import OrderRecord, RunManifestRecord, RunRecord


def build_run_index_payload(
    ledger: LocalLedger,
    *,
    limit: int = 10,
    reference_date: date | None = None,
) -> dict[str, Any]:
    manifests = ledger.list_run_manifests()
    entries = [
        _summarize_run(ledger, manifest, reference_date=reference_date)
        for manifest in manifests[: max(0, int(limit))]
    ]
    return {
        "command": "run-index",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "limit": int(limit),
        "count": len(entries),
        "entries": entries,
    }


def build_daily_summary_payload(
    ledger: LocalLedger,
    *,
    session_date: date | None = None,
    reference_date: date | None = None,
) -> dict[str, Any]:
    target_session_date = session_date or _latest_session_date(ledger) or date.today()
    manifests = [
        manifest
        for manifest in ledger.list_run_manifests()
        if manifest.session_date == target_session_date
    ]
    runs = [_summarize_run(ledger, manifest, reference_date=reference_date) for manifest in manifests]
    alerts = _merge_alerts(entry["alerts"] for entry in runs)
    status_counts = _status_counts(entry["status"] for entry in runs)

    return {
        "command": "daily-summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_date": target_session_date.isoformat(),
        "run_count": len(runs),
        "run_ids": [entry["run_id"] for entry in runs],
        "strategy_names": sorted({entry["strategy_name"] for entry in runs if entry["strategy_name"]}),
        "status_counts": status_counts,
        "order_count": sum(int(entry["order_count"]) for entry in runs),
        "open_order_count": sum(int(entry["open_order_count"]) for entry in runs),
        "fill_count": sum(int(entry["fill_count"]) for entry in runs),
        "rejected_order_count": sum(int(entry["rejected_order_count"]) for entry in runs),
        "alerts": [alert.to_dict() for alert in alerts],
        "runs": runs,
    }


def build_operator_digest_payload(
    ledger: LocalLedger,
    *,
    session_date: date | None = None,
    limit: int = 10,
    reference_date: date | None = None,
) -> dict[str, Any]:
    run_index = build_run_index_payload(ledger, limit=limit, reference_date=reference_date)
    daily_summary = build_daily_summary_payload(ledger, session_date=session_date, reference_date=reference_date)
    from stockmachine.monitoring.health_trend import build_anomaly_summary_payload, build_health_trend_payload

    health_trend = build_health_trend_payload(
        ledger,
        limit_runs=limit,
        limit_sessions=5,
        reference_date=reference_date,
    )
    anomaly_summary = build_anomaly_summary_payload(
        ledger,
        limit_runs=limit,
        reference_date=reference_date,
    )
    run_index_alerts = [
        alert
        for entry in run_index["entries"]
        for alert in entry["alerts"]
    ]
    alerts = _merge_alerts([run_index_alerts, daily_summary["alerts"], health_trend["alerts"], anomaly_summary["alerts"]])
    return {
        "command": "operator-digest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_index": run_index,
        "daily_summary": daily_summary,
        "health_trend": health_trend,
        "anomaly_summary": anomaly_summary,
        "alerts": [alert.to_dict() for alert in alerts],
    }


def _summarize_run(
    ledger: LocalLedger,
    manifest: RunManifestRecord,
    *,
    reference_date: date | None = None,
) -> dict[str, Any]:
    run = ledger.get_run(manifest.run_id)
    orders = ledger.list_orders(run_id=manifest.run_id)
    open_orders = ledger.list_open_orders(run_id=manifest.run_id)
    fills = ledger.list_fills(run_id=manifest.run_id)
    report = _build_report(run=run, manifest=manifest, orders=orders, open_orders=open_orders, fills=fills)
    alerts = build_operator_alerts(
        report=report,
        ledger=ledger,
        run_id=manifest.run_id,
        reference_date=reference_date,
    )

    return {
        "run_id": manifest.run_id,
        "session_date": manifest.session_date.isoformat(),
        "strategy_name": manifest.strategy_name,
        "model_name": manifest.model_name,
        "generated_at_utc": manifest.generated_at_utc.astimezone(timezone.utc).isoformat(),
        "dry_run": manifest.dry_run,
        "status": run.status if run is not None else report.status,
        "order_count": len(orders),
        "open_order_count": len(open_orders),
        "fill_count": len(fills),
        "rejected_order_count": len([order for order in orders if order.status.lower() in {"rejected", "reject"}]),
        "status_counts": _status_counts(order.status for order in orders),
        "manifest": _manifest_to_dict(manifest),
        "run": _run_to_dict(run),
        "report": report.to_dict(),
        "alert_codes": [alert.code for alert in alerts],
        "alerts": [alert.to_dict() for alert in alerts],
        "open_orders": [_order_to_dict(order) for order in open_orders],
    }


def _build_report(
    *,
    run: RunRecord | None,
    manifest: RunManifestRecord,
    orders: Sequence[OrderRecord],
    open_orders: Sequence[OrderRecord],
    fills: Sequence[Any],
):
    failures: list[PaperRunFailure] = []

    if run is not None and run.status.lower() not in {"open", "running", "finished", "success", "completed"}:
        failures.append(
            PaperRunFailure(
                stage="run_status",
                reason=str(run.meta.get("blocked_reason") or run.status),
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

    return build_paper_run_report(
        session_date=manifest.session_date,
        dry_run=manifest.dry_run,
        stage="ledger_summary",
        counts={
            "orders": len(orders),
            "open_orders": len(open_orders),
            "fill_count": len(fills),
        },
        failures=tuple(failures),
        meta={
            "run_manifest": _manifest_to_dict(manifest),
            "run": _run_to_dict(run),
        },
        run_id=manifest.run_id,
        manifest=_manifest_to_dict(manifest),
    )


def _merge_alerts(alert_groups: Sequence[Sequence[OperatorAlert | Mapping[str, Any]]]) -> tuple[OperatorAlert, ...]:
    merged: dict[str, OperatorAlert] = {}
    for group in alert_groups:
        for alert in group:
            candidate = alert if isinstance(alert, OperatorAlert) else OperatorAlert(**dict(alert))
            existing = merged.get(candidate.code)
            if existing is None:
                merged[candidate.code] = candidate
            else:
                merged[candidate.code] = OperatorAlert(
                    code=existing.code,
                    severity=_max_severity(existing.severity, candidate.severity),
                    title=existing.title,
                    message=existing.message,
                    details={**existing.details, **candidate.details},
                )
    return tuple(merged.values())


def _latest_session_date(ledger: LocalLedger) -> date | None:
    manifests = ledger.list_run_manifests()
    if not manifests:
        return None
    return max(manifest.session_date for manifest in manifests)


def _status_counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _max_severity(left: str, right: str) -> str:
    order = {"info": 0, "warning": 1, "critical": 2}
    return left if order.get(left, 0) >= order.get(right, 0) else right


def _manifest_to_dict(manifest: RunManifestRecord) -> dict[str, Any]:
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


def _order_to_dict(order: OrderRecord) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "run_id": order.run_id,
        "session_date": order.session_date.isoformat() if order.session_date else None,
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
