from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from stockmachine.live.reconciler import BrokerOrderSnapshot
from stockmachine.live.recovery import RecoveryPlan, build_recovery_plan
from stockmachine.state import LocalLedger
from stockmachine.state.models import OrderRecord, RunManifestRecord


@dataclass(slots=True, frozen=True)
class OrderMaintenancePolicy:
    """Policy knobs for stale-open-order maintenance."""

    stale_after_minutes: int = 60
    cancel_orphan_broker_orders: bool = True
    cancel_stale_ledger_orders: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "stale_after_minutes": self.stale_after_minutes,
            "cancel_orphan_broker_orders": self.cancel_orphan_broker_orders,
            "cancel_stale_ledger_orders": self.cancel_stale_ledger_orders,
        }


@dataclass(slots=True, frozen=True)
class MaintenanceCandidate:
    """A single open order that should be reviewed or canceled."""

    order_id: str
    symbol: str
    client_order_id: str | None
    source: str
    stale_reasons: tuple[str, ...] = ()
    age_minutes: float | None = None
    ledger_status: str | None = None
    broker_status: str | None = None
    session_date: date | None = None
    submitted_at_utc: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "client_order_id": self.client_order_id,
            "source": self.source,
            "stale_reasons": list(self.stale_reasons),
            "age_minutes": self.age_minutes,
            "ledger_status": self.ledger_status,
            "broker_status": self.broker_status,
            "session_date": self.session_date.isoformat() if self.session_date is not None else None,
            "submitted_at_utc": self.submitted_at_utc.astimezone(timezone.utc).isoformat()
            if self.submitted_at_utc is not None
            else None,
        }


@dataclass(slots=True, frozen=True)
class OrderMaintenancePlan:
    """Maintenance plan for stale and aligned open orders."""

    run_id: str | None
    session_date: date | None
    policy: OrderMaintenancePolicy
    recovery_plan: RecoveryPlan
    aligned_open_orders: tuple[MaintenanceCandidate, ...] = ()
    stale_ledger_orders: tuple[MaintenanceCandidate, ...] = ()
    stale_broker_orders: tuple[MaintenanceCandidate, ...] = ()
    cancel_candidates: tuple[MaintenanceCandidate, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_date": self.session_date.isoformat() if self.session_date is not None else None,
            "policy": self.policy.to_dict(),
            "recovery_plan": _recovery_plan_to_dict(self.recovery_plan),
            "aligned_open_orders": [candidate.to_dict() for candidate in self.aligned_open_orders],
            "stale_ledger_orders": [candidate.to_dict() for candidate in self.stale_ledger_orders],
            "stale_broker_orders": [candidate.to_dict() for candidate in self.stale_broker_orders],
            "cancel_candidates": [candidate.to_dict() for candidate in self.cancel_candidates],
        }


@dataclass(slots=True, frozen=True)
class OrderMaintenanceExecutionResult:
    """Result of applying the maintenance plan."""

    requested: bool
    executed: bool
    canceled_order_ids: tuple[str, ...] = ()
    failed_order_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "executed": self.executed,
            "canceled_order_ids": list(self.canceled_order_ids),
            "failed_order_ids": list(self.failed_order_ids),
            "notes": list(self.notes),
        }


@dataclass(slots=True, frozen=True)
class OrderMaintenanceSummary:
    """JSON-ready report for scheduler/operator consumption."""

    found: bool
    run_id: str | None
    session_date: date | None
    manifest: Mapping[str, Any]
    policy: OrderMaintenancePolicy
    plan: OrderMaintenancePlan
    execution: OrderMaintenanceExecutionResult
    notes: tuple[str, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        return {
            "aligned_open_order_count": len(self.plan.aligned_open_orders),
            "stale_ledger_order_count": len(self.plan.stale_ledger_orders),
            "stale_broker_order_count": len(self.plan.stale_broker_orders),
            "cancel_candidate_count": len(self.plan.cancel_candidates),
            "recovery_orphan_count": self.plan.recovery_plan.orphan_count,
            "recovery_stale_count": self.plan.recovery_plan.stale_count,
            "recovery_aligned_count": self.plan.recovery_plan.aligned_count,
            "executed_cancel_count": len(self.execution.canceled_order_ids),
            "failed_cancel_count": len(self.execution.failed_order_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "run_id": self.run_id,
            "session_date": self.session_date.isoformat() if self.session_date is not None else None,
            "manifest": dict(self.manifest),
            "policy": self.policy.to_dict(),
            "counts": self.counts,
            "plan": self.plan.to_dict(),
            "execution": self.execution.to_dict(),
            "notes": list(self.notes),
        }


class OrderCanceller(Protocol):
    """Minimal cancellation surface for optional execution seams."""

    def cancel_order(self, order_id: str) -> object:
        """Cancel a single open order."""


def build_order_maintenance_summary(
    ledger: LocalLedger,
    broker_open_orders: Sequence[BrokerOrderSnapshot | object],
    *,
    run_id: str | None = None,
    policy: OrderMaintenancePolicy | None = None,
    session_date: date | None = None,
    as_of_utc: datetime | None = None,
) -> OrderMaintenanceSummary:
    """Build a maintenance plan and summary from ledger and broker open orders."""

    policy = policy or OrderMaintenancePolicy()
    as_of = _ensure_utc(as_of_utc or datetime.now(timezone.utc))
    resolved_run_id, manifest = _resolve_run_context(ledger, run_id)
    if resolved_run_id is None:
        empty_plan = OrderMaintenancePlan(
            run_id=None,
            session_date=session_date,
            policy=policy,
            recovery_plan=RecoveryPlan(),
        )
        return OrderMaintenanceSummary(
            found=False,
            run_id=None,
            session_date=session_date,
            manifest={},
            policy=policy,
            plan=empty_plan,
            execution=OrderMaintenanceExecutionResult(requested=False, executed=False, notes=("no_run_found",)),
            notes=("no_run_found",),
        )

    resolved_session_date = session_date or (manifest.session_date if manifest is not None else None)
    ledger_open_orders = ledger.list_open_orders(run_id=resolved_run_id)
    broker_snapshots = tuple(
        snapshot if isinstance(snapshot, BrokerOrderSnapshot) else BrokerOrderSnapshot.from_payload(snapshot)
        for snapshot in broker_open_orders
    )
    recovery_plan = build_recovery_plan(ledger_open_orders, broker_snapshots)

    plan = _build_maintenance_plan(
        ledger_open_orders,
        broker_snapshots,
        recovery_plan=recovery_plan,
        policy=policy,
        session_date=resolved_session_date,
        as_of_utc=as_of,
    )
    plan = OrderMaintenancePlan(
        run_id=resolved_run_id,
        session_date=plan.session_date,
        policy=plan.policy,
        recovery_plan=plan.recovery_plan,
        aligned_open_orders=plan.aligned_open_orders,
        stale_ledger_orders=plan.stale_ledger_orders,
        stale_broker_orders=plan.stale_broker_orders,
        cancel_candidates=plan.cancel_candidates,
    )
    execution = apply_order_maintenance_plan(plan, execute=False)
    return OrderMaintenanceSummary(
        found=True,
        run_id=resolved_run_id,
        session_date=resolved_session_date,
        manifest=_manifest_to_dict(manifest),
        policy=policy,
        plan=plan,
        execution=execution,
        notes=_build_summary_notes(manifest, plan),
    )


def apply_order_maintenance_plan(
    plan: OrderMaintenancePlan,
    *,
    execute: bool = False,
    canceller: OrderCanceller | None = None,
) -> OrderMaintenanceExecutionResult:
    """Optionally execute cancel requests for the maintenance plan."""

    if not execute:
        return OrderMaintenanceExecutionResult(requested=False, executed=False, notes=("dry_run",))
    if canceller is None:
        raise ValueError("execute=True requires a canceller.")

    canceled_order_ids: list[str] = []
    failed_order_ids: list[str] = []
    for candidate in plan.cancel_candidates:
        try:
            canceller.cancel_order(candidate.order_id)
            canceled_order_ids.append(candidate.order_id)
        except Exception:
            failed_order_ids.append(candidate.order_id)

    notes = ("executed",) if not failed_order_ids else ("executed_with_failures",)
    return OrderMaintenanceExecutionResult(
        requested=True,
        executed=True,
        canceled_order_ids=tuple(canceled_order_ids),
        failed_order_ids=tuple(failed_order_ids),
        notes=notes,
    )


def load_broker_open_orders_from_json(path: str | Path) -> tuple[BrokerOrderSnapshot, ...]:
    """Load broker open-order snapshots from a JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Broker open-order JSON must contain a list of order objects.")
    return tuple(BrokerOrderSnapshot.from_payload(item) for item in payload)


def _build_maintenance_plan(
    ledger_open_orders: Sequence[OrderRecord],
    broker_open_orders: Sequence[BrokerOrderSnapshot],
    *,
    recovery_plan: RecoveryPlan,
    policy: OrderMaintenancePolicy,
    session_date: date | None,
    as_of_utc: datetime,
) -> OrderMaintenancePlan:
    broker_by_order_id = {order.order_id: order for order in broker_open_orders}
    broker_by_client_order_id = {
        order.client_order_id: order
        for order in broker_open_orders
        if order.client_order_id is not None
    }

    aligned_open_orders: list[MaintenanceCandidate] = []
    stale_ledger_orders: list[MaintenanceCandidate] = []
    stale_broker_orders: list[MaintenanceCandidate] = []

    for ledger_order in sorted(ledger_open_orders, key=_ledger_sort_key):
        broker_order = broker_by_order_id.get(ledger_order.order_id)
        if broker_order is None and ledger_order.client_order_id is not None:
            broker_order = broker_by_client_order_id.get(ledger_order.client_order_id)

        candidate = _candidate_from_pair(
            ledger_order=ledger_order,
            broker_order=broker_order,
            source="ledger",
            session_date=session_date,
            as_of_utc=as_of_utc,
            stale_after_minutes=policy.stale_after_minutes,
        )
        if _is_candidate_stale(candidate):
            stale_ledger_orders.append(candidate)
        else:
            aligned_open_orders.append(candidate)

    matched_broker_ids = {candidate.order_id for candidate in aligned_open_orders} | {
        candidate.order_id for candidate in stale_ledger_orders
    }
    for broker_order in sorted(broker_open_orders, key=_broker_sort_key):
        if broker_order.order_id in matched_broker_ids:
            continue
        candidate = _candidate_from_pair(
            ledger_order=None,
            broker_order=broker_order,
            source="broker",
            session_date=session_date,
            as_of_utc=as_of_utc,
            stale_after_minutes=policy.stale_after_minutes,
        )
        if _is_candidate_stale(candidate):
            stale_broker_orders.append(candidate)
        else:
            aligned_open_orders.append(candidate)

    cancel_candidates = tuple(
        list(stale_ledger_orders if policy.cancel_stale_ledger_orders else ())
        + list(stale_broker_orders if policy.cancel_orphan_broker_orders else ())
    )
    return OrderMaintenancePlan(
        run_id=None,
        session_date=session_date,
        policy=policy,
        recovery_plan=recovery_plan,
        aligned_open_orders=tuple(aligned_open_orders),
        stale_ledger_orders=tuple(stale_ledger_orders),
        stale_broker_orders=tuple(stale_broker_orders),
        cancel_candidates=cancel_candidates,
    )


def _candidate_from_pair(
    *,
    ledger_order: OrderRecord | None,
    broker_order: BrokerOrderSnapshot | None,
    source: str,
    session_date: date | None,
    as_of_utc: datetime,
    stale_after_minutes: int,
) -> MaintenanceCandidate:
    order = ledger_order or broker_order
    assert order is not None
    submitted_at_utc = ledger_order.submitted_at_utc if ledger_order is not None else broker_order.submitted_at_utc
    order_session_date = ledger_order.session_date if ledger_order is not None else None
    age_minutes = _age_minutes(submitted_at_utc, as_of_utc)
    stale_reasons: list[str] = []
    if session_date is not None and order_session_date is not None and order_session_date < session_date:
        stale_reasons.append("prior_session")
    if age_minutes is not None and age_minutes >= stale_after_minutes:
        stale_reasons.append(f"age>={stale_after_minutes}")
    if ledger_order is None:
        stale_reasons.append("orphan_broker_order")
    if ledger_order is not None and broker_order is None:
        stale_reasons.append("missing_broker_order")
    return MaintenanceCandidate(
        order_id=order.order_id,
        symbol=order.symbol,
        client_order_id=order.client_order_id if ledger_order is not None else broker_order.client_order_id,
        source=source,
        stale_reasons=tuple(stale_reasons),
        age_minutes=age_minutes,
        ledger_status=ledger_order.status if ledger_order is not None else None,
        broker_status=broker_order.status if broker_order is not None else None,
        session_date=order_session_date,
        submitted_at_utc=submitted_at_utc,
    )


def _is_candidate_stale(candidate: MaintenanceCandidate) -> bool:
    return bool(candidate.stale_reasons)


def _resolve_run_context(ledger: LocalLedger, run_id: str | None) -> tuple[str | None, RunManifestRecord | None]:
    if run_id is not None:
        return run_id, ledger.get_run_manifest(run_id)
    manifests = ledger.list_run_manifests()
    if manifests:
        return manifests[0].run_id, manifests[0]
    orders = ledger.list_orders()
    if orders and orders[0].run_id is not None:
        return orders[0].run_id, ledger.get_run_manifest(orders[0].run_id) if orders[0].run_id is not None else None
    return None, None


def _build_summary_notes(manifest: RunManifestRecord | None, plan: OrderMaintenancePlan) -> tuple[str, ...]:
    notes: list[str] = []
    if manifest is None:
        notes.append("manifest_missing")
    if plan.cancel_candidates:
        notes.append("cancel_plan_available")
    if plan.stale_ledger_orders and not plan.stale_broker_orders:
        notes.append("ledger_only_stale_orders")
    if plan.stale_broker_orders and not plan.stale_ledger_orders:
        notes.append("broker_only_stale_orders")
    return tuple(notes)


def _manifest_to_dict(manifest: RunManifestRecord | None) -> dict[str, Any]:
    if manifest is None:
        return {}
    return {
        "run_id": manifest.run_id,
        "session_date": manifest.session_date.isoformat(),
        "strategy_name": manifest.strategy_name,
        "model_name": manifest.model_name,
        "generated_at_utc": manifest.generated_at_utc.astimezone(timezone.utc).isoformat(),
        "client_order_id_prefix": manifest.client_order_id_prefix,
        "dry_run": manifest.dry_run,
        "data_snapshot": dict(manifest.data_snapshot),
        "risk_policy": dict(manifest.risk_policy),
        "execution_policy": dict(manifest.execution_policy),
        "meta": dict(manifest.meta),
    }


def _recovery_plan_to_dict(plan: RecoveryPlan) -> dict[str, Any]:
    return {
        "orphan_count": plan.orphan_count,
        "stale_count": plan.stale_count,
        "aligned_count": plan.aligned_count,
        "orphan_broker_orders": [candidate_to_recovery_dict(candidate) for candidate in plan.orphan_broker_orders],
        "stale_ledger_orders": [candidate_to_recovery_dict(candidate) for candidate in plan.stale_ledger_orders],
        "aligned_open_orders": [candidate_to_recovery_dict(candidate) for candidate in plan.aligned_open_orders],
    }


def candidate_to_recovery_dict(candidate) -> dict[str, Any]:
    return {
        "order_id": candidate.order_id,
        "symbol": candidate.symbol,
        "client_order_id": candidate.client_order_id,
        "ledger_status": candidate.ledger_status,
        "broker_status": candidate.broker_status,
        "ledger_filled_quantity": candidate.ledger_filled_quantity,
        "broker_filled_quantity": candidate.broker_filled_quantity,
    }


def _age_minutes(submitted_at_utc: datetime | None, as_of_utc: datetime) -> float | None:
    if submitted_at_utc is None:
        return None
    return (as_of_utc - _ensure_utc(submitted_at_utc)).total_seconds() / 60.0


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _ledger_sort_key(order: OrderRecord) -> tuple[str, str]:
    return order.symbol, order.order_id


def _broker_sort_key(order: BrokerOrderSnapshot) -> tuple[str, str]:
    return order.symbol, order.order_id
