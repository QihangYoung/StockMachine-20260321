from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from stockmachine.live.reconciler import PollingOrderReconciler, ReconciliationResult
from stockmachine.state import FillAuditRecord, LocalLedger, OrderDecisionRecord, OrderRecord


class OrderSnapshotProvider(Protocol):
    """Minimal broker surface used for historical order-status backfill."""

    def get_order(self, order_id: str) -> Mapping[str, Any] | object:
        """Return one broker order snapshot by id."""


@dataclass(slots=True, frozen=True)
class OrderStatusBackfillError:
    """One failed broker lookup during historical status backfill."""

    order_id: str
    symbol: str
    reason: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "reason": self.reason,
            "message": self.message,
        }


@dataclass(slots=True, frozen=True)
class OrderStatusBackfillResult:
    """Summary returned after reconciling broker history into the local ledger."""

    requested_orders: int = 0
    matched_snapshots: int = 0
    fill_audits_created: int = 0
    reconciliation: ReconciliationResult = field(default_factory=ReconciliationResult)
    errors: tuple[OrderStatusBackfillError, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_orders": self.requested_orders,
            "matched_snapshots": self.matched_snapshots,
            "fill_audits_created": self.fill_audits_created,
            "reconciliation": {
                "created_orders": self.reconciliation.created_orders,
                "updated_orders": self.reconciliation.updated_orders,
                "fill_events_created": self.reconciliation.fill_events_created,
                "status_counts": dict(self.reconciliation.status_counts),
            },
            "errors": [error.to_dict() for error in self.errors],
        }


def backfill_order_statuses(
    ledger: LocalLedger,
    order_status_provider: OrderSnapshotProvider,
    *,
    candidate_orders: Sequence[OrderRecord],
    reconciler: PollingOrderReconciler | None = None,
) -> OrderStatusBackfillResult:
    """Refresh known ledger orders from broker history and infer missing fills."""

    if not candidate_orders:
        return OrderStatusBackfillResult()

    snapshots: list[Mapping[str, Any] | object] = []
    errors: list[OrderStatusBackfillError] = []

    for order in candidate_orders:
        try:
            snapshots.append(order_status_provider.get_order(order.order_id))
        except Exception as exc:  # pragma: no cover - defensive for broker runtime
            errors.append(
                OrderStatusBackfillError(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    reason=type(exc).__name__,
                    message=str(exc),
                )
            )

    order_reconciler = reconciler or PollingOrderReconciler(ledger)
    reconciliation = order_reconciler.reconcile_orders(snapshots)
    fill_audits_created = record_fill_audits_for_orders(
        ledger,
        candidate_orders=candidate_orders,
    )
    return OrderStatusBackfillResult(
        requested_orders=len(candidate_orders),
        matched_snapshots=len(snapshots),
        fill_audits_created=fill_audits_created,
        reconciliation=reconciliation,
        errors=tuple(errors),
    )


def record_fill_audits_for_orders(
    ledger: LocalLedger,
    *,
    candidate_orders: Sequence[OrderRecord],
    context_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> int:
    """Create missing fill-audit rows for orders that now have fill records."""

    if not candidate_orders:
        return 0

    contexts = _build_fill_audit_contexts(ledger, candidate_orders)
    if context_overrides:
        for order_id, payload in context_overrides.items():
            contexts[order_id] = {
                **contexts.get(order_id, {}),
                **dict(payload),
            }
    existing_audit_ids = {
        audit.audit_id
        for order in candidate_orders
        for audit in ledger.list_fill_audits(order_id=order.order_id)
    }
    created = 0

    for order in candidate_orders:
        order_record = ledger.get_order(order.order_id)
        if order_record is None:
            continue
        context = contexts.get(order.order_id, {})
        fills = ledger.list_fills(order_id=order.order_id, run_id=order_record.run_id)
        for fill in fills:
            audit_id = f"audit:{fill.fill_id}"
            if audit_id in existing_audit_ids:
                continue
            expected_price = _coerce_optional_float(context.get("expected_price"))
            slippage = _estimate_fill_slippage(
                side=str(context.get("side") or fill.side),
                expected_price=expected_price,
                fill_price=float(fill.price),
            )
            fee = _coerce_optional_float(context.get("order_meta", {}).get("fee_estimate"))
            ledger.record_fill_audit(
                FillAuditRecord(
                    audit_id=audit_id,
                    order_id=fill.order_id,
                    run_id=order_record.run_id,
                    session_date=order_record.session_date,
                    client_order_id=str(context.get("client_order_id") or order_record.client_order_id or ""),
                    symbol=fill.symbol,
                    side=fill.side,
                    quantity=fill.quantity,
                    expected_price=expected_price,
                    price=fill.price,
                    slippage=slippage,
                    fee=fee,
                    filled_at_utc=fill.filled_at_utc,
                    meta={
                        "broker_order_status": order_record.status,
                        "backfill_source": "broker_order_status",
                    },
                )
            )
            existing_audit_ids.add(audit_id)
            created += 1

    return created


def _build_fill_audit_contexts(
    ledger: LocalLedger,
    candidate_orders: Sequence[OrderRecord],
) -> dict[str, dict[str, Any]]:
    decisions_by_client_order_id = _latest_decisions_by_client_order_id(ledger, candidate_orders)
    contexts: dict[str, dict[str, Any]] = {}
    for order in candidate_orders:
        decision = decisions_by_client_order_id.get(order.client_order_id or "")
        contexts[order.order_id] = {
            "client_order_id": order.client_order_id,
            "expected_price": decision.decision_price if decision is not None else None,
            "side": decision.side if decision is not None else order.side,
            "order_meta": dict(decision.meta) if decision is not None else {},
        }
    return contexts


def _latest_decisions_by_client_order_id(
    ledger: LocalLedger,
    candidate_orders: Sequence[OrderRecord],
) -> dict[str, OrderDecisionRecord]:
    decisions_by_client_order_id: dict[str, OrderDecisionRecord] = {}
    run_ids = sorted({order.run_id for order in candidate_orders if order.run_id})
    if run_ids:
        decisions = [
            decision
            for run_id in run_ids
            for decision in ledger.list_order_decisions(run_id=run_id)
        ]
    else:
        decisions = ledger.list_order_decisions()

    for decision in decisions:
        if not decision.client_order_id:
            continue
        decisions_by_client_order_id[decision.client_order_id] = decision
    return decisions_by_client_order_id


def _estimate_fill_slippage(
    *,
    side: str,
    expected_price: float | None,
    fill_price: float,
) -> float | None:
    if expected_price is None:
        return None
    normalized_side = side.upper()
    if normalized_side == "SELL":
        return float(expected_price) - float(fill_price)
    return float(fill_price) - float(expected_price)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
