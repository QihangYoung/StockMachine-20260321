from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from stockmachine.live.reconciler import BrokerOrderSnapshot, PollingOrderReconciler, ReconciliationResult
from stockmachine.state.ledger import LocalLedger
from stockmachine.state.models import OrderRecord


class RecoveryReconciler(Protocol):
    """Minimal reconciliation surface used by recovery helpers."""

    def reconcile_orders(
        self,
        broker_orders: Sequence[object],
        broker_fills: Sequence[object] | None = None,
    ) -> ReconciliationResult:
        """Reconcile broker state into the ledger."""


@dataclass(slots=True, frozen=True)
class RecoveryOrderLink:
    """A matched open order observed in both the ledger and broker."""

    order_id: str
    symbol: str
    client_order_id: str | None
    ledger_status: str
    broker_status: str
    ledger_filled_quantity: int
    broker_filled_quantity: int


@dataclass(slots=True, frozen=True)
class RecoveryPlan:
    """Plan describing how the current open-order state should be recovered."""

    orphan_broker_orders: tuple[RecoveryOrderLink, ...] = ()
    stale_ledger_orders: tuple[RecoveryOrderLink, ...] = ()
    aligned_open_orders: tuple[RecoveryOrderLink, ...] = ()

    @property
    def orphan_count(self) -> int:
        return len(self.orphan_broker_orders)

    @property
    def stale_count(self) -> int:
        return len(self.stale_ledger_orders)

    @property
    def aligned_count(self) -> int:
        return len(self.aligned_open_orders)


@dataclass(slots=True, frozen=True)
class RecoverySyncResult:
    """A recovery plan paired with a single reconciliation pass."""

    plan: RecoveryPlan
    reconciliation: ReconciliationResult


def build_recovery_plan(
    ledger_open_orders: Sequence[OrderRecord],
    broker_open_orders: Sequence[BrokerOrderSnapshot | object],
) -> RecoveryPlan:
    """Classify ledger and broker open orders into recovery buckets."""

    broker_snapshots = tuple(
        snapshot if isinstance(snapshot, BrokerOrderSnapshot) else BrokerOrderSnapshot.from_payload(snapshot)
        for snapshot in broker_open_orders
    )
    ledger_by_order_id = {order.order_id: order for order in ledger_open_orders}
    ledger_by_client_order_id = {
        order.client_order_id: order
        for order in ledger_open_orders
        if order.client_order_id is not None
    }

    used_ledger_order_ids: set[str] = set()
    orphan_broker_orders: list[RecoveryOrderLink] = []
    aligned_open_orders: list[RecoveryOrderLink] = []

    for broker_order in sorted(broker_snapshots, key=_broker_sort_key):
        ledger_order = ledger_by_order_id.get(broker_order.order_id)
        if ledger_order is None and broker_order.client_order_id is not None:
            ledger_order = ledger_by_client_order_id.get(broker_order.client_order_id)
        if ledger_order is None or ledger_order.order_id in used_ledger_order_ids:
            orphan_broker_orders.append(_link_from_broker(broker_order))
            continue
        used_ledger_order_ids.add(ledger_order.order_id)
        aligned_open_orders.append(_link_from_pair(ledger_order, broker_order))

    stale_ledger_orders = [
        _link_from_ledger(ledger_order)
        for ledger_order in sorted(ledger_open_orders, key=_ledger_sort_key)
        if ledger_order.order_id not in used_ledger_order_ids
    ]

    return RecoveryPlan(
        orphan_broker_orders=tuple(orphan_broker_orders),
        stale_ledger_orders=tuple(stale_ledger_orders),
        aligned_open_orders=tuple(aligned_open_orders),
    )


def recover_open_orders(
    ledger: LocalLedger,
    broker_open_orders: Sequence[BrokerOrderSnapshot | object],
    *,
    broker_fills: Sequence[object] | None = None,
    reconciler: RecoveryReconciler | None = None,
) -> RecoverySyncResult:
    """Build a recovery plan and reconcile the broker state into the local ledger."""

    plan = build_recovery_plan(ledger.list_open_orders(), broker_open_orders)
    order_reconciler = reconciler or PollingOrderReconciler(ledger)
    reconciliation = order_reconciler.reconcile_orders(broker_open_orders, broker_fills=broker_fills)
    return RecoverySyncResult(plan=plan, reconciliation=reconciliation)


def _link_from_pair(ledger_order: OrderRecord, broker_order: BrokerOrderSnapshot) -> RecoveryOrderLink:
    return RecoveryOrderLink(
        order_id=ledger_order.order_id,
        symbol=ledger_order.symbol,
        client_order_id=ledger_order.client_order_id or broker_order.client_order_id,
        ledger_status=ledger_order.status,
        broker_status=broker_order.status,
        ledger_filled_quantity=ledger_order.filled_quantity,
        broker_filled_quantity=broker_order.filled_quantity,
    )


def _link_from_ledger(ledger_order: OrderRecord) -> RecoveryOrderLink:
    return RecoveryOrderLink(
        order_id=ledger_order.order_id,
        symbol=ledger_order.symbol,
        client_order_id=ledger_order.client_order_id,
        ledger_status=ledger_order.status,
        broker_status="missing",
        ledger_filled_quantity=ledger_order.filled_quantity,
        broker_filled_quantity=0,
    )


def _link_from_broker(broker_order: BrokerOrderSnapshot) -> RecoveryOrderLink:
    return RecoveryOrderLink(
        order_id=broker_order.order_id,
        symbol=broker_order.symbol,
        client_order_id=broker_order.client_order_id,
        ledger_status="missing",
        broker_status=broker_order.status,
        ledger_filled_quantity=0,
        broker_filled_quantity=broker_order.filled_quantity,
    )


def _ledger_sort_key(order: OrderRecord) -> tuple[str, str]:
    return order.symbol, order.order_id


def _broker_sort_key(order: BrokerOrderSnapshot) -> tuple[str, str]:
    return order.symbol, order.order_id
