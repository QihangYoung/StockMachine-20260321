from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
import re

from stockmachine.state.ledger import LocalLedger
from stockmachine.state.models import FillRecord, OrderRecord


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _get_value(payload: Mapping[str, Any] | object, *names: str, default: Any = None) -> Any:
    if isinstance(payload, Mapping):
        for name in names:
            if name in payload:
                return payload[name]
        return default
    for name in names:
        if hasattr(payload, name):
            return getattr(payload, name)
    return default


def _coerce_datetime(value: Any, default: datetime | None = None) -> datetime:
    if value is None:
        if default is None:
            raise ValueError("datetime value is required")
        return default
    if isinstance(value, datetime):
        return _ensure_utc(value)
    parsed = datetime.fromisoformat(_trim_fractional_seconds(str(value)))
    return _ensure_utc(parsed)


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    return float(value)


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _trim_fractional_seconds(value: str) -> str:
    match = re.match(r"^(.*?\.\d{6})\d+(.*)$", value)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return value


@dataclass(slots=True, frozen=True)
class BrokerOrderSnapshot:
    """Generic broker order payload normalized for reconciliation."""

    order_id: str
    client_order_id: str | None
    symbol: str
    side: str
    status: str
    quantity: int
    filled_quantity: int = 0
    order_type: str | None = None
    limit_price: float | None = None
    submitted_at_utc: datetime = field(default_factory=_default_now)
    updated_at_utc: datetime = field(default_factory=_default_now)
    avg_fill_price: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | object) -> BrokerOrderSnapshot:
        return cls(
            order_id=str(_get_value(payload, "id", "order_id")),
            client_order_id=_get_value(payload, "client_order_id", "clientOrderId"),
            symbol=str(_get_value(payload, "symbol")),
            side=str(_get_value(payload, "side")),
            status=str(_get_value(payload, "status", default="unknown")),
            quantity=_coerce_int(_get_value(payload, "qty", "quantity")),
            filled_quantity=_coerce_int(_get_value(payload, "filled_qty", "filled_quantity"), default=0),
            order_type=_get_value(payload, "type", "order_type"),
            limit_price=_coerce_float(_get_value(payload, "limit_price", "limitPrice")),
            submitted_at_utc=_coerce_datetime(
                _get_value(payload, "submitted_at", "submitted_at_utc", "created_at", "created_at_utc"),
                default=_default_now(),
            ),
            updated_at_utc=_coerce_datetime(
                _get_value(payload, "updated_at", "updated_at_utc", "last_update_at"),
                default=_default_now(),
            ),
            avg_fill_price=_coerce_float(
                _get_value(payload, "avg_fill_price", "average_fill_price", "filled_avg_price")
            ),
            raw=dict(payload) if isinstance(payload, Mapping) else {},
        )


@dataclass(slots=True, frozen=True)
class BrokerFillSnapshot:
    """Generic broker fill payload normalized for reconciliation."""

    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    filled_at_utc: datetime = field(default_factory=_default_now)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | object) -> BrokerFillSnapshot:
        return cls(
            fill_id=str(_get_value(payload, "id", "fill_id")),
            order_id=str(_get_value(payload, "order_id", "orderId")),
            symbol=str(_get_value(payload, "symbol")),
            side=str(_get_value(payload, "side")),
            quantity=_coerce_int(_get_value(payload, "qty", "quantity")),
            price=float(_get_value(payload, "price")),
            filled_at_utc=_coerce_datetime(
                _get_value(payload, "filled_at", "filled_at_utc", "transaction_time", "transaction_time_utc"),
                default=_default_now(),
            ),
            raw=dict(payload) if isinstance(payload, Mapping) else {},
        )


@dataclass(slots=True, frozen=True)
class OrderReconciliationChange:
    """Single-order status transition produced by reconciliation."""

    order_id: str
    symbol: str
    previous_status: str | None
    current_status: str
    previous_filled_quantity: int
    current_filled_quantity: int
    fill_delta: int
    is_new: bool = False


@dataclass(slots=True, frozen=True)
class ReconciliationResult:
    """Summary returned by one polling reconciliation pass."""

    changes: tuple[OrderReconciliationChange, ...] = ()
    created_orders: int = 0
    updated_orders: int = 0
    fill_events_created: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)


class PollingOrderReconciler:
    """Poll broker order snapshots and keep the local ledger in sync."""

    def __init__(self, ledger: LocalLedger) -> None:
        self.ledger = ledger

    def reconcile_orders(
        self,
        broker_orders: Sequence[Mapping[str, Any] | BrokerOrderSnapshot | object],
        broker_fills: Sequence[Mapping[str, Any] | BrokerFillSnapshot | object] | None = None,
    ) -> ReconciliationResult:
        changes: list[OrderReconciliationChange] = []
        created_orders = 0
        updated_orders = 0
        fill_events_created = 0
        status_counts: dict[str, int] = {}

        for payload in broker_orders:
            snapshot = payload if isinstance(payload, BrokerOrderSnapshot) else BrokerOrderSnapshot.from_payload(payload)
            local = self.ledger.get_order(snapshot.order_id)
            if local is None and snapshot.client_order_id is not None:
                local = self.ledger.get_order_by_client_order_id(snapshot.client_order_id)

            previous_status = local.status if local is not None else None
            previous_filled_quantity = local.filled_quantity if local is not None else 0
            fill_delta = max(snapshot.filled_quantity - previous_filled_quantity, 0)
            is_new = local is None

            if local is None:
                created_orders += 1
                submitted_at_utc = snapshot.submitted_at_utc
            else:
                updated_orders += 1
                submitted_at_utc = local.submitted_at_utc

            self.ledger.upsert_order(
                OrderRecord(
                    order_id=snapshot.order_id,
                    run_id=local.run_id if local is not None else None,
                    session_date=local.session_date if local is not None else None,
                    client_order_id=snapshot.client_order_id,
                    symbol=snapshot.symbol,
                    side=snapshot.side,
                    quantity=snapshot.quantity,
                    order_type=snapshot.order_type or (local.order_type if local is not None else "unknown"),
                    limit_price=(
                        snapshot.limit_price
                        if snapshot.limit_price is not None
                        else (local.limit_price if local is not None else None)
                    ),
                    status=snapshot.status,
                    filled_quantity=snapshot.filled_quantity,
                    avg_fill_price=(
                        snapshot.avg_fill_price
                        if snapshot.avg_fill_price is not None
                        else (local.avg_fill_price if local is not None else None)
                    ),
                    submitted_at_utc=submitted_at_utc,
                    updated_at_utc=snapshot.updated_at_utc,
                    broker_payload=snapshot.raw,
                )
            )

            status_counts[snapshot.status] = status_counts.get(snapshot.status, 0) + 1
            changes.append(
                OrderReconciliationChange(
                    order_id=snapshot.order_id,
                    symbol=snapshot.symbol,
                    previous_status=previous_status,
                    current_status=snapshot.status,
                    previous_filled_quantity=previous_filled_quantity,
                    current_filled_quantity=snapshot.filled_quantity,
                    fill_delta=fill_delta,
                    is_new=is_new,
                )
            )

            if fill_delta > 0:
                fill_price = snapshot.avg_fill_price
                if fill_price is not None:
                    fill_events_created += 1
                    self.ledger.record_fill(
                        FillRecord(
                            fill_id=f"{snapshot.order_id}:{snapshot.filled_quantity}:{snapshot.updated_at_utc.isoformat()}",
                            order_id=snapshot.order_id,
                            run_id=local.run_id if local is not None else None,
                            symbol=snapshot.symbol,
                            side=snapshot.side,
                            quantity=fill_delta,
                            price=fill_price,
                            filled_at_utc=snapshot.updated_at_utc,
                            broker_payload=snapshot.raw,
                        )
                    )

        if broker_fills:
            for payload in broker_fills:
                fill = payload if isinstance(payload, BrokerFillSnapshot) else BrokerFillSnapshot.from_payload(payload)
                if self.ledger.list_fills(order_id=fill.order_id):
                    continue
                local = self.ledger.get_order(fill.order_id)
                self.ledger.record_fill(
                    FillRecord(
                        fill_id=fill.fill_id,
                        order_id=fill.order_id,
                        run_id=local.run_id if local is not None else None,
                        symbol=fill.symbol,
                        side=fill.side,
                        quantity=fill.quantity,
                        price=fill.price,
                        filled_at_utc=fill.filled_at_utc,
                        broker_payload=fill.raw,
                    )
                )
                fill_events_created += 1

        return ReconciliationResult(
            changes=tuple(changes),
            created_orders=created_orders,
            updated_orders=updated_orders,
            fill_events_created=fill_events_created,
            status_counts=status_counts,
        )
