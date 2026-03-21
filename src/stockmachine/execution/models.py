from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True, frozen=True)
class BrokerClock:
    """Minimal market-clock snapshot from a broker."""

    timestamp: datetime
    is_open: bool
    next_open: datetime | None = None
    next_close: datetime | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BrokerAccount:
    """Normalized broker account snapshot used by paper execution."""

    account_id: str
    status: str
    currency: str
    cash: float
    equity: float
    buying_power: float
    day_trade_buying_power: float | None = None
    multiplier: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BrokerPosition:
    """Normalized broker position snapshot."""

    symbol: str
    quantity: float
    side: str
    market_value: float
    avg_entry_price: float | None = None
    current_price: float | None = None
    unrealized_pl: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BrokerOrder:
    """Normalized broker order snapshot."""

    order_id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    time_in_force: str
    quantity: float
    notional: float | None
    limit_price: float | None
    stop_price: float | None
    status: str
    filled_quantity: float = 0.0
    filled_avg_price: float | None = None
    submitted_at: datetime | None = None
    updated_at: datetime | None = None
    filled_at: datetime | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class FillEvent:
    """One normalized execution fill event."""

    order_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    filled_at: datetime
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ExecutionReport:
    """Compact runtime summary for one paper-trading run."""

    run_id: str
    submitted_orders: int
    open_orders: int
    filled_orders: int
    canceled_orders: int
    rejected_orders: int
    notes: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SubmissionRetryRecord:
    """Structured summary for one order submission with an optional retry."""

    symbol: str
    side: str
    client_order_id: str | None
    original_quantity: int
    final_quantity: int
    retry_used: bool
    retry_reason: str | None = None
    retry_scale: float | None = None
    initial_error: str | None = None
    final_error: str | None = None
    final_order_id: str | None = None
    final_status: str | None = None
    initial_notional: float | None = None
    final_notional: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "client_order_id": self.client_order_id,
            "original_quantity": self.original_quantity,
            "final_quantity": self.final_quantity,
            "retry_used": self.retry_used,
            "retry_reason": self.retry_reason,
            "retry_scale": self.retry_scale,
            "initial_error": self.initial_error,
            "final_error": self.final_error,
            "final_order_id": self.final_order_id,
            "final_status": self.final_status,
            "initial_notional": self.initial_notional,
            "final_notional": self.final_notional,
            "meta": dict(self.meta),
        }


@dataclass(slots=True, frozen=True)
class SubmissionRetryReport:
    """Batch summary for submission attempts and retry behavior."""

    attempted_orders: int
    submitted_orders: int
    retried_orders: int
    failed_orders: int
    records: tuple[SubmissionRetryRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted_orders": self.attempted_orders,
            "submitted_orders": self.submitted_orders,
            "retried_orders": self.retried_orders,
            "failed_orders": self.failed_orders,
            "records": [record.to_dict() for record in self.records],
        }
