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
