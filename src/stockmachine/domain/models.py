from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True, frozen=True)
class Signal:
    """Model output consumed by portfolio and risk layers."""

    symbol: str
    side: str
    score: float
    confidence: float
    horizon_bars: int
    timestamp: datetime
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TargetPosition:
    """White-box portfolio decision after risk checks."""

    symbol: str
    target_weight: float
    max_weight: float
    reason: str
    timestamp: datetime
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OrderIntent:
    """Execution plan before orders are sent to a broker."""

    symbol: str
    side: str
    quantity: int
    order_type: str
    limit_price: float | None
    timestamp: datetime
    meta: dict[str, Any] = field(default_factory=dict)
