from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence
import re

from stockmachine.domain.models import OrderIntent


CLIENT_ORDER_ID_PREFIX = "smk"
CLIENT_ORDER_ID_PATTERN = re.compile(
    r"^(?P<prefix>[a-z0-9]+)-(?P<session>\d{8})-(?P<symbol>[A-Z0-9._-]+)-(?P<side>BUY|SELL)-(?P<sequence>\d{4})-(?P<run_tag>[a-z0-9]+)$"
)


def _sanitize_order_tag(value: str | None, *, fallback: str) -> str:
    if not value:
        return fallback
    sanitized = re.sub(r"[^a-z0-9]", "", value.lower())
    return sanitized or fallback


def build_client_order_id(
    *,
    session_date: date,
    symbol: str,
    side: str,
    sequence: int,
    run_id: str | None = None,
    prefix: str = CLIENT_ORDER_ID_PREFIX,
) -> str:
    """Build a broker-safe client order id for paper/live order tracking."""

    run_tag = _sanitize_order_tag(run_id, fallback="run")[:8]
    return f"{prefix}-{session_date:%Y%m%d}-{symbol.upper()}-{side.upper()}-{sequence:04d}-{run_tag}"


def validate_client_order_id(client_order_id: str, *, prefix: str = CLIENT_ORDER_ID_PREFIX, max_length: int = 48) -> bool:
    """Check whether a client order id follows the house convention."""

    if not client_order_id or len(client_order_id) > max_length:
        return False
    match = CLIENT_ORDER_ID_PATTERN.match(client_order_id)
    return bool(match and match.group("prefix") == prefix)


def extract_client_order_id(order: OrderIntent) -> str | None:
    value = order.meta.get("client_order_id")
    if value is None:
        value = order.meta.get("clientOrderId")
    if value is None:
        return None
    client_order_id = str(value).strip()
    return client_order_id or None


def _payload_value(payload: Mapping[str, Any] | object, *names: str, default: Any = None) -> Any:
    if isinstance(payload, Mapping):
        for name in names:
            if name in payload:
                return payload[name]
        return default
    for name in names:
        if hasattr(payload, name):
            return getattr(payload, name)
    return default


@dataclass(slots=True, frozen=True)
class OrderValidationIssue:
    """One broker-aware pre-trade validation finding."""

    code: str
    message: str
    symbol: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OrderValidationResult:
    """Approved orders plus validation findings."""

    approved_orders: tuple[OrderIntent, ...]
    blocked_orders: tuple[OrderIntent, ...]
    issues: tuple[OrderValidationIssue, ...]
    estimated_notional: float
    open_order_symbols: tuple[str, ...] = ()


@dataclass(slots=True)
class BrokerAwareOrderRiskPolicy:
    """Simple broker-aware gate for the first Alpaca paper demo."""

    require_market_open: bool = False
    block_duplicate_symbols: bool = True
    require_client_order_id: bool = False
    min_buying_power_buffer: float = 0.0
    max_order_notional: float | None = None
    max_total_notional: float | None = None
    max_total_orders: int | None = None

    def validate(
        self,
        *,
        orders: Sequence[OrderIntent],
        account_sync: object | None = None,
        open_orders: Sequence[Mapping[str, Any] | object] = (),
    ) -> OrderValidationResult:
        if not orders:
            return OrderValidationResult((), (), (), 0.0)

        issues: list[OrderValidationIssue] = []
        approved: list[OrderIntent] = []
        blocked: list[OrderIntent] = []

        broker_account = getattr(account_sync, "broker_account", None)
        clock = getattr(account_sync, "clock", None)

        if broker_account is not None:
            status = str(getattr(broker_account, "status", "")).lower()
            if status and status != "active":
                issues.append(
                    OrderValidationIssue(
                        code="inactive_account",
                        message="Broker account is not active.",
                        details={"status": status},
                    )
                )
                return OrderValidationResult((), tuple(orders), tuple(issues), 0.0)
            if bool(getattr(broker_account, "trading_blocked", False)):
                issues.append(
                    OrderValidationIssue(
                        code="trading_blocked",
                        message="Broker account is trading blocked.",
                    )
                )
                return OrderValidationResult((), tuple(orders), tuple(issues), 0.0)
            if bool(getattr(broker_account, "account_blocked", False)):
                issues.append(
                    OrderValidationIssue(
                        code="account_blocked",
                        message="Broker account is account blocked.",
                    )
                )
                return OrderValidationResult((), tuple(orders), tuple(issues), 0.0)

        if self.require_market_open and clock is not None and not bool(getattr(clock, "is_open", False)):
            issues.append(
                OrderValidationIssue(
                    code="market_closed",
                    message="Market is closed and the policy requires an open market.",
                )
            )
            return OrderValidationResult((), tuple(orders), tuple(issues), 0.0)

        open_order_symbols = tuple(
            sorted(
                {
                    str(_payload_value(order, "symbol", default=""))
                    for order in open_orders
                    if str(_payload_value(order, "status", default="")).lower()
                    not in {"filled", "canceled", "cancelled", "rejected", "expired"}
                }
            )
        )
        open_order_pairs = {
            (
                str(_payload_value(order, "symbol", default="")).upper(),
                str(_payload_value(order, "side", default="")).upper(),
            )
            for order in open_orders
            if str(_payload_value(order, "status", default="")).lower()
            not in {"filled", "canceled", "cancelled", "rejected", "expired"}
        }

        buying_power = float(getattr(broker_account, "buying_power", 0.0)) if broker_account is not None else float("inf")
        buying_power_budget = max(0.0, buying_power - self.min_buying_power_buffer)
        running_notional = 0.0
        running_orders = 0

        for order in orders:
            symbol = order.symbol.upper()
            side = order.side.upper()
            client_order_id = extract_client_order_id(order)

            if client_order_id is None:
                if self.require_client_order_id:
                    issues.append(
                        OrderValidationIssue(
                            code="missing_client_order_id",
                            message="Order is missing a client order id.",
                            symbol=symbol,
                        )
                    )
                    blocked.append(order)
                    continue
            elif not validate_client_order_id(client_order_id):
                if self.require_client_order_id:
                    issues.append(
                        OrderValidationIssue(
                            code="invalid_client_order_id",
                            message="Order client order id does not follow the house convention.",
                            symbol=symbol,
                            details={"client_order_id": client_order_id},
                        )
                    )
                    blocked.append(order)
                    continue

            if self.block_duplicate_symbols and (symbol, side) in open_order_pairs:
                issues.append(
                    OrderValidationIssue(
                        code="duplicate_open_order",
                        message="Open broker order already exists for this symbol and side.",
                        symbol=symbol,
                    )
                )
                blocked.append(order)
                continue

            reference_price = self._estimate_reference_price(order)
            if reference_price is None or reference_price <= 0:
                issues.append(
                    OrderValidationIssue(
                        code="missing_reference_price",
                        message="Order is missing a usable reference price for buying-power estimation.",
                        symbol=symbol,
                    )
                )
                blocked.append(order)
                continue

            estimated_notional = float(order.quantity) * reference_price
            is_sell = side == "SELL"
            if not is_sell and self.max_order_notional is not None and estimated_notional > self.max_order_notional:
                issues.append(
                    OrderValidationIssue(
                        code="max_order_notional_exceeded",
                        message="Order would exceed the configured max order notional.",
                        symbol=symbol,
                        details={
                            "estimated_notional": estimated_notional,
                            "max_order_notional": self.max_order_notional,
                        },
                    )
                )
                blocked.append(order)
                continue

            if not is_sell and self.max_total_orders is not None and running_orders + 1 > self.max_total_orders:
                issues.append(
                    OrderValidationIssue(
                        code="max_total_orders_exceeded",
                        message="Order would exceed the configured max total orders cap.",
                        symbol=symbol,
                        details={
                            "max_total_orders": self.max_total_orders,
                            "current_approved_orders": running_orders,
                        },
                    )
                )
                blocked.append(order)
                continue

            if not is_sell and self.max_total_notional is not None and running_notional + estimated_notional > self.max_total_notional:
                issues.append(
                    OrderValidationIssue(
                        code="max_total_notional_exceeded",
                        message="Order would exceed the configured max total notional cap.",
                        symbol=symbol,
                        details={
                            "estimated_notional": estimated_notional,
                            "current_total_notional": running_notional,
                            "max_total_notional": self.max_total_notional,
                        },
                    )
                )
                blocked.append(order)
                continue

            if not is_sell and estimated_notional > buying_power_budget:
                issues.append(
                    OrderValidationIssue(
                        code="insufficient_buying_power",
                        message="Order would exceed available buying power.",
                        symbol=symbol,
                        details={
                            "estimated_notional": estimated_notional,
                            "available_buying_power": buying_power_budget,
                        },
                    )
                )
                blocked.append(order)
                continue

            approved.append(order)
            if is_sell:
                buying_power_budget += estimated_notional
            else:
                buying_power_budget = max(0.0, buying_power_budget - estimated_notional)
                running_notional += estimated_notional
                running_orders += 1

        return OrderValidationResult(
            approved_orders=tuple(approved),
            blocked_orders=tuple(blocked),
            issues=tuple(issues),
            estimated_notional=running_notional,
            open_order_symbols=open_order_symbols,
        )

    def _estimate_reference_price(self, order: OrderIntent) -> float | None:
        if order.limit_price is not None and order.limit_price > 0:
            return float(order.limit_price)

        for key in ("reference_price", "close", "open", "last_price"):
            value = order.meta.get(key)
            if value is None:
                continue
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            if price > 0:
                return price
        return None
