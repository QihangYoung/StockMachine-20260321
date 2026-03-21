from __future__ import annotations

from typing import Protocol, Sequence

from stockmachine.domain.models import OrderIntent

from .models import BrokerAccount, BrokerClock, BrokerOrder, BrokerPosition


class BrokerAdapter(Protocol):
    """Runtime broker contract for paper and future live execution."""

    def get_account(self) -> BrokerAccount:
        """Return the latest broker account snapshot."""

    def get_clock(self) -> BrokerClock:
        """Return the latest market-clock state."""

    def list_positions(self) -> Sequence[BrokerPosition]:
        """Return the currently open positions."""

    def list_orders(self, *, status: str | None = None) -> Sequence[BrokerOrder]:
        """Return broker orders, optionally filtered by status."""

    def get_order(self, order_id: str) -> BrokerOrder:
        """Fetch one order by broker order id."""

    def submit_order(
        self,
        order: OrderIntent,
        *,
        client_order_id: str | None = None,
    ) -> BrokerOrder:
        """Submit one order intent to the broker."""

    def cancel_order(self, order_id: str) -> None:
        """Cancel one broker order."""
