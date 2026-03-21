from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Mapping

from stockmachine.backtest.protocols import AccountSnapshot, MarketBar
from stockmachine.domain.models import OrderIntent, TargetPosition


@dataclass(slots=True)
class NextOpenOrderExecutionPolicy:
    """Convert target weights into market-on-open order intents."""

    def generate_orders(
        self,
        session_date: date,
        targets: list[TargetPosition],
        bars: Mapping[str, MarketBar],
        account: AccountSnapshot,
    ) -> list[OrderIntent]:
        orders = []
        timestamp = datetime.combine(session_date, time(9, 30))
        for target in targets:
            bar = bars.get(target.symbol)
            if bar is None or bar.open <= 0:
                continue
            quantity = int((account.equity * target.target_weight) / bar.open)
            if quantity <= 0:
                continue
            orders.append(
                OrderIntent(
                    symbol=target.symbol,
                    side="BUY",
                    quantity=quantity,
                    order_type="market_on_open",
                    limit_price=None,
                    timestamp=timestamp,
                    meta=target.meta,
                )
            )
        return orders
