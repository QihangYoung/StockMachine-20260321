from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, Sequence

from stockmachine.backtest.protocols import AccountSnapshot, PositionSnapshot
from stockmachine.execution.brokers.alpaca import BrokerAccount, BrokerClock, BrokerPosition


class BrokerAccountReader(Protocol):
    """Minimal broker surface required for account sync."""

    def get_account(self) -> BrokerAccount:
        """Fetch the current broker account snapshot."""

    def get_clock(self) -> BrokerClock:
        """Fetch the current broker clock."""

    def list_positions(self) -> Sequence[BrokerPosition]:
        """Fetch open broker positions."""


@dataclass(slots=True, frozen=True)
class AccountSyncResult:
    """Combined broker and project account snapshot."""

    snapshot: AccountSnapshot
    broker_account: BrokerAccount
    broker_positions: tuple[BrokerPosition, ...]
    clock: BrokerClock


@dataclass(slots=True)
class AlpacaAccountSync:
    """Map broker account and positions into internal account snapshots."""

    broker: BrokerAccountReader

    def sync(self, session_date: date | None = None) -> AccountSyncResult:
        broker_account = self.broker.get_account()
        clock = self.broker.get_clock()
        positions = tuple(self.broker.list_positions())
        effective_date = session_date or clock.timestamp.date()
        equity = float(broker_account.equity)
        cash = float(broker_account.cash)
        position_snapshots = tuple(
            PositionSnapshot(
                symbol=position.symbol,
                quantity=int(position.quantity),
                market_value=float(position.market_value),
                weight=(float(position.market_value) / equity) if equity else 0.0,
            )
            for position in positions
        )
        gross_exposure = sum(abs(snapshot.market_value) for snapshot in position_snapshots)
        snapshot = AccountSnapshot(
            session_date=effective_date,
            cash=cash,
            equity=equity,
            gross_exposure=gross_exposure,
            positions=position_snapshots,
        )
        return AccountSyncResult(
            snapshot=snapshot,
            broker_account=broker_account,
            broker_positions=positions,
            clock=clock,
        )


def sync_account_snapshot(
    broker: BrokerAccountReader,
    *,
    session_date: date | None = None,
) -> AccountSnapshot:
    """Convenience wrapper returning the project account object only."""

    return AlpacaAccountSync(broker).sync(session_date=session_date).snapshot
