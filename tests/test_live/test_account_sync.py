from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from stockmachine.execution.brokers.alpaca import BrokerAccount, BrokerClock, BrokerPosition
from stockmachine.live import AlpacaAccountSync, sync_account_snapshot


@dataclass
class _FakeBroker:
    account: BrokerAccount
    clock: BrokerClock
    positions: list[BrokerPosition]

    def get_account(self) -> BrokerAccount:
        return self.account

    def get_clock(self) -> BrokerClock:
        return self.clock

    def list_positions(self) -> list[BrokerPosition]:
        return self.positions


def test_account_sync_maps_broker_state_into_project_snapshot() -> None:
    broker = _FakeBroker(
        account=BrokerAccount(
            account_id="acct_1",
            status="ACTIVE",
            cash=1000.0,
            equity=1200.0,
            buying_power=2400.0,
            portfolio_value=1200.0,
            long_market_value=200.0,
            short_market_value=0.0,
            pattern_day_trader=False,
            trading_blocked=False,
            account_blocked=False,
        ),
        clock=BrokerClock(
            timestamp=datetime(2026, 3, 21, 14, 30),
            is_open=True,
        ),
        positions=[
            BrokerPosition(
                symbol="AAPL",
                quantity=2,
                market_value=400.0,
                avg_entry_price=190.0,
                cost_basis=380.0,
                unrealized_pl=20.0,
                side="long",
            )
        ],
    )

    result = AlpacaAccountSync(broker).sync()
    snapshot = sync_account_snapshot(broker)

    assert result.snapshot.session_date.isoformat() == "2026-03-21"
    assert result.snapshot.cash == 1000.0
    assert result.snapshot.equity == 1200.0
    assert round(result.snapshot.gross_exposure, 6) == 400.0
    assert round(result.snapshot.positions[0].weight, 6) == round(400.0 / 1200.0, 6)
    assert snapshot.equity == 1200.0
