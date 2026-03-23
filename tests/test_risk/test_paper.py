from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

from stockmachine.domain.models import OrderIntent
from stockmachine.risk import (
    BrokerAwareOrderRiskPolicy,
    build_client_order_id,
    validate_client_order_id,
)


def test_risk_policy_blocks_duplicate_open_orders() -> None:
    policy = BrokerAwareOrderRiskPolicy()
    account_sync = SimpleNamespace(
        broker_account=SimpleNamespace(
            status="ACTIVE",
            buying_power=10_000.0,
            trading_blocked=False,
            account_blocked=False,
        ),
        clock=SimpleNamespace(is_open=True),
    )
    orders = [
        OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=10,
            order_type="market",
            limit_price=None,
            timestamp=datetime(2026, 3, 22, tzinfo=timezone.utc),
            meta={"close": 100.0},
        )
    ]
    open_orders = [{"symbol": "AAPL", "side": "BUY", "status": "new"}]

    result = policy.validate(orders=orders, account_sync=account_sync, open_orders=open_orders)

    assert len(result.approved_orders) == 0
    assert len(result.blocked_orders) == 1
    assert result.issues[0].code == "duplicate_open_order"


def test_risk_policy_enforces_buying_power_in_order_sequence() -> None:
    policy = BrokerAwareOrderRiskPolicy(min_buying_power_buffer=50.0)
    account_sync = SimpleNamespace(
        broker_account=SimpleNamespace(
            status="ACTIVE",
            buying_power=1_000.0,
            trading_blocked=False,
            account_blocked=False,
        ),
        clock=SimpleNamespace(is_open=True),
    )
    orders = [
        OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=5,
            order_type="market",
            limit_price=None,
            timestamp=datetime(2026, 3, 22, tzinfo=timezone.utc),
            meta={"close": 100.0},
        ),
        OrderIntent(
            symbol="MSFT",
            side="BUY",
            quantity=6,
            order_type="market",
            limit_price=None,
            timestamp=datetime(2026, 3, 22, tzinfo=timezone.utc),
            meta={"close": 100.0},
        ),
    ]

    result = policy.validate(orders=orders, account_sync=account_sync)

    assert len(result.approved_orders) == 1
    assert result.approved_orders[0].symbol == "AAPL"
    assert len(result.blocked_orders) == 1
    assert result.blocked_orders[0].symbol == "MSFT"
    assert result.issues[0].code == "insufficient_buying_power"


def test_risk_policy_allows_sell_orders_to_release_buying_power() -> None:
    policy = BrokerAwareOrderRiskPolicy()
    account_sync = SimpleNamespace(
        broker_account=SimpleNamespace(
            status="ACTIVE",
            buying_power=400.0,
            trading_blocked=False,
            account_blocked=False,
        ),
        clock=SimpleNamespace(is_open=True),
    )
    orders = [
        OrderIntent(
            symbol="AAPL",
            side="SELL",
            quantity=5,
            order_type="market",
            limit_price=None,
            timestamp=datetime(2026, 3, 23, tzinfo=timezone.utc),
            meta={"close": 100.0},
        ),
        OrderIntent(
            symbol="MSFT",
            side="BUY",
            quantity=8,
            order_type="market",
            limit_price=None,
            timestamp=datetime(2026, 3, 23, tzinfo=timezone.utc),
            meta={"close": 100.0},
        ),
    ]

    result = policy.validate(orders=orders, account_sync=account_sync)

    assert len(result.approved_orders) == 2
    assert [order.side for order in result.approved_orders] == ["SELL", "BUY"]
    assert len(result.blocked_orders) == 0
    assert result.estimated_notional == 800.0


def test_risk_policy_blocks_orders_above_max_order_notional() -> None:
    policy = BrokerAwareOrderRiskPolicy(max_order_notional=500.0)
    account_sync = SimpleNamespace(
        broker_account=SimpleNamespace(
            status="ACTIVE",
            buying_power=10_000.0,
            trading_blocked=False,
            account_blocked=False,
        ),
        clock=SimpleNamespace(is_open=True),
    )
    orders = [
        OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=6,
            order_type="market",
            limit_price=None,
            timestamp=datetime(2026, 3, 22, tzinfo=timezone.utc),
            meta={"close": 100.0},
        )
    ]

    result = policy.validate(orders=orders, account_sync=account_sync)

    assert len(result.approved_orders) == 0
    assert len(result.blocked_orders) == 1
    assert result.issues[0].code == "max_order_notional_exceeded"


def test_risk_policy_blocks_orders_above_max_total_notional() -> None:
    policy = BrokerAwareOrderRiskPolicy(max_total_notional=900.0)
    account_sync = SimpleNamespace(
        broker_account=SimpleNamespace(
            status="ACTIVE",
            buying_power=10_000.0,
            trading_blocked=False,
            account_blocked=False,
        ),
        clock=SimpleNamespace(is_open=True),
    )
    orders = [
        OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=5,
            order_type="market",
            limit_price=None,
            timestamp=datetime(2026, 3, 22, tzinfo=timezone.utc),
            meta={"close": 100.0},
        ),
        OrderIntent(
            symbol="MSFT",
            side="BUY",
            quantity=5,
            order_type="market",
            limit_price=None,
            timestamp=datetime(2026, 3, 22, tzinfo=timezone.utc),
            meta={"close": 100.0},
        ),
    ]

    result = policy.validate(orders=orders, account_sync=account_sync)

    assert len(result.approved_orders) == 1
    assert result.approved_orders[0].symbol == "AAPL"
    assert len(result.blocked_orders) == 1
    assert result.blocked_orders[0].symbol == "MSFT"
    assert result.issues[0].code == "max_total_notional_exceeded"


def test_risk_policy_blocks_orders_above_max_total_orders() -> None:
    policy = BrokerAwareOrderRiskPolicy(max_total_orders=1)
    account_sync = SimpleNamespace(
        broker_account=SimpleNamespace(
            status="ACTIVE",
            buying_power=10_000.0,
            trading_blocked=False,
            account_blocked=False,
        ),
        clock=SimpleNamespace(is_open=True),
    )
    orders = [
        OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=3,
            order_type="market",
            limit_price=None,
            timestamp=datetime(2026, 3, 22, tzinfo=timezone.utc),
            meta={"close": 100.0},
        ),
        OrderIntent(
            symbol="MSFT",
            side="BUY",
            quantity=3,
            order_type="market",
            limit_price=None,
            timestamp=datetime(2026, 3, 22, tzinfo=timezone.utc),
            meta={"close": 100.0},
        ),
    ]

    result = policy.validate(orders=orders, account_sync=account_sync)

    assert len(result.approved_orders) == 1
    assert result.approved_orders[0].symbol == "AAPL"
    assert len(result.blocked_orders) == 1
    assert result.blocked_orders[0].symbol == "MSFT"
    assert result.issues[0].code == "max_total_orders_exceeded"


def test_client_order_id_builder_and_validator_round_trip() -> None:
    client_order_id = build_client_order_id(
        session_date=date(2026, 3, 22),
        symbol="AAPL",
        side="BUY",
        sequence=7,
        run_id="run-abcdef123456",
    )

    assert client_order_id == "smk-20260322-AAPL-BUY-0007-runabcde"
    assert validate_client_order_id(client_order_id)


def test_risk_policy_can_require_client_order_id() -> None:
    policy = BrokerAwareOrderRiskPolicy(require_client_order_id=True)
    account_sync = SimpleNamespace(
        broker_account=SimpleNamespace(
            status="ACTIVE",
            buying_power=10_000.0,
            trading_blocked=False,
            account_blocked=False,
        ),
        clock=SimpleNamespace(is_open=True),
    )
    orders = [
        OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=10,
            order_type="market",
            limit_price=None,
            timestamp=datetime(2026, 3, 22, tzinfo=timezone.utc),
            meta={"close": 100.0, "client_order_id": "smk-20260322-AAPL-BUY-0001-runabcde"},
        )
    ]

    approved = policy.validate(orders=orders, account_sync=account_sync)
    assert len(approved.approved_orders) == 1

    blocked = policy.validate(
        orders=[
            OrderIntent(
                symbol="MSFT",
                side="BUY",
                quantity=5,
                order_type="market",
                limit_price=None,
                timestamp=datetime(2026, 3, 22, tzinfo=timezone.utc),
                meta={"close": 100.0},
            )
        ],
        account_sync=account_sync,
    )
    assert len(blocked.approved_orders) == 0
    assert len(blocked.blocked_orders) == 1
    assert blocked.issues[0].code == "missing_client_order_id"
