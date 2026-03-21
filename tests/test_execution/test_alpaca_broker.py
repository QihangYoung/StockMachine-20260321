from __future__ import annotations

import json
from datetime import datetime

from stockmachine.data.vendors.alpaca import AlpacaCredentials
from stockmachine.execution.brokers import AlpacaTradingAdapter


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def read(self) -> bytes:
        if self._payload == "":
            return b""
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_alpaca_trading_adapter_maps_broker_objects(monkeypatch) -> None:
    calls: list[tuple[str, str, bytes | None]] = []
    order_status = {"ord_1": "filled"}

    def fake_urlopen(request, timeout=None):
        body = request.data
        calls.append((request.full_url, request.get_method(), body))
        if request.full_url.endswith("/v2/account"):
            return _FakeResponse(
                {
                    "id": "acct_1",
                    "status": "ACTIVE",
                    "cash": "1000.50",
                    "equity": "1200.50",
                    "buying_power": "2400.00",
                    "portfolio_value": "1200.50",
                    "long_market_value": "200.00",
                    "short_market_value": "0.00",
                    "pattern_day_trader": False,
                    "trading_blocked": False,
                    "account_blocked": False,
                }
            )
        if request.full_url.endswith("/v2/clock"):
            return _FakeResponse(
                {
                    "timestamp": "2026-03-21T14:30:00Z",
                    "is_open": True,
                    "next_open": "2026-03-22T13:30:00Z",
                    "next_close": "2026-03-21T20:00:00Z",
                }
            )
        if request.full_url.endswith("/v2/positions"):
            return _FakeResponse(
                [
                    {
                        "symbol": "AAPL",
                        "qty": "2",
                        "market_value": "400.00",
                        "avg_entry_price": "190.00",
                        "cost_basis": "380.00",
                        "unrealized_pl": "20.00",
                        "side": "long",
                    }
                ]
            )
        if "/v2/orders/" in request.full_url and request.get_method() == "GET":
            order_id = request.full_url.rsplit("/", 1)[-1]
            return _FakeResponse(
                {
                    "id": order_id,
                    "client_order_id": "cid_1",
                    "symbol": "AAPL",
                    "side": "buy",
                    "type": "market",
                    "time_in_force": "day",
                    "status": order_status[order_id],
                    "qty": "2",
                    "filled_qty": "2",
                    "filled_avg_price": "200.25",
                    "limit_price": None,
                    "submitted_at": "2026-03-21T14:31:00Z",
                    "updated_at": "2026-03-21T14:31:05Z",
                }
            )
        if request.full_url.startswith("https://paper-api.alpaca.markets/v2/orders?") and request.get_method() == "GET":
            return _FakeResponse(
                [
                    {
                        "id": "ord_1",
                        "client_order_id": "cid_1",
                        "symbol": "AAPL",
                        "side": "buy",
                        "type": "market",
                        "time_in_force": "day",
                        "status": "filled",
                        "qty": "2",
                        "filled_qty": "2",
                        "filled_avg_price": "200.25",
                        "limit_price": None,
                        "submitted_at": "2026-03-21T14:31:00Z",
                        "updated_at": "2026-03-21T14:31:05Z",
                    }
                ]
            )
        if request.full_url.endswith("/v2/orders") and request.get_method() == "POST":
            assert json.loads(body.decode("utf-8"))["symbol"] == "MSFT"
            return _FakeResponse(
                {
                    "id": "ord_2",
                    "client_order_id": "cid_2",
                    "symbol": "MSFT",
                    "side": "buy",
                    "type": "market",
                    "time_in_force": "day",
                    "status": "new",
                    "qty": "3",
                    "filled_qty": "0",
                    "filled_avg_price": None,
                    "limit_price": None,
                    "submitted_at": "2026-03-21T14:35:00Z",
                    "updated_at": "2026-03-21T14:35:00Z",
                }
            )
        if "/v2/orders/" in request.full_url and request.get_method() == "DELETE":
            order_id = request.full_url.rsplit("/", 1)[-1]
            order_status[order_id] = "canceled"
            return _FakeResponse("")
        raise AssertionError(f"Unexpected request: {request.full_url} {request.get_method()}")

    monkeypatch.setattr("stockmachine.execution.brokers.alpaca.urlopen", fake_urlopen)

    adapter = AlpacaTradingAdapter(
        credentials=AlpacaCredentials(
            api_key_id="k",
            api_secret_key="s",
            trading_base_url="https://paper-api.alpaca.markets",
            data_base_url="https://data.alpaca.markets",
        )
    )

    account = adapter.get_account()
    clock = adapter.get_clock()
    positions = adapter.list_positions()
    orders = adapter.list_orders(status="filled")
    submitted = adapter.submit_order(symbol="MSFT", side="buy", quantity=3)
    canceled = adapter.cancel_order("ord_1")

    assert account.account_id == "acct_1"
    assert account.equity == 1200.5
    assert clock.is_open is True
    assert clock.timestamp == datetime.fromisoformat("2026-03-21T14:30:00+00:00")
    assert positions[0].symbol == "AAPL"
    assert orders[0].order_id == "ord_1"
    assert submitted.symbol == "MSFT"
    assert canceled.status == "canceled"
    assert any(method == "POST" for _, method, _ in calls)


def test_alpaca_trading_adapter_accepts_nanosecond_timestamps(monkeypatch) -> None:
    def fake_urlopen(request, timeout=None):
        if request.full_url.endswith("/v2/clock"):
            return _FakeResponse(
                {
                    "timestamp": "2026-03-21T12:22:38.634640667-04:00",
                    "is_open": False,
                    "next_open": "2026-03-22T09:30:00.123456789-04:00",
                    "next_close": "2026-03-21T16:00:00.000000001-04:00",
                }
            )
        raise AssertionError(f"Unexpected request: {request.full_url} {request.get_method()}")

    monkeypatch.setattr("stockmachine.execution.brokers.alpaca.urlopen", fake_urlopen)

    adapter = AlpacaTradingAdapter(
        credentials=AlpacaCredentials(
            api_key_id="k",
            api_secret_key="s",
            trading_base_url="https://paper-api.alpaca.markets",
            data_base_url="https://data.alpaca.markets",
        )
    )

    clock = adapter.get_clock()

    assert clock.timestamp.isoformat() == "2026-03-21T12:22:38.634640-04:00"
    assert clock.next_open.isoformat() == "2026-03-22T09:30:00.123456-04:00"


def test_alpaca_trading_adapter_detects_paper_environment() -> None:
    paper_adapter = AlpacaTradingAdapter(
        credentials=AlpacaCredentials(
            api_key_id="k",
            api_secret_key="s",
            trading_base_url="https://paper-api.alpaca.markets",
            data_base_url="https://data.alpaca.markets",
        )
    )
    live_adapter = AlpacaTradingAdapter(
        credentials=AlpacaCredentials(
            api_key_id="k",
            api_secret_key="s",
            trading_base_url="https://api.alpaca.markets",
            data_base_url="https://data.alpaca.markets",
        )
    )

    assert paper_adapter.is_paper_trading_environment() is True
    assert live_adapter.is_paper_trading_environment() is False
