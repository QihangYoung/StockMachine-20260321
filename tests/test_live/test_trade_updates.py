from __future__ import annotations

from dataclasses import dataclass

from stockmachine.live import iter_trade_update_events, poll_trade_updates
from stockmachine.live.trade_updates import stream_trade_updates
from stockmachine.data.vendors.alpaca import AlpacaCredentials
from stockmachine.execution.brokers.alpaca_stream import AlpacaTradeUpdateStream


@dataclass
class _FakeWebsocketConnection:
    frames: list[object]
    sent: list[str]
    closed: bool = False

    def send(self, payload: str) -> None:
        self.sent.append(payload)

    def recv(self):
        if not self.frames:
            return None
        return self.frames.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeStreamSource:
    def __init__(self, messages: list[object], *, raise_after: int | None = None) -> None:
        self.messages = messages
        self.raise_after = raise_after

    def iter_messages(self):
        for index, message in enumerate(self.messages):
            if self.raise_after is not None and index >= self.raise_after:
                raise ConnectionError("stream disconnected")
            yield message


def test_poll_trade_updates_emits_events_until_stream_becomes_idle() -> None:
    snapshots = iter(
        [
            [
                {
                    "id": "order-1",
                    "client_order_id": "client-1",
                    "symbol": "AAPL",
                    "side": "buy",
                    "status": "accepted",
                    "qty": 10,
                    "filled_qty": 0,
                    "type": "market",
                    "submitted_at": "2026-03-22T01:00:00+00:00",
                    "updated_at": "2026-03-22T01:00:00+00:00",
                }
            ],
            [
                {
                    "id": "order-1",
                    "client_order_id": "client-1",
                    "symbol": "AAPL",
                    "side": "buy",
                    "status": "accepted",
                    "qty": 10,
                    "filled_qty": 5,
                    "type": "market",
                    "submitted_at": "2026-03-22T01:00:00+00:00",
                    "updated_at": "2026-03-22T01:02:00+00:00",
                    "avg_fill_price": 101.0,
                }
            ],
            [
                {
                    "id": "order-1",
                    "client_order_id": "client-1",
                    "symbol": "AAPL",
                    "side": "buy",
                    "status": "filled",
                    "qty": 10,
                    "filled_qty": 10,
                    "type": "market",
                    "submitted_at": "2026-03-22T01:00:00+00:00",
                    "updated_at": "2026-03-22T01:05:00+00:00",
                    "avg_fill_price": 101.5,
                }
            ],
            [
                {
                    "id": "order-1",
                    "client_order_id": "client-1",
                    "symbol": "AAPL",
                    "side": "buy",
                    "status": "filled",
                    "qty": 10,
                    "filled_qty": 10,
                    "type": "market",
                    "submitted_at": "2026-03-22T01:00:00+00:00",
                    "updated_at": "2026-03-22T01:05:00+00:00",
                    "avg_fill_price": 101.5,
                }
            ],
        ]
    )

    def fetch_orders() -> list[dict[str, object]]:
        return next(snapshots)

    result = poll_trade_updates(fetch_orders, idle_polls_to_stop=1, max_polls=10, sleep_seconds=0.0)

    assert result.polls == 4
    assert result.stopped_reason == "idle"
    assert [event.change_type for event in result.events] == ["new", "fill", "fill"]
    assert result.events[-1].status == "filled"
    assert result.open_order_ids == ()


def test_iter_trade_update_events_exposes_generator_interface() -> None:
    snapshots = iter(
        [
            [
                {
                    "id": "order-3",
                    "client_order_id": "client-3",
                    "symbol": "SPY",
                    "side": "buy",
                    "status": "accepted",
                    "qty": 1,
                    "filled_qty": 0,
                    "type": "market",
                    "submitted_at": "2026-03-22T01:00:00+00:00",
                    "updated_at": "2026-03-22T01:00:00+00:00",
                }
            ],
            [
                {
                    "id": "order-3",
                    "client_order_id": "client-3",
                    "symbol": "SPY",
                    "side": "buy",
                    "status": "accepted",
                    "qty": 1,
                    "filled_qty": 0,
                    "type": "market",
                    "submitted_at": "2026-03-22T01:00:00+00:00",
                    "updated_at": "2026-03-22T01:00:00+00:00",
                }
            ],
        ]
    )

    events = list(iter_trade_update_events(lambda: next(snapshots), idle_polls_to_stop=1))

    assert [event.change_type for event in events] == ["new"]
    assert events[0].symbol == "SPY"


def test_poll_trade_updates_stops_at_max_polls_even_if_events_continue() -> None:
    snapshots = iter(
        [
            [
                {
                    "id": "order-2",
                    "client_order_id": "client-2",
                    "symbol": "MSFT",
                    "side": "buy",
                    "status": "accepted",
                    "qty": 8,
                    "filled_qty": 0,
                    "type": "market",
                    "submitted_at": "2026-03-22T01:00:00+00:00",
                    "updated_at": "2026-03-22T01:00:00+00:00",
                }
            ],
            [
                {
                    "id": "order-2",
                    "client_order_id": "client-2",
                    "symbol": "MSFT",
                    "side": "buy",
                    "status": "accepted",
                    "qty": 8,
                    "filled_qty": 4,
                    "type": "market",
                    "submitted_at": "2026-03-22T01:00:00+00:00",
                    "updated_at": "2026-03-22T01:01:00+00:00",
                    "avg_fill_price": 300.0,
                }
            ],
            [
                {
                    "id": "order-2",
                    "client_order_id": "client-2",
                    "symbol": "MSFT",
                    "side": "buy",
                    "status": "filled",
                    "qty": 8,
                    "filled_qty": 8,
                    "type": "market",
                    "submitted_at": "2026-03-22T01:00:00+00:00",
                    "updated_at": "2026-03-22T01:02:00+00:00",
                    "avg_fill_price": 300.5,
                }
            ],
        ]
    )

    result = poll_trade_updates(lambda: next(snapshots), max_polls=2, idle_polls_to_stop=3)

    assert result.polls == 2
    assert result.stopped_reason == "max_polls"
    assert len(result.events) == 2
    assert result.events[0].change_type == "new"
    assert result.events[1].change_type == "fill"
    assert result.open_order_ids == ("order-2",)


def test_stream_trade_updates_prefers_websocket_events_without_fallback() -> None:
    source = _FakeStreamSource(
        [
            {
                "id": "order-10",
                "client_order_id": "client-10",
                "symbol": "AAPL",
                "side": "buy",
                "status": "accepted",
                "qty": 10,
                "filled_qty": 0,
                "type": "market",
                "submitted_at": "2026-03-22T01:00:00+00:00",
                "updated_at": "2026-03-22T01:00:00+00:00",
            },
            {
                "id": "order-10",
                "client_order_id": "client-10",
                "symbol": "AAPL",
                "side": "buy",
                "status": "accepted",
                "qty": 10,
                "filled_qty": 5,
                "type": "market",
                "submitted_at": "2026-03-22T01:00:00+00:00",
                "updated_at": "2026-03-22T01:02:00+00:00",
                "avg_fill_price": 101.0,
            },
            {
                "id": "order-10",
                "client_order_id": "client-10",
                "symbol": "AAPL",
                "side": "buy",
                "status": "filled",
                "qty": 10,
                "filled_qty": 10,
                "type": "market",
                "submitted_at": "2026-03-22T01:00:00+00:00",
                "updated_at": "2026-03-22T01:05:00+00:00",
                "avg_fill_price": 101.5,
            },
        ]
    )

    result = stream_trade_updates(stream_source=source, fetch_orders=None, max_messages=10)

    assert result.source_mode == "websocket"
    assert result.fallback_used is False
    assert [event.change_type for event in result.events] == ["new", "fill", "fill"]
    assert result.websocket_messages == 3
    assert result.open_order_ids == ()


def test_stream_trade_updates_falls_back_to_polling_when_websocket_disconnects() -> None:
    websocket_source = _FakeStreamSource(
        [
            {
                "id": "order-11",
                "client_order_id": "client-11",
                "symbol": "MSFT",
                "side": "buy",
                "status": "accepted",
                "qty": 10,
                "filled_qty": 0,
                "type": "market",
                "submitted_at": "2026-03-22T01:00:00+00:00",
                "updated_at": "2026-03-22T01:00:00+00:00",
            }
            ,
            {
                "id": "order-11",
                "client_order_id": "client-11",
                "symbol": "MSFT",
                "side": "buy",
                "status": "accepted",
                "qty": 10,
                "filled_qty": 0,
                "type": "market",
                "submitted_at": "2026-03-22T01:00:00+00:00",
                "updated_at": "2026-03-22T01:00:00+00:00",
            },
        ],
        raise_after=1,
    )
    snapshots = iter(
        [
            [
                {
                    "id": "order-11",
                    "client_order_id": "client-11",
                    "symbol": "MSFT",
                    "side": "buy",
                    "status": "accepted",
                    "qty": 10,
                    "filled_qty": 5,
                    "type": "market",
                    "submitted_at": "2026-03-22T01:00:00+00:00",
                    "updated_at": "2026-03-22T01:02:00+00:00",
                    "avg_fill_price": 301.0,
                }
            ],
            [
                {
                    "id": "order-11",
                    "client_order_id": "client-11",
                    "symbol": "MSFT",
                    "side": "buy",
                    "status": "filled",
                    "qty": 10,
                    "filled_qty": 10,
                    "type": "market",
                    "submitted_at": "2026-03-22T01:00:00+00:00",
                    "updated_at": "2026-03-22T01:05:00+00:00",
                    "avg_fill_price": 301.5,
                }
            ],
            [
                {
                    "id": "order-11",
                    "client_order_id": "client-11",
                    "symbol": "MSFT",
                    "side": "buy",
                    "status": "filled",
                    "qty": 10,
                    "filled_qty": 10,
                    "type": "market",
                    "submitted_at": "2026-03-22T01:00:00+00:00",
                    "updated_at": "2026-03-22T01:05:00+00:00",
                    "avg_fill_price": 301.5,
                }
            ],
        ]
    )

    def fetch_orders() -> list[dict[str, object]]:
        return next(snapshots)

    result = stream_trade_updates(
        stream_source=websocket_source,
        fetch_orders=fetch_orders,
        max_messages=10,
        max_polls=10,
        idle_polls_to_stop=1,
    )

    assert result.source_mode == "hybrid"
    assert result.fallback_used is True
    assert result.websocket_error is not None
    assert [event.change_type for event in result.events] == ["new", "fill", "fill"]
    assert result.polls == 3
    assert result.open_order_ids == ()


def test_alpaca_trade_update_stream_uses_injected_websocket_factory() -> None:
    calls: dict[str, object] = {}

    def fake_factory(url: str, *, timeout: float, credentials: AlpacaCredentials):
        calls["url"] = url
        calls["timeout"] = timeout
        calls["credentials"] = credentials
        connection = _FakeWebsocketConnection(
            frames=[
                {
                    "id": "order-12",
                    "client_order_id": "client-12",
                    "symbol": "SPY",
                    "side": "buy",
                    "status": "accepted",
                    "qty": 1,
                    "filled_qty": 0,
                    "type": "market",
                    "submitted_at": "2026-03-22T01:00:00+00:00",
                    "updated_at": "2026-03-22T01:00:00+00:00",
                },
                {
                    "id": "order-12",
                    "client_order_id": "client-12",
                    "symbol": "SPY",
                    "side": "buy",
                    "status": "filled",
                    "qty": 1,
                    "filled_qty": 1,
                    "type": "market",
                    "submitted_at": "2026-03-22T01:00:00+00:00",
                    "updated_at": "2026-03-22T01:01:00+00:00",
                    "avg_fill_price": 500.0,
                },
            ],
            sent=[],
        )
        calls["connection"] = connection
        return connection

    stream = AlpacaTradeUpdateStream(
        credentials=AlpacaCredentials(
            api_key_id="k",
            api_secret_key="s",
            trading_base_url="https://paper-api.alpaca.markets",
            data_base_url="https://data.alpaca.markets",
        ),
        websocket_factory=fake_factory,
    )

    messages = list(stream.iter_messages())

    assert calls["url"] == "https://paper-api.alpaca.markets/stream"
    assert len(calls["connection"].sent) == 2
    assert "auth" in calls["connection"].sent[0]
    assert "listen" in calls["connection"].sent[1]
    assert messages[-1]["status"] == "filled"
