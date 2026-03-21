from __future__ import annotations

from dataclasses import dataclass

from stockmachine.data.vendors.alpaca import AlpacaCredentials
from stockmachine.execution.brokers.alpaca_stream import AlpacaTradeUpdateStream


@dataclass
class _FakeConnection:
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


def test_alpaca_trade_update_stream_uses_injected_connection_without_network() -> None:
    calls: dict[str, object] = {}

    def fake_factory(url: str, *, timeout: float, credentials: AlpacaCredentials):
        connection = _FakeConnection(
            frames=[
                {
                    "id": "order-50",
                    "client_order_id": "client-50",
                    "symbol": "AAPL",
                    "side": "buy",
                    "status": "accepted",
                    "qty": 1,
                    "filled_qty": 0,
                    "type": "market",
                    "submitted_at": "2026-03-22T01:00:00+00:00",
                    "updated_at": "2026-03-22T01:00:00+00:00",
                }
            ],
            sent=[],
        )
        calls["url"] = url
        calls["timeout"] = timeout
        calls["credentials"] = credentials
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
        channels=("trade_updates", "orders"),
    )

    messages = list(stream.iter_messages())

    assert calls["url"] == "https://paper-api.alpaca.markets/stream"
    assert calls["timeout"] == 10.0
    assert isinstance(calls["credentials"], AlpacaCredentials)
    assert len(calls["connection"].sent) == 2
    assert "auth" in calls["connection"].sent[0]
    assert "listen" in calls["connection"].sent[1]
    assert messages[0]["symbol"] == "AAPL"
