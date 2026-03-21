from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Protocol

from stockmachine.data.vendors.alpaca import AlpacaCredentials


class AlpacaStreamError(RuntimeError):
    """Raised when the Alpaca websocket seam cannot be opened or used."""


class AlpacaWebSocketConnection(Protocol):
    """Minimal websocket connection surface used by the Alpaca seam."""

    def send(self, payload: str) -> Any:
        """Send a JSON payload."""

    def recv(self) -> str | bytes | None:
        """Receive a websocket frame."""

    def close(self) -> Any:
        """Close the websocket connection."""


@dataclass(slots=True, frozen=True)
class AlpacaTradeUpdateStream:
    """Transport seam for Alpaca trade updates.

    The stream is intentionally lightweight: tests can inject a fake factory,
    while production can rely on an optional websocket-client dependency.
    """

    credentials: AlpacaCredentials
    stream_url: str | None = None
    channels: tuple[str, ...] = ("trade_updates",)
    connect_timeout_seconds: float = 10.0
    websocket_factory: Callable[..., AlpacaWebSocketConnection] | None = None
    auth_payload: Mapping[str, Any] | None = None
    listen_payload: Mapping[str, Any] | None = None

    def resolved_stream_url(self) -> str:
        return self.stream_url or f"{self.credentials.trading_base_url.rstrip('/')}/stream"

    def build_auth_payload(self) -> dict[str, Any]:
        if self.auth_payload is not None:
            return dict(self.auth_payload)
        return {
            "action": "auth",
            "key": self.credentials.api_key_id,
            "secret": self.credentials.api_secret_key,
        }

    def build_listen_payload(self) -> dict[str, Any]:
        if self.listen_payload is not None:
            return dict(self.listen_payload)
        return {
            "action": "listen",
            "data": {"streams": list(self.channels)},
        }

    def iter_messages(self) -> Iterator[dict[str, Any]]:
        connection = self.open_connection()
        try:
            self._send_json(connection, self.build_auth_payload())
            self._send_json(connection, self.build_listen_payload())
            while True:
                raw_message = connection.recv()
                if raw_message in (None, ""):
                    break
                yield self.decode_message(raw_message)
        finally:
            self._close(connection)

    def open_connection(self) -> AlpacaWebSocketConnection:
        factory = self.websocket_factory or _default_websocket_factory
        return factory(
            self.resolved_stream_url(),
            timeout=self.connect_timeout_seconds,
            credentials=self.credentials,
        )

    @staticmethod
    def decode_message(raw_message: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(raw_message, Mapping):
            return dict(raw_message)
        if isinstance(raw_message, bytes):
            raw_text = raw_message.decode("utf-8")
        else:
            raw_text = raw_message
        payload = json.loads(raw_text)
        if not isinstance(payload, dict):
            raise AlpacaStreamError("Expected Alpaca websocket messages to decode into a JSON object.")
        return payload

    @staticmethod
    def _send_json(connection: AlpacaWebSocketConnection, payload: Mapping[str, Any]) -> None:
        connection.send(json.dumps(dict(payload)))

    @staticmethod
    def _close(connection: AlpacaWebSocketConnection) -> None:
        close = getattr(connection, "close", None)
        if callable(close):
            close()


def _default_websocket_factory(
    url: str,
    *,
    timeout: float,
    credentials: AlpacaCredentials,
) -> AlpacaWebSocketConnection:
    try:
        from websocket import create_connection
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise AlpacaStreamError(
            "websocket-client is not installed; provide websocket_factory for tests or install the optional dependency."
        ) from exc

    headers = [
        f"APCA-API-KEY-ID: {credentials.api_key_id}",
        f"APCA-API-SECRET-KEY: {credentials.api_secret_key}",
    ]
    return create_connection(url, timeout=timeout, header=headers)
