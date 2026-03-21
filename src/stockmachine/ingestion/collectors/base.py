from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Protocol


@dataclass(slots=True, frozen=True)
class FetchWindow:
    """Defines what a collector should fetch in one pass."""

    stream_name: str
    start_date: date | None = None
    end_date: date | None = None
    symbols: tuple[str, ...] = ()
    full_refresh: bool = False
    cursor: str | None = None


@dataclass(slots=True, frozen=True)
class RawRecord:
    """A raw payload captured from an upstream source."""

    source_name: str
    stream_name: str
    pulled_at_utc: datetime
    payload: Mapping[str, Any]
    symbol: str | None = None
    effective_time_utc: datetime | None = None
    raw_key: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


class Collector(Protocol):
    """Collector interface shared by API and crawler jobs."""

    source_name: str
    supported_streams: tuple[str, ...]

    def collect(self, window: FetchWindow) -> list[RawRecord]:
        """Collect raw records for a requested stream and window."""
