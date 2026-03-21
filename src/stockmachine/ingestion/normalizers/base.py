from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from stockmachine.ingestion.collectors.base import RawRecord


@dataclass(slots=True, frozen=True)
class NormalizedBatch:
    """A normalized batch destined for one canonical table."""

    table_name: str
    rows: tuple[Mapping[str, Any], ...]
    meta: Mapping[str, Any] = field(default_factory=dict)


class Normalizer(Protocol):
    """Normalize raw records into canonical silver tables."""

    source_name: str
    target_tables: tuple[str, ...]

    def normalize(self, records: list[RawRecord]) -> list[NormalizedBatch]:
        """Normalize a source-specific record set."""
