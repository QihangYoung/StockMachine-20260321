from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import Enum


class SyncStatus(str, Enum):
    """Possible checkpoint states for an ingestion job."""

    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


@dataclass(slots=True, frozen=True)
class SyncCheckpoint:
    """Checkpoint describing the last known sync state for one stream."""

    source_name: str
    stream_name: str
    updated_at_utc: datetime
    status: SyncStatus
    cursor: str | None = None
    last_successful_date: date | None = None
    detail: str | None = None

    def advance(
        self,
        *,
        updated_at_utc: datetime,
        status: SyncStatus,
        cursor: str | None = None,
        last_successful_date: date | None = None,
        detail: str | None = None,
    ) -> "SyncCheckpoint":
        """Return a new checkpoint with updated sync state."""

        return replace(
            self,
            updated_at_utc=updated_at_utc,
            status=status,
            cursor=cursor if cursor is not None else self.cursor,
            last_successful_date=(
                last_successful_date
                if last_successful_date is not None
                else self.last_successful_date
            ),
            detail=detail,
        )
