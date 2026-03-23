from __future__ import annotations

from datetime import datetime
import re
from typing import Any

_ISO_FRACTION_RE = re.compile(
    r"^(?P<prefix>.+?)(?:\.(?P<fraction>\d+))?(?P<suffix>Z|[+-]\d{2}:\d{2})?$"
)


def normalize_iso_datetime_text(value: str) -> str:
    """Normalize ISO datetime text to a Python 3.10-friendly representation.

    Alpaca occasionally returns timestamps with fractional-second precision that is
    not exactly 6 digits. Python 3.10's ``datetime.fromisoformat`` rejects those
    forms, so we pad or trim to microsecond precision.
    """

    text = value.strip()
    match = _ISO_FRACTION_RE.match(text)
    if not match:
        return text.replace("Z", "+00:00")

    prefix = match.group("prefix")
    fraction = match.group("fraction")
    suffix = match.group("suffix")
    if suffix == "Z":
        suffix = "+00:00"
    elif suffix is None:
        suffix = ""

    if not fraction:
        return f"{prefix}{suffix}"

    normalized_fraction = (fraction[:6]).ljust(6, "0")
    return f"{prefix}.{normalized_fraction}{suffix}"


def parse_iso_datetime_like(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(normalize_iso_datetime_text(str(value)))
