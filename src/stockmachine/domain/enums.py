from __future__ import annotations

from enum import Enum


class Market(str, Enum):
    """Supported market scopes."""

    US_EQUITY = "US_EQUITY"
    CN_EQUITY = "CN_EQUITY"


class Frequency(str, Enum):
    """Supported data frequencies."""

    DAILY = "DAILY"
    HOURLY = "HOURLY"
    MINUTE = "MINUTE"


class RuntimeMode(str, Enum):
    """Runtime environments."""

    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


class DataLayer(str, Enum):
    """Storage layers used by the ingestion pipeline."""

    RAW = "raw"
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class DataType(str, Enum):
    """Primitive normalized types used in canonical tables."""

    STRING = "string"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    DATE = "date"
    DATETIME_UTC = "datetime_utc"
