from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from stockmachine.domain.enums import Market


class SourceKind(str, Enum):
    """Kinds of upstream data integrations."""

    API = "API"
    WEB = "WEB"
    FILE = "FILE"
    FEED = "FEED"


class UpdateCadence(str, Enum):
    """Expected upstream freshness cadence."""

    DAILY = "DAILY"
    INTRADAY = "INTRADAY"
    EVENT_DRIVEN = "EVENT_DRIVEN"
    AD_HOC = "AD_HOC"


@dataclass(slots=True, frozen=True)
class SourceDefinition:
    """Metadata describing one upstream source."""

    name: str
    market: Market
    source_kind: SourceKind
    cadence: UpdateCadence
    description: str
    streams: tuple[str, ...]
    supports_backfill: bool = True


SOURCE_CATALOG: dict[str, SourceDefinition] = {
    "yahoo_finance": SourceDefinition(
        name="yahoo_finance",
        market=Market.US_EQUITY,
        source_kind=SourceKind.API,
        cadence=UpdateCadence.DAILY,
        description="Bootstrap research source for daily bars and metadata.",
        streams=("symbol_master", "daily_bars", "benchmark_index", "industry_membership"),
    ),
    "alpaca_trading": SourceDefinition(
        name="alpaca_trading",
        market=Market.US_EQUITY,
        source_kind=SourceKind.API,
        cadence=UpdateCadence.DAILY,
        description="Alpaca trading and reference API for US equity assets.",
        streams=("assets",),
    ),
    "sec_edgar": SourceDefinition(
        name="sec_edgar",
        market=Market.US_EQUITY,
        source_kind=SourceKind.API,
        cadence=UpdateCadence.EVENT_DRIVEN,
        description="SEC filings, company facts, and disclosure metadata.",
        streams=("company_facts", "submissions", "filings"),
    ),
    "alpaca_market_data": SourceDefinition(
        name="alpaca_market_data",
        market=Market.US_EQUITY,
        source_kind=SourceKind.API,
        cadence=UpdateCadence.INTRADAY,
        description="US equity market data API for bars, trades, and quotes.",
        streams=("daily_bars", "minute_bars", "trades", "quotes"),
    ),
    "polygon_market_data": SourceDefinition(
        name="polygon_market_data",
        market=Market.US_EQUITY,
        source_kind=SourceKind.API,
        cadence=UpdateCadence.INTRADAY,
        description="US equity aggregates, reference data, and corporate actions.",
        streams=("daily_bars", "reference_tickers", "splits", "dividends"),
    ),
    "nasdaq_data_link": SourceDefinition(
        name="nasdaq_data_link",
        market=Market.US_EQUITY,
        source_kind=SourceKind.API,
        cadence=UpdateCadence.DAILY,
        description="Research-oriented equities and fundamentals datasets.",
        streams=("sharadar_prices", "sharadar_fundamentals"),
    ),
    "finra_public": SourceDefinition(
        name="finra_public",
        market=Market.US_EQUITY,
        source_kind=SourceKind.FILE,
        cadence=UpdateCadence.DAILY,
        description="Public short-interest and ATS transparency datasets.",
        streams=("short_volume", "short_interest", "ats_transparency"),
    ),
}


def get_source(name: str) -> SourceDefinition:
    """Return a source definition by name."""

    try:
        return SOURCE_CATALOG[name]
    except KeyError as exc:
        raise KeyError(f"Unknown source definition: {name}") from exc
