from stockmachine.domain.enums import Market
from stockmachine.ingestion.sources import SOURCE_CATALOG, get_source


def test_us_source_catalog_is_us_equity_only_for_v1() -> None:
    assert SOURCE_CATALOG
    assert all(source.market is Market.US_EQUITY for source in SOURCE_CATALOG.values())


def test_get_source_returns_named_source() -> None:
    source = get_source("sec_edgar")
    assert source.name == "sec_edgar"
    assert "filings" in source.streams
