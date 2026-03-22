from __future__ import annotations

import json

import pandas as pd

from stockmachine.ingestion.storage import StorageLayout
from stockmachine.research.universe import (
    build_point_in_time_metadata_history,
    load_point_in_time_universe,
    load_universe_industry_map,
    load_universe_members,
    resolve_point_in_time_universe,
)


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _symbol_master_rows() -> list[dict[str, object]]:
    return [
        {
            "as_of_date": "2025-01-02",
            "symbol": "AAPL",
            "security_id": "AAPL",
            "company_name": "Apple Old",
            "exchange_mic": "XNAS",
            "currency": "USD",
            "security_type": "COMMON_STOCK",
            "asset_class": "EQUITY",
            "is_active": True,
            "list_date": "2010-01-01",
            "delist_date": None,
            "sector": "Technology",
            "industry": "Hardware",
            "country_of_listing": "US",
            "primary_share_class": True,
            "source_name": "alpaca_market_data",
            "load_time_utc": "2025-01-02T22:00:00+00:00",
            "source_version": "v1",
        },
        {
            "as_of_date": "2025-01-02",
            "symbol": "MSFT",
            "security_id": "MSFT",
            "company_name": "Microsoft Old",
            "exchange_mic": "XNAS",
            "currency": "USD",
            "security_type": "COMMON_STOCK",
            "asset_class": "EQUITY",
            "is_active": False,
            "list_date": "1986-03-13",
            "delist_date": None,
            "sector": "Technology",
            "industry": "Software",
            "country_of_listing": "US",
            "primary_share_class": True,
            "source_name": "alpaca_market_data",
            "load_time_utc": "2025-01-02T22:00:00+00:00",
            "source_version": "v1",
        },
        {
            "as_of_date": "2025-01-04",
            "symbol": "AAPL",
            "security_id": "AAPL",
            "company_name": "Apple Future",
            "exchange_mic": "XNAS",
            "currency": "USD",
            "security_type": "COMMON_STOCK",
            "asset_class": "EQUITY",
            "is_active": True,
            "list_date": "2010-01-01",
            "delist_date": None,
            "sector": "Technology",
            "industry": "Hardware",
            "country_of_listing": "US",
            "primary_share_class": True,
            "source_name": "alpaca_market_data",
            "load_time_utc": "2025-01-04T22:00:00+00:00",
            "source_version": "v2",
        },
    ]


def _industry_rows() -> list[dict[str, object]]:
    return [
        {
            "as_of_date": "2025-01-01",
            "symbol": "AAPL",
            "industry_system": "gics",
            "sector_name": "Technology",
            "industry_group_name": "Info Tech",
            "industry_name": "Hardware",
            "subindustry_name": "Computers",
            "source_name": "alpaca_market_data",
            "load_time_utc": "2025-01-01T22:00:00+00:00",
            "source_version": "v1",
        },
        {
            "as_of_date": "2025-01-03",
            "symbol": "AAPL",
            "industry_system": "gics",
            "sector_name": "Technology New",
            "industry_group_name": "Info Tech New",
            "industry_name": "Hardware New",
            "subindustry_name": "Computers New",
            "source_name": "alpaca_market_data",
            "load_time_utc": "2025-01-03T22:00:00+00:00",
            "source_version": "v2",
        },
        {
            "as_of_date": "2025-01-03",
            "symbol": "MSFT",
            "industry_system": "gics",
            "sector_name": "Technology",
            "industry_group_name": "Info Tech",
            "industry_name": "Software",
            "subindustry_name": "Applications",
            "source_name": "alpaca_market_data",
            "load_time_utc": "2025-01-03T22:00:00+00:00",
            "source_version": "v2",
        },
    ]


def _universe_membership_rows() -> list[dict[str, object]]:
    return [
        {
            "session_date": "2025-01-02",
            "universe_name": "research_v1",
            "symbol": "AAPL",
            "is_member": True,
            "membership_source": "historical_index_membership",
            "entry_date": None,
            "exit_date": None,
            "source_name": "test_source",
            "load_time_utc": "2025-01-02T22:00:00+00:00",
            "source_version": "v1",
        },
        {
            "session_date": "2025-01-03",
            "universe_name": "research_v1",
            "symbol": "AAPL",
            "is_member": True,
            "membership_source": "historical_index_membership",
            "entry_date": None,
            "exit_date": None,
            "source_name": "test_source",
            "load_time_utc": "2025-01-03T22:00:00+00:00",
            "source_version": "v1",
        },
    ]


def test_load_point_in_time_universe_uses_latest_visible_snapshot(tmp_path) -> None:
    layout = StorageLayout(root=tmp_path)
    symbol_dir = layout.silver_table_dir("symbol_master")
    industry_dir = layout.silver_table_dir("industry_membership")
    _write_jsonl(symbol_dir / "snapshot.jsonl", _symbol_master_rows())
    _write_jsonl(industry_dir / "snapshot.jsonl", _industry_rows())

    universe = load_point_in_time_universe("2025-01-03", layout=layout)

    assert universe.session_date == pd.Timestamp("2025-01-03")
    assert universe.symbol_master_snapshot_date == pd.Timestamp("2025-01-02")
    assert universe.industry_snapshot_date == pd.Timestamp("2025-01-03")
    assert universe.members == ("AAPL",)
    assert universe.metadata.iloc[0]["company_name"] == "Apple Old"
    assert universe.metadata.iloc[0]["symbol"] == "AAPL"
    assert universe.industry_map.iloc[0]["sector"] == "Technology New"
    assert universe.industry_map.iloc[0]["industry"] == "Hardware New"
    assert universe.conservative is False
    assert universe.notes == ()

    assert load_universe_members("2025-01-03", layout=layout) == ("AAPL",)
    industry_map = load_universe_industry_map("2025-01-03", layout=layout)
    assert list(industry_map["symbol"]) == ["AAPL"]
    assert industry_map.iloc[0]["sector"] == "Technology New"


def test_load_point_in_time_universe_prefers_explicit_membership_snapshot(tmp_path) -> None:
    layout = StorageLayout(root=tmp_path)
    symbol_dir = layout.silver_table_dir("symbol_master")
    industry_dir = layout.silver_table_dir("industry_membership")
    membership_dir = layout.silver_table_dir("universe_membership")

    rows = _symbol_master_rows()
    rows[1]["is_active"] = True
    _write_jsonl(symbol_dir / "snapshot.jsonl", rows)
    _write_jsonl(industry_dir / "snapshot.jsonl", _industry_rows())
    _write_jsonl(membership_dir / "snapshot.jsonl", _universe_membership_rows())

    universe = load_point_in_time_universe("2025-01-03", layout=layout, universe_name="research_v1")

    assert universe.universe_membership_snapshot_date == pd.Timestamp("2025-01-03")
    assert universe.membership_source == "explicit_universe_membership"
    assert universe.members == ("AAPL",)
    assert list(universe.universe_membership["symbol"]) == ["AAPL"]
    assert universe.conservative is False


def test_load_point_in_time_universe_is_conservative_when_no_visible_snapshot(tmp_path) -> None:
    layout = StorageLayout(root=tmp_path)
    symbol_dir = layout.silver_table_dir("symbol_master")
    industry_dir = layout.silver_table_dir("industry_membership")

    late_symbol_rows = _symbol_master_rows()
    for row in late_symbol_rows:
        row["as_of_date"] = "2025-01-10"
        row["load_time_utc"] = "2025-01-10T22:00:00+00:00"
    late_industry_rows = _industry_rows()
    for row in late_industry_rows:
        row["as_of_date"] = "2025-01-10"
        row["load_time_utc"] = "2025-01-10T22:00:00+00:00"

    _write_jsonl(symbol_dir / "future.jsonl", late_symbol_rows)
    _write_jsonl(industry_dir / "future.jsonl", late_industry_rows)

    universe = load_point_in_time_universe("2025-01-03", layout=layout)

    assert universe.members == ()
    assert universe.metadata.empty
    assert universe.industry_membership.empty
    assert universe.industry_map.empty
    assert universe.conservative is True
    assert any("missing point-in-time symbol_master snapshot" in note for note in universe.notes)
    assert any("missing point-in-time industry_membership snapshot" in note for note in universe.notes)


def test_load_point_in_time_universe_can_require_snapshot(tmp_path) -> None:
    layout = StorageLayout(root=tmp_path)

    try:
        load_point_in_time_universe("2025-01-03", layout=layout, require_snapshot=True)
    except FileNotFoundError as exc:
        assert "No eligible point-in-time universe snapshot" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected FileNotFoundError when no snapshot is available")


def test_build_point_in_time_metadata_history_emits_date_aware_rows(tmp_path) -> None:
    layout = StorageLayout(root=tmp_path)
    symbol_dir = layout.silver_table_dir("symbol_master")
    industry_dir = layout.silver_table_dir("industry_membership")
    _write_jsonl(symbol_dir / "snapshot.jsonl", _symbol_master_rows())
    _write_jsonl(industry_dir / "snapshot.jsonl", _industry_rows())

    snapshot = resolve_point_in_time_universe(
        "2025-01-03",
        symbol_master_frame=pd.DataFrame(_symbol_master_rows()),
        industry_membership_frame=pd.DataFrame(_industry_rows()),
    )
    assert snapshot.members == ("AAPL",)

    history = build_point_in_time_metadata_history(
        pd.Index([pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]),
        symbol_master_frame=pd.DataFrame(_symbol_master_rows()),
        industry_membership_frame=pd.DataFrame(_industry_rows()),
    )

    assert list(history.columns) == [
        "date",
        "symbol",
        "company_name",
        "quote_type",
        "exchange",
        "currency",
        "country",
        "sector",
        "industry",
    ]
    assert history["date"].nunique() == 2
    assert set(history["symbol"]) == {"AAPL"}
    jan3 = history.loc[history["date"] == pd.Timestamp("2025-01-03")].iloc[0]
    assert jan3["exchange"] == "XNAS"
    assert jan3["sector"] == "Technology New"


def test_build_point_in_time_metadata_history_respects_explicit_membership() -> None:
    rows = _symbol_master_rows()
    rows[1]["is_active"] = True
    history = build_point_in_time_metadata_history(
        pd.Index([pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]),
        universe_membership_frame=pd.DataFrame(_universe_membership_rows()),
        symbol_master_frame=pd.DataFrame(rows),
        industry_membership_frame=pd.DataFrame(_industry_rows()),
        universe_name="research_v1",
    )

    assert set(history["symbol"]) == {"AAPL"}
    assert history["date"].nunique() == 2


def test_build_point_in_time_metadata_history_supports_exact_snapshot_fast_path() -> None:
    symbol_master = pd.DataFrame(
        [
            {
                "as_of_date": "2025-01-02",
                "symbol": "AAPL",
                "security_id": "AAPL",
                "company_name": "Apple",
                "exchange_mic": "XNAS",
                "currency": "USD",
                "security_type": "COMMON_STOCK",
                "asset_class": "EQUITY",
                "is_active": True,
                "list_date": "2010-01-01",
                "delist_date": None,
                "sector": None,
                "industry": None,
                "country_of_listing": "US",
                "primary_share_class": True,
                "source_name": "bootstrap",
                "load_time_utc": "2025-01-02T22:00:00+00:00",
                "source_version": "static_history_backfill_v1",
            },
            {
                "as_of_date": "2025-01-03",
                "symbol": "AAPL",
                "security_id": "AAPL",
                "company_name": "Apple",
                "exchange_mic": "XNAS",
                "currency": "USD",
                "security_type": "COMMON_STOCK",
                "asset_class": "EQUITY",
                "is_active": True,
                "list_date": "2010-01-01",
                "delist_date": None,
                "sector": None,
                "industry": None,
                "country_of_listing": "US",
                "primary_share_class": True,
                "source_name": "bootstrap",
                "load_time_utc": "2025-01-03T22:00:00+00:00",
                "source_version": "static_history_backfill_v1",
            },
        ]
    )
    industry_membership = pd.DataFrame(
        [
            {
                "as_of_date": "2025-01-02",
                "symbol": "AAPL",
                "industry_system": "yfinance_sector",
                "sector_name": "Technology",
                "industry_group_name": "Technology",
                "industry_name": "Consumer Electronics",
                "subindustry_name": "Consumer Electronics",
                "source_name": "bootstrap",
                "load_time_utc": "2025-01-02T22:00:00+00:00",
                "source_version": "static_history_backfill_v1",
            },
            {
                "as_of_date": "2025-01-03",
                "symbol": "AAPL",
                "industry_system": "yfinance_sector",
                "sector_name": "Technology",
                "industry_group_name": "Technology",
                "industry_name": "Consumer Electronics",
                "subindustry_name": "Consumer Electronics",
                "source_name": "bootstrap",
                "load_time_utc": "2025-01-03T22:00:00+00:00",
                "source_version": "static_history_backfill_v1",
            },
        ]
    )

    history = build_point_in_time_metadata_history(
        pd.Index([pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]),
        symbol_master_frame=symbol_master,
        industry_membership_frame=industry_membership,
    )

    assert len(history) == 2
    assert list(history["date"]) == [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]
    assert set(history["sector"]) == {"Technology"}
