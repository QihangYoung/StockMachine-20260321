from __future__ import annotations

import pandas as pd

from stockmachine.ingestion.jobs import bootstrap_yahoo_us_equities as bootstrap_jobs
from stockmachine.ingestion.storage import StorageLayout


def test_bootstrap_us_equities_yahoo_to_silver_supports_custom_file_stem(tmp_path, monkeypatch) -> None:
    history = pd.DataFrame(
        [
            {
                "date": "2014-01-02",
                "symbol": "AAPL",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.0,
                "adj_close": 1.0,
                "volume": 10.0,
                "dividends": 0.0,
                "stock_splits": 0.0,
            },
            {
                "date": "2014-01-02",
                "symbol": bootstrap_jobs.BENCHMARK_SYMBOL,
                "open": 2.0,
                "high": 2.1,
                "low": 1.9,
                "close": 2.0,
                "adj_close": 2.0,
                "volume": 20.0,
                "dividends": 0.0,
                "stock_splits": 0.0,
            },
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "company_name": "Apple",
                "sector": "Technology",
                "industry": "Hardware",
                "quote_type": "EQUITY",
                "exchange": "NMS",
                "currency": "USD",
                "country": "US",
            }
        ]
    )

    monkeypatch.setattr(bootstrap_jobs, "_download_history", lambda **_: history.copy())
    monkeypatch.setattr(bootstrap_jobs, "_download_metadata", lambda **_: metadata.copy())
    layout = StorageLayout(root=tmp_path)

    result = bootstrap_jobs.bootstrap_us_equities_yahoo_to_silver(
        start="2014-01-01",
        end="2014-12-31",
        layout=layout,
        silver_file_stem="custom_gap_seed",
    )

    assert result["silver_file_stem"] == "custom_gap_seed"
    assert (layout.silver_table_dir("daily_bar") / "custom_gap_seed.jsonl").exists()
    assert (layout.silver_table_dir("benchmark_index") / "custom_gap_seed.jsonl").exists()
    assert not (layout.silver_table_dir("daily_bar") / "yahoo_bootstrap.jsonl").exists()


def test_bootstrap_market_symbols_yahoo_to_silver_supports_custom_symbol_sets(tmp_path, monkeypatch) -> None:
    history = pd.DataFrame(
        [
            {
                "date": "2018-01-02",
                "symbol": "VXUS",
                "open": 10.0,
                "high": 10.1,
                "low": 9.9,
                "close": 10.0,
                "adj_close": 10.0,
                "volume": 100.0,
                "dividends": 0.0,
                "stock_splits": 0.0,
            },
            {
                "date": "2018-01-02",
                "symbol": "SPY",
                "open": 20.0,
                "high": 20.1,
                "low": 19.9,
                "close": 20.0,
                "adj_close": 20.0,
                "volume": 200.0,
                "dividends": 0.0,
                "stock_splits": 0.0,
            },
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "symbol": "VXUS",
                "company_name": "Vanguard Total International Stock ETF",
                "sector": "Unknown",
                "industry": "Unknown",
                "quote_type": "ETF",
                "exchange": "PCX",
                "currency": "USD",
                "country": "US",
            },
            {
                "symbol": "SPY",
                "company_name": "SPDR S&P 500 ETF Trust",
                "sector": "Unknown",
                "industry": "Unknown",
                "quote_type": "ETF",
                "exchange": "PCX",
                "currency": "USD",
                "country": "US",
            },
        ]
    )

    monkeypatch.setattr(bootstrap_jobs, "_download_history", lambda **_: history.copy())
    monkeypatch.setattr(bootstrap_jobs, "_download_metadata", lambda **_: metadata.copy())
    layout = StorageLayout(root=tmp_path)

    result = bootstrap_jobs.bootstrap_market_symbols_yahoo_to_silver(
        symbols=("SPY", "VXUS"),
        metadata_symbols=("SPY", "VXUS"),
        start="2018-01-01",
        end="2018-12-31",
        layout=layout,
        silver_file_stem="multi_asset_seed",
        universe_name="us_multi_asset_etfs_v1",
        membership_source="multi_asset_etf_bootstrap",
        benchmark_symbols=("SPY",),
    )

    assert result["symbol_count"] == 2
    assert result["benchmark_index_rows"] == 1
    assert result["daily_bar_rows"] == 1
    assert (layout.silver_table_dir("daily_bar") / "multi_asset_seed.jsonl").exists()
    daily_bar_payload = (layout.silver_table_dir("daily_bar") / "multi_asset_seed.jsonl").read_text(encoding="utf-8")
    benchmark_payload = (layout.silver_table_dir("benchmark_index") / "multi_asset_seed.jsonl").read_text(encoding="utf-8")
    symbol_master_payload = (layout.silver_table_dir("symbol_master") / "multi_asset_seed.jsonl").read_text(encoding="utf-8")
    universe_payload = (layout.silver_table_dir("universe_membership") / "multi_asset_seed.jsonl").read_text(encoding="utf-8")
    assert '"symbol": "VXUS"' in daily_bar_payload
    assert '"symbol": "SPY"' in benchmark_payload
    assert '"security_type": "ETF"' in symbol_master_payload
    assert '"universe_name": "us_multi_asset_etfs_v1"' in universe_payload


def test_bootstrap_fmf_validation_etfs_yahoo_to_silver_uses_rebuild_universe(tmp_path, monkeypatch) -> None:
    history = pd.DataFrame(
        [
            {
                "date": "2013-08-01",
                "symbol": "FMF",
                "open": 10.0,
                "high": 10.1,
                "low": 9.9,
                "close": 10.0,
                "adj_close": 10.0,
                "volume": 100.0,
                "dividends": 0.0,
                "stock_splits": 0.0,
            },
            {
                "date": "2013-08-01",
                "symbol": "SPY",
                "open": 20.0,
                "high": 20.1,
                "low": 19.9,
                "close": 20.0,
                "adj_close": 20.0,
                "volume": 200.0,
                "dividends": 0.0,
                "stock_splits": 0.0,
            },
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "symbol": "FMF",
                "company_name": "First Trust Managed Futures Strategy Fund",
                "sector": "Unknown",
                "industry": "Unknown",
                "quote_type": "ETF",
                "exchange": "PCX",
                "currency": "USD",
                "country": "US",
            },
            {
                "symbol": "SPY",
                "company_name": "SPDR S&P 500 ETF Trust",
                "sector": "Unknown",
                "industry": "Unknown",
                "quote_type": "ETF",
                "exchange": "PCX",
                "currency": "USD",
                "country": "US",
            },
        ]
    )

    monkeypatch.setattr(bootstrap_jobs, "_download_history", lambda **_: history.copy())
    monkeypatch.setattr(bootstrap_jobs, "_download_metadata", lambda **_: metadata.copy())
    layout = StorageLayout(root=tmp_path)

    result = bootstrap_jobs.bootstrap_fmf_validation_etfs_yahoo_to_silver(
        start="2013-08-01",
        end="2013-12-31",
        layout=layout,
        silver_file_stem="fmf_rebuild_seed",
    )

    assert result["silver_file_stem"] == "fmf_rebuild_seed"
    universe_payload = (layout.silver_table_dir("universe_membership") / "fmf_rebuild_seed.jsonl").read_text(encoding="utf-8")
    benchmark_payload = (layout.silver_table_dir("benchmark_index") / "fmf_rebuild_seed.jsonl").read_text(encoding="utf-8")
    daily_payload = (layout.silver_table_dir("daily_bar") / "fmf_rebuild_seed.jsonl").read_text(encoding="utf-8")
    assert '"universe_name": "us_multi_asset_fmf_validation_v1"' in universe_payload
    assert '"symbol": "SPY"' in benchmark_payload
    assert '"symbol": "FMF"' in daily_payload
