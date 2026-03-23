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
