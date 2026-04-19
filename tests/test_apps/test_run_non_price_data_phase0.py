from __future__ import annotations

import json

import pandas as pd

from stockmachine.apps.run_non_price_data_phase0 import (
    _build_universe_cik_mapping,
    _load_membership_universe,
    _load_sec_company_tickers,
    _summarize_submission_file,
    _ticker_keys,
)


def test_load_sec_company_tickers_exchange_shape(tmp_path):
    path = tmp_path / "company_tickers_exchange.json"
    path.write_text(
        json.dumps(
            {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"]],
            }
        ),
        encoding="utf-8",
    )

    rows = _load_sec_company_tickers(path)

    assert rows == [
        {
            "cik": "0000320193",
            "ticker": "AAPL",
            "title": "Apple Inc.",
            "exchange": "Nasdaq",
        }
    ]


def test_load_sec_company_tickers_legacy_shape(tmp_path):
    path = tmp_path / "company_tickers.json"
    path.write_text(
        json.dumps({"0": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"}}),
        encoding="utf-8",
    )

    rows = _load_sec_company_tickers(path)

    assert rows == [
        {
            "cik": "0000789019",
            "ticker": "MSFT",
            "title": "MICROSOFT CORP",
            "exchange": "",
        }
    ]


def test_build_universe_cik_mapping_matches_share_class_variants():
    universe = [
        {
            "symbol": "BRK.B",
            "name": "Berkshire Hathaway Inc. Class B",
            "exchange": "NYSE",
            "liquidity_rank": "1",
            "median_dollar_volume": "1000",
        }
    ]
    sec_rows = [
        {
            "cik": "0001067983",
            "ticker": "BRK-B",
            "title": "Berkshire Hathaway Inc.",
            "exchange": "NYSE",
        }
    ]

    rows = _build_universe_cik_mapping(universe, sec_rows)

    assert rows[0]["match_status"] == "matched"
    assert rows[0]["cik"] == "0001067983"
    assert rows[0]["sec_ticker"] == "BRK-B"


def test_ticker_keys_include_dot_and_dash_variants():
    assert set(_ticker_keys("BRK.B")) >= {"BRK.B", "BRK-B"}


def test_summarize_submission_file_extracts_recent_and_older_files(tmp_path):
    path = tmp_path / "CIK0000320193.json"
    path.write_text(
        json.dumps(
            {
                "filings": {
                    "recent": {
                        "filingDate": ["2026-01-31", "2025-10-31"],
                    },
                    "files": [
                        {"name": "CIK0000320193-submissions-001.json"},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    row = _summarize_submission_file(
        symbol="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        liquidity_rank="1",
        path=path,
        status="downloaded",
        error="",
    )

    assert row["recent_filing_count"] == 2
    assert row["oldest_recent_filing_date"] == "2025-10-31"
    assert row["newest_recent_filing_date"] == "2026-01-31"
    assert row["older_file_count"] == 1
    assert row["has_older_history_files"] is True


def test_load_membership_universe_filters_variant_and_short_eligible(tmp_path):
    membership_path = tmp_path / "membership.csv.gz"
    pd.DataFrame(
        [
            {
                "variant": "adv30m_clean_core_beta_full",
                "session_date": "2014-08-05",
                "symbol": "AAPL",
                "short_eligible_proxy": True,
                "liquidity_rank": 1,
            },
            {
                "variant": "adv30m_clean_core_beta_full",
                "session_date": "2014-08-06",
                "symbol": "AAPL",
                "short_eligible_proxy": True,
                "liquidity_rank": 2,
            },
            {
                "variant": "adv30m_clean_core_beta_full",
                "session_date": "2014-08-05",
                "symbol": "XYZ",
                "short_eligible_proxy": False,
                "liquidity_rank": 10,
            },
            {
                "variant": "top500_clean_core_beta_full",
                "session_date": "2014-08-05",
                "symbol": "MSFT",
                "short_eligible_proxy": True,
                "liquidity_rank": 1,
            },
        ]
    ).to_csv(membership_path, index=False, compression="gzip")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "symbol,name,exchange,liquidity_rank,median_dollar_volume\n"
        "AAPL,Apple Inc.,NASDAQ,6,1000\n",
        encoding="utf-8",
    )

    rows = _load_membership_universe(
        membership_path=membership_path,
        variant="adv30m_clean_core_beta_full",
        enrichment_manifest_path=manifest_path,
        short_eligible_only=True,
    )

    assert rows == [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "exchange": "NASDAQ",
            "liquidity_rank": "6",
            "median_dollar_volume": "1000",
            "universe_rows": 2,
            "first_session": "2014-08-05",
            "last_session": "2014-08-06",
            "source_universe": "adv30m_clean_core_beta_full",
        }
    ]
