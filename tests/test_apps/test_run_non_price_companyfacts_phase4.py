from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stockmachine.apps.run_non_price_companyfacts_phase4 import (
    RAW_FACT_COLUMNS,
    _build_companyfacts_events,
    _build_fundamental_panel,
    _cached_companyfacts_manifest,
    _derive_fundamental_features,
)


def _usd_fact(value: float, *, tag: str | None = None) -> dict[str, object]:
    fact = {
        "filed": "2019-01-02",
        "end": "2018-12-31",
        "form": "10-K",
        "fy": 2018,
        "fp": "FY",
        "accn": "0000000000-19-000001",
        "val": value,
    }
    payload: dict[str, object] = {"units": {"USD": [fact]}}
    if tag:
        payload["label"] = tag
    return payload


def test_build_companyfacts_events_aligns_next_session_and_prefers_tag_priority(tmp_path):
    cik = "0000320193"
    payload = {
        "facts": {
            "us-gaap": {
                "Assets": _usd_fact(100.0),
                "Liabilities": _usd_fact(60.0),
                "CashAndCashEquivalentsAtCarryingValue": _usd_fact(20.0),
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": _usd_fact(99.0),
                "ShortTermDebt": _usd_fact(5.0),
                "LongTermDebtNoncurrent": _usd_fact(25.0),
                "Revenues": _usd_fact(200.0),
                "NetIncomeLoss": _usd_fact(-10.0),
                "NetCashProvidedByUsedInOperatingActivities": _usd_fact(-5.0),
            }
        }
    }
    (tmp_path / f"CIK{cik}.json").write_text(json.dumps(payload), encoding="utf-8")
    mapping = pd.DataFrame([{"symbol": "AAPL", "cik": cik}])

    events = _build_companyfacts_events(
        mapping=mapping,
        raw_root=tmp_path,
        sessions=["2019-01-02", "2019-01-03", "2019-01-04"],
    )

    assert len(events) == 1
    row = events.iloc[0]
    assert row["session_date"] == "2019-01-03"
    assert row["assets"] == 100.0
    assert row["cash"] == 20.0
    assert row["current_debt"] == 5.0
    assert row["noncurrent_debt"] == 25.0


def test_build_fundamental_panel_carries_latest_pit_state_forward():
    membership = pd.DataFrame(
        [
            {"session_date": "2019-01-02", "symbol": "AAPL"},
            {"session_date": "2019-01-03", "symbol": "AAPL"},
            {"session_date": "2019-01-04", "symbol": "AAPL"},
        ]
    )
    fact_events = pd.DataFrame(
        [
            {
                "session_date": "2019-01-03",
                "symbol": "AAPL",
                "cik": "0000320193",
                "filed_date": "2019-01-02",
                "period_end": "2018-12-31",
                "form": "10-K",
                "fy": "2018",
                "fp": "FY",
                "accession": "0000000000-19-000001",
                "assets": 100.0,
                "liabilities": 60.0,
                "stockholders_equity": 40.0,
                "cash": 20.0,
                "cash_and_short_term_investments": np.nan,
                "current_debt": 5.0,
                "noncurrent_debt": 25.0,
                "revenue": 200.0,
                "operating_income": np.nan,
                "net_income": -10.0,
                "operating_cash_flow": -5.0,
            }
        ]
    )

    panel = _build_fundamental_panel(membership=membership, fact_events=fact_events)

    before_event = panel.loc[panel["session_date"].eq("2019-01-02")].iloc[0]
    after_event = panel.loc[panel["session_date"].eq("2019-01-04")].iloc[0]
    assert pd.isna(before_event["assets"])
    assert after_event["last_fundamental_filed_date"] == "2019-01-02"
    assert after_event["liabilities_to_assets"] == pytest.approx(0.6)
    assert after_event["total_debt"] == pytest.approx(30.0)
    assert pd.notna(after_event["fundamental_fragility_score"])
    assert not bool(after_event["test_window_used"])


def test_derive_fundamental_features_avoids_infinite_ratios_and_missing_loss_flags():
    row = {column: np.nan for column in RAW_FACT_COLUMNS}
    row.update({"session_date": "2019-01-03", "symbol": "AAPL", "assets": 0.0, "liabilities": 10.0})

    derived = _derive_fundamental_features(pd.DataFrame([row]))

    assert pd.isna(derived.loc[0, "liabilities_to_assets"])
    assert pd.isna(derived.loc[0, "negative_net_income_flag"])
    assert pd.isna(derived.loc[0, "negative_operating_cash_flow_flag"])


def test_cached_manifest_marks_corrupt_companyfacts_json_invalid(tmp_path):
    cik = "0000320193"
    (tmp_path / f"CIK{cik}.json").write_text("{", encoding="utf-8")
    mapping = pd.DataFrame([{"symbol": "AAPL", "cik": cik}])

    manifest = _cached_companyfacts_manifest(mapping=mapping, raw_root=tmp_path)
    events = _build_companyfacts_events(
        mapping=mapping,
        raw_root=tmp_path,
        sessions=["2019-01-02", "2019-01-03"],
    )

    assert manifest.loc[0, "download_status"] == "invalid_cache"
    assert not bool(manifest.loc[0, "valid_json"])
    assert "json_decode_error" in manifest.loc[0, "error"]
    assert events.empty
