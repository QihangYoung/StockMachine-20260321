from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase0 import (
    build_pit_universe_feasibility,
    classify_asset,
    run_asset_class_qa,
    write_phase0_closure_artifacts,
    write_vendor_bakeoff_artifacts,
)


def test_build_pit_universe_feasibility_uses_lagged_liquidity(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"symbol": ["AAA", "BBB"]}).to_csv(manifest_path, index=False)
    daily_path = tmp_path / "daily.jsonl"
    rows = []
    start = date(2020, 1, 1)
    for index in range(6):
        session = (start + timedelta(days=index)).isoformat()
        rows.append(_bar(session, "AAA", close=20, dollar_volume=100 + index * 10))
        rows.append(_bar(session, "BBB", close=20, dollar_volume=1_000 + index * 10))
    _write_jsonl(daily_path, rows)

    result = build_pit_universe_feasibility(
        manifest_path=manifest_path,
        daily_globs=(daily_path,),
        output_root=tmp_path / "pit",
        validation_start="2020-01-01",
        validation_end="2020-01-06",
        lookback_sessions=3,
        min_observations=2,
        top_buckets=(1, 2),
        adv_thresholds=(500,),
        beta_warmup_observations=2,
        beta_full_observations=3,
    )

    daily = pd.read_csv(tmp_path / "pit" / "candidate_universe_daily_counts_validation.csv")
    membership = pd.read_csv(
        tmp_path / "pit" / "provisional_current_top1000_lagged_liquidity_membership_validation.csv.gz"
    )
    first_membership = membership[membership["session_date"] == "2020-01-03"]

    assert result["validation_rows_loaded"] == 12
    assert daily.loc[daily["session_date"] == "2020-01-01", "liquidity_eligible_count"].iloc[0] == 0
    assert set(first_membership["symbol"]) == {"AAA", "BBB"}
    assert first_membership.loc[first_membership["symbol"] == "BBB", "liquidity_rank"].iloc[0] == 1
    assert "top1_count" in daily.columns
    assert result["key_findings"]["median_top1000_count"] is None


def test_asset_class_qa_flags_review_candidates(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "symbol": ["AAPL", "SPY", "TSM", "BRK.B"],
            "name": [
                "Apple Inc.",
                "SPDR S&P 500 ETF Trust",
                "Taiwan Semiconductor Manufacturing Company Limited",
                "Berkshire Hathaway Inc.",
            ],
            "liquidity_rank": [1, 2, 3, 4],
            "exchange": ["NASDAQ", "ARCA", "NYSE", "NYSE"],
            "tradable": [True, True, True, True],
            "marginable": [True, True, True, True],
            "shortable": [True, True, True, True],
            "easy_to_borrow": [True, True, True, True],
        }
    ).to_csv(manifest_path, index=False)
    asset_path = tmp_path / "assets.csv"
    pd.DataFrame(
        {
            "symbol": ["AAPL", "SPY", "TSM", "BRK.B"],
            "name": ["Apple Inc.", "SPDR S&P 500 ETF Trust", "TSMC Limited", "Berkshire Hathaway Inc."],
            "exchange": ["NASDAQ", "ARCA", "NYSE", "NYSE"],
            "tradable": [True, True, True, True],
            "marginable": [True, True, True, True],
            "shortable": [True, True, True, True],
            "easy_to_borrow": [True, True, True, True],
            "phase0_filter_reason": ["kept", "excluded_name_term", "kept", "kept"],
        }
    ).to_csv(asset_path, index=False)

    result = run_asset_class_qa(
        manifest_path=manifest_path,
        asset_audit_path=asset_path,
        output_root=tmp_path / "qa",
    )
    qa = pd.read_csv(tmp_path / "qa" / "top1000_asset_class_qa_by_symbol.csv")

    assert result["symbols_checked"] == 4
    assert qa.loc[qa["symbol"] == "AAPL", "classification"].iloc[0] == "common_stock_candidate"
    assert bool(qa.loc[qa["symbol"] == "AAPL", "shortable"].iloc[0]) is True
    assert bool(qa.loc[qa["symbol"] == "AAPL", "easy_to_borrow"].iloc[0]) is True
    assert qa.loc[qa["symbol"] == "SPY", "classification"].iloc[0] == "blocked_non_common_like"
    assert qa.loc[qa["symbol"] == "TSM", "classification"].iloc[0] == "review_foreign_or_adr_like"
    assert "class_share_symbol" in qa.loc[qa["symbol"] == "BRK.B", "flags"].iloc[0]


def test_vendor_bakeoff_artifacts_are_written(tmp_path) -> None:
    result = write_vendor_bakeoff_artifacts(output_root=tmp_path / "vendors")

    assert result["candidate_vendors"] >= 4
    assert (tmp_path / "vendors" / "vendor_bakeoff_candidates.csv").exists()
    assert (tmp_path / "vendors" / "vendor_bakeoff_acceptance_checks.csv").exists()
    assert (tmp_path / "vendors" / "vendor_bakeoff_plan.md").exists()


def test_phase0_closure_artifacts_are_written(tmp_path) -> None:
    asset_qa_path = tmp_path / "asset_qa.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "classification": ["common_stock_candidate", "review_foreign_or_adr_like"],
            "flags": ["", "adr_or_foreign_issuer_name_pattern"],
            "review_required": [False, True],
            "shortable": [True, True],
            "easy_to_borrow": [True, False],
        }
    ).to_csv(asset_qa_path, index=False)

    result = write_phase0_closure_artifacts(
        output_root=tmp_path / "closure",
        top1000_coverage_rollup_path=tmp_path / "missing_top1000.json",
        yahoo_gapfill_summary_path=tmp_path / "missing_yahoo.json",
        yahoo_reconciliation_summary_path=tmp_path / "missing_recon.json",
        pit_rollup_path=tmp_path / "missing_pit.json",
        asset_qa_rollup_path=tmp_path / "missing_asset_rollup.json",
        asset_qa_by_symbol_path=asset_qa_path,
        vendor_rollup_path=tmp_path / "missing_vendor.json",
    )

    assert result["phase0_status"] == "local_phase0_complete_for_phase1_plumbing"
    assert result["asset_policy_counts"]["default_core_symbols"] == 1
    assert result["asset_policy_counts"]["easy_to_borrow_symbols"] == 1
    assert "top1000_coverage" in result["missing_inputs"]
    assert (tmp_path / "closure" / "phase0_closure_memo.md").exists()
    assert (tmp_path / "closure" / "phase0_phase1_blockers.csv").exists()


def test_classify_asset_keeps_plain_common_stock() -> None:
    classification, flags = classify_asset(symbol="MSFT", name="Microsoft Corporation")

    assert classification == "common_stock_candidate"
    assert flags == []


def _bar(session_date: str, symbol: str, *, close: float, dollar_volume: float) -> dict[str, object]:
    return {
        "session_date": session_date,
        "symbol": symbol,
        "close": close,
        "volume": dollar_volume / close,
        "dollar_volume": dollar_volume,
    }


def _write_jsonl(path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
