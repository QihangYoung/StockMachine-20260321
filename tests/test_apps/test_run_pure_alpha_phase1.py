from __future__ import annotations

import json

import pandas as pd

from stockmachine.apps.run_pure_alpha_phase1 import build_phase1_universe_artifacts


def test_phase1_universe_builder_filters_clean_core_and_short_proxy(tmp_path) -> None:
    membership_path = tmp_path / "membership.csv.gz"
    asset_qa_path = tmp_path / "asset_qa.csv"
    _membership_frame().to_csv(membership_path, index=False, compression="gzip")
    _asset_qa_frame().to_csv(asset_qa_path, index=False)

    result = build_phase1_universe_artifacts(
        membership_path=membership_path,
        asset_qa_by_symbol_path=asset_qa_path,
        phase0_closure_rollup_path=tmp_path / "missing_phase0_closure.json",
        output_root=tmp_path / "phase1",
        min_long_short_names=1,
    )

    membership = pd.read_csv(
        tmp_path / "phase1" / "phase1_candidate_universe_membership_validation.csv.gz"
    )
    summary = pd.read_csv(tmp_path / "phase1" / "phase1_candidate_universe_summary_validation.csv")

    first_day = membership[
        (membership["variant"] == "top500_clean_core_beta_full")
        & (membership["session_date"] == "2020-01-02")
    ]
    assert set(first_day["symbol"]) == {"AAA", "CCC"}
    assert "BBB" not in set(membership["symbol"])
    assert first_day.loc[first_day["symbol"] == "CCC", "short_eligible_proxy"].iloc[0] == False
    assert result["recommended_validation_default"] == "top500_clean_core_beta_full"
    assert result["lockbox_status"] == "validation_only_no_test_window_performance"
    assert summary.loc[
        summary["variant"] == "top1500_clean_core_beta_full",
        "data_status",
    ].iloc[0] == "blocked_current_backfill_has_only_top1000_symbols"


def test_phase1_universe_builder_requires_lagged_beta_full_proxy(tmp_path) -> None:
    membership_path = tmp_path / "membership.csv.gz"
    asset_qa_path = tmp_path / "asset_qa.csv"
    frame = _membership_frame()
    frame.loc[frame["symbol"] == "AAA", "beta_full_lookback_proxy_ready"] = False
    frame.to_csv(membership_path, index=False, compression="gzip")
    _asset_qa_frame().to_csv(asset_qa_path, index=False)

    build_phase1_universe_artifacts(
        membership_path=membership_path,
        asset_qa_by_symbol_path=asset_qa_path,
        phase0_closure_rollup_path=tmp_path / "missing_phase0_closure.json",
        output_root=tmp_path / "phase1",
        min_long_short_names=1,
    )
    membership = pd.read_csv(
        tmp_path / "phase1" / "phase1_candidate_universe_membership_validation.csv.gz"
    )

    assert "AAA" not in set(membership["symbol"])
    assert set(membership["symbol"]) == {"CCC"}


def test_phase1_universe_builder_writes_memo_and_rollup(tmp_path) -> None:
    membership_path = tmp_path / "membership.csv.gz"
    asset_qa_path = tmp_path / "asset_qa.csv"
    closure_path = tmp_path / "phase0_closure.json"
    _membership_frame().to_csv(membership_path, index=False, compression="gzip")
    _asset_qa_frame().to_csv(asset_qa_path, index=False)
    closure_path.write_text(
        json.dumps({"phase0_status": "local_phase0_complete_for_phase1_plumbing"}),
        encoding="utf-8",
    )

    result = build_phase1_universe_artifacts(
        membership_path=membership_path,
        asset_qa_by_symbol_path=asset_qa_path,
        phase0_closure_rollup_path=closure_path,
        output_root=tmp_path / "phase1",
        min_long_short_names=1,
    )

    assert (tmp_path / "phase1" / "phase1_universe_builder_memo.md").exists()
    assert (tmp_path / "phase1" / "phase1_universe_builder_rollup.json").exists()
    assert result["clean_core_symbols"] == 2


def _membership_frame() -> pd.DataFrame:
    rows = []
    for session in ("2020-01-02", "2020-01-03"):
        rows.extend(
            [
                _membership_row(session, "AAA", liquidity_rank=1, dollar_volume=100_000_000),
                _membership_row(session, "BBB", liquidity_rank=2, dollar_volume=90_000_000),
                _membership_row(session, "CCC", liquidity_rank=3, dollar_volume=40_000_000),
            ]
        )
    return pd.DataFrame(rows)


def _membership_row(
    session_date: str,
    symbol: str,
    *,
    liquidity_rank: int,
    dollar_volume: float,
) -> dict[str, object]:
    return {
        "session_date": session_date,
        "symbol": symbol,
        "source_segment": "unit_test",
        "lagged_close": 20.0,
        "trailing_obs_20": 20,
        "trailing_median_dollar_volume_20": dollar_volume,
        "liquidity_rank": liquidity_rank,
        "beta_warmup_proxy_ready": True,
        "beta_full_lookback_proxy_ready": True,
        "in_top500": True,
        "in_top1000": True,
        "in_top1500": True,
        "in_top2000": True,
        "in_top3000": True,
        "adv_ge_10m": dollar_volume >= 10_000_000,
        "adv_ge_20m": dollar_volume >= 20_000_000,
        "adv_ge_30m": dollar_volume >= 30_000_000,
        "adv_ge_50m": dollar_volume >= 50_000_000,
    }


def _asset_qa_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "classification": [
                "common_stock_candidate",
                "review_foreign_or_adr_like",
                "common_stock_candidate",
            ],
            "flags": ["", "adr_or_foreign_issuer_name_pattern", ""],
            "review_required": [False, True, False],
            "tradable": [True, True, True],
            "marginable": [True, True, True],
            "shortable": [True, True, True],
            "easy_to_borrow": [True, True, False],
        }
    )
