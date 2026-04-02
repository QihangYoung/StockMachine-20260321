from datetime import date

from stockmachine.ingestion.jobs import us_equities_v1


def test_collect_research_seed_chunks_default_universe(monkeypatch) -> None:
    calls = []
    adj_calls = []
    symbol_master_calls = []
    membership_calls = []

    def fake_symbol_master(*, layout=None, snapshot_date=None):
        symbol_master_calls.append({"layout": layout, "snapshot_date": snapshot_date})
        return {"raw_records": 10, "normalized_rows": 10}

    def fake_daily_bars(symbols, **kwargs):
        calls.append(list(symbols))
        return {"raw_records": len(symbols), "normalized_rows": len(symbols)}

    def fake_adj_factors(symbols, **kwargs):
        adj_calls.append(list(symbols))
        return {
            "raw_bar_records": len(symbols),
            "adjusted_bar_records": len(symbols),
            "corporate_action_records": 1,
            "normalized_rows": len(symbols),
        }

    monkeypatch.setattr(us_equities_v1, "collect_symbol_master_snapshot", fake_symbol_master)
    monkeypatch.setattr(us_equities_v1, "collect_daily_bars", fake_daily_bars)
    monkeypatch.setattr(us_equities_v1, "collect_adj_factors", fake_adj_factors)
    monkeypatch.setattr(
        us_equities_v1,
        "refresh_research_membership_history",
        lambda **kwargs: membership_calls.append(kwargs) or {
            "session_dates": 5,
            "industry_membership_rows": 20,
            "universe_membership_rows": 20,
        },
    )

    result = us_equities_v1.collect_research_seed(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        chunk_size=20,
    )

    expected_symbols = len(us_equities_v1.DEFAULT_UNIVERSE) + 1
    assert result["symbols"] == expected_symbols
    assert result["chunks"] == 4
    assert sum(len(call) for call in calls) == expected_symbols
    assert sum(len(call) for call in adj_calls) == expected_symbols
    assert calls[-1][-1] == us_equities_v1.BENCHMARK_SYMBOL
    assert len(symbol_master_calls) == 1
    assert symbol_master_calls[0]["snapshot_date"] == date(2025, 1, 31)
    assert len(membership_calls) == 1
    assert membership_calls[0]["start_date"] == date(2025, 1, 1)
    assert membership_calls[0]["end_date"] == date(2025, 1, 31)
    assert result["included_adj_factor"] == 1
    assert result["included_daily_bars"] == 1
    assert result["industry_membership_rows"] == 20
    assert result["universe_membership_rows"] == 20
    assert result["membership_session_dates"] == 5
