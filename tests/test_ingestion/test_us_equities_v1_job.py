from datetime import date

from stockmachine.ingestion.jobs import us_equities_v1


def test_collect_research_seed_chunks_default_universe(monkeypatch) -> None:
    calls = []
    adj_calls = []

    def fake_symbol_master(*, layout=None):
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
    assert result["included_adj_factor"] == 1
