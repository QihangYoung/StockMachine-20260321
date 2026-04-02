import pandas as pd

from stockmachine.research.h1_us_equities import H1_FEATURE_COLUMNS, build_h1_research_frame


def test_build_h1_research_frame_produces_expected_columns() -> None:
    dates = pd.bdate_range("2025-01-01", periods=40)
    rows = []
    for symbol, offset in [("AAA", 0.0), ("BBB", 5.0), ("SPY", 2.5)]:
        for index, current_date in enumerate(dates):
            price = 100.0 + offset + index
            rows.append(
                {
                    "date": current_date,
                    "symbol": symbol,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price * 1.002,
                    "volume": 1_000_000.0 + index * 1000,
                    "adj_open": price,
                    "adj_close": price * 1.002,
                }
            )
    price_data = pd.DataFrame(rows)
    metadata = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "sector": ["Tech", "Health"],
            "industry": ["Software", "Biotech"],
        }
    )

    frame = build_h1_research_frame(price_data, benchmark_symbol="SPY", symbol_metadata=metadata)

    assert not frame.empty
    assert set(H1_FEATURE_COLUMNS).issubset(frame.columns)
    assert {"future_return", "target", "sector", "industry", "median_dollar_volume_20"}.issubset(frame.columns)
    assert frame["future_return"].notna().all()
    assert frame["target"].notna().all()
