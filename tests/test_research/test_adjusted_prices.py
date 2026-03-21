import pandas as pd

from stockmachine.research.us_equities_baseline import _ensure_adjusted_price_columns


def test_ensure_adjusted_price_columns_infers_factor_from_adj_close() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "open": [100.0, 100.0],
            "close": [100.0, 100.0],
            "adj_close": [100.0, 105.0],
            "volume": [1_000_000.0, 1_000_000.0],
            "dividends": [0.0, 5.0],
            "stock_splits": [0.0, 0.0],
        }
    )

    adjusted = _ensure_adjusted_price_columns(frame)

    assert adjusted.loc[0, "price_adjust_factor"] == 1.0
    assert adjusted.loc[1, "price_adjust_factor"] == 1.05
    assert adjusted.loc[1, "adj_open"] == 105.0
    assert adjusted.loc[1, "cash_dividend"] == 5.0
    assert adjusted.loc[1, "split_factor"] == 1.0
