import pandas as pd

from stockmachine.research.us_equities_baseline import _select_top_k_with_sector_cap, _weight_turnover


def test_weight_turnover_handles_full_replace() -> None:
    previous = {"AAPL": 0.5, "MSFT": 0.5}
    new = {"JPM": 0.5, "XOM": 0.5}
    assert _weight_turnover(previous, new) == 2.0


def test_select_top_k_with_sector_cap_limits_each_sector() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "sector": ["Tech", "Tech", "Tech", "Health", "Finance"],
            "final_score": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    selected = _select_top_k_with_sector_cap(frame, top_k=4, max_positions_per_sector=2)
    assert len(selected) == 4
    assert (selected["sector"] == "Tech").sum() == 2
