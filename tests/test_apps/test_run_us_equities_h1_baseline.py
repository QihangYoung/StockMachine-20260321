from __future__ import annotations

import json

import pandas as pd

from stockmachine.apps import run_us_equities_h1_baseline as h1_app


def _build_dataset() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=120)
    daily_rows: list[dict[str, object]] = []
    benchmark_rows: list[dict[str, object]] = []
    adj_rows: list[dict[str, object]] = []
    universe_rows: list[dict[str, object]] = []
    for index, current_date in enumerate(dates):
        benchmark_price = 500.0 + index * 0.5
        benchmark_rows.append(
            {
                "session_date": current_date,
                "symbol": "SPY",
                "open": benchmark_price,
                "high": benchmark_price * 1.01,
                "low": benchmark_price * 0.99,
                "close": benchmark_price * 1.001,
                "volume": 1_000_000.0,
            }
        )
        for symbol, offset in [("AAA", 0.0), ("BBB", 10.0), ("CCC", 20.0)]:
            price = 100.0 + offset + index * (1.0 + offset / 100.0)
            daily_rows.append(
                {
                    "session_date": current_date,
                    "symbol": symbol,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price * (1.001 + offset / 10000.0),
                    "volume": 1_500_000.0 + offset * 1000 + index * 100,
                }
            )
            adj_rows.append(
                {
                    "session_date": current_date,
                    "symbol": symbol,
                    "split_factor": 1.0,
                    "cash_dividend": 0.0,
                    "price_adjust_factor": 1.0,
                }
            )
            universe_rows.append(
                {
                    "session_date": current_date,
                    "symbol": symbol,
                    "universe_name": "us_equities_research_v1",
                }
            )

    symbol_master = pd.DataFrame(
        {
            "as_of_date": [dates[-1]] * 3,
            "symbol": ["AAA", "BBB", "CCC"],
            "company_name": ["AAA", "BBB", "CCC"],
            "quote_type": ["CS"] * 3,
            "exchange": ["XNYS", "XNAS", "XNYS"],
            "currency": ["USD"] * 3,
            "country": ["US"] * 3,
        }
    )
    industry_membership = pd.DataFrame(
        {
            "as_of_date": [dates[-1]] * 3,
            "symbol": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Health", "Industrials"],
            "industry": ["Software", "Biotech", "Machinery"],
            "industry_sector_override": ["Tech", "Health", "Industrials"],
            "industry_name_override": ["Software", "Biotech", "Machinery"],
        }
    )
    return {
        "daily_bar": pd.DataFrame(daily_rows),
        "benchmark_index": pd.DataFrame(benchmark_rows),
        "symbol_master": symbol_master,
        "industry_membership": industry_membership,
        "universe_membership": pd.DataFrame(universe_rows),
        "adj_factor": pd.DataFrame(adj_rows),
    }


def test_run_us_equities_h1_baseline_writes_expected_outputs(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "stockmachine.research.h1_us_equities.load_us_equities_dataset",
        lambda layout=None: _build_dataset(),
    )
    monkeypatch.setattr(
        "stockmachine.research.h1_us_equities.build_point_in_time_metadata_history",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "symbol": ["AAA", "BBB", "CCC"],
                "sector": ["Tech", "Health", "Industrials"],
                "industry": ["Software", "Biotech", "Machinery"],
            }
        ),
    )

    exit_code = h1_app.main(
        [
            "--predict-start",
            "2024-03-01",
            "--top-k",
            "2",
            "--output-root",
            str(tmp_path / "h1"),
            "--train-window-days",
            "40",
            "--validation-window-days",
            "20",
            "--test-window-days",
            "20",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["strategy_project"] == "us_equities_h1"
    assert (tmp_path / "h1" / "summary_metrics.csv").exists()
    assert (tmp_path / "h1" / "yearly_summary.csv").exists()
    assert (tmp_path / "h1" / "cost_stress_summary.csv").exists()
    assert (tmp_path / "h1" / "benchmark_summary.csv").exists()
    assert (tmp_path / "h1" / "promotion_gate.json").exists()
    summary = pd.read_csv(tmp_path / "h1" / "summary_metrics.csv")
    assert set(summary["model"]) == {"ridge", "hist_gbm", "extra_trees"}
