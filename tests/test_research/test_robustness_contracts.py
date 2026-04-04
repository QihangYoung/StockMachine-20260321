from __future__ import annotations

import pandas as pd
import pytest

from stockmachine.research.robustness_contracts import (
    build_robustness_artifact_bundle,
    ensure_required_backtest_record_columns,
)


def test_ensure_required_backtest_record_columns_normalizes_dates() -> None:
    records = pd.DataFrame(
        [
            {
                "entry_date": "2026-01-02",
                "exit_date": "2026-01-03",
                "net_return": 0.01,
                "benchmark_return": 0.002,
                "turnover": 0.3,
                "cost_bps": 3.0,
                "positions": 8,
            }
        ]
    )

    validated = ensure_required_backtest_record_columns(records)

    assert str(validated["entry_date"].dtype).startswith("datetime64")
    assert str(validated["exit_date"].dtype).startswith("datetime64")


def test_ensure_required_backtest_record_columns_raises_on_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        ensure_required_backtest_record_columns(pd.DataFrame([{"entry_date": "2026-01-02"}]))


def test_build_robustness_artifact_bundle_adds_model_name_to_summary() -> None:
    bundle = build_robustness_artifact_bundle(
        strategy_project="us_equities_h1",
        model_name="ridge",
        records=pd.DataFrame(
            [
                {
                    "entry_date": "2026-01-02",
                    "exit_date": "2026-01-03",
                    "net_return": 0.01,
                    "benchmark_return": 0.002,
                    "turnover": 0.3,
                    "cost_bps": 3.0,
                    "positions": 8,
                }
            ]
        ),
        summary=[{"annualized_return": 0.12}],
    )

    assert bundle.summary["model"].iloc[0] == "ridge"
    assert bundle.strategy_project == "us_equities_h1"

