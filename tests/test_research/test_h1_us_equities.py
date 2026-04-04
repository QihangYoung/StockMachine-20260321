from dataclasses import asdict

import numpy as np
import pandas as pd

from stockmachine.research.h1_us_equities import (
    H1_DEFAULT_FEATURE_VERSION,
    H1TargetConfig,
    H1_FEATURE_COLUMNS_V1,
    H1_FEATURE_COLUMNS_V2,
    H1_FEATURE_COLUMNS_V3,
    H1TurnoverControlConfig,
    build_h1_research_frame,
    fit_predict_h1_base_model,
    resolve_h1_feature_columns,
    run_h1_baseline_sweep,
)
from stockmachine.research.p1_rigor import StrictResearchBundle


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
    assert set(H1_FEATURE_COLUMNS_V1).issubset(frame.columns)
    assert set(H1_FEATURE_COLUMNS_V2).issubset(frame.columns)
    assert {"future_return", "target", "target_bucket_2", "sector", "industry", "median_dollar_volume_20"}.issubset(frame.columns)
    assert frame["future_return"].notna().all()
    assert frame["target"].notna().all()
    assert frame["target_bucket_2"].isin([0, 1]).all()


def test_resolve_h1_feature_columns_supports_v1_and_v2() -> None:
    assert resolve_h1_feature_columns("v1") == H1_FEATURE_COLUMNS_V1
    assert resolve_h1_feature_columns("v2") == H1_FEATURE_COLUMNS_V2
    assert resolve_h1_feature_columns("v3") == H1_FEATURE_COLUMNS_V3
    assert resolve_h1_feature_columns() == H1_FEATURE_COLUMNS_V3
    assert H1_DEFAULT_FEATURE_VERSION == "v3"


def test_fit_predict_h1_base_model_emits_binary_classification_scores() -> None:
    dates = pd.bdate_range("2025-01-01", periods=55)
    rows = []
    for symbol, offset in [("AAA", 0.0), ("BBB", 5.0), ("CCC", -3.0), ("SPY", 2.0)]:
        for index, current_date in enumerate(dates):
            drift = 1.0 + offset / 100.0
            price = 100.0 + offset + index * drift
            rows.append(
                {
                    "date": current_date,
                    "symbol": symbol,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price * (1.001 + offset / 10000.0),
                    "volume": 1_200_000.0 + index * 500 + abs(offset) * 1000,
                    "adj_open": price,
                    "adj_close": price * (1.001 + offset / 10000.0),
                }
            )
    price_data = pd.DataFrame(rows)
    metadata = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Health", "Industrials"],
            "industry": ["Software", "Biotech", "Machinery"],
        }
    )
    frame = build_h1_research_frame(price_data, benchmark_symbol="SPY", symbol_metadata=metadata)
    train_frame = frame.loc[frame["date"] < frame["date"].quantile(0.7)].copy()
    test_frame = frame.loc[frame["date"] >= frame["date"].quantile(0.7)].copy()

    predictions = fit_predict_h1_base_model(
        "hist_gbm",
        train_frame=train_frame,
        test_frame=test_frame,
        target_config=H1TargetConfig(),
    )

    assert not predictions.empty
    assert {"probability_positive", "predicted_bucket", "classification_confidence", "target_bucket_2"}.issubset(
        predictions.columns
    )
    assert np.isfinite(predictions["score"]).all()
    assert ((predictions["probability_positive"] >= 0.0) & (predictions["probability_positive"] <= 1.0)).all()
    assert predictions["predicted_bucket"].isin([0, 1]).all()


def test_fit_predict_h1_base_model_supports_feature_version_v3() -> None:
    dates = pd.bdate_range("2025-01-01", periods=55)
    rows = []
    for symbol, offset in [("AAA", 0.0), ("BBB", 5.0), ("CCC", -3.0), ("SPY", 2.0)]:
        for index, current_date in enumerate(dates):
            drift = 1.0 + offset / 100.0
            price = 100.0 + offset + index * drift
            rows.append(
                {
                    "date": current_date,
                    "symbol": symbol,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price * (1.001 + offset / 10000.0),
                    "volume": 1_200_000.0 + index * 500 + abs(offset) * 1000,
                    "adj_open": price,
                    "adj_close": price * (1.001 + offset / 10000.0),
                }
            )
    price_data = pd.DataFrame(rows)
    metadata = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Health", "Industrials"],
            "industry": ["Software", "Biotech", "Machinery"],
        }
    )
    frame = build_h1_research_frame(price_data, benchmark_symbol="SPY", symbol_metadata=metadata)
    train_frame = frame.loc[frame["date"] < frame["date"].quantile(0.7)].copy()
    test_frame = frame.loc[frame["date"] >= frame["date"].quantile(0.7)].copy()

    predictions = fit_predict_h1_base_model(
        "hist_gbm",
        train_frame=train_frame,
        test_frame=test_frame,
        target_config=H1TargetConfig(),
        feature_version="v3",
    )

    assert not predictions.empty
    assert np.isfinite(predictions["score"]).all()
    assert predictions["predicted_bucket"].isin([0, 1]).all()


def test_fit_predict_h1_base_model_supports_point_regression() -> None:
    dates = pd.bdate_range("2025-01-01", periods=55)
    rows = []
    for symbol, offset in [("AAA", 0.0), ("BBB", 5.0), ("CCC", -3.0), ("SPY", 2.0)]:
        for index, current_date in enumerate(dates):
            drift = 1.0 + offset / 100.0
            price = 100.0 + offset + index * drift
            rows.append(
                {
                    "date": current_date,
                    "symbol": symbol,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price * (1.001 + offset / 10000.0),
                    "volume": 1_200_000.0 + index * 500 + abs(offset) * 1000,
                    "adj_open": price,
                    "adj_close": price * (1.001 + offset / 10000.0),
                }
            )
    price_data = pd.DataFrame(rows)
    metadata = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Health", "Industrials"],
            "industry": ["Software", "Biotech", "Machinery"],
        }
    )
    frame = build_h1_research_frame(price_data, benchmark_symbol="SPY", symbol_metadata=metadata)
    train_frame = frame.loc[frame["date"] < frame["date"].quantile(0.7)].copy()
    test_frame = frame.loc[frame["date"] >= frame["date"].quantile(0.7)].copy()

    predictions = fit_predict_h1_base_model(
        "ridge",
        train_frame=train_frame,
        test_frame=test_frame,
        target_config=H1TargetConfig(task="point_regression"),
    )

    assert not predictions.empty
    assert "target_bucket_2" not in predictions.columns
    assert np.isfinite(predictions["score"]).all()
    assert predictions["probability_positive"].isna().all()
    assert predictions["predicted_bucket"].isin([0, 1]).all()


def test_run_h1_baseline_sweep_forwards_turnover_control_to_backtest(tmp_path, monkeypatch) -> None:
    predictions = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-01-02"),
                "symbol": "AAA",
                "model": "ridge",
                "score": 0.8,
                "confidence": 0.9,
            }
        ]
    )
    bundle = StrictResearchBundle(
        predict_start="2025-01-01",
        horizon=1,
        strategy_project="us_equities_h1",
        framework_id="us_equities_h1_strict_v1",
        dataset={"benchmark_index": pd.DataFrame()},
        research_frame=pd.DataFrame([{"date": pd.Timestamp("2025-01-02"), "symbol": "AAA"}]),
        predictions=predictions,
    )
    captured: dict[str, object] = {}

    def _build_strict_research_bundle(**kwargs):
        captured["prediction_options"] = kwargs.get("prediction_options")
        return bundle

    monkeypatch.setattr(
        "stockmachine.research.h1_us_equities.build_strict_research_bundle",
        _build_strict_research_bundle,
    )

    def _run_model_backtest_from_bundle(*args, **kwargs):
        captured["turnover_control_overrides"] = kwargs.get("turnover_control_overrides")
        return {
            "summary": {
                "sessions": 1,
                "total_return": 0.01,
                "annualized_return": 0.01,
                "annualized_volatility": 0.1,
                "sharpe": 0.5,
                "max_drawdown": -0.1,
                "benchmark_total_return": 0.0,
                "mean_turnover": 0.2,
                "mean_cost_bps": 2.0,
            },
            "records": pd.DataFrame(
                [
                    {
                        "signal_date": "2025-01-02",
                        "entry_date": "2025-01-03",
                        "exit_date": "2025-01-06",
                        "gross_return": 0.011,
                        "net_return": 0.01,
                        "benchmark_return": 0.0,
                        "turnover": 0.2,
                        "cost_bps": 2.0,
                        "positions": 1,
                    }
                ]
            ),
        }

    monkeypatch.setattr(
        "stockmachine.research.h1_us_equities.run_model_backtest_from_bundle",
        _run_model_backtest_from_bundle,
    )
    monkeypatch.setattr(
        "stockmachine.research.h1_us_equities.build_period_stability_summary",
        lambda *args, **kwargs: pd.DataFrame([{"model": "ridge", "period": "year", "period_label": "2025"}]),
    )
    monkeypatch.setattr(
        "stockmachine.research.h1_us_equities.build_cost_stress_summary",
        lambda *args, **kwargs: pd.DataFrame(
            [{"model": "ridge", "cost_bps_per_side": 20.0, "annualized_return": 0.005}]
        ),
    )
    monkeypatch.setattr(
        "stockmachine.research.h1_us_equities.build_h1_benchmark_summary",
        lambda **kwargs: pd.DataFrame([{"benchmark": "spy_next_open_hold"}]),
    )
    monkeypatch.setattr(
        "stockmachine.research.h1_us_equities.build_framework_promotion_gate",
        lambda *args, **kwargs: {"models": []},
    )

    turnover_control = H1TurnoverControlConfig(
        no_trade_band=0.08,
        max_turnover=0.6,
        min_weight_change=0.03,
        hold_rank_buffer=3,
        entry_rank_buffer=1,
        max_new_names_per_rebalance=1,
    )

    run_h1_baseline_sweep(
        predict_start="2025-01-01",
        model_names=("ridge",),
        output_dir=tmp_path,
        turnover_control=turnover_control,
        feature_version="v2",
    )

    assert captured["turnover_control_overrides"] == asdict(turnover_control)
    assert captured["prediction_options"]["feature_version"] == "v2"
