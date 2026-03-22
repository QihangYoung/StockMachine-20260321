from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from stockmachine.apps import run_us_equities_model_sweep as sweep


def test_model_sweep_writes_summary_csv_and_creates_directories(tmp_path, monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []

    def _run_backtest(**kwargs):
        calls.append(kwargs)
        model_name = kwargs["model_name"]
        return {
            "model": model_name,
            "summary": {
                "sessions": 12,
                "total_return": 0.34,
                "annualized_return": 0.29,
                "annualized_volatility": 0.21,
                "sharpe": 1.36,
                "max_drawdown": -0.18,
                "benchmark_total_return": 0.15,
                "mean_turnover": 1.2,
                "mean_cost_bps": 10.0,
            },
            "artifacts_dir": str(kwargs["output_dir"]),
        }

    monkeypatch.setattr(sweep, "run_silver_chain_backtest", _run_backtest)

    exit_code = sweep.main(
        [
            "--models",
            "hist_gbm",
            "random_forest",
            "--predict-start",
            "2025-01-01",
            "--output-root",
            str(tmp_path / "sweep"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    summary_path = tmp_path / "sweep" / "summary_metrics.csv"
    protocol_path = tmp_path / "sweep" / "research_protocol.json"
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["counts"] == {"requested": 2, "success": 2, "failed": 0}
    assert summary_path.exists()
    assert protocol_path.exists()
    assert (tmp_path / "sweep" / "hist_gbm").exists()
    assert (tmp_path / "sweep" / "random_forest").exists()
    frame = pd.read_csv(summary_path)
    assert list(frame["model"]) == ["hist_gbm", "random_forest"]
    assert list(frame["status"]) == ["success", "success"]
    assert set(frame["predict_start"]) == {"2025-01-01"}
    assert list(frame["total_return"]) == [0.34, 0.34]
    assert payload["research_protocol"]["walk_forward"]["train_window_months"] == 36
    assert payload["research_protocol_path"] == str(protocol_path)
    assert len(calls) == 2
    assert calls[0]["output_dir"] == tmp_path / "sweep" / "hist_gbm"
    assert calls[1]["output_dir"] == tmp_path / "sweep" / "random_forest"


def test_model_sweep_handles_empty_model_list(tmp_path, monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []

    def _run_backtest(**kwargs):
        calls.append(kwargs)
        raise AssertionError("run_silver_chain_backtest should not be called")

    monkeypatch.setattr(sweep, "run_silver_chain_backtest", _run_backtest)

    exit_code = sweep.main(
        [
            "--predict-start",
            "2025-01-01",
            "--output-root",
            str(tmp_path / "empty"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    summary_path = tmp_path / "empty" / "summary_metrics.csv"
    protocol_path = tmp_path / "empty" / "research_protocol.json"
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["empty_models"] is True
    assert payload["error"]["message"] == "No models were provided."
    assert summary_path.exists()
    assert protocol_path.exists()
    frame = pd.read_csv(summary_path)
    assert frame.empty
    assert list(frame.columns) == list(sweep.DEFAULT_SWEEP_COLUMNS)
    assert calls == []


def test_model_sweep_continues_after_model_failure(tmp_path, monkeypatch, capsys) -> None:
    calls: list[str] = []

    def _run_backtest(**kwargs):
        model_name = kwargs["model_name"]
        calls.append(model_name)
        if model_name == "broken_model":
            raise RuntimeError("boom")
        return {
            "model": model_name,
            "summary": {
                "sessions": 9,
                "total_return": 0.12,
                "annualized_return": 0.10,
                "annualized_volatility": 0.18,
                "sharpe": 0.55,
                "max_drawdown": -0.11,
                "benchmark_total_return": 0.05,
                "mean_turnover": 0.8,
                "mean_cost_bps": 10.0,
            },
            "artifacts_dir": str(kwargs["output_dir"]),
        }

    monkeypatch.setattr(sweep, "run_silver_chain_backtest", _run_backtest)

    exit_code = sweep.main(
        [
            "--models",
            "broken_model",
            "hist_gbm",
            "--predict-start",
            "2025-01-01",
            "--output-root",
            str(tmp_path / "mixed"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    frame = pd.read_csv(tmp_path / "mixed" / "summary_metrics.csv")
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["counts"] == {"requested": 2, "success": 1, "failed": 1}
    assert list(frame["model"]) == ["broken_model", "hist_gbm"]
    assert list(frame["status"]) == ["failed", "success"]
    assert frame.loc[0, "error_type"] == "RuntimeError"
    assert frame.loc[0, "error_message"] == "boom"
    assert frame.loc[1, "total_return"] == 0.12
    assert calls == ["broken_model", "hist_gbm"]
