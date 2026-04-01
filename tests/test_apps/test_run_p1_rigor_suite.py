from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stockmachine.apps import run_p1_rigor_suite as p1_app
from stockmachine.domain.project_paths import build_strategy_project_paths


def test_run_p1_rigor_suite_writes_expected_outputs(tmp_path, monkeypatch, capsys) -> None:
    class _Bundle:
        predict_start = "2025-01-01"
        horizon = 5

    def _build_bundle(*, predict_start: str, horizon: int, cache_dir=None, reuse_cache=True, rebuild_cache=False):
        assert predict_start == "2025-01-01"
        assert horizon == 5
        assert str(cache_dir).endswith(
            "artifacts\\strategy_projects\\us_equities_h5\\research\\cache\\p1_rigor_suite"
        ) or str(cache_dir).endswith(
            "artifacts/strategy_projects/us_equities_h5/research/cache/p1_rigor_suite"
        )
        assert reuse_cache is True
        assert rebuild_cache is False
        return _Bundle()

    def _strict_sweep(bundle, *, model_names, output_root, top_k, overlay_config):
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "model": "hist_gbm",
                    "status": "success",
                    "predict_start": bundle.predict_start,
                    "top_k": top_k,
                    "horizon": bundle.horizon,
                    "artifacts_dir": str(output_root / "hist_gbm"),
                    "sessions": 10,
                    "total_return": 0.2,
                    "annualized_return": 0.18,
                    "annualized_volatility": 0.2,
                    "sharpe": 0.9,
                    "max_drawdown": -0.1,
                    "benchmark_total_return": 0.1,
                    "mean_turnover": 1.0,
                    "mean_cost_bps": 10.0,
                }
            ]
        ).to_csv(output_root / "summary_metrics.csv", index=False)
        model_dir = output_root / "hist_gbm"
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "signal_date": "2025-01-02",
                    "entry_date": "2025-01-03",
                    "exit_date": "2025-01-10",
                    "gross_return": 0.02,
                    "net_return": 0.019,
                    "benchmark_return": 0.01,
                    "turnover": 1.0,
                    "cost_bps": 10.0,
                    "positions": 10,
                }
            ]
        ).to_csv(model_dir / "backtest_records.csv", index=False)
        return {"ok": True}

    def _stability(records, *, model_name, horizon, period):
        return pd.DataFrame([{"model": model_name, "period": period, "period_label": "2025", "sessions": 1}])

    def _cost(records, *, model_name, horizon, cost_levels_bps):
        return pd.DataFrame(
            [{"model": model_name, "cost_bps_per_side": float(level), "sessions": 1} for level in cost_levels_bps]
        )

    def _topk(bundle, *, model_names, top_k_values, output_root, overlay_config):
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([{"model": "hist_gbm", "top_k": int(top_k_values[0]), "sessions": 1}])
        frame.to_csv(output_root / "summary_metrics.csv", index=False)
        return frame

    monkeypatch.setattr(p1_app, "build_strict_research_bundle", _build_bundle)
    monkeypatch.setattr(p1_app, "run_strict_model_sweep_from_bundle", _strict_sweep)
    monkeypatch.setattr(p1_app, "build_period_stability_summary", _stability)
    monkeypatch.setattr(p1_app, "build_cost_stress_summary", _cost)
    monkeypatch.setattr(p1_app, "run_topk_parameter_sweep_from_bundle", _topk)

    exit_code = p1_app.main(["--models", "hist_gbm", "--output-root", str(tmp_path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["analysis_models"] == ["hist_gbm"]
    assert payload["cache"]["enabled"] is True
    assert (tmp_path / "strict_full" / "summary_metrics.csv").exists()
    assert (tmp_path / "stability" / "yearly_summary.csv").exists()
    assert (tmp_path / "cost_stress" / "summary_metrics.csv").exists()
    assert (tmp_path / "topk_sweep" / "summary_metrics.csv").exists()


def test_run_p1_rigor_suite_defaults_to_project_scoped_research_root(tmp_path, monkeypatch, capsys) -> None:
    class _Bundle:
        predict_start = "2025-01-01"
        horizon = 5

    workspace = build_strategy_project_paths("us_equities_h5", artifact_root=tmp_path / "artifacts")

    def _build_bundle(*, predict_start: str, horizon: int, cache_dir=None, reuse_cache=True, rebuild_cache=False):
        assert predict_start == "2025-01-01"
        assert horizon == 5
        assert cache_dir == workspace.research_root / "cache" / "p1_rigor_suite"
        assert reuse_cache is True
        assert rebuild_cache is False
        return _Bundle()

    def _strict_sweep(bundle, *, model_names, output_root, top_k, overlay_config):
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "model": "hist_gbm",
                    "status": "success",
                    "predict_start": bundle.predict_start,
                    "top_k": top_k,
                    "horizon": bundle.horizon,
                    "artifacts_dir": str(output_root / "hist_gbm"),
                    "sessions": 2,
                    "total_return": 0.1,
                    "annualized_return": 0.09,
                    "annualized_volatility": 0.11,
                    "sharpe": 0.8,
                    "max_drawdown": -0.05,
                    "benchmark_total_return": 0.03,
                    "mean_turnover": 0.7,
                    "mean_cost_bps": 10.0,
                }
            ]
        ).to_csv(output_root / "summary_metrics.csv", index=False)
        model_dir = output_root / "hist_gbm"
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "signal_date": "2025-01-02",
                    "entry_date": "2025-01-03",
                    "exit_date": "2025-01-10",
                    "gross_return": 0.01,
                    "net_return": 0.009,
                    "benchmark_return": 0.004,
                    "turnover": 0.7,
                    "cost_bps": 10.0,
                    "positions": 10,
                }
            ]
        ).to_csv(model_dir / "backtest_records.csv", index=False)
        return {"ok": True}

    def _stability(records, *, model_name, horizon, period):
        return pd.DataFrame([{"model": model_name, "period": period, "period_label": "2025", "sessions": 1}])

    def _cost(records, *, model_name, horizon, cost_levels_bps):
        return pd.DataFrame([{"model": model_name, "cost_bps_per_side": float(cost_levels_bps[0]), "sessions": 1}])

    def _topk(bundle, *, model_names, top_k_values, output_root, overlay_config):
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([{"model": "hist_gbm", "top_k": int(top_k_values[0]), "sessions": 1}])
        frame.to_csv(output_root / "summary_metrics.csv", index=False)
        return frame

    monkeypatch.setattr(p1_app, "build_strict_research_bundle", _build_bundle)
    monkeypatch.setattr(p1_app, "run_strict_model_sweep_from_bundle", _strict_sweep)
    monkeypatch.setattr(p1_app, "build_period_stability_summary", _stability)
    monkeypatch.setattr(p1_app, "build_cost_stress_summary", _cost)
    monkeypatch.setattr(p1_app, "run_topk_parameter_sweep_from_bundle", _topk)

    exit_code = p1_app.main(["--models", "hist_gbm", "--artifact-root", str(tmp_path / "artifacts")])
    payload = json.loads(capsys.readouterr().out)

    expected_root = workspace.research_root / "p1_rigor_suite"
    assert exit_code == 0
    assert payload["strategy_project"] == "us_equities_h5"
    assert payload["strategy_workspace"] == workspace.to_dict()
    assert payload["output_root"] == str(expected_root)
    assert payload["cache"]["cache_dir"] == str(workspace.research_root / "cache" / "p1_rigor_suite")
    assert (expected_root / "strict_full" / "summary_metrics.csv").exists()
