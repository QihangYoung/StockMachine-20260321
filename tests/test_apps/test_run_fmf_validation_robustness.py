from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stockmachine.apps import run_fmf_validation_robustness as app


def _write_records(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "signal_date": "2018-01-02",
                "entry_date": "2018-01-02",
                "exit_date": "2018-01-02",
                "gross_return": 0.010,
                "net_return": 0.010,
                "benchmark_return": 0.006,
                "turnover": 0.10,
                "cost_bps": 0.0,
                "positions": 7,
            },
            {
                "signal_date": "2018-04-02",
                "entry_date": "2018-04-02",
                "exit_date": "2018-04-02",
                "gross_return": -0.004,
                "net_return": -0.004,
                "benchmark_return": -0.002,
                "turnover": 0.12,
                "cost_bps": 0.0,
                "positions": 7,
            },
            {
                "signal_date": "2019-01-02",
                "entry_date": "2019-01-02",
                "exit_date": "2019-01-02",
                "gross_return": 0.008,
                "net_return": 0.008,
                "benchmark_return": 0.004,
                "turnover": 0.09,
                "cost_bps": 0.0,
                "positions": 7,
            },
            {
                "signal_date": "2019-04-01",
                "entry_date": "2019-04-01",
                "exit_date": "2019-04-01",
                "gross_return": -0.003,
                "net_return": -0.003,
                "benchmark_return": -0.001,
                "turnover": 0.11,
                "cost_bps": 0.0,
                "positions": 7,
            },
        ]
    ).to_csv(path, index=False)


def _write_strategy_dir(root: Path, strategy_name: str) -> None:
    strategy_dir = root / strategy_name
    strategy_dir.mkdir(parents=True, exist_ok=True)
    _write_records(strategy_dir / "backtest_records.csv")


def _write_baseline_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for strategy_name in (
        "static_equal_weight_non_cash",
        "rolling_erc_core",
        "rolling_c2_v0_seed_core",
    ):
        _write_strategy_dir(root, strategy_name)
    (root / "run_meta.json").write_text(
        json.dumps(
            {
                "lockbox_policy": {
                    "test_window_locked": True,
                    "include_test_window": False,
                }
            }
        ),
        encoding="utf-8",
    )


def _write_grid_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    strategy_names = (
        "rolling_c2_v0_seed_core",
        "rolling_fmf_c2_e40_c10_d20_i20_t10",
        "rolling_fmf_c2_e42_c10_d22_i20_t06",
        "rolling_fmf_c2_e42_c10_d22_i18_t08",
    )
    for strategy_name in strategy_names:
        _write_strategy_dir(root, strategy_name)

    pd.DataFrame(
        [
            {
                "strategy_name": "rolling_c2_v0_seed_core",
                "policy_name": "fmf_c2_v0_seed",
                "equity_total": 0.35,
                "equity_us": 0.24,
                "equity_ex_us": 0.11,
                "duration": 0.20,
                "credit": 0.10,
                "inflation_hedge": 0.15,
                "trend": 0.20,
                "diversifier_total": 0.35,
                "reserve_cash": 0.05,
            },
            {
                "strategy_name": "rolling_fmf_c2_e40_c10_d20_i20_t10",
                "policy_name": "fmf_c2_e40_c10_d20_i20_t10",
                "equity_total": 0.40,
                "equity_us": 0.2743,
                "equity_ex_us": 0.1257,
                "duration": 0.20,
                "credit": 0.10,
                "inflation_hedge": 0.20,
                "trend": 0.10,
                "diversifier_total": 0.30,
                "reserve_cash": 0.05,
            },
            {
                "strategy_name": "rolling_fmf_c2_e42_c10_d22_i20_t06",
                "policy_name": "fmf_c2_e42_c10_d22_i20_t06",
                "equity_total": 0.42,
                "equity_us": 0.2880,
                "equity_ex_us": 0.1320,
                "duration": 0.22,
                "credit": 0.10,
                "inflation_hedge": 0.20,
                "trend": 0.06,
                "diversifier_total": 0.26,
                "reserve_cash": 0.05,
            },
            {
                "strategy_name": "rolling_fmf_c2_e42_c10_d22_i18_t08",
                "policy_name": "fmf_c2_e42_c10_d22_i18_t08",
                "equity_total": 0.42,
                "equity_us": 0.2880,
                "equity_ex_us": 0.1320,
                "duration": 0.22,
                "credit": 0.10,
                "inflation_hedge": 0.18,
                "trend": 0.08,
                "diversifier_total": 0.26,
                "reserve_cash": 0.05,
            },
        ]
    ).to_csv(root / "policy_grid.csv", index=False)

    pd.DataFrame(
        [
            {
                "strategy_name": "rolling_fmf_c2_e42_c10_d22_i20_t06",
                "sessions": 100,
                "total_return": 0.20,
                "annualized_return": 0.070,
                "annualized_volatility": 0.050,
                "sharpe": 1.20,
                "max_drawdown": -0.06,
                "benchmark_total_return": 0.15,
                "mean_turnover": 0.02,
                "mean_cost_bps": 0.0,
            },
            {
                "strategy_name": "rolling_fmf_c2_e42_c10_d22_i18_t08",
                "sessions": 100,
                "total_return": 0.19,
                "annualized_return": 0.066,
                "annualized_volatility": 0.050,
                "sharpe": 1.15,
                "max_drawdown": -0.06,
                "benchmark_total_return": 0.15,
                "mean_turnover": 0.02,
                "mean_cost_bps": 0.0,
            },
            {
                "strategy_name": "rolling_fmf_c2_e40_c10_d20_i20_t10",
                "sessions": 100,
                "total_return": 0.18,
                "annualized_return": 0.063,
                "annualized_volatility": 0.050,
                "sharpe": 1.10,
                "max_drawdown": -0.06,
                "benchmark_total_return": 0.15,
                "mean_turnover": 0.02,
                "mean_cost_bps": 0.0,
            },
            {
                "strategy_name": "rolling_c2_v0_seed_core",
                "sessions": 100,
                "total_return": 0.16,
                "annualized_return": 0.058,
                "annualized_volatility": 0.050,
                "sharpe": 1.00,
                "max_drawdown": -0.06,
                "benchmark_total_return": 0.15,
                "mean_turnover": 0.02,
                "mean_cost_bps": 0.0,
            },
        ]
    ).to_csv(root / "summary_metrics_c2_candidates_validation_common_window.csv", index=False)

    (root / "sweep_meta.json").write_text(
        json.dumps(
            {
                "lockbox_policy": {
                    "test_window_locked": True,
                    "test_window_exposed": False,
                }
            }
        ),
        encoding="utf-8",
    )


def test_run_fmf_validation_robustness_writes_expected_outputs(tmp_path, capsys) -> None:
    baseline_root = tmp_path / "baseline"
    broad_grid_root = tmp_path / "grid_broad"
    narrow_grid_root = tmp_path / "grid_narrow"
    output_root = tmp_path / "robustness"

    _write_baseline_root(baseline_root)
    _write_grid_root(broad_grid_root)
    _write_grid_root(narrow_grid_root)

    exit_code = app.main(
        [
            "--baseline-root",
            str(baseline_root),
            "--broad-grid-root",
            str(broad_grid_root),
            "--narrow-grid-root",
            str(narrow_grid_root),
            "--output-root",
            str(output_root),
            "--top-n-candidates",
            "2",
        ]
    )
    _ = capsys.readouterr()

    assert exit_code == 0

    candidate_manifest = pd.read_csv(output_root / "manifests" / "candidate_manifest.csv")
    assert list(candidate_manifest["report_name"][:3]) == [
        "static_equal_weight_non_cash",
        "rolling_erc_core",
        "rolling_c2_v0_seed_core",
    ]
    assert len(candidate_manifest) == 5

    suite_root = output_root / "robustness_suite"
    run_meta = json.loads((suite_root / "run_meta.json").read_text(encoding="utf-8"))
    assert run_meta["strategy_project"] == "multi_asset_fmf_validation"
    assert run_meta["analyzers"]["selection_bias"] is True
    assert run_meta["analyzers"]["parameter_stability"] is True
    assert run_meta["analyzers"]["universe_stability"] is False

    assert (suite_root / "robustness_overview.csv").exists()
    assert (
        suite_root
        / "suite_level"
        / "fmf_validation_broad_search_selection_bias_summary.csv"
    ).exists()
    assert (
        suite_root
        / "suite_level"
        / "fmf_validation_narrow_search_parameter_stability_summary.csv"
    ).exists()
    assert (
        suite_root
        / "per_model"
        / "rolling_fmf_c2_e42_c10_d22_i20_t06"
        / "time_stability_yearly.csv"
    ).exists()


def test_run_fmf_validation_robustness_rejects_exposed_source_root(tmp_path) -> None:
    baseline_root = tmp_path / "baseline"
    broad_grid_root = tmp_path / "grid_broad"
    narrow_grid_root = tmp_path / "grid_narrow"
    output_root = tmp_path / "robustness"

    _write_baseline_root(baseline_root)
    _write_grid_root(broad_grid_root)
    _write_grid_root(narrow_grid_root)

    (baseline_root / "run_meta.json").write_text(
        json.dumps(
            {
                "lockbox_policy": {
                    "test_window_locked": True,
                    "include_test_window": True,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exposed the test window"):
        app.main(
            [
                "--baseline-root",
                str(baseline_root),
                "--broad-grid-root",
                str(broad_grid_root),
                "--narrow-grid-root",
                str(narrow_grid_root),
                "--output-root",
                str(output_root),
            ]
        )
