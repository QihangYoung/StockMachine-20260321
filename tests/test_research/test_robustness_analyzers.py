from __future__ import annotations

import pandas as pd

from stockmachine.research.robustness_analyzers import (
    build_cost_execution_stress_summary,
    build_parameter_stability_summary,
    build_selection_bias_summary,
    build_tail_dependence_summary,
    build_time_stability_summary,
    build_turnover_concentration_summary,
    build_universe_stability_summary,
)
from stockmachine.research.robustness_frameworks import RobustnessFrameworkSpec, resolve_robustness_framework


def _sample_records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entry_date": "2024-12-31",
                "exit_date": "2025-01-02",
                "gross_return": 0.10,
                "net_return": 0.10,
                "benchmark_return": 0.01,
                "turnover": 0.1,
                "cost_bps": 1.0,
                "positions": 8,
            },
            {
                "entry_date": "2025-01-02",
                "exit_date": "2025-01-03",
                "gross_return": -0.05,
                "net_return": -0.05,
                "benchmark_return": -0.01,
                "turnover": 0.2,
                "cost_bps": 2.0,
                "positions": 8,
            },
            {
                "entry_date": "2025-01-03",
                "exit_date": "2025-01-06",
                "gross_return": 0.02,
                "net_return": 0.02,
                "benchmark_return": 0.00,
                "turnover": 0.1,
                "cost_bps": 1.0,
                "positions": 8,
            },
        ]
    )


def test_build_time_stability_summary_uses_framework_attribution_column() -> None:
    records = _sample_records()
    framework = RobustnessFrameworkSpec(
        framework_id="test",
        strategy_project="demo",
        default_horizon=1,
        attribution_date_column="exit_date",
    )

    summary = build_time_stability_summary(
        records,
        model_name="ridge",
        horizon=1,
        framework=framework,
        period="year",
    )

    assert set(summary["period_label"]) == {"2025"}
    assert summary["attribution_date_column"].iloc[0] == "exit_date"


def test_build_tail_dependence_summary_reports_trimmed_effects() -> None:
    records = _sample_records()
    framework = resolve_robustness_framework(strategy_project="us_equities_h1", horizon=1)

    summary = build_tail_dependence_summary(
        records,
        model_name="ridge",
        horizon=1,
        framework=framework,
        trim_counts=(1,),
    )

    top = summary[(summary["tail_side"] == "top") & (summary["trim_count"] == 1)].iloc[0]
    bottom = summary[(summary["tail_side"] == "bottom") & (summary["trim_count"] == 1)].iloc[0]

    assert top["removed_return_sum"] > 0
    assert top["delta_annualized_return"] < 0
    assert bottom["removed_return_sum"] < 0
    assert bottom["delta_annualized_return"] > 0


def test_build_tail_dependence_summary_handles_empty_records() -> None:
    framework = resolve_robustness_framework(strategy_project="us_equities_h1", horizon=1)

    summary = build_tail_dependence_summary(
        pd.DataFrame(),
        model_name="ridge",
        horizon=1,
        framework=framework,
    )

    assert summary.empty


def test_build_parameter_stability_summary_scores_leader_vs_neighbors() -> None:
    surface = pd.DataFrame(
        [
            {"top_k": 6, "max_turnover": 0.3, "min_weight_change": 0.02, "sharpe": 0.92},
            {"top_k": 6, "max_turnover": 0.3, "min_weight_change": 0.03, "sharpe": 0.73},
            {"top_k": 8, "max_turnover": 0.3, "min_weight_change": 0.02, "sharpe": 0.80},
            {"top_k": 8, "max_turnover": 0.3, "min_weight_change": 0.03, "sharpe": 0.78},
            {"top_k": 6, "max_turnover": 0.4, "min_weight_change": 0.02, "sharpe": 0.62},
            {"top_k": 6, "max_turnover": 0.4, "min_weight_change": 0.03, "sharpe": 0.61},
        ]
    )

    summary = build_parameter_stability_summary(
        surface,
        metric_column="sharpe",
        parameter_columns=("top_k", "max_turnover", "min_weight_change"),
        higher_is_better=True,
        neighborhood_radius=1,
        run_label="turnover_grid",
    )

    row = summary.iloc[0]
    assert row["leader_parameter_signature"].startswith("top_k=6")
    assert row["neighborhood_size"] >= 2
    assert row["leader_vs_neighbor_median_gap"] > 0


def test_build_cost_execution_stress_summary_adds_slope_and_break_even() -> None:
    records = _sample_records()
    framework = resolve_robustness_framework(strategy_project="us_equities_h1", horizon=1)

    summary = build_cost_execution_stress_summary(
        records,
        model_name="ridge",
        horizon=1,
        framework=framework,
        cost_levels_bps=(0.0, 500.0, 5000.0, 10000.0),
    )

    assert set(summary["cost_bps_per_side"]) == {0.0, 500.0, 5000.0, 10000.0}
    assert (summary["annualized_return_slope_per_10bps"] < 0).all()
    assert summary["break_even_cost_bps_per_side"].notna().all()


def test_build_turnover_concentration_summary_reports_top_day_shares() -> None:
    records = _sample_records()

    summary = build_turnover_concentration_summary(
        records,
        model_name="ridge",
        trim_counts=(1, 2),
    )

    row = summary.loc[summary["trim_count"] == 1].iloc[0]
    assert row["top_turnover_share"] > 0
    assert row["top_cost_share"] > 0
    assert row["days_to_reach_50pct_turnover"] >= 1


def test_build_universe_stability_summary_compares_base_vs_perturbations() -> None:
    surface = pd.DataFrame(
        [
            {"experiment": "base", "annualized_return": 0.25, "sharpe": 0.70, "max_drawdown": -0.38},
            {"experiment": "dv20_100m", "annualized_return": 0.246, "sharpe": 0.698, "max_drawdown": -0.382},
            {"experiment": "dv20_150m", "annualized_return": 0.239, "sharpe": 0.683, "max_drawdown": -0.382},
            {"experiment": "dv20_200m", "annualized_return": 0.243, "sharpe": 0.691, "max_drawdown": -0.382},
        ]
    )

    summary = build_universe_stability_summary(
        surface,
        label_column="experiment",
        base_label_value="base",
        run_label="h1_dv20",
    )

    row = summary.iloc[0]
    assert row["base_label"] == "base"
    assert row["perturbation_count"] == 3
    assert row["worst_annualized_return_drop_vs_base"] < 0
    assert row["worst_sharpe_drop_vs_base"] < 0


def test_build_selection_bias_summary_reports_trial_pressure_and_family_share() -> None:
    surface = pd.DataFrame(
        [
            {"model_family": "linear", "sharpe": 0.70},
            {"model_family": "linear", "sharpe": 0.75},
            {"model_family": "tree", "sharpe": 0.82},
            {"model_family": "tree", "sharpe": 0.91},
            {"model_family": "ranker", "sharpe": 0.68},
        ]
    )

    summary = build_selection_bias_summary(
        surface,
        metric_column="sharpe",
        higher_is_better=True,
        family_column="model_family",
        run_label="h1_model_compare",
    )

    row = summary.iloc[0]
    assert row["trial_count"] == 5
    assert row["family_count"] == 3
    assert row["leader_metric"] == 0.91
    assert row["leader_vs_median_gap"] > 0
    assert row["leader_family"] == "tree"
    assert row["leader_family_share_of_trials"] > 0
