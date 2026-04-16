import numpy as np
import pandas as pd
import pytest

from stockmachine.research.multi_asset import (
    AffineProxySegment,
    BucketDefinition,
    CovarianceConfig,
    DEFAULT_C2_V0_POLICY_CONFIG,
    DEFAULT_ERC_POLICY_CONFIG,
    RollingAllocationConfig,
    SharpeTargetVolConfig,
    ThresholdRebalanceConfig,
    allocate_vol_capped_portfolio_into_cash,
    apply_affine_proxy_chain,
    backtest_weight_schedule,
    build_bucket_return_frame,
    build_equity_duration_shifted_policy_config,
    build_full_portfolio_weights,
    build_periodic_weight_schedule,
    build_threshold_aware_weight_schedule,
    build_symbol_return_frame,
    estimate_blended_covariance,
    estimate_ewma_covariance,
    estimate_sample_covariance,
    estimate_trailing_asset_sharpe_scores,
    evaluate_risk_budget,
    generate_rolling_configured_allocations,
    generate_rolling_sharpe_target_vol_allocations,
    generate_state_conditioned_rolling_allocations,
    run_periodic_constant_weight_backtest,
    run_rolling_sharpe_target_vol_backtest,
    run_state_conditioned_rolling_risk_budget_backtest,
    run_threshold_aware_weight_schedule_backtest,
    solve_configured_risk_budget_allocation,
    solve_equal_risk_contribution_weights,
    solve_long_only_sharpe_score_weights,
    solve_risk_budget_weights,
)


def test_build_bucket_return_frame_uses_adjusted_prices_and_common_intersection() -> None:
    daily_bar = pd.DataFrame(
        {
            "session_date": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-06",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-06",
                ]
            ),
            "symbol": ["SPY", "SPY", "SPY", "GLDM", "GLDM", "GLDM"],
            "close": [100.0, 110.0, 121.0, 50.0, 50.0, 50.0],
        }
    )
    adj_factor = pd.DataFrame(
        {
            "session_date": daily_bar["session_date"],
            "symbol": daily_bar["symbol"],
            "price_adjust_factor": [1.0, 1.0, 1.0, 1.0, 1.10, 1.10],
        }
    )

    symbol_returns = build_symbol_return_frame(daily_bar, adj_factor=adj_factor)
    bucket_returns = build_bucket_return_frame(
        symbol_returns,
        (
            BucketDefinition(name="balanced", symbols=("SPY", "GLDM"), symbol_weights=(0.6, 0.4)),
            BucketDefinition(name="equity_us", symbols=("SPY",)),
        ),
    )

    assert list(symbol_returns.index) == list(pd.to_datetime(["2026-01-03", "2026-01-06"]))
    assert symbol_returns.loc[pd.Timestamp("2026-01-03"), "GLDM"] == pytest.approx(0.10)
    assert symbol_returns.loc[pd.Timestamp("2026-01-06"), "GLDM"] == pytest.approx(0.0)
    assert bucket_returns.loc[pd.Timestamp("2026-01-03"), "balanced"] == pytest.approx(0.10)
    assert bucket_returns.loc[pd.Timestamp("2026-01-06"), "balanced"] == pytest.approx(0.06)
    assert bucket_returns.loc[pd.Timestamp("2026-01-06"), "equity_us"] == pytest.approx(0.10)


def test_estimate_blended_covariance_matches_weighted_components() -> None:
    returns = pd.DataFrame(
        {
            "equity_us": [0.01, 0.02, -0.01, 0.03, 0.00],
            "trend": [0.00, 0.01, 0.02, -0.01, 0.01],
        },
        index=pd.to_datetime(
            ["2026-01-02", "2026-01-03", "2026-01-06", "2026-01-07", "2026-01-08"]
        ),
    )
    config = CovarianceConfig(
        long_lookback=4,
        short_lookback=3,
        ewma_lambda=0.5,
        long_weight=0.6,
        short_weight=0.4,
    )

    estimate = estimate_blended_covariance(returns, config=config)
    expected_long = estimate_sample_covariance(returns, lookback=4)
    expected_short = estimate_ewma_covariance(returns, lookback=3, ewma_lambda=0.5)
    expected_blend = 0.6 * expected_long + 0.4 * expected_short

    assert estimate.observation_count == 5
    assert np.allclose(estimate.long_covariance.to_numpy(), expected_long.to_numpy())
    assert np.allclose(estimate.short_covariance.to_numpy(), expected_short.to_numpy())
    assert np.allclose(estimate.covariance.to_numpy(), expected_blend.to_numpy())


def test_evaluate_risk_budget_contributions_sum_to_portfolio_volatility() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["equity_us", "trend"],
        columns=["equity_us", "trend"],
    )

    result = evaluate_risk_budget(covariance, weights={"equity_us": 0.6, "trend": 0.4})

    assert result.portfolio_volatility == pytest.approx(result.total_risk_contributions.sum())
    assert result.risk_shares.sum() == pytest.approx(1.0)


def test_estimate_trailing_asset_sharpe_scores_returns_expected_metrics() -> None:
    returns = pd.DataFrame(
        {
            "equity_us": [0.01, 0.02, -0.01, 0.02],
            "duration": [0.001, 0.001, 0.001, 0.001],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]),
    )

    scores, summary = estimate_trailing_asset_sharpe_scores(
        returns,
        asset_names=("equity_us", "duration"),
        lookback=4,
    )

    assert scores.index.tolist() == ["equity_us", "duration"]
    assert summary["asset"].tolist() == ["equity_us", "duration"]
    assert scores["equity_us"] > 0.0
    assert scores["duration"] == pytest.approx(0.0)


def test_solve_long_only_sharpe_score_weights_prefers_high_score_asset_under_diagonal_covariance() -> None:
    covariance = pd.DataFrame(
        np.diag([0.04, 0.09, 0.16]),
        index=["equity_us", "duration", "trend"],
        columns=["equity_us", "duration", "trend"],
    )

    weights = solve_long_only_sharpe_score_weights(
        covariance,
        sharpe_scores={"equity_us": 1.0, "duration": 0.5, "trend": -0.2},
    )

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert weights["trend"] == pytest.approx(0.0)
    assert weights["equity_us"] > weights["duration"]


def test_allocate_vol_capped_portfolio_into_cash_scales_down_risky_sleeve() -> None:
    covariance = pd.DataFrame(
        [
            [0.04, 0.00, 0.00],
            [0.00, 0.09, 0.00],
            [0.00, 0.00, 0.000001],
        ],
        index=["equity_us", "duration", "cash"],
        columns=["equity_us", "duration", "cash"],
    )

    final_weights, risk_scale, risky_volatility, final_volatility = allocate_vol_capped_portfolio_into_cash(
        {"equity_us": 0.6, "duration": 0.4},
        covariance=covariance,
        strategic_buckets=("equity_us", "duration"),
        cash_bucket="cash",
        target_volatility=0.10,
    )

    assert risky_volatility > 0.10
    assert final_volatility <= 0.10 + 1e-6
    assert 0.0 < risk_scale < 1.0
    assert final_weights["cash"] > 0.0
    assert final_weights.sum() == pytest.approx(1.0)


def test_solve_equal_risk_contribution_weights_balances_diagonal_covariance() -> None:
    covariance = pd.DataFrame(
        np.diag([0.04, 0.09, 0.16]),
        index=["equity_us", "duration", "trend"],
        columns=["equity_us", "duration", "trend"],
    )

    result = solve_equal_risk_contribution_weights(covariance, tolerance=1e-10)

    assert result.converged is True
    assert result.risk_shares.to_numpy() == pytest.approx(np.array([1 / 3, 1 / 3, 1 / 3]), abs=1e-6)
    assert result.weights.to_numpy() == pytest.approx(
        np.array([0.46153846, 0.30769231, 0.23076923]),
        abs=1e-6,
    )


def test_solve_risk_budget_weights_matches_custom_target_on_diagonal_covariance() -> None:
    covariance = pd.DataFrame(
        np.diag([0.04, 0.09, 0.16]),
        index=["equity_us", "duration", "trend"],
        columns=["equity_us", "duration", "trend"],
    )
    risk_budgets = {"equity_us": 0.5, "duration": 0.3, "trend": 0.2}

    result = solve_risk_budget_weights(covariance, risk_budgets=risk_budgets, tolerance=1e-10)

    assert result.converged is True
    assert result.risk_shares.to_numpy() == pytest.approx(np.array([0.5, 0.3, 0.2]), abs=1e-6)
    assert result.weights.to_numpy() == pytest.approx(
        np.array([0.54566521, 0.28178030, 0.17255449]),
        abs=1e-6,
    )


def test_default_policy_configs_freeze_erc_and_c2_v0_targets() -> None:
    erc_budgets = DEFAULT_ERC_POLICY_CONFIG.normalized_risk_budgets()
    c2_v0_budgets = DEFAULT_C2_V0_POLICY_CONFIG.normalized_risk_budgets()

    assert erc_budgets.index.tolist() == [
        "equity_us",
        "equity_ex_us",
        "duration",
        "credit",
        "inflation_hedge",
        "trend",
    ]
    assert erc_budgets.to_numpy() == pytest.approx(np.full(6, 1 / 6), abs=1e-12)
    assert DEFAULT_ERC_POLICY_CONFIG.reserve_capital_series().to_dict() == {"cash": 0.0}

    assert c2_v0_budgets.to_dict() == pytest.approx(
        {
            "equity_us": 0.24,
            "equity_ex_us": 0.11,
            "duration": 0.20,
            "credit": 0.10,
            "inflation_hedge": 0.15,
            "trend": 0.20,
        },
        abs=1e-12,
    )
    assert DEFAULT_C2_V0_POLICY_CONFIG.reserve_capital_series().to_dict() == {"cash": 0.05}
    assert DEFAULT_C2_V0_POLICY_CONFIG.strategic_capital_fraction == pytest.approx(0.95)


def test_build_full_portfolio_weights_scales_strategic_weights_and_keeps_cash_reserve() -> None:
    strategic_weights = {
        "equity_us": 0.50,
        "equity_ex_us": 0.10,
        "duration": 0.20,
        "credit": 0.05,
        "inflation_hedge": 0.05,
        "trend": 0.10,
    }

    full_weights = build_full_portfolio_weights(strategic_weights, config=DEFAULT_C2_V0_POLICY_CONFIG)

    assert full_weights.to_dict() == pytest.approx(
        {
            "equity_us": 0.475,
            "equity_ex_us": 0.095,
            "duration": 0.190,
            "credit": 0.0475,
            "inflation_hedge": 0.0475,
            "trend": 0.095,
            "cash": 0.05,
        },
        abs=1e-12,
    )
    assert full_weights.sum() == pytest.approx(1.0)


def test_build_equity_duration_shifted_policy_config_moves_risk_budget_between_equity_and_duration() -> None:
    shifted = build_equity_duration_shifted_policy_config(
        DEFAULT_C2_V0_POLICY_CONFIG,
        name="shifted",
        equity_duration_shift=0.04,
    )

    shifted_budgets = shifted.normalized_risk_budgets()
    assert shifted_budgets.to_dict() == pytest.approx(
        {
            "equity_us": 0.26742857142857146,
            "equity_ex_us": 0.12257142857142858,
            "duration": 0.16,
            "credit": 0.10,
            "inflation_hedge": 0.15,
            "trend": 0.20,
        },
        abs=1e-12,
    )
    assert shifted.reserve_capital_series().to_dict() == {"cash": 0.05}


def test_solve_configured_risk_budget_allocation_respects_v0_budgets_and_cash_reserve() -> None:
    covariance = pd.DataFrame(
        np.diag([0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.0001]),
        index=[
            "equity_us",
            "equity_ex_us",
            "duration",
            "credit",
            "inflation_hedge",
            "trend",
            "cash",
        ],
        columns=[
            "equity_us",
            "equity_ex_us",
            "duration",
            "credit",
            "inflation_hedge",
            "trend",
            "cash",
        ],
    )

    allocation = solve_configured_risk_budget_allocation(
        covariance,
        config=DEFAULT_C2_V0_POLICY_CONFIG,
        tolerance=1e-10,
    )

    assert allocation.config.name == "c2_v0_balanced_defensive"
    assert allocation.strategic_result.converged is True
    assert allocation.strategic_result.risk_shares.to_dict() == pytest.approx(
        {
            "equity_us": 0.24,
            "equity_ex_us": 0.11,
            "duration": 0.20,
            "credit": 0.10,
            "inflation_hedge": 0.15,
            "trend": 0.20,
        },
        abs=1e-6,
    )
    assert allocation.full_weights["cash"] == pytest.approx(0.05)
    assert allocation.full_weights.sum() == pytest.approx(1.0)


def test_generate_rolling_configured_allocations_uses_next_session_effective_dates() -> None:
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-03",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
            "2026-01-09",
        ]
    )
    bucket_returns = pd.DataFrame(
        {
            "equity_us": [0.01, 0.02, -0.01, 0.00, 0.01, 0.02],
            "equity_ex_us": [0.00, 0.01, 0.00, 0.01, -0.01, 0.00],
            "duration": [0.00, -0.01, 0.01, 0.00, 0.01, -0.01],
            "credit": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01],
            "inflation_hedge": [0.00, 0.01, 0.02, -0.01, 0.00, 0.01],
            "trend": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00],
            "cash": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
        },
        index=dates,
    )

    result = generate_rolling_configured_allocations(
        bucket_returns,
        covariance_config=CovarianceConfig(long_lookback=3, short_lookback=2, ewma_lambda=0.8),
        policy_config=DEFAULT_C2_V0_POLICY_CONFIG,
        rolling_config=RollingAllocationConfig(rebalance_frequency=2, effective_lag=1),
    )

    assert result.weight_schedule.index.tolist() == [dates[3], dates[5]]
    assert np.allclose(result.weight_schedule.sum(axis=1).to_numpy(), np.array([1.0, 1.0]))
    assert np.allclose(result.weight_schedule["cash"].to_numpy(), np.array([0.05, 0.05]))
    assert result.diagnostics["decision_date"].tolist() == [dates[2], dates[4]]
    assert result.diagnostics["effective_date"].tolist() == [dates[3], dates[5]]
    assert result.diagnostics["iterations"].gt(0).all()
    assert result.diagnostics["observation_count"].tolist() == [3, 5]


def test_generate_rolling_sharpe_target_vol_allocations_caps_risky_scale_when_needed() -> None:
    dates = pd.date_range("2026-01-02", periods=8, freq="B")
    bucket_returns = pd.DataFrame(
        {
            "equity_us": [0.03, 0.02, -0.02, 0.03, 0.02, -0.01, 0.03, 0.02],
            "equity_ex_us": [0.02, 0.01, -0.01, 0.02, 0.01, -0.01, 0.02, 0.01],
            "duration": [0.00, -0.01, 0.01, 0.00, -0.01, 0.01, 0.00, -0.01],
            "credit": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01, -0.01, 0.00],
            "inflation_hedge": [0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01],
            "trend": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01],
            "cash": [0.0001] * 8,
        },
        index=dates,
    )

    result = generate_rolling_sharpe_target_vol_allocations(
        bucket_returns,
        covariance_config=CovarianceConfig(long_lookback=4, short_lookback=2, ewma_lambda=0.8),
        rolling_config=RollingAllocationConfig(rebalance_frequency=2, effective_lag=1, min_history=4),
        target_vol_config=SharpeTargetVolConfig(
            strategic_buckets=("equity_us", "equity_ex_us", "duration", "credit", "inflation_hedge", "trend"),
            cash_bucket="cash",
            score_lookback=4,
            target_volatility=0.10,
        ),
    )

    assert result.weight_schedule.index.tolist() == [dates[4], dates[6]]
    assert np.allclose(result.weight_schedule.sum(axis=1).to_numpy(), np.array([1.0, 1.0]))
    assert "cash" in result.weight_schedule.columns
    assert "risk_scale" in result.diagnostics.columns
    assert "score_equity_us" in result.diagnostics.columns
    assert result.diagnostics["final_portfolio_volatility"].le(0.10 + 1e-6).all()


def test_run_rolling_sharpe_target_vol_backtest_returns_records() -> None:
    dates = pd.date_range("2026-01-02", periods=8, freq="B")
    bucket_returns = pd.DataFrame(
        {
            "equity_us": [0.03, 0.02, -0.02, 0.03, 0.02, -0.01, 0.03, 0.02],
            "equity_ex_us": [0.02, 0.01, -0.01, 0.02, 0.01, -0.01, 0.02, 0.01],
            "duration": [0.00, -0.01, 0.01, 0.00, -0.01, 0.01, 0.00, -0.01],
            "credit": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01, -0.01, 0.00],
            "inflation_hedge": [0.00, 0.01, 0.02, -0.01, 0.00, 0.01, 0.02, -0.01],
            "trend": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01],
            "cash": [0.0001] * 8,
        },
        index=dates,
    )

    allocations, backtest = run_rolling_sharpe_target_vol_backtest(
        bucket_returns,
        covariance_config=CovarianceConfig(long_lookback=4, short_lookback=2, ewma_lambda=0.8),
        rolling_config=RollingAllocationConfig(rebalance_frequency=2, effective_lag=1, min_history=4),
        target_vol_config=SharpeTargetVolConfig(
            strategic_buckets=("equity_us", "equity_ex_us", "duration", "credit", "inflation_hedge", "trend"),
            cash_bucket="cash",
            score_lookback=4,
            target_volatility=0.10,
        ),
        benchmark_column="equity_us",
    )

    assert allocations.diagnostics["risk_scale"].between(0.0, 1.0).all()
    assert backtest.records["rebalanced"].any()
    assert backtest.summary["sessions"] == 4


def test_generate_state_conditioned_rolling_allocations_switches_configs_by_state() -> None:
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
            "2026-01-09",
        ]
    )
    bucket_returns = pd.DataFrame(
        {
            "equity_us": [0.01, 0.02, -0.01, 0.00, 0.01, 0.02],
            "equity_ex_us": [0.00, 0.01, 0.00, 0.01, -0.01, 0.00],
            "duration": [0.00, -0.01, 0.01, 0.00, 0.01, -0.01],
            "credit": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01],
            "inflation_hedge": [0.00, 0.01, 0.02, -0.01, 0.00, 0.01],
            "trend": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00],
            "cash": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
        },
        index=dates,
    )
    risk_on = build_equity_duration_shifted_policy_config(
        DEFAULT_C2_V0_POLICY_CONFIG,
        name="risk_on",
        equity_duration_shift=0.04,
    )
    state_result = generate_state_conditioned_rolling_allocations(
        bucket_returns,
        state_by_date=pd.Series(["risk_on", "neutral"], index=[dates[2], dates[4]], dtype="object"),
        state_to_policy_config={
            "neutral": DEFAULT_C2_V0_POLICY_CONFIG,
            "risk_on": risk_on,
        },
        default_state="neutral",
        covariance_config=CovarianceConfig(long_lookback=3, short_lookback=2, ewma_lambda=0.8),
        rolling_config=RollingAllocationConfig(rebalance_frequency=2, effective_lag=1),
    )

    assert state_result.weight_schedule.index.tolist() == [dates[3], dates[5]]
    assert state_result.diagnostics["state_name"].tolist() == ["risk_on", "neutral"]
    assert state_result.diagnostics["config_name"].tolist() == ["risk_on", "c2_v0_balanced_defensive"]
    assert state_result.weight_schedule.iloc[0].sum() == pytest.approx(1.0, abs=1e-12)
    assert state_result.weight_schedule.iloc[1].sum() == pytest.approx(1.0, abs=1e-12)
    assert state_result.weight_schedule.iloc[0].equals(state_result.weight_schedule.iloc[1]) is False


def test_run_state_conditioned_rolling_risk_budget_backtest_returns_records() -> None:
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
            "2026-01-09",
        ]
    )
    bucket_returns = pd.DataFrame(
        {
            "equity_us": [0.01, 0.02, -0.01, 0.00, 0.01, 0.02],
            "equity_ex_us": [0.00, 0.01, 0.00, 0.01, -0.01, 0.00],
            "duration": [0.00, -0.01, 0.01, 0.00, 0.01, -0.01],
            "credit": [0.01, 0.00, 0.01, -0.01, 0.00, 0.01],
            "inflation_hedge": [0.00, 0.01, 0.02, -0.01, 0.00, 0.01],
            "trend": [0.01, -0.01, 0.00, 0.02, -0.01, 0.00],
            "cash": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
        },
        index=dates,
    )
    defensive = build_equity_duration_shifted_policy_config(
        DEFAULT_C2_V0_POLICY_CONFIG,
        name="defensive",
        equity_duration_shift=-0.04,
    )

    allocations, backtest = run_state_conditioned_rolling_risk_budget_backtest(
        bucket_returns,
        state_by_date=pd.Series(["defensive"], index=[dates[2]], dtype="object"),
        state_to_policy_config={
            "neutral": DEFAULT_C2_V0_POLICY_CONFIG,
            "defensive": defensive,
        },
        default_state="neutral",
        covariance_config=CovarianceConfig(long_lookback=3, short_lookback=2, ewma_lambda=0.8),
        rolling_config=RollingAllocationConfig(rebalance_frequency=2, effective_lag=1),
        benchmark_column="equity_us",
    )

    assert allocations.diagnostics.loc[0, "state_name"] == "defensive"
    assert backtest.records["rebalanced"].any()
    assert backtest.summary["sessions"] == 3


def test_backtest_weight_schedule_rebalances_only_on_schedule_and_keeps_drift() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-06"])
    bucket_returns = pd.DataFrame(
        {
            "equity_us": [0.10, 0.00, 0.00],
            "duration": [0.00, 0.00, 0.10],
        },
        index=dates,
    )
    weight_schedule = pd.DataFrame(
        {
            "equity_us": [0.5, 0.0],
            "duration": [0.5, 1.0],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-06"]),
    )

    result = backtest_weight_schedule(
        bucket_returns,
        weight_schedule,
        benchmark_column="equity_us",
        cost_bps_per_side=100.0,
    )

    records = result.records
    assert records["turnover"].tolist() == pytest.approx([1.0, 0.0, 1.0476190476190477], abs=1e-12)
    assert records["rebalanced"].tolist() == [True, False, True]
    assert records.iloc[1]["applied_weights"] == pytest.approx(
        {
            "equity_us": 0.5238095238095238,
            "duration": 0.47619047619047616,
        },
        abs=1e-12,
    )
    assert float(records.iloc[0]["net_return"]) == pytest.approx(0.04, abs=1e-12)
    assert float(records.iloc[2]["net_return"]) == pytest.approx(0.08952380952380952, abs=1e-12)
    assert result.summary["sessions"] == 3
    assert result.summary["mean_turnover"] == pytest.approx((1.0 + 1.0476190476190477) / 3.0, abs=1e-12)


def test_build_periodic_weight_schedule_repeats_target_on_fixed_interval() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])

    schedule = build_periodic_weight_schedule(
        dates,
        {"equity_us": 0.6, "duration": 0.4},
        rebalance_frequency=2,
    )

    assert schedule.index.tolist() == [dates[0], dates[2], dates[4]]
    assert schedule.iloc[0].to_dict() == pytest.approx({"equity_us": 0.6, "duration": 0.4}, abs=1e-12)
    assert np.allclose(schedule.sum(axis=1).to_numpy(), np.array([1.0, 1.0, 1.0]))


def test_run_periodic_constant_weight_backtest_rebalances_back_to_target() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    bucket_returns = pd.DataFrame(
        {
            "equity_us": [0.10, 0.00, 0.00],
            "duration": [0.00, 0.00, 0.10],
        },
        index=dates,
    )

    result = run_periodic_constant_weight_backtest(
        bucket_returns,
        weights={"equity_us": 0.5, "duration": 0.5},
        rebalance_frequency=2,
        benchmark_column="equity_us",
    )

    assert result.weight_schedule.index.tolist() == [dates[0], dates[2]]
    assert result.records["rebalanced"].tolist() == [True, False, True]
    assert result.records.iloc[1]["applied_weights"] == pytest.approx(
        {
            "equity_us": 0.5238095238095238,
            "duration": 0.47619047619047616,
        },
        abs=1e-12,
    )
    assert result.records.iloc[2]["applied_weights"] == pytest.approx(
        {
            "equity_us": 0.5,
            "duration": 0.5,
        },
        abs=1e-12,
    )


def test_build_threshold_aware_weight_schedule_skips_small_target_update_then_rebalances_on_drift() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
    bucket_returns = pd.DataFrame(
        {
            "equity_us": [0.10, 0.00, 0.10, 0.00],
            "duration": [0.00, 0.00, 0.00, 0.00],
        },
        index=dates,
    )
    target_schedule = pd.DataFrame(
        {
            "equity_us": [0.5, 0.5],
            "duration": [0.5, 0.5],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-06"]),
    )

    threshold_result = build_threshold_aware_weight_schedule(
        bucket_returns,
        target_schedule,
        config=ThresholdRebalanceConfig(drift_threshold=0.03, stale_time_cap=None),
    )

    assert threshold_result.weight_schedule.index.tolist() == [dates[0], dates[3]]
    assert bool(threshold_result.diagnostics.loc[2, "target_updated_today"]) is True
    assert bool(threshold_result.diagnostics.loc[2, "rebalanced"]) is False
    assert bool(threshold_result.diagnostics.loc[3, "drift_triggered"]) is True
    assert bool(threshold_result.diagnostics.loc[3, "rebalanced"]) is True


def test_build_threshold_aware_weight_schedule_respects_stale_time_cap() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    bucket_returns = pd.DataFrame(
        {
            "equity_us": [0.00, 0.00, 0.00],
            "duration": [0.00, 0.00, 0.00],
        },
        index=dates,
    )
    target_schedule = pd.DataFrame(
        {
            "equity_us": [0.5],
            "duration": [0.5],
        },
        index=pd.to_datetime(["2026-01-02"]),
    )

    threshold_result = build_threshold_aware_weight_schedule(
        bucket_returns,
        target_schedule,
        config=ThresholdRebalanceConfig(drift_threshold=0.10, stale_time_cap=2),
    )

    assert threshold_result.weight_schedule.index.tolist() == [dates[0], dates[2]]
    assert bool(threshold_result.diagnostics.loc[2, "stale_triggered"]) is True


def test_run_threshold_aware_weight_schedule_backtest_uses_filtered_schedule() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
    bucket_returns = pd.DataFrame(
        {
            "equity_us": [0.10, 0.00, 0.10, 0.00],
            "duration": [0.00, 0.00, 0.00, 0.00],
        },
        index=dates,
    )
    target_schedule = pd.DataFrame(
        {
            "equity_us": [0.5, 0.5],
            "duration": [0.5, 0.5],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-06"]),
    )

    threshold_result, backtest_result = run_threshold_aware_weight_schedule_backtest(
        bucket_returns,
        target_schedule,
        config=ThresholdRebalanceConfig(drift_threshold=0.03),
        benchmark_column="equity_us",
    )

    assert threshold_result.weight_schedule.index.tolist() == [dates[0], dates[3]]
    assert backtest_result.records["rebalanced"].tolist() == [True, False, False, True]
    assert backtest_result.records.iloc[2]["applied_weights"] == pytest.approx(
        {"equity_us": 0.5238095238095238, "duration": 0.47619047619047616},
        abs=1e-12,
    )


def test_apply_affine_proxy_chain_fills_pre_live_history_and_preserves_live_returns() -> None:
    dates = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
            "2026-01-09",
        ]
    )
    proxy = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], index=dates)
    target = pd.Series([np.nan, np.nan, np.nan, np.nan, 0.101, 0.121], index=dates)
    symbol_returns = pd.DataFrame({"CTA": target, "FMF": proxy}, index=dates)

    result = apply_affine_proxy_chain(
        symbol_returns,
        (
            AffineProxySegment(
                target_symbol="CTA",
                proxy_symbol="FMF",
                segment_name="cta_proxy",
                apply_end="2026-01-07",
                min_observations=2,
            ),
        ),
    )

    expected = pd.Series([0.021, 0.041, 0.061, 0.081, 0.101, 0.121], index=dates)
    assert result.symbol_returns["CTA"].to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-12)
    assert result.symbol_returns["FMF"].to_numpy() == pytest.approx(proxy.to_numpy(), abs=1e-12)
    assert result.diagnostics.loc[0, "segment_name"] == "cta_proxy"
    assert int(result.diagnostics.loc[0, "applied_observations"]) == 4
    assert float(result.diagnostics.loc[0, "intercept"]) == pytest.approx(0.001, abs=1e-12)
    assert float(result.diagnostics.loc[0, "slope"]) == pytest.approx(2.0, abs=1e-12)
