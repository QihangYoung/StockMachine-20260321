# U.S. Zero-Shot Multi-Expert Report

## Scope

This report summarizes the current end-to-end experiment set for `A股训练 -> 美股 zero-shot -> 白盒风控 -> Alpaca-style historical execution`.

The aligned comparison window used for the main cross-expert table is **2025-03-13 to 2025-12-23**. Single experts were filtered to that common window so they can be compared with the 5-expert ensemble on the same date range.

Core datasets:
- A-share training universe: [large_cap_50_20200101_20241231_hfq_normalized.csv](C:/Users/Apricity/Desktop/股票/data/interim/akshare/universes/large_cap_50_20200101_20241231_hfq_normalized.csv)
- U.S. zero-shot universe: [us_large_cap_30_20200101_20260320_hfq_normalized.csv](C:/Users/Apricity/Desktop/股票/data/interim/stooq/universes/us_large_cap_30_20200101_20260320_hfq_normalized.csv)
- U.S. metadata: [us_large_cap_30_metadata.csv](C:/Users/Apricity/Desktop/股票/data/interim/stooq/universes/us_large_cap_30_metadata.csv)
- Consolidated aligned table: [aligned_expert_ensemble_comparison.csv](C:/Users/Apricity/Desktop/股票/docs/us_a_share_multi_expert_report_assets/aligned_expert_ensemble_comparison.csv)

## Benchmark

The current `benchmark` is an **internal same-window, same-pool benchmark**, not `SPY` and not a tradable ETF baseline.

At the white-box risk layer, the benchmark return for each rebalance date is the mean forward return of the available pool on that date. The implementation is in [risk_pipeline.py](C:/Users/Apricity/Desktop/股票/risk_management/white_box/risk_pipeline.py): it computes `benchmark_realized = date_slice["realized_return"]` and then `benchmark_return = benchmark_realized.mean()`.

At the Alpaca-style execution layer, the backtest does not create a new benchmark; it reuses the `benchmark_return` path already written into `risk_periods.csv`. The implementation is in [backtest_alpaca_style.py](C:/Users/Apricity/Desktop/股票/execution/scripts/backtest_alpaca_style.py), where it loads `risk_periods.csv`, reads `benchmark_return`, and compounds it into `benchmark_total_return`.

This means the benchmark is best read as **“同窗同池内部参考线”**:
- same time window
- same candidate pool
- same rebalance rhythm

It also means `benchmark_total_return` is **not a universal market index benchmark**, so the cleanest cross-expert comparison right now is the combination of:
- aligned date window
- absolute execution return
- execution drawdown
- transaction cost

## Shared Setting

White-box risk parameters used in the aligned comparison:
- `top_k=5`
- `min_confidence=0.7`
- `min_close=5`
- `min_amount=100000000`
- `group_column=industry_group`, `max_per_group=1`
- `secondary_group_column=amount_bucket`, `secondary_max_per_group=2`
- `weighting=score_confidence`
- `max_position_weight=0.35`
- `transaction_cost_bps=10`

Alpaca-style execution replay settings:
- `default_account_equity=100000`
- `allow_fractional=true`
- `min_order_notional=50`
- `max_position_weight=0.35`
- `long_only=true`
- `order_type=market`
- `time_in_force=day`
- `cancel_open_orders_first=true`
- `order_sizing_mode=hybrid`
- `buying_power_buffer=0.97`
- `transaction_cost_bps=10`

## Single Experts

| Expert | Corr | Dir Acc | Risk Return | Risk Benchmark | Execution Return | Execution Benchmark | Excess | Max DD | Tx Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer | 0.0208 | 49.61% | 217.58% | 183.58% | 28.39% | 183.58% | -155.18% | -17.49% | 8519.42 |
| lstm | 0.1187 | 48.13% | 470.02% | 146.75% | 21.06% | 146.75% | -125.69% | -14.42% | 2315.30 |
| lightgbm | 0.0677 | 53.80% | 321.94% | 183.58% | 9.51% | 183.58% | -174.06% | -15.67% | 8337.50 |
| xgboost | 0.1652 | 56.84% | 108.53% | 183.58% | 8.40% | 183.58% | -175.18% | -17.11% | 11172.87 |
| catboost | 0.1161 | 56.65% | 142.43% | 39.00% | -14.01% | 39.00% | -53.00% | -21.92% | 2044.73 |

Key read:
- Best single-expert execution return on the aligned window is **transformer = 28.39%**.
- Strongest single-expert risk-layer return is `lstm = 470.02%`.
- Highest aligned zero-shot correlation among single experts is `xgboost = 0.1652`.

Aligned single-expert artifacts:
- LightGBM: [aligned lightgbm dir](C:/Users/Apricity/Desktop/股票/execution/experiments/us_a_share_expert_suite_aligned_20250313_20251223/lightgbm_regression_balanced)
- XGBoost: [aligned xgboost dir](C:/Users/Apricity/Desktop/股票/execution/experiments/us_a_share_expert_suite_aligned_20250313_20251223/xgboost_regression_balanced)
- CatBoost: [aligned catboost dir](C:/Users/Apricity/Desktop/股票/execution/experiments/us_a_share_expert_suite_aligned_20250313_20251223/catboost_regression_balanced)
- LSTM: [aligned lstm dir](C:/Users/Apricity/Desktop/股票/execution/experiments/us_a_share_expert_suite_aligned_20250313_20251223/lstm_regression_balanced)
- Transformer: [aligned transformer dir](C:/Users/Apricity/Desktop/股票/execution/experiments/us_a_share_expert_suite_aligned_20250313_20251223/transformer_regression_balanced)

## Ensemble

The ensemble combiner is implemented in:
- [predict_ensemble.py](C:/Users/Apricity/Desktop/股票/model_prediction/ensemble/scripts/predict_ensemble.py)
- [train_ensemble.py](C:/Users/Apricity/Desktop/股票/model_prediction/ensemble/scripts/train_ensemble.py)
- [shared.py](C:/Users/Apricity/Desktop/股票/model_prediction/ensemble/shared.py)
- [expert_registry.py](C:/Users/Apricity/Desktop/股票/model_prediction/common/expert_registry.py)

Aligned ensemble results:

| Ensemble | Corr | Dir Acc | Risk Return | Risk Benchmark | Execution Return | Execution Benchmark | Excess | Max DD | Tx Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ensemble_mean_score | 0.0535 | 48.20% | 412.70% | 183.58% | 36.32% | 183.58% | -147.26% | -15.19% | 7435.86 |
| ensemble_rank_average | 0.0591 | 56.65% | 354.19% | 183.58% | 17.20% | 183.58% | -166.38% | -14.83% | 9492.18 |
| ensemble_vote | 0.0756 | 56.65% | 331.84% | 152.62% | 15.75% | 152.62% | -136.86% | -19.51% | 8358.80 |

Key read:
- Best ensemble execution return is **ensemble_mean_score = 36.32%**.
- On this aligned setting, `mean_score` beats the other ensemble methods at the execution layer.
- Even the best ensemble still trails the internal same-window benchmark, which reinforces that the main bottleneck is execution realism rather than pure signal generation.

Ensemble artifacts:
- Rank average: [ensemble rank_average dir](C:/Users/Apricity/Desktop/股票/execution/experiments/us_a_share_expert_suite/ensemble_rank_average_regression_balanced)
- Mean score: [ensemble mean_score dir](C:/Users/Apricity/Desktop/股票/execution/experiments/us_a_share_expert_suite/ensemble_mean_score_regression_balanced)
- Vote: [ensemble vote dir](C:/Users/Apricity/Desktop/股票/execution/experiments/us_a_share_expert_suite/ensemble_vote_regression_balanced)

## Takeaways

- Infrastructure-wise, the project is now genuinely multi-expert: `lightgbm`, `xgboost`, `catboost`, `lstm`, `transformer`, and `ensemble` all run through the same downstream `signal -> white_box_risk -> execution` chain.
- Signal quality does not translate linearly into executable return. Several experts show strong risk-layer returns but much weaker Alpaca-style execution returns once turnover, weight caps, and transaction costs are applied.
- On the aligned window, the current best single expert by execution return is **transformer**, while the current best ensemble is **ensemble_mean_score**.
- The aligned results support a cautious conclusion: the system can generate transferable signals from A-share-trained models into U.S. equities, but the current execution layer is still too lossy for those signals to beat the internal same-window benchmark consistently.
- The next most valuable improvement is to add a **fixed external benchmark** such as `SPY buy-and-hold` or a fixed equal-weight U.S. pool benchmark, so future reports do not rely only on the internal same-window benchmark.
