# Peer Comparison Backtest Protocol

This document freezes the current externally shareable backtest protocol for
peer-to-peer comparison as of 2026-03-22.

It supersedes the older six-model comparison note in
[peer-comparison-backtest-setting.md](./peer-comparison-backtest-setting.md).

## Repository Version

- branch: `codex/stable-ops-status-20260322`
- commit: `b7a59b0`

## Purpose

This protocol is intended for another team that wants to run:

- the same data scope
- the same feature and label definitions
- the same walk-forward split logic
- the same white-box portfolio overlay
- the same backtest engine assumptions

The goal is to make model-level performance comparisons as apples-to-apples as
possible.

## Comparison Mode

Use the current `strict` protocol driven by:

- [research-protocol.md](./research-protocol.md)
- [historical-universe-contract.md](./historical-universe-contract.md)
- [run_p1_rigor_suite.py](/E:/CodeX/StockMachine-260321/src/stockmachine/apps/run_p1_rigor_suite.py)

Reference output from the latest all-model rerun:

- [summary_metrics.csv](/E:/CodeX/StockMachine-260321/artifacts/p1_rigor_suite_refresh_20260322/strict_full/summary_metrics.csv)

## Market Scope

- market: US equities
- frequency: daily
- benchmark: `SPY`
- prediction start: `2025-01-01`
- holding horizon: `5` sessions

## Data Coverage

The current strict rerun loads the full `silver` tables by concatenating every
JSONL file under each table directory and deduplicating by canonical primary
key.

Loader:

- [silver.py](/E:/CodeX/StockMachine-260321/src/stockmachine/data/loaders/silver.py)

Loaded dataset coverage at current head:

| Table | Rows | Symbols | Date Column | Min Date | Max Date |
| --- | ---: | ---: | --- | --- | --- |
| `universe_membership` | `119,724` | `66` | `session_date` | `2019-01-02` | `2026-03-20` |
| `daily_bar` | `119,724` | `66` | `session_date` | `2019-01-02` | `2026-03-20` |
| `adj_factor` | `121,538` | `67` | `session_date` | `2019-01-02` | `2026-03-20` |
| `benchmark_index` | `1,814` | `1` | `session_date` | `2019-01-02` | `2026-03-20` |
| `industry_membership` | `119,724` | `66` | `as_of_date` | `2019-01-02` | `2026-03-20` |
| `symbol_master` | `133,132` | `13,408` | `as_of_date` | `2019-01-02` | `2026-03-21` |

## Universe

The default research universe is `us_equities_research_v1`.

Resolver:

- [universe.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/universe.py)

Current default point-in-time universe size:

- `66` symbols per active session

Current fixed research basket:

```text
AAPL, MSFT, NVDA, AMZN, META, GOOGL, GOOG, TSLA, AVGO, ORCL, CRM, ADBE, NFLX, AMD, INTC, QCOM, TXN, CSCO, IBM, AMAT, JPM, BAC, WFC, GS, MS, C, V, MA, AXP, BLK, SCHW, JNJ, UNH, PFE, ABBV, MRK, LLY, ABT, TMO, DHR, AMGN, GILD, XOM, CVX, COP, SLB, HD, LOW, COST, WMT, TGT, KO, PEP, MCD, NKE, SBUX, CAT, DE, GE, HON, BA, LMT, DIS, CMCSA, VZ, T
```

### Current point-in-time behavior

- membership is resolved from explicit `universe_membership` when present
- missing explicit membership falls back conservatively to `symbol_master`
- metadata is resolved using the latest visible snapshot at or before each
  `session_date`

Important limitation:

- `universe_membership` and `industry_membership` currently come from static
  historical backfill, not true constituent-history events

## Price And Adjustment Rules

Research and backtest use:

- `daily_bar` for OHLCV
- `adj_factor` for split/dividend adjustment
- `benchmark_index` for `SPY`

Price-panel builder:

- [us_equities_baseline.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/us_equities_baseline.py)

Return construction uses adjusted opens:

- entry price: `adj_open(T+1)`
- exit price: `adj_open(T+6)`

## Feature Sets

### Shared tabular feature set

Most non-sequence models use the shared 12-column daily feature set:

```text
gap_1
ret_1d
mom_5
mom_10
mom_20
mom_60
vol_20
vol_60
range_1d
volume_ratio_20
rel_mom_20
rel_mom_60
```

Definition:

- [common.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/builders/common.py)

### Sequence-model feature set

`lstm_regressor` and `transformer_regressor` use a richer sequence feature set:

```text
intraday_return
range_pct
return_1d
return_5d
return_10d
return_20d
close_ma5_gap
close_ma10_gap
close_ma20_gap
close_ma60_gap
volume_ma5_ratio
volume_ma20_ratio
volatility_5d
volatility_10d
volatility_20d
breakout_20d
distance_to_low_20d
price_position_20d
volume_zscore_20d
market_return_1d
market_return_5d
relative_return_1d
relative_return_5d
cs_rank_return_1d
cs_rank_return_5d
cs_rank_return_20d
cs_rank_volume_ma5_ratio
cs_rank_dollar_volume
dollar_volume_ma5_ratio
```

Definition:

- [sequence_models.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/builders/sequence_models.py)

## Label Definition

Task:

- cross-sectional prediction of future excess return versus `SPY`

Timing contract:

- features are available after session `T` close
- signal is generated after session `T` close
- entry is the next session open, `T+1`
- exit is the open of `T+6`
- holding period is `5` sessions

Per-symbol future return:

- `adj_open(T+6) / adj_open(T+1) - 1`

Benchmark future return:

- `SPY_adj_open(T+6) / SPY_adj_open(T+1) - 1`

Training target:

- `target = stock_future_return - benchmark_future_return`

## Walk-Forward Protocol

Strict walk-forward settings:

- train window: `36 months`
- validation window: `6 months`
- test window: `6 months`
- roll frequency: `monthly`
- purge window: `6 sessions`
- embargo window: `1 session`

Implementation:

- [splitting.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/splitting.py)

Important implementation detail:

- each test fold is scored after fitting on `train + validation`
- the validation segment currently acts as a temporal buffer and protocol
  scaffold, not as a nested hyperparameter-search layer

## Portfolio Construction

Portfolio rules:

- long-only
- `top_k = 10`
- equal-weight selected names
- integer-share sizing
- no fractional shares

White-box filters:

- `close >= 10`
- `median_dollar_volume_20 >= 50,000,000`
- `vol_20 <= 0.04`
- max `2` names per sector
- sector neutralization enabled

Sector neutralization rule:

- for each session, subtract the sector mean score before ranking

Policy:

- [policies.py](/E:/CodeX/StockMachine-260321/src/stockmachine/portfolio/policies.py)

## Execution And Cost Assumptions

Execution policy:

- market-on-open

Order sizing:

- `shares = floor(account_equity * target_weight / next_open_price)`

Backtest engine assumptions:

- next-open entry
- next-open exit after `5` bars
- initial equity: `1,000,000`
- cost per side: `10 bps`

Net-return logic:

- gross holding-period return
- turnover-based cost deducted as `turnover * 10 bps`

Engine:

- [simple_engine.py](/E:/CodeX/StockMachine-260321/src/stockmachine/backtest/simple_engine.py)

Execution policy:

- [policies.py](/E:/CodeX/StockMachine-260321/src/stockmachine/execution/policies.py)

## Model Coverage

The latest strict all-model rerun covers every currently registered alpha
expert in [registry.py](/E:/CodeX/StockMachine-260321/src/stockmachine/alpha/registry.py):

```text
factor_baseline
ridge
huber_regression
elastic_net
hist_gbm
extra_trees
random_forest
lightgbm_regressor
lightgbm_ranker
catboost_regressor
xgboost_regressor
lstm_regressor
transformer_regressor
ensemble_hist_gbm_ridge_mean
ensemble_hist_gbm_ridge_rank
ensemble_hist_gbm_random_forest_mean
ensemble_hist_gbm_random_forest_rank
ensemble_hist_gbm_lightgbm_regressor_mean
ensemble_hist_gbm_lightgbm_regressor_rank
ensemble_random_forest_lightgbm_regressor_mean
ensemble_random_forest_lightgbm_regressor_rank
ensemble_hist_gbm_ridge_random_forest_mean
ensemble_hist_gbm_ridge_random_forest_rank
ensemble_hist_gbm_random_forest_lightgbm_regressor_mean
ensemble_hist_gbm_random_forest_lightgbm_regressor_rank
```

Total models in the current full sweep:

- `25`

## What `run_p1_rigor_suite` Does

Default CLI:

- [run_p1_rigor_suite.py](/E:/CodeX/StockMachine-260321/src/stockmachine/apps/run_p1_rigor_suite.py)

With default arguments, it does all of the following:

1. strict full rerun for all registered models
2. chooses the top `3` successful models by Sharpe unless `--analysis-models`
   is provided
3. builds yearly and quarterly stability summaries for those analysis models
4. runs cost-stress analysis on those analysis models
5. runs a `top_k` sweep for those analysis models

So:

- `strict_full` covers all `25` models
- `stability`, `cost_stress`, and `topk_sweep` cover the current top `3` by default

## Output Metrics

Primary summary fields:

- `sessions`
- `total_return`
- `annualized_return`
- `annualized_volatility`
- `sharpe`
- `max_drawdown`
- `benchmark_total_return`
- `mean_turnover`
- `mean_cost_bps`

## Latest Full-Rerun Leaderboard

Latest strict full rerun result:

- [summary_metrics.csv](/E:/CodeX/StockMachine-260321/artifacts/p1_rigor_suite_refresh_20260322/strict_full/summary_metrics.csv)

Top five models in the latest full rerun:

| Rank | Model | Total Return | Excess vs SPY | Annualized Return | Sharpe | Max Drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `ensemble_hist_gbm_random_forest_lightgbm_regressor_rank` | `29.90%` | `12.43%` | `26.02%` | `1.28` | `-17.60%` |
| 2 | `ensemble_hist_gbm_lightgbm_regressor_rank` | `30.48%` | `13.02%` | `26.52%` | `1.27` | `-19.05%` |
| 3 | `ensemble_random_forest_lightgbm_regressor_mean` | `24.90%` | `7.44%` | `21.73%` | `1.09` | `-18.99%` |
| 4 | `ensemble_hist_gbm_random_forest_lightgbm_regressor_mean` | `24.75%` | `7.29%` | `21.60%` | `1.08` | `-18.05%` |
| 5 | `hist_gbm` | `23.25%` | `5.78%` | `20.30%` | `1.02` | `-20.18%` |

Benchmark reference:

- `SPY total return = 17.46%`

## Reproduction Commands

### 1. Materialize current historical metadata backfill

```powershell
$env:PYTHONPATH='src'
@'
from stockmachine.ingestion.jobs import backfill_static_metadata_history_from_silver
print(backfill_static_metadata_history_from_silver())
'@ | python -
```

### 2. Run the full strict rigor suite

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_p1_rigor_suite --output-root artifacts/p1_rigor_suite_refresh_20260322
```

### 3. Run only the strict full sweep for a subset of models

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_p1_rigor_suite --models hist_gbm lightgbm_regressor ensemble_hist_gbm_random_forest_lightgbm_regressor_rank --analysis-models hist_gbm lightgbm_regressor ensemble_hist_gbm_random_forest_lightgbm_regressor_rank --output-root artifacts/peer_subset_compare
```

## Known Limitations

- research universe is still a fixed 66-name basket, not true historical
  constituent evolution
- `universe_membership` and `industry_membership` are currently based on
  static historical backfill, not event-accurate historical updates
- transaction costs are still modeled as simplified turnover-based `bps`
  costs, not a full broker fee plus spread plus slippage model
- no bootstrap confidence intervals or multiple-testing correction are applied
  in this protocol
- the silver loader still emits a pandas concat `FutureWarning`
