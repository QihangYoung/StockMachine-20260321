# Peer Comparison Backtest Setting

This document freezes the current strict backtest setting used for peer
comparison as of 2026-03-22.

It is intended to be shared externally so another team can reproduce a close
match of the current StockMachine research protocol.

## Repository Version

- branch: `codex/stable-ops-status-20260322`
- commit: `83cbdc5`

## Recommended Comparison Mode

Use the current `P0 Strict` protocol, not the earlier looser protocol.

Reference outputs:

- [summary_metrics.csv](/E:/CodeX/StockMachine-260321/artifacts/p0_rigor_rerun/summary_metrics.csv)
- [research_protocol.json](/E:/CodeX/StockMachine-260321/artifacts/p0_rigor_rerun/research_protocol.json)
- [p0-research-rigor-report.md](/E:/CodeX/StockMachine-260321/docs/p0-research-rigor-report.md)

## Market Scope

- market: US equities
- frequency: daily
- benchmark: `SPY`
- prediction start: `2025-01-01`

## Universe

Current comparison universe is a fixed 66-name large-cap, liquid US equities
basket:

```text
AAPL, MSFT, NVDA, AMZN, META, GOOGL, GOOG, TSLA, AVGO, ORCL, CRM, ADBE, NFLX, AMD, INTC, QCOM, TXN, CSCO, IBM, AMAT, JPM, BAC, WFC, GS, MS, C, V, MA, AXP, BLK, SCHW, JNJ, UNH, PFE, ABBV, MRK, LLY, ABT, TMO, DHR, AMGN, GILD, XOM, CVX, COP, SLB, HD, LOW, COST, WMT, TGT, KO, PEP, MCD, NKE, SBUX, CAT, DE, GE, HON, BA, LMT, DIS, CMCSA, VZ, T
```

Universe anchor:

- [us_equities_baseline.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/us_equities_baseline.py)

## Data Inputs

The strict rerun uses silver-layer tables:

- `daily_bar`
- `adj_factor`
- `benchmark_index`
- `symbol_master`
- `industry_membership`

Relevant loaders and helpers:

- [silver.py](/E:/CodeX/StockMachine-260321/src/stockmachine/data/loaders/silver.py)
- [universe.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/universe.py)

## Point-in-Time Metadata Rules

The current strict rerun resolves metadata using the latest visible snapshot at
or before each `session_date`.

Important note:

- the original silver metadata history was incomplete
- to remove future-dated metadata leakage, we explicitly materialize static
  metadata snapshots across historical sessions
- this is a bootstrap compromise, not a true historical constituent history

Backfill helper:

- [bootstrap_yahoo_us_equities.py](/E:/CodeX/StockMachine-260321/src/stockmachine/ingestion/jobs/bootstrap_yahoo_us_equities.py)

## Feature Set

All current comparison models use the same 12 baseline daily features:

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

Feature definition:

- [common.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/builders/common.py)

## Label Definition

Task type:

- cross-sectional prediction of future excess return

Holding horizon:

- `5` sessions

Timing contract:

- features available by `T` close
- signal generated after `T` close
- entry at `T+1` open
- exit at `T+6` open

Per-symbol future return:

- `adj_open(T+6) / adj_open(T+1) - 1`

Benchmark future return:

- `SPY adj_open(T+6) / SPY adj_open(T+1) - 1`

Training target:

- `target = stock_future_return - benchmark_future_return`

Protocol anchor:

- [research-protocol.md](/E:/CodeX/StockMachine-260321/docs/research-protocol.md)

## Walk-Forward Validation

Strict walk-forward settings:

- train window: `36 months`
- validation window: `6 months`
- test window: `6 months`
- roll frequency: `monthly`
- purge window: `6 sessions`
- embargo window: `1 session`

Implementation note:

- each fold is fit on `train + validation`
- held-out scoring happens on `test`
- the validation segment is currently used as a temporal buffer and protocol
  scaffold, not yet as a separate hyperparameter search layer

Splitter implementation:

- [splitting.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/splitting.py)

## Portfolio Construction

Portfolio rules:

- long-only
- `top_k = 10`
- equal-weight selected names
- integer-share sizing
- no fractional shares

Risk and selection filters:

- `close >= 10`
- `median_dollar_volume_20 >= 50,000,000`
- `vol_20 <= 0.04`
- max `2` names per sector
- sector neutralization enabled

Sector neutralization rule:

- per date, subtract sector mean score before ranking

Portfolio policy:

- [policies.py](/E:/CodeX/StockMachine-260321/src/stockmachine/portfolio/policies.py)

## Execution Assumption

Execution policy:

- market-on-open

Order sizing:

- quantity = `floor(account_equity * target_weight / next_open_price)`

Execution policy:

- [policies.py](/E:/CodeX/StockMachine-260321/src/stockmachine/execution/policies.py)

## Backtest Engine

Engine type:

- fixed-horizon next-open entry and next-open exit

Parameters:

- horizon bars: `5`
- initial equity: `1,000,000`
- cost per side: `10 bps`

Net-return calculation:

- gross portfolio return over the holding period
- turnover-based cost deducted as `turnover * 10 bps`

Engine implementation:

- [simple_engine.py](/E:/CodeX/StockMachine-260321/src/stockmachine/backtest/simple_engine.py)

## Current Comparison Models

The current strict rerun compares these 6 models:

- `factor_baseline`
- `hist_gbm`
- `random_forest`
- `lightgbm_regressor`
- `ensemble_hist_gbm_random_forest_rank`
- `ensemble_random_forest_lightgbm_regressor_rank`

Registry:

- [registry.py](/E:/CodeX/StockMachine-260321/src/stockmachine/alpha/registry.py)

### Model Defaults

`hist_gbm`

- `learning_rate=0.05`
- `max_depth=4`
- `max_iter=200`
- `min_samples_leaf=40`
- `random_state=7`

`random_forest`

- `n_estimators=300`
- `max_depth=6`
- `max_features='sqrt'`
- `min_samples_leaf=40`
- `bootstrap=True`
- `random_state=7`

`lightgbm_regressor`

- `objective='regression'`
- `learning_rate=0.03`
- `n_estimators=400`
- `num_leaves=31`
- `min_child_samples=40`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `reg_alpha=0.0`
- `reg_lambda=1.0`
- `random_state=7`
- `n_jobs=-1`

`ensemble_hist_gbm_random_forest_rank`

- component models: `hist_gbm + random_forest`
- combine rule: percentile-rank average per date

`ensemble_random_forest_lightgbm_regressor_rank`

- component models: `random_forest + lightgbm_regressor`
- combine rule: percentile-rank average per date

Builder anchors:

- [sklearn_models.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/builders/sklearn_models.py)
- [lightgbm_models.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/builders/lightgbm_models.py)

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

## Strict-Rerun Results

Current strict rerun leaderboard:

| Model | Sessions | Total Return | Excess vs SPY | Annualized Return | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hist_gbm` | 57 | `23.25%` | `5.78%` | `20.30%` | `1.02` | `-20.18%` |
| `lightgbm_regressor` | 57 | `20.69%` | `3.22%` | `18.09%` | `0.92` | `-22.72%` |
| `ensemble_hist_gbm_random_forest_rank` | 57 | `18.76%` | `1.30%` | `16.42%` | `0.86` | `-18.74%` |
| `ensemble_random_forest_lightgbm_regressor_rank` | 57 | `17.00%` | `-0.47%` | `14.89%` | `0.82` | `-19.58%` |
| `factor_baseline` | 57 | `9.22%` | `-8.24%` | `8.11%` | `0.54` | `-14.86%` |
| `random_forest` | 57 | `10.08%` | `-7.39%` | `8.86%` | `0.52` | `-21.08%` |

Benchmark reference:

- `SPY total return = 17.46%`

## Reproduction Commands

Backfill metadata history:

```powershell
$env:PYTHONPATH='src'
@'
from stockmachine.ingestion.jobs import backfill_static_metadata_history_from_silver
print(backfill_static_metadata_history_from_silver())
'@ | python -
```

Run strict comparison sweep:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_us_equities_model_sweep --models factor_baseline hist_gbm random_forest lightgbm_regressor ensemble_hist_gbm_random_forest_rank ensemble_random_forest_lightgbm_regressor_rank --predict-start 2025-01-01 --output-root artifacts/p0_rigor_rerun
```

## Known Limitations

- fixed 66-name universe, not true historical constituent membership
- metadata history currently comes from a static bootstrap expanded across dates
- no bootstrap confidence intervals or multiple-testing correction in this pass
- silver loader still emits a pandas concat `FutureWarning`
