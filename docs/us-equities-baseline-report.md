# US Equities Baseline Report

## Run Date

2026-03-21

## Scope

This is the first end-to-end baseline experiment for the repository.

- market: US equities
- data source: Yahoo Finance via `yfinance`
- universe: 66 current large-cap US stocks plus `SPY` as benchmark
- history window: 2019-01-01 to 2025-12-31
- prediction start: 2024-01-01
- validation period: 2024
- test period: 2025
- label: next 5-session excess return from next open
- rebalance for strategy metric: every 5 sessions
- portfolio metric: equal-weight top 10 predicted names

## Features

The baseline uses 12 simple daily-bar features:

- `gap_1`
- `ret_1d`
- `mom_5`
- `mom_10`
- `mom_20`
- `mom_60`
- `vol_20`
- `vol_60`
- `range_1d`
- `volume_ratio_20`
- `rel_mom_20`
- `rel_mom_60`

## Models

- `factor_baseline`: white-box ranked factor blend
- `ridge`: linear regression with scaling and median imputation
- `hist_gbm`: histogram gradient boosting regressor

## Core Results

Validation and test metrics are stored in
`artifacts/us_equities_baseline/summary_metrics.csv`.

### Validation 2024

| Model | Mean Rank IC | Top-Bottom Spread | Top-10 Total Return | Top-10 Sharpe | Benchmark Total Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| factor_baseline | 0.0213 | 0.00118 | 0.2500 | 1.3885 | 0.2514 |
| ridge | -0.0054 | 0.00048 | 0.1774 | 0.7585 | 0.2514 |
| hist_gbm | 0.0223 | 0.00348 | 0.4398 | 2.0011 | 0.2514 |

### Test 2025

| Model | Mean Rank IC | Top-Bottom Spread | Top-10 Total Return | Top-10 Sharpe | Benchmark Total Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| factor_baseline | -0.0203 | -0.00356 | 0.1298 | 0.6911 | 0.1755 |
| ridge | 0.0338 | 0.00449 | 0.3812 | 1.2770 | 0.1755 |
| hist_gbm | 0.0338 | 0.00304 | 0.4048 | 1.4826 | 0.1755 |

## Quick Takeaways

- the white-box factor blend was not stable enough out of sample
- the linear and tree baselines both produced positive 2025 sample-out-of-sample
  rank IC
- `hist_gbm` was the strongest first-pass model on the 2025 test set
- `ridge` was competitive and may be easier to interpret and regularize

## Rough Cost Sensitivity

These numbers are not yet integrated into the main report pipeline. They are a
rough sanity check assuming a flat round-trip cost per 5-session rebalance.

### 2025 Test Period

| Model | Gross Total Return | Net at 20 bps | Net at 40 bps |
| --- | ---: | ---: | ---: |
| ridge | 0.3812 | 0.2529 | 0.1362 |
| hist_gbm | 0.4048 | 0.2743 | 0.1557 |

## Known Limits

- public Yahoo Finance data was used because Alpaca credentials were not
  available in the environment
- the universe uses current large-cap names, so this run has survivorship bias
- no point-in-time fundamentals were included
- no formal transaction-cost model is baked into the main evaluation loop yet
- no sector-neutralization or exposure control was applied yet
- the strategy metric is a simple top-10 long-only cohort simulation

## Artifacts

- raw price panel: `artifacts/us_equities_baseline/price_data.csv`
- research frame: `artifacts/us_equities_baseline/research_frame.csv`
- predictions: `artifacts/us_equities_baseline/predictions.csv`
- summary metrics: `artifacts/us_equities_baseline/summary_metrics.csv`
