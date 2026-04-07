# H5 Target Engineering Plan

## Current State

The current h5 target is defined in [us_equities_baseline.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/us_equities_baseline.py):

- `future_return = adj_open(t+h+1) / adj_open(t+1) - 1`
- `benchmark_future_return = benchmark_adj_open(t+h+1) / benchmark_adj_open(t+1) - 1`
- `target = future_return - benchmark_future_return`

Relevant code:
- [build_research_frame()](/E:/CodeX/StockMachine-260321/src/stockmachine/research/us_equities_baseline.py#L649)
- [target assignment](/E:/CodeX/StockMachine-260321/src/stockmachine/research/us_equities_baseline.py#L717)
- [model fitting uses `prepared_train[\"target\"]`](/E:/CodeX/StockMachine-260321/src/stockmachine/research/us_equities_baseline.py#L1134)

This target is already better than raw future return because it removes a first-order benchmark trend. But it still leaves at least three issues:

1. It is not beta-neutral.
2. It is not sector-neutral.
3. It still rewards long-only directional implementations more than pure cross-sectional alpha implementations.

That diagnosis is consistent with the pure-alpha implementation study in:
- [summary_metrics.csv](/E:/CodeX/StockMachine-260321/artifacts/h5_pure_alpha_impl_compare_20260405/summary_metrics.csv)


## Goal

Redefine the h5 target so the model learns a cleaner cross-sectional alpha signal:

- less market-direction contamination
- less sector-direction contamination
- better transfer into long-short / neutral implementations
- ideally better standalone alpha Sharpe, not just better long-only beta capture


## Priority Targets

### 1. Benchmark / Beta Residual Target

#### Motivation

The current target subtracts benchmark return one-for-one. That assumes every stock has beta exactly `1.0`, which is false.

Stocks with:
- `beta > 1`
- `beta < 1`
- time-varying beta

are mis-specified by the current target. A beta-residual target tries to remove only the stock's own market exposure.

#### Definition

Estimate a lagged rolling beta for each stock:

`beta_{i,t} = Cov(ret_{i,1d}, benchmark_ret_{1d}) / Var(benchmark_ret_{1d})`

using only data available up to date `t`.

Then define:

`target_beta_residual = future_return - beta_{i,t} * benchmark_future_return`

#### Implementation Notes

- Use a rolling window such as `60` trading days first.
- Require a minimum history threshold such as `40` valid observations.
- Use only lagged returns. No future information may enter beta estimation.
- Clip or shrink beta if needed for stability, for example to `[0, 2.5]` or with mild shrinkage toward `1.0`.

#### Expected Effect

- Should reduce directional market contamination.
- Should help if current long-only strength partly comes from high-beta selection.
- May reduce raw long-only return, but improve pure-alpha quality.


### 2. Sector Residual Target

#### Motivation

If the model is mostly learning "buy the strongest sector" and then choosing names inside it, it is not learning pure stock selection.

Sector residualization forces the model to explain:

- which stock is better than its sector peers

rather than:

- which sector is better than the market

#### Definition

For each date and sector:

`sector_future_return_mean_{s,t} = mean(future_return_{j,t} for j in sector s)`

Then define:

`target_sector_residual = future_return - sector_future_return_mean`

#### Implementation Notes

- This uses future cross-sectional outcomes on the same date as labels. That is acceptable because labels are realized outcomes, not features.
- Sector membership must remain point-in-time correct, which the existing metadata handling already supports.
- This target is naturally aligned with stronger sector-neutral or long-short implementations.

#### Expected Effect

- Should reduce sector chasing.
- Should improve "top vs bottom within sector" discrimination.
- Likely to help pure-alpha more than long-only headline return.


## Recommended Combined Target

After the two single-target tests, the most natural combined version is:

`target_beta_sector_residual = future_return - beta_{i,t} * benchmark_future_return - sector_mean(beta-adjusted future return)`

Operationally:

1. compute `beta_adjusted_future_return = future_return - beta_{i,t} * benchmark_future_return`
2. compute within-date, within-sector mean of that adjusted future return
3. subtract the sector mean

This is the cleanest "market-adjusted, sector-adjusted" target, but it is not the right first experiment because it combines two new ideas at once.


## Suggested Rollout Order

### Phase 1

Add and test these target variants:

- `benchmark_excess`
  - current baseline
- `beta_residual_60d`
- `sector_residual`

### Phase 2

If at least one of Phase 1 improves pure-alpha diagnostics:

- `beta_sector_residual_60d`

### Phase 3

Only if needed:

- volatility-scaled residual target
- bucketed residual target


## Evaluation Framework

The current problem is not just headline long-only return. So each target variant should be tested on three layers:

### A. Model Layer

- daily rank IC
- top-bottom spread
- hit rate

### B. Current Production-Like Long-Only Layer

- annualized return
- annualized volatility
- Sharpe
- max drawdown
- turnover
- cost stress

### C. Pure-Alpha Diagnostic Layer

- market-neutral long-short diagnostic
- sector-neutral long-short diagnostic
- benchmark beta
- benchmark correlation

This third layer matters most for target engineering.


## Model Priority

Target engineering should not be rolled out to every model equally at first.

### First priority

- `lightgbm_ranker`

Reason:
- current main alpha sleeve
- strongest h5 product value
- likely most sensitive to cleaner cross-sectional targets

### Second priority

- `hist_gbm`

Reason:
- good h5 candidate
- already benefits from richer features

### Third priority

- `extra_trees`

Reason:
- useful baseline
- but current evidence suggests it is less promising as the main pure-alpha engine than `lightgbm_ranker`


## Integration Design

The h1 line already has a clean target abstraction:
- [H1TargetConfig](/E:/CodeX/StockMachine-260321/src/stockmachine/research/h1_us_equities.py#L123)

h5 should follow the same pattern instead of hard-coding a single target formula.

### Proposed h5 target config

Add a small immutable config, for example in [us_equities_baseline.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/us_equities_baseline.py):

- `target_kind: str = "benchmark_excess"`
- `beta_lookback_days: int = 60`
- `beta_min_obs: int = 40`
- `beta_clip_low: float = 0.0`
- `beta_clip_high: float = 2.5`

### Proposed target columns in research frame

Add:

- `target_benchmark_excess`
- `target_beta_residual`
- `target_sector_residual`
- `target_beta_sector_residual`
- `target_beta_estimate`

Then resolve:

- `panel["target"] = panel[target_config.selected_column]`

This keeps downstream training code unchanged.


## Exact Code Touchpoints

### Research frame

- [build_research_frame()](/E:/CodeX/StockMachine-260321/src/stockmachine/research/us_equities_baseline.py#L649)

This is where:
- benchmark future return is built
- stock future return is built
- target is currently assigned

### Prediction assembly

- [_assemble_predictions()](/E:/CodeX/StockMachine-260321/src/stockmachine/research/us_equities_baseline.py#L1427)

This should continue to export:
- `target`
- `future_return`
- `benchmark_future_return`

and ideally also:
- `target_kind`
- `target_beta_estimate`

for auditability.

### Model fit path

- [fit_predict_base_model()](/E:/CodeX/StockMachine-260321/src/stockmachine/research/us_equities_baseline.py#L1129)

This already trains on `prepared_train["target"]`, so if the frame's target is switched cleanly, model code does not need to change.


## What Success Looks Like

A target redesign is successful if it improves at least one of these without badly damaging the others:

1. Higher long-short diagnostic Sharpe
2. Lower benchmark beta / correlation in the diagnostic implementations
3. Similar or better long-only Sharpe after costs
4. Better sleeve usefulness when combined with `C policy`

The highest priority success condition is:

**improving pure-alpha diagnostics, not just long-only headline return**


## Recommended Next Experiment

Run a tight experiment matrix on h5:

- models:
  - `lightgbm_ranker`
  - `hist_gbm`
  - `extra_trees`
- targets:
  - `benchmark_excess`
  - `beta_residual_60d`
  - `sector_residual`
- evaluations:
  - current long-only
  - current pure-alpha diagnostic evaluator

If either `beta_residual_60d` or `sector_residual` improves pure-alpha quality for `lightgbm_ranker`, then promote that target into the next h5 mainline comparison.
