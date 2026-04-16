# C2 Candidate Triage 2026-04-09

## Purpose

This memo compresses the current `C2` candidate set into three buckets:

- retain
- observe
- de-prioritize

It is based on the latest:

- risk-budget frontier sweep
- implementation sweep
- apples-to-apples periodic-rebalance `current C` baseline

This is a research triage memo, not a final promotion memo.

## Retain

### 1. `current_c_policy_core_periodic_rebalance`

Role:

- best current standalone baseline

Why retain:

- still the strongest Sharpe baseline after removing the passive-drift
  objection
- common-window metrics: `11.57% / 7.44% / Sharpe 1.510 / MDD -11.48%`
- this means `C2` must beat a clean, explicit, monthly-rebalanced version of
  `current C`, not a straw-man baseline

Decision:

- retain as the standing benchmark

### 2. `c2_impl_lead_cash05_calendar`

Equivalent policy region:

- `equity total 35%`
- `credit 5%`
- `duration 15%`
- `inflation_hedge 20%`
- `trend 25%`
- `cash reserve 5%`
- monthly calendar rebalance

Why retain:

- best current `C2` implementation candidate
- common-window metrics: `8.32% / 5.79% / Sharpe 1.409 / MDD -9.39%`
- live-window metrics: `8.36% / 6.11% / Sharpe 1.345 / MDD -5.50%`
- it is materially better than `C2 v0`
- it preserves the lower-volatility, shallower-drawdown character of the `C2`
  family

Decision:

- retain as the lead `C2` candidate

## Observe

### 3. `rolling_c2_region_e35_c05_d15_i15_t30`

Why observe:

- very close to the lead candidate on Sharpe
- common-window metrics: `8.11% / 5.68% / Sharpe 1.401 / MDD -8.95%`
- slightly lower return, but also slightly lower volatility and shallower
  drawdown
- it is the cleanest near-neighbor to the lead candidate inside the same strong
  region

Decision:

- keep as the main alternate candidate

### 4. `rolling_c2_region_e40_c05_d15_i20_t20`

Why observe:

- best higher-equity variant tested so far
- common-window metrics: `8.28% / 5.89% / Sharpe 1.381 / MDD -9.81%`
- it did not beat the `35%` equity lead, but it is still the best challenge
  candidate if we later want to revisit whether `C2` is too conservative

Decision:

- keep only as a secondary challenge point, not as a mainline candidate

## De-Prioritize

### 5. `cash00` implementation variants

Examples:

- `c2_impl_lead_cash00_calendar`
- `c2_impl_lead_cash00_band_t06_s21`

Why de-prioritize:

- they have higher return than the `cash05` lead, but worse Sharpe and worse
  drawdown
- the implementation sweep consistently preferred `cash reserve 5%`

Decision:

- do not expand this branch further unless future live evidence specifically
  points against the cash sleeve

### 6. Threshold-aware rebalance variants

Examples:

- `c2_impl_lead_cash05_band_t02_s21`
- `c2_impl_lead_cash05_band_t04_s42`

Why de-prioritize:

- none of them beat the simple monthly calendar implementation
- some are nearly identical to calendar because the threshold rarely binds
- the rest mainly add complexity without increasing Sharpe

Decision:

- do not use threshold-aware rebalance as the default `C2` implementation path

### 7. Higher-credit or higher-duration `C2` regions

Examples:

- `rolling_c2_region_e35_c10_d15_i20_t20`
- `rolling_c2_region_e35_c10_d15_i15_t25`
- `rolling_c2_region_e35_c05_d20_i20_t20`

Why de-prioritize:

- the frontier sweep consistently favored `credit 5%` over `credit 10%`
- the stronger candidates also clustered around `duration 15%`, not `20%` or
  `25%`

Decision:

- do not expand these regions unless a new out-of-sample result points back to
  them

## Working Shortlist

If we compress everything to the smallest useful shortlist, it should be:

- benchmark: `current_c_policy_core_periodic_rebalance`
- lead `C2`: `c2_impl_lead_cash05_calendar`
- alternate `C2`: `rolling_c2_region_e35_c05_d15_i15_t30`

## Recommended Next Step

The next step should not be another broad sweep.

It should be one of:

- live-era side-by-side monitoring of:
  - `current_c_policy_core_periodic_rebalance`
  - `c2_impl_lead_cash05_calendar`
- one final narrow policy pass around the retained `C2` pair:
  - `e35 / c05 / d15 / i20 / t25`
  - `e35 / c05 / d15 / i15 / t30`

Everything else should now be treated as explored background, not active
mainline research.
