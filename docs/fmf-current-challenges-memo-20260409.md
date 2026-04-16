# FMF Rebuild Current Challenges Memo

Date: 2026-04-09

## Purpose

This memo summarizes the current major challenges in the rebuilt `FMF`
multi-asset validation line and evaluates whether the main contradiction is the
time-series nonstationarity of macro environments.

Current rebuilt setup:

- universe: `SPY / VXUS / IEF / LQD / GLD / FMF / BIL`
- validation: `2013-08-01 ~ 2019-12-31`
- lockbox test: `2020-01-02 ~ 2026-04-08`
- frozen shortlist:
  - `rolling_fmf_c2_e42_c10_d22_i20_t06`
  - `rolling_fmf_c2_e42_c10_d22_i18_t08`
  - `rolling_c2_v0_seed_core`
  - plus `ERC` and naive baselines

## Current Major Challenges

### 1. Validation/Test Regime Mismatch

This is the first-order statistical challenge.

The validation window is mostly a pre-COVID, low-inflation, low-rate period
with one notable `2018 Q4` risk-off shock. The lockbox test window contains:

- the `2020` liquidity shock
- the `2020-2021` policy-driven bull market
- the `2022` inflation and tightening regime
- the `2023-2026` post-inflation mixed environment

These are not small perturbations around one stable data-generating process.
They are materially different macro environments.

Therefore a parameter region that looks best on validation can fail badly on
test without implying a coding error. It may simply be adapted to the wrong
environment.

### 2. Trend-Bucket Representative Quality

The rebuilt data architecture solved one problem by introducing another.

We replaced the older `CTA`-centered setup with `FMF` to obtain a longer clean
ETF-only live window and a proper validation/test split. That was the right
move from an evaluation-architecture perspective.

But the trend representative itself is now a live issue:

- on overlapping windows, `GLDM -> GLD` looks close
- on overlapping windows, `SGOV -> BIL` looks reasonably close
- on overlapping windows, `CTA -> FMF` shows the largest quality gap

So part of the rebuilt-line underperformance may come from a stricter
validation architecture, but part may also come from a weaker `trend` sleeve
implementation.

This means we are not only testing a new evaluation system. We are also testing
a changed `trend` product implementation.

### 3. Lockbox Scarcity

We now finally have a meaningful untouched test window. That is good, but it is
also scarce.

Every additional test-window exposure reduces the value of the final product
decision. So the research process must now live under a hard budget:

- validation can generate hypotheses
- robustness can shrink the candidate set
- the lockbox can only be opened for a formal decision round

This makes research discipline more important than it was in the previous
all-sample-like workflow.

### 4. Search Bias On The Validation Window

The current validation-only search already shows nontrivial selection-bias
pressure.

The lead candidate survived robustness review, but the search-leader z-scores
and moderate parameter-stability results tell us that continued blind grid
search would quickly turn validation into an optimization target rather than a
filter.

So the system is now at a point where more validation search is likely to harm
future test reliability.

### 5. Dynamic-Model Risk

If we respond to nonstationarity by making the model itself adaptive, we create
a second challenge:

- the regime model can overfit
- the hyperparameter-switching rule can overfit
- the state variables themselves can leak or become unstable

So dynamic adaptation is not a free solution. It can easily move overfitting
one layer up instead of solving it.

## Is Nonstationarity The Main Contradiction?

Yes, mostly.

The user diagnosis is directionally correct and does capture the main
contradiction:

- the macro environment is temporally nonstationary
- the validation and test windows are materially different
- static hyperparameters chosen on one regime may not transfer to another

This is the central reason why validation winners cannot be interpreted as
future winners.

But it is not the only major contradiction.

A more complete statement is:

> The primary challenge is regime nonstationarity, but it is currently coupled
> with a second major problem: the rebuilt `trend` bucket may not be represented
> by an equally strong ETF implementation.

So if a candidate later fails on test, there are at least two plausible causes:

- the environment changed
- the representative ETF choice for one bucket was weaker than intended

## Should We Model Market Drift And Dynamically Adjust Hyperparameters?

Potentially yes, but only under tight constraints.

The basic idea is valid:

- if environments drift
- and if the role of buckets changes across environments
- then a static policy can be systematically mis-specified

So discussing explicit regime or drift modeling is reasonable and timely.

However, the dangerous version of this idea is:

- estimate many latent states
- switch many hyperparameters
- re-optimize aggressively across windows

That would likely consume the validation set and create a more complex form of
overfitting.

The safer version is:

- use a small number of interpretable state variables
- define a coarse regime map
- let the strategy adjust only a small number of top-level knobs
- prefer bucket-level risk-budget changes over full re-optimization

Examples of acceptable adaptive levers:

- `equity_total` risk budget band
- `duration` risk budget band
- `trend` sleeve floor or cap
- rebalance frequency or threshold band

Examples of dangerous adaptive levers:

- continuously re-optimizing all bucket targets
- using a high-dimensional regime classifier
- letting the state model itself drift without a frozen protocol

## Working Judgment

Current working judgment:

- Yes, temporal nonstationarity is the main contradiction.
- Yes, it is worth discussing explicit drift/regime modeling.
- No, we should not jump straight to unconstrained dynamic hyperparameter
  switching.
- The right next discussion is whether we can define a small, white-box,
  low-dimensional adaptation layer above the current bucket-risk-budget system.

## Immediate Implication

The rebuilt line should now be interpreted as follows:

- the validation process has done its job by shrinking the candidate set
- the next conceptual step is not more static grid search
- the next conceptual step is to decide whether to keep a static shortlist into
  the lockbox or to introduce a tightly constrained regime-aware overlay before
  any final evaluation round
