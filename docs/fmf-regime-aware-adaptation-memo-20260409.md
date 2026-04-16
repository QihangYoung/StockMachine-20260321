# FMF Regime-Aware Adaptation Memo

Date: 2026-04-09

## Purpose

This memo proposes a constrained way to discuss regime-aware or drift-aware
adaptation for the rebuilt `FMF` multi-asset line.

The goal is not to create a high-dimensional meta-optimizer. The goal is to
identify a very small set of top-level levers that could reasonably adapt when
macro conditions change materially.

## Starting Point

Current rebuilt setup:

- universe: `SPY / VXUS / IEF / LQD / GLD / FMF / BIL`
- validation: `2013-08-01 ~ 2019-12-31`
- lockbox test: `2020-01-02 ~ 2026-04-08`
- current lead region:
  - equity risk roughly `40% ~ 42%`
  - credit `8% ~ 10%`
  - duration around `22%`
  - inflation hedge `18% ~ 20%`
  - trend `6% ~ 10%`

The motivation for adaptation is clear:

- validation and test live in materially different macro environments
- the relative attractiveness of bucket exposures can drift through time
- a single static parameter point may be structurally fragile

## What The Dynamic Layer Should And Should Not Do

### What it should do

- respond to large, interpretable environment shifts
- move only a few top-level knobs
- preserve the bucket-based architecture
- remain auditable and explainable

### What it should not do

- continuously re-optimize the full portfolio
- switch many parameters at once
- rely on a high-dimensional latent-state classifier
- turn validation into a second meta-search problem

## Parameters That May Be Allowed To Move

These are the only parameters that currently look reasonable to place under a
future regime-aware layer.

### 1. Equity-total risk budget band

This is the cleanest top-level adaptation lever.

Reason:

- it expresses overall risk-on versus risk-off posture
- it is economically interpretable
- it does not require changing the whole bucket system

Suggested treatment:

- static core target remains the default
- allow only coarse bands, for example:
  - low-risk band
  - neutral band
  - high-risk band

The adaptation should move between bands, not continuously search for a new
equity optimum.

### 2. Duration risk budget band

This is the second cleanest lever.

Reason:

- duration behavior changes sharply across inflation and tightening regimes
- this was one of the most obvious differences between pre-2020 and post-2020
  environments

Suggested treatment:

- keep a default duration target
- permit only a small upward or downward adjustment band

This should be thought of as a macro hedge lever, not a return-chasing lever.

### 3. Trend sleeve floor and cap

This is a candidate lever, but only after the `trend` representative question is
handled more carefully.

Reason:

- trend is the most regime-sensitive sleeve in the current rebuild
- but it is also the sleeve with the largest implementation-quality uncertainty

Suggested treatment:

- do not dynamically optimize trend weight
- only allow a floor/cap framework once the bucket representative is better
  understood

So trend adaptation is a second-stage topic, not the first adaptation topic.

### 4. Rebalance rule band

This is allowed in principle, but only coarsely.

Reason:

- some environments may reward calendar-only rebalancing
- others may justify wider thresholds or lower turnover

Suggested treatment:

- allow only a small menu:
  - monthly calendar
  - monthly calendar plus light threshold
- do not let rebalance cadence become a free search dimension

## Parameters That Should Stay Frozen

These should remain static unless a much stronger case appears later.

### 1. Full bucket structure

Do not dynamically change what the buckets are.

The current bucket architecture is already the main explanatory frame. Changing
bucket definitions by regime would make attribution too unstable.

### 2. Equity split between US and ex-US

This is too secondary and too easy to overfit.

The current problem is not that the system needs dynamic fine control over
`SPY` versus `VXUS`. The more important issue is total equity risk.

### 3. Credit as a separate fast-moving tactical dial

Credit can remain in the static allocation for now, but it should not become a
high-frequency macro switch.

If credit starts moving dynamically, the model will quickly become harder to
interpret and more sensitive to validation noise.

### 4. Covariance hyperparameters

Do not make the covariance windows and blending weights dynamic in the first
regime-aware version.

Those are implementation details. Making them state-dependent would be a hidden
meta-optimization layer.

## Preferred Adaptation Architecture

If we decide to build a regime-aware layer, the safest first version is:

1. keep a static bucket-risk-budget core
2. define a small number of coarse macro states
3. map each state to a small adjustment on:
   - equity-total band
   - duration band
   - optionally trend floor/cap later
4. keep all other components frozen

This makes the adaptation layer:

- low-dimensional
- explainable
- easier to validate
- less likely to consume the lockbox by over-searching

## What Would Count As A Dangerous Version

The following should be treated as anti-patterns:

- dynamic optimization of all bucket weights
- regime-specific covariance hyperparameters
- regime-specific ETF representatives
- machine-learned hidden-state switching without a frozen interpretability layer
- using the lockbox to tune the state model

## Working Judgment

Current working judgment:

- yes, regime-aware adaptation is worth exploring
- yes, it should directly target nonstationarity
- no, it should not begin as a full dynamic hyperparameter framework
- the right first design is a narrow overlay on top of the current risk-budget
  architecture

## Immediate Next Question

Before implementing any regime-aware version, the next design question should
be:

- what minimal set of state variables can justify switching the
  `equity_total` and `duration` bands without turning the system into a new
  overfit search problem?
