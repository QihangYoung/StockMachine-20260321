# C 2.0 Risk-Budget Core Experiment Spec

Date: 2026-04-08

## Scope

This spec defines the next multi-asset research line after the current
`Benchmark C Policy`.

It is intentionally limited to the multi-asset core only.

It excludes:

- stock alpha sleeves
- single-name equity selection
- broker operations
- live deployment decisions

## Objective

Build and evaluate a stronger replacement candidate for the current fixed-weight
`C policy` by moving from:

- fixed ETF capital weights

to:

- bucket-based risk budgeting with explicit crisis-diversifier sleeves

## Why This Experiment Exists

The current `C policy` has a credible macro-diversification story, but the
review found two important gaps:

1. the evidence package is not yet strong enough to call the strategy robust on
   standalone multi-asset grounds
2. the current fixed-weight form leaves money on the table in how risk is
   distributed across sleeves

The next step should therefore not be "tune another fixed vector."

The next step should be:

- redesign the core around bucket-level risk structure

## Primary Research Question

Can a bucket-based risk-budgeted multi-asset core outperform the current
fixed-weight `C policy` on risk-adjusted terms while remaining more explainable
and more robust than a pure max-Sharpe weight search?

## Secondary Research Questions

1. Does an `ERC` or risk-budgeted core improve diversification quality versus
   the current fixed weights?
2. Is a split bond sleeve better than a single `AGG` sleeve?
3. How much incremental value comes from formalizing `CTA` and gold as explicit
   crisis-diversifier sleeves?
4. Does threshold-aware rebalancing improve the turnover-versus-responsiveness
   tradeoff?
5. Is a light tactical overlay helpful after the strategic core is rebuilt?

## Design Principles

- keep the strategy white-box
- define risk at the bucket level, not just the ETF level
- prefer simple, economically interpretable sleeves
- avoid full unconstrained mean-variance optimization as the mainline design
- treat robustness as a first-class output, not an afterthought

## Candidate Buckets

The first version should use these strategic buckets:

- `equity_us`
- `equity_ex_us`
- `duration`
- `credit`
- `inflation_hedge`
- `trend`
- `cash`

Bucket roles:

- `equity_us`: primary growth beta
- `equity_ex_us`: non-US equity diversification
- `duration`: growth scare and rate-cut hedge
- `credit`: spread carry and intermediate-risk income sleeve
- `inflation_hedge`: inflation and non-fiat shock sleeve
- `trend`: crisis-diversifier and macro-trend sleeve
- `cash`: drawdown control, dry powder, and rebalance anchor

## Candidate ETF Mapping

The first experiment should freeze one primary ETF per bucket:

- `equity_us` -> `SPY`
- `equity_ex_us` -> `VXUS`
- `duration` -> `IEF`
- `credit` -> `LQD`
- `inflation_hedge` -> `GLDM`
- `trend` -> `CTA`
- `cash` -> `SGOV`

Optional secondary ETFs for robustness or replacement tests:

- `equity_us`: `QQQ`, `IWM`
- `duration`: `TLT`
- `credit`: `HYG`
- `inflation_hedge`: `GLD`, `DBC`
- `trend`: `DBMF`, `KMLM`
- `cash`: `BIL`

## Core Allocation Families To Test

### Family A: Fixed-weight bucket baseline

Use a transparent manually chosen bucket mix.

Purpose:

- baseline for comparison
- sanity check against the current `C policy`

### Family B: ERC core

`ERC` means:

- `equal risk contribution`

Definition:

- choose weights so each strategic bucket contributes roughly the same share of
  total portfolio volatility

Purpose:

- test whether simple risk balancing is a better core than fixed capital
  weights

### Family C: Constrained risk-budget core

Definition:

- assign non-equal target risk shares to buckets

Example target structure:

- growth buckets together: `30%` to `40%`
- duration plus cash together: `20%` to `30%`
- inflation hedge plus trend together: `30%` to `40%`
- credit: `5%` to `15%`

Purpose:

- allow an economically intentional risk profile rather than mechanically equal
  risk

### Family D: Vol-control overlay on top of the core

Definition:

- keep the same strategic core but scale gross exposure down when realized
  volatility breaches a threshold

Purpose:

- test whether a simple risk-control overlay improves drawdown efficiency

## Rebalancing Policies To Test

### Policy 1: Calendar-only baseline

- rebalance every `5` sessions
- rebalance every `21` sessions

### Policy 2: Threshold-aware rebalance

Use:

- absolute drift threshold at bucket level
- optional volatility-aware wider bands in high-cost or high-noise sleeves
- maximum stale-time cap so no sleeve goes unreviewed for too long

Initial test grid:

- drift threshold: `2%`, `4%`, `6%`
- stale-time cap: `21` sessions, `42` sessions

## Tactical Overlay Layer

The first pass should keep tactical logic simple.

Allowed rule families:

- absolute momentum at bucket level
- relative strength among offensive buckets
- defensive fallback into duration, trend, inflation hedge, or cash
- volatility scaling

Not allowed in the first pass:

- opaque machine-learning allocation signals
- macro forecasting models
- unconstrained optimization driven by unstable expected-return estimates

## Data And History Rules

The research line should:

- preserve a frozen bucket-to-ETF mapping
- document when live ETF history starts for each sleeve
- explicitly separate actual ETF history from proxy-chain history
- run proxy replacement checks as a required robustness step

Priority rule:

- do not treat a result as strong unless it survives reduced reliance on proxy
  assumptions

## Experimental Matrix

The first matrix should include:

### Baselines

- current `Benchmark C Policy`
- equal-weight all buckets
- simple `60/40` style proxy
- current best constrained fixed-weight candidate

### Core families

- fixed-weight bucket baseline
- `ERC` core
- constrained risk-budget core

### Bond splits

- `AGG` single-bucket baseline
- `IEF + LQD + SGOV` split-bucket version

### Diversifier tests

- with and without `trend`
- with and without `inflation_hedge`
- `CTA` replacement checks
- `GLDM` replacement checks

### Rebalance tests

- 5-session calendar
- 21-session calendar
- threshold-aware variants

### Tactical tests

- strategic core only
- strategic core plus absolute-momentum tilt
- strategic core plus defensive fallback

## Primary Evaluation Metrics

- annualized return
- annualized volatility
- Sharpe
- max drawdown
- downside capture versus `SPY`
- worst rolling `3`-month drawdown
- turnover
- mean cost bps

## Required Structural Diagnostics

- bucket risk contribution
- rolling correlation between major sleeves
- drawdown decomposition by bucket
- calendar-year returns
- crisis-period contribution review
- proxy sensitivity comparison

## Promotion Criteria

The first successful `C 2.0` candidate should beat the current fixed-weight
`C policy` on most of the following:

- equal or better Sharpe
- equal or better max drawdown
- stronger bucket diversification
- stronger evidence under proxy replacement checks
- similar or better turnover realism

It should also satisfy the standalone multi-asset observation checklist.

## Default Implementation Order

1. Freeze buckets and primary ETF mapping.
2. Build bucket return series and correlation estimates.
3. Implement `ERC` core.
4. Implement constrained risk-budget core.
5. Add bond-split version.
6. Add threshold-aware rebalancing.
7. Add simple tactical overlays.
8. Run robustness and gate review.

## Deliverables

The first output bundle should contain:

- summary metrics table
- bucket risk contribution table
- parameter sensitivity report
- proxy replacement report
- gate-review memo
- recommendation on whether `C 2.0` should replace the current fixed-weight
  `C policy`

## What ERC Core Means

`ERC` stands for:

- `equal risk contribution`

An `ERC core` is a portfolio where the strategic buckets are weighted so each
bucket contributes approximately the same share of total portfolio volatility.

It does **not** mean equal capital weights.

Because volatile sleeves require less capital to contribute the same amount of
risk, an `ERC` portfolio often looks very different from a naive equal-weight
portfolio.

Why it is useful:

- it reduces accidental risk concentration
- it is more stable than expected-return-heavy optimization
- it often produces a cleaner strategic core for multi-asset portfolios

## Recommended First Direction

The recommended first direction is:

- implement `ERC` as the first rebuilt strategic core
- then extend it into a constrained risk-budget core once the bucket behaviors
  are verified

This keeps the research path disciplined:

- first prove that simple risk balancing helps
- then prove that economically intentional non-equal risk budgets help even
  more
