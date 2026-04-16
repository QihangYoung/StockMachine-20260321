# FMF Regime Detector vNext Spec (2026-04-09)

## Goal

Upgrade the current `FMF` regime layer from:

- single-benchmark
- hard-threshold
- low-information

to a more industry-aligned design:

- multi-signal
- probability-based
- low-action-width

This document is a **design spec**, not a promotion decision.

The lockbox test window remains closed. All development and calibration for this detector family must stay inside the validation framework.

## Why v1/v2 Are Not Enough

The first two overlay versions established three important facts:

1. The original detector was too weak.
2. A better detector can improve regime alignment materially.
3. Detector improvement alone still does not make the overlay beat the static lead candidate.

So the main issue is no longer just threshold tuning. The architecture itself is too thin.

Current detector characteristics:

- benchmark: `equity_us`
- inputs:
  - trailing return
  - drawdown
  - realized volatility
- output:
  - hard label
- action:
  - shift `equity_total <-> duration`

This is too narrow relative to the economic problem we are trying to solve.

## Design Principle

The next detector should follow a stricter separation:

- **state layer**: may become richer
- **action layer**: must remain narrow

In other words:

- add information to the detector
- do **not** add a large number of moving knobs to the strategy

The target architecture is:

- static risk-budget core
- thin regime overlay

not:

- fully dynamic optimizer
- or parameter-retraining engine

## vNext State Model

### State Objective

The detector should answer:

- is the environment supportive of growth risk?
- is inflation pressure rising?
- is market stress broadening?

It should not try to forecast exact turning points at daily precision.

### State Space

We should move from a hard `bull/correction/bear/rebound` label to a probability-style state representation.

Recommended first-pass state space:

- `risk_on_score`
- `defensive_score`
- optional later:
  - `inflation_pressure_score`
  - `growth_slowdown_score`

The first implementation can still collapse these scores into:

- `risk_on`
- `neutral`
- `defensive`

but the underlying engine should store continuous scores.

## Signal Stack

The detector should no longer rely only on `SPY`.

### Tier 1: Benchmark Price-State Signals

Keep, but demote from sole driver to one block among several.

- `equity_us` trailing return
- `equity_us` drawdown
- `equity_us` realized volatility

### Tier 2: Cross-Asset Relative Signals

These are the most important additions.

Recommended first-pass signals:

- `equity_us` vs `duration`
  - broad growth-risk preference proxy
- `equity_us` vs `inflation_hedge`
  - weak but useful inflation/risk mix signal
- `credit` vs `duration`
  - credit stress / spread-like proxy
- `trend` absolute momentum
  - trend sleeve participation in the regime

Suggested implementations using the rebuilt `FMF` universe:

- `equity_us / duration`: `SPY` vs `IEF`
- `credit / duration`: `LQD` vs `IEF`
- `inflation_hedge`: `GLD`
- `trend`: `FMF`

### Tier 3: Correlation / Stress Signals

These should help identify when nominal diversification is failing.

Recommended first pass:

- rolling `equity_us-duration` correlation
- rolling cross-bucket average volatility
- rolling cross-bucket dispersion / breadth

The key use is not to perfectly identify crisis onset, but to detect when the portfolio environment is becoming less forgiving.

### Tier 4: Optional Macro Proxies

This tier should be optional for vNext, not mandatory for first implementation.

Later candidates:

- breakeven inflation proxy
- curve slope proxy
- dollar proxy
- commodity basket proxy

These may improve interpretation, but they should not block the first upgraded detector.

## Scoring Logic

### Recommended Structure

Use a block-scoring model instead of a rule tree.

For each date:

1. normalize each signal to a bounded score
2. aggregate within blocks
3. aggregate blocks into top-level state scores
4. smooth the final scores
5. map to policy state only at rebalance time

### Example

Illustrative layout:

- `risk_on_score`
  - positive `SPY` trend
  - positive `SPY vs IEF`
  - positive `LQD vs IEF`
  - low realized stress

- `defensive_score`
  - negative `SPY` trend
  - deep `SPY` drawdown
  - high realized volatility
  - rising `SPY-IEF` correlation
  - weak `LQD vs IEF`

### Smoothing

To avoid regime flapping:

- smooth scores with short EWMA
- require score margin before switching
- allow asymmetry:
  - easier to move into `defensive`
  - harder to move back into `risk_on`

This is closer to how industry systems are typically stabilized.

## Action Layer

The action layer must stay intentionally narrow.

### Allowed for vNext

- `equity_total` shift band
- `duration` shift band
- optional later:
  - `trend` floor/cap adjustment

### Frozen for vNext

- bucket structure
- `equity_us / equity_ex_us` split
- `credit` tactical resizing
- covariance estimator parameters
- optimization objective

### Suggested Initial Action Map

Keep the same action width as overlay v1/v2 for the first upgraded detector:

- `risk_on`: `equity_total +4%`, `duration -4%`
- `neutral`: no change
- `defensive`: `equity_total -4%`, `duration +4%`

Only after detector quality is clearly better should we test:

- asymmetric action widths
- `trend` floor increase in defensive states

## Validation Protocol

### Calibration Scope

All detector design and calibration must remain inside:

- validation window only

The lockbox test window must not be used for:

- signal selection
- threshold tuning
- score weighting
- state/action design

### Internal Evaluation

The upgraded detector should be judged on two levels.

#### Level 1: Regime-label Quality

Reference-style evaluation metrics:

- daily match ratio
- targeted stress-window recall
- defensive precision
- defensive share realism
- transition smoothness

Important:

- this is diagnostic only
- not the final objective

#### Level 2: Strategy Utility

Validation-only overlay comparison versus static lead:

- annualized return
- annualized volatility
- Sharpe
- max drawdown
- turnover

Promotion criterion for the detector branch should be strict:

- improved state quality **and**
- at least non-worse risk-adjusted portfolio outcome

If state quality improves but Sharpe and drawdown do not, the detector should remain research-only.

## Recommended Build Order

### Step 1

Implement a feature builder for validation-only regime signals:

- price-state block
- cross-asset relative block
- stress block

### Step 2

Implement a simple score model:

- z-score or bounded percentile transforms
- weighted additive aggregation
- EWMA smoothing

### Step 3

Map scores into `risk_on / neutral / defensive`

- hysteresis
- asymmetric transition rules

### Step 4

Run validation-only comparison:

- static lead
- overlay v2
- overlay vNext

### Step 5

Only if vNext is clearly better, consider one second-stage extension:

- add `trend` floor/cap action

## Working Recommendation

The next detector iteration should **not** be:

- more threshold tuning on the current single-benchmark design

It **should** be:

- multi-signal
- score-based
- smoothed
- low-action-width

This is the narrowest upgrade path that is still meaningfully closer to industry practice.
