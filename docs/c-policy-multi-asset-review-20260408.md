# C Policy Multi-Asset Review

Date: 2026-04-08

## Scope

This review evaluates only the multi-asset core currently frozen as:

- `Benchmark C Policy`
- weights: `SPY 10%`, `VXUS 10%`, `AGG 25%`, `CTA 35%`, `GLDM 20%`

It does **not** evaluate the stock-alpha sleeve.

## Decision

Current decision:

- `Reject or redesign`

This is a serious and promising multi-asset strategy, but it is **not yet**
strong enough to pass the stricter observation gate as a standalone multi-asset
allocation strategy.

## Gate Review

### Gate 1: Economic Logic

Result:

- `PASS`

Why:

- the bucket roles are economically coherent
- equity, bond, managed-futures, and gold sleeves each have a clear job
- the strategy can be explained in macro portfolio terms rather than as a pure
  backtest artifact

### Gate 2: Cross-Regime And Holdout Robustness

Result:

- `FAIL`

Why:

- the strategy was tested across a broad `2018-2026` multi-regime window
- however, the current evidence package does not show a clean holdout period
  that was fully excluded from weight selection
- regime coverage is good, but it is not enough by itself to establish
  robustness

### Gate 3: Parameter And Structure Robustness

Result:

- `PASS`

Why:

- the chosen `10/10/25/35/20` mix sits inside a believable local neighborhood
- nearby variants remain competitive instead of collapsing immediately
- the selected configuration does not look like an isolated optimization spike

### Gate 4: Cost And Implementation Robustness

Result:

- `FAIL`

Why:

- turnover is low and the instruments are liquid
- however, the long-window evidence still relies materially on proxy-chain
  history, especially for `CTA`
- the current package does not yet provide a convincing proxy replacement check
  for this exact strategy configuration

### Gate 5: Risk Structure Robustness

Result:

- `PASS`

Why:

- the implied risk distribution is diversified
- no unintended bucket appears to dominate total portfolio risk
- crisis diversifiers appear to be doing real work rather than only being
  cosmetic holdings

Approximate annualized risk-share estimate using current single-asset
volatilities and correlations:

- `SPY`: about `16%`
- `VXUS`: about `17%`
- `AGG`: about `9%`
- `CTA`: about `29%`
- `GLDM`: about `28%`

This is materially more balanced than the older `45/15/20/10/10` style
allocation, which was still dominated by equity risk.

## Overall Interpretation

The current `C policy` has three attractive properties:

1. It has a credible macro-diversification story.
2. Its local weight neighborhood looks stable enough to be believable.
3. Its realized risk structure appears intentionally diversified rather than
   accidentally concentrated.

The two main reasons it still fails the stronger gate are:

1. lack of a clean post-selection holdout proof
2. continued dependence on proxy-chain history for core sleeves

## Next Required Work

Before this strategy should be treated as observation-ready on multi-asset
merit alone, the next required work is:

1. Freeze the current `C` weights and test them on a clean holdout period not
   used in the selection process.
2. Run explicit proxy replacement checks for the `CTA` and `GLDM` sleeves and
   document whether the main conclusion survives.
3. Add formal risk-contribution and stress-period attribution to the standard
   review pack.

## Bottom Line

`C policy` should be treated as:

- a strong multi-asset research idea
- not yet observation-ready under the stricter standalone multi-asset gate
- in need of a redesigned evidence package before promotion
