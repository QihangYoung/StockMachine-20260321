# Multi-Asset Strategy Memo

## Scope

This memo focuses only on strengthening the multi-asset allocation strategy
itself.

It explicitly excludes:

- stock alpha sleeves
- single-name equity selection
- broker or paper-trading operations
- generic productization work that does not change allocation quality

## Goal

Build a stronger multi-asset allocator that is:

- structurally diversified
- robust across inflation and growth regimes
- implementable with liquid ETFs
- explainable in white-box terms

## Recommended Strategy Upgrades

### 1. Separate strategic core from tactical tilts

The portfolio should have:

- a stable strategic core that reflects long-run risk preferences
- a tactical overlay that can lean risk-on or risk-off using simple rules

This is stronger than one undifferentiated fixed-weight portfolio because it
separates:

- long-run diversification decisions
- shorter-horizon opportunity capture

### 2. Move from capital weights to risk-budgeted buckets

Do not let dollar weights implicitly determine risk.

Instead:

- define buckets such as equity, duration, credit, inflation hedge, trend, and
  cash
- assign each bucket a target share of total portfolio risk
- translate those risk budgets into capital weights using realized volatility
  and cross-bucket correlation

This is the highest-priority upgrade because it prevents the portfolio from
quietly becoming an equity-dominated portfolio just because equities are more
volatile.

### 3. Split the current bond bucket into distinct roles

A single aggregate-bond ETF is too coarse for a serious multi-asset strategy.

At minimum, separate:

- cash or ultra-short duration
- duration hedge
- spread or credit carry

This matters because these sleeves behave differently in:

- growth scares
- inflation shocks
- credit stress
- policy easing cycles

### 4. Formalize crisis-diversifier sleeves

Treat managed futures and gold as explicit sleeves with clear jobs, not just
miscellaneous diversifiers.

Managed-futures or trend sleeves should be evaluated for:

- convexity during equity stress
- low correlation to equity beta
- persistence during inflation or macro trend shocks

Gold or inflation-defense sleeves should be evaluated for:

- inflation surprise resilience
- diversification when stocks and bonds struggle together

### 5. Add white-box tactical allocation rules on top of the core

The first tactical layer should stay simple and auditable.

Recommended rule families:

- absolute momentum
- relative strength across offensive assets
- defensive fallback when offensive assets fail filters
- realized-volatility scaling

This should be the first source of dynamic edge before any more complex regime
logic is attempted.

### 6. Replace calendar-only rebalancing with threshold-aware rebalancing

A stronger allocator should not rebalance just because the calendar says so.

Use:

- drift thresholds
- volatility-aware rebalance bands
- a maximum stale-time cap so the portfolio is still periodically refreshed

This usually improves the tradeoff between responsiveness and turnover.

### 7. Measure contribution by bucket, not only by total Sharpe

A stronger multi-asset strategy must know which sleeves are doing real work.

Evaluate each bucket by:

- marginal contribution to volatility
- marginal contribution to drawdown reduction
- correlation to the rest of the portfolio
- crisis-period contribution

This helps prevent false diversification, where a sleeve looks diversified by
name but not by realized behavior.

## Suggested First Build Order

1. Freeze buckets and the ETF mapping for each bucket.
2. Build a risk-budgeted strategic core.
3. Split bonds into cash, duration, and credit roles.
4. Add simple tactical overlays: trend, relative strength, defensive fallback.
5. Add threshold-aware rebalancing.
6. Report sleeve-level contribution and crisis-period behavior.

## Practical First Version

The first serious multi-asset version should aim for:

- bucket-based strategic risk budgets
- explicit crisis-diversifier sleeves
- tactical risk-on or risk-off tilts
- turnover-aware rebalancing

That is a much stronger direction than simply tuning one more fixed ETF weight
vector.
