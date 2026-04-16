# C2 Risk-Budget Frontier Sweep 2026-04-09

## Purpose

This memo records the second disciplined `C2` sweep after the first regional
pass identified a more promising zone around:

- lower `duration`
- higher `inflation_hedge`
- unchanged or moderately higher `trend`

The goal of this pass was to test two next-step questions:

- does `credit < 10%` help the `C2` family
- does `equity total > 35%` help the `C2` family

## Sweep Region

Artifact root:

- `artifacts/c2_risk_budget_sweep_frontier_20260409`

The frontier sweep fixed:

- cash reserve at `5%`
- internal equity split proportional to the original `24 / 11`

And tested:

- equity total: `35%`, `40%`, `45%`
- credit: `5%`, `10%`
- duration: `15%`, `20%`
- inflation hedge: `15%`, `20%`
- trend: residual budget

## Main Results

Common window:

- `2019-01-04` to `2026-04-08`

Best `C2` candidate on the common window:

- `rolling_c2_region_e35_c05_d15_i20_t25`
- annualized return `8.32%`
- annualized volatility `5.79%`
- Sharpe `1.409`
- max drawdown `-9.39%`

Best `C2` candidate on the live window:

- `rolling_c2_region_e35_c05_d15_i20_t25`
- annualized return `8.36%`
- annualized volatility `6.11%`
- Sharpe `1.345`
- max drawdown `-5.50%`

Reference baseline on the common window:

- `current_c_policy_core_periodic_rebalance`
- annualized return `11.57%`
- annualized volatility `7.44%`
- Sharpe `1.510`
- max drawdown `-11.48%`

## What This Sweep Suggests

- The strongest improvement over `C2 v0` came from reducing `credit` from
  `10%` to `5%`.
- Increasing `equity total` above `35%` did not produce the best candidate in
  this sweep.
- The top of the tested frontier stayed clustered around:
  - `equity total = 35%`
  - `credit = 5%`
  - `duration = 15%`
  - `inflation_hedge = 15%` to `20%`
  - `trend = 25%` to `30%`
- `C2` still behaves like a lower-volatility, shallower-drawdown family, but
  it still does not beat the periodic-rebalance `current C` baseline on Sharpe.

## What We Are Not Allowed To Conclude

This sweep does **not** prove:

- that the final `C2` policy is now known
- that `credit = 5%` is permanently correct
- that `equity total` should never be above `35%`

These are still sample-generated research signals.

## Recommended Next Step

The next pass should stay focused on the currently strongest region:

- `equity total`: `35%` to `40%`
- `credit`: `5%`
- `duration`: `15%`
- `inflation_hedge`: `15%` to `20%`
- `trend`: residual

The next useful dimension to test is no longer broad capital mix. It should be:

- rebalance policy:
  - calendar-only vs threshold-aware
- reserve cash:
  - `0%` vs `5%`
- live-era only robustness:
  - keep the same candidate family but evaluate only the actual-CTA era
