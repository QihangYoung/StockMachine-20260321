# C2 Implementation Sweep 2026-04-09

## Purpose

This memo records the first implementation-layer sweep around the current lead
`C2` candidate:

- `equity_total 35%`
- `credit 5%`
- `duration 15%`
- `inflation_hedge 20%`
- `trend 25%`

The goal of this pass was to test:

- `cash reserve`: `0%` vs `5%`
- rebalance style:
  - calendar-only
  - threshold-aware

## Artifact Root

- `artifacts/c2_implementation_sweep_20260409`

## Test Matrix

Cash reserve:

- `0%`
- `5%`

Threshold grid:

- drift threshold: `2%`, `4%`, `6%`
- stale-time cap: `21`, `42` sessions

Reference baselines kept in the same run:

- `current_c_policy_core`
- `current_c_policy_core_periodic_rebalance`
- `rolling_c2_v0_core`

## Main Results

Common window:

- `2019-01-04` to `2026-04-08`

Best implementation candidate on the common window:

- `c2_impl_lead_cash05_calendar`
- annualized return `8.32%`
- annualized volatility `5.79%`
- Sharpe `1.409`
- max drawdown `-9.39%`

Best implementation candidate on the live window:

- `c2_impl_lead_cash05_calendar`
- annualized return `8.36%`
- annualized volatility `6.11%`
- Sharpe `1.345`
- max drawdown `-5.50%`

Reference baseline:

- `current_c_policy_core_periodic_rebalance`
- annualized return `11.57%`
- annualized volatility `7.44%`
- Sharpe `1.510`
- max drawdown `-11.48%`

## What This Sweep Suggests

- `cash reserve 5%` remained better than `cash reserve 0%` for the lead `C2`
  candidate.
- Threshold-aware rebalance did **not** improve on the calendar-only monthly
  baseline in this sample.
- The best threshold variants stayed close, but they still trailed the simple
  calendar baseline.
- Higher stale caps reduced turnover more aggressively, but usually at the cost
  of lower Sharpe and slightly weaker drawdown control.
- `stale_time_cap = 21` with a high drift threshold (`6%`) was effectively very
  close to the plain monthly calendar policy, which is what the rebalance count
  shows.

## Practical Read

At the current stage, the simplest implementation remains the strongest one:

- keep `cash reserve = 5%`
- keep calendar-only monthly rebalance
- do not add threshold-aware rebalance as the default implementation path

## What We Are Not Allowed To Conclude

This sweep does **not** prove:

- that threshold-aware rebalancing is never useful
- that `cash reserve = 5%` is permanently optimal in every future window

It only says that, for the current lead `C2` candidate and this sample, the
simple monthly calendar implementation still wins.

## Recommended Next Step

The next step should move away from rebalance mechanics and back toward
robustness:

- keep `c2_impl_lead_cash05_calendar` as the working `C2` implementation
  candidate
- compare it against `current_c_policy_core_periodic_rebalance` in live-era
  monitoring
- if `C2` remains structurally weaker on return, test one last narrow policy
  pass inside the already-identified strong region rather than adding more
  implementation complexity
