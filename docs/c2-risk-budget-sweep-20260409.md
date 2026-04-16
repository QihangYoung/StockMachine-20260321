# C2 Risk-Budget Sweep 2026-04-09

## Purpose

This memo records the first disciplined `C2` regional sweep after adding an
apples-to-apples `current C` periodic-rebalance baseline.

The sweep follows the rule in
`post-hoc-hypothesis-discipline-memo.md`:

- treat `duration down / crisis diversifier up` as a hypothesis region
- evaluate a region rather than a single point
- compare against both `buy-and-drift current C` and
  `periodic-rebalance current C`

## Sweep Region

The sweep fixed:

- total equity risk budget at `35%`
- internal equity split at `24% / 11%`
- credit at `10%`
- cash reserve at `5%`

And tested:

- `duration`: `15%`, `20%`, `25%`
- `inflation_hedge`: `10%`, `15%`, `20%`
- `trend`: residual budget after the above choices

The center point `20 / 15 / 20` was already covered by `C2 v0`, so the sweep
tested the surrounding region only.

## Main Results

Artifact root:

- `artifacts/c2_risk_budget_sweep_20260409`

Common window:

- `2019-01-04` to `2026-04-08`

Live window:

- `2022-03-09` to `2026-04-08`

Best common-window `C2` candidate:

- `rolling_c2_region_d15_i20_t20`
- annualized return `8.06%`
- annualized volatility `5.88%`
- Sharpe `1.348`
- max drawdown `-10.51%`

Reference rows on the same common window:

- `current_c_policy_core_periodic_rebalance`: `11.57% / 7.44% / 1.510 / -11.48%`
- `current_c_policy_core`: `11.90% / 8.18% / 1.417 / -11.06%`
- `rolling_c2_v0_core`: `7.46% / 5.64% / 1.306 / -9.76%`
- `rolling_erc_core`: `7.54% / 6.19% / 1.206 / -12.04%`

Best live-window `C2` candidate:

- `rolling_c2_region_d15_i20_t20`
- annualized return `8.16%`
- annualized volatility `6.09%`
- Sharpe `1.318`
- max drawdown `-5.82%`

Reference rows on the same live window:

- `current_c_policy_core_periodic_rebalance`: `11.59% / 7.74% / 1.456 / -6.71%`
- `rolling_c2_v0_core`: `7.43% / 5.87% / 1.249 / -5.88%`

## What This Sweep Suggests

- Lowering `duration` from `20%` toward `15%` improved the `C2` family versus
  `v0` in this sample.
- Within the tested region, higher `inflation_hedge` helped more than pushing
  residual budget further into `trend`; the best point was `duration 15 /
  inflation_hedge 20 / trend 20`.
- The `C2` family still has a clear lower-volatility character, but none of the
  tested candidates beat `current C` periodic rebalance on Sharpe.
- The new periodic-rebalance `current C` baseline matters: after removing the
  passive-drift objection, `current C` still remained the strongest Sharpe
  baseline in this sample.

## What We Are Not Allowed To Conclude

This sweep does **not** prove:

- that the final `C2` policy should use `duration 15 / inflation_hedge 20 /
  trend 20`
- that `current C` is the final winning standalone multi-asset policy
- that the next official strategy should be promoted from this same sample

These are still sample-generated hypotheses, not validated final truths.

## Recommended Next Step

The next `C2` pass should:

- keep `duration` focused at `15%` to `20%`
- test whether reducing `credit` below `10%` helps close the return gap
- test whether total equity risk budget above `35%` can improve return while
  preserving drawdown control
- continue reporting proxy-era and live-era windows separately
