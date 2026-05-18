# Phase7F Momentum Horizon Grid Memo

Date: 2026-05-17

Scope: validation-only. The test lockbox is not used.

## Purpose

Phase7F answers:

```text
When does cross-sectional momentum pay?
```

Unlike Phase7E, this experiment rebuilds forward labels directly from adjusted
prices, so the holding horizon is explicit:

```text
h5, h10, h20, h60
```

Momentum formation is also explicit:

```text
lookback = 20, 60, 120, 252 sessions
skip     = 0, 5, 20 sessions
```

`lookback=60, skip=20` means a 60-session return ending 20 sessions before the
signal date.

## Command

```text
$env:PYTHONPATH='src'; python -m stockmachine.apps.run_pure_alpha_phase7f
```

Default output:

```text
artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7f_momentum_horizon_grid_20260517
```

## Setup

Validation window:

```text
2014-11-11 through 2019-12-31
```

Panel size:

- membership rows: 923,359;
- feature/target panel rows: 923,359;
- evaluation payoff rows: 71,022;
- styles tested: 56.

The payoff target is:

```text
forward_beta_residual_return_hN
= forward_return_hN - beta * benchmark_forward_return_hN
```

## Horizon Summary

Best momentum row by holding horizon:

| Holding | Best style | Mean payoff | Naive t-stat | Newey-West t-stat |
|---:|---|---:|---:|---:|
| h5 | `momentum_l60_s20_h5` | 9.82 bps | 2.10 | 1.14 |
| h10 | `momentum_l60_s20_h10` | 20.24 bps | 3.09 | 1.26 |
| h20 | `momentum_l120_s20_h20` | 37.71 bps | 3.94 | 1.20 |
| h60 | `momentum_l60_s20_h60` | 86.68 bps | 6.18 | 1.26 |

The economic payoff increases with holding horizon. The Newey-West t-stat is
much lower because forward labels overlap, especially for h20/h60.

## Main Interpretation

Momentum seems to pay slowly.

The strongest and cleanest family is:

```text
60-120 session formation
20-session skip
h10 to h60 holding
```

The skip matters because it removes the short-term reversal window. This avoids
confusing medium-term momentum with the exact opposite of 5-day reversal.

Examples:

| Style | Mean payoff | Hit rate | Corr vs same-horizon reversal |
|---|---:|---:|---:|
| `momentum_l60_s20_h10` | 20.24 bps | 55.15% | -0.11 |
| `momentum_l120_s20_h20` | 37.71 bps | 55.90% | -0.20 |
| `momentum_l60_s20_h60` | 86.68 bps | 62.99% | -0.00 |
| `momentum_l120_s0_h60` | 87.98 bps | 58.44% | -0.20 |

This says momentum is not `-reversal`. Properly skipped momentum has low
correlation to reversal and a longer payoff horizon.

## Baseline Contrast

Short-horizon reversal:

| Style | Mean payoff | Hit rate | Newey-West t-stat |
|---|---:|---:|---:|
| `reversal_5d_h5` | 10.09 bps | 52.99% | 1.54 |
| `reversal_5d_h10` | 23.49 bps | 54.99% | 2.27 |
| `reversal_5d_h20` | 34.64 bps | 55.42% | 1.98 |
| `reversal_5d_h60` | 17.17 bps | 50.32% | 0.57 |

Reversal is strongest around h10/h20 and fades by h60. Momentum is weak at h5
but becomes economically large at h20/h60.

## Non-Overlap Sanity Check

To avoid trusting overlapping labels too much, selected sleeves were also
checked by sampling every holding horizon across all offsets.

| Style | Holding | Offset positive rate | Median offset mean |
|---|---:|---:|---:|
| `momentum_l60_s20_h10` | h10 | 100.00% | 20.14 bps |
| `momentum_l120_s20_h20` | h20 | 100.00% | 39.64 bps |
| `momentum_l60_s20_h60` | h60 | 91.67% | 94.01 bps |
| `momentum_l120_s0_h60` | h60 | 93.33% | 97.75 bps |

The sign is broadly stable across offsets, though h60 has much fewer
independent observations per offset.

## Conclusion

The current evidence says:

```text
reversal: best around h10/h20
momentum: starts showing up around h10, is more economically meaningful at h20/h60
```

For the alpha line, this means momentum should not be tested as a direct h10
replacement for reversal only. It should be a separate, slower sleeve with its
own holding horizon and turnover assumptions.

Recommended next experiment:

```text
reversal sleeve: h10/h20
momentum sleeve: 60-120 formation, 20 skip, h20/h60
allocation: walk-forward, not full-window optimized
```

Promotion criterion:

```text
Momentum must improve portfolio-level drawdown / t-stat after turnover and
overlapping-label adjustment, not merely show a large naive h60 mean.
```
