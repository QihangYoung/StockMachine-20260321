# Phase7E Momentum / Trend Sleeve Memo

Date: 2026-05-17

Scope: validation-only. The test lockbox is not used.

## Purpose

Phase7E answers a narrow question:

```text
Is momentum/trend just -reversal?
```

The answer is no. `-reversal_5d` is only 5-session short-term continuation.
Intermediate momentum must use a different horizon, preferably excluding the
most recent reversal window.

## Definitions

Current code defines:

```text
return_5d = past 5-session return
reversal_5d = -return_5d
momentum_20d = past 20-session return
momentum_60d = past 60-session return
```

Phase7E adds:

```text
momentum_20d_ex_recent5 = (1 + momentum_20d) / (1 + return_5d) - 1
momentum_60d_ex_recent5 = (1 + momentum_60d) / (1 + return_5d) - 1
```

Each style sleeve is a daily top-minus-bottom 20% long-short basket on
`forward_beta_residual_return_5d`.

## Command

```text
$env:PYTHONPATH='src'; python -m stockmachine.apps.run_pure_alpha_phase7e
```

Default output:

```text
artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7e_momentum_trend_sleeves_20260517
```

## Main Result

Validation window:

```text
2014-11-11 through 2019-12-31
```

Panel size:

- joined stock-date rows: 915,312;
- style payoff rows: 10,256 in the eval window;
- styles tested: 8.

Style payoff summary:

| Style | Mean payoff | Hit rate | t-stat | Payoff corr vs reversal |
|---|---:|---:|---:|---:|
| `reversal_5d_loser_minus_winner` | 23.49 bps | 54.99% | 4.27 | 1.00 |
| `short_term_continuation_5d` | -23.49 bps | 45.01% | -4.27 | -1.00 |
| `momentum_20d_winner_minus_loser` | -29.21 bps | 45.16% | -4.85 | -0.53 |
| `momentum_60d_winner_minus_loser` | 1.97 bps | 53.67% | 0.31 | -0.30 |
| `momentum_20d_ex_recent5` | -21.03 bps | 48.44% | -3.63 | -0.15 |
| `momentum_60d_ex_recent5` | 6.08 bps | 52.96% | 0.95 | -0.09 |

Key score correlations:

| Score pair | Mean daily Spearman |
|---|---:|
| `reversal_5d` vs `return_5d` | -1.00 |
| `reversal_5d` vs `momentum_20d` | -0.44 |
| `reversal_5d` vs `momentum_60d` | -0.25 |
| `reversal_5d` vs `momentum_20d_ex_recent5` | 0.02 |
| `reversal_5d` vs `momentum_60d_ex_recent5` | 0.01 |

## Interpretation

`return_5d` is exactly `-reversal_5d`, so it provides no independent risk
source. It is the same short-horizon signal with the sign flipped.

Raw 20-session momentum is still heavily contaminated by the recent 5-session
move:

```text
payoff correlation vs reversal = -0.53
mean payoff = -29.21 bps
```

This behaves less like a useful momentum sleeve and more like anti-reversal in
this target/window.

The cleaner independent candidate is:

```text
momentum_60d_ex_recent5
```

It has near-zero payoff correlation with reversal:

```text
payoff corr vs reversal = -0.09
rolling corr vs reversal = -0.05
```

But its standalone payoff is weak:

```text
mean payoff = 6.08 bps
t-stat = 0.95
```

So it is not yet a strong alpha sleeve. It is a possible diversifier, not a
replacement for reversal.

## Static Combo Diagnostic

Validation-only static combinations:

| Combo | Mean payoff | Hit rate | Vol bps | t-stat |
|---|---:|---:|---:|---:|
| `reversal_only` | 23.49 bps | 54.99% | 196.80 | 4.27 |
| `75% reversal + 25% momentum_60d_ex_recent5` | 19.14 bps | 56.55% | 153.53 | 4.46 |
| `50% reversal + 50% momentum_60d_ex_recent5` | 14.78 bps | 53.28% | 144.16 | 3.67 |

The 75/25 combo gives up mean payoff but reduces volatility enough to improve
the diagnostic t-stat. That suggests `momentum_60d_ex_recent5` may be useful as
a small risk diversifier, but not as a primary sleeve.

## Next Step

The next version should test walk-forward style allocation rather than static
weights:

```text
reversal weight high when reliable_core_score is strong
momentum_60d_ex_recent5 weight small but persistent if rolling payoff remains positive
raw 20d momentum excluded unless it is explicitly converted into anti-momentum
```

Promotion criterion:

```text
The combined sleeve must improve calendar t-stat / drawdown after costs,
without relying on full-window optimized weights.
```
