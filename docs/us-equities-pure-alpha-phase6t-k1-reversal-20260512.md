# Phase6T K=1 Reversal-Only Dynamic Style Test

Date: 2026-05-12

Scope: validation-only. Test lockbox remains closed.

## Setup

This run keeps the corrected Phase6T framework but reduces the style dimension to \(K=1\):

\[
s_{i,t} = b_{i,t}^{reversal} \tilde f_t^{reversal}
\]

Only `reversal_5d` is used. The universe remains symmetric:

- long universe: `adv30m_clean_core_beta_full`;
- short universe: `adv30m_clean_core_beta_full`;
- long buys high-score names;
- short sells low-score names;
- corrected validation window: 2014-11-11 to 2019-12-31.

Artifacts:

- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6t_dynamic_price_style_allocation_k1_reversal_20260512/phase6t_curve_metrics.csv`
- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6t_dynamic_price_style_allocation_k1_reversal_20260512/phase6t_dynamic_style_allocation_plot.png`

## Results

| Portfolio | Final Equity | Ann. Return | Ann. Vol | Sharpe | Max DD | Rolling 60 Positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sota_size_hard_neutral` | 1.996 | 14.42% | 7.51% | 1.83 | -7.78% | 85.15% |
| `static_mean` | 1.265 | 4.69% | 7.13% | 0.68 | -11.92% | 67.93% |
| `ewma_hl120` | 1.144 | 2.66% | 7.13% | 0.40 | -17.88% | 61.50% |
| `ewma_hl20` | 1.168 | 3.07% | 8.88% | 0.38 | -16.34% | 57.63% |
| `kalman_phi099_q0004_shrink025` | 0.971 | -0.59% | 6.97% | -0.05 | -14.69% | 48.05% |
| `ewma_hl60` | 0.897 | -2.11% | 8.91% | -0.19 | -26.95% | 54.38% |
| `kalman_phi099_q0004` | 0.806 | -4.12% | 8.79% | -0.43 | -22.00% | 46.55% |

## Interpretation

K=1 does not rescue the dynamic style allocation route. The best K=1 variant is `static_mean`, not an adaptive estimator. That is important: with only the core reversal prior, the most robust choice is effectively to not time the payoff aggressively.

Compared with corrected K=4:

- K=4 `static_mean`: 8.03% annualized return, Sharpe 0.93, max DD -21.67%.
- K=1 `static_mean`: 4.69% annualized return, Sharpe 0.68, max DD -11.92%.

So K=1 reduces drawdown but gives up too much return. It is cleaner and more interpretable, but not competitive with SOTA.

The result supports the earlier concern: many adaptive variants are still members of the same reversal family. Reducing \(K\) to one makes the family structure explicit, but it does not solve the regime problem. The next useful step is not more half-life tuning; it is either a principled prior-preserving Bayesian overlay on SOTA, or a genuinely different source of evidence about when the reversal mechanism is active.
