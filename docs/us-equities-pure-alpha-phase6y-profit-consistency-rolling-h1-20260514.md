# Phase6Y Profit-Consistency Rolling H1 Ranker

Date: 2026-05-14

Scope: validation-only walk-forward. The test lockbox is not used.

## Question

Phase6X selected model complexity with pairwise logloss early stopping. Phase6Y
tests whether a profit-path selection rule is better:

- Keep the same 12m train / 3m validation / 1m prediction walk-forward setup.
- Keep the same h1 enhanced feature panel and pairwise LightGBM model family.
- Evaluate candidate iterations on train and 3m validation day1 top-bottom
  residual spread.
- Penalize left-tail daily spread and consecutive loss streaks.
- Penalize train/validation mean-spread overfit gaps.
- Final report remains prediction-month day1 only.

## Utility Definition

For each candidate iteration, Phase6Y computes daily Top20-Bottom20 h1 strict
residual spread on a sampled utility panel.

Utility is:

```text
mean_spread_bps
- 0.20 * max(0, -p10_spread_bps)
- 0.05 * worst_consecutive_loss_streak_bps
- positive_rate_shortfall_penalty
```

Then the final objective combines train and validation:

```text
0.70 * validation_utility
+ 0.30 * min(train_utility, validation_utility)
- 0.10 * max(0, train_mean_spread_bps - validation_mean_spread_bps)
```

The intent is to avoid selecting models that look good only because they have a
few large right-tail days or a strong train-only mean.

## Overall Result

| model | sessions | mean Rank IC | day1 spread | spread t-stat | positive rate | p10 spread | median spread | p90 spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase6X logloss early stop | 492 | 0.0092 | 2.93 bps/day | 3.03 | 55.7% | -23.31 | 3.93 | 27.69 |
| Phase6Y profit consistency | 492 | 0.0074 | 1.94 bps/day | 1.93 | 54.7% | -24.52 | 1.70 | 28.49 |

Phase6Y underperforms Phase6X. The penalty does not improve the left tail, and
it lowers average day1 spread.

## Selected Iterations

Phase6Y selected a wide range of model depths:

| selected iteration | count |
| ---: | ---: |
| 1 | 6 |
| 2 | 1 |
| 8 | 2 |
| 16 | 1 |
| 32 | 3 |
| 64 | 2 |
| 96 | 4 |
| 128 | 2 |
| 192 | 3 |

This shows the selection rule is active, not trivially choosing the same depth.
But the selected depth does not translate into better next-month performance.

## Interpretation

This version is a useful negative result.

The train/validation profit-path utility is intuitive, but the 3-month
validation path is itself noisy. Penalizing left tail and loss streaks inside
that short window can select models that were safer in the recent past without
being more predictive next month.

The result suggests the issue is not just LightGBM logloss overfitting. The
short-horizon payoff itself is unstable enough that a hand-built path utility
can overfit the validation slice too.

## Practical Takeaway

Do not replace Phase6X with Phase6Y as currently specified.

Phase6X remains the better rolling h1 candidate:

- Higher mean day1 spread.
- Higher Rank IC.
- Better spread t-stat.
- Similar or better left-tail behavior.

If we revisit profit-consistency selection, it should be simpler and less
parameterized, for example:

- require validation utility to be above a hard floor, otherwise choose a very
  shallow/default model;
- use only validation positive-rate and p10 gates, not a weighted utility;
- aggregate utility across multiple rolling validation slices instead of one
  3-month slice;
- or move path consistency into portfolio-level construction rather than model
  iteration selection.

## Artifacts

- Output directory: `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6y_profit_consistency_rolling_h1_20260514`
- Daily metrics: `phase6y_day1_daily_metrics.csv`
- Monthly summary: `phase6y_day1_monthly_summary.csv`
- Candidate utility: `phase6y_candidate_utility.csv`
- Model diagnostics: `phase6y_model_diagnostics.csv`
- Score panel: `phase6y_score_panel.csv.gz`
