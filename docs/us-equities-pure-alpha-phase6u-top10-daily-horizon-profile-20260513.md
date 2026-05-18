# Phase6U Top10-Bottom10 Daily Horizon Profile

Date: 2026-05-13

Scope: validation-era diagnostic only. This memo does not use the test lockbox.

## Method

This is the same Phase6U diagnostic as the Top20-Bottom20 run, but buckets are
changed to daily Top 10% minus Bottom 10% by `pairwise_rank_score`.

Important caveat: the daily strict residual day1-10 sum is not identical to
directly residualizing one aggregate h10 return. It is a horizon-profile
diagnostic.

## Pairwise Score: Strict Residual Daily Spread

Mean Top 10% minus Bottom 10%, bps per holding day.

| split | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | day1-10 sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 5.92 | 5.91 | 7.44 | 6.65 | 7.21 | 6.29 | 6.74 | 5.75 | 4.11 | 4.37 | 60.38 |
| model_validation_2018 | 3.90 | -0.03 | 1.59 | 0.74 | -1.69 | 0.77 | 0.89 | 2.24 | 1.04 | 0.30 | 9.76 |
| final_validation_2019 | 3.27 | -0.46 | 0.60 | 1.02 | -1.83 | -0.32 | -0.23 | 0.42 | -1.13 | -1.12 | 0.22 |

## Pairwise Score: Other Target Views

Mean Top 10% minus Bottom 10%, bps per holding day.

### Final Validation 2019

| target | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | day1-10 sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw open-to-open | -0.53 | -6.29 | -4.49 | -3.14 | -3.98 | -6.39 | -2.45 | -6.84 | -5.01 | -5.40 | -44.52 |
| beta residual | 3.93 | -0.51 | 0.54 | 2.04 | 0.67 | -1.43 | 2.44 | -1.28 | 0.28 | -0.89 | 5.79 |
| strict residual | 3.27 | -0.46 | 0.60 | 1.02 | -1.83 | -0.32 | -0.23 | 0.42 | -1.13 | -1.12 | 0.22 |

### Model Validation 2018

| target | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | day1-10 sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw open-to-open | 2.99 | 0.66 | -0.67 | -1.63 | -3.55 | -3.27 | -2.43 | -0.07 | -1.76 | -4.74 | -14.47 |
| beta residual | 2.06 | -0.85 | -2.24 | -2.53 | -4.63 | -4.17 | -4.24 | -1.93 | -3.63 | -5.46 | -27.61 |
| strict residual | 3.90 | -0.03 | 1.59 | 0.74 | -1.69 | 0.77 | 0.89 | 2.24 | 1.04 | 0.30 | 9.76 |

## Interpretation

Top10-Bottom10 increases in-sample strength sharply: strict residual day1-10
sum rises from about 39.57 bps in the Top20 run to about 60.38 bps.

The same concentration does not generalize to 2019. Final validation strict
residual day1-10 sum falls from about 10.75 bps in the Top20 run to about
0.22 bps in the Top10 run. The most extreme names are not better OOS; they look
more fragile.

This is an anti-silver-bullet result for the pairwise ranker: the model can
rank the training sample more aggressively, but the extreme tail is not where
the validation alpha lives.

## Practical Takeaway

For this specific Phase6P ranker, Top10-Bottom10 is not an improvement over
Top20-Bottom20. If this score is used at all, it should likely be used in a
broader, more diversified bucket or inside an optimizer rather than as an
extreme-tail selector.

## Artifacts

- Output directory: `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6u_pairwise_ranker_daily_horizon_profile_20260513_top10`
- Daily metrics: `phase6u_daily_horizon_metrics.csv`
- Horizon summary: `phase6u_horizon_summary.csv`
- Cumulative summary: `phase6u_horizon_cumulative_summary.csv`
- Report: `phase6u_pairwise_ranker_daily_horizon_profile.md`
