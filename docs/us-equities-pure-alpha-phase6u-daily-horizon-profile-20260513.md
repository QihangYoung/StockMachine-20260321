# Phase6U Daily Horizon Profile for the Phase6P Pairwise Ranker

Date: 2026-05-13

Scope: validation-era diagnostic only. This memo does not use the test lockbox.

## Question

Phase6P evaluated the sparse pairwise ranker mainly on an aggregate h10 target.
This can hide whether the model earns steadily over the full 10-session holding
window, earns early and then decays, or earns in a few isolated days and gives
the edge back later.

Phase6U decomposes the same score into holding-day 1 through holding-day 10.

## Method

- Score source: Phase6P `pairwise_rank_score`.
- Universe/sample source: Phase6P score panel.
- Bucket definition: daily Top 20% minus Bottom 20% by score.
- Holding-day 1: adjusted-open return from decision date +1 to +2.
- Holding-day 10: adjusted-open return from decision date +10 to +11.
- Reported targets: raw open-to-open, beta residual open-to-open, and daily
  strict size/style/SIC2 residual open-to-open.
- Daily strict residual controls: beta, market-cap/size proxy, liquidity,
  reversal, momentum, beta-residual momentum, volatility-adjusted momentum,
  and SIC2.

Important caveat: daily strict residuals summed over day 1..10 are not
identical to residualizing one aggregate h10 return. They are a horizon-profile
diagnostic, not a replacement for the original h10 label.

## Pairwise Score: Strict Residual Daily Spread

Mean Top 20% minus Bottom 20%, bps per holding day.

| split | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | day1-10 sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 4.51 | 4.08 | 4.30 | 4.21 | 4.01 | 4.38 | 4.30 | 3.54 | 3.38 | 2.86 | 39.57 |
| model_validation_2018 | 2.44 | 2.60 | 1.40 | 1.25 | -1.07 | -0.51 | 0.27 | -0.97 | 0.39 | 0.23 | 6.03 |
| final_validation_2019 | 2.73 | 1.14 | 0.45 | 1.58 | 0.59 | 1.38 | 2.22 | 1.04 | -0.35 | -0.03 | 10.75 |

## Pairwise Score: Other Target Views

Mean Top 20% minus Bottom 20%, bps per holding day.

### Final Validation 2019

| target | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | day1-10 sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw open-to-open | -1.17 | -4.26 | -3.69 | -2.54 | -2.46 | -4.37 | -1.43 | -5.48 | -4.88 | -4.20 | -34.46 |
| beta residual | 2.98 | 0.97 | 0.86 | 2.17 | 1.86 | 0.16 | 2.99 | -0.70 | -0.25 | -0.16 | 10.89 |
| strict residual | 2.73 | 1.14 | 0.45 | 1.58 | 0.59 | 1.38 | 2.22 | 1.04 | -0.35 | -0.03 | 10.75 |

### Model Validation 2018

| target | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | day1-10 sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw open-to-open | 2.39 | 3.93 | 0.40 | 1.04 | -1.77 | -2.40 | -1.23 | -1.93 | -0.54 | -3.17 | -3.28 |
| beta residual | 1.91 | 2.86 | -0.55 | 0.30 | -2.90 | -3.32 | -2.76 | -3.42 | -2.20 | -3.81 | -13.88 |
| strict residual | 2.44 | 2.60 | 1.40 | 1.25 | -1.07 | -0.51 | 0.27 | -0.97 | 0.39 | 0.23 | 6.03 |

## Interpretation

The training sample shows a broad but decaying edge: days 1 through 7 are all
around 4 bps, while days 8 through 10 fade toward 3 bps.

The 2018 validation sample is front-loaded. Days 1 through 4 are positive, but
days 5, 6, and 8 give a meaningful amount back. This means a clean h10 average
is not the natural shape of the signal in that window.

The 2019 validation sample is weak but not completely flat. It has visible
positive residual days around day 1, day 4, and day 7, but days 9 and 10 are
near zero or slightly negative. This supports the view that the model learned a
short-lived price/style payoff more than a stable 10-day alpha.

The raw 2019 spread is negative even though beta-residual and strict-residual
spreads are positive. That is a reminder that this ranker is not a standalone
raw-return selector; it only has meaning after beta/style accounting.

## Practical Takeaway

Phase6P still does not look like a robust pure-alpha breakthrough. The daily
profile says the model has a mild, horizon-dependent residual signal, but the
signal is too small and unstable to justify increasing complexity by itself.

Better next experiments are horizon-aware rather than simply bigger-model
experiments:

- Test whether the same score should be used with a shorter effective holding
  window or earlier exit logic.
- Train horizon-specific labels, for example day 1..3 and day 4..7 separately,
  instead of one aggregate h10 label.
- Compare against simpler style/payoff timing models on the same daily
  horizon-profile basis.

## Artifacts

- Daily metrics: `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6u_pairwise_ranker_daily_horizon_profile_20260513/phase6u_daily_horizon_metrics.csv`
- Horizon summary: `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6u_pairwise_ranker_daily_horizon_profile_20260513/phase6u_horizon_summary.csv`
- Cumulative summary: `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6u_pairwise_ranker_daily_horizon_profile_20260513/phase6u_horizon_cumulative_summary.csv`
- Report: `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6u_pairwise_ranker_daily_horizon_profile_20260513/phase6u_pairwise_ranker_daily_horizon_profile.md`
