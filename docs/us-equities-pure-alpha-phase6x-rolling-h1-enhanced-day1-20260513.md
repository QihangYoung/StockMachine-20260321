# Phase6X Rolling H1 Enhanced Pairwise Ranker

Date: 2026-05-13

Scope: validation-only walk-forward. The test lockbox is not used.

## Question

Phase6W showed that h1-specific price features are useful inputs, but the
static model still overfits. Phase6X tests whether a realistic walk-forward
procedure can control this:

- Train: previous 12 calendar months.
- Early-stopping validation: next 3 calendar months.
- Predict: next 1 calendar month.
- Roll monthly through 2018-01 to 2019-12.
- Metric: day1 strict residual only.
- Bucket: Top20% minus Bottom20% by rolling pairwise score.

## Overall Result

| period | sessions | mean Rank IC | Rank IC t-stat | day1 spread | spread t-stat | positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2018-01 to 2019-12 | 492 | 0.0092 | 4.00 | 2.93 bps/day | 3.03 | 55.7% |

This is a meaningful improvement over the static h1 enhanced model. It also
slightly exceeds the static h10-supervised model on day1-only validation
spread.

## Day1 Comparison

Weighted by validation sessions.

| model | day1 spread | positive rate note |
| --- | ---: | --- |
| Static h10-supervised Phase6P | 2.58 bps/day | 2018: 53.4%, 2019: 57.3% |
| Static h1 old-feature Phase6V | 2.21 bps/day | 2018: 55.8%, 2019: 55.2% |
| Static h1 enhanced-feature Phase6W | 1.41 bps/day | 2018: 51.0%, 2019: 52.7% |
| Rolling h1 enhanced Phase6X | 2.93 bps/day | 55.7% combined |

## Monthly Profile

The rolling model is not uniformly positive. Strong months include 2018-11,
2018-12, 2019-05, 2019-11, and 2019-12. Weak months include 2018-01, 2018-04,
2019-01, 2019-04, and 2019-09.

The important change versus Phase6W is not that every month becomes positive;
it is that the walk-forward procedure reduces static overfit and recovers a
small but statistically visible day1 edge.

## Model Diagnostics

The early-stopping pair AUC is usually only slightly above 0.50, and several
months stop at very low iteration counts. This is a healthy warning: the model
should not be trusted to fit deeply on h1 labels. The 3-month validation window
is doing real work by forcing shallow models in weak windows.

## Interpretation

The result supports a narrow conclusion:

Rolling h1 enhanced features can produce a modest day1 validation edge when
the model is constrained by recent out-of-sample validation.

It does not yet support a broader conclusion that h1 ML is a production-ready
replacement for the existing SOTA portfolio. The edge is small, noisy, and
measured on standalone top-bottom residual ranking, not after full portfolio
construction, costs, borrow constraints, and beta/sector/size optimization.

## Next Checks

- Re-run with fewer/more pairs per date to confirm the result is not a sampling
  artifact.
- Compare Top10, Top20, and optimizer-weighted usage on day1 only.
- Test a simpler regularized model, because early stopping often prefers very
  shallow tree ensembles.
- If promoted, evaluate inside the actual beta/sector/size-neutral portfolio
  optimizer rather than standalone buckets.

## Artifacts

- Output directory: `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6x_rolling_h1_enhanced_pairwise_20260513`
- Daily metrics: `phase6x_rolling_day1_daily_metrics.csv`
- Monthly summary: `phase6x_rolling_day1_monthly_summary.csv`
- Model diagnostics: `phase6x_rolling_model_diagnostics.csv`
- Score panel: `phase6x_rolling_score_panel.csv.gz`
- Plot: `phase6x_rolling_day1_cumulative_spread.png`
