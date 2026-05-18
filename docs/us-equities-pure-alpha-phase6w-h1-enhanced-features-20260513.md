# Phase6W H1 Enhanced Feature Pairwise Ranker

Date: 2026-05-13

Scope: validation-only diagnostic. The test lockbox is not used.

## Question

Phase6V changed the pairwise ranker supervision target from h10 strict residual
return to h1 strict residual return, but did not improve validation performance.
One likely reason was feature-horizon mismatch: the old feature set mostly
described 5-60 day price states, not next-day open-to-open behavior.

Phase6W keeps h1 supervision and adds h1/short-horizon price features.

## Added Features

The new point-in-time features are known by the close of decision date `t` and
are used to predict the first future holding-day open-to-open residual return:

- `close_reversal_1d`, `close_reversal_2d`, `close_reversal_3d`
- `open_to_close_return_1d`
- `overnight_gap_1d`
- `high_low_range_1d`
- `close_location_1d`
- `abs_close_return_1d`
- `abs_open_to_close_return_1d`
- `abs_overnight_gap_1d`
- `realized_vol_3d`, `realized_vol_5d`
- `volume_shock_1d`
- `dollar_volume_shock_1d`
- `range_shock_1d`

## H1 Label Model Quality

| model | train AUC | 2018 AUC | 2019 AUC | train h1 spread | 2018 h1 spread | 2019 h1 spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase6V h1 old features | 0.5498 | 0.5070 | 0.5055 | 7.71 bps | 2.09 bps | 2.40 bps |
| Phase6W h1 enhanced features | 0.5568 | 0.5080 | 0.5046 | 12.46 bps | 1.14 bps | 1.60 bps |

The enhanced feature set materially increases in-sample h1 fit, but does not
improve validation AUC or validation h1 label spread.

## Day1-10 Forward Profile

Mean Top20% minus Bottom20% by `pairwise_rank_score`, strict daily residual,
bps per holding day.

| model | train day1 | train sum | 2018 day1 | 2018 sum | 2019 day1 | 2019 sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase6P h10-supervised | 4.51 | 39.57 | 2.44 | 6.03 | 2.73 | 10.75 |
| Phase6V h1 old features | 7.35 | 17.18 | 1.87 | 1.98 | 2.55 | 1.78 |
| Phase6W h1 enhanced features | 12.20 | 15.52 | 0.99 | -0.91 | 1.85 | 5.74 |

Full Phase6W strict residual profile:

| split | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | day1-10 sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 12.20 | 1.25 | 1.11 | 1.22 | -0.32 | 0.60 | -0.48 | 0.50 | -0.50 | -0.07 | 15.52 |
| model_validation_2018 | 0.99 | 0.56 | 0.47 | -0.81 | -1.77 | 0.55 | -1.28 | 0.59 | -1.21 | 1.00 | -0.91 |
| final_validation_2019 | 1.85 | 0.88 | -0.24 | -0.50 | 2.01 | 1.99 | 1.90 | 0.66 | -0.50 | -2.30 | 5.74 |

## Feature Importance

The enhanced model uses the new h1 features heavily. Top gain features:

| rank | feature |
| ---: | --- |
| 1 | `abs_close_return_1d` |
| 2 | `overnight_gap_1d` |
| 3 | `high_low_range_1d` |
| 4 | `abs_overnight_gap_1d` |
| 5 | `close_reversal_1d` |
| 10 | `realized_vol_5d` |
| 11 | `realized_vol_3d` |
| 20 | `volume_shock_1d` |

This confirms that the model can use the new h1 inputs. The problem is not
that the features are ignored; the problem is that the in-sample h1 structure
does not generalize cleanly.

## Interpretation

Adding h1-aligned price features helps relative to the old h1-feature mismatch:
2019 day1-10 strict residual sum improves from 1.78 bps to 5.74 bps.

However, it still does not beat the original h10-supervised Phase6P profile.
More importantly, the enhanced h1 model shows classic overfit behavior:
training day1 spread jumps to 12.20 bps, while validation day1 is only 0.99 bps
in 2018 and 1.85 bps in 2019.

This suggests that h1 prediction needs either stronger regularization, simpler
models, or non-price information. Purely adding short-horizon price features is
not enough.

## Practical Takeaway

Do not replace the current h10-supervised pairwise ranker with Phase6W.

Phase6W is still useful evidence: h1-specific inputs matter, but a flexible tree
model overfits them. A better next version would be a simpler regularized model
or a multi-horizon target such as day1-day4 residual blend, rather than pure h1.

## Artifacts

- Enhanced h1 model: `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6w_h1_enhanced_feature_pairwise_ranker_20260513`
- Enhanced h1 day1-10 profile: `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6w_h1_enhanced_daily_horizon_profile_20260513`
