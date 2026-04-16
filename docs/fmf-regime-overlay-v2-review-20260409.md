# FMF Regime Overlay V2 Review (2026-04-09)

## Purpose

This note records the validation-only follow-up after the first conservative regime overlay prototype.

The immediate goal was not to build a richer meta-strategy, but to answer a narrower question:

- can a better detector materially improve the conservative `equity_total <-> duration` overlay?

The lockbox test window remained closed throughout this work.

## What Changed

The first overlay review showed two issues:

- the detector under-identified `defensive` states, especially in `2018 Q4`
- most days were still classified as `risk_on`, so the overlay mostly added risk without improving defense quality

To address that, a validation-only detector sensitivity pass was run across:

- `trend_lookback`: `42 / 63 / 84 / 126`
- `vol_lookback`: `21 / 42 / 63`
- `drawdown_threshold`: `-5% / -6% / -8%`
- `high_vol_threshold`: `12% / 14% / 16%`

The selection score emphasized `2018 Q4` alignment:

- `50%` `q4_2018_match_ratio`
- `30%` `q4_2018_defensive_recall`
- `20%` `daily_match_ratio`

The best detector was:

- `trend_lookback = 84`
- `vol_lookback = 42`
- `drawdown_threshold = -5%`
- `high_vol_threshold = 16%`

## Detector Improvement

Relative to the original detector, the new one materially improved state alignment.

Original overlay/reference comparison:

- `daily_match_ratio = 67.0%`
- `overlay_defensive_share = 6.2%`

Best sensitivity detector:

- `daily_match_ratio = 76.8%`
- `q4_2018_match_ratio = 76.2%`
- `q4_2018_defensive_recall = 76.2%`
- `q4_2018_defensive_precision = 100%`
- `overlay_defensive_share = 17.0%`

In plain language:

- the detector now catches much more of `2018 Q4`
- it produces a state mix that is much closer to the consensus-style reference regime timeline

## Policy Application Improvement

The improvement was not only cosmetic at the daily-label level. It propagated into actual monthly policy choices.

During the key stress window:

- overlay v1:
  - `2018-11-02 ~ 2018-12-03`: `neutral`
  - `2018-12-04 ~ 2019-01-06`: `neutral`
  - `2019-01-07 ~ 2019-02-05`: `defensive`
- overlay v2:
  - `2018-11-02 ~ 2018-12-03`: `defensive`
  - `2018-12-04 ~ 2019-01-06`: `defensive`
  - `2019-01-07 ~ 2019-02-05`: `defensive`
  - `2019-02-06 ~ 2019-03-05`: `defensive`

Overall monthly decision-state counts changed from:

- v1: `risk_on 43 / neutral 18 / defensive 4`
- v2: `risk_on 46 / neutral 8 / defensive 11`

So the detector revision did meaningfully change the applied overlay, especially in the windows that originally motivated the rework.

## Validation Outcome

Even after the detector improvement, the overlay still did not beat the static lead candidate.

Common validation window (`2014-08-05 ~ 2019-12-31`):

| Strategy | Annualized Return | Annualized Vol | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| `rolling_fmf_c2_e42_c10_d22_i20_t06` | `4.995%` | `4.201%` | `1.181` | `-5.245%` |
| `overlay v1` | `5.055%` | `4.266%` | `1.177` | `-5.446%` |
| `overlay v2` | `5.035%` | `4.264%` | `1.173` | `-5.548%` |

Additional note:

- v2 does change the realized daily path relative to v1
- `315` daily returns changed
- `max_abs_daily_return_diff ≈ 0.089%`

So this is not a stale-file issue. The overlay really changed, but the PnL trade-off still did not improve.

## Interpretation

The main finding is subtle but important:

- the detector problem was real
- we improved it
- but detector improvement alone still did not justify the overlay

This suggests the main bottleneck is no longer only "state recognition."

At least in this validation regime, the conservative overlay seems structurally limited because:

- it only moves `equity_total` and `duration`
- it leaves the rest of the policy unchanged
- the static lead is already fairly efficient

So even a more sensible detector does not create enough incremental edge.

## Working Conclusion

The regime-aware idea should not be abandoned, but `overlay v2` still does **not** earn promotion over the static lead.

Current status:

- static lead remains the primary candidate:
  - `rolling_fmf_c2_e42_c10_d22_i20_t06`
- `overlay v2` remains an exploratory branch
- if dynamic adaptation continues, the next step should focus on:
  - better state variables
  - or a more meaningful action layer
  - not more detector micro-tuning alone

## Files

- detector sensitivity:
  - `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_regime_detector_sensitivity_20260409/detector_sensitivity_summary.csv`
- overlay v1:
  - `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_regime_overlay_experiment_20260409`
- overlay v2:
  - `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_regime_overlay_experiment_v2_20260409`
