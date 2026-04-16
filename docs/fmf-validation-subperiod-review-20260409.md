# FMF Validation Subperiod Review (2026-04-09)

## Scope

This note summarizes the first validation-internal subperiod review on the frozen `FMF` shortlist.

- shortlist manifest:
  - `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_robustness_20260409/frozen_shortlist_manifest.csv`
- output root:
  - `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_subperiod_review_20260409`
- lockbox status:
  - the `2020-01-02 ~ 2026-04-08` test window remained closed

## Common Validation Window

The frozen shortlist aligns to the following common validation window:

- `2014-08-05 ~ 2019-12-31`

This is later than the full validation start because the rolling candidates need warm-up history before their first live allocation.

## Validation Blocks

The review used three validation-only blocks:

- `block_2014_2015`: `2014-08-05 ~ 2015-12-31`
- `block_2016_2017`: `2016-01-01 ~ 2017-12-31`
- `block_2018_2019`: `2018-01-01 ~ 2019-12-31`

## Main Results

### Block-level summary

Key candidate results by block:

- `rolling_c2_v0_seed_core`
  - `2014-2015`: annualized return `0.22%`, Sharpe `0.070`
  - `2016-2017`: annualized return `6.96%`, Sharpe `1.636`
  - `2018-2019`: annualized return `5.09%`, Sharpe `1.299`

- `rolling_fmf_c2_e42_c10_d22_i20_t06`
  - `2014-2015`: annualized return `0.09%`, Sharpe `0.042`
  - `2016-2017`: annualized return `7.63%`, Sharpe `1.857`
  - `2018-2019`: annualized return `5.96%`, Sharpe `1.471`

- `rolling_fmf_c2_e42_c10_d22_i18_t08`
  - `2014-2015`: annualized return `0.18%`, Sharpe `0.062`
  - `2016-2017`: annualized return `7.47%`, Sharpe `1.824`
  - `2018-2019`: annualized return `5.79%`, Sharpe `1.444`

- `rolling_erc_core`
  - `2014-2015`: annualized return `-0.57%`, Sharpe `-0.094`
  - `2016-2017`: annualized return `7.50%`, Sharpe `1.646`
  - `2018-2019`: annualized return `5.19%`, Sharpe `1.249`

- `static_equal_weight_non_cash`
  - `2014-2015`: annualized return `-1.78%`, Sharpe `-0.269`
  - `2016-2017`: annualized return `8.69%`, Sharpe `1.598`
  - `2018-2019`: annualized return `5.47%`, Sharpe `0.879`

### Stability ranking

From `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_subperiod_review_20260409/stability_overview.csv`:

- `rolling_c2_v0_seed_core`
  - positive block ratio: `1.000`
  - worst-block Sharpe: `0.070`
  - average-block Sharpe: `1.002`

- `rolling_fmf_c2_e42_c10_d22_i18_t08`
  - positive block ratio: `1.000`
  - worst-block Sharpe: `0.062`
  - average-block Sharpe: `1.110`

- `rolling_fmf_c2_e42_c10_d22_i20_t06`
  - positive block ratio: `1.000`
  - worst-block Sharpe: `0.042`
  - average-block Sharpe: `1.123`

- `rolling_erc_core`
  - positive block ratio: `0.667`
  - worst-block Sharpe: `-0.094`

- `static_equal_weight_non_cash`
  - positive block ratio: `0.667`
  - worst-block Sharpe: `-0.269`

## Interpretation

### 1. The lead `C2` candidate is still the strongest on average

`rolling_fmf_c2_e42_c10_d22_i20_t06` remains the best candidate by average block Sharpe and by the stronger `2016-2017` and `2018-2019` blocks.

### 2. The seed is more conservative, but genuinely more even

`rolling_c2_v0_seed_core` is no longer the validation leader on headline Sharpe, but it is the most even candidate in this block review:

- all three blocks remain positive
- worst-block Sharpe is the least weak among the shortlist

This makes the seed a meaningful retained baseline, not just a stale reference point.

### 3. The neighbor candidate is valuable

`rolling_fmf_c2_e42_c10_d22_i18_t08` is almost as stable as the seed in the weak early block, while still staying close to the lead candidate in the later stronger blocks.

This makes it the most useful near-neighbor to keep beside the lead candidate.

### 4. `ERC` and naive equal-weight still matter, but only as baselines

Both `rolling_erc_core` and `static_equal_weight_non_cash` go negative in the weakest block.

That does not make them useless:

- `ERC` remains the right equal-risk baseline
- `static_equal_weight_non_cash` remains the right naive baseline

But neither currently looks like a serious promotion candidate versus the rebuilt `C2` region.

## Recommended Status

No change to the frozen shortlist.

But the internal interpretation should tighten:

- primary candidate:
  - `rolling_fmf_c2_e42_c10_d22_i20_t06`
- near-neighbor candidate:
  - `rolling_fmf_c2_e42_c10_d22_i18_t08`
- conservative stability anchor:
  - `rolling_c2_v0_seed_core`
- retained baselines:
  - `rolling_erc_core`
  - `static_equal_weight_non_cash`

## Recommended Next Step

Do **not** reopen the test lockbox yet.

Preferred next move:

- freeze the above interpretation
- stop validation grid expansion
- wait until a formal evaluation round is explicitly authorized before exposing the `2020-01-02 ~ 2026-04-08` test window
