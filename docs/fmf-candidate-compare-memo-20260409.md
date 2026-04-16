# FMF Candidate Compare Memo (2026-04-09)

## Scope

This memo compares the three most important validation-only `FMF` candidates:

- `rolling_c2_v0_seed_core`
- `rolling_fmf_c2_e42_c10_d22_i20_t06`
- `rolling_fmf_c2_e42_c10_d22_i18_t08`

All comparisons below use the already-exposed validation-only window:

- `2014-08-05 ~ 2019-12-31`

The lockbox test window remains closed.

## Reference Charts

- cumulative equity comparison:
  - `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_curve_review_20260409/seed_lead_neighbor_equity_comparison.png`
- drawdown comparison:
  - `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_curve_review_20260409/seed_lead_neighbor_drawdown_comparison.png`
- monthly return distributions:
  - `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_curve_review_20260409/seed_lead_neighbor_monthly_return_distributions.png`

## Headline Comparison

### Total return and monthly hit rate

- `rolling_c2_v0_seed_core`
  - total return: `26.68%`
  - positive months: `44 / 65`
  - monthly hit rate: `67.69%`

- `rolling_fmf_c2_e42_c10_d22_i20_t06`
  - total return: `30.14%`
  - positive months: `42 / 65`
  - monthly hit rate: `64.62%`

- `rolling_fmf_c2_e42_c10_d22_i18_t08`
  - total return: `29.52%`
  - positive months: `42 / 65`
  - monthly hit rate: `64.62%`

### Max drawdown

- `rolling_c2_v0_seed_core`: `-5.27%`
- `rolling_fmf_c2_e42_c10_d22_i20_t06`: `-5.25%`
- `rolling_fmf_c2_e42_c10_d22_i18_t08`: `-5.24%`

So the lead/neighbor pair is not winning because drawdowns are materially shallower. The edge is mainly coming from higher return generation at similar drawdown depth.

## Monthly Distribution Comparison

### Seed

- mean monthly return: `0.37%`
- median monthly return: `0.45%`
- monthly volatility: `1.20%`
- best month: `3.04%`
- worst month: `-2.63%`

### Lead

- mean monthly return: `0.41%`
- median monthly return: `0.44%`
- monthly volatility: `1.26%`
- best month: `3.30%`
- worst month: `-2.64%`

### Neighbor

- mean monthly return: `0.41%`
- median monthly return: `0.46%`
- monthly volatility: `1.24%`
- best month: `3.22%`
- worst month: `-2.61%`

Interpretation:

- the seed wins on month-to-month consistency
- the lead and neighbor win on average monthly payoff
- the gap is not caused by meaningfully worse left tails for the lead pair

## Interpretation

### 1. Seed is the stability anchor

`rolling_c2_v0_seed_core` still deserves to stay in the final frozen shortlist because:

- it has the best monthly hit rate
- it stayed positive in all three validation blocks
- its max drawdown is effectively the same as the stronger candidates

This makes it the cleanest conservative reference point.

### 2. Lead is still the primary candidate

`rolling_fmf_c2_e42_c10_d22_i20_t06` remains the primary candidate because:

- it has the highest total return
- it kept the strongest average block-level Sharpe in the subperiod review
- it achieved that without paying a materially deeper drawdown penalty

### 3. Neighbor is the most useful nearby alternative

`rolling_fmf_c2_e42_c10_d22_i18_t08` remains the best nearby alternative because:

- it is close to the lead on total return
- it is slightly more even than the lead in the weakest block
- it belongs to the same preferred validation region

## Working Conclusion

The current interpretation should remain:

- primary candidate:
  - `rolling_fmf_c2_e42_c10_d22_i20_t06`
- near-neighbor candidate:
  - `rolling_fmf_c2_e42_c10_d22_i18_t08`
- conservative anchor:
  - `rolling_c2_v0_seed_core`

The choice among them is now mostly a choice between:

- slightly better consistency
- versus slightly higher payoff at similar drawdown

That is a useful frontier.

It is **not** a case where one candidate is clearly dominating all others on every axis inside validation.
