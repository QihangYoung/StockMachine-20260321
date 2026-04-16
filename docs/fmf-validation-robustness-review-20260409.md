# FMF Validation Robustness Review (2026-04-09)

## Scope

This note summarizes the first shared-robustness pass for the rebuilt `FMF` validation line.

- universe: `SPY / VXUS / IEF / LQD / GLD / FMF / BIL`
- validation window only: `2013-08-05 ~ 2019-12-31`
- untouched lockbox remains closed: `2020-01-02 ~ 2026-04-08`
- source artifacts:
  - `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_baseline_validation_only_20260409`
  - `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_c2_grid_search_20260409`
  - `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_c2_grid_search_narrow_20260409`
- robustness output root:
  - `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_robustness_20260409`

## What Was Evaluated

The shared robustness suite evaluated:

- time stability by year
- tail dependence after trimming top/bottom extreme days
- cost and execution stress
- turnover concentration
- search-level selection bias
- parameter stability around the search leader

`universe stability` was not run in this pass because no ETF-universe perturbation experiment has been staged yet.

## Candidate Set Reviewed

The robustness pass reviewed the following `8` validation-only candidates:

- `static_equal_weight_non_cash`
- `rolling_erc_core`
- `rolling_c2_v0_seed_core`
- `rolling_fmf_c2_e42_c10_d22_i20_t06`
- `rolling_fmf_c2_e42_c10_d22_i18_t08`
- `rolling_fmf_c2_e40_c10_d22_i20_t08`
- `rolling_fmf_c2_e42_c10_d20_i20_t08`
- `rolling_fmf_c2_e42_c08_d22_i20_t08`

The exact manifest is saved at:

- `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_robustness_20260409/manifests/candidate_manifest.csv`

## Main Findings

### 1. The rebuilt `C2` leader remains ahead after robustness checks

The strongest candidate in the robustness overview remains:

- `rolling_fmf_c2_e42_c10_d22_i20_t06`

Key comparison from the robustness overview:

- `rolling_fmf_c2_e42_c10_d22_i20_t06`: annualized return `4.995%`, Sharpe `1.181`, max drawdown `-5.25%`
- `rolling_c2_v0_seed_core`: annualized return `4.473%`, Sharpe `1.069`, max drawdown `-5.27%`
- `rolling_erc_core`: annualized return `4.490%`, Sharpe `1.010`, max drawdown `-6.24%`
- `static_equal_weight_non_cash`: annualized return `4.981%`, Sharpe `0.861`, max drawdown `-10.24%`

This means the validation winner survives time/tail/cost turnover checks and still looks better than the seed and `ERC` baselines on a risk-adjusted basis.

### 2. Time stability is acceptable, but not uniformly strong

For `rolling_fmf_c2_e42_c10_d22_i20_t06`, yearly validation results show:

- positive years: `2014`, `2016`, `2017`, `2019`
- negative years: `2015`, `2018`
- positive-year ratio in the overview: `0.667`

This is good enough for shortlist retention, but not strong enough to claim unusually broad temporal dominance.

### 3. Tail dependence is present, but not pathological

For the lead candidate:

- dropping the best `5` days reduces annualized return from `4.995%` to `4.110%`
- Sharpe falls from `1.181` to `1.000`
- dropping the worst `5` days lifts annualized return to `6.160%`

Interpretation:

- the candidate does benefit from a small number of strong upside days
- but the result is not driven by a vanishingly small handful of extremes
- left-tail damage is also not concentrated enough to disqualify the strategy

### 4. Cost sensitivity is mild

For the lead candidate:

- at `20 bps` per side, annualized return remains `4.834%`
- at `40 bps` per side, annualized return remains `4.674%`
- annualized-return slope is about `-0.080%` per additional `10 bps`

This is consistent with the low-turnover structure of the rolling multi-asset line.

### 5. Turnover concentration looks healthy for a monthly allocator

For the lead candidate:

- active-day ratio is about `4.8%`
- top `5` turnover days account for about `37.0%` of total turnover
- it takes `12` active days to reach `50%` of cumulative turnover

This does not look like a fragile, over-traded implementation.

### 6. Search-bias risk is now the main warning sign

Broad-grid selection-bias summary:

- leader Sharpe: `1.141`
- median Sharpe: `0.991`
- leader vs median gap: `0.150`
- leader z-score: `2.292`

Narrow-grid selection-bias summary:

- leader Sharpe: `1.181`
- median Sharpe: `1.114`
- leader vs median gap: `0.067`
- leader z-score: `2.352`

Interpretation:

- the search leader is clearly above the search-surface center
- this is good enough to keep the region
- but it also means more blind validation-only grid search would raise overfitting risk quickly

### 7. Parameter stability is only moderate

Broad-grid parameter stability:

- leader: `equity_total=0.40, credit=0.10, duration=0.20, inflation_hedge=0.20, trend=0.10`
- neighborhood size: `3`
- neighbor-only mean Sharpe: `1.088`
- leader advantage vs neighbor mean: `0.053`

Narrow-grid parameter stability:

- leader: `equity_total=0.42, credit=0.10, duration=0.22, inflation_hedge=0.20, trend=0.06`
- neighborhood size: `1`

Interpretation:

- the broad search does show a real high-quality region
- the narrow search winner now sits on the boundary
- so the right move is to stop fine-grained grid descent, not to keep pushing toward a sharper local optimum

## Frozen Shortlist

The following shortlist is the recommended freeze set after the first robustness pass:

- `static_equal_weight_non_cash`
  - role: naive capital-weight baseline
- `rolling_erc_core`
  - role: equal-risk baseline
- `rolling_c2_v0_seed_core`
  - role: rebuilt-seed baseline
- `rolling_fmf_c2_e42_c10_d22_i20_t06`
  - role: lead candidate
- `rolling_fmf_c2_e42_c10_d22_i18_t08`
  - role: near-neighbor candidate in the same preferred region

The frozen shortlist manifest is stored at:

- `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_robustness_20260409/frozen_shortlist_manifest.csv`

## Recommended Next Step

Do **not** continue broad or narrow validation-only grid search by default.

Preferred next action:

- run validation-internal sub-period checks on the frozen shortlist only

Still forbidden unless explicitly approved for a formal decision round:

- reopening the `2020-01-02 ~ 2026-04-08` lockbox test window
