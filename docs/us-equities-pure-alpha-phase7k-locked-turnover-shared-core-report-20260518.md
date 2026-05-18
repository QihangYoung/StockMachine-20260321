# Phase7K Locked-Turnover Shared-Core Strategy Report

Generated: 2026-05-18

Scope: validation-only research report. No test lockbox performance is used.

## 1. Executive Summary

Phase7K is a shared-capital multi-factor long-short strategy. It keeps one
shared long-short book, but assigns each position a factor-reason ledger. New
or increased position weight is split into reason-level lots, and each lot has
its own minimum holding clock. This lets the same dollar of capital carry
several factor exposures while preventing the optimizer from overreacting to
daily signal noise.

The primary candidate is:

`shared_core_lambda_0p005_tb0p15`

It uses:

- shared factor score: reversal, momentum, small size, low beta, cash quality
- daily signal generation
- reason-level minimum holds
- hard daily turnover budget of `0.15` NAV after initial build
- transaction cost assumption of `4 bps/side`
- strict open-to-open daily path

On the validation window `2014-11-13` through `2019-12-31`, the primary
candidate produced:

| strategy | net ann. return | net vol | net Sharpe | max drawdown | mean daily turnover | mean cost |
|---|---:|---:|---:|---:|---:|---:|
| `shared_core_lambda_0p005_tb0p15` | 21.35% | 10.84% | 1.97 | -10.77% | 0.153 | 0.61 bps/day |
| `shared_no_momentum_lambda_0p005_tb0p15` | 16.73% | 10.35% | 1.62 | -11.71% | 0.153 | 0.61 bps/day |
| `shared_slow_core_lambda_0p005_tb0p15` | 12.23% | 11.62% | 1.05 | -21.40% | 0.023 | 0.09 bps/day |

The main result is not simply cost savings. Compared with the earlier no-hard-hold
Phase7J strict path, the shared-core daily turnover fell from about `0.658` to
`0.153`, while net Sharpe rose from about `0.55` to `1.97`. The reason-level
holding rule appears to remove a large amount of unstable daily re-optimization.

## 2. Strategy Idea

The strategy starts from a simple observation: a stock can be attractive for
more than one reason at the same time. For example, one long position may be
supported by reversal, momentum confirmation, smaller size, lower beta, and
cash-rich fundamentals. If the strategy allocates separate NAV budgets to
separate factor sleeves, it may fail to reuse capital efficiently. Phase7K
instead keeps one shared book and lets the same position carry multiple factor
exposures.

The hard part is turnover. If a shared-capital optimizer fully rebuilds the book
every day, it can trade too much. Phase7K addresses this by separating actual
capital from position reasons:

- actual capital: one shared long-short portfolio
- reason ledger: factor-level lots attached to each position
- holding constraint: applied to factor-reason lots
- turnover control: applied at portfolio level

This means a position can partially expire. If the reversal reason expires but
the cash-quality reason is still locked, the optimizer can reduce the reversal
portion while keeping the longer-horizon reason.

## 3. Factor Definitions

All stock-level factors are cross-sectionally standardized within SIC2 sector by
date, with date-level fallback, then clipped to `[-3, 3]`.

| score column | raw direction | interpretation |
|---|---|---|
| `reversal_score` | `-return_5d` | recent loser is better |
| `momentum_score` | `momentum_l120_s20` | 120-session momentum, skipping recent 20 sessions |
| `small_size_score` | `-market_cap_log` | smaller capitalization is better |
| `low_beta_score` | `-beta` | lower estimated beta is better |
| `cash_quality_score` | `cash_to_assets` | higher cash/assets is better |

For the primary shared-core portfolio, factor-score weights are:

| factor | score weight |
|---|---:|
| `reversal_score` | 0.25 |
| `momentum_score` | 0.10 |
| `small_size_score` | 0.30 |
| `low_beta_score` | 0.20 |
| `cash_quality_score` | 0.15 |

The composite score is the weighted average of available factor scores, then
z-scored cross-sectionally on that date.

## 4. Portfolio Construction

Each day:

1. Compute all factor scores for the active universe.
2. Compute composite score for each portfolio.
3. Build candidate sets:
   - long candidates: top `120` by composite score plus incumbents and locked names
   - short candidates: bottom `120` by composite score plus incumbents and locked names
4. Solve a linear program for long and short side weights.
5. Attach new or increased weight to factor-reason lots.
6. Run the strict daily path with turnover cost.

Portfolio constraints:

| constraint | value |
|---|---:|
| long gross | 1.0 |
| short gross | 1.0 |
| total gross | 2.0 |
| dollar net exposure | approximately 0 |
| max single-name side weight | `1/30 = 3.33%` |
| beta neutrality | hard ex-ante equality |
| SIC2 sector neutrality | soft penalty, `25.0` |
| daily turnover budget | `0.15` after initial build |
| turnover objective penalty | `0.005` |
| transaction cost | `4 bps/side` |

Minimum holding periods by factor reason:

| factor | min hold |
|---|---:|
| `reversal_score` | 5 sessions |
| `momentum_score` | 10 sessions |
| `small_size_score` | 20 sessions |
| `low_beta_score` | 20 sessions |
| `cash_quality_score` | 20 sessions |

The first build day is allowed to trade up to full gross construction. After
that, the hard turnover budget applies.

## 5. Reason-Level Lot Mechanics

When a position is newly opened or increased, the increment is attributed to
factor reasons using directional factor support.

For a long position, positive support comes from:

```text
factor_weight * factor_score
```

For a short position, positive support comes from:

```text
factor_weight * (-factor_score)
```

The positive supports are normalized to sum to one, then the new weight is split
across factor lots. Each lot stores:

```text
symbol, factor, weight, birth_idx, min_hold
```

If a lot has not reached its minimum hold, it cannot be reduced. Expired lots
may be kept, reduced, or replaced. If an expired lot is kept unchanged, it is not
given a new lock unless the position weight is increased.

This is an accounting ledger only. It does not mean separate capital sleeves are
funded. The actual portfolio remains one shared long-short book.

## 6. Backtest Semantics

Strict path contract:

| item | definition |
|---|---|
| signal session | `T` |
| rebalance | adjusted open at `T+1` |
| PnL | adjusted open-to-open return from `T+1` to `T+2` |
| cost | `sum(abs(target_weight - previous_weight)) * cost_bps_per_side` |
| benchmark | SPY open-to-open return |

Important: this is not a forward-label diagnostic. It builds an actual daily
target-weight path and charges realized target turnover.

## 7. Main Results

Full Phase7K run:

- artifact root:
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7k_locked_turnover_shared_capital_20260518_tb0p15`
- rows:
  - panel: `923,359`
  - positions: `416,359`
  - diagnostics: `3,879`
  - skipped sessions: `0`
  - strict curve: `3,869`
  - strict metrics: `6`

Net strict daily metrics:

| portfolio | final equity | ann. return | ann. vol | Sharpe | max drawdown | mean daily bps | turnover | cost bps/day |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `shared_core_lambda_0p005_tb0p15` | 2.691 | 21.35% | 10.84% | 1.970 | -10.77% | 7.91 | 0.153 | 0.61 |
| `shared_no_momentum_lambda_0p005_tb0p15` | 2.206 | 16.73% | 10.35% | 1.616 | -11.71% | 6.35 | 0.153 | 0.61 |
| `shared_slow_core_lambda_0p005_tb0p15` | 1.806 | 12.23% | 11.62% | 1.052 | -21.40% | 4.85 | 0.023 | 0.09 |

Factor exposure summary for shared core:

| factor | mean exposure | positive exposure rate |
|---|---:|---:|
| `reversal_score` | 0.559 | 92.03% |
| `momentum_score` | 0.983 | 100.00% |
| `small_size_score` | 2.702 | 99.15% |
| `low_beta_score` | 0.190 | 98.92% |
| `cash_quality_score` | 1.637 | 100.00% |

The shared-core book has persistent positive exposure to all intended factors.
Low-beta exposure remains small because hard beta neutrality compresses that
dimension.

## 8. Comparison Against Alpha SOTA And SPY

Comparison artifact root:

`artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7k_shared_core_vs_sota_spy_20260518`

The comparison uses the common window `2014-11-13` through `2019-12-31`, rebased
to `1.0`. Both strategy curves are net of `4 bps/side` turnover cost:

- shared core: Phase7K net return
- alpha SOTA: Phase6 daily records, recomputed as
  `gross_return - turnover * 4 / 10000`
- SPY: benchmark open-to-open return

![Shared Core vs Alpha SOTA vs SPY](../artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7k_shared_core_vs_sota_spy_20260518/shared_core_vs_alpha_sota_spy_equity_drawdown.png)

| series | final equity | ann. return | ann. vol | Sharpe | max drawdown | mean daily bps |
|---|---:|---:|---:|---:|---:|---:|
| Shared Core net 4bps | 2.684 | 21.29% | 10.84% | 1.964 | -10.77% | 7.91 |
| Alpha SOTA net 4bps | 1.712 | 11.08% | 7.49% | 1.480 | -8.69% | 4.28 |
| SPY | 1.774 | 11.86% | 13.24% | 0.896 | -19.02% | 4.83 |

Newey-West t-statistics, using daily returns and lag `10`:

| series | mean daily bps | NW t |
|---|---:|---:|
| Shared Core net 4bps | 7.91 | 4.14 |
| Alpha SOTA net 4bps | 4.28 | 3.02 |
| SPY | 4.83 | 2.35 |

Pairwise Newey-West t-statistics, also lag `10`:

| comparison | mean daily diff bps | NW t |
|---|---:|---:|
| Shared Core - Alpha SOTA | 3.63 | 1.65 |
| Shared Core - SPY | 3.09 | 1.21 |
| Alpha SOTA - SPY | -0.55 | -0.25 |

Interpretation: shared core has a stronger standalone daily return stream than
both comparators in this validation window. However, the paired difference
versus Alpha SOTA has NW t around `1.6`, so the claim should be "promising
validation improvement", not "statistically decisive replacement".

## 9. Reproduction Conditions

Repository context used for this report:

- working directory: `E:\CodeX\StockMachine-260321`
- git head at report time: `5a0dd80`
- script required: `src/stockmachine/apps/run_pure_alpha_phase7k.py`
- note: the workspace had uncommitted/untracked research files. A fresh clone of
  only `5a0dd80` will not reproduce unless the Phase7K script and required
  artifacts are present.

Python/runtime dependencies:

- Python 3.10 was used in this workspace.
- Required packages include `numpy`, `pandas`, `scipy`, and `matplotlib` for the
  comparison plot.
- Run from PowerShell with `PYTHONPATH=src`.

Required input artifacts:

| input | path |
|---|---|
| universe membership | `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase1_universe_builder_20260418/phase1_candidate_universe_membership_validation.csv.gz` |
| beta panel | `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase2_beta_panel_20260418/phase2_beta_panel_validation.csv.gz` |
| size panel | `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6j_true_size_mvp_20260508/phase6j_true_size_panel_validation.csv.gz` |
| non-price panel | `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/non_price_phase1_top1000_two_factor_build_20260428/non_price_two_factor_panel.csv.gz` |
| fundamentals panel | `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/non_price_phase4_companyfacts_fundamentals_20260505/companyfacts_fundamental_panel.csv.gz` |
| SEC CIK map | `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/non_price_phase0_two_factor_data_prep_20260419/sec_universe_cik_mapping.csv` |
| SEC submissions | `data/raw/sec/submissions` |
| stock daily bars | `data/silver/daily_bar/phase0_top1000_yahoo_gap_20130805_20151231_chunk*.jsonl`; `data/silver/daily_bar/phase0_top1000_sip_raw_20160104_20260416_chunk*.jsonl` |
| stock adjustment factors | `data/silver/adj_factor/phase0_top1000_yahoo_gap_20130805_20151231_adjfactor_chunk*.jsonl`; `data/silver/adj_factor/phase0_top1000_sip_adjfactor_20160104_20260416_chunk*.jsonl` |
| SPY benchmark bars | `data/silver/benchmark_index/fmf_validation_etf_bootstrap.jsonl` |
| SPY adjustment factors | `data/silver/adj_factor/fmf_validation_etf_bootstrap.jsonl` |

Required Alpha SOTA comparison artifact:

`artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6_sota_robustness_20260507/phase6_sota_backtest_records.csv`

## 10. Reproduction Commands

Compile-check the Phase7K script:

```powershell
python -m py_compile src\stockmachine\apps\run_pure_alpha_phase7k.py
```

Run the primary Phase7K experiment:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase7k `
  --turnover-budget-grid 0.15 `
  --cost-bps-per-side 4 `
  --output-root artifacts\strategy_projects\us_equities_pure_alpha_h5\research\phase7k_locked_turnover_shared_capital_20260518_tb0p15
```

Expected primary outputs:

| output | path |
|---|---|
| rollup | `phase7k_rollup.json` |
| strict curve | `phase7k_strict_daily_curve.csv` |
| strict metrics | `phase7k_strict_daily_metrics.csv` |
| turnover summary | `phase7k_turnover_summary.csv` |
| positions | `phase7k_locked_positions.csv.gz` |
| diagnostics | `phase7k_daily_diagnostics.csv` |
| factor exposure | `phase7k_factor_exposure_summary.csv` |
| memo | `phase7k_locked_turnover_memo.md` |

The rollup should report:

```json
{
  "panel": 923359,
  "positions": 416359,
  "diagnostics": 3879,
  "skipped": 0,
  "curve": 3869,
  "strict_metrics": 6
}
```

Optional turnover-grid check:

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase7k `
  --turnover-budget-grid 0.10,0.20 `
  --cost-bps-per-side 4 `
  --output-root artifacts\strategy_projects\us_equities_pure_alpha_h5\research\phase7k_locked_turnover_shared_capital_20260518_tb0p10_tb0p20
```

Observed net Sharpe across turnover budgets:

| portfolio | tb0.10 | tb0.15 | tb0.20 |
|---|---:|---:|---:|
| shared core | 1.68 | 1.97 | 1.67 |
| no momentum | 1.67 | 1.62 | 1.53 |
| slow core | 1.02 | 1.05 | 0.97 |

## 11. How To Rebuild The Comparison Plot

After the primary Phase7K run and the Phase6 SOTA artifact exist, rebuild the
comparison plot from:

- `phase7k_strict_daily_curve.csv`
- `phase6_sota_backtest_records.csv`

The chart in this report was saved to:

`artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7k_shared_core_vs_sota_spy_20260518/shared_core_vs_alpha_sota_spy_equity_drawdown.png`

The corresponding machine-readable outputs are:

- `shared_core_vs_alpha_sota_spy_curves.csv`
- `shared_core_vs_alpha_sota_spy_summary.csv`
- `shared_core_vs_alpha_sota_spy_newey_west_t.csv`
- `shared_core_vs_alpha_sota_spy_pairwise_newey_west_t.csv`

Reproduction rule for the comparison:

```text
shared_core_return = Phase7K net_return
alpha_sota_return = Phase6 gross_return - Phase6 turnover * 4 / 10000
spy_return = benchmark open-to-open return
```

All three curves are aligned by date and rebased to `1.0` on the common window.

## 12. Caveats And Open Risks

1. This is validation-only. It should not be treated as final production
   evidence.
2. The test lockbox is not used here.
3. Realized beta is still non-trivial. For shared core, realized beta to SPY is
   about `0.15` even though the optimizer enforces hard ex-ante beta neutrality.
4. Costs include turnover cost only. There is no borrow cost, financing cost,
   spread model, market impact model, or capacity model.
5. The data lake is not final survivorship-bias-free full-market data.
6. Pairwise NW t versus Alpha SOTA is positive but not decisive.
7. The strategy was developed after reviewing validation behavior, so further
   robustness checks are needed before promoting it.

## 13. Recommended Next Checks

1. Re-run Phase7K on a truly frozen test window only once a promotion decision is
   ready.
2. Add borrow and financing stress similar to Phase6 robustness.
3. Audit realized beta drift by year and by rolling 60/126/252-session windows.
4. Run cost stress at `2`, `4`, `8`, and `10 bps/side`.
5. Add capacity checks using ADV participation and short borrow availability.
6. Compare against Alpha SOTA on yearly and quarterly windows, not only the full
   sample.
7. Freeze the exact code and input artifact manifest before any further tuning.

## 14. Bottom Line

Phase7K is a meaningful improvement candidate. The mechanism is economically
plausible: shared capital captures multi-factor concurrence, while reason-level
minimum holds and hard turnover budgets reduce overtrading. The validation
curve is materially stronger than the prior alpha SOTA after equalized
`4 bps/side` transaction costs.

The result is strong enough to justify a formal robustness packet, but not yet
strong enough to call the strategy production-ready.
