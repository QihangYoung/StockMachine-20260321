# P1 Research Rigor Report

This note summarizes the first P1 rigor pass completed on 2026-03-22.

Artifacts:

- [strict full summary](/E:/CodeX/StockMachine-260321/artifacts/p1_rigor_suite/strict_full/summary_metrics.csv)
- [yearly stability](/E:/CodeX/StockMachine-260321/artifacts/p1_rigor_suite/stability/yearly_summary.csv)
- [quarterly stability](/E:/CodeX/StockMachine-260321/artifacts/p1_rigor_suite/stability/quarterly_summary.csv)
- [cost stress summary](/E:/CodeX/StockMachine-260321/artifacts/p1_rigor_suite/cost_stress/summary_metrics.csv)
- [top-k sweep summary](/E:/CodeX/StockMachine-260321/artifacts/p1_rigor_suite/topk_sweep/summary_metrics.csv)

## What This Pass Adds

P1 extends the P0 strict rerun workflow with four follow-on checks:

- full-model strict rerun over every registered alpha expert
- yearly and quarterly stability slices for the current leaders
- cost stress revaluation at `10 / 20 / 40 / 60 bps per side`
- small `top_k` robustness sweeps

The entrypoint is:

- [run_p1_rigor_suite.py](/E:/CodeX/StockMachine-260321/src/stockmachine/apps/run_p1_rigor_suite.py)

The reusable helpers live in:

- [p1_rigor.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/p1_rigor.py)

## Full Strict Leaderboard

Current run settings:

- `predict_start = 2025-01-01`
- current P0 strict protocol
- all `23` registered alpha experts
- `top_k = 10`

Top models by Sharpe:

| Model | Total Return | Excess vs SPY | Annualized Return | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ensemble_hist_gbm_random_forest_lightgbm_regressor_rank` | `29.90%` | `12.43%` | `26.02%` | `1.28` | `-17.60%` |
| `ensemble_hist_gbm_lightgbm_regressor_rank` | `30.48%` | `13.02%` | `26.52%` | `1.27` | `-19.05%` |
| `ensemble_random_forest_lightgbm_regressor_mean` | `24.90%` | `7.44%` | `21.73%` | `1.09` | `-18.99%` |
| `ensemble_hist_gbm_random_forest_lightgbm_regressor_mean` | `24.75%` | `7.29%` | `21.60%` | `1.08` | `-18.05%` |
| `hist_gbm` | `23.25%` | `5.78%` | `20.30%` | `1.02` | `-20.18%` |
| `ensemble_hist_gbm_lightgbm_regressor_mean` | `22.38%` | `4.91%` | `19.55%` | `1.01` | `-20.60%` |

Reference benchmark:

- `SPY total return = 17.46%`

Main change from P0:

- the best strict models are no longer the two-model leaders from the earlier
  reruns
- a three-model `rank-average` ensemble now sits at the top of the strict
  leaderboard

## Stability Readout

This pass sliced the top three strict models by year and quarter.

Models analyzed:

- `ensemble_hist_gbm_random_forest_lightgbm_regressor_rank`
- `ensemble_hist_gbm_lightgbm_regressor_rank`
- `ensemble_random_forest_lightgbm_regressor_mean`

Yearly summary:

| Model | Year | Sessions | Total Return | Excess vs SPY | Sharpe | Max Drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ensemble_hist_gbm_random_forest_lightgbm_regressor_rank` | `2025` | `50` | `23.57%` | `6.73%` | `1.13` | `-17.60%` |
| `ensemble_hist_gbm_random_forest_lightgbm_regressor_rank` | `2026` | `7` | `5.12%` | `4.59%` | `3.67` | `-1.86%` |
| `ensemble_hist_gbm_lightgbm_regressor_rank` | `2025` | `50` | `22.13%` | `5.28%` | `1.06` | `-19.05%` |
| `ensemble_hist_gbm_lightgbm_regressor_rank` | `2026` | `7` | `6.84%` | `6.31%` | `4.84` | `-1.43%` |
| `ensemble_random_forest_lightgbm_regressor_mean` | `2025` | `50` | `22.07%` | `5.22%` | `1.06` | `-18.99%` |
| `ensemble_random_forest_lightgbm_regressor_mean` | `2026` | `7` | `2.32%` | `1.79%` | `2.00` | `-2.08%` |

Quarterly readout:

- all three models had a weak `2025Q1`
- the strongest rebound came in `2025Q2`
- `2025Q3` and `2025Q4` stayed positive, but excess narrowed for the mean
  ensemble and for one of the rank ensembles

That pattern suggests the new leaders are not just one-quarter spikes, but they
still rely on a strong mid-2025 regime.

## Cost Stress

The top three strict models were revalued under higher transaction costs.

At `10 bps/side`:

- `ensemble_hist_gbm_random_forest_lightgbm_regressor_rank`: `12.43%` excess
- `ensemble_hist_gbm_lightgbm_regressor_rank`: `13.02%` excess
- `ensemble_random_forest_lightgbm_regressor_mean`: `7.44%` excess

At `20 bps/side`:

- `ensemble_hist_gbm_random_forest_lightgbm_regressor_rank`: `3.08%` excess
- `ensemble_hist_gbm_lightgbm_regressor_rank`: `3.26%` excess
- `ensemble_random_forest_lightgbm_regressor_mean`: `-1.17%` excess

At `40 bps/side`:

- all three fall below zero excess

Takeaway:

- the two rank ensembles still have a modest buffer at `20 bps/side`
- none of the current leaders survive `40 bps/side` as positive-excess
  strategies under the current assumptions

## Top-K Robustness

This pass ran a small `top_k` sweep over the same top three models with:

- `top_k = 5, 10, 15, 20`

Main readout:

- `ensemble_hist_gbm_lightgbm_regressor_rank` is strongest at `top_k = 10`
- `ensemble_hist_gbm_random_forest_lightgbm_regressor_rank` is weak at
  `top_k = 5`, but gets stronger at `15` and `20`
- `ensemble_random_forest_lightgbm_regressor_mean` improves as breadth rises
  and is strongest at `top_k = 20`

This suggests the current strict leaders are not all optimized at the same
portfolio breadth:

- rank ensembles prefer tighter books, but not too tight
- the mean ensemble benefits from broader diversification

## Main Takeaways

- P1 changes the leaderboard again; the strict winners now lean heavily on
  `lightgbm`
- `hist_gbm` remains strong, but it is no longer the top strict model
- three-model and two-model `lightgbm` ensembles now dominate the strict table
- the leading models are still sensitive to costs; `40 bps/side` is enough to
  wipe out excess returns
- `top_k` is an important sensitivity and should not be treated as a fixed
  constant going forward

## Recommended Next Step

Move to the next P1 rigor slice:

- expand stability analysis to all current shortlist models
- add larger parameter grids for `horizon`, liquidity, and volatility filters
- start statistical significance work after the shortlist is frozen
