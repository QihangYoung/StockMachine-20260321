# P1 Regime-Window Strict Rerun Report

This report summarizes the wider strict backtest rerun that extends the held-out
evaluation window to cover more bull and bear market regimes.

Reference output root:

- [p1_rigor_suite_regime_window](/E:/CodeX/StockMachine-260321/artifacts/p1_rigor_suite_regime_window)

## Scope

- dataset coverage after safe history extension: `2014-01-02` to `2026-03-20`
- prediction start: `2018-01-01`
- frequency: daily
- benchmark: `SPY`
- holding horizon: `5` sessions
- portfolio mode: long-only, equal-weight, `top_k=10`
- strict protocol:
  - point-in-time metadata and explicit universe loading
  - `36m` train / `6m` validation / `6m` test
  - monthly walk-forward
  - purge `6`
  - embargo `1`
  - sector-neutral overlay
  - `10 bps/side` turnover-based transaction cost

Primary protocol references:

- [research-protocol.md](/E:/CodeX/StockMachine-260321/docs/research-protocol.md)
- [historical-universe-contract.md](/E:/CodeX/StockMachine-260321/docs/historical-universe-contract.md)
- [regime-coverage-backtest.md](/E:/CodeX/StockMachine-260321/docs/regime-coverage-backtest.md)

## Coverage

The strict full rerun completed all `25` registered models.

Main outputs:

- [strict_full/summary_metrics.csv](/E:/CodeX/StockMachine-260321/artifacts/p1_rigor_suite_regime_window/strict_full/summary_metrics.csv)
- [stability/yearly_summary.csv](/E:/CodeX/StockMachine-260321/artifacts/p1_rigor_suite_regime_window/stability/yearly_summary.csv)
- [stability/quarterly_summary.csv](/E:/CodeX/StockMachine-260321/artifacts/p1_rigor_suite_regime_window/stability/quarterly_summary.csv)
- [cost_stress/summary_metrics.csv](/E:/CodeX/StockMachine-260321/artifacts/p1_rigor_suite_regime_window/cost_stress/summary_metrics.csv)
- [topk_sweep/summary_metrics.csv](/E:/CodeX/StockMachine-260321/artifacts/p1_rigor_suite_regime_window/topk_sweep/summary_metrics.csv)

Benchmark reference:

- `SPY total return = 159.16%`

## Top Leaderboard

Ranked by Sharpe:

| Rank | Model | Total Return | Excess vs SPY | Annualized Return | Sharpe | Max Drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `extra_trees` | `338.53%` | `179.36%` | `20.03%` | `0.93` | `-29.88%` |
| 2 | `ensemble_hist_gbm_random_forest_mean` | `257.66%` | `98.50%` | `17.05%` | `0.85` | `-27.09%` |
| 3 | `ensemble_hist_gbm_ridge_rank` | `268.15%` | `108.98%` | `17.47%` | `0.85` | `-27.54%` |
| 4 | `ensemble_hist_gbm_random_forest_rank` | `227.40%` | `68.24%` | `15.78%` | `0.82` | `-27.77%` |
| 5 | `ensemble_hist_gbm_random_forest_lightgbm_regressor_rank` | `223.40%` | `64.24%` | `15.60%` | `0.80` | `-27.30%` |
| 6 | `hist_gbm` | `212.28%` | `53.12%` | `15.10%` | `0.78` | `-30.15%` |

Notable findings:

- `extra_trees` becomes the most stable top model in the wider regime window.
- `lightgbm_ranker` produces the largest absolute return in the full table, but
  not the best Sharpe because volatility and drawdown are materially higher.
- the previous strict short-window winner,
  `ensemble_hist_gbm_random_forest_lightgbm_regressor_rank`, remains strong but
  no longer leads the wider regime leaderboard.
- `lstm_regressor` remains competitive in the broader window, while
  `transformer_regressor` improves versus the short window but still sits below
  the first tier.

## Return Curves

Top-5 cumulative return curves:

![Top 5 Regime-Window Return Curves](/E:/CodeX/StockMachine-260321/artifacts/p1_rigor_suite_regime_window/return_curves_top5_final.svg)

## Cost Stress

The default suite runs cost stress for the current top `3` models.

| Model | 10 bps/side Excess | 20 bps/side Excess | 40 bps/side Excess | 60 bps/side Excess |
| --- | ---: | ---: | ---: | ---: |
| `extra_trees` | `179.36%` | `47.96%` | `-108.69%` | `-185.55%` |
| `ensemble_hist_gbm_random_forest_mean` | `98.50%` | `-29.54%` | `-164.66%` | `-220.36%` |
| `ensemble_hist_gbm_ridge_rank` | `108.98%` | `-33.67%` | `-174.73%` | `-227.63%` |

Interpretation:

- `extra_trees` is clearly the most cost-resilient of the current top three.
- all three strategies break under sufficiently heavy costs.
- the two ensemble leaders are still attractive in the default `10 bps/side`
  setting, but their margin of safety versus realistic execution friction is
  not especially thick.

## `top_k` Robustness

The default suite runs a `top_k` scan over `{5, 10, 15, 20}` for the current
top `3` models.

| Model | Best `top_k` By Sharpe | Notes |
| --- | --- | --- |
| `extra_trees` | `10` | Best Sharpe at `10`; `15` and `20` remain decent but weaker. |
| `ensemble_hist_gbm_random_forest_mean` | `10` | `5` improves total return, but `10` remains the best risk-adjusted setting. |
| `ensemble_hist_gbm_ridge_rank` | `10` | Very clear drop-off once expanded to `15` or `20`. |

Interpretation:

- the current default `top_k=10` remains a reasonable setting in the broader
  regime window.
- the broader-window result does not support a blanket move to deeper books.

## Conclusions

1. Expanding the historical window changes the leaderboard meaningfully.
   `extra_trees` is now the most stable strategy across the broader regime set.
2. Some ensemble strategies remain strong, but the short-window winner is no
   longer the clear long-window winner.
3. Cost sensitivity remains one of the main remaining realism risks.
4. The wider test window strengthens confidence that the surviving top models
   are not just artifacts of the post-2023 regime.

## Suggested Next Steps

1. promote `extra_trees` into the current paper shortlist
2. run paper-vs-backtest attribution on `extra_trees` and the best ensemble
3. move the cost model beyond flat `bps` assumptions
4. continue the historical-universe work so the wider-window result is backed
   by true constituent-history data
