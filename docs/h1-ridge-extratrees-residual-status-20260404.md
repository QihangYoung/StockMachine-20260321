# H1 Ridge / ExtraTrees / Residual Status

Date: 2026-04-04

## Summary

This note summarizes the latest `h1` findings on:

- `ridge`
- `extra_trees`
- `extra_trees` residualized against `ridge`
- whether any of them should be introduced into the current `C policy` core+sleeve product line

The short conclusion is:

- `ridge` and `extra_trees` are highly related on the latest `h1` line.
- `extra_trees` is **not** fully degenerated to a linear model.
- However, most of its economic value appears to overlap with the same core signal family captured by `ridge`.
- The residual component of `extra_trees` still has positive standalone edge, but it does **not** add useful product value when blended into `C policy`.
- Current product priority remains unchanged:
  1. `C + 6% lightgbm_ranker`
  2. `C + 7% extra_trees`
  3. `C only`

## Latest H1 Optimized Pair

The latest `h1` correlation study compares these two optimized configurations:

- `ridge`: `A_topk6_mt0p3_mwc0p02`
- `extra_trees`: `B_topk8_mt0p3_mwc0p03`

Artifacts:

- `E:/CodeX/StockMachine-260321/artifacts/strategy_projects/us_equities_h1/research/feature_v3_compare_20260404/v3/A_topk6_mt0p3_mwc0p02/backtest_summary.csv`
- `E:/CodeX/StockMachine-260321/artifacts/strategy_projects/us_equities_h1/research/model_compare_v3_20260404/extra_trees/B_topk8_mt0p3_mwc0p03/backtest_summary.csv`
- `E:/CodeX/StockMachine-260321/artifacts/strategy_projects/us_equities_h1/research/correlation_ridgeA_vs_extratreesB_20260404/summary.csv`

### Backtest Metrics

| Model | Annualized Return | Volatility | Sharpe | Max Drawdown | Mean Turnover |
|---|---:|---:|---:|---:|---:|
| `ridge_latest_A` | 21.13% | 22.08% | 0.979 | -32.00% | 0.261 |
| `extra_trees_latest_B` | 14.27% | 13.62% | 1.048 | -25.68% | 0.079 |

### Similarity Diagnostics

The latest pair is clearly related, but not identical.

- Backtest net return correlation: `0.841`
- Prediction score Pearson correlation: `0.750`
- Prediction score Spearman correlation: `0.752`
- Daily Top10 mean overlap: `67.7%`
- Ridge vs ExtraTrees mean turnover:
  - `ridge`: `0.261`
  - `extra_trees`: `0.079`

Interpretation:

- The two models are driven by a very similar core signal family.
- But they are not equivalent models:
  - score correlation is high, not near-1
  - holdings are overlapping, not identical
  - turnover behavior is materially different

## Residual Alpha Test

Question:

- After removing the linear component explained by `ridge`, how much independent edge remains in `extra_trees`?

Method:

- For each date cross-section, regress:
  - `extra_score = alpha_t + beta_t * ridge_score + residual`
- Use the residual as the new ranking score
- Re-run the `h1` backtest using the same `extra_trees B` policy settings:
  - `top_k = 8`
  - `max_turnover = 0.3`
  - `min_weight_change = 0.03`
  - `hold_rank_buffer = 2`
  - `entry_rank_buffer = 2`
  - `max_new_names_per_rebalance = 2`

Artifacts:

- `E:/CodeX/StockMachine-260321/artifacts/h1_residual_alpha_ridge_vs_extratrees_20260404/residual_backtest_summary.csv`
- `E:/CodeX/StockMachine-260321/artifacts/h1_residual_alpha_ridge_vs_extratrees_20260404/comparison_summary.csv`
- `E:/CodeX/StockMachine-260321/artifacts/h1_residual_alpha_ridge_vs_extratrees_20260404/diagnostics_summary.json`

### Residual Results

| Strategy | Annualized Return | Volatility | Sharpe | Max Drawdown | Mean Turnover |
|---|---:|---:|---:|---:|---:|
| `ridge_latest_A` | 21.13% | 22.08% | 0.979 | -32.00% | 0.261 |
| `extra_trees_latest_B` | 14.27% | 13.62% | 1.048 | -25.68% | 0.079 |
| `extra_trees_residual_vs_ridge_B_policy` | 11.47% | 22.01% | 0.583 | -20.90% | 0.031 |

### Residual Diagnostics

- Ridge vs ExtraTrees score correlation:
  - Pearson: `0.750`
  - Spearman: `0.752`
- Ridge vs Residual score correlation:
  - Pearson: `~0`
  - Spearman: `0.016`
- Top10 overlap:
  - Ridge vs ExtraTrees: `67.7%`
  - Ridge vs Residual: `25.2%`
  - ExtraTrees vs Residual: `52.2%`

Return-path correlations:

- `ridge` vs `extra_trees`: `0.841`
- `ridge` vs `residual`: `0.408`
- `extra_trees` vs `residual`: `0.432`

Interpretation:

- `extra_trees` does retain independent non-ridge edge.
- Therefore it is not accurate to say the model has completely collapsed into a linear model.
- But the independent component is much weaker than the full model.
- Most of the economic value of `extra_trees` appears to come from the same shared latent signal family already captured by `ridge`.

## C Policy Sleeve Experiments

Question:

- Does `ridge` or the residualized `extra_trees` component improve the current `C policy` core?

Artifacts:

- Ridge sleeve:
  - `E:/CodeX/StockMachine-260321/artifacts/c_policy_h1_ridge_sleeve_20260404/summary_metrics.csv`
  - `E:/CodeX/StockMachine-260321/artifacts/c_policy_h1_ridge_sleeve_risk_matched_20260404/summary_metrics.csv`
- Ridge + residual combo:
  - `E:/CodeX/StockMachine-260321/artifacts/c_policy_ridge_residual_combo_20260404/summary_metrics.csv`
  - `E:/CodeX/StockMachine-260321/artifacts/c_policy_ridge_residual_combo_20260404/candidate_compare.csv`

### Sleeve Results

| Combo | Annualized Return | Volatility | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| `C only` | 11.17% | 6.94% | 1.562 | -9.90% |
| `C + 4% ridge` | 10.91% | 6.77% | 1.564 | -9.85% |
| `C + 1% residual(extra_trees)` | 11.07% | 6.89% | 1.561 | -9.87% |
| `C + 4% ridge + 1% residual` | 10.80% | 6.72% | 1.562 | -9.81% |

### Interpretation

- `ridge` has slightly lower correlation to `C` and can lift Sharpe marginally.
- But the improvement is economically tiny.
- The residualized `extra_trees` sleeve does not improve product quality.
- Combining `ridge` and `residual` also fails to beat the already-best non-ridge sleeve candidates.

## Product-Level Conclusion

These findings imply:

1. `extra_trees` should not be described as "linearized" or "fully degenerated to ridge".
2. The residual part of `extra_trees` is real, but it is not strong enough to justify its own sleeve role in the current `C policy` product.
3. `ridge` is interesting as an `h1` research result, but it is not currently competitive enough as a `C policy` sleeve candidate.
4. Current product ranking remains:
   1. `C + 6% lightgbm_ranker`
   2. `C + 7% extra_trees`
   3. `C only`

## Recommended Next Step

Do not move `ridge` or `extra_trees residual` into the main product line for now.

If this line is revisited later, the more promising questions would be:

- whether `ridge` should be evaluated inside a truly daily core+sleeve engine instead of only on `C`'s 5-session rebalance nodes
- whether `ridge` is useful as a research benchmark for "shared linear alpha" rather than a production sleeve
