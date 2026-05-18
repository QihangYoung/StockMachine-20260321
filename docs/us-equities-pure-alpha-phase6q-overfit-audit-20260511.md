# Phase6Q Pairwise Ranker Overfit Audit

Date: 2026-05-11

Scope: validation only. Test lockbox remains unused.

## Purpose

Phase6P produced a large train top-bottom residual spread but weak 2019 validation spread. Phase6Q audits whether the sparse pairwise nonlinear ranker survives stricter anti-overfit controls:

- purged walk-forward yearly folds for 2017, 2018, and 2019;
- weaker LightGBM models (`tiny_lgbm`, `shallow_lgbm`);
- capped sparse pair sampling with at most 700 pairs per date;
- per-stock pair appearance cap of 6 per date;
- max 350,000 train pairs per fold/model/label;
- label ladder from beta residual to stricter multi-factor/style residual.

## Key Result

The strict style-residual label does not retain enough out-of-sample ranking spread.

| Label | Fold | Model | Rank IC | Top-bottom spread | Positive rate |
| --- | --- | --- | ---: | ---: | ---: |
| strict style residual | 2017 | tiny_lgbm | 0.0084 | 3.48 bps | 55.4% |
| strict style residual | 2017 | shallow_lgbm | 0.0128 | 12.17 bps | 57.8% |
| strict style residual | 2018 | tiny_lgbm | 0.0245 | 4.50 bps | 54.2% |
| strict style residual | 2018 | shallow_lgbm | 0.0227 | 1.76 bps | 52.6% |
| strict style residual | 2019 | tiny_lgbm | 0.0103 | -2.22 bps | 47.7% |
| strict style residual | 2019 | shallow_lgbm | 0.0107 | -0.19 bps | 50.2% |

This is not a product-ready pure-alpha selector. The model can learn some ordering information, but the top-bottom economic spread after strict style neutralization is essentially gone by 2019.

## Label Ladder

The same price-state features look much stronger when the target is a looser residual label.

| 2019 label | Model | Rank IC | Top-bottom spread | Positive rate |
| --- | --- | ---: | ---: | ---: |
| beta residual | tiny_lgbm | 0.0647 | 72.70 bps | 64.3% |
| beta residual | shallow_lgbm | 0.0698 | 81.23 bps | 63.9% |
| beta + size + liquidity + SIC2 residual | tiny_lgbm | 0.0187 | 17.84 bps | 56.0% |
| beta + size + liquidity + SIC2 residual | shallow_lgbm | 0.0189 | 14.49 bps | 58.1% |
| strict style residual | tiny_lgbm | 0.0103 | -2.22 bps | 47.7% |
| strict style residual | shallow_lgbm | 0.0107 | -0.19 bps | 50.2% |

Interpretation: price-only nonlinear ML is still mostly learning known price-style payoff. Once beta, size, liquidity, sector, and reversal/momentum-style effects are removed from the label, the remaining residual-alpha signal is too small and unstable.

## Baseline Cross-Check

In 2019, simple anti-momentum baselines also collapse under the strict style residual label:

| 2019 label | Score | Rank IC | Top-bottom spread | Positive rate |
| --- | --- | ---: | ---: | ---: |
| beta residual | anti momentum 20d | 0.0588 | 73.00 bps | 61.4% |
| beta + size + liquidity + SIC2 residual | anti momentum 20d | 0.0321 | 41.51 bps | 61.8% |
| strict style residual | anti momentum 20d | -0.0013 | -0.29 bps | 49.4% |
| strict style residual | anti beta-residual momentum 20d | -0.0011 | 0.67 bps | 51.5% |
| strict style residual | reversal 5d | 0.0041 | 2.68 bps | 56.8% |

This supports the same diagnosis: the attractive beta-residual result is not yet pure alpha; it is largely price-style payoff.

## Research Implication

The sparse pairwise ML architecture is not the bottleneck yet. The bottleneck is input information.

Recommended next step:

- Keep the current SOTA as the operational baseline, not the pairwise ML model.
- Do not connect Phase6P/6Q to the optimizer as a candidate production selector yet.
- Ask the non-price line for features that can explain residual winner/loser behavior after beta, size, liquidity, sector, and reversal/momentum effects are neutralized.
- If price-only work continues, treat it as a label-design and regime-conditioning research line, not as an immediate product-upgrade path.

Artifacts:

- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6q_sparse_pairwise_overfit_audit_20260511/phase6q_sparse_pairwise_overfit_audit_memo.md`
- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6q_sparse_pairwise_overfit_audit_20260511/phase6q_oos_spread_audit.png`
