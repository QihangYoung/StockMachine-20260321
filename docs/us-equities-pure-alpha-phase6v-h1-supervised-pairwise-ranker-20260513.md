# Phase6V H1-Supervised Pairwise Ranker

Date: 2026-05-13

Scope: validation-only diagnostic. The test lockbox is not used.

## Question

Phase6P trained the sparse pairwise ranker on aggregate h10 strict residual
return. Phase6U showed that the validation edge is front-loaded and unstable
across holding days. This raises a natural question: should the supervised
target be h1 residual return instead of h10 residual return?

Phase6V changes only the label horizon:

- Phase6P label: aggregate h10 strict residual return.
- Phase6V label: first holding-day strict residual return.

The model family, features, pair sampling framework, and train/validation
windows are kept the same.

## H1 Label Model Quality

| split | pairwise AUC | h1 label Top20-Bottom20 spread | h1 label positive rate |
| --- | ---: | ---: | ---: |
| train | 0.5498 | 7.71 bps | 67.8% |
| model_validation_2018 | 0.5070 | 2.09 bps | 55.0% |
| final_validation_2019 | 0.5055 | 2.40 bps | 54.8% |

The h1-supervised model has a real in-sample day1 signal, but out-of-sample
pairwise AUC is only slightly above random.

## Day1-10 Forward Profile

Mean Top20% minus Bottom20% by `pairwise_rank_score`, strict daily residual,
bps per holding day.

| split | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | day1-10 sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 7.35 | 2.56 | 1.80 | 1.45 | 1.42 | 0.24 | 1.60 | 0.38 | 0.58 | -0.19 | 17.18 |
| model_validation_2018 | 1.87 | 1.76 | -0.62 | 0.94 | 0.69 | 1.02 | -2.15 | -0.85 | -0.55 | -0.14 | 1.98 |
| final_validation_2019 | 2.55 | 0.86 | 0.27 | 0.68 | -0.66 | 0.50 | -0.18 | -1.13 | 0.49 | -1.61 | 1.78 |

## Comparison Against H10-Supervised Phase6P

Same Top20-Bottom20 strict daily residual diagnostic.

| model | train day1 | train sum | 2018 day1 | 2018 sum | 2019 day1 | 2019 sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase6P h10-supervised | 4.51 | 39.57 | 2.44 | 6.03 | 2.73 | 10.75 |
| Phase6V h1-supervised | 7.35 | 17.18 | 1.87 | 1.98 | 2.55 | 1.78 |

H1 supervision improves in-sample day1 strength, but does not improve validation
day1. It also destroys most of the post-day1 residual profile.

## Interpretation

H1 residual return is probably too noisy to be a better supervision target for
this model and feature set. The h10 target, although slower, appears to act as
a denoised label for a broader price/style payoff. Once we train directly on
h1, the model chases short-horizon noise: train day1 jumps, but validation
barely improves and cumulative day1-10 validation performance deteriorates.

This result does not prove h1 labels are useless in general. It says that a
plain replacement of h10 with h1 inside the current sparse pairwise framework
is not an improvement.

## Practical Takeaway

Do not switch the Phase6P pairwise ranker to h1 supervision as-is.

Better variants would be:

- Use a multi-horizon target, for example a weighted blend of day1-day4 strict
  residuals, instead of pure h1.
- Train horizon-specific auxiliary heads but keep selection based on a
  stabilized ensemble.
- Add stronger regularization or simpler linear/ridge models for h1, because
  the tree model has enough flexibility to overfit short-horizon noise.

## Artifacts

- H1 model output: `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6v_h1_sparse_pairwise_ranker_20260513`
- H1 day1-10 profile: `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6v_h1_daily_horizon_profile_20260513`
- H1 model memo: `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6v_h1_sparse_pairwise_ranker_20260513/phase6v_h1_sparse_pairwise_ranker_memo.md`
