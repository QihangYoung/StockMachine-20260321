# Phase6R Strict-Residual Complexity Ladder

Date: 2026-05-11

Scope: validation only. Test lockbox remains unused.

## Question

Phase6Q used stricter anti-overfit controls and weaker models. Strict residual performance declined versus Phase6P. We need to separate two explanations:

- underfitting: the weaker model is too conservative and cannot capture real nonlinear signal;
- weak-signal overfitting: stronger models fit training noise, while out-of-sample strict residual payoff remains thin.

Phase6R keeps Phase6Q's sampling discipline fixed and varies only model complexity.

## Controls Held Fixed

- Label: `true_size_style_sic2_residual`
- Pairs per date: 700
- Per-stock pair appearance cap: 6
- Purged gap: 10 sessions
- Max train pairs per fold: 350,000
- Folds: 2017, 2018, 2019 walk-forward validation years
- Test lockbox: unused

## Result

| Fold | Model | Train AUC | OOS AUC | Train-OOS gap | Rank IC | Spread | Positive rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2017 | tiny | 0.5185 | 0.5087 | 0.0098 | 0.0082 | 5.08 bps | 57.0% |
| 2017 | shallow | 0.5291 | 0.5106 | 0.0185 | 0.0118 | 10.27 bps | 57.0% |
| 2017 | medium | 0.5473 | 0.5081 | 0.0392 | 0.0098 | 10.85 bps | 53.4% |
| 2017 | Phase6P-like | 0.6188 | 0.5058 | 0.1130 | 0.0053 | 8.59 bps | 51.4% |
| 2018 | tiny | 0.5181 | 0.5196 | -0.0015 | 0.0248 | 6.02 bps | 55.8% |
| 2018 | shallow | 0.5261 | 0.5172 | 0.0089 | 0.0231 | 5.87 bps | 52.6% |
| 2018 | medium | 0.5429 | 0.5160 | 0.0270 | 0.0204 | 4.18 bps | 51.8% |
| 2018 | Phase6P-like | 0.6183 | 0.5127 | 0.1056 | 0.0163 | 4.30 bps | 51.8% |
| 2019 | tiny | 0.5184 | 0.5034 | 0.0150 | 0.0101 | -3.16 bps | 49.0% |
| 2019 | shallow | 0.5266 | 0.5028 | 0.0239 | 0.0102 | -2.18 bps | 50.2% |
| 2019 | medium | 0.5406 | 0.5043 | 0.0363 | 0.0121 | 0.66 bps | 47.3% |
| 2019 | Phase6P-like | 0.6158 | 0.5046 | 0.1112 | 0.0123 | 3.55 bps | 49.8% |

Average by model:

| Model | Avg train AUC | Avg OOS AUC | Avg gap | Avg spread |
| --- | ---: | ---: | ---: | ---: |
| tiny | 0.5183 | 0.5106 | 0.0078 | 2.65 bps |
| shallow | 0.5273 | 0.5102 | 0.0171 | 4.65 bps |
| medium | 0.5436 | 0.5095 | 0.0341 | 5.23 bps |
| Phase6P-like | 0.6176 | 0.5077 | 0.1099 | 5.48 bps |

## Interpretation

There is some underfitting in the weakest models: 2019 strict spread improves from negative to positive as capacity increases.

But the larger conclusion is still weak-signal overfitting, not a clean underfitting story. The Phase6P-like model creates a very large train-OOS AUC gap while OOS AUC stays close to 0.505 and average strict residual spread improves only modestly.

This means the strict residual price-only signal exists, but it is thin. Stronger models can extract a little more OOS spread in some folds, but most of the added model capacity is spent fitting in-sample structure that does not reliably transfer.

## Decision

Do not promote the pairwise strict-residual model to SOTA.

For product work, keep the current SOTA as the operational baseline. For research, pairwise ML remains useful as a diagnostic framework, but the next improvement likely needs better input information or a more explicitly regime-conditioned label design rather than simply larger price-only models.

Artifact plot:

- `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6r_pairwise_complexity_ladder_20260511/phase6r_complexity_ladder_plot.png`
