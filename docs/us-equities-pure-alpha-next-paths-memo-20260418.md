# US Equities Pure Alpha Next Paths Memo

Date: 2026-04-18

## Purpose

The pure-alpha line should not move directly into Phase 5. Phase 4G, 4H, and
4I exposed two structural issues:

- the current `top1000_clean_core_beta_full` universe has a positive
  beta-residual prior;
- the current target removes only market beta and leaves daily universe drift
  and style effects in the residual.

This memo defines the next three validation-only paths.

## Path 1: Residual-Target-Aware Short Selector

Goal:

Re-evaluate the short selector under multiple residual targets instead of
optimizing only against `forward_beta_residual_return_5d`.

Targets:

- `beta_residual`;
- `cs_demeaned_beta_residual`;
- `risk_neutral_beta_residual`.

Selector family:

- high `momentum_20d`;
- high `beta_residual_momentum_20d_z`;
- high `vol_adjusted_momentum_20d`;
- high `exhausted_winner_20_5`;
- high `residual_overextension_20_5`;
- transparent composites of the above.

Decision rule:

A short selector is more credible if it works on the beta residual and keeps
its edge on cross-section-demeaned or risk-neutral residuals.

## Path 2: Asymmetric Long/Short Universe Constructor

Goal:

Test whether the long and short books should draw from different candidate
pools.

Default candidates:

- long side: `top500_clean_core_beta_full`, `top1000_clean_core_beta_full`;
- short side: `adv50m_clean_core_beta_full`,
  `adv30m_clean_core_beta_full`, `adv20m_clean_core_beta_full`;
- control: symmetric `top1000_clean_core_beta_full`.

Rationale:

The long side may benefit from the high-quality / high-liquidity top universe,
while the short side may need a broader but still liquid candidate pool.

Guardrails:

- beta matching remains mandatory;
- no test-window performance;
- no borrow-cost claims yet;
- no use of low-liquidity names without explicit capacity checks.

## Path 3: Top2000 Feasibility Audit

Goal:

Determine whether `top2000` can be studied safely. Do not run a top2000
performance experiment until membership is timestamp-safe enough.

Required checks:

- does a `top2000` or broader point-in-time membership artifact exist;
- can it be aligned to the Beta-thread validation window;
- are adjusted prices, volume, beta, and forward labels available;
- are shortability / borrow proxies available or explicitly missing;
- does the resulting universe remain tradable at `<= 500,000 USD`;
- does it avoid look-ahead and survivorship bias.

Decision rule:

If top2000 membership is missing or only current-date derived, the path remains
blocked and the correct near-term proxy is `adv20m` / `adv30m`, not an unsafe
top2000 backtest.

## Recommended Order

1. Run the residual-target-aware short selector.
2. Run asymmetric long/short universe construction using existing safe
   variants.
3. Audit top2000 feasibility and data gaps before any top2000 performance
   claim.

The test lockbox remains closed for all three paths.

## Execution Results

Completed on 2026-04-18:

- Path 1 produced Phase 4J artifacts at
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase4j_residual_target_short_selector_20260418`.
  The best transparent selector was `short_core_plus_overextension`, with
  positive short contribution on `beta_residual`, `cs_demeaned_beta_residual`,
  and `risk_neutral_beta_residual`.
- Path 2 produced Phase 4K artifacts at
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase4k_asymmetric_universe_constructor_20260418`.
  The cleanest first-pass asymmetric pair was
  `top1000_clean_core_beta_full__short_adv30m_clean_core_beta_full`, with mean
  CS-demeaned residual spread `0.001684` and effectively zero net beta, using
  `reversal_5d` on the long side and `short_core_plus_overextension` on the
  short side.
- Path 3 produced Phase 4L artifacts at
  `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase4l_top2000_feasibility_audit_20260418`.
  `top1500`, `top2000`, and `top3000` remain blocked because they are absent
  from the Phase 3 signal panel and Phase 1 marks them as outside the current
  top1000 bootstrap data scope.

Near-term decision:

Do not run a top2000 performance experiment yet. Use `top1000` long plus
`adv30m` short as the safe validation-only proxy while building a timestamp-safe
full-market membership/data layer. Keep the signals asymmetric: `reversal_5d`
for long selection and overextension/residual-momentum features for short
selection.
