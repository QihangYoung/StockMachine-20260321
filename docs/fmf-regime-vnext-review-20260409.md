# FMF Regime vNext Review (2026-04-09)

## Scope

This review covers the first implementation of the `FMF` regime detector vNext:

- multi-signal
- score-based
- smoothed
- still using a thin `equity_total <-> duration` action layer

The lockbox test window remained closed. Everything here is validation-only.

## What Was Built

Implementation files:

- detector:
  - `/E:/CodeX/StockMachine-260321/src/stockmachine/risk/regime.py`
- validation-only runner:
  - `/E:/CodeX/StockMachine-260321/src/stockmachine/apps/run_fmf_validation_regime_vnext_experiment.py`

The detector now uses:

- benchmark trend
- `equity_us vs duration`
- `credit vs duration`
- inflation sleeve momentum
- trend sleeve momentum
- realized volatility
- `equity_us-duration` rolling correlation

These signals are converted into:

- `risk_on_score`
- `defensive_score`
- `net_score`

Then smoothed and mapped into:

- `risk_on`
- `neutral`
- `defensive`

## Validation Result

Common validation window (`2014-08-05 ~ 2019-12-31`):

| Strategy | Annualized Return | Annualized Vol | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| `rolling_fmf_c2_e42_c10_d22_i20_t06` | `4.995%` | `4.201%` | `1.181` | `-5.245%` |
| `rolling_fmf_c2_regime_overlay_v2` | `5.035%` | `4.264%` | `1.173` | `-5.548%` |
| `rolling_fmf_c2_regime_overlay_vnext` | `4.989%` | `4.229%` | `1.172` | `-5.429%` |

Interpretation:

- vNext did run successfully and produced a genuinely different state path
- but it still did **not** beat the static lead
- it also did not clearly improve on overlay v2

So vNext should remain an exploratory branch for now.

## State Distribution

Trading-day state counts from the actual validation backtest:

- `neutral`: `947`
- `risk_on`: `356`
- `defensive`: `311`

This is meaningfully different from overlay v2:

- vNext is much less aggressively `risk_on`
- it spends much more time in `neutral`

That is directionally sensible, but it also means the overlay becomes more conservative most of the time.

## Reference Timeline Comparison

A reference-style regime comparison was also built against the same consensus timeline used for overlay v1/v2 review.

Files:

- timeline:
  - `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_regime_vnext_experiment_20260409/overlay_vs_reference_timeline.png`
- daily summary:
  - `/E:/CodeX/StockMachine-260321/artifacts/fmf_validation_regime_vnext_experiment_20260409/overlay_vs_reference_summary.csv`

Exact daily match ratio is only about `27.5%`.

This sounds worse than v2, but the number needs context:

- the reference timeline has only `risk_on` / `defensive`
- vNext uses a large `neutral` region
- exact-label matching therefore punishes any middle state heavily

So the result should not be read as "vNext is random."

The more useful interpretation is:

- vNext is structurally more cautious
- but its current action layer does not monetize that caution well enough

## Working Conclusion

vNext validates the architecture direction more than the current parameterization.

What we learned:

- a thicker state layer is implementable
- richer signals alone do not automatically improve portfolio performance
- with the current thin action map, the detector still cannot beat the static lead

Current decision:

- keep static lead as the primary candidate
- keep vNext as a research branch
- if the dynamic path continues, the next upgrade should target:
  - better action mapping
  - not another round of detector-only micro-tuning
