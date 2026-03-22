# P0 Research Rigor Report

This note summarizes the first P0 research-rigor hardening pass completed on
2026-03-22.

## What Changed

- Added a frozen research contract in [research-protocol.md](./research-protocol.md).
- Replaced the ad-hoc monthly training loop with a formal walk-forward splitter:
  - train window: `36 months`
  - validation window: `6 months`
  - test window: `6 months`
  - roll frequency: `monthly`
  - purge window: `6 sessions`
  - embargo window: `1 session`
- Added point-in-time universe helpers in
  [universe.py](/E:/CodeX/StockMachine-260321/src/stockmachine/research/universe.py).
- Updated the silver-chain backtest and paper signal path to consume date-aware
  metadata.
- Added protocol output to the model sweep runner so each rerun now writes
  `research_protocol.json`.

## Data Gap Found

The original silver metadata tables only had late snapshots:

- `symbol_master`: `2026-03-21`
- `industry_membership`: `2025-12-30`

Under a strict point-in-time contract, that meant the historical research frame
before those dates collapsed to an empty set.

To make the contract executable on the current fixed US-equities research
universe, a repeatable bootstrap helper was added:

- [bootstrap_yahoo_us_equities.py](/E:/CodeX/StockMachine-260321/src/stockmachine/ingestion/jobs/bootstrap_yahoo_us_equities.py)

The helper materializes static metadata snapshots across every observed trading
session in silver. This removes future-dated metadata leakage, but it is still a
bootstrap compromise and not a true historical constituent history.

## Strict-Rerun Results

Artifacts:

- [summary_metrics.csv](/E:/CodeX/StockMachine-260321/artifacts/p0_rigor_rerun/summary_metrics.csv)
- [research_protocol.json](/E:/CodeX/StockMachine-260321/artifacts/p0_rigor_rerun/research_protocol.json)

Current strict rerun uses:

- `predict_start = 2025-01-01`
- point-in-time metadata join
- formal walk-forward splitter
- purge + embargo
- sector-neutral overlay
- `10 bps/side` costs

Results:

| Model | Total Return | Excess vs SPY | Annualized Return | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hist_gbm` | `23.25%` | `5.78%` | `20.30%` | `1.02` | `-20.18%` |
| `lightgbm_regressor` | `20.69%` | `3.22%` | `18.09%` | `0.92` | `-22.72%` |
| `ensemble_hist_gbm_random_forest_rank` | `18.76%` | `1.30%` | `16.42%` | `0.86` | `-18.74%` |
| `ensemble_random_forest_lightgbm_regressor_rank` | `17.00%` | `-0.47%` | `14.89%` | `0.82` | `-19.58%` |
| `factor_baseline` | `9.22%` | `-8.24%` | `8.11%` | `0.54` | `-14.86%` |
| `random_forest` | `10.08%` | `-7.39%` | `8.86%` | `0.52` | `-21.08%` |

Reference benchmark:

- `SPY total return = 17.46%`

## Main Takeaways

- The stricter protocol materially changed the leaderboard.
- `hist_gbm` is the strongest model under the current rigorous rerun.
- `lightgbm_regressor` remains competitive and is the second-best single model.
- `random_forest` lost most of its previous edge once the stricter walk-forward
  and point-in-time rules were enforced.
- The newer ensemble winners from the looser protocol still look usable, but
  they no longer dominate.

## Remaining Risks

- The metadata history is currently a static bootstrap expanded across session
  dates; this still carries survivorship and static-classification risk.
- We still do not have a true historical universe-membership source.
- Statistical significance checks are not part of this P0 pass yet.
- The silver loader still emits a pandas `FutureWarning` during concatenation.

## Recommended Next Step

Move to P1 rigor work:

- yearly / rolling stability analysis
- cost stress tests
- parameter robustness sweeps
- unified rerun leaderboard for the current top models
