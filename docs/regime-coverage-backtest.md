# Regime Coverage Backtest

This note describes the recommended way to expand the current US-equities
research dataset so the test window can cover both bull and bear market
regimes.

## Why Extend The History

The older default silver bootstrap started at `2019-01-01`. That is already
useful, but it compresses the strict test window too heavily toward the recent
post-2023 regime.

To evaluate models across more distinct market environments, we want the test
window to include at least:

- late-cycle 2018 weakness
- the 2020 crash
- the 2020-2021 recovery and bull phase
- the 2022 bear market
- the 2023-2025 recovery and AI-driven bull phase

## Safe Historical Extension Strategy

Do **not** overwrite the current `yahoo_bootstrap.jsonl` with a new
full-history pull, because a fresh Yahoo bootstrap may end up overriding newer
and better Alpaca rows in overlapping periods.

Instead, backfill only the missing historical gap into a separate silver file.

Current recommended gap-fill window:

- `2014-01-01` to `2018-12-31`

This leaves the existing `2019+` silver stack intact while extending the
research window far enough back to support a broader strict test sample.

## One-Step Extension Command

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.extend_us_equities_history --start 2014-01-01 --end 2018-12-31 --silver-file-stem yahoo_bootstrap_2014_2018
```

What this does:

1. downloads the historical gap from Yahoo Finance
2. writes it into separate silver files such as:
   - `data/silver/daily_bar/yahoo_bootstrap_2014_2018.jsonl`
   - `data/silver/adj_factor/yahoo_bootstrap_2014_2018.jsonl`
3. reruns the static metadata backfill so point-in-time loaders see the full
   session history

## Resulting Dataset Coverage

After the current extension pass, the main research tables now cover:

| Table | Min Date | Max Date |
| --- | --- | --- |
| `universe_membership` | `2014-01-02` | `2026-03-20` |
| `daily_bar` | `2014-01-02` | `2026-03-20` |
| `adj_factor` | `2014-01-02` | `2026-03-20` |
| `benchmark_index` | `2014-01-02` | `2026-03-20` |
| `industry_membership` | `2014-01-02` | `2026-03-20` |
| `symbol_master` | `2014-01-02` | `2026-03-21` |

## Recommended Strict Test Window

To make the held-out period cover multiple regimes, the recommended strict test
start is now:

- `predict_start = 2018-01-01`

That still leaves enough earlier history for the current:

- `36 month` train window
- `6 month` validation window
- `6 month` test window

while making the evaluation period broad enough to include both bull and bear
states.

## Recommended Rerun Command

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_p1_rigor_suite --predict-start 2018-01-01 --output-root artifacts/p1_rigor_suite_regime_window
```

## Notes

- This improves regime coverage, but it still does **not** turn the dataset
  into a true historical constituent-history archive.
- `universe_membership` and `industry_membership` remain based on static
  historical backfill, not event-accurate constituent changes.
- If we later add true historical membership data, this workflow should be
  rerun again under that stronger universe contract.
