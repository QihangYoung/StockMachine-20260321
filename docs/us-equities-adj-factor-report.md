# US Equities Adjustment-Factor Report

## Run Date

2026-03-21

## Goal

Upgrade the US equities silver-chain prototype so that:

- Yahoo bootstrap writes a canonical `adj_factor` table
- research labels use adjusted open-to-open returns
- the backtest engine uses `adj_open` when available
- Yahoo bootstrap retries missing symbols instead of silently writing broken data

## What Changed

- `bootstrap_yahoo_us_equities.py` now writes `adj_factor/yahoo_bootstrap.jsonl`
- Yahoo bootstrap now retries missing tickers one by one and falls back to
  `Ticker.history()` for stubborn symbols such as `AVGO`
- `load_us_equities_dataset()` now loads `adj_factor`
- `build_price_panel_from_silver()` now merges `price_adjust_factor`,
  `cash_dividend`, and `split_factor`
- `build_research_frame()` keeps features on raw prices but moves labels to
  adjusted `adj_open`
- `DailyOpenHoldBacktestEngine` now realizes returns from `adj_open` when the
  field exists

## Validation

Local verification completed with:

```text
python -m compileall src tests
python -m pytest tests/test_research/test_adjusted_prices.py tests/test_backtest/test_simple_engine.py tests/test_domain/test_datasets.py tests/test_backtest/test_protocols.py tests/test_research/test_overlay_helpers.py
```

The targeted pytest set passed: `8 passed`.

## Data Sanity Checks

- `adj_factor` rows: `117,853`
- non-unit adjustment factors: `98,406`
- non-zero dividend rows: `1,494`
- label rows changed versus raw-open labels: `101,311`

This confirms the adjustment path is active and materially changes the research
target, rather than being a no-op.

## 2025 `hist_gbm` Backtest Impact

Old result from the pre-adjustment silver-chain run:

- total return: `17.81%`
- annualized return: `18.78%`
- annualized volatility: `22.42%`
- Sharpe: `0.88`
- max drawdown: `-19.05%`
- mean turnover: `1.30`

New result after `adj_factor + adj_open`:

- total return: `13.88%`
- annualized return: `14.62%`
- annualized volatility: `21.78%`
- Sharpe: `0.74`
- max drawdown: `-20.00%`
- mean turnover: `1.28`

Delta:

- total return: `-3.93 pct`
- annualized return: `-4.16 pct`
- Sharpe: `-0.15`
- max drawdown: `-0.95 pct`
- turnover: `-0.03`

## Interpretation

The earlier silver-chain result was directionally right, but it overstated the
edge. Once dividend and split adjustments are included in the label and
realized-return path, the signal remains alive but clearly weaker.

That is the outcome we want from a validation perspective:

- the strategy did not collapse to noise
- the pipeline now penalizes us for a more realistic return definition
- model rankings and selected positions changed, so the new result is not just
  a cosmetic rewrite

## Current Limitation

This is still a Yahoo-based bootstrap. The `price_adjust_factor` is suitable as
an interim total-return label aid, but it is not a full institutional
point-in-time corporate-action archive. We should still replace this bootstrap
with our own higher-quality ingestion path later.

## Artifacts

- old summary: `artifacts/us_equities_silver_chain_2025_hist_gbm/backtest_summary.csv`
- new summary: `artifacts/us_equities_silver_chain_2025_hist_gbm_adj_v2/backtest_summary.csv`
- new backtest records: `artifacts/us_equities_silver_chain_2025_hist_gbm_adj_v2/backtest_records.csv`
- new predictions: `artifacts/us_equities_silver_chain_2025_hist_gbm_adj_v2/predictions.csv`
