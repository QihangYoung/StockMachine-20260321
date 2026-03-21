# US Equities Alpaca Migration Report

## Run Date

2026-03-21

## Goal

Move the 2025-plus US equities silver layer away from the temporary Yahoo
bootstrap and toward Alpaca as the primary structured source for:

- `daily_bar`
- `benchmark_index`
- `adj_factor`

## What Was Added

- `research-seed` can now batch-sync the default research universe
- `adj-factors` can now build `adj_factor` from:
  - Alpaca raw bars
  - Alpaca adjusted bars
  - Alpaca corporate actions
- silver-table loading now dedupes by canonical primary key so newer Alpaca
  rows override older Yahoo bootstrap rows
- the Alpaca client now retries timeouts and incomplete reads

## Commands Used

```text
python -m stockmachine.apps.collect_us_equities_v1 research-seed --start 2025-01-01 --end 2025-12-31 --chunk-size 25 --feed sip --adjustment raw --skip-symbol-master
python -m stockmachine.apps.collect_us_equities_v1 research-seed --start 2026-01-01 --end 2026-03-21 --chunk-size 25 --feed iex --adjustment raw --skip-symbol-master
python -m stockmachine.apps.run_us_equities_silver_chain --model hist_gbm --predict-start 2025-01-01 --output-dir artifacts/us_equities_silver_chain_2025_hist_gbm_alpaca_adj
```

## Coverage After Migration

- `daily_bar` Alpaca rows: `20,064`
- `benchmark_index` Alpaca rows: `304`
- `adj_factor` Alpaca rows: `20,368`

Coverage now spans:

- symbols: `66` equities plus `SPY`
- dates: `2025-01-02` through `2026-03-20`

## Feed Split

We used a mixed feed policy:

- `SIP` for the 2025 historical window
- `IEX` for the recent 2026 window

This was necessary because a recent-window `SIP` request returned:

```text
403: subscription does not permit querying recent SIP data
```

That matches Alpaca's plan-based market-data restrictions.

## 2025 Backtest Impact

Using the newer Alpaca-backed `daily_bar + benchmark_index + adj_factor`
combination, the current `hist_gbm` result is:

- sessions: `59`
- total return: `15.60%`
- annualized return: `13.18%`
- annualized volatility: `21.22%`
- Sharpe: `0.69`
- max drawdown: `-21.96%`
- benchmark total return: `15.34%`
- mean turnover: `1.25`

Compared with the earlier Alpaca-bar / older-adjustment mix, this is weaker.
That is a useful result: the more realistic data path is shrinking paper alpha
instead of inflating it.

## Interpretation

The pipeline is now healthier:

- price bars for the active research window come from Alpaca
- adjustment factors are no longer relying only on Yahoo
- newer data correctly overrides older bootstrap rows

The strategy still has signal, but the edge is narrower than earlier,
less-rigorous runs suggested.

## Artifacts

- backtest summary: `artifacts/us_equities_silver_chain_2025_hist_gbm_alpaca_adj/backtest_summary.csv`
- predictions: `artifacts/us_equities_silver_chain_2025_hist_gbm_alpaca_adj/predictions.csv`
