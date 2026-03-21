# US Equities Risk-Aware Report

## Run Date

2026-03-21

## Scope

This report upgrades the first baseline into a more strategy-like workflow.

- market: US equities
- data source: Yahoo Finance via `yfinance`
- universe: 66 current large-cap US stocks plus `SPY`
- history window: 2019-01-01 to 2025-12-31
- validation period: 2024
- test period: 2025
- label: next 5-session excess return from next open
- rebalance frequency for portfolio metrics: every 5 sessions

## Overlay Rules

- sector neutralization: enabled
- minimum close price: `10 USD`
- minimum 20-day median dollar volume: `50M USD`
- maximum 20-day realized volatility: `4%`
- maximum positions per sector: `2`
- trading cost assumption: `10 bps` per side

## Test-Period Comparison 2025

### Raw Model Results

| Model | Raw Total Return | Raw Annualized Return | Raw Sharpe | Benchmark Total Return |
| --- | ---: | ---: | ---: | ---: |
| factor_baseline | 0.1298 | 0.1338 | 0.6911 | 0.1755 |
| ridge | 0.3812 | 0.3941 | 1.2770 | 0.1755 |
| hist_gbm | 0.4048 | 0.4185 | 1.4826 | 0.1755 |

### After Overlay And Costs

| Model | Net Total Return | Net Annualized Return | Net Sharpe | Annualized Net Excess | Mean Turnover | Mean Cost bps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| factor_baseline | 0.0372 | 0.0383 | 0.2952 | -0.1427 | 0.8857 | 8.86 |
| ridge | 0.1564 | 0.1613 | 0.7078 | -0.0197 | 1.1429 | 11.43 |
| hist_gbm | 0.1895 | 0.1954 | 0.9166 | 0.0144 | 1.3061 | 13.06 |

## Main Findings

- the overlay compresses performance materially, which is exactly what we want
  from a realism check
- `hist_gbm` remains the strongest model after sector neutralization, sector
  caps, and trading costs
- `ridge` still retains signal, but its 2025 net annualized excess return turns
  slightly negative after the overlay
- `factor_baseline` does not survive the stronger out-of-sample test period

## Interpretation

The earlier raw backtest was picking up useful information, but part of the
headline return was coming from concentrated sector bets and unconstrained
turnover. Once we force the strategy to spread risk and pay trading costs, the
edge becomes much smaller.

That is a good sign for the research process:

- the raw result was not entirely fake, because `hist_gbm` still stays positive
- the raw result was too optimistic, because a large fraction of the excess
  return disappears after realistic constraints

## Sector Footprint

The 2025 test-period selections for both `ridge` and `hist_gbm` were still led
most often by:

- Technology
- Healthcare
- Communication Services
- Financial Services

The sector cap held the average maximum sector weight at `20%`, matching the
configured `2 of 10` position limit.

## Artifacts

- raw summary: `artifacts/us_equities_risk_aware/raw_summary_metrics.csv`
- overlay summary: `artifacts/us_equities_risk_aware/overlay_summary_metrics.csv`
- selected positions: `artifacts/us_equities_risk_aware/selected_positions.csv`
- symbol metadata: `artifacts/us_equities_risk_aware/symbol_metadata.csv`
