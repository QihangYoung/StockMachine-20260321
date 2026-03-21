# US Equities Silver-Chain Report

## Run Date

2026-03-21

## Goal

Move the current best baseline out of the standalone research script and into
the project's formal flow:

- canonical silver tables
- signal contract
- portfolio policy
- execution policy
- backtest engine

## What Was Added

- Yahoo bootstrap into silver tables
- silver-table loader
- `Signal -> TargetPosition -> OrderIntent` backtest path
- minimal daily open-to-open hold engine

## Commands Used

```text
python -m stockmachine.apps.bootstrap_us_equities_silver --start 2019-01-01 --end 2025-12-31
python -m stockmachine.apps.run_us_equities_silver_chain --model hist_gbm --predict-start 2024-01-01
python -m stockmachine.apps.run_us_equities_silver_chain --model hist_gbm --predict-start 2025-01-01
python -m stockmachine.apps.run_us_equities_silver_chain --model ridge --predict-start 2025-01-01
```

## 2024-2025 Combined Backtest

Model: `hist_gbm`

- sessions: `98`
- total return: `34.96%`
- annualized return: `16.67%`
- annualized volatility: `21.24%`
- Sharpe: `0.83`
- max drawdown: `-26.98%`
- benchmark total return: `44.38%`
- mean turnover: `1.28`
- mean cost: `12.78 bps`

## 2025 Only Backtest

### `hist_gbm`

- total return: `17.81%`
- annualized return: `18.78%`
- annualized volatility: `22.42%`
- Sharpe: `0.88`
- max drawdown: `-19.05%`
- benchmark total return: `15.33%`
- mean turnover: `1.30`
- mean cost: `13.04 bps`

### `ridge`

- total return: `13.91%`
- annualized return: `14.65%`
- annualized volatility: `26.04%`
- Sharpe: `0.65`
- max drawdown: `-22.44%`
- benchmark total return: `15.33%`
- mean turnover: `1.13`
- mean cost: `11.33 bps`

## Comparison To The Earlier Overlay Report

The silver-chain results are directionally consistent with the earlier
risk-aware report:

- `hist_gbm` remains stronger than `ridge`
- performance survives costs and sector caps, but the edge is much smaller than
  in the unconstrained raw model
- once the strategy is forced through a formal backtest path, performance still
  looks plausible rather than collapsing completely

The exact numbers differ because:

- the engine computes realized open-to-open returns directly from silver tables
- the 2025-only run uses the formal backtest session schedule rather than the
  previous report's simplified cohort summary
- the final incomplete holding window is naturally excluded by the engine

## Current Limitation

The silver tables were still seeded from Yahoo Finance as a temporary bridge.
So this run validates our architecture and wiring more than it validates final
data quality.

## Artifacts

- combined summary: `artifacts/us_equities_silver_chain/backtest_summary.csv`
- 2025 hist_gbm summary: `artifacts/us_equities_silver_chain_2025_hist_gbm/backtest_summary.csv`
- 2025 ridge summary: `artifacts/us_equities_silver_chain_2025_ridge/backtest_summary.csv`
