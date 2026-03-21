# Alpaca Setup

## Goal

Use Alpaca as the first structured US-equities source for:

- `symbol_master` snapshots
- historical daily bars
- paper-trading execution checks
- later live trading integration

## What Alpaca Gives This Project

- Trading API access for paper or live brokerage workflows
- Market Data API access for historical and real-time US equities data
- a cleaner path from `signal -> target -> order -> fill` than the temporary
  Yahoo bootstrap

## Credentials

Create or open your Alpaca dashboard, then generate:

- `ALPACA_API_KEY_ID`
- `ALPACA_API_SECRET_KEY`

For paper trading, the default base URLs in this project are already correct:

- trading: `https://paper-api.alpaca.markets`
- market data: `https://data.alpaca.markets`

If you need overrides, copy [`.env.example`](/E:/CodeX/StockMachine-260321/.env.example)
to a local `.env` file and edit the values.

Optional HTTP tuning:

- `ALPACA_HTTP_TIMEOUT_SECONDS`
- `ALPACA_HTTP_MAX_RETRIES`

## Important Feed Note

The free Basic plan is useful for development, but it is not full-market
coverage. Alpaca's official Market Data docs say:

- Basic equities real-time coverage is `IEX`
- Algo Trader Plus adds all US stock exchanges
- Basic historical access is limited near the latest 15 minutes

So for this project:

- Basic is fine for ingestion wiring, research prototyping, and paper checks
- higher-confidence intraday execution work should later use better coverage

## First Validation Command

Run either after `python -m pip install -e .` or with `PYTHONPATH=src`.

```text
python -m stockmachine.apps.check_alpaca_access --symbol AAPL --benchmark SPY --feed iex
```

Expected outcome:

- asset metadata is returned for `AAPL` and `SPY`
- recent bars are returned for both symbols
- the script prints a JSON summary instead of raising a credential or HTTP error

## First Collection Commands

Once validation passes:

```text
python -m stockmachine.apps.collect_us_equities_v1 symbol-master
python -m stockmachine.apps.collect_us_equities_v1 daily-bars --symbols AAPL MSFT SPY --start 2024-01-01 --end 2024-03-31
python -m stockmachine.apps.collect_us_equities_v1 research-seed --start 2025-01-01 --end 2026-03-21 --chunk-size 25
```

## How We Should Use Alpaca In This Repo

Recommended rollout order:

1. Replace Yahoo bootstrap for `symbol_master` and `daily_bar`
2. Keep research and backtest reading canonical silver tables
3. Add paper-trading execution against Alpaca Trading API
4. Add trade-update streaming and post-trade reconciliation

## Official References

- [About Trading API](https://docs.alpaca.markets/docs/trading-api)
- [About Market Data API](https://docs.alpaca.markets/v1.3/docs/about-market-data-api)
- [Working with Assets](https://docs.alpaca.markets/docs/working-with-assets)
- [Paper Trading](https://docs.alpaca.markets/docs/paper-trading)
- [Websocket Streaming](https://docs.alpaca.markets/v1.3/docs/websocket-streaming)
