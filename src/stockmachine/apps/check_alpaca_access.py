from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from sys import stderr

from stockmachine.data.vendors import AlpacaCredentials, AlpacaHttpClient, AlpacaRequestError


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Alpaca trading and market-data credentials."
    )
    parser.add_argument("--symbol", default="AAPL", help="Primary symbol to check.")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark or ETF symbol to check.")
    parser.add_argument(
        "--start",
        default=(date.today() - timedelta(days=14)).isoformat(),
        help="Start date in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--feed",
        default="iex",
        help="Market-data feed to request, for example iex or sip.",
    )
    args = parser.parse_args()

    try:
        credentials = AlpacaCredentials.from_env()
        client = AlpacaHttpClient(credentials)
        assets = client.list_assets(status="active", asset_class="us_equity")
        bars = client.get_stock_bars(
            symbols=[args.symbol, args.benchmark],
            start=args.start,
            end=args.end,
            feed=args.feed,
            adjustment="raw",
        )
    except AlpacaRequestError as exc:
        print(str(exc), file=stderr)
        raise SystemExit(1) from exc

    bars_by_symbol: dict[str, int] = {}
    latest_bar_by_symbol: dict[str, str | None] = {}
    for bar in bars:
        symbol = str(bar.get("symbol"))
        bars_by_symbol[symbol] = bars_by_symbol.get(symbol, 0) + 1
        latest_bar_by_symbol[symbol] = str(bar.get("t")) if bar.get("t") else None

    requested = {args.symbol, args.benchmark}
    assets_by_symbol = {
        str(asset.get("symbol")): asset
        for asset in assets
        if isinstance(asset, dict) and asset.get("symbol") in requested
    }

    summary = {
        "trading_base_url": credentials.trading_base_url,
        "data_base_url": credentials.data_base_url,
        "feed": args.feed,
        "window": {
            "start": args.start,
            "end": args.end,
        },
        "requested_symbols": sorted(requested),
        "asset_checks": {
            symbol: {
                "found": symbol in assets_by_symbol,
                "tradable": bool(assets_by_symbol.get(symbol, {}).get("tradable")),
                "shortable": bool(assets_by_symbol.get(symbol, {}).get("shortable")),
                "fractionable": bool(assets_by_symbol.get(symbol, {}).get("fractionable")),
                "status": assets_by_symbol.get(symbol, {}).get("status"),
                "exchange": assets_by_symbol.get(symbol, {}).get("exchange"),
            }
            for symbol in sorted(requested)
        },
        "bar_checks": {
            symbol: {
                "count": bars_by_symbol.get(symbol, 0),
                "latest_timestamp": latest_bar_by_symbol.get(symbol),
            }
            for symbol in sorted(requested)
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
