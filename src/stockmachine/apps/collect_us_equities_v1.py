from __future__ import annotations

import argparse
from datetime import date

from stockmachine.ingestion.jobs import (
    collect_adj_factors,
    collect_daily_bars,
    collect_research_seed,
    collect_symbol_master_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect US equities V1 data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("symbol-master", help="Collect Alpaca asset metadata.")

    bars_parser = subparsers.add_parser("daily-bars", help="Collect Alpaca daily bars.")
    bars_parser.add_argument("--symbols", nargs="+", required=True, help="Ticker symbols to fetch.")
    bars_parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD.")
    bars_parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD.")
    bars_parser.add_argument("--feed", default="iex", help="Market-data feed, for example iex or sip.")
    bars_parser.add_argument("--adjustment", default="raw", help="Adjustment mode, for example raw or all.")

    adj_parser = subparsers.add_parser("adj-factors", help="Collect Alpaca adjustment factors.")
    adj_parser.add_argument("--symbols", nargs="+", required=True, help="Ticker symbols to fetch.")
    adj_parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD.")
    adj_parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD.")
    adj_parser.add_argument("--feed", default="iex", help="Market-data feed, for example iex or sip.")

    seed_parser = subparsers.add_parser(
        "research-seed",
        help="Collect the default research universe plus benchmark into silver.",
    )
    seed_parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD.")
    seed_parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD.")
    seed_parser.add_argument("--chunk-size", type=int, default=25, help="Symbols per request chunk.")
    seed_parser.add_argument("--feed", default="iex", help="Market-data feed, for example iex or sip.")
    seed_parser.add_argument("--adjustment", default="raw", help="Adjustment mode, for example raw or all.")
    seed_parser.add_argument("--skip-symbol-master", action="store_true", help="Skip the asset snapshot step.")
    seed_parser.add_argument("--skip-adj-factor", action="store_true", help="Skip Alpaca adj_factor collection.")

    args = parser.parse_args()

    if args.command == "symbol-master":
        result = collect_symbol_master_snapshot()
    elif args.command == "adj-factors":
        result = collect_adj_factors(
            symbols=args.symbols,
            start_date=date.fromisoformat(args.start),
            end_date=date.fromisoformat(args.end),
            feed=args.feed,
        )
    elif args.command == "research-seed":
        result = collect_research_seed(
            start_date=date.fromisoformat(args.start),
            end_date=date.fromisoformat(args.end),
            chunk_size=args.chunk_size,
            feed=args.feed,
            adjustment=args.adjustment,
            include_symbol_master=not args.skip_symbol_master,
            include_adj_factor=not args.skip_adj_factor,
        )
    else:
        result = collect_daily_bars(
            symbols=args.symbols,
            start_date=date.fromisoformat(args.start),
            end_date=date.fromisoformat(args.end),
            feed=args.feed,
            adjustment=args.adjustment,
        )

    print(result)


if __name__ == "__main__":
    main()
