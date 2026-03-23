from __future__ import annotations

import argparse
import json

from stockmachine.ingestion.jobs import bootstrap_us_equities_yahoo_to_silver


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap US equities silver tables from Yahoo Finance.")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--silver-file-stem", default="yahoo_bootstrap")
    args = parser.parse_args()

    result = bootstrap_us_equities_yahoo_to_silver(
        start=args.start,
        end=args.end,
        silver_file_stem=args.silver_file_stem,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
