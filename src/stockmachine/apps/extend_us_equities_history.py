from __future__ import annotations

import argparse
import json

from stockmachine.ingestion.jobs import (
    backfill_static_metadata_history_from_silver,
    bootstrap_us_equities_yahoo_to_silver,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely extend US-equities silver history with an older non-overlapping Yahoo window."
    )
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2018-12-31")
    parser.add_argument("--silver-file-stem", default="yahoo_bootstrap_2014_2018")
    parser.add_argument("--skip-backfill", action="store_true")
    args = parser.parse_args()

    bootstrap_result = bootstrap_us_equities_yahoo_to_silver(
        start=args.start,
        end=args.end,
        silver_file_stem=args.silver_file_stem,
    )
    payload = {
        "bootstrap": bootstrap_result,
        "metadata_backfill": None,
    }
    if not args.skip_backfill:
        payload["metadata_backfill"] = backfill_static_metadata_history_from_silver()

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
