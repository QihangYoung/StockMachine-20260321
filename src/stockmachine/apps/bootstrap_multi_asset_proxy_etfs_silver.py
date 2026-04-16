from __future__ import annotations

import argparse
import json
from pathlib import Path

from stockmachine.ingestion.jobs import bootstrap_multi_asset_proxy_etfs_yahoo_to_silver
from stockmachine.ingestion.storage import StorageLayout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap the default proxy ETF set for long-window multi-asset research."
    )
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--silver-file-stem", default="multi_asset_proxy_bootstrap")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    layout = StorageLayout(root=Path(args.data_root)) if args.data_root else StorageLayout()
    result = bootstrap_multi_asset_proxy_etfs_yahoo_to_silver(
        start=args.start,
        end=args.end,
        layout=layout,
        silver_file_stem=args.silver_file_stem,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
