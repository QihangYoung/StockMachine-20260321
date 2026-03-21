from __future__ import annotations

import argparse
import json

from stockmachine.alpha import list_alpha_expert_names
from stockmachine.research.us_equities_baseline import OverlayConfig, run_silver_chain_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the silver-table US equities backtest.")
    parser.add_argument("--model", default="hist_gbm", choices=list_alpha_expert_names())
    parser.add_argument("--predict-start", default="2024-01-01")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--output-dir", default="artifacts/us_equities_silver_chain")
    parser.add_argument("--min-close", type=float, default=10.0)
    parser.add_argument("--min-median-dollar-volume-20", type=float, default=50_000_000.0)
    parser.add_argument("--max-vol-20", type=float, default=0.04)
    parser.add_argument("--max-positions-per-sector", type=int, default=2)
    parser.add_argument("--cost-bps-per-side", type=float, default=10.0)
    parser.add_argument("--disable-sector-neutral", action="store_true")
    args = parser.parse_args()

    overlay_config = OverlayConfig(
        min_close=args.min_close,
        min_median_dollar_volume_20=args.min_median_dollar_volume_20,
        max_vol_20=args.max_vol_20,
        max_positions_per_sector=args.max_positions_per_sector,
        cost_bps_per_side=args.cost_bps_per_side,
        sector_neutral=not args.disable_sector_neutral,
    )
    result = run_silver_chain_backtest(
        predict_start=args.predict_start,
        model_name=args.model,
        top_k=args.top_k,
        horizon=args.horizon,
        output_dir=args.output_dir,
        overlay_config=overlay_config,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
