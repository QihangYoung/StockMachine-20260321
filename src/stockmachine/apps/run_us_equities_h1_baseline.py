from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from stockmachine.apps.research_paths import resolve_research_output_root, resolve_research_workspace
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.research.h1_us_equities import (
    H1_BASE_MODEL_NAMES,
    H1PromotionGateConfig,
    H1TurnoverControlConfig,
    build_h1_walk_forward_split_config,
    run_h1_baseline_sweep,
)
from stockmachine.research.us_equities_baseline import OverlayConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the first h1 US equities baseline sweep.")
    parser.add_argument("--predict-start", default="2025-01-01")
    parser.add_argument("--models", nargs="*", default=H1_BASE_MODEL_NAMES)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--strategy-project", default="us_equities_h1")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--min-close", type=float, default=10.0)
    parser.add_argument("--min-median-dollar-volume-20", type=float, default=50_000_000.0)
    parser.add_argument("--max-vol-20", type=float, default=0.04)
    parser.add_argument("--max-positions-per-sector", type=int, default=2)
    parser.add_argument("--cost-bps-per-side", type=float, default=10.0)
    parser.add_argument("--disable-sector-neutral", action="store_true")
    parser.add_argument("--no-trade-band", type=float, default=0.05)
    parser.add_argument("--max-turnover", type=float, default=1.0)
    parser.add_argument("--min-weight-change", type=float, default=0.02)
    parser.add_argument("--hold-rank-buffer", type=int, default=2)
    parser.add_argument("--entry-rank-buffer", type=int, default=2)
    parser.add_argument("--max-new-names-per-rebalance", type=int, default=2)
    parser.add_argument("--cost-bps-levels", nargs="*", type=float, default=(10.0, 15.0, 20.0, 30.0))
    parser.add_argument("--train-window-days", type=int, default=None)
    parser.add_argument("--validation-window-days", type=int, default=None)
    parser.add_argument("--test-window-days", type=int, default=None)
    parser.add_argument("--purge-window-days", type=int, default=None)
    parser.add_argument("--embargo-window-days", type=int, default=None)
    parser.add_argument("--roll-frequency", default=None)
    parser.add_argument("--storage-root", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    workspace = resolve_research_workspace(
        strategy_project=args.strategy_project,
        horizon=1,
        artifact_root=args.artifact_root,
    )
    output_root = resolve_research_output_root(
        output_root=args.output_root,
        default_dirname="us_equities_h1_baseline",
        strategy_project=args.strategy_project,
        horizon=1,
        artifact_root=args.artifact_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    overlay_config = OverlayConfig(
        min_close=args.min_close,
        min_median_dollar_volume_20=args.min_median_dollar_volume_20,
        max_vol_20=args.max_vol_20,
        max_positions_per_sector=args.max_positions_per_sector,
        cost_bps_per_side=args.cost_bps_per_side,
        sector_neutral=not args.disable_sector_neutral,
    )
    turnover_control = H1TurnoverControlConfig(
        no_trade_band=args.no_trade_band,
        max_turnover=args.max_turnover,
        min_weight_change=args.min_weight_change,
        hold_rank_buffer=args.hold_rank_buffer,
        entry_rank_buffer=args.entry_rank_buffer,
        max_new_names_per_rebalance=args.max_new_names_per_rebalance,
    )
    split_config = build_h1_walk_forward_split_config(
        train_window_days=args.train_window_days,
        validation_window_days=args.validation_window_days,
        test_window_days=args.test_window_days,
        purge_window_days=args.purge_window_days,
        embargo_window_days=args.embargo_window_days,
        roll_frequency=args.roll_frequency,
    )
    storage = StorageLayout(root=Path(args.storage_root)) if args.storage_root else StorageLayout()

    payload = run_h1_baseline_sweep(
        predict_start=args.predict_start,
        model_names=tuple(args.models),
        top_k=args.top_k,
        output_dir=output_root,
        overlay_config=overlay_config,
        turnover_control=turnover_control,
        cost_levels_bps=tuple(args.cost_bps_levels),
        gate_config=H1PromotionGateConfig(),
        layout=storage,
        split_config=split_config,
    )
    payload["strategy_project"] = workspace.project_id
    payload["strategy_workspace"] = workspace.to_dict()
    payload["output_root"] = str(output_root)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
