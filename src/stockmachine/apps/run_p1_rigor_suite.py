from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from stockmachine.apps.research_paths import (
    resolve_research_cache_dir,
    resolve_research_output_root,
    resolve_research_workspace,
)
from stockmachine.alpha import list_alpha_expert_names
from stockmachine.research.p1_rigor import (
    build_cost_stress_summary,
    build_period_stability_summary,
    build_strict_research_bundle,
    run_strict_model_sweep_from_bundle,
    run_topk_parameter_sweep_from_bundle,
)
from stockmachine.research.strict_preflight import build_strict_research_preflight
from stockmachine.research.strict_reports import write_csv_artifact
from stockmachine.research.us_equities_baseline import OverlayConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the P1 research-rigor suite.")
    parser.add_argument("--models", nargs="*", default=list_alpha_expert_names())
    parser.add_argument("--analysis-models", nargs="*", default=())
    parser.add_argument("--analysis-top-n", type=int, default=3)
    parser.add_argument("--predict-start", default="2025-01-01")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-k-grid", nargs="*", type=int, default=(5, 10, 15, 20))
    parser.add_argument("--cost-bps-levels", nargs="*", type=float, default=(10.0, 20.0, 40.0, 60.0))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--strategy-project", default=None)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--min-close", type=float, default=10.0)
    parser.add_argument("--min-median-dollar-volume-20", type=float, default=50_000_000.0)
    parser.add_argument("--max-vol-20", type=float, default=0.04)
    parser.add_argument("--max-positions-per-sector", type=int, default=2)
    parser.add_argument("--cost-bps-per-side", type=float, default=10.0)
    parser.add_argument("--disable-sector-neutral", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    workspace = resolve_research_workspace(
        strategy_project=getattr(args, "strategy_project", None),
        horizon=args.horizon,
        artifact_root=getattr(args, "artifact_root", "artifacts"),
    )
    output_root = resolve_research_output_root(
        output_root=getattr(args, "output_root", None),
        default_dirname="p1_rigor_suite",
        strategy_project=getattr(args, "strategy_project", None),
        horizon=args.horizon,
        artifact_root=getattr(args, "artifact_root", "artifacts"),
    )
    cache_dir = resolve_research_cache_dir(
        cache_dir=getattr(args, "cache_dir", None),
        default_dirname="p1_rigor_suite",
        strategy_project=getattr(args, "strategy_project", None),
        horizon=args.horizon,
        artifact_root=getattr(args, "artifact_root", "artifacts"),
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

    preflight = build_strict_research_preflight(
        horizon=args.horizon,
        strategy_project=getattr(args, "strategy_project", None),
    )
    if not preflight.ok:
        payload = {
            "ok": False,
            "predict_start": args.predict_start,
            "horizon": args.horizon,
            "strategy_project": workspace.project_id,
            "strategy_workspace": workspace.to_dict(),
            "output_root": str(output_root),
            "preflight": preflight.to_dict(),
            "cache": {
                "enabled": not args.disable_cache,
                "cache_dir": None if args.disable_cache else str(cache_dir),
                "bundle_cache_hit": False,
                "prediction_cache_hit": False,
                "bundle_cache_key": None,
                "prediction_cache_key": None,
                "rebuild_cache": bool(args.rebuild_cache),
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    bundle = build_strict_research_bundle(
        predict_start=args.predict_start,
        horizon=args.horizon,
        strategy_project=getattr(args, "strategy_project", None),
        prediction_options={"model_names": tuple(args.models)},
        cache_dir=None if args.disable_cache else cache_dir,
        reuse_cache=not args.disable_cache,
        rebuild_cache=bool(args.rebuild_cache),
        source_inputs=preflight.source_inputs,
    )

    strict_root = output_root / "strict_full"
    strict_payload = run_strict_model_sweep_from_bundle(
        bundle,
        model_names=args.models,
        output_root=strict_root,
        top_k=args.top_k,
        overlay_config=overlay_config,
    )
    strict_summary = pd.read_csv(strict_root / "summary_metrics.csv")

    if args.analysis_models:
        analysis_models = [str(model).strip() for model in args.analysis_models if str(model).strip()]
    else:
        analysis_models = (
            strict_summary.loc[strict_summary["status"] == "success"]
            .sort_values("sharpe", ascending=False)
            .head(args.analysis_top_n)["model"]
            .tolist()
        )

    yearly_frames: list[pd.DataFrame] = []
    quarterly_frames: list[pd.DataFrame] = []
    cost_frames: list[pd.DataFrame] = []
    for model_name in analysis_models:
        record_path = strict_root / model_name / "backtest_records.csv"
        if not record_path.exists():
            continue
        records = pd.read_csv(record_path)
        yearly_frames.append(
            build_period_stability_summary(records, model_name=model_name, horizon=args.horizon, period="year")
        )
        quarterly_frames.append(
            build_period_stability_summary(records, model_name=model_name, horizon=args.horizon, period="quarter")
        )
        cost_frames.append(
            build_cost_stress_summary(
                records,
                model_name=model_name,
                horizon=args.horizon,
                cost_levels_bps=args.cost_bps_levels,
            )
        )

    stability_root = output_root / "stability"
    stability_root.mkdir(parents=True, exist_ok=True)
    yearly_summary = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    quarterly_summary = pd.concat(quarterly_frames, ignore_index=True) if quarterly_frames else pd.DataFrame()
    write_csv_artifact(stability_root / "yearly_summary.csv", yearly_summary)
    write_csv_artifact(stability_root / "quarterly_summary.csv", quarterly_summary)

    cost_root = output_root / "cost_stress"
    cost_root.mkdir(parents=True, exist_ok=True)
    cost_summary = pd.concat(cost_frames, ignore_index=True) if cost_frames else pd.DataFrame()
    write_csv_artifact(cost_root / "summary_metrics.csv", cost_summary)

    topk_root = output_root / "topk_sweep"
    topk_summary = run_topk_parameter_sweep_from_bundle(
        bundle,
        model_names=analysis_models,
        top_k_values=args.top_k_grid,
        output_root=topk_root,
        overlay_config=overlay_config,
    )

    payload = {
        "ok": bool(strict_payload.get("ok", False)),
        "predict_start": args.predict_start,
        "horizon": args.horizon,
        "strategy_project": workspace.project_id,
        "strategy_workspace": workspace.to_dict(),
        "output_root": str(output_root),
        "strict_summary_path": str(strict_root / "summary_metrics.csv"),
        "analysis_models": analysis_models,
        "stability_yearly_path": str(stability_root / "yearly_summary.csv"),
        "stability_quarterly_path": str(stability_root / "quarterly_summary.csv"),
        "cost_stress_path": str(cost_root / "summary_metrics.csv"),
        "topk_sweep_path": str(topk_root / "summary_metrics.csv"),
        "preflight": preflight.to_dict(),
        "counts": {
            "strict_models": int(len(strict_summary)),
            "analysis_models": int(len(analysis_models)),
            "topk_rows": int(len(topk_summary)),
        },
        "cache": {
            "enabled": not args.disable_cache,
            "cache_dir": None if args.disable_cache else str(cache_dir),
            "bundle_cache_hit": bool(getattr(bundle, "bundle_cache_hit", False)),
            "prediction_cache_hit": bool(getattr(bundle, "prediction_cache_hit", False)),
            "bundle_cache_key": getattr(bundle, "bundle_cache_key", None),
            "prediction_cache_key": getattr(bundle, "prediction_cache_key", None),
            "rebuild_cache": bool(args.rebuild_cache),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
