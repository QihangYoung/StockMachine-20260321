from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from stockmachine.research import get_default_research_protocol
from stockmachine.research.us_equities_baseline import OverlayConfig, run_silver_chain_backtest


DEFAULT_SWEEP_COLUMNS: tuple[str, ...] = (
    "model",
    "status",
    "predict_start",
    "top_k",
    "horizon",
    "artifacts_dir",
    "sessions",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "benchmark_total_return",
    "mean_turnover",
    "mean_cost_bps",
    "error_type",
    "error_message",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a batch sweep of US equities backtests.")
    parser.add_argument("--models", nargs="*", default=(), help="Model names to sweep.")
    parser.add_argument("--predict-start", default="2024-01-01")
    parser.add_argument("--output-root", default="artifacts/us_equities_model_sweep")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--min-close", type=float, default=10.0)
    parser.add_argument("--min-median-dollar-volume-20", type=float, default=50_000_000.0)
    parser.add_argument("--max-vol-20", type=float, default=0.04)
    parser.add_argument("--max-positions-per-sector", type=int, default=2)
    parser.add_argument("--cost-bps-per-side", type=float, default=10.0)
    parser.add_argument("--disable-sector-neutral", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true", default=False)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    parser = build_arg_parser()
    return parser.parse_args(raw_args)


def run_model_sweep(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    research_protocol = get_default_research_protocol()

    models = [str(model).strip() for model in getattr(args, "models", ()) if str(model).strip()]
    overlay_config = OverlayConfig(
        min_close=args.min_close,
        min_median_dollar_volume_20=args.min_median_dollar_volume_20,
        max_vol_20=args.max_vol_20,
        max_positions_per_sector=args.max_positions_per_sector,
        cost_bps_per_side=args.cost_bps_per_side,
        sector_neutral=not args.disable_sector_neutral,
    )

    rows: list[dict[str, Any]] = []
    ok = True
    empty_models = len(models) == 0

    if empty_models:
        ok = False
    else:
        for model_name in models:
            model_dir = output_root / _safe_component(model_name)
            model_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = run_silver_chain_backtest(
                    predict_start=args.predict_start,
                    model_name=model_name,
                    top_k=args.top_k,
                    horizon=args.horizon,
                    output_dir=model_dir,
                    overlay_config=overlay_config,
                )
                row = _build_success_row(
                    model_name=model_name,
                    result=result,
                    predict_start=args.predict_start,
                    top_k=args.top_k,
                    horizon=args.horizon,
                )
            except Exception as exc:
                ok = False
                row = _build_error_row(
                    model_name=model_name,
                    model_dir=model_dir,
                    predict_start=args.predict_start,
                    top_k=args.top_k,
                    horizon=args.horizon,
                    exc=exc,
                )
                if getattr(args, "stop_on_error", False):
                    rows.append(row)
                    break
            rows.append(row)

    summary_frame = pd.DataFrame(rows, columns=DEFAULT_SWEEP_COLUMNS)
    summary_path = output_root / "summary_metrics.csv"
    summary_frame.to_csv(summary_path, index=False)
    protocol_path = output_root / "research_protocol.json"
    protocol_path.write_text(
        json.dumps(research_protocol.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    payload = {
        "command": "sweep",
        "ok": ok,
        "empty_models": empty_models,
        "predict_start": args.predict_start,
        "output_root": str(output_root),
        "summary_metrics_path": str(summary_path),
        "research_protocol_path": str(protocol_path),
        "research_protocol": research_protocol.to_dict(),
        "overlay_config": asdict(overlay_config),
        "counts": {
            "requested": len(models),
            "success": sum(1 for row in rows if row.get("status") == "success"),
            "failed": sum(1 for row in rows if row.get("status") == "failed"),
        },
        "results": rows,
    }
    if empty_models:
        payload["error"] = {
            "type": "ValueError",
            "message": "No models were provided.",
        }
    return payload


def _build_success_row(
    *,
    model_name: str,
    result: Mapping[str, Any],
    predict_start: str,
    top_k: int,
    horizon: int,
) -> dict[str, Any]:
    summary = result.get("summary")
    summary = dict(summary) if isinstance(summary, Mapping) else {}
    row: dict[str, Any] = {
        "model": model_name,
        "status": "success",
        "predict_start": predict_start,
        "top_k": top_k,
        "horizon": horizon,
        "artifacts_dir": result.get("artifacts_dir"),
        "sessions": summary.get("sessions"),
        "total_return": summary.get("total_return"),
        "annualized_return": summary.get("annualized_return"),
        "annualized_volatility": summary.get("annualized_volatility"),
        "sharpe": summary.get("sharpe"),
        "max_drawdown": summary.get("max_drawdown"),
        "benchmark_total_return": summary.get("benchmark_total_return"),
        "mean_turnover": summary.get("mean_turnover"),
        "mean_cost_bps": summary.get("mean_cost_bps"),
        "error_type": None,
        "error_message": None,
    }
    return row


def _build_error_row(
    *,
    model_name: str,
    model_dir: Path,
    predict_start: str,
    top_k: int,
    horizon: int,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "status": "failed",
        "predict_start": predict_start,
        "top_k": top_k,
        "horizon": horizon,
        "artifacts_dir": str(model_dir),
        "sessions": None,
        "total_return": None,
        "annualized_return": None,
        "annualized_volatility": None,
        "sharpe": None,
        "max_drawdown": None,
        "benchmark_total_return": None,
        "mean_turnover": None,
        "mean_cost_bps": None,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _safe_component(value: str) -> str:
    cleaned = value.strip().replace("\\", "_").replace("/", "_")
    return cleaned or "model"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_model_sweep(args)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
