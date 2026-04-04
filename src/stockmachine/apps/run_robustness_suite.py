from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from stockmachine.apps.research_paths import resolve_research_output_root, resolve_research_workspace
from stockmachine.research.p1_rigor import summarize_backtest_records
from stockmachine.research.robustness_analyzers import (
    build_cost_execution_stress_summary,
    build_parameter_stability_summary,
    build_selection_bias_summary,
    build_tail_dependence_summary,
    build_time_stability_summary,
    build_turnover_concentration_summary,
    build_universe_stability_summary,
)
from stockmachine.research.robustness_contracts import (
    build_robustness_artifact_bundle,
    load_backtest_records_artifact,
)
from stockmachine.research.robustness_frameworks import (
    build_robustness_gate_config,
    resolve_robustness_framework,
    should_run_parameter_stability,
)
from stockmachine.research.robustness_reports import (
    write_cost_execution_stress_artifact,
    write_parameter_stability_artifact,
    write_robustness_overview_artifact,
    write_selection_bias_artifact,
    write_tail_dependence_artifact,
    write_time_stability_artifact,
    write_turnover_concentration_artifact,
    write_universe_stability_artifact,
)
from stockmachine.research.strict_reports import write_json_artifact


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the shared research robustness suite.")
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--strategy-project", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--selection-bias-manifest", default=None)
    parser.add_argument("--universe-stability-manifest", default=None)
    parser.add_argument("--parameter-stability-manifest", default=None)
    parser.add_argument("--include-parameter-stability", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    framework = resolve_robustness_framework(
        strategy_project=args.strategy_project,
        horizon=args.horizon,
    )
    workspace = resolve_research_workspace(
        strategy_project=args.strategy_project,
        horizon=args.horizon,
        artifact_root=args.artifact_root,
    )
    output_root = resolve_research_output_root(
        output_root=args.output_root,
        default_dirname="robustness_suite",
        strategy_project=args.strategy_project,
        horizon=args.horizon,
        artifact_root=args.artifact_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    candidate_manifest_path = Path(args.candidate_manifest)
    candidates = _load_manifest(candidate_manifest_path)
    if candidates.empty:
        raise ValueError("candidate manifest is empty.")

    overview_rows: list[dict[str, Any]] = []
    per_model_root = output_root / "per_model"
    per_model_root.mkdir(parents=True, exist_ok=True)

    for row in candidates.to_dict(orient="records"):
        report_name = str(row.get("report_name") or row.get("model") or "candidate").strip()
        model_name = str(row.get("model") or report_name).strip()
        records_path = _resolve_manifest_path(candidate_manifest_path, row.get("records_path"))
        summary_path = _resolve_manifest_path(candidate_manifest_path, row.get("summary_path"))
        predictions_path = _resolve_manifest_path(candidate_manifest_path, row.get("predictions_path"))

        records = load_backtest_records_artifact(records_path)
        summary_frame = _load_summary_frame(summary_path, records=records, model_name=model_name, horizon=args.horizon)
        predictions = pd.read_csv(predictions_path) if predictions_path is not None and predictions_path.exists() else None
        bundle = build_robustness_artifact_bundle(
            strategy_project=args.strategy_project,
            model_name=model_name,
            records=records,
            summary=summary_frame,
            predictions=predictions,
        )

        yearly = build_time_stability_summary(
            bundle.records,
            model_name=model_name,
            horizon=args.horizon,
            framework=framework,
            period="year",
        )
        quarterly = build_time_stability_summary(
            bundle.records,
            model_name=model_name,
            horizon=args.horizon,
            framework=framework,
            period="quarter",
        )
        tail = build_tail_dependence_summary(
            bundle.records,
            model_name=model_name,
            horizon=args.horizon,
            framework=framework,
        )
        cost = build_cost_execution_stress_summary(
            bundle.records,
            model_name=model_name,
            horizon=args.horizon,
            framework=framework,
        )
        turnover = build_turnover_concentration_summary(bundle.records, model_name=model_name)

        model_dir = per_model_root / report_name
        model_dir.mkdir(parents=True, exist_ok=True)
        write_time_stability_artifact(model_dir / "time_stability_yearly.csv", yearly)
        write_time_stability_artifact(model_dir / "time_stability_quarterly.csv", quarterly)
        write_tail_dependence_artifact(model_dir / "tail_dependence_summary.csv", tail)
        write_cost_execution_stress_artifact(model_dir / "cost_execution_stress_summary.csv", cost)
        write_turnover_concentration_artifact(model_dir / "turnover_concentration_summary.csv", turnover)

        overview_rows.append(
            _build_overview_row(
                report_name=report_name,
                source=row.get("source"),
                config_id=row.get("config_id"),
                records_path=records_path,
                bundle=bundle,
                yearly=yearly,
                tail=tail,
                cost=cost,
                turnover=turnover,
            )
        )

    overview = write_robustness_overview_artifact(output_root / "robustness_overview.csv", pd.DataFrame(overview_rows))

    suite_level_root = output_root / "suite_level"
    suite_level_root.mkdir(parents=True, exist_ok=True)
    selection_bias_outputs: list[str] = []
    if args.selection_bias_manifest:
        for result in _run_selection_bias_suite(
            manifest_path=Path(args.selection_bias_manifest),
            suite_level_root=suite_level_root,
        ):
            selection_bias_outputs.append(result)

    universe_outputs: list[str] = []
    if args.universe_stability_manifest:
        for result in _run_universe_stability_suite(
            manifest_path=Path(args.universe_stability_manifest),
            suite_level_root=suite_level_root,
        ):
            universe_outputs.append(result)

    parameter_outputs: list[str] = []
    run_parameter = should_run_parameter_stability(
        framework,
        override=args.include_parameter_stability,
    )
    if run_parameter and args.parameter_stability_manifest:
        for result in _run_parameter_stability_suite(
            manifest_path=Path(args.parameter_stability_manifest),
            suite_level_root=suite_level_root,
            framework=framework,
        ):
            parameter_outputs.append(result)

    run_meta = {
        "ok": True,
        "strategy_project": workspace.project_id,
        "strategy_workspace": workspace.to_dict(),
        "framework": {
            "framework_id": framework.framework_id,
            "default_horizon": framework.default_horizon,
            "attribution_date_column": framework.attribution_date_column,
            "tail_trim_counts": list(framework.tail_trim_counts),
            "cost_stress_levels": list(framework.cost_stress_levels),
            "gate_defaults": build_robustness_gate_config(framework),
        },
        "output_root": str(output_root),
        "candidate_manifest": str(candidate_manifest_path),
        "counts": {
            "candidates": int(len(overview)),
            "selection_bias_runs": int(len(selection_bias_outputs)),
            "universe_stability_runs": int(len(universe_outputs)),
            "parameter_stability_runs": int(len(parameter_outputs)),
        },
        "analyzers": {
            "time_stability": True,
            "tail_dependence": True,
            "cost_execution_stress": True,
            "turnover_concentration": True,
            "selection_bias": bool(selection_bias_outputs),
            "universe_stability": bool(universe_outputs),
            "parameter_stability": bool(parameter_outputs),
        },
    }
    write_json_artifact(output_root / "run_meta.json", run_meta)
    print(json.dumps(run_meta, indent=2, sort_keys=True, default=str))
    return 0


def _load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.fillna("")


def _resolve_manifest_path(manifest_path: Path, raw_path: Any) -> Path | None:
    if raw_path in (None, ""):
        return None
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def _load_summary_frame(
    summary_path: Path | None,
    *,
    records: pd.DataFrame,
    model_name: str,
    horizon: int,
) -> pd.DataFrame:
    if summary_path is not None and summary_path.exists():
        frame = pd.read_csv(summary_path)
        if frame.empty:
            frame = pd.DataFrame([{"model": model_name}])
        if "model" not in frame.columns:
            frame["model"] = model_name
        return frame
    summary_row = summarize_backtest_records(records, horizon=horizon)
    return pd.DataFrame([{**summary_row, "model": model_name}])


def _build_overview_row(
    *,
    report_name: str,
    source: Any,
    config_id: Any,
    records_path: Path,
    bundle,
    yearly: pd.DataFrame,
    tail: pd.DataFrame,
    cost: pd.DataFrame,
    turnover: pd.DataFrame,
) -> dict[str, Any]:
    summary_row = bundle.summary.iloc[0].to_dict()

    positive_year_ratio = float((yearly["annualized_return"].astype(float) > 0).mean()) if not yearly.empty else None
    top5_row = _select_row(tail, tail_side="top", trim_count=5)
    bottom5_row = _select_row(tail, tail_side="bottom", trim_count=5)
    cost_20 = _select_cost_row(cost, target_bps=20.0)
    turnover_5 = _select_row(turnover, trim_count=5)

    return {
        "report_name": report_name,
        "model": bundle.model_name,
        "source": str(source or ""),
        "config_id": str(config_id or ""),
        "annualized_return": _to_scalar(summary_row.get("annualized_return")),
        "sharpe": _to_scalar(summary_row.get("sharpe")),
        "max_drawdown": _to_scalar(summary_row.get("max_drawdown")),
        "annualized_return_20bps": _to_scalar(cost_20.get("annualized_return") if cost_20 is not None else None),
        "mean_turnover": _to_scalar(summary_row.get("mean_turnover")),
        "positive_year_ratio": positive_year_ratio,
        "top5_positive_trade_abs_share": _to_scalar(
            top5_row.get("removed_return_share_of_abs_sum") if top5_row is not None else None
        ),
        "top5_negative_trade_abs_share": _to_scalar(
            bottom5_row.get("removed_return_share_of_abs_sum") if bottom5_row is not None else None
        ),
        "annualized_return_after_drop_top5": _to_scalar(
            top5_row.get("trimmed_annualized_return") if top5_row is not None else None
        ),
        "annualized_return_after_drop_bottom5": _to_scalar(
            bottom5_row.get("trimmed_annualized_return") if bottom5_row is not None else None
        ),
        "cost_slope_per_10bps": _to_scalar(
            cost.iloc[0]["annualized_return_slope_per_10bps"] if not cost.empty else None
        ),
        "break_even_cost_bps_per_side": _to_scalar(
            cost.iloc[0]["break_even_cost_bps_per_side"] if not cost.empty else None
        ),
        "top5_turnover_share": _to_scalar(turnover_5.get("top_turnover_share") if turnover_5 is not None else None),
        "active_day_ratio": _to_scalar(turnover_5.get("active_day_ratio") if turnover_5 is not None else None),
        "days_to_reach_50pct_turnover": _to_scalar(
            turnover_5.get("days_to_reach_50pct_turnover") if turnover_5 is not None else None
        ),
        "days_to_reach_80pct_turnover": _to_scalar(
            turnover_5.get("days_to_reach_80pct_turnover") if turnover_5 is not None else None
        ),
        "records_path": str(records_path),
    }


def _select_cost_row(frame: pd.DataFrame, *, target_bps: float) -> pd.Series | None:
    if frame.empty or "cost_bps_per_side" not in frame.columns:
        return None
    mask = frame["cost_bps_per_side"].astype(float) == float(target_bps)
    if not bool(mask.any()):
        return None
    return frame.loc[mask].iloc[0]


def _select_row(frame: pd.DataFrame, **filters: Any) -> pd.Series | None:
    if frame.empty:
        return None
    working = frame.copy()
    for column, value in filters.items():
        if column not in working.columns:
            return None
        working = working.loc[working[column] == value]
    if working.empty:
        return None
    return working.iloc[0]


def _to_scalar(value: Any) -> Any:
    if value is None or value == "":
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _run_selection_bias_suite(*, manifest_path: Path, suite_level_root: Path) -> list[str]:
    manifest = _load_manifest(manifest_path)
    outputs: list[str] = []
    for row in manifest.to_dict(orient="records"):
        surface_path = _resolve_manifest_path(manifest_path, row.get("surface_path") or row.get("path"))
        if surface_path is None:
            continue
        frame = pd.read_csv(surface_path)
        summary = build_selection_bias_summary(
            frame,
            metric_column=str(row["metric_column"]),
            higher_is_better=_parse_bool(row.get("higher_is_better"), default=True),
            family_column=str(row.get("family_column") or "") or None,
            run_label=str(row.get("run_label") or surface_path.stem),
        )
        output_path = suite_level_root / f"{summary.iloc[0]['run_label']}_selection_bias_summary.csv"
        write_selection_bias_artifact(output_path, summary)
        outputs.append(str(output_path))
    return outputs


def _run_universe_stability_suite(*, manifest_path: Path, suite_level_root: Path) -> list[str]:
    manifest = _load_manifest(manifest_path)
    outputs: list[str] = []
    for row in manifest.to_dict(orient="records"):
        surface_path = _resolve_manifest_path(manifest_path, row.get("surface_path") or row.get("path"))
        if surface_path is None:
            continue
        frame = pd.read_csv(surface_path)
        summary = build_universe_stability_summary(
            frame,
            label_column=str(row["label_column"]),
            annualized_return_column=str(row.get("annualized_return_column") or "annualized_return"),
            sharpe_column=str(row.get("sharpe_column") or "sharpe"),
            max_drawdown_column=str(row.get("max_drawdown_column") or "max_drawdown"),
            base_label_value=_empty_to_none(row.get("base_label_value")),
            run_label=str(row.get("run_label") or surface_path.stem),
        )
        output_path = suite_level_root / f"{summary.iloc[0]['run_label']}_universe_stability_summary.csv"
        write_universe_stability_artifact(output_path, summary)
        outputs.append(str(output_path))
    return outputs


def _run_parameter_stability_suite(
    *,
    manifest_path: Path,
    suite_level_root: Path,
    framework,
) -> list[str]:
    manifest = _load_manifest(manifest_path)
    outputs: list[str] = []
    for row in manifest.to_dict(orient="records"):
        surface_path = _resolve_manifest_path(manifest_path, row.get("surface_path") or row.get("path"))
        if surface_path is None:
            continue
        frame = pd.read_csv(surface_path)
        parameter_columns = [
            column.strip()
            for column in str(row.get("parameter_columns") or "").split(",")
            if column.strip()
        ]
        summary = build_parameter_stability_summary(
            frame,
            metric_column=str(row["metric_column"]),
            parameter_columns=parameter_columns,
            higher_is_better=_parse_bool(row.get("higher_is_better"), default=True),
            neighborhood_radius=int(row.get("neighborhood_radius") or framework.parameter_neighborhood_radius),
            run_label=str(row.get("run_label") or surface_path.stem),
        )
        output_path = suite_level_root / f"{summary.iloc[0]['run_label']}_parameter_stability_summary.csv"
        write_parameter_stability_artifact(output_path, summary)
        outputs.append(str(output_path))
    return outputs


def _parse_bool(raw_value: Any, *, default: bool) -> bool:
    if raw_value in (None, ""):
        return bool(default)
    lowered = str(raw_value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    return bool(default)


def _empty_to_none(raw_value: Any) -> Any:
    return None if raw_value in (None, "") else raw_value


if __name__ == "__main__":
    raise SystemExit(main())
