from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stockmachine.alpha import list_alpha_experts
from stockmachine.backtest import DailyOpenHoldBacktestEngine, DataFrameSignalModel
from stockmachine.data.loaders import load_us_equities_dataset
from stockmachine.execution import NextOpenOrderExecutionPolicy
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.portfolio import RiskAwareTopKPortfolioPolicy
from stockmachine.research import get_default_research_protocol
from stockmachine.research.strict_preflight import (
    StrictResearchSourceInputs,
    build_strict_research_preflight,
    ensure_strict_research_preflight_ok,
)
from stockmachine.research.strict_reports import (
    write_csv_artifact,
    write_json_artifact,
    write_model_backtest_artifacts,
    write_summary_metrics_artifact,
)
from stockmachine.research.strict_frameworks import StrictFrameworkSpec, resolve_strict_framework
from stockmachine.research.us_equities_baseline import (
    BENCHMARK_SYMBOL,
    OverlayConfig,
    build_point_in_time_metadata_history,
    build_price_panel_from_silver,
    build_research_frame,
    generate_walk_forward_predictions,
    resolve_model_whitebox_policy,
)

STRICT_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model",
    "status",
    "predict_start",
    "strategy_project",
    "framework_id",
    "requested_top_k",
    "top_k",
    "horizon",
    "artifacts_dir",
    "route_model_whitebox",
    "effective_min_median_dollar_volume_20",
    "effective_max_vol_20",
    "sessions",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "benchmark_total_return",
    "mean_turnover",
    "mean_cost_bps",
)
STRICT_CACHE_DATASET_TABLES: tuple[str, ...] = (
    "universe_membership",
    "daily_bar",
    "adj_factor",
    "benchmark_index",
    "industry_membership",
    "symbol_master",
)
STRICT_CACHE_DEPENDENCY_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "torch",
)
STRICT_CACHE_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True, frozen=True)
class StrictResearchBundle:
    """Reusable strict-research inputs shared across multiple model reruns."""

    predict_start: str
    horizon: int
    strategy_project: str
    framework_id: str
    dataset: Mapping[str, pd.DataFrame]
    research_frame: pd.DataFrame
    predictions: pd.DataFrame
    bundle_cache_key: str | None = None
    prediction_cache_key: str | None = None
    bundle_cache_hit: bool = False
    prediction_cache_hit: bool = False


@dataclass(slots=True, frozen=True)
class RepositoryCacheState:
    """Repository state used to determine whether cache reuse is safe."""

    head: str
    clean: bool
    cache_allowed: bool
    reason: str


def _resolve_research_session_dates(price_data: pd.DataFrame) -> pd.Index:
    return pd.Index(
        pd.to_datetime(price_data.loc[price_data["symbol"] != BENCHMARK_SYMBOL, "date"], errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )


def _build_repository_cache_state() -> RepositoryCacheState:
    try:
        head = _run_git_command("rev-parse", "HEAD").strip()
        status_output = _run_git_command("status", "--porcelain")
    except RuntimeError as exc:
        return RepositoryCacheState(
            head="unavailable",
            clean=False,
            cache_allowed=False,
            reason=f"git_unavailable:{exc}",
        )

    clean = status_output.strip() == ""
    if not clean:
        return RepositoryCacheState(
            head=head,
            clean=False,
            cache_allowed=False,
            reason="repository_dirty",
        )
    return RepositoryCacheState(
        head=head,
        clean=True,
        cache_allowed=True,
        reason="clean",
    )


def _run_git_command(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(STRICT_CACHE_REPOSITORY_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def _collect_dependency_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package_name in STRICT_CACHE_DEPENDENCY_PACKAGES:
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def _build_alpha_registry_snapshot() -> list[dict[str, object]]:
    return [spec.to_dict() for spec in list_alpha_experts()]


def _framework_protocol_dict(framework: StrictFrameworkSpec) -> dict[str, object]:
    if framework.protocol_family == "h1":
        from stockmachine.research import get_h1_research_protocol

        return get_h1_research_protocol().to_dict()
    return get_default_research_protocol().to_dict()


def _build_silver_input_fingerprint(layout: StorageLayout) -> dict[str, object]:
    root = layout.root.resolve()
    tables: dict[str, object] = {}
    for table_name in STRICT_CACHE_DATASET_TABLES:
        table_dir = layout.silver_table_dir(table_name)
        if not table_dir.exists():
            tables[table_name] = {"present": False, "files": []}
            continue

        files: list[dict[str, object]] = []
        for path in sorted(table_dir.glob("*.jsonl")):
            stat = path.stat()
            files.append(
                {
                    "name": path.name,
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "sha256": _hash_file_contents(path),
                }
            )
        tables[table_name] = {"present": True, "files": files}

    return {
        "layout_root": str(root),
        "tables": tables,
    }


def _hash_file_contents(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_bundle_cache_signature(
    *,
    framework: StrictFrameworkSpec,
    layout: StorageLayout,
    horizon: int,
    repository_state: RepositoryCacheState,
) -> dict[str, object]:
    return {
        "cache_kind": "strict_bundle",
        "cache_version": framework.bundle_cache_version,
        "framework_id": framework.framework_id,
        "strategy_project": framework.strategy_project,
        "repository_commit": repository_state.head,
        "dependency_versions": _collect_dependency_versions(),
        "protocol": _framework_protocol_dict(framework),
        "horizon": int(horizon),
        "universe_name": framework.universe_name,
        "silver_inputs": _build_silver_input_fingerprint(layout),
    }


def _build_prediction_cache_signature(
    *,
    framework: StrictFrameworkSpec,
    bundle_key: str,
    predict_start: str,
    prediction_options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "cache_kind": "prediction",
        "cache_version": framework.prediction_cache_version,
        "framework_id": framework.framework_id,
        "strategy_project": framework.strategy_project,
        "bundle_cache_key": bundle_key,
        "predict_start": str(predict_start),
        "prediction_options": _normalize_prediction_options(prediction_options),
        "alpha_registry": _build_alpha_registry_snapshot(),
    }


def _make_cache_key(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(payload, sort_keys=True))


def _normalize_prediction_options(payload: Mapping[str, object] | None) -> dict[str, object]:
    if not payload:
        return {}
    return json.loads(json.dumps(dict(payload), sort_keys=True, default=str))


def _strict_bundle_cache_dir(cache_dir: str | Path, bundle_key: str) -> Path:
    return Path(cache_dir) / "strict_bundle" / bundle_key


def _prediction_cache_dir(bundle_cache_dir: Path, prediction_key: str) -> Path:
    return bundle_cache_dir / "predictions" / prediction_key


def _bundle_manifest_path(bundle_cache_dir: Path) -> Path:
    return bundle_cache_dir / "manifest.json"


def _prediction_manifest_path(prediction_cache_dir: Path) -> Path:
    return prediction_cache_dir / "manifest.json"


def _bundle_dataset_path(bundle_cache_dir: Path, table_name: str) -> Path:
    return bundle_cache_dir / f"{table_name}.pkl"


def _bundle_price_data_path(bundle_cache_dir: Path) -> Path:
    return bundle_cache_dir / "price_data.pkl"


def _bundle_metadata_path(bundle_cache_dir: Path) -> Path:
    return bundle_cache_dir / "metadata.pkl"


def _bundle_research_frame_path(bundle_cache_dir: Path) -> Path:
    return bundle_cache_dir / "research_frame.pkl"


def _prediction_frame_path(prediction_cache_dir: Path) -> Path:
    return prediction_cache_dir / "predictions.pkl"


def _read_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_strict_bundle_cache(
    bundle_cache_dir: Path,
    *,
    expected_manifest: Mapping[str, object],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame] | None:
    manifest = _read_manifest(_bundle_manifest_path(bundle_cache_dir))
    if manifest != _normalize_manifest(expected_manifest):
        return None

    expected_paths = [
        _bundle_price_data_path(bundle_cache_dir),
        _bundle_metadata_path(bundle_cache_dir),
        _bundle_research_frame_path(bundle_cache_dir),
        *[_bundle_dataset_path(bundle_cache_dir, table_name) for table_name in STRICT_CACHE_DATASET_TABLES],
    ]
    if any(not path.exists() for path in expected_paths):
        return None

    try:
        dataset = {
            table_name: pd.read_pickle(_bundle_dataset_path(bundle_cache_dir, table_name))
            for table_name in STRICT_CACHE_DATASET_TABLES
        }
        research_frame = pd.read_pickle(_bundle_research_frame_path(bundle_cache_dir))
    except Exception:
        return None

    return dataset, research_frame


def _write_strict_bundle_cache(
    bundle_cache_dir: Path,
    *,
    manifest: Mapping[str, object],
    dataset: Mapping[str, pd.DataFrame],
    price_data: pd.DataFrame,
    metadata: pd.DataFrame,
    research_frame: pd.DataFrame,
) -> None:
    bundle_cache_dir.mkdir(parents=True, exist_ok=True)
    normalized_manifest = _normalize_manifest(manifest)
    for table_name in STRICT_CACHE_DATASET_TABLES:
        dataset.get(table_name, pd.DataFrame()).to_pickle(_bundle_dataset_path(bundle_cache_dir, table_name))
    price_data.to_pickle(_bundle_price_data_path(bundle_cache_dir))
    metadata.to_pickle(_bundle_metadata_path(bundle_cache_dir))
    research_frame.to_pickle(_bundle_research_frame_path(bundle_cache_dir))
    _bundle_manifest_path(bundle_cache_dir).write_text(
        json.dumps(normalized_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_prediction_cache(
    prediction_cache_dir: Path,
    *,
    expected_manifest: Mapping[str, object],
) -> pd.DataFrame | None:
    manifest = _read_manifest(_prediction_manifest_path(prediction_cache_dir))
    if manifest != _normalize_manifest(expected_manifest):
        return None

    prediction_path = _prediction_frame_path(prediction_cache_dir)
    if not prediction_path.exists():
        return None

    try:
        return pd.read_pickle(prediction_path)
    except Exception:
        return None


def _write_prediction_cache(
    prediction_cache_dir: Path,
    *,
    manifest: Mapping[str, object],
    predictions: pd.DataFrame,
) -> None:
    prediction_cache_dir.mkdir(parents=True, exist_ok=True)
    normalized_manifest = _normalize_manifest(manifest)
    predictions.to_pickle(_prediction_frame_path(prediction_cache_dir))
    _prediction_manifest_path(prediction_cache_dir).write_text(
        json.dumps(normalized_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_strict_research_bundle(
    *,
    predict_start: str,
    horizon: int = 5,
    strategy_project: str | None = None,
    prediction_options: Mapping[str, object] | None = None,
    layout: StorageLayout | None = None,
    cache_dir: str | Path | None = None,
    reuse_cache: bool = True,
    rebuild_cache: bool = False,
    source_inputs: StrictResearchSourceInputs | None = None,
) -> StrictResearchBundle:
    """Build one strict point-in-time research bundle once and reuse it."""

    storage = layout or StorageLayout()
    framework = resolve_strict_framework(strategy_project=strategy_project, horizon=horizon)
    repository_state = _build_repository_cache_state()

    bundle_cache_key: str | None = None
    prediction_cache_key: str | None = None
    bundle_cache_hit = False
    prediction_cache_hit = False
    bundle_cache_dir: Path | None = None
    bundle_manifest: dict[str, object] | None = None

    dataset: Mapping[str, pd.DataFrame]
    research_frame: pd.DataFrame

    if cache_dir is not None and repository_state.cache_allowed:
        bundle_manifest = _build_bundle_cache_signature(
            framework=framework,
            layout=storage,
            horizon=horizon,
            repository_state=repository_state,
        )
        bundle_cache_key = _make_cache_key(bundle_manifest)
        bundle_cache_dir = _strict_bundle_cache_dir(cache_dir, bundle_cache_key)
        if reuse_cache and not rebuild_cache:
            cached_bundle = _load_strict_bundle_cache(bundle_cache_dir, expected_manifest=bundle_manifest)
            if cached_bundle is not None:
                dataset, research_frame = cached_bundle
                bundle_cache_hit = True
            else:
                dataset, price_data, metadata, research_frame = _build_uncached_strict_bundle_inputs(
                    storage=storage,
                    horizon=horizon,
                    framework=framework,
                    source_inputs=source_inputs,
                )
                _write_strict_bundle_cache(
                    bundle_cache_dir,
                    manifest=bundle_manifest,
                    dataset=dataset,
                    price_data=price_data,
                    metadata=metadata,
                    research_frame=research_frame,
                )
        else:
            dataset, price_data, metadata, research_frame = _build_uncached_strict_bundle_inputs(
                storage=storage,
                horizon=horizon,
                framework=framework,
                source_inputs=source_inputs,
            )
            _write_strict_bundle_cache(
                bundle_cache_dir,
                manifest=bundle_manifest,
                dataset=dataset,
                price_data=price_data,
                metadata=metadata,
                research_frame=research_frame,
            )
    else:
        dataset, _, _, research_frame = _build_uncached_strict_bundle_inputs(
            storage=storage,
            horizon=horizon,
            framework=framework,
            source_inputs=source_inputs,
        )

    if bundle_cache_dir is not None and bundle_cache_key is not None and repository_state.cache_allowed:
        prediction_manifest = _build_prediction_cache_signature(
            framework=framework,
            bundle_key=bundle_cache_key,
            predict_start=predict_start,
            prediction_options=prediction_options,
        )
        prediction_cache_key = _make_cache_key(prediction_manifest)
        prediction_dir = _prediction_cache_dir(bundle_cache_dir, prediction_cache_key)
        if reuse_cache and not rebuild_cache:
            cached_predictions = _load_prediction_cache(prediction_dir, expected_manifest=prediction_manifest)
            if cached_predictions is not None:
                predictions = cached_predictions
                prediction_cache_hit = True
            else:
                predictions = _generate_framework_predictions(
                    framework=framework,
                    research_frame=research_frame,
                    predict_start=predict_start,
                    prediction_options=prediction_options,
                )
                _write_prediction_cache(
                    prediction_dir,
                    manifest=prediction_manifest,
                    predictions=predictions,
                )
        else:
            predictions = _generate_framework_predictions(
                framework=framework,
                research_frame=research_frame,
                predict_start=predict_start,
                prediction_options=prediction_options,
            )
            _write_prediction_cache(
                prediction_dir,
                manifest=prediction_manifest,
                predictions=predictions,
            )
    else:
        predictions = _generate_framework_predictions(
            framework=framework,
            research_frame=research_frame,
            predict_start=predict_start,
            prediction_options=prediction_options,
        )
    return StrictResearchBundle(
        predict_start=predict_start,
        horizon=horizon,
        strategy_project=framework.strategy_project,
        framework_id=framework.framework_id,
        dataset=dataset,
        research_frame=research_frame,
        predictions=predictions,
        bundle_cache_key=bundle_cache_key,
        prediction_cache_key=prediction_cache_key,
        bundle_cache_hit=bundle_cache_hit,
        prediction_cache_hit=prediction_cache_hit,
    )


def _build_uncached_strict_bundle_inputs(
    *,
    storage: StorageLayout,
    horizon: int,
    framework: StrictFrameworkSpec,
    source_inputs: StrictResearchSourceInputs | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if source_inputs is None:
        dataset = load_us_equities_dataset(layout=storage)
        price_data = build_price_panel_from_silver(dataset)
        source_inputs = StrictResearchSourceInputs(
            dataset=dataset,
            price_data=price_data,
            session_dates=_resolve_research_session_dates(price_data),
        )
    preflight = build_strict_research_preflight(
        horizon=horizon,
        strategy_project=framework.strategy_project,
        layout=storage,
        source_inputs=source_inputs,
    )
    ensure_strict_research_preflight_ok(preflight)
    inputs = preflight.source_inputs or source_inputs
    if inputs is None:
        raise ValueError("Strict research preflight did not provide source inputs.")
    dataset = inputs.dataset
    price_data = inputs.price_data
    session_dates = inputs.session_dates
    metadata = _build_strict_bundle_metadata(dataset, session_dates=session_dates, framework=framework)
    research_frame = _build_framework_research_frame(
        framework=framework,
        price_data=price_data,
        metadata=metadata,
        horizon=horizon,
    )
    return dataset, price_data, metadata, research_frame


def _build_strict_bundle_metadata(
    dataset: Mapping[str, pd.DataFrame],
    *,
    session_dates: pd.Index | None = None,
    framework: StrictFrameworkSpec,
) -> pd.DataFrame:
    if session_dates is None:
        price_data = build_price_panel_from_silver(dict(dataset))
        session_dates = _resolve_research_session_dates(price_data)
    return build_point_in_time_metadata_history(
        session_dates,
        universe_membership_frame=dataset.get("universe_membership", pd.DataFrame()),
        symbol_master_frame=dataset["symbol_master"],
        industry_membership_frame=dataset["industry_membership"],
        require_snapshot=True,
        universe_name=framework.universe_name,
    )


def _build_framework_research_frame(
    *,
    framework: StrictFrameworkSpec,
    price_data: pd.DataFrame,
    metadata: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    if framework.protocol_family == "h1":
        from stockmachine.research.h1_us_equities import build_h1_research_frame

        return build_h1_research_frame(
            price_data,
            benchmark_symbol=BENCHMARK_SYMBOL,
            symbol_metadata=metadata,
        )

    return build_research_frame(
        price_data,
        benchmark_symbol=BENCHMARK_SYMBOL,
        horizon=horizon,
        symbol_metadata=metadata,
    )


def _generate_framework_predictions(
    *,
    framework: StrictFrameworkSpec,
    research_frame: pd.DataFrame,
    predict_start: str,
    prediction_options: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    options = dict(prediction_options or {})
    if framework.prediction_family == "h1":
        from stockmachine.research.h1_us_equities import (
            build_h1_walk_forward_split_config,
            generate_h1_walk_forward_predictions,
        )

        split_option_keys = (
            "train_window_days",
            "validation_window_days",
            "test_window_days",
            "purge_window_days",
            "embargo_window_days",
            "roll_frequency",
        )
        split_kwargs = {
            key: options.get(key)
            for key in split_option_keys
            if key in options and options.get(key) is not None
        }
        split_config = build_h1_walk_forward_split_config(**split_kwargs) if split_kwargs else None
        model_names = tuple(options["model_names"]) if options.get("model_names") else None
        generate_kwargs: dict[str, object] = {
            "predict_start": predict_start,
            "split_config": split_config,
        }
        target_option_keys = ("target_task", "bucket_count", "positive_threshold_bps")
        target_kwargs = {
            key: options.get(key)
            for key in target_option_keys
            if key in options and options.get(key) is not None
        }
        if target_kwargs:
            from stockmachine.research.h1_us_equities import H1TargetConfig

            generate_kwargs["target_config"] = H1TargetConfig(
                task=str(target_kwargs.get("target_task", "bucket_classification")),
                bucket_count=int(target_kwargs.get("bucket_count", 2)),
                positive_threshold_bps=float(target_kwargs.get("positive_threshold_bps", 0.0)),
            )
        if model_names is not None:
            generate_kwargs["model_names"] = model_names
        if options.get("feature_version") is not None:
            generate_kwargs["feature_version"] = str(options["feature_version"])

        return generate_h1_walk_forward_predictions(research_frame, **generate_kwargs)

    return generate_walk_forward_predictions(
        research_frame,
        predict_start=predict_start,
        requested_models=tuple(options["model_names"]) if options.get("model_names") else None,
        train_window_days=options.get("train_window_days"),
        validation_window_days=options.get("validation_window_days"),
        test_window_days=options.get("test_window_days"),
        purge_window_days=options.get("purge_window_days"),
        embargo_window_days=options.get("embargo_window_days"),
        roll_frequency=options.get("roll_frequency"),
    )


def run_model_backtest_from_bundle(
    bundle: StrictResearchBundle,
    *,
    model_name: str,
    top_k: int = 10,
    overlay_config: OverlayConfig | None = None,
    route_model_whitebox: bool = True,
    turnover_control_overrides: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one backtest from a precomputed strict bundle without retraining."""

    config = overlay_config or OverlayConfig()
    effective_top_k = int(top_k)
    effective_config = config
    if route_model_whitebox:
        effective_top_k, effective_config = resolve_model_whitebox_policy(
            model_name,
            top_k=top_k,
            overlay_config=config,
        )
    framework = resolve_strict_framework(
        strategy_project=getattr(bundle, "strategy_project", None),
        horizon=bundle.horizon,
    )
    selected_predictions = bundle.predictions[bundle.predictions["model"] == model_name].copy()
    if selected_predictions.empty:
        raise RuntimeError(f"No predictions produced for model '{model_name}'.")

    engine = _build_framework_backtest_engine(
        framework=framework,
        predictions=selected_predictions,
        dataset=bundle.dataset,
        horizon=bundle.horizon,
        top_k=effective_top_k,
        overlay_config=effective_config,
        turnover_control_overrides=turnover_control_overrides,
    )

    start_date = selected_predictions["date"].min().date()
    end_date = selected_predictions["date"].max().date()
    result = engine.run(start_date=start_date, end_date=end_date)
    summary = _build_summary_mapping(result)
    records = pd.DataFrame(result.meta.get("records", []))

    if output_dir is not None:
        output_path = Path(output_dir)
        write_model_backtest_artifacts(
            output_path,
            model_name=model_name,
            summary_row=summary,
            records=records,
            predictions=selected_predictions,
        )

    return {
        "model": model_name,
        "requested_top_k": int(top_k),
        "effective_top_k": effective_top_k,
        "effective_overlay_config": effective_config,
        "route_model_whitebox": bool(route_model_whitebox),
        "summary": summary,
        "records": records,
        "artifacts_dir": str(Path(output_dir)) if output_dir is not None else None,
    }


def _build_framework_backtest_engine(
    *,
    framework: StrictFrameworkSpec,
    predictions: pd.DataFrame,
    dataset: Mapping[str, pd.DataFrame],
    horizon: int,
    top_k: int,
    overlay_config: OverlayConfig,
    turnover_control_overrides: Mapping[str, Any] | None = None,
) -> Any:
    portfolio_policy_kwargs = dict(
        top_k=top_k,
        min_close=overlay_config.min_close,
        min_median_dollar_volume_20=overlay_config.min_median_dollar_volume_20,
        max_vol_20=overlay_config.max_vol_20,
        max_positions_per_sector=overlay_config.max_positions_per_sector,
        sector_neutral=overlay_config.sector_neutral,
    )

    if framework.backtest_family == "daily_rebalance":
        from stockmachine.backtest import DailyRebalanceOpenHoldBacktestEngine

        turnover_control = dict(framework.turnover_control_defaults)
        if turnover_control_overrides:
            turnover_control.update(
                {
                    key: value
                    for key, value in turnover_control_overrides.items()
                    if value is not None
                }
            )
        portfolio_policy_kwargs.update(
            hold_rank_buffer=int(turnover_control.get("hold_rank_buffer", 0)),
            entry_rank_buffer=int(turnover_control.get("entry_rank_buffer", 0)),
            max_new_names_per_rebalance=turnover_control.get("max_new_names_per_rebalance"),
        )
        return DailyRebalanceOpenHoldBacktestEngine(
            predictions=predictions,
            daily_bar=dataset["daily_bar"],
            benchmark_index=dataset["benchmark_index"],
            signal_model=DataFrameSignalModel(predictions, horizon_bars=horizon),
            portfolio_policy=RiskAwareTopKPortfolioPolicy(**portfolio_policy_kwargs),
            execution_policy=NextOpenOrderExecutionPolicy(),
            horizon_bars=horizon,
            cost_bps_per_side=overlay_config.cost_bps_per_side,
            no_trade_band=float(turnover_control.get("no_trade_band", 0.0)),
            max_turnover=turnover_control.get("max_turnover"),
            min_weight_change=float(turnover_control.get("min_weight_change", 0.0)),
        )

    return DailyOpenHoldBacktestEngine(
        predictions=predictions,
        daily_bar=dataset["daily_bar"],
        benchmark_index=dataset["benchmark_index"],
        signal_model=DataFrameSignalModel(predictions, horizon_bars=horizon),
        portfolio_policy=RiskAwareTopKPortfolioPolicy(**portfolio_policy_kwargs),
        execution_policy=NextOpenOrderExecutionPolicy(),
        horizon_bars=horizon,
        cost_bps_per_side=overlay_config.cost_bps_per_side,
    )


def run_strict_model_sweep_from_bundle(
    bundle: StrictResearchBundle,
    *,
    model_names: Sequence[str],
    output_root: str | Path,
    top_k: int = 10,
    overlay_config: OverlayConfig | None = None,
    route_model_whitebox: bool = True,
) -> dict[str, Any]:
    """Run one strict sweep over many models while reusing the same predictions."""

    config = overlay_config or OverlayConfig()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    framework = resolve_strict_framework(
        strategy_project=getattr(bundle, "strategy_project", None),
        horizon=bundle.horizon,
    )
    protocol = _framework_protocol_dict(framework)

    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        model_dir = root / _safe_component(model_name)
        try:
            result = run_model_backtest_from_bundle(
                bundle,
                model_name=model_name,
                top_k=top_k,
                overlay_config=config,
                route_model_whitebox=route_model_whitebox,
                output_dir=model_dir,
            )
            effective_config = result["effective_overlay_config"]
            rows.append(
                {
                    "model": model_name,
                    "status": "success",
                    "predict_start": bundle.predict_start,
                    "strategy_project": framework.strategy_project,
                    "framework_id": framework.framework_id,
                    "requested_top_k": int(top_k),
                    "top_k": int(result["effective_top_k"]),
                    "horizon": bundle.horizon,
                    "artifacts_dir": str(model_dir),
                    "route_model_whitebox": bool(route_model_whitebox),
                    "effective_min_median_dollar_volume_20": effective_config.min_median_dollar_volume_20,
                    "effective_max_vol_20": effective_config.max_vol_20,
                    **result["summary"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "status": "failed",
                    "predict_start": bundle.predict_start,
                    "strategy_project": framework.strategy_project,
                    "framework_id": framework.framework_id,
                    "requested_top_k": int(top_k),
                    "top_k": int(top_k),
                    "horizon": bundle.horizon,
                    "artifacts_dir": str(model_dir),
                    "route_model_whitebox": bool(route_model_whitebox),
                    "effective_min_median_dollar_volume_20": None,
                    "effective_max_vol_20": None,
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
            )

    summary_frame = pd.DataFrame(rows)
    summary_path = root / "summary_metrics.csv"
    summary_frame = write_summary_metrics_artifact(
        summary_path,
        summary_frame,
        ordered_columns=STRICT_SUMMARY_COLUMNS,
    )
    write_json_artifact(root / "research_protocol.json", protocol)

    return {
        "ok": bool((summary_frame["status"] == "success").all()) if not summary_frame.empty else False,
        "summary_metrics_path": str(summary_path),
        "research_protocol_path": str(root / "research_protocol.json"),
        "results": rows,
    }


def summarize_backtest_records(
    records: pd.DataFrame,
    *,
    horizon: int,
    cost_bps_per_side: float | None = None,
) -> dict[str, float]:
    """Summarize one backtest record frame under one cost assumption."""

    if records.empty:
        return {
            "sessions": 0,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "benchmark_total_return": np.nan,
            "mean_turnover": np.nan,
            "mean_cost_bps": np.nan,
        }

    frame = records.copy()
    if cost_bps_per_side is None:
        net_returns = frame["net_return"].astype(float)
        mean_cost_bps = float(frame["cost_bps"].astype(float).mean())
    else:
        net_returns = frame["gross_return"].astype(float) - frame["turnover"].astype(float) * (cost_bps_per_side / 10000.0)
        mean_cost_bps = float((frame["turnover"].astype(float) * cost_bps_per_side).mean())

    benchmark_returns = frame["benchmark_return"].astype(float)
    total_return = float((1.0 + net_returns).prod() - 1.0)
    benchmark_total_return = float((1.0 + benchmark_returns).prod() - 1.0)
    annualized_return = _annualize_total_return(total_return, len(frame), horizon)
    annualized_volatility = float(net_returns.std(ddof=1) * np.sqrt(252 / horizon)) if len(frame) > 1 else np.nan
    sharpe = (
        float((net_returns.mean() / net_returns.std(ddof=1)) * np.sqrt(252 / horizon))
        if len(frame) > 1 and net_returns.std(ddof=1) > 0
        else np.nan
    )
    equity_curve = (1.0 + net_returns).cumprod()
    peaks = equity_curve.cummax()
    max_drawdown = float((equity_curve / peaks - 1.0).min()) if not equity_curve.empty else np.nan

    return {
        "sessions": int(len(frame)),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "benchmark_total_return": benchmark_total_return,
        "mean_turnover": float(frame["turnover"].astype(float).mean()),
        "mean_cost_bps": mean_cost_bps,
    }


def build_period_stability_summary(
    records: pd.DataFrame,
    *,
    model_name: str,
    horizon: int,
    period: str,
) -> pd.DataFrame:
    """Build yearly or quarterly stability summaries from backtest records."""

    if records.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "period",
                "period_label",
                "sessions",
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe",
                "max_drawdown",
                "benchmark_total_return",
                "mean_turnover",
                "mean_cost_bps",
            ]
        )

    frame = records.copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    if period == "year":
        periods = frame["entry_date"].dt.to_period("Y")
    elif period == "quarter":
        periods = frame["entry_date"].dt.to_period("Q")
    else:
        raise ValueError("period must be 'year' or 'quarter'.")

    rows: list[dict[str, Any]] = []
    for period_key, group in frame.groupby(periods, sort=True):
        summary = summarize_backtest_records(group, horizon=horizon)
        rows.append(
            {
                "model": model_name,
                "period": period,
                "period_label": str(period_key),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def build_cost_stress_summary(
    records: pd.DataFrame,
    *,
    model_name: str,
    horizon: int,
    cost_levels_bps: Sequence[float],
) -> pd.DataFrame:
    """Revalue one backtest record frame under multiple cost assumptions."""

    rows: list[dict[str, Any]] = []
    for cost_bps in cost_levels_bps:
        summary = summarize_backtest_records(records, horizon=horizon, cost_bps_per_side=float(cost_bps))
        rows.append(
            {
                "model": model_name,
                "cost_bps_per_side": float(cost_bps),
                "excess_total_return": summary["total_return"] - summary["benchmark_total_return"]
                if np.isfinite(summary["total_return"]) and np.isfinite(summary["benchmark_total_return"])
                else np.nan,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def run_topk_parameter_sweep_from_bundle(
    bundle: StrictResearchBundle,
    *,
    model_names: Sequence[str],
    top_k_values: Sequence[int],
    output_root: str | Path,
    overlay_config: OverlayConfig | None = None,
) -> pd.DataFrame:
    """Run a small top-k robustness sweep without retraining models."""

    config = overlay_config or OverlayConfig()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    framework = resolve_strict_framework(
        strategy_project=getattr(bundle, "strategy_project", None),
        horizon=bundle.horizon,
    )

    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        for top_k in top_k_values:
            run_dir = root / _safe_component(model_name) / f"top_k_{top_k}"
            result = run_model_backtest_from_bundle(
                bundle,
                model_name=model_name,
                top_k=int(top_k),
                overlay_config=config,
                route_model_whitebox=False,
                output_dir=run_dir,
            )
            rows.append(
                {
                    "model": model_name,
                    "strategy_project": framework.strategy_project,
                    "framework_id": framework.framework_id,
                    "top_k": int(top_k),
                    "artifacts_dir": str(run_dir),
                    **result["summary"],
                    "excess_total_return": result["summary"]["total_return"] - result["summary"]["benchmark_total_return"]
                    if np.isfinite(result["summary"]["total_return"]) and np.isfinite(result["summary"]["benchmark_total_return"])
                    else np.nan,
                }
            )

    summary = pd.DataFrame(rows)
    summary_path = root / "summary_metrics.csv"
    write_summary_metrics_artifact(summary_path, summary)
    return summary


def _build_summary_mapping(result: Any) -> dict[str, float]:
    return {
        "sessions": result.sessions,
        "total_return": result.total_return,
        "annualized_return": result.annualized_return,
        "annualized_volatility": result.annualized_volatility,
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "benchmark_total_return": result.meta.get("benchmark_total_return"),
        "mean_turnover": result.meta.get("mean_turnover"),
        "mean_cost_bps": result.meta.get("mean_cost_bps"),
        "mean_gross_exposure": result.meta.get("mean_gross_exposure"),
        "mean_changed_symbols": result.meta.get("mean_changed_symbols"),
    }


def _safe_component(value: str) -> str:
    cleaned = str(value).strip().replace("\\", "_").replace("/", "_")
    return cleaned or "model"


def _annualize_total_return(total_return: float, periods: int, horizon: int) -> float:
    if periods <= 0 or not np.isfinite(total_return) or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (252 / (periods * horizon)) - 1.0)
