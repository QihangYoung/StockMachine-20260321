from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from stockmachine.ingestion.storage import StorageLayout
from stockmachine.research.builders import prepare_model_frame
from stockmachine.research.builders.common import build_linear_model_pipeline, build_tree_model_pipeline
from stockmachine.research.p1_rigor import (
    build_cost_stress_summary,
    build_period_stability_summary,
    build_strict_research_bundle,
    run_model_backtest_from_bundle,
)
from stockmachine.research.protocols import get_h1_research_protocol
from stockmachine.research.strict_frameworks import (
    build_framework_overlay_config,
    build_framework_promotion_gate,
    resolve_framework_cost_stress_levels,
    resolve_strict_framework,
)
from stockmachine.research.strict_preflight import StrictResearchSourceInputs
from stockmachine.research.strict_reports import (
    write_csv_artifact,
    write_json_artifact,
    write_summary_metrics_artifact,
)
from stockmachine.research.splitting import WalkForwardSplit, WalkForwardSplitConfig, build_walk_forward_splits
from stockmachine.research.universe import (
    DEFAULT_RESEARCH_UNIVERSE_NAME,
    build_point_in_time_metadata_history,
)
from stockmachine.research.us_equities_baseline import BENCHMARK_SYMBOL, OverlayConfig, build_price_panel_from_silver

H1_FEATURE_COLUMNS_V1: tuple[str, ...] = (
    "gap_1",
    "gap_z_20",
    "intraday_return",
    "ret_1d",
    "ret_2d",
    "mom_3",
    "range_1d",
    "range_5",
    "vol_5",
    "vol_20",
    "volume_ratio_5",
    "volume_ratio_20",
    "rel_ret_1d",
    "rel_mom_3",
    "sector_rel_ret_1d",
    "sector_rel_mom_3",
)
H1_FEATURE_FAMILY_BENCHMARK_CONTEXT: tuple[str, ...] = (
    "benchmark_gap_1",
    "rel_gap_1",
    "rel_intraday_return",
)
H1_FEATURE_FAMILY_RISK_SCALED: tuple[str, ...] = (
    "gap_to_vol_20",
    "ret_1d_to_vol_20",
    "true_range_1d",
    "atr_5",
    "downside_vol_20",
)
H1_FEATURE_FAMILY_CANDLE_SHAPE: tuple[str, ...] = (
    "range_position_1d",
    "body_to_range_1d",
)
H1_FEATURE_FAMILY_SECTOR_INTRADAY: tuple[str, ...] = (
    "sector_rel_gap_1",
    "sector_rel_intraday_return",
)
H1_FEATURE_COLUMNS_V2: tuple[str, ...] = (
    H1_FEATURE_COLUMNS_V1
    + H1_FEATURE_FAMILY_BENCHMARK_CONTEXT
    + H1_FEATURE_FAMILY_RISK_SCALED
    + H1_FEATURE_FAMILY_CANDLE_SHAPE
    + H1_FEATURE_FAMILY_SECTOR_INTRADAY
)
H1_FEATURE_COLUMNS_V3: tuple[str, ...] = (
    H1_FEATURE_COLUMNS_V1
    + H1_FEATURE_FAMILY_BENCHMARK_CONTEXT
    + H1_FEATURE_FAMILY_RISK_SCALED
)
H1_DEFAULT_FEATURE_VERSION = "v3"
H1_FEATURE_COLUMNS: tuple[str, ...] = H1_FEATURE_COLUMNS_V3

H1_BASE_MODEL_NAMES: tuple[str, ...] = ("ridge", "hist_gbm", "extra_trees")
TRADING_DAYS_PER_MONTH = 21


@dataclass(slots=True, frozen=True)
class H1TurnoverControlConfig:
    no_trade_band: float = 0.05
    max_turnover: float = 1.0
    min_weight_change: float = 0.02
    hold_rank_buffer: int = 2
    entry_rank_buffer: int = 2
    max_new_names_per_rebalance: int = 2


@dataclass(slots=True, frozen=True)
class H1PromotionGateConfig:
    min_sharpe: float = 0.5
    min_annualized_return_at_20bps: float = 0.0
    max_drawdown_floor: float = -0.35
    max_mean_turnover: float = 1.25
    min_positive_year_ratio: float = 0.5


@dataclass(slots=True, frozen=True)
class H1TargetConfig:
    task: str = "bucket_classification"
    bucket_count: int = 2
    positive_threshold_bps: float = 0.0

    @property
    def label_column(self) -> str:
        if self.task == "point_regression":
            return "target"
        return _h1_bucket_label_column(self.bucket_count)

    @property
    def positive_threshold_return(self) -> float:
        return float(self.positive_threshold_bps) / 10_000.0

    @property
    def is_classification(self) -> bool:
        return self.task == "bucket_classification"


def resolve_h1_feature_columns(feature_version: str = H1_DEFAULT_FEATURE_VERSION) -> tuple[str, ...]:
    normalized = str(feature_version).strip().lower()
    if normalized == "v1":
        return H1_FEATURE_COLUMNS_V1
    if normalized == "v2":
        return H1_FEATURE_COLUMNS_V2
    if normalized == "v3":
        return H1_FEATURE_COLUMNS_V3
    raise ValueError(f"Unsupported h1 feature_version '{feature_version}'. Expected one of: v1, v2, v3.")


def build_h1_research_frame(
    price_data: pd.DataFrame,
    *,
    benchmark_symbol: str = BENCHMARK_SYMBOL,
    symbol_metadata: pd.DataFrame | None = None,
    drop_unlabeled_rows: bool = True,
) -> pd.DataFrame:
    price_data = _ensure_adjusted_price_columns(price_data.copy())
    benchmark = (
        price_data.loc[price_data["symbol"] == benchmark_symbol, ["date", "open", "close", "adj_open"]]
        .rename(columns={"open": "benchmark_open", "close": "benchmark_close", "adj_open": "benchmark_adj_open"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    benchmark["benchmark_prev_close"] = benchmark["benchmark_close"].shift(1)
    benchmark["benchmark_ret_1d"] = benchmark["benchmark_close"].pct_change(1, fill_method=None)
    benchmark["benchmark_mom_3"] = benchmark["benchmark_close"].pct_change(3, fill_method=None)
    benchmark["benchmark_gap_1"] = benchmark["benchmark_open"] / benchmark["benchmark_prev_close"] - 1.0
    benchmark["benchmark_intraday_return"] = benchmark["benchmark_close"] / benchmark["benchmark_open"] - 1.0
    benchmark["benchmark_future_return"] = (
        benchmark["benchmark_adj_open"].shift(-2) / benchmark["benchmark_adj_open"].shift(-1) - 1.0
    )

    panel = price_data.loc[price_data["symbol"] != benchmark_symbol].copy()
    panel = panel.merge(
        benchmark[["date", "benchmark_ret_1d", "benchmark_mom_3", "benchmark_gap_1", "benchmark_future_return"]],
        on="date",
        how="left",
    )
    panel = panel.merge(
        benchmark[["date", "benchmark_intraday_return"]],
        on="date",
        how="left",
    )

    group = panel.groupby("symbol", group_keys=False)
    panel["prev_close"] = group["close"].shift(1)
    panel["ret_1d"] = group["close"].pct_change(1, fill_method=None)
    panel["ret_2d"] = group["close"].pct_change(2, fill_method=None)
    panel["mom_3"] = group["close"].pct_change(3, fill_method=None)
    panel["intraday_return"] = panel["close"] / panel["open"] - 1.0
    panel["range_1d"] = panel["high"] / panel["low"] - 1.0
    panel["range_5"] = group["range_1d"].rolling(5).mean().reset_index(level=0, drop=True)
    panel["gap_1"] = panel["open"] / panel["prev_close"] - 1.0
    gap_mean_20 = group["gap_1"].rolling(20).mean().reset_index(level=0, drop=True)
    gap_std_20 = group["gap_1"].rolling(20).std().reset_index(level=0, drop=True)
    panel["gap_z_20"] = ((panel["gap_1"] - gap_mean_20) / gap_std_20.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    panel["gap_z_20"] = panel["gap_z_20"].fillna(0.0)
    panel["vol_5"] = group["ret_1d"].rolling(5).std().reset_index(level=0, drop=True)
    panel["vol_20"] = group["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)
    panel["dollar_volume"] = panel["close"] * panel["volume"]
    panel["median_dollar_volume_20"] = (
        group["dollar_volume"].rolling(20).median().reset_index(level=0, drop=True)
    )
    panel["volume_ratio_5"] = panel["volume"] / group["volume"].rolling(5).mean().reset_index(level=0, drop=True)
    panel["volume_ratio_20"] = panel["volume"] / group["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    panel["rel_ret_1d"] = panel["ret_1d"] - panel["benchmark_ret_1d"]
    panel["rel_mom_3"] = panel["mom_3"] - panel["benchmark_mom_3"]
    panel["rel_gap_1"] = panel["gap_1"] - panel["benchmark_gap_1"]
    panel["rel_intraday_return"] = panel["intraday_return"] - panel["benchmark_intraday_return"]
    panel["gap_to_vol_20"] = _safe_divide(panel["gap_1"], panel["vol_20"])
    panel["ret_1d_to_vol_20"] = _safe_divide(panel["ret_1d"], panel["vol_20"])
    day_range = (panel["high"] - panel["low"]).replace(0.0, np.nan)
    panel["range_position_1d"] = (((panel["close"] - panel["low"]) / day_range) - 0.5).fillna(0.0) * 2.0
    panel["body_to_range_1d"] = ((panel["close"] - panel["open"]) / day_range).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    true_range = pd.concat(
        [
            (panel["high"] - panel["low"]).abs(),
            (panel["high"] - panel["prev_close"]).abs(),
            (panel["low"] - panel["prev_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    panel["true_range_1d"] = _safe_divide(true_range, panel["prev_close"])
    panel["atr_5"] = group["true_range_1d"].rolling(5).mean().reset_index(level=0, drop=True)
    negative_ret_sq = panel["ret_1d"].clip(upper=0.0).pow(2)
    panel["future_return"] = group["adj_open"].shift(-2) / group["adj_open"].shift(-1) - 1.0
    panel["target"] = panel["future_return"] - panel["benchmark_future_return"]
    panel[_h1_bucket_label_column(2)] = _build_h1_bucket_labels(panel["target"], bucket_count=2, positive_threshold_bps=0.0)

    if symbol_metadata is not None:
        merge_keys = ["symbol"]
        if "date" in symbol_metadata.columns:
            merge_keys = ["date", "symbol"]
        panel = panel.merge(symbol_metadata, on=merge_keys, how="left", indicator="_metadata_match")
        if merge_keys == ["date", "symbol"]:
            panel = panel.loc[panel["_metadata_match"] == "both"].copy()
        panel = panel.drop(columns="_metadata_match")
        panel["sector"] = panel["sector"].fillna("Unknown")
        panel["industry"] = panel["industry"].fillna("Unknown")
    else:
        panel["sector"] = "Unknown"
        panel["industry"] = "Unknown"

    sector_group = panel.groupby(["date", "sector"], group_keys=False)
    panel["sector_rel_ret_1d"] = panel["ret_1d"] - sector_group["ret_1d"].transform("mean")
    panel["sector_rel_mom_3"] = panel["mom_3"] - sector_group["mom_3"].transform("mean")
    panel["sector_rel_gap_1"] = panel["gap_1"] - sector_group["gap_1"].transform("mean")
    panel["sector_rel_intraday_return"] = panel["intraday_return"] - sector_group["intraday_return"].transform("mean")

    downside_vol_20 = (
        negative_ret_sq.groupby(panel["symbol"])
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
        .pow(0.5)
    )
    panel["downside_vol_20"] = downside_vol_20

    required_columns = list(H1_FEATURE_COLUMNS_V2)
    if drop_unlabeled_rows:
        required_columns += ["future_return", "target"]
    panel = panel.dropna(subset=required_columns).reset_index(drop=True)
    return panel


def get_h1_model_builders(
    *,
    task: str = "bucket_classification",
    feature_version: str = H1_DEFAULT_FEATURE_VERSION,
) -> Mapping[str, Callable[[], object]]:
    feature_columns = resolve_h1_feature_columns(feature_version)
    if task == "point_regression":
        return {
            "ridge": lambda: build_linear_model_pipeline(
                Ridge(alpha=1.0),
                feature_columns=feature_columns,
            ),
            "hist_gbm": lambda: build_tree_model_pipeline(
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=4,
                    max_iter=200,
                    min_samples_leaf=40,
                    random_state=7,
                )
            ),
            "extra_trees": lambda: build_tree_model_pipeline(
                ExtraTreesRegressor(
                    bootstrap=False,
                    max_depth=8,
                    max_features="sqrt",
                    min_samples_leaf=40,
                    n_estimators=400,
                    n_jobs=-1,
                    random_state=7,
                )
            ),
        }

    return {
        "ridge": lambda: build_linear_model_pipeline(
            LogisticRegression(
                C=1.0,
                max_iter=2_000,
                random_state=7,
                solver="lbfgs",
            ),
            feature_columns=feature_columns,
        ),
        "hist_gbm": lambda: build_tree_model_pipeline(
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=4,
                max_iter=200,
                min_samples_leaf=40,
                random_state=7,
            )
        ),
        "extra_trees": lambda: build_tree_model_pipeline(
            ExtraTreesClassifier(
                bootstrap=False,
                max_depth=8,
                max_features="sqrt",
                min_samples_leaf=40,
                n_estimators=400,
                n_jobs=-1,
                random_state=7,
            )
        ),
    }


def build_h1_walk_forward_split_config(
    *,
    train_window_days: int | None = None,
    validation_window_days: int | None = None,
    test_window_days: int | None = None,
    purge_window_days: int | None = None,
    embargo_window_days: int | None = None,
    roll_frequency: str | None = None,
) -> WalkForwardSplitConfig:
    protocol = get_h1_research_protocol()
    return WalkForwardSplitConfig(
        train_window_days=train_window_days or protocol.walk_forward.train_window_months * TRADING_DAYS_PER_MONTH,
        validation_window_days=validation_window_days or protocol.walk_forward.validation_window_months * TRADING_DAYS_PER_MONTH,
        test_window_days=test_window_days or protocol.walk_forward.test_window_months * TRADING_DAYS_PER_MONTH,
        purge_window_days=protocol.walk_forward.purge_window_sessions if purge_window_days is None else purge_window_days,
        embargo_window_days=protocol.walk_forward.embargo_window_sessions if embargo_window_days is None else embargo_window_days,
        roll_frequency=roll_frequency or protocol.walk_forward.roll_frequency,
    )


def generate_h1_walk_forward_predictions(
    panel: pd.DataFrame,
    *,
    predict_start: str,
    model_names: Sequence[str] = H1_BASE_MODEL_NAMES,
    split_config: WalkForwardSplitConfig | None = None,
    target_config: H1TargetConfig | None = None,
    feature_version: str = H1_DEFAULT_FEATURE_VERSION,
) -> pd.DataFrame:
    predict_start_ts = pd.Timestamp(predict_start).normalize()
    config = split_config or build_h1_walk_forward_split_config()
    resolved_target = target_config or H1TargetConfig()
    splits = build_walk_forward_splits(panel, config=config)

    prediction_frames: list[pd.DataFrame] = []
    for split in splits:
        test_frame = split.test_frame.loc[split.test_frame["date"] >= predict_start_ts].copy()
        if test_frame.empty:
            continue
        train_frame = pd.concat([split.train_frame, split.validation_frame], ignore_index=True)
        train_frame = train_frame.sort_values(["date", "symbol"]).reset_index(drop=True)

        for model_name in model_names:
            predictions = fit_predict_h1_base_model(
                model_name,
                train_frame=train_frame,
                test_frame=test_frame,
                target_config=resolved_target,
                feature_version=feature_version,
            )
            prediction_frames.append(_attach_split_metadata(predictions, split))

    if not prediction_frames:
        empty_columns = [
            "date",
            "symbol",
            "sector",
            "industry",
            "close",
            "vol_20",
            "median_dollar_volume_20",
            "target",
            "future_return",
            "benchmark_future_return",
            "score",
            "confidence",
            "probability_positive",
            "predicted_bucket",
            "classification_confidence",
            "model",
            "split_fold_index",
            "split_anchor_date",
            "split_train_start",
            "split_train_end",
            "split_validation_start",
            "split_validation_end",
            "split_test_start",
            "split_test_end",
        ]
        if resolved_target.is_classification:
            empty_columns.insert(8, resolved_target.label_column)
        return pd.DataFrame(
            columns=empty_columns
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["split_anchor_date"] = pd.to_datetime(predictions["split_anchor_date"], utc=False)
    predictions = predictions.sort_values(
        ["model", "date", "symbol", "split_anchor_date", "score"],
        ascending=[True, True, True, True, False],
    )
    predictions = predictions.drop_duplicates(subset=["model", "date", "symbol"], keep="last")
    return predictions.sort_values(["model", "date", "score"], ascending=[True, True, False]).reset_index(drop=True)


def fit_predict_h1_base_model(
    name: str,
    *,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_config: H1TargetConfig | None = None,
    feature_version: str = H1_DEFAULT_FEATURE_VERSION,
) -> pd.DataFrame:
    prepared_train = prepare_model_frame(train_frame)
    prepared_test = prepare_model_frame(test_frame)
    resolved_target = target_config or H1TargetConfig()
    feature_columns = resolve_h1_feature_columns(feature_version)
    builders = get_h1_model_builders(task=resolved_target.task, feature_version=feature_version)
    try:
        model = builders[name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported h1 model '{name}'.") from exc

    features_train = prepared_train[list(feature_columns)]
    features_test = prepared_test[list(feature_columns)]
    if resolved_target.is_classification:
        target_labels = _build_h1_bucket_labels(
            prepared_train["target"],
            bucket_count=resolved_target.bucket_count,
            positive_threshold_bps=resolved_target.positive_threshold_bps,
        )
        model.fit(features_train, target_labels.astype(int))
        scores = _predict_h1_positive_class_probability(model, features_test)
    else:
        model.fit(features_train, prepared_train["target"].astype(float))
        scores = np.asarray(model.predict(features_test), dtype=float).reshape(-1)
    return _assemble_predictions(
        prepared_test,
        np.asarray(scores),
        name,
        target_config=resolved_target,
    )


def run_h1_baseline_sweep(
    *,
    predict_start: str = "2025-01-01",
    model_names: Sequence[str] = H1_BASE_MODEL_NAMES,
    top_k: int = 10,
    output_dir: str | Path = "artifacts/us_equities_h1_baseline",
    overlay_config: OverlayConfig | None = None,
    turnover_control: H1TurnoverControlConfig | None = None,
    target_config: H1TargetConfig | None = None,
    cost_levels_bps: Sequence[float] | None = None,
    gate_config: H1PromotionGateConfig | None = None,
    strategy_project: str = "us_equities_h1",
    layout: StorageLayout | None = None,
    split_config: WalkForwardSplitConfig | None = None,
    cache_dir: str | Path | None = None,
    reuse_cache: bool = True,
    rebuild_cache: bool = False,
    source_inputs: StrictResearchSourceInputs | None = None,
    feature_version: str = H1_DEFAULT_FEATURE_VERSION,
) -> dict[str, object]:
    framework = resolve_strict_framework(strategy_project=strategy_project, horizon=1)
    config = overlay_config or build_framework_overlay_config(framework)
    turnover = turnover_control or H1TurnoverControlConfig(**dict(framework.turnover_control_defaults))
    if target_config is None:
        prediction_defaults = dict(framework.prediction_defaults)
        target = H1TargetConfig(
            task=str(prediction_defaults.get("target_task", "bucket_classification")),
            bucket_count=int(prediction_defaults.get("bucket_count", 2)),
            positive_threshold_bps=float(prediction_defaults.get("positive_threshold_bps", 0.0)),
        )
    else:
        target = target_config
    gate = gate_config or H1PromotionGateConfig(**dict(framework.promotion_gate_defaults))
    resolved_cost_levels = resolve_framework_cost_stress_levels(framework, override_levels=tuple(cost_levels_bps) if cost_levels_bps else None)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prediction_options = {
        "model_names": tuple(model_names),
        "train_window_days": split_config.train_window_days if split_config is not None else None,
        "validation_window_days": split_config.validation_window_days if split_config is not None else None,
        "test_window_days": split_config.test_window_days if split_config is not None else None,
        "purge_window_days": split_config.purge_window_days if split_config is not None else None,
        "embargo_window_days": split_config.embargo_window_days if split_config is not None else None,
        "roll_frequency": split_config.roll_frequency if split_config is not None else None,
        "target_task": target.task,
        "bucket_count": target.bucket_count,
        "positive_threshold_bps": target.positive_threshold_bps,
        "feature_version": feature_version,
    }
    bundle = build_strict_research_bundle(
        predict_start=predict_start,
        horizon=1,
        strategy_project=strategy_project,
        prediction_options=prediction_options,
        layout=layout or StorageLayout(),
        cache_dir=cache_dir,
        reuse_cache=reuse_cache,
        rebuild_cache=rebuild_cache,
        source_inputs=source_inputs,
    )
    predictions = bundle.predictions.copy()
    research_frame = bundle.research_frame.copy()
    dataset = bundle.dataset
    write_csv_artifact(output_path / "predictions.csv", predictions)
    write_csv_artifact(output_path / "research_frame.csv", research_frame)
    research_contract = get_h1_research_protocol().to_dict()
    research_contract["target_definition"] = asdict(target)
    write_json_artifact(output_path / "research_protocol.json", research_contract)
    write_json_artifact(output_path / "target_definition.json", asdict(target))

    rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    cost_frames: list[pd.DataFrame] = []
    for model_name in model_names:
        model_dir = output_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        selected_predictions = predictions[predictions["model"] == model_name].copy()
        if selected_predictions.empty:
            continue

        result = run_model_backtest_from_bundle(
            bundle,
            model_name=model_name,
            top_k=top_k,
            overlay_config=config,
            turnover_control_overrides=asdict(turnover),
            output_dir=model_dir,
        )
        records = result["records"].copy()
        summary_row = {
            "model": model_name,
            **result["summary"],
            "artifacts_dir": str(model_dir),
        }
        rows.append(summary_row)
        yearly_frames.append(build_period_stability_summary(records, model_name=model_name, horizon=1, period="year"))
        cost_frames.append(
            build_cost_stress_summary(records, model_name=model_name, horizon=1, cost_levels_bps=resolved_cost_levels)
        )

    summary_frame = write_summary_metrics_artifact(output_path / "summary_metrics.csv", rows)
    yearly_summary = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    write_csv_artifact(output_path / "yearly_summary.csv", yearly_summary)
    cost_summary = pd.concat(cost_frames, ignore_index=True) if cost_frames else pd.DataFrame()
    write_csv_artifact(output_path / "cost_stress_summary.csv", cost_summary)
    benchmark_summary = build_h1_benchmark_summary(predictions=predictions, benchmark_index=dataset["benchmark_index"])
    write_csv_artifact(output_path / "benchmark_summary.csv", benchmark_summary)

    gate_payload = build_framework_promotion_gate(
        framework,
        summary_frame=summary_frame,
        yearly_summary=yearly_summary,
        cost_summary=cost_summary,
        gate_config=gate,
    )
    if gate_payload is None:
        gate_payload = {"strategy_project": framework.strategy_project, "framework_id": framework.framework_id, "models": []}
    write_json_artifact(output_path / "promotion_gate.json", gate_payload)

    return {
        "ok": not summary_frame.empty,
        "artifacts_dir": str(output_path),
        "summary_metrics_path": str(output_path / "summary_metrics.csv"),
        "yearly_summary_path": str(output_path / "yearly_summary.csv"),
        "cost_stress_summary_path": str(output_path / "cost_stress_summary.csv"),
        "benchmark_summary_path": str(output_path / "benchmark_summary.csv"),
        "promotion_gate_path": str(output_path / "promotion_gate.json"),
        "research_protocol": get_h1_research_protocol().to_dict(),
        "target_definition": asdict(target),
        "feature_version": feature_version,
        "overlay_config": asdict(config),
        "turnover_control": asdict(turnover),
        "cache": {
            "enabled": cache_dir is not None,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "bundle_cache_hit": bool(getattr(bundle, "bundle_cache_hit", False)),
            "prediction_cache_hit": bool(getattr(bundle, "prediction_cache_hit", False)),
            "bundle_cache_key": getattr(bundle, "bundle_cache_key", None),
            "prediction_cache_key": getattr(bundle, "prediction_cache_key", None),
            "rebuild_cache": bool(rebuild_cache),
        },
        "promotion_gate": gate_payload,
    }


def build_h1_benchmark_summary(
    *,
    predictions: pd.DataFrame,
    benchmark_index: pd.DataFrame,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "benchmark",
                "sessions",
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe",
                "max_drawdown",
                "mean_turnover",
                "mean_cost_bps",
            ]
        )

    benchmark_frame = benchmark_index.rename(columns={"session_date": "date"}).copy()
    benchmark_frame["date"] = pd.to_datetime(benchmark_frame["date"])
    benchmark_frame = benchmark_frame.sort_values("date").reset_index(drop=True)
    if "adj_open" not in benchmark_frame.columns:
        benchmark_frame["adj_open"] = benchmark_frame["open"]

    prediction_dates = sorted(predictions["date"].dt.normalize().unique())
    returns: list[float] = []
    for idx in range(0, len(prediction_dates)):
        entry_index = idx + 1
        exit_index = idx + 2
        if entry_index >= len(prediction_dates) or exit_index >= len(prediction_dates):
            break
        entry_date = pd.Timestamp(prediction_dates[entry_index]).normalize()
        exit_date = pd.Timestamp(prediction_dates[exit_index]).normalize()
        entry_row = benchmark_frame.loc[benchmark_frame["date"] == entry_date]
        exit_row = benchmark_frame.loc[benchmark_frame["date"] == exit_date]
        if entry_row.empty or exit_row.empty:
            continue
        entry_open = float(entry_row["adj_open"].iloc[0])
        exit_open = float(exit_row["adj_open"].iloc[0])
        if entry_open <= 0:
            continue
        returns.append(exit_open / entry_open - 1.0)

    spy_series = pd.Series(returns, dtype=float)
    cash_series = pd.Series(np.zeros(len(returns), dtype=float))
    return pd.DataFrame(
        [
            _benchmark_summary_row("spy_next_open_hold", spy_series),
            _benchmark_summary_row("cash_zero", cash_series),
        ]
    )


def build_h1_promotion_gate(
    *,
    summary_frame: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    gate_config: H1PromotionGateConfig,
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for row in summary_frame.itertuples(index=False):
        yearly = yearly_summary[yearly_summary["model"] == row.model].copy()
        cost_20 = cost_summary[
            (cost_summary["model"] == row.model)
            & (cost_summary["cost_bps_per_side"].astype(float) == 20.0)
        ].copy()
        positive_year_ratio = float((yearly["total_return"].astype(float) > 0).mean()) if not yearly.empty else np.nan
        annualized_return_20bps = (
            float(cost_20["annualized_return"].iloc[0])
            if not cost_20.empty and pd.notna(cost_20["annualized_return"].iloc[0])
            else np.nan
        )

        checks = {
            "sharpe": bool(pd.notna(row.sharpe) and float(row.sharpe) >= gate_config.min_sharpe),
            "cost_20bps_annualized_return": bool(
                pd.notna(annualized_return_20bps)
                and float(annualized_return_20bps) >= gate_config.min_annualized_return_at_20bps
            ),
            "max_drawdown": bool(
                pd.notna(row.max_drawdown) and float(row.max_drawdown) >= gate_config.max_drawdown_floor
            ),
            "mean_turnover": bool(
                pd.notna(row.mean_turnover) and float(row.mean_turnover) <= gate_config.max_mean_turnover
            ),
            "positive_year_ratio": bool(
                pd.notna(positive_year_ratio) and positive_year_ratio >= gate_config.min_positive_year_ratio
            ),
        }
        decisions.append(
            {
                "model": row.model,
                "promote_to_paper_research": all(checks.values()),
                "checks": checks,
                "metrics": {
                    "annualized_return": row.annualized_return,
                    "sharpe": row.sharpe,
                    "max_drawdown": row.max_drawdown,
                    "mean_turnover": row.mean_turnover,
                    "positive_year_ratio": positive_year_ratio,
                    "annualized_return_at_20bps": annualized_return_20bps,
                },
            }
        )

    return {
        "gate_config": asdict(gate_config),
        "models": decisions,
    }


def _assemble_predictions(
    frame: pd.DataFrame,
    scores: np.ndarray,
    model_name: str,
    *,
    target_config: H1TargetConfig,
) -> pd.DataFrame:
    label_column = target_config.label_column
    realized_label = (
        _build_h1_bucket_labels(
            frame["target"],
            bucket_count=target_config.bucket_count,
            positive_threshold_bps=target_config.positive_threshold_bps,
        ).astype("Int64")
        if target_config.is_classification
        else pd.Series(pd.NA, index=frame.index, dtype="Int64")
    )
    prediction_frame = frame[
        [
            "date",
            "symbol",
            "sector",
            "industry",
            "close",
            "vol_20",
            "median_dollar_volume_20",
            "target",
            "future_return",
            "benchmark_future_return",
        ]
    ].copy()
    if target_config.is_classification:
        prediction_frame[label_column] = realized_label.to_numpy()
    prediction_frame["score"] = scores
    if target_config.is_classification:
        prediction_frame["probability_positive"] = scores
        prediction_frame["predicted_bucket"] = (scores >= 0.5).astype(int)
        prediction_frame["classification_confidence"] = np.maximum(scores, 1.0 - scores)
    else:
        prediction_frame["probability_positive"] = np.nan
        prediction_frame["predicted_bucket"] = (
            pd.Series(scores, index=prediction_frame.index) > target_config.positive_threshold_return
        ).astype(int)
        prediction_frame["classification_confidence"] = np.nan
    prediction_frame["confidence"] = (
        pd.Series(scores, index=prediction_frame.index)
        .groupby(prediction_frame["date"])
        .rank(pct=True)
        .to_numpy()
    )
    prediction_frame["model"] = model_name
    return prediction_frame


def _attach_split_metadata(frame: pd.DataFrame, split: WalkForwardSplit) -> pd.DataFrame:
    tagged = frame.copy()
    tagged["split_fold_index"] = split.fold_index
    tagged["split_anchor_date"] = split.anchor_date
    tagged["split_train_start"] = split.train_start
    tagged["split_train_end"] = split.train_end
    tagged["split_validation_start"] = split.validation_start
    tagged["split_validation_end"] = split.validation_end
    tagged["split_test_start"] = split.test_start
    tagged["split_test_end"] = split.test_end
    return tagged


def _benchmark_summary_row(name: str, returns: pd.Series) -> dict[str, float | str]:
    total_return = _compound_total_return(returns)
    annualized_return = _annualize_total_return(total_return, len(returns), 1)
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else np.nan
    sharpe = _annualized_sharpe(returns, 1)
    max_drawdown = _max_drawdown_from_returns(returns)
    return {
        "benchmark": name,
        "sessions": int(len(returns)),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "mean_turnover": 0.0,
        "mean_cost_bps": 0.0,
    }


def _ensure_adjusted_price_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "adj_close" not in normalized.columns:
        normalized["adj_close"] = normalized.get("close")
    if "price_adjust_factor" not in normalized.columns:
        normalized["price_adjust_factor"] = np.nan
    if "cash_dividend" not in normalized.columns:
        normalized["cash_dividend"] = normalized.get("dividends", 0.0)
    if "split_factor" not in normalized.columns:
        normalized["split_factor"] = normalized.get("stock_splits", 1.0)

    valid_close = normalized["close"].notna() & (normalized["close"] > 0)
    inferred_factor = np.where(valid_close, normalized["adj_close"] / normalized["close"], np.nan)
    normalized["price_adjust_factor"] = (
        normalized["price_adjust_factor"].where(normalized["price_adjust_factor"].notna(), inferred_factor)
    )
    normalized["price_adjust_factor"] = normalized["price_adjust_factor"].fillna(1.0)
    normalized["cash_dividend"] = normalized["cash_dividend"].fillna(0.0)
    normalized["split_factor"] = normalized["split_factor"].replace(0.0, 1.0).fillna(1.0)
    normalized["adj_open"] = normalized["open"] * normalized["price_adjust_factor"]
    normalized["adj_close"] = normalized["close"] * normalized["price_adjust_factor"]
    return normalized


def _h1_bucket_label_column(bucket_count: int) -> str:
    return f"target_bucket_{int(bucket_count)}"


def _build_h1_bucket_labels(
    target: pd.Series,
    *,
    bucket_count: int,
    positive_threshold_bps: float,
) -> pd.Series:
    if int(bucket_count) != 2:
        raise NotImplementedError("h1 bucket classification currently supports only 2 buckets.")

    threshold = float(positive_threshold_bps) / 10_000.0
    labels = pd.Series(pd.NA, index=target.index, dtype="Int64")
    numeric_target = pd.to_numeric(target, errors="coerce")
    valid_mask = numeric_target.notna()
    labels.loc[valid_mask] = (numeric_target.loc[valid_mask] > threshold).astype(int)
    return labels


def _predict_h1_positive_class_probability(model: object, features_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features_test), dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
        return probabilities.reshape(-1)

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(features_test), dtype=float).reshape(-1)
        return 1.0 / (1.0 + np.exp(-decision))

    predicted = np.asarray(model.predict(features_test), dtype=float).reshape(-1)
    return np.clip(predicted, 0.0, 1.0)


def _annualized_sharpe(returns: pd.Series, horizon: int) -> float:
    if returns.std(ddof=1) == 0 or returns.empty:
        return np.nan
    return float((returns.mean() / returns.std(ddof=1)) * np.sqrt(252 / horizon))


def _compound_total_return(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    return float((1.0 + returns).prod() - 1.0)


def _annualize_total_return(total_return: float, periods: int, horizon: int) -> float:
    if periods <= 0 or not np.isfinite(total_return) or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (252 / (periods * horizon)) - 1.0)


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    peaks = equity.cummax()
    drawdowns = equity / peaks - 1.0
    return float(drawdowns.min())


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    quotient = numerator / denominator.replace(0.0, np.nan)
    return quotient.replace([np.inf, -np.inf], np.nan).fillna(0.0)
