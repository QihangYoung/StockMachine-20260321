"""Phase6W: h1 pairwise ranker with short-horizon price features.

Phase6V showed that h1 supervision alone does not help when the feature set is
mostly 5-60 day price state. Phase6W keeps the h1 label and pairwise framework,
but adds point-in-time features that are more naturally aligned with next-day
open-to-open residual returns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from stockmachine.apps import run_pure_alpha_phase6p as phase6p
from stockmachine.apps import run_pure_alpha_phase6v as phase6v
from stockmachine.apps.run_pure_alpha_phase3 import (
    DEFAULT_ADJ_FACTOR_GLOBS,
    DEFAULT_DAILY_GLOBS,
    _load_jsonl_frames,
)
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT


OUTDIR = RESEARCH_ROOT / "phase6w_h1_enhanced_feature_pairwise_ranker_20260513"
OUTPUT_PREFIX = "phase6w_h1_enhanced"
LABEL_COLUMN = "true_size_style_sic2_h1_enhanced_residual"
RANDOM_SEED = 20260513

SHORT_HORIZON_FEATURES = (
    "close_reversal_1d",
    "close_reversal_2d",
    "close_reversal_3d",
    "open_to_close_return_1d",
    "overnight_gap_1d",
    "high_low_range_1d",
    "close_location_1d",
    "abs_close_return_1d",
    "abs_open_to_close_return_1d",
    "abs_overnight_gap_1d",
    "realized_vol_3d",
    "realized_vol_5d",
    "volume_shock_1d",
    "dollar_volume_shock_1d",
    "range_shock_1d",
)

ENHANCED_FEATURES = tuple(dict.fromkeys([*phase6p.BASE_FEATURES, *SHORT_HORIZON_FEATURES]))


def main(argv: Sequence[str] | None = None) -> None:
    _configure_phase6v()
    rollup = phase6v.build_phase6v_h1_sparse_pairwise_ranker()
    print(json.dumps(rollup, indent=2, ensure_ascii=True))


def _configure_phase6v() -> None:
    phase6v.OUTDIR = OUTDIR
    phase6v.PHASE_NAME = "phase6w_h1_enhanced_feature_pairwise_ranker"
    phase6v.OUTPUT_PREFIX = OUTPUT_PREFIX
    phase6v.MEMO_TITLE = "Phase6W H1 Pairwise Ranker with Short-Horizon Features"
    phase6v.LABEL_COLUMN = LABEL_COLUMN
    phase6v.RANDOM_SEED = RANDOM_SEED
    phase6v.FEATURE_COLUMNS = ENHANCED_FEATURES
    phase6v.PREBUILT_PANEL_PATH = OUTDIR / f"{OUTPUT_PREFIX}_label_feature_panel.csv.gz"
    phase6v.WRITE_PAIR_SAMPLE = False
    phase6v._add_extra_features = _add_short_horizon_price_features


def _add_short_horizon_price_features(panel: pd.DataFrame) -> pd.DataFrame:
    feature_frame = _load_short_horizon_price_feature_panel(
        symbols=panel["symbol"].astype(str).unique(),
        end_date=phase6v.VALIDATION_PRICE_END,
    )
    return panel.merge(feature_frame, on=["session_date", "symbol"], how="left")


def _load_short_horizon_price_feature_panel(symbols: Sequence[str], end_date: str) -> pd.DataFrame:
    symbol_set = set(map(str, symbols))
    daily = _load_jsonl_frames(
        DEFAULT_DAILY_GLOBS,
        columns=(
            "session_date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "dollar_volume",
        ),
        symbols=symbol_set,
        end_date=end_date,
    )
    adj = _load_jsonl_frames(
        DEFAULT_ADJ_FACTOR_GLOBS,
        columns=("session_date", "symbol", "price_adjust_factor"),
        symbols=symbol_set,
        end_date=end_date,
    )
    if daily.empty:
        raise ValueError("No daily bars available for Phase6W short-horizon features.")

    frame = daily.merge(adj, on=["session_date", "symbol"], how="left")
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    frame["symbol"] = frame["symbol"].astype(str)
    frame["price_adjust_factor"] = pd.to_numeric(
        frame["price_adjust_factor"],
        errors="coerce",
    ).fillna(1.0)
    for column in ("open", "high", "low", "close", "volume", "dollar_volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("open", "high", "low", "close"):
        frame[f"adjusted_{column}"] = frame[column] * frame["price_adjust_factor"]

    frame["dollar_volume"] = frame["dollar_volume"].where(
        frame["dollar_volume"].notna(),
        frame["close"] * frame["volume"],
    )
    frame = frame.sort_values(["symbol", "session_date"]).drop_duplicates(
        ["symbol", "session_date"],
        keep="last",
    )
    grouped = frame.groupby("symbol", group_keys=False)

    frame["close_return_1d"] = grouped["adjusted_close"].pct_change(1)
    frame["close_return_2d"] = grouped["adjusted_close"].pct_change(2)
    frame["close_return_3d"] = grouped["adjusted_close"].pct_change(3)
    frame["close_reversal_1d"] = -frame["close_return_1d"]
    frame["close_reversal_2d"] = -frame["close_return_2d"]
    frame["close_reversal_3d"] = -frame["close_return_3d"]
    frame["open_to_close_return_1d"] = frame["adjusted_close"] / frame["adjusted_open"] - 1.0
    frame["overnight_gap_1d"] = frame["adjusted_open"] / grouped["adjusted_close"].shift(1) - 1.0
    frame["high_low_range_1d"] = frame["adjusted_high"] / frame["adjusted_low"] - 1.0

    range_denominator = frame["adjusted_high"] - frame["adjusted_low"]
    frame["close_location_1d"] = np.where(
        range_denominator > 0,
        (frame["adjusted_close"] - frame["adjusted_low"]) / range_denominator - 0.5,
        np.nan,
    )
    frame["abs_close_return_1d"] = frame["close_return_1d"].abs()
    frame["abs_open_to_close_return_1d"] = frame["open_to_close_return_1d"].abs()
    frame["abs_overnight_gap_1d"] = frame["overnight_gap_1d"].abs()
    frame["realized_vol_3d"] = grouped["close_return_1d"].transform(
        lambda series: series.rolling(3, min_periods=2).std()
    )
    frame["realized_vol_5d"] = grouped["close_return_1d"].transform(
        lambda series: series.rolling(5, min_periods=3).std()
    )

    volume_median_20_prior = grouped["volume"].transform(
        lambda series: series.shift(1).rolling(20, min_periods=10).median()
    )
    dollar_volume_median_20_prior = grouped["dollar_volume"].transform(
        lambda series: series.shift(1).rolling(20, min_periods=10).median()
    )
    range_median_20_prior = grouped["high_low_range_1d"].transform(
        lambda series: series.shift(1).rolling(20, min_periods=10).median()
    )
    frame["volume_shock_1d"] = np.log(frame["volume"] / volume_median_20_prior)
    frame["dollar_volume_shock_1d"] = np.log(frame["dollar_volume"] / dollar_volume_median_20_prior)
    frame["range_shock_1d"] = np.log(frame["high_low_range_1d"] / range_median_20_prior)

    feature_frame = frame[["session_date", "symbol", *SHORT_HORIZON_FEATURES]].copy()
    return feature_frame.replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":
    main()
