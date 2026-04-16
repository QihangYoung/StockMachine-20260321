from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from stockmachine.apps.run_fmf_validation_baseline_experiment import (
    DEFAULT_VALIDATION_END_DATE,
    _require_return_columns,
    load_symbol_returns_for_symbols,
    slice_return_frame,
)
from stockmachine.apps.run_fmf_validation_regime_overlay_experiment import (
    FMF_VALIDATION_UNIVERSE_NAME,
    build_validation_regime_state_frame,
)
from stockmachine.ingestion.storage import StorageLayout
from stockmachine.research.multi_asset import FMF_VALIDATION_BUCKETS, build_bucket_return_frame
from stockmachine.research.strict_reports import write_csv_artifact, write_json_artifact


DEFAULT_TREND_LOOKBACK_GRID = (42, 63, 84, 126)
DEFAULT_VOL_LOOKBACK_GRID = (21, 42, 63)
DEFAULT_DRAWDOWN_THRESHOLD_GRID = (-0.05, -0.06, -0.08)
DEFAULT_HIGH_VOL_THRESHOLD_GRID = (0.12, 0.14, 0.16)
DEFAULT_REFERENCE_SEGMENTS = (
    ("2014-08-05", "2015-07-31", "risk_on", "late_cycle_bull"),
    ("2015-08-03", "2016-02-29", "defensive", "china_commodity_scare"),
    ("2016-03-01", "2018-09-30", "risk_on", "reflation_growth"),
    ("2018-10-01", "2018-12-31", "defensive", "q4_2018_selloff"),
    ("2019-01-01", "2019-12-31", "risk_on", "fed_pivot_rebound"),
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run validation-only sensitivity checks for the FMF regime detector."
    )
    parser.add_argument("--data-root", default=None, help="Optional storage root for silver tables.")
    parser.add_argument(
        "--output-root",
        default="artifacts/fmf_validation_regime_detector_sensitivity",
        help="Artifact directory for validation-only detector sensitivity outputs.",
    )
    parser.add_argument("--start-date", default=None, help="Optional inclusive return start date.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive return end date.")
    parser.add_argument(
        "--validation-end-date",
        default=DEFAULT_VALIDATION_END_DATE,
        help="Inclusive validation-window end date. Lockbox test remains closed.",
    )
    parser.add_argument("--benchmark-column", default="equity_us")
    parser.add_argument(
        "--trend-lookback-grid",
        default=",".join(str(value) for value in DEFAULT_TREND_LOOKBACK_GRID),
    )
    parser.add_argument(
        "--vol-lookback-grid",
        default=",".join(str(value) for value in DEFAULT_VOL_LOOKBACK_GRID),
    )
    parser.add_argument(
        "--drawdown-threshold-grid",
        default=",".join(f"{value:.2f}" for value in DEFAULT_DRAWDOWN_THRESHOLD_GRID),
    )
    parser.add_argument(
        "--high-vol-threshold-grid",
        default=",".join(f"{value:.2f}" for value in DEFAULT_HIGH_VOL_THRESHOLD_GRID),
    )
    return parser


def parse_int_grid(raw_value: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in raw_value.split(",") if token.strip())
    if not values:
        raise ValueError("Expected at least one integer grid value.")
    return values


def parse_float_grid(raw_value: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in raw_value.split(",") if token.strip())
    if not values:
        raise ValueError("Expected at least one float grid value.")
    return values


def build_reference_regime_segments() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "segment_start": pd.Timestamp(start_date),
                "segment_end": pd.Timestamp(end_date),
                "reference_state": state_name,
                "reference_label": label,
            }
            for start_date, end_date, state_name, label in DEFAULT_REFERENCE_SEGMENTS
        ]
    )
    return frame


def expand_segment_frame_to_daily(
    segments: pd.DataFrame,
    *,
    start_column: str,
    end_column: str,
    state_column: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    daily_parts: list[pd.Series] = []
    for row in segments.to_dict(orient="records"):
        daily_index = pd.date_range(row[start_column], row[end_column], freq="D")
        daily_parts.append(pd.Series(row[state_column], index=daily_index, dtype="object"))
    if not daily_parts:
        raise ValueError("segments must contain at least one row.")
    daily = pd.concat(daily_parts).sort_index()
    daily = daily.loc[(daily.index >= start_date) & (daily.index <= end_date)]
    daily = daily.loc[~daily.index.duplicated(keep="last")]
    return daily


def summarize_detector_against_reference(
    regime_state_frame: pd.DataFrame,
    *,
    reference_daily: pd.Series,
) -> dict[str, object]:
    overlay_daily = pd.Series(
        regime_state_frame["overlay_state"].to_numpy(),
        index=pd.to_datetime(regime_state_frame["entry_date"], utc=False),
        dtype="object",
    ).sort_index()
    common = pd.DataFrame(
        {
            "overlay_state": overlay_daily,
            "reference_state": reference_daily,
        }
    ).dropna()
    common["match"] = common["overlay_state"] == common["reference_state"]

    q4_mask = (common.index >= pd.Timestamp("2018-10-01")) & (common.index <= pd.Timestamp("2018-12-31"))
    defensive_mask = common["reference_state"] == "defensive"
    q4_reference_defensive_mask = q4_mask & defensive_mask
    q4_overlay_defensive_mask = q4_mask & (common["overlay_state"] == "defensive")

    summary = {
        "days": int(len(common)),
        "daily_match_ratio": float(common["match"].mean()),
        "q4_2018_match_ratio": float(common.loc[q4_mask, "match"].mean()),
        "q4_2018_defensive_recall": (
            float((common.loc[q4_reference_defensive_mask, "overlay_state"] == "defensive").mean())
            if int(q4_reference_defensive_mask.sum()) > 0
            else float("nan")
        ),
        "q4_2018_defensive_precision": (
            float((common.loc[q4_overlay_defensive_mask, "reference_state"] == "defensive").mean())
            if int(q4_overlay_defensive_mask.sum()) > 0
            else float("nan")
        ),
        "overlay_risk_on_share": float((common["overlay_state"] == "risk_on").mean()),
        "overlay_defensive_share": float((common["overlay_state"] == "defensive").mean()),
        "reference_risk_on_share": float((common["reference_state"] == "risk_on").mean()),
        "reference_defensive_share": float((common["reference_state"] == "defensive").mean()),
    }
    return summary


def rank_detector_summary(summary_frame: pd.DataFrame) -> pd.DataFrame:
    ranked = summary_frame.copy()
    ranked["rank_score"] = (
        0.50 * ranked["q4_2018_match_ratio"]
        + 0.30 * ranked["q4_2018_defensive_recall"]
        + 0.20 * ranked["daily_match_ratio"]
    )
    ranked = ranked.sort_values(
        ["rank_score", "q4_2018_match_ratio", "q4_2018_defensive_recall", "daily_match_ratio"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    ordered_columns = [
        "rank",
        "rank_score",
        "trend_lookback",
        "vol_lookback",
        "drawdown_threshold",
        "high_vol_threshold",
        "daily_match_ratio",
        "q4_2018_match_ratio",
        "q4_2018_defensive_recall",
        "q4_2018_defensive_precision",
        "overlay_risk_on_share",
        "overlay_defensive_share",
        "reference_risk_on_share",
        "reference_defensive_share",
        "days",
    ]
    return ranked.loc[:, ordered_columns]


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    validation_end = pd.Timestamp(args.validation_end_date)
    layout = StorageLayout(root=Path(args.data_root)) if args.data_root else StorageLayout()
    required_symbols = tuple(
        dict.fromkeys(symbol for bucket in FMF_VALIDATION_BUCKETS for symbol in bucket.symbols)
    )
    symbol_returns = load_symbol_returns_for_symbols(layout=layout, symbols=required_symbols)
    _require_return_columns(
        symbol_returns,
        required_columns=required_symbols,
        context="FMF detector sensitivity universe",
    )
    symbol_returns = symbol_returns.loc[:, list(required_symbols)]
    bucket_returns = build_bucket_return_frame(symbol_returns, FMF_VALIDATION_BUCKETS)
    bucket_returns = slice_return_frame(
        bucket_returns,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    validation_bucket_returns = bucket_returns.loc[bucket_returns.index <= validation_end].copy()
    if validation_bucket_returns.empty:
        raise ValueError("No validation rows remain after applying validation_end_date.")

    reference_segments = build_reference_regime_segments()
    validation_start = pd.Timestamp(validation_bucket_returns.index.min())
    reference_daily = expand_segment_frame_to_daily(
        reference_segments,
        start_column="segment_start",
        end_column="segment_end",
        state_column="reference_state",
        start_date=validation_start,
        end_date=validation_end,
    )

    trend_lookback_grid = parse_int_grid(args.trend_lookback_grid)
    vol_lookback_grid = parse_int_grid(args.vol_lookback_grid)
    drawdown_threshold_grid = parse_float_grid(args.drawdown_threshold_grid)
    high_vol_threshold_grid = parse_float_grid(args.high_vol_threshold_grid)

    summary_rows: list[dict[str, object]] = []
    best_regime_state_frame: pd.DataFrame | None = None
    best_key: tuple[int, int, float, float] | None = None
    best_score = float("-inf")
    for trend_lookback, vol_lookback, drawdown_threshold, high_vol_threshold in product(
        trend_lookback_grid,
        vol_lookback_grid,
        drawdown_threshold_grid,
        high_vol_threshold_grid,
    ):
        regime_state_frame = build_validation_regime_state_frame(
            validation_bucket_returns,
            benchmark_column=str(args.benchmark_column),
            trend_lookback=int(trend_lookback),
            vol_lookback=int(vol_lookback),
            drawdown_threshold=float(drawdown_threshold),
            high_vol_threshold=float(high_vol_threshold),
        )
        summary = summarize_detector_against_reference(
            regime_state_frame,
            reference_daily=reference_daily,
        )
        rank_score = (
            0.50 * summary["q4_2018_match_ratio"]
            + 0.30 * summary["q4_2018_defensive_recall"]
            + 0.20 * summary["daily_match_ratio"]
        )
        row = {
            "trend_lookback": int(trend_lookback),
            "vol_lookback": int(vol_lookback),
            "drawdown_threshold": float(drawdown_threshold),
            "high_vol_threshold": float(high_vol_threshold),
            "rank_score": float(rank_score),
            **summary,
        }
        summary_rows.append(row)
        if rank_score > best_score:
            best_score = float(rank_score)
            best_regime_state_frame = regime_state_frame
            best_key = (
                int(trend_lookback),
                int(vol_lookback),
                float(drawdown_threshold),
                float(high_vol_threshold),
            )

    summary_frame = pd.DataFrame(summary_rows)
    ranked_summary = rank_detector_summary(summary_frame)
    write_csv_artifact(output_root / "detector_sensitivity_summary.csv", ranked_summary)
    write_csv_artifact(output_root / "reference_regime_segments.csv", reference_segments)

    if best_regime_state_frame is None or best_key is None:
        raise RuntimeError("Detector sensitivity search produced no candidates.")
    write_csv_artifact(output_root / "best_regime_state_frame.csv", best_regime_state_frame)
    write_json_artifact(
        output_root / "run_meta.json",
        {
            "experiment_family": "fmf_validation_regime_detector_sensitivity",
            "universe_name": FMF_VALIDATION_UNIVERSE_NAME,
            "lockbox_policy": {
                "test_window_locked": True,
                "test_window_exposed": False,
            },
            "validation_window": {
                "start_date": validation_start.date().isoformat(),
                "end_date": validation_end.date().isoformat(),
            },
            "benchmark_column": str(args.benchmark_column),
            "grids": {
                "trend_lookback": list(trend_lookback_grid),
                "vol_lookback": list(vol_lookback_grid),
                "drawdown_threshold": list(drawdown_threshold_grid),
                "high_vol_threshold": list(high_vol_threshold_grid),
            },
            "best_detector": {
                "trend_lookback": best_key[0],
                "vol_lookback": best_key[1],
                "drawdown_threshold": best_key[2],
                "high_vol_threshold": best_key[3],
                "summary_row": ranked_summary.iloc[0].to_dict(),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
