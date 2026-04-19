from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ID = "us_equities_pure_alpha_h5"
DEFAULT_POSITIONS_PATH = (
    "artifacts/strategy_projects/us_equities_pure_alpha_h5/research/"
    "phase4s_sector_industry_mf_residual_v2_20260419/phase4s_positions_validation.csv.gz"
)
DEFAULT_EVENTS_PATH = (
    "artifacts/strategy_projects/us_equities_pure_alpha_h5/research/"
    "non_price_phase2_sparse_event_study_20260419/non_price_sparse_event_rows_validation.csv"
)
DEFAULT_OUTPUT_ROOT = (
    "artifacts/strategy_projects/us_equities_pure_alpha_h5/research/"
    "non_price_phase3_position_overlay_20260419"
)
DEFAULT_TARGET_COLUMN = "forward_beta_residual_return_5d"


def build_non_price_position_overlay(
    *,
    positions_path: str | Path = DEFAULT_POSITIONS_PATH,
    events_path: str | Path = DEFAULT_EVENTS_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    target_column: str = DEFAULT_TARGET_COLUMN,
    min_event_rows: int = 30,
) -> dict[str, Any]:
    """Overlay sparse non-price events on existing long/short validation positions."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = _load_positions(Path(positions_path), target_column=target_column)
    events = _load_event_keys(Path(events_path))
    positions = _add_signed_alpha(positions, target_column=target_column)

    exposures = _position_event_exposures(positions, events)
    summary = _overlay_summary(
        positions,
        events,
        min_event_rows=min_event_rows,
    )
    paths = _write_outputs(output_dir=output_dir, exposures=exposures, summary=summary)
    rollup = {
        "created_at_utc": _utc_now(),
        "project_id": PROJECT_ID,
        "artifact_dir": output_dir.as_posix(),
        "positions_path": str(positions_path),
        "events_path": str(events_path),
        "target_column": target_column,
        "positions_rows": int(len(positions)),
        "event_key_rows": int(len(events)),
        "position_event_exposure_rows": int(len(exposures)),
        "summary_rows": int(len(summary)),
        "min_event_rows": int(min_event_rows),
        "artifacts": paths,
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "non_price_sparse_event_overlay_on_existing_phase4s_positions",
        "limitations": [
            "This is a diagnostic overlay, not a new portfolio backtest.",
            "Position construction is inherited from existing Phase4S validation artifacts.",
            "No test-window performance, transaction costs, borrow costs, or turnover model is computed.",
        ],
    }
    rollup_path = output_dir / "phase3_non_price_position_overlay_rollup.json"
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    memo_path = output_dir / "phase3_non_price_position_overlay_memo.md"
    _write_memo(memo_path, rollup=rollup, summary=summary)
    rollup["artifacts"]["rollup"] = rollup_path.as_posix()
    rollup["artifacts"]["memo"] = memo_path.as_posix()
    return rollup


def _load_positions(path: Path, *, target_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Positions file not found: {path}")
    usecols = [
        "session_date",
        "portfolio",
        "side",
        "symbol",
        target_column,
    ]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    missing = [column for column in usecols if column not in frame.columns]
    if missing:
        raise ValueError(f"Positions file is missing required columns: {missing}")
    frame["session_date"] = frame["session_date"].astype(str)
    frame["portfolio"] = frame["portfolio"].astype(str)
    frame["side"] = frame["side"].astype(str).str.lower()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    return frame.dropna(subset=[target_column]).reset_index(drop=True)


def _load_event_keys(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Event file not found: {path}")
    usecols = [
        "effective_session",
        "symbol",
        "event_family",
        "event_direction",
        "event_subtype",
        "event_strength",
    ]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    missing = [column for column in usecols if column not in raw.columns]
    if missing:
        raise ValueError(f"Event file is missing required columns: {missing}")
    raw["session_date"] = raw["effective_session"].astype(str)
    raw["symbol"] = raw["symbol"].astype(str).str.upper()
    raw["event_family"] = raw["event_family"].astype(str)
    raw["event_direction"] = raw["event_direction"].astype(str)
    raw["event_subtype"] = raw["event_subtype"].astype(str)
    raw["event_strength"] = pd.to_numeric(raw["event_strength"], errors="coerce").fillna(0.0)
    grouped = raw.groupby(["session_date", "symbol", "event_family"], as_index=False).agg(
        event_direction=("event_direction", _join_unique),
        event_subtypes=("event_subtype", _join_unique),
        event_count=("event_subtype", "size"),
        event_strength=("event_strength", "sum"),
    )
    return grouped.sort_values(["session_date", "symbol", "event_family"]).reset_index(drop=True)


def _add_signed_alpha(positions: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    frame = positions.copy()
    frame["signed_alpha"] = np.where(
        frame["side"].eq("short"),
        -frame[target_column],
        frame[target_column],
    )
    return frame


def _position_event_exposures(positions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    exposures = positions.merge(events, on=["session_date", "symbol"], how="inner")
    if exposures.empty:
        return exposures
    exposures["intended_use"] = [
        _event_intent(event_family, side)
        for event_family, side in zip(exposures["event_family"], exposures["side"], strict=False)
    ]
    return exposures.sort_values(["session_date", "portfolio", "side", "symbol", "event_family"])


def _overlay_summary(
    positions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    min_event_rows: int,
) -> pd.DataFrame:
    if positions.empty or events.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    families = sorted(events["event_family"].unique())
    portfolio_values = ["ALL"] + sorted(positions["portfolio"].unique())
    for event_family in families:
        family_keys = events[events["event_family"].eq(event_family)][
            ["session_date", "symbol", "event_count", "event_strength"]
        ].copy()
        family_keys = family_keys.groupby(["session_date", "symbol"], as_index=False).agg(
            event_count=("event_count", "sum"),
            event_strength=("event_strength", "sum"),
        )
        family_keys["has_event"] = True
        enriched = positions.merge(family_keys, on=["session_date", "symbol"], how="left")
        enriched["has_event"] = enriched["has_event"].eq(True)
        enriched["event_count"] = enriched["event_count"].fillna(0).astype(int)
        enriched["event_strength"] = enriched["event_strength"].fillna(0.0)
        for portfolio in portfolio_values:
            portfolio_frame = (
                enriched if portfolio == "ALL" else enriched[enriched["portfolio"].eq(portfolio)]
            )
            if portfolio_frame.empty:
                continue
            for side in ("long", "short"):
                side_frame = portfolio_frame[portfolio_frame["side"].eq(side)]
                if side_frame.empty:
                    continue
                event_frame = side_frame[side_frame["has_event"]]
                no_event_frame = side_frame[~side_frame["has_event"]]
                if event_frame.empty or no_event_frame.empty:
                    continue
                edge_mean = _safe_mean(event_frame["signed_alpha"]) - _safe_mean(
                    no_event_frame["signed_alpha"]
                )
                edge_median = _safe_median(event_frame["signed_alpha"]) - _safe_median(
                    no_event_frame["signed_alpha"]
                )
                edge_hit_rate = _hit_rate(event_frame["signed_alpha"]) - _hit_rate(
                    no_event_frame["signed_alpha"]
                )
                intended_use = _event_intent(event_family, side)
                rows.append(
                    {
                        "portfolio": portfolio,
                        "side": side,
                        "event_family": event_family,
                        "intended_use": intended_use,
                        "positions_rows": int(len(side_frame)),
                        "event_position_rows": int(len(event_frame)),
                        "event_symbol_days": int(
                            event_frame[["session_date", "symbol"]].drop_duplicates().shape[0]
                        ),
                        "no_event_position_rows": int(len(no_event_frame)),
                        "event_signed_alpha_mean": _safe_mean(event_frame["signed_alpha"]),
                        "no_event_signed_alpha_mean": _safe_mean(no_event_frame["signed_alpha"]),
                        "edge_signed_alpha_mean": edge_mean,
                        "event_signed_alpha_median": _safe_median(event_frame["signed_alpha"]),
                        "no_event_signed_alpha_median": _safe_median(no_event_frame["signed_alpha"]),
                        "edge_signed_alpha_median": edge_median,
                        "event_hit_rate": _hit_rate(event_frame["signed_alpha"]),
                        "no_event_hit_rate": _hit_rate(no_event_frame["signed_alpha"]),
                        "edge_hit_rate": edge_hit_rate,
                        "mean_event_count": _safe_mean(event_frame["event_count"]),
                        "mean_event_strength": _safe_mean(event_frame["event_strength"]),
                        "overlay_label": _overlay_label(
                            intended_use=intended_use,
                            event_rows=len(event_frame),
                            min_event_rows=min_event_rows,
                            edge_mean=edge_mean,
                            edge_median=edge_median,
                            edge_hit_rate=edge_hit_rate,
                        ),
                        "test_window_used": False,
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["event_family", "portfolio", "side"],
    )


def _event_intent(event_family: str, side: str) -> str:
    side = str(side).lower()
    if event_family == "filing_hard_red_event_v2":
        return "short_confirmation" if side == "short" else "long_veto"
    if event_family == "insider_cluster_buy_event_v2":
        return "long_confirmation" if side == "long" else "short_veto"
    return "diagnostic"


def _overlay_label(
    *,
    intended_use: str,
    event_rows: int,
    min_event_rows: int,
    edge_mean: float,
    edge_median: float,
    edge_hit_rate: float,
) -> str:
    if event_rows < min_event_rows:
        return "insufficient_event_rows"
    if intended_use.endswith("confirmation"):
        supported = edge_mean > 0 and edge_median > 0 and edge_hit_rate >= 0
    elif intended_use.endswith("veto"):
        supported = edge_mean < 0 and edge_median < 0 and edge_hit_rate <= 0
    else:
        supported = False
    return "overlay_supported" if supported else "overlay_not_supported"


def _write_outputs(
    *,
    output_dir: Path,
    exposures: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, str]:
    paths = {
        "position_event_exposures": output_dir / "non_price_position_event_exposures_validation.csv.gz",
        "summary": output_dir / "non_price_position_overlay_summary_validation.csv",
    }
    exposures.to_csv(paths["position_event_exposures"], index=False, compression="gzip")
    summary.to_csv(paths["summary"], index=False)
    return {key: value.as_posix() for key, value in paths.items()}


def _write_memo(path: Path, *, rollup: dict[str, Any], summary: pd.DataFrame) -> None:
    lines = [
        "# Phase3 Non-Price Position Overlay Memo",
        "",
        f"Generated: {rollup['created_at_utc']}",
        "",
        "## Scope",
        "",
        "Validation-only overlay of sparse non-price events on existing Phase4S long/short positions.",
        "This evaluates confirmation/veto usefulness, not standalone factor power.",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("No event overlays were available.")
    else:
        display = summary[summary["portfolio"].eq("ALL")].copy()
        if display.empty:
            display = summary.copy()
        lines.extend(
            [
                "| Event | Side | Intended Use | Event Rows | Event Mean | No-Event Mean | Edge Mean | Event Median | No-Event Median | Edge Median | Label |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in display.to_dict("records"):
            lines.append(
                f"| {row['event_family']} | {row['side']} | {row['intended_use']} | "
                f"{row['event_position_rows']} | {row['event_signed_alpha_mean']} | "
                f"{row['no_event_signed_alpha_mean']} | {row['edge_signed_alpha_mean']} | "
                f"{row['event_signed_alpha_median']} | {row['no_event_signed_alpha_median']} | "
                f"{row['edge_signed_alpha_median']} | {row['overlay_label']} |"
            )
    lines.extend(
        [
            "",
            "## Run Metadata",
            "",
            f"- positions rows: `{rollup['positions_rows']}`",
            f"- event key rows: `{rollup['event_key_rows']}`",
            f"- exposure rows: `{rollup['position_event_exposure_rows']}`",
            f"- min event rows for label: `{rollup['min_event_rows']}`",
            "- lockbox status: validation only, no test-window performance.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _join_unique(values: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in values if str(value) and str(value) != "nan"}))


def _safe_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def _safe_median(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.median()) if not clean.empty else float("nan")


def _hit_rate(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.gt(0).mean()) if not clean.empty else float("nan")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions-path", default=DEFAULT_POSITIONS_PATH)
    parser.add_argument("--events-path", default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--min-event-rows", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_non_price_position_overlay(
        positions_path=args.positions_path,
        events_path=args.events_path,
        output_root=args.output_root,
        target_column=args.target_column,
        min_event_rows=args.min_event_rows,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
