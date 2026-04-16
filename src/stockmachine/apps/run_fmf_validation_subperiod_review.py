from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stockmachine.research.multi_asset import summarize_multi_asset_backtest_records
from stockmachine.research.strict_reports import write_csv_artifact, write_json_artifact


DEFAULT_SHORTLIST_MANIFEST = Path(
    "artifacts/fmf_validation_robustness_20260409/frozen_shortlist_manifest.csv"
)
DEFAULT_OUTPUT_ROOT = Path("artifacts/fmf_validation_subperiod_review_20260409")
DEFAULT_VALIDATION_END_DATE = "2019-12-31"


@dataclass(frozen=True, slots=True)
class SubperiodSpec:
    label: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run validation-only subperiod review on the frozen FMF shortlist."
    )
    parser.add_argument("--shortlist-manifest", default=str(DEFAULT_SHORTLIST_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--validation-end-date", default=DEFAULT_VALIDATION_END_DATE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    shortlist_manifest_path = Path(args.shortlist_manifest)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    shortlist = pd.read_csv(shortlist_manifest_path).fillna("")
    if shortlist.empty:
        raise ValueError("Shortlist manifest is empty.")

    validation_end = pd.Timestamp(args.validation_end_date)
    candidate_records = load_shortlist_records(
        shortlist,
        validation_end=validation_end,
    )
    common_window = resolve_common_window(candidate_records)
    if common_window is None:
        raise ValueError("Unable to resolve a common validation window for the shortlist.")

    subperiods = build_default_subperiods(
        common_start=common_window[0],
        validation_end=common_window[1],
    )
    subperiod_summary = build_subperiod_summary(
        shortlist=shortlist,
        candidate_records=candidate_records,
        subperiods=subperiods,
    )
    stability_overview = build_stability_overview(
        subperiod_summary=subperiod_summary,
    )

    write_csv_artifact(output_root / "subperiod_summary.csv", subperiod_summary)
    write_csv_artifact(output_root / "stability_overview.csv", stability_overview)
    _write_report(
        output_root / "subperiod_report.md",
        common_window=common_window,
        subperiods=subperiods,
        stability_overview=stability_overview,
    )
    write_json_artifact(
        output_root / "run_meta.json",
        {
            "ok": True,
            "shortlist_manifest": str(shortlist_manifest_path),
            "validation_end_date": validation_end.date().isoformat(),
            "common_window": {
                "start_date": common_window[0].date().isoformat(),
                "end_date": common_window[1].date().isoformat(),
            },
            "subperiods": [
                {
                    "label": spec.label,
                    "start_date": spec.start_date.date().isoformat(),
                    "end_date": spec.end_date.date().isoformat(),
                }
                for spec in subperiods
            ],
            "candidate_count": int(len(shortlist)),
        },
    )
    return 0


def load_shortlist_records(
    shortlist: pd.DataFrame,
    *,
    validation_end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    records_map: dict[str, pd.DataFrame] = {}
    for row in shortlist.to_dict(orient="records"):
        report_name = str(row["report_name"])
        records_path = Path(str(row["records_path"]))
        records = pd.read_csv(records_path)
        records["entry_date"] = pd.to_datetime(records["entry_date"], utc=False)
        if records.empty:
            raise ValueError(f"Shortlist candidate has empty records: {report_name}")
        max_entry = pd.Timestamp(records["entry_date"].max())
        if max_entry > validation_end:
            raise ValueError(
                f"Shortlist candidate '{report_name}' extends beyond the validation lockbox boundary: "
                f"{max_entry.date().isoformat()} > {validation_end.date().isoformat()}"
            )
        records_map[report_name] = records.copy()
    return records_map


def resolve_common_window(candidate_records: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for records in candidate_records.values():
        entry_dates = pd.to_datetime(records["entry_date"], utc=False)
        windows.append((pd.Timestamp(entry_dates.min()), pd.Timestamp(entry_dates.max())))
    if not windows:
        return None
    common_start = max(start for start, _ in windows)
    common_end = min(end for _, end in windows)
    if common_end < common_start:
        return None
    return common_start, common_end


def build_default_subperiods(
    *,
    common_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> tuple[SubperiodSpec, ...]:
    candidate_periods = (
        ("block_2014_2015", pd.Timestamp("2014-08-05"), pd.Timestamp("2015-12-31")),
        ("block_2016_2017", pd.Timestamp("2016-01-01"), pd.Timestamp("2017-12-31")),
        ("block_2018_2019", pd.Timestamp("2018-01-01"), validation_end),
    )
    subperiods: list[SubperiodSpec] = []
    for label, raw_start, raw_end in candidate_periods:
        start = max(common_start, raw_start)
        end = min(validation_end, raw_end)
        if end < start:
            continue
        subperiods.append(SubperiodSpec(label=label, start_date=start, end_date=end))
    if not subperiods:
        raise ValueError("No usable validation subperiods remain after clipping to the common window.")
    return tuple(subperiods)


def build_subperiod_summary(
    *,
    shortlist: pd.DataFrame,
    candidate_records: dict[str, pd.DataFrame],
    subperiods: tuple[SubperiodSpec, ...],
) -> pd.DataFrame:
    shortlist_by_name = shortlist.set_index("report_name", drop=False)
    rows: list[dict[str, object]] = []
    for report_name, records in candidate_records.items():
        shortlist_row = shortlist_by_name.loc[report_name]
        for spec in subperiods:
            sliced = records.loc[
                (records["entry_date"] >= spec.start_date) & (records["entry_date"] <= spec.end_date)
            ].reset_index(drop=True)
            summary = summarize_multi_asset_backtest_records(sliced, horizon=1)
            rows.append(
                {
                    "report_name": report_name,
                    "source": shortlist_row["source"],
                    "config_id": shortlist_row["config_id"],
                    "shortlist_role": shortlist_row["shortlist_role"],
                    "subperiod_label": spec.label,
                    "start_date": spec.start_date.date().isoformat(),
                    "end_date": spec.end_date.date().isoformat(),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def build_stability_overview(*, subperiod_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for report_name, group in subperiod_summary.groupby("report_name", sort=False):
        annualized_returns = group["annualized_return"].astype(float)
        sharpes = group["sharpe"].astype(float)
        drawdowns = group["max_drawdown"].astype(float)
        rows.append(
            {
                "report_name": report_name,
                "source": group["source"].iloc[0],
                "config_id": group["config_id"].iloc[0],
                "shortlist_role": group["shortlist_role"].iloc[0],
                "block_count": int(len(group)),
                "positive_block_ratio": float((annualized_returns > 0).mean()),
                "average_block_annualized_return": float(annualized_returns.mean()),
                "median_block_annualized_return": float(annualized_returns.median()),
                "worst_block_annualized_return": float(annualized_returns.min()),
                "average_block_sharpe": float(sharpes.mean()),
                "median_block_sharpe": float(sharpes.median()),
                "worst_block_sharpe": float(sharpes.min()),
                "best_block_sharpe": float(sharpes.max()),
                "worst_block_max_drawdown": float(drawdowns.min()),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["worst_block_sharpe", "average_block_sharpe", "average_block_annualized_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _write_report(
    path: Path,
    *,
    common_window: tuple[pd.Timestamp, pd.Timestamp],
    subperiods: tuple[SubperiodSpec, ...],
    stability_overview: pd.DataFrame,
) -> None:
    lines = [
        "# FMF Validation Subperiod Review",
        "",
        f"- common validation window: `{common_window[0].date().isoformat()} ~ {common_window[1].date().isoformat()}`",
        "",
        "## Subperiods",
        "",
    ]
    for spec in subperiods:
        lines.append(f"- `{spec.label}`: `{spec.start_date.date().isoformat()} ~ {spec.end_date.date().isoformat()}`")
    lines.append("")
    if not stability_overview.empty:
        leader = stability_overview.iloc[0]
        lines.extend(
            [
                "## Best By Worst-Block Sharpe",
                "",
                (
                    f"- `{leader['report_name']}`: worst-block Sharpe `{leader['worst_block_sharpe']:.3f}`, "
                    f"average-block Sharpe `{leader['average_block_sharpe']:.3f}`, "
                    f"positive-block ratio `{leader['positive_block_ratio']:.3f}`"
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
