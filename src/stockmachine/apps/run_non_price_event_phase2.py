from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ID = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT_ID / "research"
DEFAULT_SIGNAL_PANEL = (
    RESEARCH_ROOT
    / "phase3_baseline_signals_h10_probe_20260418"
    / "phase3_baseline_signal_panel_validation.csv.gz"
)
DEFAULT_NON_PRICE_PHASE1 = (
    RESEARCH_ROOT
    / "non_price_phase1_adv30m_two_factor_build_20260419"
)
DEFAULT_FILINGS = DEFAULT_NON_PRICE_PHASE1 / "sec_submission_filings_normalized.csv.gz"
DEFAULT_INSIDER_EVENTS = DEFAULT_NON_PRICE_PHASE1 / "sec_insider_net_buy_events.csv"
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "non_price_phase2_sparse_event_study_20260419"
DEFAULT_VARIANT = "adv30m_clean_core_beta_full"
DEFAULT_TARGET = "forward_beta_residual_return_5d"
DEFAULT_FORWARD_RETURN = "forward_return_5d"
DEFAULT_BENCHMARK_RETURN = "benchmark_forward_return_5d"

HARD_RED_8K_ITEMS = {
    "1.03": "bankruptcy_or_receivership",
    "2.04": "default_or_acceleration",
    "2.05": "exit_or_disposal_costs",
    "2.06": "material_impairment",
    "3.01": "delisting_notice",
    "4.01": "auditor_change",
    "4.02": "non_reliance_or_restatement",
}


def build_non_price_phase2_event_study(
    *,
    signal_panel_path: str | Path = DEFAULT_SIGNAL_PANEL,
    filings_path: str | Path = DEFAULT_FILINGS,
    insider_events_path: str | Path = DEFAULT_INSIDER_EVENTS,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant: str = DEFAULT_VARIANT,
    target_column: str = DEFAULT_TARGET,
    loser_quantile: float = 0.20,
    winner_quantile: float = 0.20,
    cluster_window_sessions: int = 30,
    min_cluster_buy_events: int = 2,
    min_cluster_buy_owners: int = 2,
) -> dict[str, Any]:
    """Evaluate v2 sparse non-price events without training or lockbox usage."""

    _validate_settings(
        loser_quantile=loser_quantile,
        winner_quantile=winner_quantile,
        cluster_window_sessions=cluster_window_sessions,
        min_cluster_buy_events=min_cluster_buy_events,
        min_cluster_buy_owners=min_cluster_buy_owners,
    )
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal = _load_signal_panel(
        Path(signal_panel_path),
        variant=variant,
        target_column=target_column,
    )
    sessions = sorted(signal["session_date"].unique().tolist())
    session_index = {session: index for index, session in enumerate(sessions)}

    hard_red_events = _build_hard_red_events(
        filings_path=Path(filings_path),
        sessions=sessions,
        signal_symbols=set(signal["symbol"]),
    )
    cluster_buy_events = _build_cluster_buy_events(
        insider_events_path=Path(insider_events_path),
        sessions=sessions,
        session_index=session_index,
        signal_symbols=set(signal["symbol"]),
        cluster_window_sessions=cluster_window_sessions,
        min_cluster_buy_events=min_cluster_buy_events,
        min_cluster_buy_owners=min_cluster_buy_owners,
    )

    hard_red_joined = _join_events_to_signal(
        hard_red_events,
        signal=signal,
        target_column=target_column,
        event_direction="short",
    )
    cluster_buy_joined = _join_events_to_signal(
        cluster_buy_events,
        signal=signal,
        target_column=target_column,
        event_direction="long",
    )
    event_rows = pd.concat([hard_red_joined, cluster_buy_joined], ignore_index=True)

    baseline = _baseline_summary(
        signal,
        target_column=target_column,
        loser_quantile=loser_quantile,
        winner_quantile=winner_quantile,
    )
    bottom_cutoff = float(signal[target_column].quantile(loser_quantile))
    top_cutoff = float(signal[target_column].quantile(1.0 - winner_quantile))
    summary = _event_summary(
        event_rows,
        target_column=target_column,
        bottom_cutoff=bottom_cutoff,
        top_cutoff=top_cutoff,
        baseline=baseline,
    )

    paths = _write_outputs(
        output_dir=output_dir,
        hard_red_events=hard_red_events,
        cluster_buy_events=cluster_buy_events,
        event_rows=event_rows,
        baseline=baseline,
        summary=summary,
    )
    rollup = {
        "created_at_utc": _utc_now(),
        "project_id": PROJECT_ID,
        "artifact_dir": output_dir.as_posix(),
        "variant": variant,
        "target_column": target_column,
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "filings_path": Path(filings_path).as_posix(),
        "insider_events_path": Path(insider_events_path).as_posix(),
        "signal_rows": int(len(signal)),
        "sessions": int(len(sessions)),
        "hard_red_event_rows": int(len(hard_red_events)),
        "cluster_buy_event_rows": int(len(cluster_buy_events)),
        "joined_event_rows": int(len(event_rows)),
        "loser_quantile": float(loser_quantile),
        "winner_quantile": float(winner_quantile),
        "cluster_window_sessions": int(cluster_window_sessions),
        "min_cluster_buy_events": int(min_cluster_buy_events),
        "min_cluster_buy_owners": int(min_cluster_buy_owners),
        "artifacts": paths,
        "lockbox_status": "validation_only_no_test_window_performance",
        "method": "sparse_event_study_for_v2_non_price_candidates",
        "limitations": [
            "This is an event-level validation diagnostic, not a portfolio or model backtest.",
            "Event availability is conservatively aligned to the next signal session after filing date.",
            "Hard-red filing categories are white-box high-confidence filters, not tuned on returns.",
            "Cluster-buy events aggregate Form 4 open-market buy evidence over a fixed session window.",
            "No test-window performance is computed.",
        ],
    }
    (output_dir / "phase2_non_price_event_study_rollup.json").write_text(
        json.dumps(rollup, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _write_memo(
        output_dir / "phase2_non_price_event_study_memo.md",
        baseline=baseline,
        summary=summary,
        rollup=rollup,
    )
    return rollup


def _load_signal_panel(path: Path, *, variant: str, target_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Signal panel not found: {path}")
    columns = [
        "variant",
        "session_date",
        "symbol",
        target_column,
        DEFAULT_FORWARD_RETURN,
        DEFAULT_BENCHMARK_RETURN,
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame = frame[frame["variant"].astype(str).eq(variant)].copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    for column in [target_column, DEFAULT_FORWARD_RETURN, DEFAULT_BENCHMARK_RETURN]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_column])
    return frame.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _build_hard_red_events(
    *,
    filings_path: Path,
    sessions: Sequence[str],
    signal_symbols: set[str],
) -> pd.DataFrame:
    if not filings_path.exists():
        raise FileNotFoundError(f"Filings file not found: {filings_path}")
    usecols = [
        "symbol",
        "cik",
        "accession_number",
        "filing_date",
        "form",
        "items",
        "primary_doc_description",
    ]
    filings = pd.read_csv(filings_path, usecols=usecols, low_memory=False)
    filings["symbol"] = filings["symbol"].astype(str).str.upper()
    filings = filings[filings["symbol"].isin(signal_symbols)].copy()
    events = []
    for row in filings.to_dict("records"):
        categories = _hard_red_categories(row)
        if not categories:
            continue
        filing_date = _iso_date(row.get("filing_date"))
        if not _date_in_session_range(filing_date, sessions):
            continue
        effective_session = _next_session_after(filing_date, sessions)
        if not effective_session:
            continue
        events.append(
            {
                "event_family": "filing_hard_red_event_v2",
                "event_direction": "short",
                "event_subtype": "|".join(categories),
                "event_strength": float(len(categories)),
                "symbol": str(row["symbol"]),
                "cik": str(row.get("cik") or "").zfill(10),
                "accession_number": str(row.get("accession_number") or ""),
                "filing_date": filing_date,
                "effective_session": effective_session,
                "form": str(row.get("form") or ""),
                "items": str(row.get("items") or ""),
            }
        )
    if not events:
        return _empty_event_frame()
    return (
        pd.DataFrame(events)
        .drop_duplicates(["event_family", "symbol", "accession_number"])
        .sort_values(["effective_session", "symbol", "accession_number"])
        .reset_index(drop=True)
    )


def _hard_red_categories(row: Mapping[str, Any]) -> list[str]:
    form = str(row.get("form") or "").upper().strip()
    items = _parse_items(row.get("items"))
    categories = []
    if form.startswith("NT 10-K") or form.startswith("NT 10-Q"):
        categories.append("late_10kq")
    if "8-K" in form:
        categories.extend(HARD_RED_8K_ITEMS[item] for item in items if item in HARD_RED_8K_ITEMS)
    return sorted(set(categories))


def _build_cluster_buy_events(
    *,
    insider_events_path: Path,
    sessions: Sequence[str],
    session_index: Mapping[str, int],
    signal_symbols: set[str],
    cluster_window_sessions: int,
    min_cluster_buy_events: int,
    min_cluster_buy_owners: int,
) -> pd.DataFrame:
    if not insider_events_path.exists():
        raise FileNotFoundError(f"Insider event file not found: {insider_events_path}")
    usecols = [
        "symbol",
        "cik",
        "accession_number",
        "filing_date",
        "effective_session",
        "buy_value_role_weighted",
        "buy_owner_count",
        "insider_buy_intensity",
    ]
    raw = pd.read_csv(insider_events_path, usecols=usecols, low_memory=False)
    raw["symbol"] = raw["symbol"].astype(str).str.upper()
    raw = raw[raw["symbol"].isin(signal_symbols)].copy()
    raw["filing_date"] = raw["filing_date"].map(_iso_date)
    raw = raw[raw["filing_date"].map(lambda value: _date_in_session_range(value, sessions))].copy()
    raw["effective_session"] = raw["effective_session"].astype(str)
    raw = raw[raw["effective_session"].isin(session_index)].copy()
    raw["buy_value_role_weighted"] = pd.to_numeric(
        raw["buy_value_role_weighted"],
        errors="coerce",
    ).fillna(0.0)
    raw["buy_owner_count"] = pd.to_numeric(raw["buy_owner_count"], errors="coerce").fillna(0.0)
    raw["insider_buy_intensity"] = pd.to_numeric(
        raw["insider_buy_intensity"],
        errors="coerce",
    ).fillna(0.0)
    raw = raw[raw["buy_value_role_weighted"].gt(0) & raw["buy_owner_count"].gt(0)].copy()
    if raw.empty:
        return _empty_event_frame()
    raw["session_index"] = raw["effective_session"].map(session_index).astype(int)

    events = []
    for symbol, group in raw.groupby("symbol", sort=True):
        group = group.sort_values(["session_index", "accession_number"]).reset_index(drop=True)
        sessions_with_buys = sorted(group["session_index"].unique().tolist())
        for session_int in sessions_with_buys:
            window = group[
                group["session_index"].between(session_int - cluster_window_sessions + 1, session_int)
            ]
            buy_events = int(window["accession_number"].nunique())
            owner_count_proxy = float(window["buy_owner_count"].sum())
            if buy_events < min_cluster_buy_events or owner_count_proxy < min_cluster_buy_owners:
                continue
            latest = window[window["session_index"].eq(session_int)]
            if latest.empty:
                continue
            filing_dates = sorted(set(str(value) for value in window["filing_date"] if str(value)))
            accessions = sorted(set(str(value) for value in window["accession_number"] if str(value)))
            events.append(
                {
                    "event_family": "insider_cluster_buy_event_v2",
                    "event_direction": "long",
                    "event_subtype": f"cluster_{cluster_window_sessions}s",
                    "event_strength": float(min(buy_events, 10)),
                    "symbol": symbol,
                    "cik": str(latest.iloc[-1].get("cik") or "").zfill(10),
                    "accession_number": "|".join(accessions[-10:]),
                    "filing_date": filing_dates[-1] if filing_dates else "",
                    "effective_session": _session_from_index(session_int, sessions),
                    "cluster_buy_event_count": buy_events,
                    "cluster_buy_owner_count_proxy": owner_count_proxy,
                    "cluster_buy_value_role_weighted": float(window["buy_value_role_weighted"].sum()),
                    "cluster_buy_intensity": float(window["insider_buy_intensity"].sum()),
                }
            )
    if not events:
        return _empty_event_frame()
    return (
        pd.DataFrame(events)
        .drop_duplicates(["event_family", "symbol", "effective_session"])
        .sort_values(["effective_session", "symbol"])
        .reset_index(drop=True)
    )


def _join_events_to_signal(
    events: pd.DataFrame,
    *,
    signal: pd.DataFrame,
    target_column: str,
    event_direction: str,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "event_family",
                "event_direction",
                "event_subtype",
                "event_strength",
                "symbol",
                "effective_session",
                target_column,
                DEFAULT_FORWARD_RETURN,
                DEFAULT_BENCHMARK_RETURN,
            ]
        )
    joined = events.merge(
        signal[
            [
                "session_date",
                "symbol",
                target_column,
                DEFAULT_FORWARD_RETURN,
                DEFAULT_BENCHMARK_RETURN,
            ]
        ],
        left_on=["effective_session", "symbol"],
        right_on=["session_date", "symbol"],
        how="inner",
        validate="many_to_one",
    ).drop(columns=["session_date"])
    joined["event_direction"] = event_direction
    joined["expected_alpha_contribution"] = np.where(
        joined["event_direction"].eq("short"),
        -joined[target_column],
        joined[target_column],
    )
    return joined.sort_values(["event_family", "effective_session", "symbol"]).reset_index(
        drop=True
    )


def _baseline_summary(
    signal: pd.DataFrame,
    *,
    target_column: str,
    loser_quantile: float,
    winner_quantile: float,
) -> pd.DataFrame:
    bottom_cutoff = float(signal[target_column].quantile(loser_quantile))
    top_cutoff = float(signal[target_column].quantile(1.0 - winner_quantile))
    return pd.DataFrame(
        [
            {
                "sample": "adv30m_signal_universe",
                "rows": int(len(signal)),
                "target_mean": float(signal[target_column].mean()),
                "target_median": float(signal[target_column].median()),
                "target_trimmed_mean_10_90": _trimmed_mean(signal[target_column]),
                "bottom_loser_cutoff": bottom_cutoff,
                "top_winner_cutoff": top_cutoff,
                "bottom_loser_share": float((signal[target_column] <= bottom_cutoff).mean()),
                "top_winner_share": float((signal[target_column] >= top_cutoff).mean()),
            }
        ]
    )


def _event_summary(
    events: pd.DataFrame,
    *,
    target_column: str,
    bottom_cutoff: float,
    top_cutoff: float,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    baseline_row = baseline.iloc[0]
    rows = []
    for (family, direction, subtype), group in events.groupby(
        ["event_family", "event_direction", "event_subtype"],
        sort=True,
    ):
        expected = (
            -group[target_column]
            if direction == "short"
            else group[target_column]
        )
        target_median = float(group[target_column].median())
        target_trimmed = _trimmed_mean(group[target_column])
        expected_median = float(expected.median())
        expected_trimmed = _trimmed_mean(expected)
        bottom_share = float((group[target_column] <= bottom_cutoff).mean())
        top_share = float((group[target_column] >= top_cutoff).mean())
        if direction == "short":
            intended_supported = (
                target_median < float(baseline_row["target_median"])
                and target_trimmed < float(baseline_row["target_trimmed_mean_10_90"])
                and bottom_share > float(baseline_row["bottom_loser_share"])
                and top_share <= float(baseline_row["top_winner_share"]) + 0.02
            )
        else:
            intended_supported = (
                target_median > float(baseline_row["target_median"])
                and target_trimmed > float(baseline_row["target_trimmed_mean_10_90"])
                and top_share > float(baseline_row["top_winner_share"])
                and bottom_share <= float(baseline_row["bottom_loser_share"]) + 0.02
            )
        rows.append(
            {
                "event_family": family,
                "event_direction": direction,
                "event_subtype": subtype,
                "rows": int(len(group)),
                "unique_symbols": int(group["symbol"].nunique()),
                "target_mean": float(group[target_column].mean()),
                "target_median": target_median,
                "target_trimmed_mean_10_90": target_trimmed,
                "expected_alpha_median": expected_median,
                "expected_alpha_trimmed_mean_10_90": expected_trimmed,
                "bottom_loser_share": bottom_share,
                "top_winner_share": top_share,
                "median_edge_vs_baseline": target_median - float(baseline_row["target_median"]),
                "trimmed_edge_vs_baseline": target_trimmed
                - float(baseline_row["target_trimmed_mean_10_90"]),
                "bottom_loser_share_edge_vs_baseline": bottom_share
                - float(baseline_row["bottom_loser_share"]),
                "top_winner_share_edge_vs_baseline": top_share
                - float(baseline_row["top_winner_share"]),
                "intended_direction_label": (
                    "intended_direction_supported"
                    if intended_supported
                    else "intended_direction_not_supported"
                ),
                "test_window_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["event_family", "event_direction", "event_subtype"]
    )


def _write_outputs(
    *,
    output_dir: Path,
    hard_red_events: pd.DataFrame,
    cluster_buy_events: pd.DataFrame,
    event_rows: pd.DataFrame,
    baseline: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, str]:
    paths = {
        "hard_red_events": output_dir / "filing_hard_red_events_v2.csv",
        "cluster_buy_events": output_dir / "insider_cluster_buy_events_v2.csv",
        "joined_event_rows": output_dir / "non_price_sparse_event_rows_validation.csv",
        "baseline": output_dir / "non_price_sparse_event_baseline_validation.csv",
        "summary": output_dir / "non_price_sparse_event_summary_validation.csv",
    }
    hard_red_events.to_csv(paths["hard_red_events"], index=False)
    cluster_buy_events.to_csv(paths["cluster_buy_events"], index=False)
    event_rows.to_csv(paths["joined_event_rows"], index=False)
    baseline.to_csv(paths["baseline"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return {key: value.as_posix() for key, value in paths.items()}


def _write_memo(
    path: Path,
    *,
    baseline: pd.DataFrame,
    summary: pd.DataFrame,
    rollup: Mapping[str, Any],
) -> None:
    lines = [
        "# Phase2 Non-Price Sparse Event Study Memo",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "Validation-only event study for v2 Non-Price candidates on the ADV30M short universe.",
        "No model training, portfolio construction, or test-window performance is computed.",
        "",
        "## Baseline",
        "",
        "| Rows | Median | Trimmed Mean | Bottom Share | Top Share |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in baseline.iterrows():
        lines.append(
            f"| {row['rows']} | {row['target_median']} | "
            f"{row['target_trimmed_mean_10_90']} | {row['bottom_loser_share']} | "
            f"{row['top_winner_share']} |"
        )
    lines.extend(
        [
            "",
            "## Event Results",
            "",
            "| Event | Direction | Subtype | Rows | Symbols | Target Median | Target Trimmed | Expected Alpha Median | Bottom Share | Top Share | Label |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['event_family']} | {row['event_direction']} | {row['event_subtype']} | "
            f"{row['rows']} | {row['unique_symbols']} | {row['target_median']} | "
            f"{row['target_trimmed_mean_10_90']} | {row['expected_alpha_median']} | "
            f"{row['bottom_loser_share']} | {row['top_winner_share']} | "
            f"{row['intended_direction_label']} |"
        )
    lines.extend(
        [
            "",
            "## Run Metadata",
            "",
            f"- variant: `{rollup['variant']}`",
            f"- target: `{rollup['target_column']}`",
            f"- signal rows: `{rollup['signal_rows']}`",
            f"- hard-red event rows: `{rollup['hard_red_event_rows']}`",
            f"- cluster-buy event rows: `{rollup['cluster_buy_event_rows']}`",
            f"- joined event rows: `{rollup['joined_event_rows']}`",
            "- lockbox status: validation only, no test-window performance.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _validate_settings(
    *,
    loser_quantile: float,
    winner_quantile: float,
    cluster_window_sessions: int,
    min_cluster_buy_events: int,
    min_cluster_buy_owners: int,
) -> None:
    if not 0 < loser_quantile < 1:
        raise ValueError("loser_quantile must be between 0 and 1.")
    if not 0 < winner_quantile < 1:
        raise ValueError("winner_quantile must be between 0 and 1.")
    if loser_quantile + winner_quantile >= 1:
        raise ValueError("loser_quantile + winner_quantile must be below 1.")
    if cluster_window_sessions <= 0:
        raise ValueError("cluster_window_sessions must be positive.")
    if min_cluster_buy_events <= 0:
        raise ValueError("min_cluster_buy_events must be positive.")
    if min_cluster_buy_owners <= 0:
        raise ValueError("min_cluster_buy_owners must be positive.")


def _empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_family",
            "event_direction",
            "event_subtype",
            "event_strength",
            "symbol",
            "cik",
            "accession_number",
            "filing_date",
            "effective_session",
        ]
    )


def _session_from_index(session_index: int, sessions: Sequence[str]) -> str:
    return sessions[session_index]


def _next_session_after(date_value: str, sessions: Sequence[str]) -> str | None:
    if not date_value:
        return None
    position = int(np.searchsorted(np.array(sessions), date_value, side="right"))
    if position >= len(sessions):
        return None
    return sessions[position]


def _date_in_session_range(date_value: str, sessions: Sequence[str]) -> bool:
    if not date_value or not sessions:
        return False
    return sessions[0] <= date_value <= sessions[-1]


def _parse_items(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    return [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]


def _iso_date(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.date().isoformat()


def _trimmed_mean(series: pd.Series, lower: float = 0.10, upper: float = 0.90) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    if len(clean) < 10:
        return float(clean.mean())
    low = clean.quantile(lower)
    high = clean.quantile(upper)
    trimmed = clean[(clean >= low) & (clean <= high)]
    return float(trimmed.mean()) if not trimmed.empty else float(clean.mean())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run v2 sparse Non-Price event studies.")
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_SIGNAL_PANEL))
    parser.add_argument("--filings-path", default=str(DEFAULT_FILINGS))
    parser.add_argument("--insider-events-path", default=str(DEFAULT_INSIDER_EVENTS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--target-column", default=DEFAULT_TARGET)
    parser.add_argument("--loser-quantile", type=float, default=0.20)
    parser.add_argument("--winner-quantile", type=float, default=0.20)
    parser.add_argument("--cluster-window-sessions", type=int, default=30)
    parser.add_argument("--min-cluster-buy-events", type=int, default=2)
    parser.add_argument("--min-cluster-buy-owners", type=int, default=2)
    args = parser.parse_args(argv)

    result = build_non_price_phase2_event_study(
        signal_panel_path=args.signal_panel_path,
        filings_path=args.filings_path,
        insider_events_path=args.insider_events_path,
        output_root=args.output_root,
        variant=args.variant,
        target_column=args.target_column,
        loser_quantile=args.loser_quantile,
        winner_quantile=args.winner_quantile,
        cluster_window_sessions=args.cluster_window_sessions,
        min_cluster_buy_events=args.min_cluster_buy_events,
        min_cluster_buy_owners=args.min_cluster_buy_owners,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
