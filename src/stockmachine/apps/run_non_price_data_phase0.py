from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ID = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT_ID / "research"
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "non_price_phase0_two_factor_data_prep_20260419"
DEFAULT_UNIVERSE_MANIFEST = (
    RESEARCH_ROOT / "top1000_data_backfill_20260417" / "top1000_manifest.csv"
)
DEFAULT_PHASE1_MEMBERSHIP_PATH = (
    RESEARCH_ROOT
    / "phase1_universe_builder_20260418"
    / "phase1_candidate_universe_membership_validation.csv.gz"
)
DEFAULT_RAW_ROOT = Path("data") / "raw" / "sec" / "reference"
DEFAULT_SUBMISSIONS_RAW_ROOT = Path("data") / "raw" / "sec" / "submissions"

SEC_COMPANY_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSION_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"

VALIDATION_START = "2013-08-01"
PURE_ALPHA_SESSION_START = "2013-08-05"
VALIDATION_END = "2019-12-31"
TEST_START = "2020-01-02"
TEST_END = "2026-04-08"


def run_phase0_data_prep(
    *,
    universe_manifest: str | Path = DEFAULT_UNIVERSE_MANIFEST,
    membership_path: str | Path | None = None,
    variant: str | None = None,
    short_eligible_only: bool = False,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    submissions_raw_root: str | Path = DEFAULT_SUBMISSIONS_RAW_ROOT,
    sec_user_agent: str | None = None,
    refresh: bool = False,
    download_submissions: bool = False,
    download_older_submission_files: bool = False,
    max_symbols: int | None = None,
    sleep_seconds: float = 0.12,
) -> dict[str, Any]:
    """Prepare SEC reference artifacts for non-price factor research.

    This phase intentionally stops at data coverage and mapping. It does not
    build factors, train models, or inspect lockbox performance.
    """

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(raw_root)
    raw_dir.mkdir(parents=True, exist_ok=True)

    universe_manifest_path = Path(universe_manifest)
    if membership_path and variant:
        universe = _load_membership_universe(
            membership_path=Path(membership_path),
            variant=variant,
            enrichment_manifest_path=universe_manifest_path,
            short_eligible_only=short_eligible_only,
        )
        universe_source = f"membership:{Path(membership_path).as_posix()}#{variant}"
    else:
        universe = _load_universe_manifest(universe_manifest_path)
        universe_source = f"manifest:{universe_manifest_path.as_posix()}"
    company_ticker_snapshot = _ensure_sec_company_ticker_snapshot(
        raw_dir=raw_dir,
        sec_user_agent=sec_user_agent,
        refresh=refresh,
    )
    sec_rows = _load_sec_company_tickers(company_ticker_snapshot)
    mapping_rows = _build_universe_cik_mapping(universe, sec_rows)

    mapping_path = output_dir / "sec_universe_cik_mapping.csv"
    mapping_fieldnames = _mapping_fieldnames()
    _write_csv(mapping_path, mapping_rows, fieldnames=mapping_fieldnames)

    missing_path = output_dir / "sec_universe_cik_missing.csv"
    _write_csv(
        missing_path,
        [row for row in mapping_rows if row["match_status"] != "matched"],
        fieldnames=mapping_fieldnames,
    )

    source_manifest = _source_feasibility_manifest()
    source_manifest_path = output_dir / "non_price_source_feasibility.csv"
    _write_csv(source_manifest_path, source_manifest, fieldnames=_source_manifest_fieldnames())

    factor_manifest = _factor_channel_manifest()
    factor_manifest_path = output_dir / "two_factor_channel_manifest.csv"
    _write_csv(factor_manifest_path, factor_manifest, fieldnames=_factor_manifest_fieldnames())

    submissions_manifest_path = output_dir / "sec_submissions_download_manifest.csv"
    submissions_rows: list[dict[str, Any]] = []
    if download_submissions:
        submissions_rows = _prepare_sec_submissions(
            mapping_rows=mapping_rows,
            raw_dir=Path(submissions_raw_root),
            sec_user_agent=sec_user_agent,
            refresh=refresh,
            max_symbols=max_symbols,
            sleep_seconds=sleep_seconds,
            download_older_files=download_older_submission_files,
        )
        _write_csv(
            submissions_manifest_path,
            submissions_rows,
            fieldnames=_submissions_manifest_fieldnames(),
        )

    summary = _build_summary(
        universe=universe,
        universe_source=universe_source,
        mapping_rows=mapping_rows,
        company_ticker_snapshot=company_ticker_snapshot,
        mapping_path=mapping_path,
        missing_path=missing_path,
        source_manifest_path=source_manifest_path,
        factor_manifest_path=factor_manifest_path,
        submissions_manifest_path=submissions_manifest_path if download_submissions else None,
        submissions_rows=submissions_rows,
    )
    summary_path = output_dir / "phase0_non_price_data_prep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def _load_universe_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Universe manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            symbol = _clean_symbol(row.get("symbol"))
            if not symbol:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "name": row.get("name") or row.get("company_name") or "",
                    "exchange": row.get("exchange") or row.get("exchange_mic") or "",
                    "liquidity_rank": row.get("liquidity_rank") or row.get("rank") or "",
                    "median_dollar_volume": row.get("median_dollar_volume") or "",
                    "universe_rows": row.get("universe_rows") or "",
                    "first_session": row.get("first_session") or "",
                    "last_session": row.get("last_session") or "",
                    "source_universe": row.get("source_universe") or "manifest",
                }
            )
    if not rows:
        raise ValueError(f"No symbols loaded from universe manifest: {path}")
    return rows


def _load_membership_universe(
    *,
    membership_path: Path,
    variant: str,
    enrichment_manifest_path: Path,
    short_eligible_only: bool,
) -> list[dict[str, Any]]:
    if not membership_path.exists():
        raise FileNotFoundError(f"Membership path not found: {membership_path}")
    membership = _read_membership_columns(membership_path)
    if not membership or "variant" not in membership[0]:
        raise ValueError(f"Membership file has no variant column: {membership_path}")

    enrichment = _load_enrichment_by_symbol(enrichment_manifest_path)
    stats: dict[str, dict[str, Any]] = {}
    for row in membership:
        if row.get("variant") != variant:
            continue
        if short_eligible_only and not _truthy(row.get("short_eligible_proxy")):
            continue
        symbol = _clean_symbol(row.get("symbol"))
        session_date = str(row.get("session_date") or "")
        if not symbol:
            continue
        item = stats.setdefault(
            symbol,
            {
                "symbol": symbol,
                "universe_rows": 0,
                "first_session": session_date,
                "last_session": session_date,
                "max_liquidity_rank": "",
                "min_liquidity_rank": "",
            },
        )
        item["universe_rows"] += 1
        if session_date:
            item["first_session"] = min(item["first_session"], session_date)
            item["last_session"] = max(item["last_session"], session_date)
        rank = _safe_float(row.get("liquidity_rank"))
        if rank is not None:
            item["min_liquidity_rank"] = (
                rank
                if item["min_liquidity_rank"] == ""
                else min(float(item["min_liquidity_rank"]), rank)
            )
            item["max_liquidity_rank"] = (
                rank
                if item["max_liquidity_rank"] == ""
                else max(float(item["max_liquidity_rank"]), rank)
            )

    rows: list[dict[str, Any]] = []
    for symbol in sorted(stats):
        enriched = enrichment.get(symbol, {})
        item = stats[symbol]
        rows.append(
            {
                "symbol": symbol,
                "name": enriched.get("name", ""),
                "exchange": enriched.get("exchange", ""),
                "liquidity_rank": enriched.get("liquidity_rank", item["min_liquidity_rank"]),
                "median_dollar_volume": enriched.get("median_dollar_volume", ""),
                "universe_rows": item["universe_rows"],
                "first_session": item["first_session"],
                "last_session": item["last_session"],
                "source_universe": variant,
            }
        )
    if not rows:
        raise ValueError(f"No rows found for variant {variant!r} in {membership_path}")
    return rows


def _read_membership_columns(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    requested = ["variant", "session_date", "symbol", "short_eligible_proxy", "liquidity_rank"]
    header = pd.read_csv(path, nrows=0)
    usecols = [column for column in requested if column in header.columns]
    frame = pd.read_csv(path, usecols=usecols)
    return frame.to_dict("records")


def _load_enrichment_by_symbol(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _load_universe_manifest(path)
    return {row["symbol"]: row for row in rows}


def _ensure_sec_company_ticker_snapshot(
    *,
    raw_dir: Path,
    sec_user_agent: str | None,
    refresh: bool,
) -> Path:
    existing = sorted(raw_dir.glob("company_tickers_exchange_*.json"))
    if existing and not refresh:
        return existing[-1]

    timestamp = _utc_stamp()
    path = raw_dir / f"company_tickers_exchange_{timestamp}.json"
    try:
        _download(SEC_COMPANY_TICKERS_EXCHANGE_URL, path, sec_user_agent=sec_user_agent)
    except Exception:
        fallback_path = raw_dir / f"company_tickers_{timestamp}.json"
        _download(SEC_COMPANY_TICKERS_URL, fallback_path, sec_user_agent=sec_user_agent)
        return fallback_path
    return path


def _download(url: str, output_path: Path, *, sec_user_agent: str | None) -> None:
    user_agent = (
        sec_user_agent
        or os.environ.get("SEC_USER_AGENT")
        or "StockMachine research contact@example.com"
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "identity",
        },
    )
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with tmp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _load_sec_company_tickers(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    if isinstance(payload, dict) and "fields" in payload and "data" in payload:
        fields = [str(field).lower() for field in payload["fields"]]
        for values in payload["data"]:
            row = dict(zip(fields, values))
            rows.append(
                {
                    "cik": _format_cik(row.get("cik") or row.get("cik_str")),
                    "ticker": _clean_symbol(row.get("ticker")),
                    "title": str(row.get("name") or row.get("title") or ""),
                    "exchange": str(row.get("exchange") or ""),
                }
            )
        return rows

    if isinstance(payload, dict):
        for value in payload.values():
            if not isinstance(value, Mapping):
                continue
            rows.append(
                {
                    "cik": _format_cik(value.get("cik") or value.get("cik_str")),
                    "ticker": _clean_symbol(value.get("ticker")),
                    "title": str(value.get("title") or value.get("name") or ""),
                    "exchange": str(value.get("exchange") or ""),
                }
            )
        return rows

    raise ValueError(f"Unsupported SEC company ticker payload shape: {path}")


def _build_universe_cik_mapping(
    universe_rows: Iterable[Mapping[str, Any]],
    sec_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in sec_rows:
        ticker = _clean_symbol(row.get("ticker"))
        if not ticker:
            continue
        for key in _ticker_keys(ticker):
            index.setdefault(key, row)

    output = []
    for row in universe_rows:
        symbol = _clean_symbol(row.get("symbol"))
        match = None
        matched_key = ""
        for key in _ticker_keys(symbol):
            if key in index:
                match = index[key]
                matched_key = key
                break
        output.append(
            {
                "symbol": symbol,
                "company_name": row.get("name", ""),
                "exchange": row.get("exchange", ""),
                "liquidity_rank": row.get("liquidity_rank", ""),
                "median_dollar_volume": row.get("median_dollar_volume", ""),
                "universe_rows": row.get("universe_rows", ""),
                "first_session": row.get("first_session", ""),
                "last_session": row.get("last_session", ""),
                "source_universe": row.get("source_universe", ""),
                "cik": match.get("cik", "") if match else "",
                "sec_ticker": match.get("ticker", "") if match else "",
                "sec_title": match.get("title", "") if match else "",
                "sec_exchange": match.get("exchange", "") if match else "",
                "matched_key": matched_key,
                "match_status": "matched" if match else "missing",
                "mapping_confidence": "ticker_exact_or_class_normalized" if match else "missing",
            }
        )
    return output


def _prepare_sec_submissions(
    *,
    mapping_rows: list[dict[str, Any]],
    raw_dir: Path,
    sec_user_agent: str | None,
    refresh: bool,
    max_symbols: int | None,
    sleep_seconds: float,
    download_older_files: bool,
) -> list[dict[str, Any]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in mapping_rows if row.get("cik")]
    rows = sorted(rows, key=lambda row: _safe_int(row.get("liquidity_rank")))
    if max_symbols is not None and max_symbols > 0:
        rows = rows[:max_symbols]

    output = []
    for index, row in enumerate(rows, start=1):
        cik = str(row["cik"])
        symbol = str(row["symbol"])
        target_path = raw_dir / f"CIK{cik}.json"
        status = "cached"
        error = ""
        if refresh or not target_path.exists():
            status = "downloaded"
            url = SEC_SUBMISSION_URL_TEMPLATE.format(cik=cik)
            try:
                _download(url, target_path, sec_user_agent=sec_user_agent)
                if sleep_seconds > 0 and index < len(rows):
                    time.sleep(sleep_seconds)
            except Exception as exc:  # pragma: no cover - network failure path
                status = "failed"
                error = str(exc)

        summary = _summarize_submission_file(
            symbol=symbol,
            cik=cik,
            company_name=str(row.get("company_name", "")),
            liquidity_rank=str(row.get("liquidity_rank", "")),
            path=target_path,
            status=status,
            error=error,
        )
        if download_older_files and status != "failed" and target_path.exists():
            _download_older_submission_files(
                summary=summary,
                raw_dir=raw_dir,
                sec_user_agent=sec_user_agent,
                refresh=refresh,
                sleep_seconds=sleep_seconds,
            )
        _add_combined_submission_coverage(summary, raw_dir=raw_dir)
        output.append(summary)
    return output


def _summarize_submission_file(
    *,
    symbol: str,
    cik: str,
    company_name: str,
    liquidity_rank: str,
    path: Path,
    status: str,
    error: str,
) -> dict[str, Any]:
    base = {
        "symbol": symbol,
        "cik": cik,
        "company_name": company_name,
        "liquidity_rank": liquidity_rank,
        "raw_path": path.as_posix(),
        "download_status": status,
        "error": error,
        "recent_filing_count": "",
        "oldest_recent_filing_date": "",
        "newest_recent_filing_date": "",
        "older_file_count": "",
        "older_files": "",
        "has_older_history_files": "",
        "older_files_downloaded_count": "",
        "older_files_failed_count": "",
        "combined_filing_count": "",
        "oldest_combined_filing_date": "",
        "newest_combined_filing_date": "",
        "validation_window_overlap_status": "",
    }
    if status == "failed" or not path.exists():
        return base

    payload = json.loads(path.read_text(encoding="utf-8"))
    recent = payload.get("filings", {}).get("recent", {}) if isinstance(payload, dict) else {}
    filing_dates = [
        str(value)
        for value in recent.get("filingDate", [])
        if isinstance(value, str) and value
    ]
    older_files = []
    for item in payload.get("filings", {}).get("files", []):
        if isinstance(item, Mapping):
            name = item.get("name")
            if name:
                older_files.append(str(name))

    base.update(
        {
            "recent_filing_count": len(filing_dates),
            "oldest_recent_filing_date": min(filing_dates) if filing_dates else "",
            "newest_recent_filing_date": max(filing_dates) if filing_dates else "",
            "older_file_count": len(older_files),
            "older_files": "|".join(older_files),
            "has_older_history_files": bool(older_files),
        }
    )
    return base


def _download_older_submission_files(
    *,
    summary: dict[str, Any],
    raw_dir: Path,
    sec_user_agent: str | None,
    refresh: bool,
    sleep_seconds: float,
) -> None:
    names = [name for name in str(summary.get("older_files", "")).split("|") if name]
    downloaded = 0
    failed = 0
    for name in names:
        target_path = raw_dir / name
        if not refresh and target_path.exists():
            downloaded += 1
            continue
        try:
            _download(f"https://data.sec.gov/submissions/{name}", target_path, sec_user_agent=sec_user_agent)
            downloaded += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        except Exception:  # pragma: no cover - network failure path
            failed += 1
    summary["older_files_downloaded_count"] = downloaded
    summary["older_files_failed_count"] = failed


def _add_combined_submission_coverage(summary: dict[str, Any], *, raw_dir: Path) -> None:
    paths = [Path(str(summary["raw_path"]))]
    for name in str(summary.get("older_files", "")).split("|"):
        if name:
            paths.append(raw_dir / name)
    dates = []
    for path in paths:
        if path.exists():
            dates.extend(_submission_filing_dates(path))
    summary["combined_filing_count"] = len(dates)
    summary["oldest_combined_filing_date"] = min(dates) if dates else ""
    summary["newest_combined_filing_date"] = max(dates) if dates else ""
    summary["validation_window_overlap_status"] = _validation_overlap_status(dates)


def _submission_filing_dates(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return []
    if "filingDate" in payload:
        return [
            str(value)
            for value in payload.get("filingDate", [])
            if isinstance(value, str) and value
        ]
    recent = payload.get("filings", {}).get("recent", {})
    if not isinstance(recent, Mapping):
        return []
    return [
        str(value)
        for value in recent.get("filingDate", [])
        if isinstance(value, str) and value
    ]


def _validation_overlap_status(filing_dates: list[str]) -> str:
    if not filing_dates:
        return "no_filings"
    oldest = min(filing_dates)
    newest = max(filing_dates)
    if oldest <= VALIDATION_START and newest >= VALIDATION_END:
        return "spans_full_validation_window"
    if newest < VALIDATION_START:
        return "ends_before_validation_window"
    if oldest > VALIDATION_END:
        return "starts_after_validation_window"
    return "partial_validation_overlap"


def _source_feasibility_manifest() -> list[dict[str, Any]]:
    return [
        {
            "source_family": "sec_filing_metadata",
            "candidate_factor": "filing_red_flag_score",
            "primary_channel": "short_evidence_score",
            "aligned_window_status": "full_validation_and_test_candidate",
            "first_use": "phase1_short_evidence_mvp",
            "pit_risk": "medium",
            "notes": "Use filing acceptance time or conservative next-session availability.",
        },
        {
            "source_family": "sec_insider_transactions",
            "candidate_factor": "insider_net_buy_score",
            "primary_channel": "long_evidence_score",
            "aligned_window_status": "full_validation_and_test_candidate",
            "first_use": "phase2_long_evidence_mvp",
            "pit_risk": "medium",
            "notes": "Classify open-market purchases separately from grants, exercises, and planned sales.",
        },
        {
            "source_family": "sec_structured_fundamentals",
            "candidate_factor": "fundamental_quality_change_score",
            "primary_channel": "long_and_short_evidence",
            "aligned_window_status": "full_validation_and_test_candidate_with_warmup",
            "first_use": "phase3_fundamental_quality",
            "pit_risk": "medium_high",
            "notes": "Requires taxonomy normalization, restatement handling, and fiscal-period alignment.",
        },
        {
            "source_family": "sec_13f",
            "candidate_factor": "crowding_score",
            "primary_channel": "risk_context",
            "aligned_window_status": "mostly_aligned_from_2013_07",
            "first_use": "later_crowding_layer",
            "pit_risk": "medium_high",
            "notes": "Quarterly and lagged; use as crowding/risk context before directional alpha.",
        },
        {
            "source_family": "finra_short_data",
            "candidate_factor": "short_pressure_score",
            "primary_channel": "short_evidence_or_squeeze_context",
            "aligned_window_status": "not_first_line_full_window",
            "first_use": "later_if_vendor_coverage_exists",
            "pit_risk": "medium",
            "notes": "Daily short-sale volume starts too late for full validation; short-interest history needs coverage audit.",
        },
        {
            "source_family": "attention_news",
            "candidate_factor": "attention_mismatch_score",
            "primary_channel": "context_interaction",
            "aligned_window_status": "later_start_exploratory",
            "first_use": "deferred",
            "pit_risk": "high",
            "notes": "Wikipedia/GDELT 2.0 start too late for full-window first-line validation.",
        },
    ]


def _factor_channel_manifest() -> list[dict[str, Any]]:
    return [
        {
            "channel": "long_evidence_score",
            "first_factor": "insider_net_buy_score",
            "direction": "higher_is_better_for_long",
            "initial_role": "long boost",
            "validation_question": "Does the high-score bucket outperform the eligible median after beta/sector controls?",
        },
        {
            "channel": "short_evidence_score",
            "first_factor": "filing_red_flag_score",
            "direction": "higher_is_worse_for_stock",
            "initial_role": "short candidate or long exclusion",
            "validation_question": "Does the high-score bucket underperform the eligible median and improve the short leg?",
        },
        {
            "channel": "shared_quality_score",
            "first_factor": "fundamental_quality_change_score",
            "direction": "higher_is_better_for_long_lower_can_support_short",
            "initial_role": "core non-price alpha candidate",
            "validation_question": "Does accounting-quality change add value after price, beta, sector, and liquidity features?",
        },
    ]


def _build_summary(
    *,
    universe: list[dict[str, Any]],
    universe_source: str,
    mapping_rows: list[dict[str, Any]],
    company_ticker_snapshot: Path,
    mapping_path: Path,
    missing_path: Path,
    source_manifest_path: Path,
    factor_manifest_path: Path,
    submissions_manifest_path: Path | None,
    submissions_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    mapped = [row for row in mapping_rows if row["match_status"] == "matched"]
    missing = [row for row in mapping_rows if row["match_status"] != "matched"]
    missing_preview = [row["symbol"] for row in missing[:50]]
    return {
        "created_at_utc": _utc_now(),
        "project_id": PROJECT_ID,
        "purpose": "non_price_two_factor_phase0_data_prep",
        "universe_source": universe_source,
        "validation_start": VALIDATION_START,
        "pure_alpha_session_start": PURE_ALPHA_SESSION_START,
        "validation_end": VALIDATION_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "lockbox_status": "closed_no_performance_computed",
        "universe_count": len(universe),
        "mapped_cik_count": len(mapped),
        "missing_cik_count": len(missing),
        "mapped_cik_rate": round(len(mapped) / len(universe), 6) if universe else 0.0,
        "missing_symbol_preview": missing_preview,
        "sec_company_ticker_snapshot": company_ticker_snapshot.as_posix(),
        "mapping_artifact": mapping_path.as_posix(),
        "missing_mapping_artifact": missing_path.as_posix(),
        "source_feasibility_artifact": source_manifest_path.as_posix(),
        "factor_channel_artifact": factor_manifest_path.as_posix(),
        "submissions_download_artifact": (
            submissions_manifest_path.as_posix() if submissions_manifest_path else None
        ),
        "submissions_download_count": len(submissions_rows),
        "submissions_downloaded_or_cached_count": sum(
            1
            for row in submissions_rows
            if row.get("download_status") in {"downloaded", "cached"}
        ),
        "submissions_failed_count": sum(
            1 for row in submissions_rows if row.get("download_status") == "failed"
        ),
        "submissions_validation_overlap_counts": _count_by_key(
            submissions_rows,
            "validation_window_overlap_status",
        ),
        "next_steps": [
            "Review missing CIK mappings and decide whether to exclude or manually map them.",
            "Build filing metadata ingestion for short_evidence_score.",
            "Build Form 4 ingestion for long_evidence_score.",
            "Keep all feature discovery inside the validation window.",
        ],
        "limitations": [
            "Universe is the current top1000 bootstrap manifest, not a final survivorship-bias-free historical universe.",
            "Ticker-to-CIK mapping uses current SEC ticker reference and must be audited for historical ticker changes.",
            "No alpha features, model training, or portfolio performance are computed in this phase.",
        ],
    }


def _write_csv(
    path: Path,
    rows: list[Mapping[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fieldnames = fieldnames or sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mapping_fieldnames() -> list[str]:
    return [
        "symbol",
        "company_name",
        "exchange",
        "liquidity_rank",
        "median_dollar_volume",
        "universe_rows",
        "first_session",
        "last_session",
        "source_universe",
        "cik",
        "sec_ticker",
        "sec_title",
        "sec_exchange",
        "matched_key",
        "match_status",
        "mapping_confidence",
    ]


def _source_manifest_fieldnames() -> list[str]:
    return [
        "source_family",
        "candidate_factor",
        "primary_channel",
        "aligned_window_status",
        "first_use",
        "pit_risk",
        "notes",
    ]


def _factor_manifest_fieldnames() -> list[str]:
    return [
        "channel",
        "first_factor",
        "direction",
        "initial_role",
        "validation_question",
    ]


def _submissions_manifest_fieldnames() -> list[str]:
    return [
        "symbol",
        "cik",
        "company_name",
        "liquidity_rank",
        "raw_path",
        "download_status",
        "error",
        "recent_filing_count",
        "oldest_recent_filing_date",
        "newest_recent_filing_date",
        "older_file_count",
        "older_files",
        "has_older_history_files",
        "older_files_downloaded_count",
        "older_files_failed_count",
        "combined_filing_count",
        "oldest_combined_filing_date",
        "newest_combined_filing_date",
        "validation_window_overlap_status",
    ]


def _safe_int(value: Any) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 10**12


def _safe_float(value: Any) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _count_by_key(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "missing")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _ticker_keys(symbol: str | None) -> tuple[str, ...]:
    clean = _clean_symbol(symbol)
    if not clean:
        return ()
    keys = {
        clean,
        clean.replace(".", "-"),
        clean.replace("-", "."),
        clean.replace("/", "-"),
        clean.replace("/", "."),
    }
    return tuple(key for key in keys if key)


def _clean_symbol(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _format_cik(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{int(value):010d}"
    except (TypeError, ValueError):
        text = str(value).strip()
        return text.zfill(10) if text.isdigit() else text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare non-price phase0 reference data.")
    parser.add_argument("--universe-manifest", default=str(DEFAULT_UNIVERSE_MANIFEST))
    parser.add_argument("--membership-path", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--short-eligible-only", action="store_true")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--submissions-raw-root", default=str(DEFAULT_SUBMISSIONS_RAW_ROOT))
    parser.add_argument("--sec-user-agent", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--download-submissions", action="store_true")
    parser.add_argument("--download-older-submission-files", action="store_true")
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.12)
    args = parser.parse_args()

    summary = run_phase0_data_prep(
        universe_manifest=args.universe_manifest,
        membership_path=args.membership_path,
        variant=args.variant,
        short_eligible_only=args.short_eligible_only,
        output_root=args.output_root,
        raw_root=args.raw_root,
        submissions_raw_root=args.submissions_raw_root,
        sec_user_agent=args.sec_user_agent,
        refresh=args.refresh,
        download_submissions=args.download_submissions,
        download_older_submission_files=args.download_older_submission_files,
        max_symbols=args.max_symbols,
        sleep_seconds=args.sleep_seconds,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
