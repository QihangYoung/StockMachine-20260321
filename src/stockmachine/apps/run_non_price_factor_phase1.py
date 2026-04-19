from __future__ import annotations

import argparse
import bisect
import json
import math
import shutil
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


PROJECT_ID = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT_ID / "research"
DEFAULT_PHASE1_MEMBERSHIP_PATH = (
    RESEARCH_ROOT
    / "phase1_universe_builder_20260418"
    / "phase1_candidate_universe_membership_validation.csv.gz"
)
DEFAULT_ADV30M_MAPPING_PATH = (
    RESEARCH_ROOT
    / "non_price_phase0_adv30m_short_data_prep_20260419"
    / "sec_universe_cik_mapping.csv"
)
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "non_price_phase1_two_factor_build_20260419"
DEFAULT_SUBMISSIONS_RAW_ROOT = Path("data") / "raw" / "sec" / "submissions"
DEFAULT_INSIDER_RAW_ROOT = Path("data") / "raw" / "sec" / "insider_transactions"

SEC_INSIDER_ZIP_URL_TEMPLATE = (
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets/{quarter}_form345.zip"
)

DEFAULT_VARIANT = "adv30m_clean_core_beta_full"
DEFAULT_START = "2013-08-05"
DEFAULT_END = "2019-12-31"

RED_8K_ITEM_WEIGHTS = {
    "1.03": 2.5,  # Bankruptcy or receivership.
    "2.04": 2.0,  # Default / acceleration trigger.
    "2.05": 1.5,  # Exit or disposal costs.
    "2.06": 2.0,  # Material impairments.
    "3.01": 2.0,  # Delisting notice.
    "4.01": 2.5,  # Auditor change.
    "4.02": 4.0,  # Non-reliance on previous financials.
    "5.02": 0.75,  # Officer/director changes. Weak by itself.
}


def build_non_price_phase1_factors(
    *,
    mapping_path: str | Path = DEFAULT_ADV30M_MAPPING_PATH,
    membership_path: str | Path = DEFAULT_PHASE1_MEMBERSHIP_PATH,
    variant: str = DEFAULT_VARIANT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    submissions_raw_root: str | Path = DEFAULT_SUBMISSIONS_RAW_ROOT,
    insider_raw_root: str | Path = DEFAULT_INSIDER_RAW_ROOT,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    sec_user_agent: str | None = None,
    download_insider_zips: bool = False,
    sleep_seconds: float = 0.12,
    max_symbols: int | None = None,
) -> dict[str, Any]:
    """Build validation-only non-price factor panels for the first two channels."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = _load_mapping(Path(mapping_path), max_symbols=max_symbols)
    membership = _load_membership(
        Path(membership_path),
        variant=variant,
        start=start,
        end=end,
        symbols=sorted(mapping["symbol"].unique()),
    )
    if membership.empty:
        raise ValueError("No membership rows after filtering.")

    sessions = sorted(membership["session_date"].unique().tolist())
    session_index = {session: index for index, session in enumerate(sessions)}

    filing_events, filing_all = _build_filing_events(
        mapping=mapping,
        submissions_raw_root=Path(submissions_raw_root),
        sessions=sessions,
    )
    filing_panel = _build_filing_panel(
        membership=membership,
        events=filing_events,
        session_index=session_index,
    )

    if download_insider_zips:
        _ensure_insider_zips(
            raw_root=Path(insider_raw_root),
            start=start,
            end=end,
            sec_user_agent=sec_user_agent,
            sleep_seconds=sleep_seconds,
        )
    insider_events, insider_transactions = _build_insider_events(
        mapping=mapping,
        raw_root=Path(insider_raw_root),
        start=start,
        end=end,
        sessions=sessions,
    )
    insider_panel = _build_insider_panel(
        membership=membership,
        events=insider_events,
        session_index=session_index,
    )

    factor_panel = filing_panel.merge(
        insider_panel,
        on=["session_date", "symbol"],
        how="outer",
        validate="one_to_one",
    ).fillna(0.0)
    factor_panel = factor_panel.sort_values(["session_date", "symbol"]).reset_index(drop=True)

    paths = _write_outputs(
        output_dir=output_dir,
        filing_events=filing_events,
        filing_all=filing_all,
        filing_panel=filing_panel,
        insider_events=insider_events,
        insider_transactions=insider_transactions,
        insider_panel=insider_panel,
        factor_panel=factor_panel,
    )
    summary = _build_summary(
        output_dir=output_dir,
        mapping=mapping,
        membership=membership,
        filing_events=filing_events,
        filing_panel=filing_panel,
        insider_events=insider_events,
        insider_transactions=insider_transactions,
        insider_panel=insider_panel,
        factor_panel=factor_panel,
        paths=paths,
        variant=variant,
        start=start,
        end=end,
        download_insider_zips=download_insider_zips,
    )
    summary_path = output_dir / "phase1_non_price_two_factor_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def _load_mapping(path: Path, *, max_symbols: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")
    frame = pd.read_csv(path, dtype=str)
    frame = frame[frame["match_status"].eq("matched")].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["cik"] = frame["cik"].astype(str).str.zfill(10)
    frame["median_dollar_volume_float"] = pd.to_numeric(
        frame.get("median_dollar_volume", 0.0),
        errors="coerce",
    ).fillna(0.0)
    frame["liquidity_rank_float"] = pd.to_numeric(
        frame.get("liquidity_rank", 10**12),
        errors="coerce",
    ).fillna(10**12)
    frame = frame.sort_values(["liquidity_rank_float", "symbol"])
    if max_symbols is not None and max_symbols > 0:
        frame = frame.head(max_symbols).copy()
    return frame.reset_index(drop=True)


def _load_membership(
    path: Path,
    *,
    variant: str,
    start: str,
    end: str,
    symbols: Sequence[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Membership file not found: {path}")
    usecols = ["variant", "session_date", "symbol"]
    frame = pd.read_csv(path, usecols=usecols, dtype=str)
    symbol_set = set(symbols)
    frame = frame[
        frame["variant"].eq(variant)
        & frame["symbol"].isin(symbol_set)
        & frame["session_date"].ge(start)
        & frame["session_date"].le(end)
    ].copy()
    frame = frame[["session_date", "symbol"]].drop_duplicates()
    return frame.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _build_filing_events(
    *,
    mapping: pd.DataFrame,
    submissions_raw_root: Path,
    sessions: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events: list[dict[str, Any]] = []
    all_filings: list[dict[str, Any]] = []
    for row in mapping.to_dict("records"):
        symbol = str(row["symbol"])
        cik = str(row["cik"])
        for filing in _iter_submission_filings(cik, submissions_raw_root):
            filing_date = filing.get("filing_date")
            if not filing_date:
                continue
            effective_session = _next_session_after(filing_date, sessions)
            record = {
                "symbol": symbol,
                "cik": cik,
                "accession_number": filing.get("accession_number", ""),
                "filing_date": filing_date,
                "effective_session": effective_session or "",
                "report_date": filing.get("report_date", ""),
                "acceptance_datetime": filing.get("acceptance_datetime", ""),
                "form": filing.get("form", ""),
                "items": filing.get("items", ""),
                "primary_document": filing.get("primary_document", ""),
                "primary_doc_description": filing.get("primary_doc_description", ""),
            }
            all_filings.append(record)
            red = _filing_red_flag_components(record)
            if not effective_session or red["filing_red_flag_event_weight"] <= 0:
                continue
            events.append({**record, **red})

    all_frame = pd.DataFrame(all_filings)
    event_frame = pd.DataFrame(events)
    if not all_frame.empty:
        all_frame = all_frame.drop_duplicates(["symbol", "accession_number"])
    if not event_frame.empty:
        event_frame = event_frame.drop_duplicates(["symbol", "accession_number"])
    return event_frame, all_frame


def _iter_submission_filings(cik: str, raw_root: Path) -> Iterable[dict[str, str]]:
    paths = sorted(raw_root.glob(f"CIK{cik}*.json"))
    seen: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = _submission_rows(payload)
        for row in rows:
            accession = row.get("accessionNumber") or row.get("accession_number") or ""
            if not accession or accession in seen:
                continue
            seen.add(accession)
            yield {
                "accession_number": str(accession),
                "filing_date": _sec_date_to_iso(row.get("filingDate")),
                "report_date": _sec_date_to_iso(row.get("reportDate")),
                "acceptance_datetime": str(row.get("acceptanceDateTime") or ""),
                "form": str(row.get("form") or ""),
                "items": str(row.get("items") or ""),
                "primary_document": str(row.get("primaryDocument") or ""),
                "primary_doc_description": str(row.get("primaryDocDescription") or ""),
            }


def _submission_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "filings" in payload:
        recent = payload.get("filings", {}).get("recent", {})
        if isinstance(recent, Mapping):
            return _columnar_to_rows(recent)
    return _columnar_to_rows(payload)


def _columnar_to_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    lengths = [len(value) for value in payload.values() if isinstance(value, list)]
    if not lengths:
        return []
    row_count = min(lengths)
    rows = []
    for index in range(row_count):
        rows.append(
            {
                key: value[index] if isinstance(value, list) and index < len(value) else ""
                for key, value in payload.items()
            }
        )
    return rows


def _filing_red_flag_components(record: Mapping[str, Any]) -> dict[str, Any]:
    form = str(record.get("form") or "").upper().strip()
    items = _parse_items(record.get("items"))

    nt_weight = 0.0
    if form.startswith("NT "):
        nt_weight = 3.0 if any(code in form for code in ("10-K", "10-Q")) else 1.5

    amendment_weight = 0.0
    if form in {"10-K/A", "10-Q/A"}:
        amendment_weight = 1.5
    elif form.endswith("/A") and "8-K" in form:
        amendment_weight = 0.5

    red_8k_weight = 0.0
    if "8-K" in form:
        red_8k_weight = sum(RED_8K_ITEM_WEIGHTS.get(item, 0.0) for item in items)

    periodic_delay_weight = _periodic_delay_weight(record)
    total = nt_weight + amendment_weight + red_8k_weight + periodic_delay_weight
    return {
        "filing_red_flag_event_weight": float(total),
        "nt_10kq_event_weight": float(nt_weight),
        "amendment_event_weight": float(amendment_weight),
        "red_8k_event_weight": float(red_8k_weight),
        "periodic_delay_event_weight": float(periodic_delay_weight),
        "red_8k_items": "|".join(item for item in items if item in RED_8K_ITEM_WEIGHTS),
    }


def _periodic_delay_weight(record: Mapping[str, Any]) -> float:
    form = str(record.get("form") or "").upper().strip()
    if form not in {"10-K", "10-Q"}:
        return 0.0
    filing_date = _parse_iso_date(record.get("filing_date"))
    report_date = _parse_iso_date(record.get("report_date"))
    if filing_date is None or report_date is None:
        return 0.0
    delay_days = (filing_date - report_date).days
    if form == "10-K" and delay_days > 95:
        return 1.0
    if form == "10-Q" and delay_days > 50:
        return 1.0
    return 0.0


def _build_filing_panel(
    *,
    membership: pd.DataFrame,
    events: pd.DataFrame,
    session_index: Mapping[str, int],
) -> pd.DataFrame:
    base = membership[["session_date", "symbol"]].drop_duplicates().copy()
    columns = [
        "filing_red_flag_score",
        "filing_red_flag_events_20d",
        "filing_red_flag_events_60d",
        "nt_10kq_events_252d",
        "amendment_events_60d",
        "red_8k_events_60d",
        "periodic_delay_events_252d",
        "days_since_last_filing_red_flag",
    ]
    return _build_decayed_panel(
        base=base,
        events=events,
        session_index=session_index,
        score_column="filing_red_flag_event_weight",
        output_score_column="filing_red_flag_score",
        component_specs={
            "filing_red_flag_events_20d": ("filing_red_flag_event_weight", 20),
            "filing_red_flag_events_60d": ("filing_red_flag_event_weight", 60),
            "nt_10kq_events_252d": ("nt_10kq_event_weight", 252),
            "amendment_events_60d": ("amendment_event_weight", 60),
            "red_8k_events_60d": ("red_8k_event_weight", 60),
            "periodic_delay_events_252d": ("periodic_delay_event_weight", 252),
        },
        days_since_column="days_since_last_filing_red_flag",
        output_columns=columns,
        half_life_sessions=20.0,
        max_age_sessions=252,
    )


def _ensure_insider_zips(
    *,
    raw_root: Path,
    start: str,
    end: str,
    sec_user_agent: str | None,
    sleep_seconds: float,
) -> list[Path]:
    raw_root.mkdir(parents=True, exist_ok=True)
    paths = []
    quarters = _quarter_labels(start, end)
    for index, quarter in enumerate(quarters):
        path = raw_root / f"{quarter}_form345.zip"
        if not path.exists():
            _download(
                SEC_INSIDER_ZIP_URL_TEMPLATE.format(quarter=quarter),
                path,
                sec_user_agent=sec_user_agent,
            )
            if sleep_seconds > 0 and index < len(quarters) - 1:
                time.sleep(sleep_seconds)
        paths.append(path)
    return paths


def _build_insider_events(
    *,
    mapping: pd.DataFrame,
    raw_root: Path,
    start: str,
    end: str,
    sessions: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    zips = sorted(raw_root.glob("*_form345.zip"))
    if not zips:
        return pd.DataFrame(), pd.DataFrame()

    cik_to_symbols = (
        mapping.groupby("cik")["symbol"].apply(list).to_dict()
    )
    cik_to_adv = mapping.drop_duplicates("cik").set_index("cik")["median_dollar_volume_float"].to_dict()
    target_ciks = set(cik_to_symbols)
    transaction_frames = []

    for path in zips:
        quarter = path.name.split("_", 1)[0]
        if not _quarter_overlaps(quarter, start, end):
            continue
        parsed = _parse_insider_zip(path, target_ciks=target_ciks)
        if not parsed.empty:
            transaction_frames.append(parsed)

    if not transaction_frames:
        return pd.DataFrame(), pd.DataFrame()

    transactions = pd.concat(transaction_frames, ignore_index=True)
    transactions["filing_date"] = transactions["filing_date"].map(_sec_date_to_iso)
    transactions["transaction_date"] = transactions["transaction_date"].map(_sec_date_to_iso)
    transactions = transactions[
        transactions["filing_date"].ge(start) & transactions["filing_date"].le(end)
    ].copy()
    if transactions.empty:
        return pd.DataFrame(), transactions

    transactions["shares"] = pd.to_numeric(transactions["shares"], errors="coerce").fillna(0.0)
    transactions["price"] = pd.to_numeric(transactions["price"], errors="coerce").fillna(0.0)
    transactions["dollar_value"] = transactions["shares"] * transactions["price"]
    transactions = transactions[transactions["dollar_value"].gt(0)].copy()
    transactions["role_weight"] = transactions.apply(_owner_role_weight, axis=1)

    events: list[dict[str, Any]] = []
    grouped = transactions.groupby(["issuer_cik", "accession_number", "filing_date"], dropna=False)
    for (issuer_cik, accession, filing_date), group in grouped:
        symbols = cik_to_symbols.get(str(issuer_cik).zfill(10), [])
        if not symbols:
            continue
        effective_session = _next_session_after(filing_date, sessions)
        if not effective_session:
            continue
        buy = group[group["trans_code"].eq("P")].copy()
        sell = group[group["trans_code"].eq("S")].copy()
        buy_value = float((buy["dollar_value"] * buy["role_weight"]).sum())
        sell_value = float((sell["dollar_value"] * sell["role_weight"]).sum())
        owner_buy_count = int(buy["owner_cik"].nunique()) if not buy.empty else 0
        owner_sell_count = int(sell["owner_cik"].nunique()) if not sell.empty else 0
        for symbol in symbols:
            median_adv = float(cik_to_adv.get(str(issuer_cik).zfill(10), 0.0) or 0.0)
            denominator = median_adv if median_adv > 0 else 1.0
            buy_intensity = min((buy_value / denominator) * 10_000.0, 25.0)
            sell_intensity = min((sell_value / denominator) * 10_000.0, 25.0)
            event_score = buy_intensity - 0.25 * sell_intensity
            if abs(event_score) <= 0:
                continue
            events.append(
                {
                    "symbol": symbol,
                    "cik": str(issuer_cik).zfill(10),
                    "accession_number": accession,
                    "filing_date": filing_date,
                    "effective_session": effective_session,
                    "buy_value_role_weighted": buy_value,
                    "sell_value_role_weighted": sell_value,
                    "buy_owner_count": owner_buy_count,
                    "sell_owner_count": owner_sell_count,
                    "open_market_buy_event": owner_buy_count > 0,
                    "open_market_sell_event": owner_sell_count > 0,
                    "insider_net_buy_event_score": float(event_score),
                    "insider_buy_intensity": float(buy_intensity),
                    "insider_sell_intensity": float(sell_intensity),
                    "median_dollar_volume": median_adv,
                }
            )

    event_frame = pd.DataFrame(events)
    if not event_frame.empty:
        event_frame = event_frame.drop_duplicates(["symbol", "accession_number"])
    return event_frame, transactions


def _parse_insider_zip(path: Path, *, target_ciks: set[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        submission = pd.read_csv(
            archive.open("SUBMISSION.tsv"),
            sep="\t",
            dtype=str,
            usecols=["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK"],
        )
        submission["issuer_cik"] = submission["ISSUERCIK"].astype(str).str.zfill(10)
        submission = submission[
            submission["issuer_cik"].isin(target_ciks)
            & submission["DOCUMENT_TYPE"].astype(str).str.upper().isin({"4", "4/A"})
        ].copy()
        if submission.empty:
            return pd.DataFrame()

        nonderiv = pd.read_csv(
            archive.open("NONDERIV_TRANS.tsv"),
            sep="\t",
            dtype=str,
            usecols=[
                "ACCESSION_NUMBER",
                "TRANS_DATE",
                "TRANS_CODE",
                "TRANS_SHARES",
                "TRANS_PRICEPERSHARE",
                "TRANS_ACQUIRED_DISP_CD",
            ],
        )
        nonderiv = nonderiv[
            nonderiv["ACCESSION_NUMBER"].isin(set(submission["ACCESSION_NUMBER"]))
            & nonderiv["TRANS_CODE"].isin({"P", "S"})
            & nonderiv["TRANS_ACQUIRED_DISP_CD"].isin({"A", "D"})
        ].copy()
        if nonderiv.empty:
            return pd.DataFrame()

        owners = pd.read_csv(
            archive.open("REPORTINGOWNER.tsv"),
            sep="\t",
            dtype=str,
            usecols=[
                "ACCESSION_NUMBER",
                "RPTOWNERCIK",
                "RPTOWNERNAME",
                "RPTOWNER_RELATIONSHIP",
                "RPTOWNER_TITLE",
            ],
        )
    owners = owners[owners["ACCESSION_NUMBER"].isin(set(nonderiv["ACCESSION_NUMBER"]))].copy()
    owner_summary = _summarize_reporting_owners(owners)
    merged = nonderiv.merge(
        submission,
        on="ACCESSION_NUMBER",
        how="inner",
        validate="many_to_one",
    ).merge(
        owner_summary,
        on="ACCESSION_NUMBER",
        how="left",
        validate="many_to_one",
    )
    return merged.rename(
        columns={
            "ACCESSION_NUMBER": "accession_number",
            "FILING_DATE": "filing_date",
            "DOCUMENT_TYPE": "document_type",
            "TRANS_DATE": "transaction_date",
            "TRANS_CODE": "trans_code",
            "TRANS_SHARES": "shares",
            "TRANS_PRICEPERSHARE": "price",
            "TRANS_ACQUIRED_DISP_CD": "acquired_disposed",
            "RPTOWNERCIK": "owner_cik",
            "RPTOWNERNAME": "owner_name",
            "RPTOWNER_RELATIONSHIP": "owner_relationship",
            "RPTOWNER_TITLE": "owner_title",
        }
    )


def _summarize_reporting_owners(owners: pd.DataFrame) -> pd.DataFrame:
    if owners.empty:
        return pd.DataFrame(
            columns=[
                "ACCESSION_NUMBER",
                "RPTOWNERCIK",
                "RPTOWNERNAME",
                "RPTOWNER_RELATIONSHIP",
                "RPTOWNER_TITLE",
            ]
        )
    grouped = owners.groupby("ACCESSION_NUMBER", dropna=False).agg(
        {
            "RPTOWNERCIK": lambda values: "|".join(sorted(set(map(str, values.dropna())))),
            "RPTOWNERNAME": lambda values: "|".join(sorted(set(map(str, values.dropna())))),
            "RPTOWNER_RELATIONSHIP": lambda values: "|".join(sorted(set(map(str, values.dropna())))),
            "RPTOWNER_TITLE": lambda values: "|".join(sorted(set(map(str, values.dropna())))),
        }
    )
    return grouped.reset_index()


def _owner_role_weight(row: Mapping[str, Any]) -> float:
    relationship = str(row.get("owner_relationship") or "").lower()
    title = str(row.get("owner_title") or "").lower()
    weight = 1.0
    if "officer" in relationship:
        weight += 0.3
    if "director" in relationship:
        weight += 0.1
    if "ten percent" in relationship or "10%" in relationship:
        weight -= 0.2
    if any(token in title for token in ("chief executive", " ceo", "president")):
        weight += 0.2
    if any(token in title for token in ("chief financial", " cfo")):
        weight += 0.25
    return max(weight, 0.5)


def _build_insider_panel(
    *,
    membership: pd.DataFrame,
    events: pd.DataFrame,
    session_index: Mapping[str, int],
) -> pd.DataFrame:
    base = membership[["session_date", "symbol"]].drop_duplicates().copy()
    columns = [
        "insider_net_buy_score",
        "insider_buy_events_20d",
        "insider_sell_events_20d",
        "insider_cluster_buy_events_60d",
        "insider_buy_intensity_60d",
        "insider_sell_intensity_60d",
        "days_since_last_insider_buy",
    ]
    return _build_decayed_panel(
        base=base,
        events=events,
        session_index=session_index,
        score_column="insider_net_buy_event_score",
        output_score_column="insider_net_buy_score",
        component_specs={
            "insider_buy_events_20d": ("open_market_buy_event", 20),
            "insider_sell_events_20d": ("open_market_sell_event", 20),
            "insider_cluster_buy_events_60d": ("buy_owner_count", 60),
            "insider_buy_intensity_60d": ("insider_buy_intensity", 60),
            "insider_sell_intensity_60d": ("insider_sell_intensity", 60),
        },
        days_since_column="days_since_last_insider_buy",
        days_since_event_column="open_market_buy_event",
        output_columns=columns,
        half_life_sessions=20.0,
        max_age_sessions=126,
    )


def _build_decayed_panel(
    *,
    base: pd.DataFrame,
    events: pd.DataFrame,
    session_index: Mapping[str, int],
    score_column: str,
    output_score_column: str,
    component_specs: Mapping[str, tuple[str, int]],
    days_since_column: str,
    output_columns: Sequence[str],
    half_life_sessions: float,
    max_age_sessions: int,
    days_since_event_column: str | None = None,
) -> pd.DataFrame:
    frame = base.sort_values(["symbol", "session_date"]).reset_index(drop=True).copy()
    for column in output_columns:
        frame[column] = 0.0
    frame[days_since_column] = math.nan

    if events.empty:
        frame[days_since_column] = 9999.0
        return frame

    grouped_events = {
        symbol: group.sort_values("effective_session").to_dict("records")
        for symbol, group in events.groupby("symbol")
    }
    output_frames = []
    for symbol, group in frame.groupby("symbol", sort=False):
        local = group.copy()
        sessions = local["session_date"].tolist()
        local_indices = [session_index[session] for session in sessions]
        values = {column: [0.0] * len(local) for column in output_columns}
        symbol_events = grouped_events.get(symbol, [])
        days_since_event_indices = sorted(
            session_index[str(event.get("effective_session"))]
            for event in symbol_events
            if str(event.get("effective_session")) in session_index
            and (
                days_since_event_column is None
                or _float(event.get(days_since_event_column)) > 0
            )
        )

        for event in symbol_events:
            effective = str(event.get("effective_session") or "")
            if effective not in session_index:
                continue
            event_index = session_index[effective]
            start_pos = bisect.bisect_left(local_indices, event_index)
            score_value = _float(event.get(score_column))
            for pos in range(start_pos, len(local_indices)):
                age = local_indices[pos] - event_index
                if age < 0:
                    continue
                if age > max_age_sessions:
                    break
                decay = 0.5 ** (age / half_life_sessions)
                values[output_score_column][pos] += score_value * decay
                for output_column, (event_column, window) in component_specs.items():
                    component = _float(event.get(event_column))
                    if component > 0 and age <= window:
                        if "_events_" in output_column or "event_count" in output_column:
                            values[output_column][pos] += 1.0
                        else:
                            values[output_column][pos] += component * decay

        for pos, session_int in enumerate(local_indices):
            prior_position = bisect.bisect_right(days_since_event_indices, session_int) - 1
            values[days_since_column][pos] = (
                float(session_int - days_since_event_indices[prior_position])
                if prior_position >= 0
                else 9999.0
            )

        for column, column_values in values.items():
            local[column] = column_values
        output_frames.append(local)

    out = pd.concat(output_frames, ignore_index=True)
    return out[["session_date", "symbol", *output_columns]]


def _write_outputs(
    *,
    output_dir: Path,
    filing_events: pd.DataFrame,
    filing_all: pd.DataFrame,
    filing_panel: pd.DataFrame,
    insider_events: pd.DataFrame,
    insider_transactions: pd.DataFrame,
    insider_panel: pd.DataFrame,
    factor_panel: pd.DataFrame,
) -> dict[str, str]:
    paths = {
        "filing_events": output_dir / "sec_filing_red_flag_events.csv",
        "all_filings": output_dir / "sec_submission_filings_normalized.csv.gz",
        "filing_panel": output_dir / "filing_red_flag_factor_panel.csv.gz",
        "insider_events": output_dir / "sec_insider_net_buy_events.csv",
        "insider_transactions": output_dir / "sec_insider_open_market_transactions.csv.gz",
        "insider_panel": output_dir / "insider_net_buy_factor_panel.csv.gz",
        "factor_panel": output_dir / "non_price_two_factor_panel.csv.gz",
    }
    filing_events.to_csv(paths["filing_events"], index=False)
    filing_all.to_csv(paths["all_filings"], index=False, compression="gzip")
    filing_panel.to_csv(paths["filing_panel"], index=False, compression="gzip")
    insider_events.to_csv(paths["insider_events"], index=False)
    insider_transactions.to_csv(paths["insider_transactions"], index=False, compression="gzip")
    insider_panel.to_csv(paths["insider_panel"], index=False, compression="gzip")
    factor_panel.to_csv(paths["factor_panel"], index=False, compression="gzip")
    return {key: path.as_posix() for key, path in paths.items()}


def _build_summary(
    *,
    output_dir: Path,
    mapping: pd.DataFrame,
    membership: pd.DataFrame,
    filing_events: pd.DataFrame,
    filing_panel: pd.DataFrame,
    insider_events: pd.DataFrame,
    insider_transactions: pd.DataFrame,
    insider_panel: pd.DataFrame,
    factor_panel: pd.DataFrame,
    paths: Mapping[str, str],
    variant: str,
    start: str,
    end: str,
    download_insider_zips: bool,
) -> dict[str, Any]:
    return {
        "created_at_utc": _utc_now(),
        "project_id": PROJECT_ID,
        "artifact_dir": output_dir.as_posix(),
        "variant": variant,
        "start": start,
        "end": end,
        "lockbox_status": "validation_features_only_no_performance_computed",
        "symbols": int(mapping["symbol"].nunique()),
        "membership_rows": int(len(membership)),
        "sessions": int(membership["session_date"].nunique()),
        "filing_red_flag_event_rows": int(len(filing_events)),
        "filing_panel_rows": int(len(filing_panel)),
        "filing_panel_nonzero_rows": int(
            filing_panel["filing_red_flag_score"].abs().gt(0).sum()
            if "filing_red_flag_score" in filing_panel
            else 0
        ),
        "insider_download_requested": bool(download_insider_zips),
        "insider_open_market_transaction_rows": int(len(insider_transactions)),
        "insider_event_rows": int(len(insider_events)),
        "insider_panel_rows": int(len(insider_panel)),
        "insider_panel_nonzero_rows": int(
            insider_panel["insider_net_buy_score"].abs().gt(0).sum()
            if "insider_net_buy_score" in insider_panel
            else 0
        ),
        "factor_panel_rows": int(len(factor_panel)),
        "artifacts": dict(paths),
        "filing_red_flag_form_counts": _value_counts(filing_events, "form", 15),
        "insider_event_symbol_counts_top10": _value_counts(insider_events, "symbol", 10),
        "limitations": [
            "Factor availability is aligned to the next membership session after SEC filing date.",
            "Filing red-flag weights are first-pass white-box weights, not tuned on returns.",
            "Insider score uses non-derivative open-market P/S codes and ADV-normalized dollar value.",
            "Insider sell impact is intentionally downweighted because routine sales are noisy.",
            "No model training, portfolio construction, or lockbox performance is computed.",
        ],
    }


def _value_counts(frame: pd.DataFrame, column: str, limit: int) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts().head(limit).to_dict().items()
    }


def _quarter_labels(start: str, end: str) -> list[str]:
    start_dt = _parse_iso_date(start)
    end_dt = _parse_iso_date(end)
    if start_dt is None or end_dt is None:
        raise ValueError("Invalid start/end date.")
    quarters = []
    for year in range(start_dt.year, end_dt.year + 1):
        for quarter in range(1, 5):
            q_start_month = 3 * (quarter - 1) + 1
            q_end_month = q_start_month + 2
            if year == start_dt.year and q_end_month < start_dt.month:
                continue
            if year == end_dt.year and q_start_month > end_dt.month:
                continue
            quarters.append(f"{year}q{quarter}")
    return quarters


def _quarter_overlaps(quarter: str, start: str, end: str) -> bool:
    try:
        year = int(quarter[:4])
        q = int(quarter[-1])
    except ValueError:
        return False
    start_dt = _parse_iso_date(start)
    end_dt = _parse_iso_date(end)
    if start_dt is None or end_dt is None:
        return False
    q_start_month = 3 * (q - 1) + 1
    q_end_month = q_start_month + 2
    q_start = datetime(year, q_start_month, 1).date()
    if q_end_month == 12:
        q_end = datetime(year, 12, 31).date()
    else:
        q_end = datetime(year, q_end_month + 1, 1).date() - pd.Timedelta(days=1)
    return q_end >= start_dt and q_start <= end_dt


def _download(url: str, output_path: Path, *, sec_user_agent: str | None) -> None:
    user_agent = sec_user_agent or "StockMachine research contact@example.com"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "identity",
        },
    )
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with tmp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _next_session_after(date_value: str, sessions: Sequence[str]) -> str | None:
    if not date_value:
        return None
    position = bisect.bisect_right(sessions, date_value)
    if position >= len(sessions):
        return None
    return sessions[position]


def _parse_items(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    return [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]


def _sec_date_to_iso(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
        return text[:10]
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.date().isoformat()


def _parse_iso_date(value: Any):
    text = _sec_date_to_iso(value)
    if not text:
        return None
    return datetime.strptime(text, "%Y-%m-%d").date()


def _float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(result):
        return 0.0
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build first non-price factor panels.")
    parser.add_argument("--mapping-path", default=str(DEFAULT_ADV30M_MAPPING_PATH))
    parser.add_argument("--membership-path", default=str(DEFAULT_PHASE1_MEMBERSHIP_PATH))
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--submissions-raw-root", default=str(DEFAULT_SUBMISSIONS_RAW_ROOT))
    parser.add_argument("--insider-raw-root", default=str(DEFAULT_INSIDER_RAW_ROOT))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--sec-user-agent", default=None)
    parser.add_argument("--download-insider-zips", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.12)
    parser.add_argument("--max-symbols", type=int, default=None)
    args = parser.parse_args()

    summary = build_non_price_phase1_factors(
        mapping_path=args.mapping_path,
        membership_path=args.membership_path,
        variant=args.variant,
        output_root=args.output_root,
        submissions_raw_root=args.submissions_raw_root,
        insider_raw_root=args.insider_raw_root,
        start=args.start,
        end=args.end,
        sec_user_agent=args.sec_user_agent,
        download_insider_zips=args.download_insider_zips,
        sleep_seconds=args.sleep_seconds,
        max_symbols=args.max_symbols,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
