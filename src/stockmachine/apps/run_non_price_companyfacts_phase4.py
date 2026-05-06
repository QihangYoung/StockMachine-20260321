from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_non_price_data_phase0 import _download
from stockmachine.apps.run_non_price_factor_phase1 import (
    DEFAULT_PHASE1_MEMBERSHIP_PATH,
    _load_mapping,
    _load_membership,
    _next_session_after,
)


PROJECT_ID = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT_ID / "research"
DEFAULT_MAPPING_PATH = (
    RESEARCH_ROOT
    / "non_price_phase0_two_factor_data_prep_20260419"
    / "sec_universe_cik_mapping.csv"
)
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "non_price_phase4_companyfacts_fundamentals_20260505"
DEFAULT_COMPANYFACTS_RAW_ROOT = Path("data") / "raw" / "sec" / "companyfacts"
DEFAULT_VARIANT = "top1000_clean_core_beta_full"
DEFAULT_START = "2013-08-05"
DEFAULT_END = "2019-12-31"
SEC_COMPANYFACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

FACT_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A"}

FACT_SPECS: dict[str, tuple[str, ...]] = {
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "stockholders_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "cash": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ),
    "cash_and_short_term_investments": (
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsAndShortTermInvestments",
    ),
    "current_debt": (
        "ShortTermBorrowings",
        "ShortTermDebt",
        "LongTermDebtCurrent",
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
    ),
    "noncurrent_debt": (
        "LongTermDebtNoncurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
    ),
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
}

RAW_FACT_COLUMNS = tuple(FACT_SPECS.keys())
NUMERIC_PANEL_COLUMNS = (
    *RAW_FACT_COLUMNS,
    "cash_or_st_investments",
    "total_debt",
    "liabilities_to_assets",
    "total_debt_to_assets",
    "current_debt_to_assets",
    "debt_to_cash",
    "cash_to_assets",
    "equity_to_assets",
    "net_income_margin",
    "operating_cash_flow_margin",
    "revenue_change_252d",
    "net_income_change_252d",
    "operating_cash_flow_change_252d",
    "fundamental_leverage_pressure_score",
    "fundamental_profit_stress_score",
    "fundamental_fragility_score",
)


def build_non_price_companyfacts_phase4_data(
    *,
    mapping_path: str | Path = DEFAULT_MAPPING_PATH,
    membership_path: str | Path = DEFAULT_PHASE1_MEMBERSHIP_PATH,
    variant: str = DEFAULT_VARIANT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    companyfacts_raw_root: str | Path = DEFAULT_COMPANYFACTS_RAW_ROOT,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    sec_user_agent: str | None = None,
    download_companyfacts: bool = False,
    refresh: bool = False,
    sleep_seconds: float = 0.12,
    max_symbols: int | None = None,
) -> dict[str, Any]:
    """Build point-in-time SEC companyfacts fundamentals for non-price research."""

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

    raw_root = Path(companyfacts_raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    download_manifest = _ensure_companyfacts(
        mapping=mapping,
        raw_root=raw_root,
        sec_user_agent=sec_user_agent,
        refresh=refresh,
        sleep_seconds=sleep_seconds,
    ) if download_companyfacts else _cached_companyfacts_manifest(mapping=mapping, raw_root=raw_root)

    sessions = sorted(membership["session_date"].unique().tolist())
    fact_events = _build_companyfacts_events(
        mapping=mapping,
        raw_root=raw_root,
        sessions=sessions,
    )
    fundamental_panel = _build_fundamental_panel(
        membership=membership,
        fact_events=fact_events,
    )

    paths = _write_outputs(
        output_dir=output_dir,
        download_manifest=download_manifest,
        fact_events=fact_events,
        fundamental_panel=fundamental_panel,
        factor_schema=_factor_schema(),
    )
    summary = _build_summary(
        output_dir=output_dir,
        mapping=mapping,
        membership=membership,
        download_manifest=download_manifest,
        fact_events=fact_events,
        fundamental_panel=fundamental_panel,
        paths=paths,
        variant=variant,
        start=start,
        end=end,
        download_companyfacts=download_companyfacts,
    )
    summary_path = output_dir / "phase4_companyfacts_fundamentals_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def _ensure_companyfacts(
    *,
    mapping: pd.DataFrame,
    raw_root: Path,
    sec_user_agent: str | None,
    refresh: bool,
    sleep_seconds: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in mapping.to_dict("records"):
        symbol = str(row["symbol"])
        cik = str(row["cik"]).zfill(10)
        path = raw_root / f"CIK{cik}.json"
        status = "cached"
        error = ""
        cache_error = ""
        valid_json = False
        if path.exists() and not refresh:
            _, cache_error = _load_companyfacts_payload(path)
            valid_json = not cache_error
        needs_download = refresh or not path.exists() or bool(cache_error)
        if needs_download:
            url = SEC_COMPANYFACTS_URL_TEMPLATE.format(cik=cik)
            was_invalid = bool(cache_error)
            try:
                _download(url, path, sec_user_agent=sec_user_agent)
                _, cache_error = _load_companyfacts_payload(path)
                valid_json = not cache_error
                if cache_error:
                    status = "downloaded_invalid"
                    error = cache_error
                else:
                    status = "redownloaded_invalid_cache" if was_invalid else "downloaded"
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            except Exception as exc:  # pragma: no cover - network failure path
                status = "failed"
                error = str(exc)
                valid_json = False
        rows.append(
            {
                "symbol": symbol,
                "cik": cik,
                "path": path.as_posix(),
                "exists": path.exists(),
                "valid_json": valid_json,
                "download_status": status,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def _cached_companyfacts_manifest(*, mapping: pd.DataFrame, raw_root: Path) -> pd.DataFrame:
    rows = []
    for row in mapping.to_dict("records"):
        cik = str(row["cik"]).zfill(10)
        path = raw_root / f"CIK{cik}.json"
        _, cache_error = _load_companyfacts_payload(path) if path.exists() else (None, "")
        rows.append(
            {
                "symbol": str(row["symbol"]),
                "cik": cik,
                "path": path.as_posix(),
                "exists": path.exists(),
                "valid_json": path.exists() and not cache_error,
                "download_status": "cached"
                if path.exists() and not cache_error
                else "invalid_cache"
                if path.exists()
                else "missing_not_requested",
                "error": cache_error,
            }
        )
    return pd.DataFrame(rows)


def _build_companyfacts_events(
    *,
    mapping: pd.DataFrame,
    raw_root: Path,
    sessions: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in mapping.to_dict("records"):
        symbol = str(item["symbol"])
        cik = str(item["cik"]).zfill(10)
        path = raw_root / f"CIK{cik}.json"
        if not path.exists():
            continue
        payload, error = _load_companyfacts_payload(path)
        if error or payload is None:
            continue
        rows.extend(
            _extract_fact_rows(
                payload,
                symbol=symbol,
                cik=cik,
                sessions=sessions,
            )
        )
    if not rows:
        return pd.DataFrame(columns=_event_columns())
    raw = pd.DataFrame(rows)
    raw = raw.sort_values(
        [
            "symbol",
            "filed_date",
            "period_end",
            "accession",
            "concept",
            "_source_priority",
        ],
        kind="mergesort",
    )
    raw = raw.drop_duplicates(
        [
            "session_date",
            "symbol",
            "cik",
            "filed_date",
            "period_end",
            "form",
            "fy",
            "fp",
            "accession",
            "concept",
        ],
        keep="first",
    )
    pivot = (
        raw.pivot_table(
            index=[
                "session_date",
                "symbol",
                "cik",
                "filed_date",
                "period_end",
                "form",
                "fy",
                "fp",
                "accession",
            ],
            columns="concept",
            values="value",
            aggfunc="last",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for column in RAW_FACT_COLUMNS:
        if column not in pivot.columns:
            pivot[column] = np.nan
    pivot = pivot[_event_columns()].sort_values(
        ["session_date", "symbol", "filed_date", "period_end", "accession"]
    )
    return pivot.reset_index(drop=True)


def _extract_fact_rows(
    payload: Mapping[str, Any],
    *,
    symbol: str,
    cik: str,
    sessions: Sequence[str],
) -> list[dict[str, Any]]:
    facts = payload.get("facts", {})
    if not isinstance(facts, Mapping):
        return []
    us_gaap = facts.get("us-gaap", {})
    if not isinstance(us_gaap, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    for concept, tags in FACT_SPECS.items():
        for source_priority, tag in enumerate(tags):
            tag_payload = us_gaap.get(tag)
            if not isinstance(tag_payload, Mapping):
                continue
            for unit, values in _iter_fact_units(tag_payload):
                if unit != "USD":
                    continue
                for fact in values:
                    if not isinstance(fact, Mapping):
                        continue
                    filed_date = _clean_date(fact.get("filed"))
                    period_end = _clean_date(fact.get("end"))
                    accession = str(fact.get("accn") or "")
                    form = str(fact.get("form") or "")
                    if not filed_date or not period_end or form not in FACT_FORMS:
                        continue
                    session_date = _next_session_after(filed_date, sessions)
                    if not session_date:
                        continue
                    value = _safe_float(fact.get("val"))
                    if value is None or not np.isfinite(value):
                        continue
                    rows.append(
                        {
                            "session_date": session_date,
                            "symbol": symbol,
                            "cik": cik,
                            "filed_date": filed_date,
                            "period_end": period_end,
                            "form": form,
                            "fy": str(fact.get("fy") or ""),
                            "fp": str(fact.get("fp") or ""),
                            "accession": accession,
                            "concept": concept,
                            "value": value,
                            "_source_tag": tag,
                            "_source_priority": source_priority,
                        }
                    )
    return rows


def _load_companyfacts_payload(path: Path) -> tuple[Mapping[str, Any] | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error:{exc}"
    except OSError as exc:
        return None, f"read_error:{exc}"
    if not isinstance(payload, Mapping):
        return None, "non_mapping_payload"
    return payload, ""


def _iter_fact_units(tag_payload: Mapping[str, Any]) -> list[tuple[str, list[Any]]]:
    units = tag_payload.get("units", {})
    if not isinstance(units, Mapping):
        return []
    ordered = []
    for unit in ("USD", "USD/shares", "shares"):
        values = units.get(unit)
        if isinstance(values, list):
            ordered.append((unit, values))
    for unit, values in units.items():
        if unit in {"USD", "USD/shares", "shares"}:
            continue
        if isinstance(values, list):
            ordered.append((str(unit), values))
    return ordered


def _build_fundamental_panel(
    *,
    membership: pd.DataFrame,
    fact_events: pd.DataFrame,
) -> pd.DataFrame:
    event_columns = list(RAW_FACT_COLUMNS)
    rows: list[dict[str, Any]] = []
    fact_events = fact_events.copy()
    if fact_events.empty:
        fact_events = pd.DataFrame(columns=["symbol", "session_date", "filed_date", *event_columns])

    events_by_symbol = {
        symbol: group.sort_values(["session_date", "filed_date", "period_end"]).reset_index(drop=True)
        for symbol, group in fact_events.groupby("symbol", sort=False)
    }
    for symbol, member_group in membership.groupby("symbol", sort=True):
        symbol_events = events_by_symbol.get(symbol, pd.DataFrame(columns=fact_events.columns))
        state: dict[str, Any] = {column: np.nan for column in event_columns}
        state.update({"last_fundamental_filed_date": "", "last_fundamental_period_end": ""})
        event_idx = 0
        event_records = symbol_events.to_dict("records")
        for session_date in member_group["session_date"].sort_values().tolist():
            while event_idx < len(event_records) and str(event_records[event_idx]["session_date"]) <= session_date:
                event = event_records[event_idx]
                for column in event_columns:
                    value = _safe_float(event.get(column))
                    if value is not None and np.isfinite(value):
                        state[column] = value
                state["last_fundamental_filed_date"] = str(event.get("filed_date") or "")
                state["last_fundamental_period_end"] = str(event.get("period_end") or "")
                event_idx += 1
            row = {"session_date": session_date, "symbol": symbol, **state}
            rows.append(row)
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    panel = _derive_fundamental_features(panel)
    return panel.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _derive_fundamental_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    for column in RAW_FACT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["cash_or_st_investments"] = frame["cash_and_short_term_investments"].where(
        frame["cash_and_short_term_investments"].notna(),
        frame["cash"],
    )
    frame["total_debt"] = frame[["current_debt", "noncurrent_debt"]].sum(axis=1, min_count=1)

    frame["liabilities_to_assets"] = _safe_div(frame["liabilities"], frame["assets"])
    frame["total_debt_to_assets"] = _safe_div(frame["total_debt"], frame["assets"])
    frame["current_debt_to_assets"] = _safe_div(frame["current_debt"], frame["assets"])
    frame["debt_to_cash"] = _safe_div(frame["total_debt"], frame["cash_or_st_investments"].abs())
    frame["cash_to_assets"] = _safe_div(frame["cash_or_st_investments"], frame["assets"])
    frame["equity_to_assets"] = _safe_div(frame["stockholders_equity"], frame["assets"])
    frame["net_income_margin"] = _safe_div(frame["net_income"], frame["revenue"].abs())
    frame["operating_cash_flow_margin"] = _safe_div(
        frame["operating_cash_flow"],
        frame["revenue"].abs(),
    )
    frame["negative_net_income_flag"] = np.where(
        frame["net_income"].notna(),
        frame["net_income"].lt(0).astype(float),
        np.nan,
    )
    frame["negative_operating_cash_flow_flag"] = np.where(
        frame["operating_cash_flow"].notna(),
        frame["operating_cash_flow"].lt(0).astype(float),
        np.nan,
    )

    frame = frame.sort_values(["symbol", "session_date"]).reset_index(drop=True)
    for source, target in (
        ("revenue", "revenue_change_252d"),
        ("net_income", "net_income_change_252d"),
        ("operating_cash_flow", "operating_cash_flow_change_252d"),
    ):
        frame[target] = (
            frame.groupby("symbol", sort=False)[source]
            .pct_change(252, fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
        )

    percentile_specs = (
        ("liabilities_to_assets", "liabilities_to_assets_pct", False),
        ("total_debt_to_assets", "total_debt_to_assets_pct", False),
        ("current_debt_to_assets", "current_debt_to_assets_pct", False),
        ("debt_to_cash", "debt_to_cash_pct", False),
        ("cash_to_assets", "cash_to_assets_low_pct", True),
        ("equity_to_assets", "equity_to_assets_low_pct", True),
        ("net_income_margin", "net_income_margin_low_pct", True),
        ("operating_cash_flow_margin", "operating_cash_flow_margin_low_pct", True),
        ("revenue_change_252d", "revenue_decline_252d_pct", True),
        ("net_income_change_252d", "net_income_decline_252d_pct", True),
        ("operating_cash_flow_change_252d", "operating_cash_flow_decline_252d_pct", True),
    )
    for source, target, inverse in percentile_specs:
        frame[target] = _session_rank_pct(frame, source=source, inverse=inverse)

    frame["fundamental_leverage_pressure_score"] = _mean_available(
        frame,
        [
            "liabilities_to_assets_pct",
            "total_debt_to_assets_pct",
            "current_debt_to_assets_pct",
            "debt_to_cash_pct",
            "cash_to_assets_low_pct",
            "equity_to_assets_low_pct",
        ],
    )
    frame["fundamental_profit_stress_score"] = _mean_available(
        frame,
        [
            "negative_net_income_flag",
            "negative_operating_cash_flow_flag",
            "net_income_margin_low_pct",
            "operating_cash_flow_margin_low_pct",
            "revenue_decline_252d_pct",
            "net_income_decline_252d_pct",
            "operating_cash_flow_decline_252d_pct",
        ],
    )
    frame["fundamental_fragility_score"] = _mean_available(
        frame,
        [
            "fundamental_leverage_pressure_score",
            "fundamental_profit_stress_score",
        ],
    )
    frame["test_window_used"] = False
    return frame.sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _session_rank_pct(frame: pd.DataFrame, *, source: str, inverse: bool) -> pd.Series:
    values = pd.to_numeric(frame[source], errors="coerce")
    ranked = -values if inverse else values
    return ranked.groupby(frame["session_date"]).rank(method="average", pct=True)


def _mean_available(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return frame[list(columns)].mean(axis=1, skipna=True)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denom


def _factor_schema() -> pd.DataFrame:
    rows = [
        {
            "factor_name": "fundamental_leverage_pressure_score",
            "source": "SEC companyfacts us-gaap instant balance-sheet tags",
            "economic_mechanism": "High leverage, low cash, and weak equity cushion raise the risk that a loser is a financing-stressed falling knife.",
            "point_in_time_availability": "available_next_membership_session_after_companyfacts_filed_date",
            "expected_help_on_phi_long": "high",
            "expected_help_on_phi_short": "low",
            "stock_level_use": "penalize_or_veto_long_candidates_with_high_balance_sheet_fragility",
            "basket_level_use": "aggregate leverage pressure across long candidate basket",
            "known_failure_modes": "Bank and insurance balance sheets are structurally different; reported debt/cash tags can be sparse or taxonomy-dependent.",
        },
        {
            "factor_name": "fundamental_profit_stress_score",
            "source": "SEC companyfacts us-gaap duration income/cash-flow tags",
            "economic_mechanism": "Negative earnings, weak cash flow, and deteriorating revenue/profit measures suggest real business deterioration rather than pure price dislocation.",
            "point_in_time_availability": "available_next_membership_session_after_companyfacts_filed_date",
            "expected_help_on_phi_long": "medium_high",
            "expected_help_on_phi_short": "medium",
            "stock_level_use": "diagnose long candidates with worsening business quality",
            "basket_level_use": "aggregate profit stress across long candidate basket",
            "known_failure_modes": "Quarterly vs annual reporting windows are mixed in the MVP; revenue tags vary across filers.",
        },
        {
            "factor_name": "fundamental_fragility_score",
            "source": "composite of leverage pressure and profit stress",
            "economic_mechanism": "Combines financing fragility and operating stress into the first structured-fundamental non-price fragility proxy.",
            "point_in_time_availability": "available_next_membership_session_after_companyfacts_filed_date",
            "expected_help_on_phi_long": "high",
            "expected_help_on_phi_short": "low",
            "stock_level_use": "candidate-level structural loser / falling-knife screen",
            "basket_level_use": "candidate basket fragility temperature",
            "known_failure_modes": "Needs sector-specific normalization before promotion; this phase is data build, not alpha selection.",
        },
    ]
    return pd.DataFrame(rows)


def _write_outputs(
    *,
    output_dir: Path,
    download_manifest: pd.DataFrame,
    fact_events: pd.DataFrame,
    fundamental_panel: pd.DataFrame,
    factor_schema: pd.DataFrame,
) -> dict[str, str]:
    paths = {
        "download_manifest": output_dir / "companyfacts_download_manifest.csv",
        "fact_events": output_dir / "companyfacts_fundamental_events.csv.gz",
        "fundamental_panel": output_dir / "companyfacts_fundamental_panel.csv.gz",
        "factor_schema": output_dir / "companyfacts_factor_schema.csv",
    }
    download_manifest.to_csv(paths["download_manifest"], index=False)
    fact_events.to_csv(paths["fact_events"], index=False, compression="gzip")
    fundamental_panel.to_csv(paths["fundamental_panel"], index=False, compression="gzip")
    factor_schema.to_csv(paths["factor_schema"], index=False)
    return {key: value.as_posix() for key, value in paths.items()}


def _build_summary(
    *,
    output_dir: Path,
    mapping: pd.DataFrame,
    membership: pd.DataFrame,
    download_manifest: pd.DataFrame,
    fact_events: pd.DataFrame,
    fundamental_panel: pd.DataFrame,
    paths: Mapping[str, str],
    variant: str,
    start: str,
    end: str,
    download_companyfacts: bool,
) -> dict[str, Any]:
    coverage = _panel_coverage(fundamental_panel)
    return {
        "created_at_utc": _utc_now(),
        "project_id": PROJECT_ID,
        "artifact_dir": output_dir.as_posix(),
        "variant": variant,
        "start": start,
        "end": end,
        "mapping_symbols": int(len(mapping)),
        "membership_rows": int(len(membership)),
        "sessions": int(membership["session_date"].nunique()),
        "download_companyfacts_requested": bool(download_companyfacts),
        "companyfacts_files_existing": int(download_manifest["exists"].astype(bool).sum())
        if "exists" in download_manifest
        else 0,
        "companyfacts_files_valid_json": int(download_manifest["valid_json"].astype(bool).sum())
        if "valid_json" in download_manifest
        else 0,
        "companyfacts_files_failed": int(download_manifest["download_status"].eq("failed").sum())
        if "download_status" in download_manifest
        else 0,
        "fact_event_rows": int(len(fact_events)),
        "fact_event_symbols": int(fact_events["symbol"].nunique()) if not fact_events.empty else 0,
        "fundamental_panel_rows": int(len(fundamental_panel)),
        "fundamental_panel_symbols": int(fundamental_panel["symbol"].nunique())
        if not fundamental_panel.empty
        else 0,
        "coverage": coverage,
        "artifacts": dict(paths),
        "method": "sec_companyfacts_structured_fundamental_point_in_time_panel",
        "lockbox_status": "validation_features_only_no_test_window_performance",
        "limitations": [
            "Companyfacts taxonomy coverage differs by filer and concept.",
            "This MVP mixes quarterly and annual duration facts; downstream alpha tests should add sector and reporting-period guardrails.",
            "Bank, broker, insurance, and REIT balance sheets need specialized interpretation.",
            "No model training, portfolio construction, or test-window performance is computed.",
        ],
    }


def _panel_coverage(panel: pd.DataFrame) -> dict[str, Any]:
    if panel.empty:
        return {}
    rows = int(len(panel))
    output: dict[str, Any] = {}
    for column in (
        "assets",
        "liabilities",
        "cash_or_st_investments",
        "total_debt",
        "revenue",
        "net_income",
        "operating_cash_flow",
        "fundamental_leverage_pressure_score",
        "fundamental_profit_stress_score",
        "fundamental_fragility_score",
    ):
        if column in panel:
            output[f"{column}_non_null_rate"] = float(panel[column].notna().sum() / rows)
    return output


def _event_columns() -> list[str]:
    return [
        "session_date",
        "symbol",
        "cik",
        "filed_date",
        "period_end",
        "form",
        "fy",
        "fp",
        "accession",
        *RAW_FACT_COLUMNS,
    ]


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _clean_date(value: Any) -> str:
    text = str(value or "")
    return text if len(text) == 10 and text[4] == "-" and text[7] == "-" else ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build SEC companyfacts fundamental non-price data."
    )
    parser.add_argument("--mapping-path", default=str(DEFAULT_MAPPING_PATH))
    parser.add_argument("--membership-path", default=str(DEFAULT_PHASE1_MEMBERSHIP_PATH))
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--companyfacts-raw-root", default=str(DEFAULT_COMPANYFACTS_RAW_ROOT))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--sec-user-agent", default=None)
    parser.add_argument("--download-companyfacts", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.12)
    parser.add_argument("--max-symbols", type=int, default=None)
    args = parser.parse_args(argv)

    summary = build_non_price_companyfacts_phase4_data(
        mapping_path=args.mapping_path,
        membership_path=args.membership_path,
        variant=args.variant,
        output_root=args.output_root,
        companyfacts_raw_root=args.companyfacts_raw_root,
        start=args.start,
        end=args.end,
        sec_user_agent=args.sec_user_agent,
        download_companyfacts=args.download_companyfacts,
        refresh=args.refresh,
        sleep_seconds=args.sleep_seconds,
        max_symbols=args.max_symbols,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
