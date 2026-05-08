"""Build a validation-only MVP point-in-time size panel for pure alpha.

This phase extracts SEC companyfacts shares-outstanding facts, carries them
forward by filing availability, and combines them with lagged raw close prices
to create a market-cap proxy suitable for validation-window neutrality work.

It deliberately does not evaluate test-window performance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stockmachine.apps.run_non_price_companyfacts_phase4 import (
    DEFAULT_COMPANYFACTS_RAW_ROOT,
    FACT_FORMS,
    _cached_companyfacts_manifest,
    _clean_date,
    _iter_fact_units,
    _load_companyfacts_payload,
    _safe_float,
)
from stockmachine.apps.run_non_price_factor_phase1 import _load_mapping, _next_session_after
from stockmachine.apps.run_pure_alpha_phase4d import RESEARCH_ROOT
from stockmachine.apps.run_pure_alpha_phase4s import DEFAULT_H10_SIGNAL_PANEL


DEFAULT_MAPPING_PATH = (
    RESEARCH_ROOT
    / "non_price_phase0_two_factor_data_prep_20260419"
    / "sec_universe_cik_mapping.csv"
)
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "phase6j_true_size_mvp_20260508"
DEFAULT_START = "2013-08-05"
DEFAULT_END = "2019-12-31"
DEFAULT_VARIANTS = (
    "top1000_clean_core_beta_full",
    "adv30m_clean_core_beta_full",
)
DEFAULT_PRICE_GLOBS = (
    "data/silver/daily_bar/phase0_top1000_yahoo_gap_20130805_20151231_chunk*.jsonl",
    "data/silver/daily_bar/phase0_top1000_sip_raw_20160104_20260416_chunk*.jsonl",
)
DEFAULT_SOTA_POSITIONS_PATH = (
    RESEARCH_ROOT
    / "phase6e_adv_floor_hardening_20260507"
    / "phase6e_positions_validation.csv.gz"
)
DEFAULT_SOTA_PORTFOLIO = "adv_floor_1m"
TRADING_DAYS_PER_YEAR = 252
STALE_SHARES_DAYS = 550


@dataclass(frozen=True)
class ShareFactSpec:
    feature: str
    namespace: str
    tags: tuple[str, ...]
    source_priority: int
    is_spot_shares: bool


SHARE_FACT_SPECS = (
    ShareFactSpec(
        feature="entity_common_stock_shares_outstanding",
        namespace="dei",
        tags=("EntityCommonStockSharesOutstanding",),
        source_priority=0,
        is_spot_shares=True,
    ),
    ShareFactSpec(
        feature="weighted_average_basic_shares",
        namespace="us-gaap",
        tags=(
            "WeightedAverageNumberOfSharesOutstandingBasic",
            "WeightedAverageSharesOutstandingBasic",
        ),
        source_priority=10,
        is_spot_shares=False,
    ),
    ShareFactSpec(
        feature="weighted_average_diluted_shares",
        namespace="us-gaap",
        tags=(
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageSharesOutstandingDiluted",
        ),
        source_priority=20,
        is_spot_shares=False,
    ),
)


def build_phase6j_true_size_mvp_artifacts(
    *,
    signal_panel_path: str | Path = DEFAULT_H10_SIGNAL_PANEL,
    mapping_path: str | Path = DEFAULT_MAPPING_PATH,
    companyfacts_raw_root: str | Path = DEFAULT_COMPANYFACTS_RAW_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    price_globs: Sequence[str] = DEFAULT_PRICE_GLOBS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    stale_shares_days: int = STALE_SHARES_DAYS,
    sota_positions_path: str | Path = DEFAULT_SOTA_POSITIONS_PATH,
    sota_portfolio: str = DEFAULT_SOTA_PORTFOLIO,
) -> dict[str, Any]:
    """Build the MVP true-size panel and validation QA artifacts."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal = _load_signal_membership(
        signal_panel_path=Path(signal_panel_path),
        variants=tuple(variants),
        start=start,
        end=end,
    )
    if signal.empty:
        raise ValueError("No validation signal rows found for requested variants/date range.")

    sessions = sorted(signal["session_date"].unique().tolist())
    symbol_set = set(signal["symbol"].unique().tolist())
    mapping = _load_mapping(Path(mapping_path), max_symbols=None)
    mapping = mapping[mapping["symbol"].isin(symbol_set)].copy()
    missing_mapping = sorted(symbol_set - set(mapping["symbol"]))

    companyfacts_manifest = _cached_companyfacts_manifest(
        mapping=mapping,
        raw_root=Path(companyfacts_raw_root),
    )
    share_events = _build_share_events(
        mapping=mapping,
        raw_root=Path(companyfacts_raw_root),
        sessions=sessions,
    )
    share_state = _build_share_state_panel(
        signal[["session_date", "symbol"]].drop_duplicates(),
        share_events,
        stale_shares_days=stale_shares_days,
    )
    raw_prices = _load_lagged_raw_close(
        price_globs=tuple(price_globs),
        symbols=symbol_set,
        start=start,
        end=end,
    )
    panel = _build_size_panel(signal=signal, share_state=share_state, raw_prices=raw_prices)
    coverage = _build_coverage_table(
        panel=panel,
        share_events=share_events,
        companyfacts_manifest=companyfacts_manifest,
        missing_mapping=missing_mapping,
    )
    sota_exposure = _build_sota_size_exposure(
        panel=panel,
        positions_path=Path(sota_positions_path),
        portfolio=sota_portfolio,
    )

    paths = _write_outputs(
        output_dir=output_dir,
        share_events=share_events,
        panel=panel,
        coverage=coverage,
        companyfacts_manifest=companyfacts_manifest,
        sota_exposure=sota_exposure,
    )
    memo_path = output_dir / "phase6j_true_size_mvp_memo.md"
    memo_path.write_text(
        _memo(
            panel=panel,
            coverage=coverage,
            sota_exposure=sota_exposure,
            paths=paths,
            missing_mapping=missing_mapping,
        ),
        encoding="utf-8",
    )
    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "signal_panel_path": Path(signal_panel_path).as_posix(),
        "mapping_path": Path(mapping_path).as_posix(),
        "companyfacts_raw_root": Path(companyfacts_raw_root).as_posix(),
        "variants": list(variants),
        "start": start,
        "end": end,
        "stale_shares_days": int(stale_shares_days),
        "signal_rows": int(len(signal)),
        "signal_symbols": int(signal["symbol"].nunique()),
        "mapping_symbols": int(mapping["symbol"].nunique()),
        "missing_mapping_symbols": missing_mapping,
        "share_events_rows": int(len(share_events)),
        "panel_rows": int(len(panel)),
        "market_cap_non_null_rate": _non_null_rate(panel, "market_cap"),
        "spot_shares_market_cap_rate": _mean_true(panel, "market_cap_uses_spot_shares"),
        "raw_price_market_cap_rate": float(panel["market_cap_price_source"].eq("lagged_raw_close").mean())
        if not panel.empty
        else np.nan,
        "coverage_artifact": paths["coverage"].as_posix(),
        "share_events_artifact": paths["share_events"].as_posix(),
        "size_panel_artifact": paths["panel"].as_posix(),
        "sota_exposure_artifact": paths["sota_exposure"].as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "method": "validation_only_sec_companyfacts_pit_shares_raw_lagged_close_market_cap_mvp",
        "lockbox_status": "validation_only_no_test_window_performance",
        "limitations": [
            "SEC companyfacts coverage and taxonomy differ by filer.",
            "EntityCommonStockSharesOutstanding is preferred; weighted-average shares are fallback only.",
            "Symbol-to-CIK mapping is current-symbol based and remains a known survivorship/mapping limitation.",
            "Yahoo 2013-2015 raw closes are reconstructed by the existing gap-fill source.",
            "This phase builds data and QA only; it does not tune or evaluate test-window performance.",
        ],
    }
    rollup_path = output_dir / "phase6j_true_size_mvp_rollup.json"
    rollup_path.write_text(json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8")
    return rollup


def _load_signal_membership(
    *,
    signal_panel_path: Path,
    variants: tuple[str, ...],
    start: str,
    end: str,
) -> pd.DataFrame:
    usecols = [
        "variant",
        "session_date",
        "symbol",
        "lagged_close",
        "trailing_median_dollar_volume_20",
        "liquidity_rank",
    ]
    frame = pd.read_csv(signal_panel_path, usecols=usecols, dtype={"symbol": str})
    frame["variant"] = frame["variant"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame[
        frame["variant"].isin(set(variants))
        & frame["session_date"].ge(start)
        & frame["session_date"].le(end)
    ].copy()
    for column in ("lagged_close", "trailing_median_dollar_volume_20", "liquidity_rank"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["test_window_used"] = False
    return frame.sort_values(["variant", "session_date", "symbol"]).reset_index(drop=True)


def _build_share_events(
    *,
    mapping: pd.DataFrame,
    raw_root: Path,
    sessions: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in mapping.to_dict("records"):
        symbol = str(item["symbol"]).upper()
        cik = str(item["cik"]).zfill(10)
        path = raw_root / f"CIK{cik}.json"
        if not path.exists():
            continue
        payload, error = _load_companyfacts_payload(path)
        if error or payload is None:
            continue
        rows.extend(_extract_share_fact_rows(payload, symbol=symbol, cik=cik, sessions=sessions))
    if not rows:
        return pd.DataFrame(columns=_share_event_columns())

    raw = pd.DataFrame(rows)
    # A single filing can contain one spot share count plus many weighted-average
    # duration rows for older periods. Prefer spot shares; within the same
    # source priority, keep the latest period end.
    raw = raw.sort_values(
        [
            "symbol",
            "session_date",
            "filed_date",
            "accession",
            "source_priority",
            "period_end",
        ],
        ascending=[True, True, True, True, True, False],
        kind="mergesort",
    )
    raw = raw.drop_duplicates(
        ["symbol", "session_date", "filed_date", "accession"],
        keep="first",
    )
    return raw[_share_event_columns()].reset_index(drop=True)


def _extract_share_fact_rows(
    payload: Mapping[str, Any],
    *,
    symbol: str,
    cik: str,
    sessions: Sequence[str],
) -> list[dict[str, Any]]:
    facts = payload.get("facts", {})
    if not isinstance(facts, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    for spec in SHARE_FACT_SPECS:
        namespace_payload = facts.get(spec.namespace, {})
        if not isinstance(namespace_payload, Mapping):
            continue
        for tag_offset, tag in enumerate(spec.tags):
            tag_payload = namespace_payload.get(tag)
            if not isinstance(tag_payload, Mapping):
                continue
            for unit, values in _iter_fact_units(tag_payload):
                if unit != "shares":
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
                    if value is None or not np.isfinite(value) or value <= 0:
                        continue
                    if value < 1_000 or value > 100_000_000_000_000:
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
                            "shares_outstanding": float(value),
                            "share_source": spec.feature,
                            "source_namespace": spec.namespace,
                            "source_tag": tag,
                            "source_priority": int(spec.source_priority + tag_offset),
                            "is_spot_shares": bool(spec.is_spot_shares),
                        }
                    )
    return rows


def _build_share_state_panel(
    membership: pd.DataFrame,
    share_events: pd.DataFrame,
    *,
    stale_shares_days: int,
) -> pd.DataFrame:
    membership = membership.copy()
    membership["session_date"] = membership["session_date"].astype(str)
    membership["symbol"] = membership["symbol"].astype(str).str.upper()
    if share_events.empty:
        return pd.DataFrame(
            columns=[
                "session_date",
                "symbol",
                "shares_outstanding",
                "share_source",
                "share_source_tag",
                "share_filed_date",
                "share_period_end",
                "share_form",
                "share_accession",
                "share_source_priority",
                "share_is_spot",
                "shares_age_days",
                "shares_stale",
            ]
        )

    events_by_symbol = {
        symbol: group.sort_values(
            ["session_date", "filed_date", "source_priority", "period_end"],
            ascending=[True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        for symbol, group in share_events.groupby("symbol", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for symbol, member_group in membership.groupby("symbol", sort=True):
        symbol_events = events_by_symbol.get(symbol, pd.DataFrame(columns=share_events.columns))
        records = symbol_events.to_dict("records")
        event_idx = 0
        state: dict[str, Any] = {
            "shares_outstanding": np.nan,
            "share_source": "",
            "share_source_tag": "",
            "share_filed_date": "",
            "share_period_end": "",
            "share_form": "",
            "share_accession": "",
            "share_source_priority": np.nan,
            "share_is_spot": False,
        }
        for session_date in sorted(member_group["session_date"].unique().tolist()):
            while event_idx < len(records) and str(records[event_idx]["session_date"]) <= session_date:
                event = records[event_idx]
                shares = _safe_float(event.get("shares_outstanding"))
                if shares is not None and np.isfinite(shares) and shares > 0:
                    state = {
                        "shares_outstanding": float(shares),
                        "share_source": str(event.get("share_source") or ""),
                        "share_source_tag": str(event.get("source_tag") or ""),
                        "share_filed_date": str(event.get("filed_date") or ""),
                        "share_period_end": str(event.get("period_end") or ""),
                        "share_form": str(event.get("form") or ""),
                        "share_accession": str(event.get("accession") or ""),
                        "share_source_priority": int(event.get("source_priority") or 0),
                        "share_is_spot": bool(event.get("is_spot_shares")),
                    }
                event_idx += 1
            age = _date_diff_days(session_date, str(state.get("share_filed_date") or ""))
            rows.append(
                {
                    "session_date": session_date,
                    "symbol": symbol,
                    **state,
                    "shares_age_days": age,
                    "shares_stale": bool(age is not None and age > stale_shares_days),
                }
            )
    return pd.DataFrame(rows).sort_values(["session_date", "symbol"]).reset_index(drop=True)


def _load_lagged_raw_close(
    *,
    price_globs: tuple[str, ...],
    symbols: set[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pattern in price_globs:
        for path in sorted(Path().glob(pattern)):
            for chunk in pd.read_json(path, lines=True, chunksize=200_000):
                if chunk.empty:
                    continue
                for column in ("session_date", "symbol", "close", "source_name", "source_version"):
                    if column not in chunk.columns:
                        chunk[column] = np.nan
                frame = chunk[["session_date", "symbol", "close", "source_name", "source_version"]].copy()
                frame["session_date"] = frame["session_date"].astype(str)
                frame["symbol"] = frame["symbol"].astype(str).str.upper()
                frame = frame[
                    frame["symbol"].isin(symbols)
                    & frame["session_date"].ge(start)
                    & frame["session_date"].le(end)
                ].copy()
                if not frame.empty:
                    frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "session_date",
                "symbol",
                "raw_close",
                "lagged_raw_close",
                "lagged_raw_close_date",
                "price_source_name",
                "price_source_version",
            ]
        )
    prices = pd.concat(frames, ignore_index=True)
    prices["raw_close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["session_date", "symbol", "raw_close"])
    prices = prices.sort_values(["symbol", "session_date"], kind="mergesort")
    prices = prices.drop_duplicates(["session_date", "symbol"], keep="last")
    grouped = prices.groupby("symbol", group_keys=False)
    prices["lagged_raw_close"] = grouped["raw_close"].shift(1)
    prices["lagged_raw_close_date"] = grouped["session_date"].shift(1)
    prices = prices.rename(
        columns={
            "source_name": "price_source_name",
            "source_version": "price_source_version",
        }
    )
    return prices[
        [
            "session_date",
            "symbol",
            "raw_close",
            "lagged_raw_close",
            "lagged_raw_close_date",
            "price_source_name",
            "price_source_version",
        ]
    ].reset_index(drop=True)


def _build_size_panel(
    *,
    signal: pd.DataFrame,
    share_state: pd.DataFrame,
    raw_prices: pd.DataFrame,
) -> pd.DataFrame:
    frame = signal.merge(share_state, on=["session_date", "symbol"], how="left")
    frame = frame.merge(raw_prices, on=["session_date", "symbol"], how="left")

    frame["market_cap_price"] = frame["lagged_raw_close"].where(
        frame["lagged_raw_close"].notna(),
        frame["lagged_close"],
    )
    frame["market_cap_price_source"] = np.where(
        frame["lagged_raw_close"].notna(),
        "lagged_raw_close",
        np.where(frame["lagged_close"].notna(), "signal_lagged_close_fallback", ""),
    )
    frame["market_cap"] = frame["market_cap_price"] * frame["shares_outstanding"]
    frame.loc[~np.isfinite(frame["market_cap"]) | (frame["market_cap"] <= 0), "market_cap"] = np.nan
    frame["market_cap_log"] = np.log(frame["market_cap"])
    frame["market_cap_log_z"] = frame.groupby(["variant", "session_date"])["market_cap_log"].transform(
        _zscore
    )
    frame["market_cap_percentile"] = frame.groupby(["variant", "session_date"])[
        "market_cap"
    ].rank(pct=True)
    frame["market_cap_uses_spot_shares"] = frame["share_is_spot"].fillna(False).astype(bool)
    frame["market_cap_uses_weighted_avg_shares"] = (
        frame["share_source"].fillna("").astype(str).str.startswith("weighted_average")
    )
    frame["market_cap_mvp_available"] = frame["market_cap"].notna()
    frame["test_window_used"] = False
    return frame.sort_values(["variant", "session_date", "symbol"]).reset_index(drop=True)


def _build_coverage_table(
    *,
    panel: pd.DataFrame,
    share_events: pd.DataFrame,
    companyfacts_manifest: pd.DataFrame,
    missing_mapping: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "scope": "companyfacts_cache",
            "rows": int(len(companyfacts_manifest)),
            "symbols": int(companyfacts_manifest["symbol"].nunique())
            if "symbol" in companyfacts_manifest
            else 0,
            "valid_json_rate": float(companyfacts_manifest["valid_json"].astype(bool).mean())
            if not companyfacts_manifest.empty
            else np.nan,
            "market_cap_non_null_rate": np.nan,
            "spot_shares_rate": np.nan,
            "raw_price_rate": np.nan,
            "stale_shares_rate": np.nan,
            "notes": f"missing_mapping_symbols={len(missing_mapping)}",
        }
    )
    rows.append(
        {
            "scope": "share_events",
            "rows": int(len(share_events)),
            "symbols": int(share_events["symbol"].nunique()) if not share_events.empty else 0,
            "valid_json_rate": np.nan,
            "market_cap_non_null_rate": np.nan,
            "spot_shares_rate": _mean_true(share_events, "is_spot_shares"),
            "raw_price_rate": np.nan,
            "stale_shares_rate": np.nan,
            "notes": _source_mix_note(share_events, "share_source"),
        }
    )
    for variant, group in panel.groupby("variant", sort=True):
        rows.append(
            {
                "scope": f"panel:{variant}",
                "rows": int(len(group)),
                "symbols": int(group["symbol"].nunique()),
                "valid_json_rate": np.nan,
                "market_cap_non_null_rate": _non_null_rate(group, "market_cap"),
                "spot_shares_rate": _mean_true(group, "market_cap_uses_spot_shares"),
                "raw_price_rate": float(group["market_cap_price_source"].eq("lagged_raw_close").mean()),
                "stale_shares_rate": _mean_true(group, "shares_stale"),
                "notes": _source_mix_note(group, "share_source"),
            }
        )
    rows.append(
        {
            "scope": "panel:all",
            "rows": int(len(panel)),
            "symbols": int(panel["symbol"].nunique()) if not panel.empty else 0,
            "valid_json_rate": np.nan,
            "market_cap_non_null_rate": _non_null_rate(panel, "market_cap"),
            "spot_shares_rate": _mean_true(panel, "market_cap_uses_spot_shares"),
            "raw_price_rate": float(panel["market_cap_price_source"].eq("lagged_raw_close").mean())
            if not panel.empty
            else np.nan,
            "stale_shares_rate": _mean_true(panel, "shares_stale"),
            "notes": _source_mix_note(panel, "share_source"),
        }
    )
    return pd.DataFrame(rows)


def _build_sota_size_exposure(
    *,
    panel: pd.DataFrame,
    positions_path: Path,
    portfolio: str,
) -> pd.DataFrame:
    if not positions_path.exists():
        return pd.DataFrame()
    usecols = [
        "session_date",
        "portfolio",
        "long_variant",
        "short_variant",
        "side",
        "symbol",
        "side_weight",
        "signed_weight",
        "test_window_used",
    ]
    positions = pd.read_csv(positions_path, usecols=usecols)
    positions = positions[positions["portfolio"].astype(str).eq(portfolio)].copy()
    if positions.empty:
        return pd.DataFrame()
    positions["session_date"] = positions["session_date"].astype(str)
    positions["symbol"] = positions["symbol"].astype(str).str.upper()
    positions["position_variant"] = np.where(
        positions["side"].astype(str).eq("long"),
        positions["long_variant"].astype(str),
        positions["short_variant"].astype(str),
    )
    merge_cols = [
        "variant",
        "session_date",
        "symbol",
        "market_cap",
        "market_cap_log",
        "market_cap_log_z",
        "market_cap_uses_spot_shares",
        "market_cap_price_source",
    ]
    merged = positions.merge(
        panel[merge_cols],
        left_on=["position_variant", "session_date", "symbol"],
        right_on=["variant", "session_date", "symbol"],
        how="left",
    )
    merged["signed_weight"] = pd.to_numeric(merged["signed_weight"], errors="coerce")
    merged["side_weight"] = pd.to_numeric(merged["side_weight"], errors="coerce")
    merged["signed_size_z"] = merged["signed_weight"] * merged["market_cap_log_z"]

    daily = (
        merged.groupby("session_date", as_index=False)
        .agg(
            position_rows=("symbol", "size"),
            market_cap_rows=("market_cap", lambda s: int(s.notna().sum())),
            spot_share_rows=("market_cap_uses_spot_shares", lambda s: int(s.fillna(False).sum())),
            raw_price_rows=(
                "market_cap_price_source",
                lambda s: int(s.astype(str).eq("lagged_raw_close").sum()),
            ),
            net_market_cap_log_z=("signed_size_z", "sum"),
            long_market_cap_log_z=(
                "market_cap_log_z",
                lambda s: _weighted_side_mean(merged.loc[s.index], "long"),
            ),
            short_market_cap_log_z=(
                "market_cap_log_z",
                lambda s: _weighted_side_mean(merged.loc[s.index], "short"),
            ),
        )
        .sort_values("session_date")
    )
    if daily.empty:
        return daily
    summary = {
        "session_date": "SUMMARY",
        "position_rows": int(merged["symbol"].count()),
        "market_cap_rows": int(merged["market_cap"].notna().sum()),
        "spot_share_rows": int(merged["market_cap_uses_spot_shares"].fillna(False).sum()),
        "raw_price_rows": int(merged["market_cap_price_source"].astype(str).eq("lagged_raw_close").sum()),
        "net_market_cap_log_z": float(daily["net_market_cap_log_z"].mean()),
        "long_market_cap_log_z": float(daily["long_market_cap_log_z"].mean()),
        "short_market_cap_log_z": float(daily["short_market_cap_log_z"].mean()),
        "mean_abs_net_market_cap_log_z": float(daily["net_market_cap_log_z"].abs().mean()),
        "p90_abs_net_market_cap_log_z": float(daily["net_market_cap_log_z"].abs().quantile(0.90)),
        "max_abs_net_market_cap_log_z": float(daily["net_market_cap_log_z"].abs().max()),
        "market_cap_coverage_rate": float(merged["market_cap"].notna().mean()),
        "spot_share_rate": float(merged["market_cap_uses_spot_shares"].fillna(False).mean()),
        "raw_price_rate": float(merged["market_cap_price_source"].astype(str).eq("lagged_raw_close").mean()),
    }
    daily["market_cap_coverage_rate"] = daily["market_cap_rows"] / daily["position_rows"]
    daily["spot_share_rate"] = daily["spot_share_rows"] / daily["position_rows"]
    daily["raw_price_rate"] = daily["raw_price_rows"] / daily["position_rows"]
    return pd.concat([pd.DataFrame([summary]), daily], ignore_index=True, sort=False)


def _weighted_side_mean(frame: pd.DataFrame, side: str) -> float:
    sub = frame[frame["side"].astype(str).eq(side)].dropna(subset=["market_cap_log_z", "side_weight"])
    if sub.empty:
        return np.nan
    return float((sub["side_weight"].astype(float) * sub["market_cap_log_z"].astype(float)).sum())


def _write_outputs(
    *,
    output_dir: Path,
    share_events: pd.DataFrame,
    panel: pd.DataFrame,
    coverage: pd.DataFrame,
    companyfacts_manifest: pd.DataFrame,
    sota_exposure: pd.DataFrame,
) -> dict[str, Path]:
    paths = {
        "companyfacts_manifest": output_dir / "phase6j_companyfacts_cache_manifest.csv",
        "share_events": output_dir / "phase6j_true_size_share_events.csv.gz",
        "panel": output_dir / "phase6j_true_size_panel_validation.csv.gz",
        "coverage": output_dir / "phase6j_true_size_coverage.csv",
        "sota_exposure": output_dir / "phase6j_sota_size_exposure_validation.csv",
    }
    companyfacts_manifest.to_csv(paths["companyfacts_manifest"], index=False)
    share_events.to_csv(paths["share_events"], index=False, compression="gzip")
    panel.to_csv(paths["panel"], index=False, compression="gzip")
    coverage.to_csv(paths["coverage"], index=False)
    sota_exposure.to_csv(paths["sota_exposure"], index=False)
    return paths


def _memo(
    *,
    panel: pd.DataFrame,
    coverage: pd.DataFrame,
    sota_exposure: pd.DataFrame,
    paths: dict[str, Path],
    missing_mapping: Sequence[str],
) -> str:
    lines = [
        "# Phase6J MVP True Size Data",
        "",
        "Scope: validation-window only. Test lockbox remains closed.",
        "",
        "This phase builds a first point-in-time market-cap panel for size-neutral pure-alpha research.",
        "It uses SEC companyfacts shares-outstanding facts, makes them available on the next validation session after filing date, and carries them forward by symbol.",
        "",
        "## Price and share convention",
        "",
        "- Preferred share source: `EntityCommonStockSharesOutstanding` from SEC `dei` facts.",
        "- Fallback share source: basic then diluted weighted-average shares from `us-gaap` facts.",
        "- Preferred price source: lagged raw close from the local Phase0 raw daily bars.",
        "- Fallback price source: signal-panel `lagged_close` if raw close is missing.",
        "- Market cap: `market_cap = market_cap_price * shares_outstanding`.",
        "",
        "## Coverage",
        "",
        "```text",
        coverage.to_string(index=False),
        "```",
        "",
    ]
    if not sota_exposure.empty:
        summary = sota_exposure[sota_exposure["session_date"].astype(str).eq("SUMMARY")]
        lines.extend(
            [
                "## Current SOTA size exposure",
                "",
                "The table below is descriptive only. It does not use test data and does not change strategy parameters.",
                "",
                "```text",
                summary.to_string(index=False),
                "```",
                "",
            ]
        )
    if missing_mapping:
        lines.extend(
            [
                "## Missing mapping symbols",
                "",
                ", ".join(missing_mapping[:100]) + (" ..." if len(missing_mapping) > 100 else ""),
                "",
            ]
        )
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- share events: `{paths['share_events'].as_posix()}`",
            f"- size panel: `{paths['panel'].as_posix()}`",
            f"- coverage QA: `{paths['coverage'].as_posix()}`",
            f"- SOTA size exposure: `{paths['sota_exposure'].as_posix()}`",
            "",
            "## Limitations",
            "",
            "- This is an MVP, not a final production-grade market-cap dataset.",
            "- Current symbol-to-CIK mapping can miss historical ticker changes and class-level changes.",
            "- Weighted-average shares are fallback only and are not identical to spot shares outstanding.",
            "- Float market cap is not available yet.",
            "- This data should be used first for validation-window neutrality experiments and exposure QA, not for test-window tuning.",
        ]
    )
    return "\n".join(lines) + "\n"


def _share_event_columns() -> list[str]:
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
        "shares_outstanding",
        "share_source",
        "source_namespace",
        "source_tag",
        "source_priority",
        "is_spot_shares",
    ]


def _zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.nan, index=values.index)
    return (values - values.mean()) / std


def _date_diff_days(date_value: str, prior_date_value: str) -> int | None:
    if not date_value or not prior_date_value:
        return None
    try:
        return int((pd.Timestamp(date_value) - pd.Timestamp(prior_date_value)).days)
    except (TypeError, ValueError):
        return None


def _non_null_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return float(frame[column].notna().mean())


def _mean_true(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return float(frame[column].fillna(False).astype(bool).mean())


def _source_mix_note(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    counts = frame[column].fillna("missing").astype(str).value_counts(normalize=True).head(5)
    return "; ".join(f"{idx}={value:.3f}" for idx, value in counts.items())


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build pure-alpha MVP true-size data.")
    parser.add_argument("--signal-panel-path", default=str(DEFAULT_H10_SIGNAL_PANEL))
    parser.add_argument("--mapping-path", default=str(DEFAULT_MAPPING_PATH))
    parser.add_argument("--companyfacts-raw-root", default=str(DEFAULT_COMPANYFACTS_RAW_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--price-globs", nargs="+", default=list(DEFAULT_PRICE_GLOBS))
    parser.add_argument("--stale-shares-days", type=int, default=STALE_SHARES_DAYS)
    parser.add_argument("--sota-positions-path", default=str(DEFAULT_SOTA_POSITIONS_PATH))
    parser.add_argument("--sota-portfolio", default=DEFAULT_SOTA_PORTFOLIO)
    args = parser.parse_args(argv)

    rollup = build_phase6j_true_size_mvp_artifacts(
        signal_panel_path=args.signal_panel_path,
        mapping_path=args.mapping_path,
        companyfacts_raw_root=args.companyfacts_raw_root,
        output_root=args.output_root,
        variants=tuple(args.variants),
        price_globs=tuple(args.price_globs),
        start=args.start,
        end=args.end,
        stale_shares_days=args.stale_shares_days,
        sota_positions_path=args.sota_positions_path,
        sota_portfolio=args.sota_portfolio,
    )
    print(json.dumps(rollup, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
