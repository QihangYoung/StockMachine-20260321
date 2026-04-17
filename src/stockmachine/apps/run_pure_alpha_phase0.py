from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


PROJECT_ID = "us_equities_pure_alpha_h5"
RESEARCH_ROOT = Path("artifacts") / "strategy_projects" / PROJECT_ID / "research"
DEFAULT_MANIFEST = RESEARCH_ROOT / "top1000_data_backfill_20260417" / "top1000_manifest.csv"
DEFAULT_ASSET_AUDIT = (
    RESEARCH_ROOT / "top1000_data_backfill_20260417" / "candidate_asset_filter_audit.csv"
)
DEFAULT_PIT_ROOT = RESEARCH_ROOT / "phase0_provisional_pit_universe_feasibility_20260417"
DEFAULT_ASSET_QA_ROOT = RESEARCH_ROOT / "phase0_asset_class_qa_20260417"
DEFAULT_VENDOR_ROOT = RESEARCH_ROOT / "phase0_vendor_bakeoff_20260417"
DEFAULT_DAILY_GLOBS = (
    "data/silver/daily_bar/phase0_top1000_yahoo_gap_20130805_20151231_chunk*.jsonl",
    "data/silver/daily_bar/phase0_top1000_sip_raw_20160104_20260416_chunk*.jsonl",
)
DEFAULT_TOP_BUCKETS = (500, 1000, 1500, 2000, 3000)
DEFAULT_ADV_THRESHOLDS = (10_000_000, 20_000_000, 30_000_000, 50_000_000)

ETF_PATTERN = re.compile(
    r"\b(ETF|ETN|FUND|FUNDS|INDEX FUND|EXCHANGE[- ]TRADED|SPDR|ISHARES|VANGUARD|"
    r"PROSHARES|DIREXION|GLOBAL X|WISDOMTREE|FIRST TRUST|VANECK)\b",
    re.IGNORECASE,
)
NON_COMMON_PATTERN = re.compile(
    r"\b(WARRANTS?|RIGHTS?|UNITS?|PREFERRED|PREF|DEBENTURES?|NOTES?|BONDS?|"
    r"ACQUISITION|BLANK CHECK|SPAC)\b",
    re.IGNORECASE,
)
ADR_PATTERN = re.compile(
    r"\b(ADR|ADS|AMERICAN DEPOSITARY|DEPOSITARY SHARES?|PLC|P\.L\.C\.|N\.V\.|"
    r"S\.A\.|SE|LTD|LIMITED|COMPANHIA|BANCO|HOLDING[S]? PLC)\b",
    re.IGNORECASE,
)
TRUST_PATTERN = re.compile(r"\b(REIT|REAL ESTATE INVESTMENT TRUST|ROYALTY TRUST)\b", re.IGNORECASE)


def build_pit_universe_feasibility(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    daily_globs: Sequence[str | Path] = DEFAULT_DAILY_GLOBS,
    output_root: str | Path = DEFAULT_PIT_ROOT,
    validation_start: str = "2013-08-05",
    validation_end: str = "2019-12-31",
    lookback_sessions: int = 20,
    min_observations: int = 15,
    price_floor: float = 10.0,
    beta_warmup_observations: int = 126,
    beta_full_observations: int = 252,
    top_buckets: Sequence[int] = DEFAULT_TOP_BUCKETS,
    adv_thresholds: Sequence[int] = DEFAULT_ADV_THRESHOLDS,
) -> dict[str, Any]:
    """Build validation-only lagged-liquidity universe artifacts."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    symbols = set(manifest["symbol"].astype(str))
    files = _expand_globs(daily_globs)
    if not files:
        raise FileNotFoundError(f"No daily files matched: {list(map(str, daily_globs))}")

    rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for path in files:
        source = _source_segment(path)
        source_counts.setdefault(source, 0)
        for row in _iter_jsonl(path):
            session_date = row.get("session_date")
            symbol = row.get("symbol")
            if not isinstance(session_date, str) or not isinstance(symbol, str):
                continue
            if session_date < validation_start or session_date > validation_end or symbol not in symbols:
                continue
            rows.append(
                {
                    "session_date": session_date,
                    "symbol": symbol,
                    "close": _to_float(row.get("close")),
                    "volume": _to_float(row.get("volume")),
                    "dollar_volume": _to_float(row.get("dollar_volume")),
                    "source_segment": source,
                }
            )
            source_counts[source] += 1
    if not rows:
        raise ValueError("No rows loaded for the requested validation window.")

    frame = pd.DataFrame(rows)
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    frame = frame.sort_values(["symbol", "session_date"]).drop_duplicates(
        ["symbol", "session_date"], keep="last"
    )
    frame = frame.reset_index(drop=True)

    eligible = _lagged_liquidity(
        frame,
        lookback_sessions=lookback_sessions,
        min_observations=min_observations,
        price_floor=price_floor,
        beta_warmup_observations=beta_warmup_observations,
        beta_full_observations=beta_full_observations,
    )
    for bucket in top_buckets:
        eligible[f"in_top{bucket}"] = eligible["liquidity_rank"] <= bucket
    for threshold in adv_thresholds:
        eligible[_adv_label(threshold)] = eligible["trailing_median_dollar_volume_20"] >= threshold

    membership_path = output_dir / "provisional_current_top1000_lagged_liquidity_membership_validation.csv.gz"
    eligible[_membership_columns(top_buckets, adv_thresholds)].to_csv(
        membership_path, index=False, compression="gzip"
    )
    daily = _daily_counts(frame, eligible, top_buckets=top_buckets, adv_thresholds=adv_thresholds)
    daily_path = output_dir / "candidate_universe_daily_counts_validation.csv"
    daily.to_csv(daily_path, index=False)
    summary = _candidate_summary(daily, top_buckets=top_buckets, adv_thresholds=adv_thresholds)
    summary_path = output_dir / "candidate_universe_feasibility_summary_validation.csv"
    summary.to_csv(summary_path, index=False)
    source_path = output_dir / "source_segment_daily_counts_validation.csv"
    _source_by_date(eligible).to_csv(source_path, index=False)

    rollup = {
        "created_at_utc": _utc_now(),
        "validation_start": validation_start,
        "validation_end": validation_end,
        "lookback_sessions": lookback_sessions,
        "min_liquidity_observations": min_observations,
        "price_floor": price_floor,
        "input_daily_files": [path.as_posix() for path in files],
        "manifest_path": Path(manifest_path).as_posix(),
        "artifact_dir": output_dir.as_posix(),
        "source_rows_loaded": source_counts,
        "validation_rows_loaded": int(len(frame)),
        "symbols_loaded": int(frame["symbol"].nunique()),
        "validation_sessions": int(len(daily)),
        "eligible_membership_rows": int(len(eligible)),
        "membership_artifact": membership_path.as_posix(),
        "daily_counts_artifact": daily_path.as_posix(),
        "summary_artifact": summary_path.as_posix(),
        "source_segment_daily_counts_artifact": source_path.as_posix(),
        "key_findings": _pit_key_findings(daily, summary),
        "limitations": [
            "Provisional and limited to the current top1000 bootstrap symbol set.",
            "Top1500/top2000/top3000 cannot be evaluated from a top1000-only backfill.",
            "No alpha signal, portfolio return, or test-window performance was computed.",
        ],
    }
    (output_dir / "phase0_provisional_pit_universe_feasibility_rollup.json").write_text(
        json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return rollup


def run_asset_class_qa(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    asset_audit_path: str | Path = DEFAULT_ASSET_AUDIT,
    output_root: str | Path = DEFAULT_ASSET_QA_ROOT,
) -> dict[str, Any]:
    """Generate current top1000 common-stock cleanliness QA artifacts."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    assets = pd.read_csv(asset_audit_path)
    frame = manifest.merge(
        assets.drop_duplicates("symbol"),
        on="symbol",
        how="left",
        suffixes=("_manifest", "_asset"),
    )
    rows = []
    for _, row in frame.iterrows():
        symbol = str(row["symbol"])
        name = _row_str(row, "name_manifest") or _row_str(row, "name_asset")
        classification, flags = classify_asset(symbol=symbol, name=name, row=row)
        rows.append(
            {
                "symbol": symbol,
                "name": name,
                "exchange": _row_str(row, "exchange_asset") or _row_str(row, "exchange_manifest"),
                "liquidity_rank": row.get("liquidity_rank"),
                "classification": classification,
                "flags": "|".join(flags),
                "review_required": classification != "common_stock_candidate" or bool(flags),
                "tradable": row.get("tradable"),
                "marginable": row.get("marginable"),
                "shortable": row.get("shortable"),
                "easy_to_borrow": row.get("easy_to_borrow"),
                "asset_filter_reason": row.get("phase0_filter_reason"),
            }
        )
    qa = pd.DataFrame(rows)
    by_symbol_path = output_dir / "top1000_asset_class_qa_by_symbol.csv"
    qa.to_csv(by_symbol_path, index=False)
    summary = (
        qa.groupby("classification")
        .agg(symbols=("symbol", "nunique"), review_required=("review_required", "sum"))
        .reset_index()
        .sort_values(["review_required", "symbols"], ascending=[False, False])
    )
    summary_path = output_dir / "top1000_asset_class_qa_summary.csv"
    summary.to_csv(summary_path, index=False)
    flag_counts = _flag_counts(qa)
    flag_counts_path = output_dir / "top1000_asset_class_flag_counts.csv"
    flag_counts.to_csv(flag_counts_path, index=False)
    review_path = output_dir / "top1000_asset_class_review_queue.csv"
    qa[qa["review_required"]].sort_values("liquidity_rank").to_csv(review_path, index=False)

    rollup = {
        "created_at_utc": _utc_now(),
        "manifest_path": Path(manifest_path).as_posix(),
        "asset_audit_path": Path(asset_audit_path).as_posix(),
        "artifact_dir": output_dir.as_posix(),
        "symbols_checked": int(qa["symbol"].nunique()),
        "common_stock_candidates": int((qa["classification"] == "common_stock_candidate").sum()),
        "review_required_symbols": int(qa["review_required"].sum()),
        "blocked_non_common_like_symbols": int((qa["classification"] == "blocked_non_common_like").sum()),
        "foreign_or_adr_review_symbols": int((qa["classification"] == "review_foreign_or_adr_like").sum()),
        "trust_or_reit_review_symbols": int((qa["classification"] == "review_trust_or_reit_like").sum()),
        "by_symbol_artifact": by_symbol_path.as_posix(),
        "summary_artifact": summary_path.as_posix(),
        "flag_counts_artifact": flag_counts_path.as_posix(),
        "review_queue_artifact": review_path.as_posix(),
        "limitations": [
            "Name/metadata QA is not a legal security master.",
            "Review flags do not automatically exclude a symbol.",
            "ADR and foreign issuer policy must be chosen before final universe freeze.",
        ],
    }
    (output_dir / "top1000_asset_class_qa_rollup.json").write_text(
        json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return rollup


def classify_asset(*, symbol: str, name: str, row: pd.Series | None = None) -> tuple[str, list[str]]:
    flags: list[str] = []
    if ETF_PATTERN.search(name):
        flags.append("etf_or_fund_name_pattern")
    if NON_COMMON_PATTERN.search(name):
        flags.append("non_common_security_name_pattern")
    if ADR_PATTERN.search(name):
        flags.append("adr_or_foreign_issuer_name_pattern")
    if TRUST_PATTERN.search(name):
        flags.append("trust_or_reit_name_pattern")
    if "." in symbol:
        flags.append("class_share_symbol")
    if row is not None:
        if _row_bool(row, "tradable") is False:
            flags.append("not_tradable_current_metadata")
        if _row_bool(row, "shortable") is False:
            flags.append("not_shortable_current_metadata")
        if _row_bool(row, "easy_to_borrow") is False:
            flags.append("not_easy_to_borrow_current_metadata")

    if "etf_or_fund_name_pattern" in flags or "non_common_security_name_pattern" in flags:
        classification = "blocked_non_common_like"
    elif "adr_or_foreign_issuer_name_pattern" in flags:
        classification = "review_foreign_or_adr_like"
    elif "trust_or_reit_name_pattern" in flags:
        classification = "review_trust_or_reit_like"
    else:
        classification = "common_stock_candidate"
    return classification, flags


def write_vendor_bakeoff_artifacts(
    *,
    output_root: str | Path = DEFAULT_VENDOR_ROOT,
) -> dict[str, Any]:
    """Write the Phase 0 vendor bake-off matrix and acceptance checks."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    vendors = _vendor_rows()
    checks = _vendor_checks()
    vendors_path = output_dir / "vendor_bakeoff_candidates.csv"
    checks_path = output_dir / "vendor_bakeoff_acceptance_checks.csv"
    _write_csv(vendors_path, vendors)
    _write_csv(checks_path, checks)
    memo_path = output_dir / "vendor_bakeoff_plan.md"
    memo_path.write_text(_vendor_bakeoff_markdown(vendors, checks), encoding="utf-8")
    rollup = {
        "created_at_utc": _utc_now(),
        "artifact_dir": output_dir.as_posix(),
        "candidate_vendors": len(vendors),
        "acceptance_checks": len(checks),
        "recommended_order": ["Norgate Data", "Sharadar / Nasdaq Data Link", "CRSP / WRDS", "Polygon.io"],
        "vendor_candidates_artifact": vendors_path.as_posix(),
        "acceptance_checks_artifact": checks_path.as_posix(),
        "memo_artifact": memo_path.as_posix(),
        "decision_rule": (
            "Select the first vendor that passes survivorship-bias, adjustment, "
            "common-stock classification, 2013-2015 coverage, and repeatable-ingestion checks."
        ),
    }
    (output_dir / "vendor_bakeoff_rollup.json").write_text(
        json.dumps(rollup, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return rollup


def _lagged_liquidity(
    frame: pd.DataFrame,
    *,
    lookback_sessions: int,
    min_observations: int,
    price_floor: float,
    beta_warmup_observations: int,
    beta_full_observations: int,
) -> pd.DataFrame:
    grouped = frame.groupby("symbol", group_keys=False)
    frame["lagged_close"] = grouped["close"].shift(1)
    frame["prior_bar_count"] = grouped.cumcount()
    frame["trailing_median_dollar_volume_20"] = grouped["dollar_volume"].transform(
        lambda series: series.shift(1).rolling(lookback_sessions, min_periods=min_observations).median()
    )
    frame["trailing_obs_20"] = grouped["dollar_volume"].transform(
        lambda series: series.shift(1).rolling(lookback_sessions, min_periods=1).count()
    )
    frame["beta_warmup_proxy_ready"] = frame["prior_bar_count"] >= beta_warmup_observations
    frame["beta_full_lookback_proxy_ready"] = frame["prior_bar_count"] >= beta_full_observations
    frame["liquidity_eligible"] = (
        (frame["trailing_obs_20"] >= min_observations)
        & frame["trailing_median_dollar_volume_20"].notna()
        & (frame["trailing_median_dollar_volume_20"] > 0)
        & frame["lagged_close"].notna()
        & (frame["lagged_close"] >= price_floor)
    )
    eligible = frame[frame["liquidity_eligible"]].copy()
    eligible["liquidity_rank"] = eligible.groupby("session_date")[
        "trailing_median_dollar_volume_20"
    ].rank(method="first", ascending=False)
    eligible["liquidity_rank"] = eligible["liquidity_rank"].astype(int)
    return eligible


def _daily_counts(
    frame: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    top_buckets: Sequence[int],
    adv_thresholds: Sequence[int],
) -> pd.DataFrame:
    dates = pd.DataFrame({"session_date": sorted(frame["session_date"].unique())})
    daily = frame.groupby("session_date").agg(
        symbols_with_bar=("symbol", "nunique"),
        raw_rows=("symbol", "size"),
    ).reset_index()
    base = eligible.groupby("session_date").agg(
        liquidity_eligible_count=("symbol", "nunique"),
        beta_warmup_proxy_count=("beta_warmup_proxy_ready", "sum"),
        beta_full_lookback_proxy_count=("beta_full_lookback_proxy_ready", "sum"),
    ).reset_index()
    daily = dates.merge(daily, on="session_date", how="left").merge(base, on="session_date", how="left")
    for column in daily.columns:
        if column != "session_date":
            daily[column] = daily[column].fillna(0).astype(int)
    for bucket in top_buckets:
        daily = _merge_counts(daily, eligible[eligible[f"in_top{bucket}"]], f"top{bucket}")
    for threshold in adv_thresholds:
        label = _adv_label(threshold)
        daily = _merge_counts(daily, eligible[eligible[label]], label)
    daily["session_date"] = pd.to_datetime(daily["session_date"]).dt.date.astype(str)
    return daily


def _merge_counts(daily: pd.DataFrame, subset: pd.DataFrame, prefix: str) -> pd.DataFrame:
    counts = subset.groupby("session_date").agg(
        **{
            f"{prefix}_count": ("symbol", "nunique"),
            f"{prefix}_beta_warmup_proxy_count": ("beta_warmup_proxy_ready", "sum"),
            f"{prefix}_beta_full_lookback_proxy_count": ("beta_full_lookback_proxy_ready", "sum"),
        }
    ).reset_index()
    daily = daily.merge(counts, on="session_date", how="left")
    for suffix in ("count", "beta_warmup_proxy_count", "beta_full_lookback_proxy_count"):
        column = f"{prefix}_{suffix}"
        daily[column] = daily[column].fillna(0).astype(int)
    return daily


def _candidate_summary(
    daily: pd.DataFrame,
    *,
    top_buckets: Sequence[int],
    adv_thresholds: Sequence[int],
) -> pd.DataFrame:
    rows = [
        _summarize(
            daily,
            "liquidity_eligible_all_current_top1000_scope",
            "liquidity_eligible_count",
            "beta_warmup_proxy_count",
            "beta_full_lookback_proxy_count",
            "provisional_current_top1000_scope",
        )
    ]
    for bucket in top_buckets:
        status = (
            "supported_within_current_top1000_scope"
            if bucket <= 1000
            else "blocked_current_backfill_has_only_top1000_symbols"
        )
        rows.append(
            _summarize(
                daily,
                f"top{bucket}_by_lagged_trailing_median_dollar_volume_20d",
                f"top{bucket}_count",
                f"top{bucket}_beta_warmup_proxy_count",
                f"top{bucket}_beta_full_lookback_proxy_count",
                status,
            )
        )
    for threshold in adv_thresholds:
        label = _adv_label(threshold)
        rows.append(
            _summarize(
                daily,
                f"{label}_lagged_trailing_median_dollar_volume_20d",
                f"{label}_count",
                f"{label}_beta_warmup_proxy_count",
                f"{label}_beta_full_lookback_proxy_count",
                "supported_only_within_current_top1000_scope",
            )
        )
    return pd.DataFrame(rows)


def _summarize(
    daily: pd.DataFrame,
    name: str,
    count_col: str,
    beta_col: str,
    full_beta_col: str,
    status: str,
) -> dict[str, Any]:
    counts = daily[count_col]
    beta = daily[beta_col]
    full_beta = daily[full_beta_col]
    nonzero = daily.loc[counts > 0, "session_date"]
    beta40 = daily.loc[beta >= 40, "session_date"]
    beta60 = daily.loc[beta >= 60, "session_date"]
    fullbeta40 = daily.loc[full_beta >= 40, "session_date"]
    return {
        "candidate": name,
        "data_status": status,
        "sessions": int(len(daily)),
        "first_nonzero_date": nonzero.iloc[0] if len(nonzero) else None,
        "median_members": float(counts.median()),
        "p10_members": float(counts.quantile(0.10)),
        "min_members": int(counts.min()),
        "max_members": int(counts.max()),
        "median_beta_warmup_proxy_members": float(beta.median()),
        "first_beta_warmup_proxy_20_20_date": beta40.iloc[0] if len(beta40) else None,
        "first_beta_warmup_proxy_30_30_date": beta60.iloc[0] if len(beta60) else None,
        "share_sessions_beta_warmup_proxy_20_20": float((beta >= 40).mean()),
        "share_sessions_beta_warmup_proxy_30_30": float((beta >= 60).mean()),
        "median_full_beta_lookback_proxy_members": float(full_beta.median()),
        "first_full_beta_lookback_proxy_20_20_date": fullbeta40.iloc[0] if len(fullbeta40) else None,
        "share_sessions_full_beta_lookback_proxy_20_20": float((full_beta >= 40).mean()),
    }


def _pit_key_findings(daily: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    findings: dict[str, Any] = {
        "median_symbols_with_bar": float(daily["symbols_with_bar"].median()),
        "min_symbols_with_bar": int(daily["symbols_with_bar"].min()),
        "median_liquidity_eligible_count": float(daily["liquidity_eligible_count"].median()),
        "min_liquidity_eligible_count": int(daily["liquidity_eligible_count"].min()),
        "median_top500_count": float(daily["top500_count"].median()) if "top500_count" in daily else None,
        "median_top1000_count": float(daily["top1000_count"].median()) if "top1000_count" in daily else None,
    }
    for bucket in (500, 1000):
        row = summary[summary["candidate"].str.startswith(f"top{bucket}_")]
        if not row.empty:
            findings[f"first_top{bucket}_beta_warmup_proxy_30_30_date"] = row[
                "first_beta_warmup_proxy_30_30_date"
            ].iloc[0]
    return findings


def _source_by_date(eligible: pd.DataFrame) -> pd.DataFrame:
    frame = eligible.groupby(["session_date", "source_segment"]).size().reset_index(name="eligible_rows")
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date.astype(str)
    return frame


def _membership_columns(top_buckets: Sequence[int], adv_thresholds: Sequence[int]) -> list[str]:
    columns = [
        "session_date",
        "symbol",
        "source_segment",
        "lagged_close",
        "trailing_obs_20",
        "trailing_median_dollar_volume_20",
        "liquidity_rank",
        "beta_warmup_proxy_ready",
        "beta_full_lookback_proxy_ready",
    ]
    columns.extend(f"in_top{bucket}" for bucket in top_buckets)
    columns.extend(_adv_label(threshold) for threshold in adv_thresholds)
    return columns


def _flag_counts(qa: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for raw_flags in qa["flags"].fillna(""):
        for flag in str(raw_flags).split("|"):
            if flag:
                counts[flag] = counts.get(flag, 0) + 1
    return pd.DataFrame([{"flag": flag, "symbols": count} for flag, count in sorted(counts.items())])


def _vendor_rows() -> list[dict[str, str]]:
    return [
        {
            "vendor": "Norgate Data",
            "priority": "1",
            "role": "preferred independent-research source",
            "expected_strength": "survivorship-bias-free US equities, delisted stocks, historical membership/context",
            "main_risk": "Windows/local tooling integration and subscription workflow",
            "trial_task": "Export 2013-08-05, 2014-01-02, 2015-12-31 bars plus delisted coverage sample.",
            "source_url": "https://norgatedata.com/data-content-tables.php",
        },
        {
            "vendor": "Sharadar / Nasdaq Data Link",
            "priority": "2",
            "role": "preferred API-style source",
            "expected_strength": "active plus delisted equity prices, repeatable ingestion, corporate actions",
            "main_risk": "subscription access and point-in-time membership still built by us",
            "trial_task": "Pull price/ticker/action tables for 2013-2015 coverage and corporate-action QA.",
            "source_url": "https://docs.data.nasdaq.com/",
        },
        {
            "vendor": "CRSP / WRDS",
            "priority": "3",
            "role": "gold standard if accessible",
            "expected_strength": "research-grade returns, distributions, delistings, identifiers",
            "main_risk": "institutional access requirement and ingestion complexity",
            "trial_task": "Confirm access and export daily stock plus delisting-return sample.",
            "source_url": "https://wrds-www.wharton.upenn.edu/",
        },
        {
            "vendor": "Polygon.io",
            "priority": "4",
            "role": "API fallback",
            "expected_strength": "stock aggregates, tickers, splits/dividends endpoints, automation-friendly API",
            "main_risk": "total-return adjustment must be rebuilt and audited",
            "trial_task": "Pull grouped daily bars and corporate actions for a 2013-2015 sample.",
            "source_url": "https://polygon.io/docs/rest/stocks/aggregates/daily-market-summary",
        },
        {
            "vendor": "QuantQuote / HistoricalData.net / EODHD",
            "priority": "5",
            "role": "low-cost backup",
            "expected_strength": "bulk historical coverage may fill practical gaps quickly",
            "main_risk": "quality and metadata must be independently audited",
            "trial_task": "Run split/dividend/delisting and ETF-filter QA before research use.",
            "source_url": "https://quantquote.com/stock-data",
        },
        {
            "vendor": "Yahoo / Stooq",
            "priority": "6",
            "role": "sanity-check or provisional gap fill only",
            "expected_strength": "quick access for plumbing and price-scale reconciliation",
            "main_risk": "not a final survivorship-bias-free institutional source",
            "trial_task": "Keep as cross-check and temporary data-engineering bridge only.",
            "source_url": "https://finance.yahoo.com/",
        },
    ]


def _vendor_checks() -> list[dict[str, str]]:
    checks = [
        ("coverage_2013_2015", "Daily bars exist on 2013-08-05, 2014-01-02, and 2015-12-31."),
        ("active_and_delisted", "Active and delisted names are present, or survivorship bias is quantified."),
        ("corporate_actions", "Split and dividend adjustment factors reconcile on sampled events."),
        ("common_stock_filter", "Common stocks can be separated from ETFs, units, warrants, preferreds, and SPACs."),
        ("lagged_membership", "Lagged dollar-volume topN membership can be generated without future leakage."),
        ("alpaca_overlap", "2016+ overlap sample reconciles against current Alpaca SIP backfill."),
        ("automation", "Ingestion can be repeated from a documented command or export procedure."),
        ("cost_fit", "Subscription cost and operational burden are acceptable for this research line."),
    ]
    return [
        {"check_id": check_id, "acceptance_check": text, "required": "yes"}
        for check_id, text in checks
    ]


def _vendor_bakeoff_markdown(vendors: list[dict[str, str]], checks: list[dict[str, str]]) -> str:
    lines = [
        "# Pure Alpha Phase 0 Vendor Bake-Off Plan",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Decision Rule",
        "",
        "Select the first vendor that passes survivorship-bias, adjustment, common-stock "
        "classification, 2013-2015 coverage, and repeatable-ingestion checks.",
        "",
        "## Candidate Order",
        "",
        "| Priority | Vendor | Role | Main Risk |",
        "|---:|---|---|---|",
    ]
    for vendor in vendors:
        lines.append(
            f"| {vendor['priority']} | {vendor['vendor']} | {vendor['role']} | {vendor['main_risk']} |"
        )
    lines.extend(["", "## Acceptance Checks", "", "| Check | Required |", "|---|---|"])
    for check in checks:
        lines.append(f"| {check['acceptance_check']} | {check['required']} |")
    lines.extend(
        [
            "",
            "Do not use test-window strategy performance during vendor selection.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _adv_label(threshold: int) -> str:
    return f"adv_ge_{int(threshold / 1_000_000)}m"


def _expand_globs(patterns: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(str(pattern)))
    return sorted(dict.fromkeys(paths))


def _source_segment(path: Path) -> str:
    name = path.name.lower()
    if "yahoo_gap" in name:
        return "yahoo_gapfill"
    if "sip_raw" in name or "alpaca" in name:
        return "alpaca_sip"
    return "unknown"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_str(row: pd.Series, column: str) -> str:
    value = row.get(column)
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _row_bool(row: pd.Series, column: str) -> bool | None:
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pure-alpha Phase 0 utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pit = subparsers.add_parser("pit-universe", help="Build lagged-liquidity universe artifacts.")
    pit.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    pit.add_argument("--daily-glob", action="append", dest="daily_globs")
    pit.add_argument("--output-root", default=str(DEFAULT_PIT_ROOT))
    pit.add_argument("--validation-start", default="2013-08-05")
    pit.add_argument("--validation-end", default="2019-12-31")
    pit.add_argument("--lookback-sessions", type=int, default=20)
    pit.add_argument("--min-observations", type=int, default=15)
    pit.add_argument("--price-floor", type=float, default=10.0)

    qa = subparsers.add_parser("asset-qa", help="Run current top1000 asset-class QA.")
    qa.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    qa.add_argument("--asset-audit-path", default=str(DEFAULT_ASSET_AUDIT))
    qa.add_argument("--output-root", default=str(DEFAULT_ASSET_QA_ROOT))

    vendor = subparsers.add_parser("vendor-bakeoff", help="Write vendor bake-off artifacts.")
    vendor.add_argument("--output-root", default=str(DEFAULT_VENDOR_ROOT))

    all_parser = subparsers.add_parser("all", help="Run PIT universe, asset QA, and vendor bake-off.")
    all_parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    all_parser.add_argument("--asset-audit-path", default=str(DEFAULT_ASSET_AUDIT))
    all_parser.add_argument("--daily-glob", action="append", dest="daily_globs")
    all_parser.add_argument("--pit-output-root", default=str(DEFAULT_PIT_ROOT))
    all_parser.add_argument("--asset-qa-output-root", default=str(DEFAULT_ASSET_QA_ROOT))
    all_parser.add_argument("--vendor-output-root", default=str(DEFAULT_VENDOR_ROOT))

    args = parser.parse_args(argv)
    if args.command == "pit-universe":
        result = build_pit_universe_feasibility(
            manifest_path=args.manifest_path,
            daily_globs=tuple(args.daily_globs or DEFAULT_DAILY_GLOBS),
            output_root=args.output_root,
            validation_start=args.validation_start,
            validation_end=args.validation_end,
            lookback_sessions=args.lookback_sessions,
            min_observations=args.min_observations,
            price_floor=args.price_floor,
        )
    elif args.command == "asset-qa":
        result = run_asset_class_qa(
            manifest_path=args.manifest_path,
            asset_audit_path=args.asset_audit_path,
            output_root=args.output_root,
        )
    elif args.command == "vendor-bakeoff":
        result = write_vendor_bakeoff_artifacts(output_root=args.output_root)
    else:
        result = {
            "pit_universe": build_pit_universe_feasibility(
                manifest_path=args.manifest_path,
                daily_globs=tuple(args.daily_globs or DEFAULT_DAILY_GLOBS),
                output_root=args.pit_output_root,
            ),
            "asset_qa": run_asset_class_qa(
                manifest_path=args.manifest_path,
                asset_audit_path=args.asset_audit_path,
                output_root=args.asset_qa_output_root,
            ),
            "vendor_bakeoff": write_vendor_bakeoff_artifacts(output_root=args.vendor_output_root),
        }

    print(json.dumps({"ok": True, "result": result}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
