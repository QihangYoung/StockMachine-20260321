from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from stockmachine.data.loaders.silver import load_silver_table
from stockmachine.ingestion.storage import StorageLayout

_EMPTY_METADATA_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "symbol",
    "security_id",
    "company_name",
    "exchange_mic",
    "currency",
    "security_type",
    "asset_class",
    "is_active",
    "list_date",
    "delist_date",
    "sector",
    "industry",
    "country_of_listing",
    "primary_share_class",
    "source_name",
    "load_time_utc",
    "source_version",
)

DEFAULT_RESEARCH_UNIVERSE_NAME = "us_equities_research_v1"

_EMPTY_UNIVERSE_MEMBERSHIP_COLUMNS: tuple[str, ...] = (
    "session_date",
    "universe_name",
    "symbol",
    "is_member",
    "membership_source",
    "entry_date",
    "exit_date",
    "source_name",
    "load_time_utc",
    "source_version",
)

_EMPTY_INDUSTRY_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "symbol",
    "industry_system",
    "sector_name",
    "industry_group_name",
    "industry_name",
    "subindustry_name",
    "source_name",
    "load_time_utc",
    "source_version",
)

_EMPTY_INDUSTRY_MAP_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "symbol",
    "industry_system",
    "sector",
    "industry_group",
    "industry",
    "subindustry",
    "source_name",
    "load_time_utc",
    "source_version",
)


@dataclass(slots=True, frozen=True)
class PointInTimeUniverse:
    """One point-in-time universe snapshot for research and backtests."""

    session_date: pd.Timestamp
    universe_membership_snapshot_date: pd.Timestamp | None
    symbol_master_snapshot_date: pd.Timestamp | None
    industry_snapshot_date: pd.Timestamp | None
    members: tuple[str, ...]
    metadata: pd.DataFrame
    universe_membership: pd.DataFrame
    industry_membership: pd.DataFrame
    industry_map: pd.DataFrame
    membership_source: str
    conservative: bool
    notes: tuple[str, ...] = ()

    @property
    def member_count(self) -> int:
        return len(self.members)


def load_point_in_time_universe(
    session_date: str | pd.Timestamp,
    *,
    layout: StorageLayout | None = None,
    active_only: bool = True,
    require_snapshot: bool = False,
    universe_name: str | None = DEFAULT_RESEARCH_UNIVERSE_NAME,
) -> PointInTimeUniverse:
    """Load the latest silver snapshots visible on or before one session date.

    The helper is conservative by default:
    - it never uses a snapshot dated after ``session_date``;
    - if no eligible snapshot exists, it returns an empty universe;
    - it only raises when ``require_snapshot`` is set.
    """

    session_ts = _coerce_session_date(session_date)
    storage = layout or StorageLayout()

    universe_membership_raw = load_silver_table("universe_membership", layout=storage)
    symbol_master_raw = load_silver_table("symbol_master", layout=storage)
    industry_raw = load_silver_table("industry_membership", layout=storage)

    return resolve_point_in_time_universe(
        session_ts,
        universe_membership_frame=universe_membership_raw,
        symbol_master_frame=symbol_master_raw,
        industry_membership_frame=industry_raw,
        active_only=active_only,
        require_snapshot=require_snapshot,
        universe_name=universe_name,
    )


def resolve_point_in_time_universe(
    session_date: str | pd.Timestamp,
    *,
    universe_membership_frame: pd.DataFrame | None = None,
    symbol_master_frame: pd.DataFrame,
    industry_membership_frame: pd.DataFrame,
    active_only: bool = True,
    require_snapshot: bool = False,
    universe_name: str | None = None,
) -> PointInTimeUniverse:
    """Resolve one point-in-time universe from already-loaded silver frames."""

    session_ts = _coerce_session_date(session_date)
    universe_membership, universe_snapshot_date = _select_latest_snapshot(
        universe_membership_frame if universe_membership_frame is not None else pd.DataFrame(),
        date_column="session_date",
        session_date=session_ts,
    )
    explicit_membership = _normalize_universe_membership(
        universe_membership,
        session_date=session_ts,
        universe_name=universe_name,
    )
    symbol_master, symbol_snapshot_date = _select_latest_snapshot(
        symbol_master_frame,
        date_column="as_of_date",
        session_date=session_ts,
    )
    industry_membership, industry_snapshot_date = _select_latest_snapshot(
        industry_membership_frame,
        date_column="as_of_date",
        session_date=session_ts,
    )

    notes: list[str] = []
    metadata = _empty_metadata_frame()
    membership_frame = _empty_universe_membership_frame()
    industry_map = _empty_industry_map_frame()
    members: tuple[str, ...] = ()
    membership_source = "symbol_master_fallback"
    conservative = False

    if symbol_master.empty:
        notes.append(
            f"missing point-in-time symbol_master snapshot on or before {session_ts.date().isoformat()}"
        )
        conservative = True
    else:
        metadata = _normalize_symbol_master(
            symbol_master,
            session_date=session_ts,
            active_only=active_only,
        )
        members = tuple(metadata["symbol"].dropna().astype(str).tolist())

    if industry_membership.empty:
        notes.append(
            f"missing point-in-time industry_membership snapshot on or before {session_ts.date().isoformat()}"
        )
        conservative = True
    else:
        industry_membership = _normalize_industry_membership(industry_membership)
        industry_map = _collapse_industry_membership(industry_membership, allowed_symbols=members)

    if not explicit_membership.empty:
        membership_frame = explicit_membership
        membership_source = "explicit_universe_membership"
        allowed_symbols = tuple(
            membership_frame.loc[membership_frame["is_member"].fillna(False).astype(bool), "symbol"]
            .dropna()
            .astype(str)
            .tolist()
        )
        if metadata.empty:
            metadata = _placeholder_metadata_from_membership(
                membership_frame,
                session_date=session_ts,
            )
        else:
            metadata = metadata.loc[metadata["symbol"].astype(str).isin(set(allowed_symbols))].copy()
        if not industry_membership.empty:
            industry_membership = industry_membership.loc[
                industry_membership["symbol"].astype(str).isin(set(allowed_symbols))
            ].copy()
        industry_map = _collapse_industry_membership(industry_membership, allowed_symbols=allowed_symbols)
        members = tuple(metadata["symbol"].dropna().astype(str).tolist())
        missing_metadata_symbols = sorted(set(allowed_symbols) - set(members))
        if missing_metadata_symbols:
            notes.append(
                "explicit universe members missing metadata fields were retained with Unknown fills: "
                + ", ".join(missing_metadata_symbols)
            )
            conservative = True
    elif universe_membership_frame is not None and not universe_membership_frame.empty:
        notes.append(
            f"missing explicit universe_membership snapshot on or before {session_ts.date().isoformat()}, "
            "falling back to symbol_master active snapshot"
        )
        conservative = True

    if require_snapshot and (symbol_master.empty or industry_membership.empty):
        raise FileNotFoundError(
            "No eligible point-in-time universe snapshot was found for "
            f"{session_ts.date().isoformat()}."
        )

    return PointInTimeUniverse(
        session_date=session_ts,
        universe_membership_snapshot_date=universe_snapshot_date,
        symbol_master_snapshot_date=symbol_snapshot_date,
        industry_snapshot_date=industry_snapshot_date,
        members=members,
        metadata=metadata,
        universe_membership=membership_frame,
        industry_membership=industry_membership,
        industry_map=industry_map,
        membership_source=membership_source,
        conservative=conservative,
        notes=tuple(notes),
    )


def build_point_in_time_metadata_history(
    session_dates: pd.Series | pd.Index | tuple[pd.Timestamp, ...] | list[pd.Timestamp],
    *,
    universe_membership_frame: pd.DataFrame | None = None,
    symbol_master_frame: pd.DataFrame,
    industry_membership_frame: pd.DataFrame,
    active_only: bool = True,
    require_snapshot: bool = False,
    universe_name: str | None = DEFAULT_RESEARCH_UNIVERSE_NAME,
) -> pd.DataFrame:
    """Build one date-aware metadata panel for research-frame joins."""

    resolved_dates = (
        pd.Series(list(session_dates), dtype="datetime64[ns]")
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not resolved_dates:
        return _empty_research_metadata_frame()

    fast_path = _build_exact_snapshot_metadata_history(
        resolved_dates,
        universe_membership_frame=universe_membership_frame,
        symbol_master_frame=symbol_master_frame,
        industry_membership_frame=industry_membership_frame,
        active_only=active_only,
        universe_name=universe_name,
    )
    if fast_path is not None:
        return fast_path

    frames: list[pd.DataFrame] = []
    for session_date in resolved_dates:
        snapshot = resolve_point_in_time_universe(
            session_date,
            universe_membership_frame=universe_membership_frame,
            symbol_master_frame=symbol_master_frame,
            industry_membership_frame=industry_membership_frame,
            active_only=active_only,
            require_snapshot=require_snapshot,
            universe_name=universe_name,
        )
        if snapshot.metadata.empty:
            continue
        frames.append(_standardize_research_metadata(snapshot, session_date=pd.Timestamp(session_date)))

    if not frames:
        return _empty_research_metadata_frame()
    return pd.concat(frames, ignore_index=True)


def load_universe_members(
    session_date: str | pd.Timestamp,
    *,
    layout: StorageLayout | None = None,
    active_only: bool = True,
    require_snapshot: bool = False,
) -> tuple[str, ...]:
    """Convenience wrapper that returns only the member symbols."""

    snapshot = load_point_in_time_universe(
        session_date,
        layout=layout,
        active_only=active_only,
        require_snapshot=require_snapshot,
    )
    return snapshot.members


def load_universe_industry_map(
    session_date: str | pd.Timestamp,
    *,
    layout: StorageLayout | None = None,
    active_only: bool = True,
    require_snapshot: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper that returns only the collapsed industry mapping."""

    snapshot = load_point_in_time_universe(
        session_date,
        layout=layout,
        active_only=active_only,
        require_snapshot=require_snapshot,
    )
    return snapshot.industry_map.copy()


def _coerce_session_date(session_date: str | pd.Timestamp) -> pd.Timestamp:
    coerced = pd.Timestamp(session_date)
    if coerced.tzinfo is not None:
        coerced = coerced.tz_convert(None)
    return coerced.normalize()


def _select_latest_snapshot(
    frame: pd.DataFrame,
    *,
    date_column: str,
    session_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    if frame.empty or date_column not in frame.columns:
        return pd.DataFrame(), None

    working = frame.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working = working.dropna(subset=[date_column]).copy()
    if working.empty:
        return pd.DataFrame(), None

    eligible = working.loc[working[date_column] <= session_date]
    if eligible.empty:
        return pd.DataFrame(), None

    snapshot_date = eligible[date_column].max()
    snapshot = eligible.loc[eligible[date_column] == snapshot_date].copy().reset_index(drop=True)
    return snapshot, pd.Timestamp(snapshot_date)


def _normalize_symbol_master(
    frame: pd.DataFrame,
    *,
    session_date: pd.Timestamp,
    active_only: bool,
) -> pd.DataFrame:
    working = frame.copy()
    has_is_active = "is_active" in working.columns
    working["as_of_date"] = pd.to_datetime(working["as_of_date"], errors="coerce")

    rename_map = {
        "country_of_listing": "country_of_listing",
        "asset_class": "asset_class",
        "exchange_mic": "exchange_mic",
    }
    working = working.rename(columns=rename_map)
    working = working.reindex(columns=_EMPTY_METADATA_COLUMNS)

    if "list_date" in working.columns:
        working["list_date"] = pd.to_datetime(working["list_date"], errors="coerce")
    if "delist_date" in working.columns:
        working["delist_date"] = pd.to_datetime(working["delist_date"], errors="coerce")

    if active_only and has_is_active:
        active_mask = working["is_active"].fillna(False).astype(bool)
        working = working.loc[active_mask].copy()

    if "list_date" in working.columns:
        listed_mask = working["list_date"].isna() | (working["list_date"] <= session_date)
        working = working.loc[listed_mask].copy()
    if "delist_date" in working.columns:
        not_delisted_mask = working["delist_date"].isna() | (working["delist_date"] > session_date)
        working = working.loc[not_delisted_mask].copy()

    if working.empty:
        return _empty_metadata_frame()

    working["sector"] = working["sector"].fillna("Unknown")
    working["industry"] = working["industry"].fillna("Unknown")
    working["exchange_mic"] = working["exchange_mic"].fillna("Unknown")
    working["currency"] = working["currency"].fillna("USD")
    working["security_type"] = working["security_type"].fillna("Unknown")
    working["asset_class"] = working["asset_class"].fillna("Unknown")
    working["country_of_listing"] = working["country_of_listing"].fillna("Unknown")
    working["source_name"] = working["source_name"].fillna("unknown")
    working["source_version"] = working["source_version"].fillna("unknown")

    return working.sort_values(["symbol"]).reset_index(drop=True)


def _normalize_industry_membership(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["as_of_date"] = pd.to_datetime(working["as_of_date"], errors="coerce")
    working = working.reindex(columns=_EMPTY_INDUSTRY_COLUMNS)
    if working.empty:
        return _empty_industry_membership_frame()

    for column in ("sector_name", "industry_group_name", "industry_name", "subindustry_name"):
        working[column] = working[column].fillna("Unknown")
    working["industry_system"] = working["industry_system"].fillna("Unknown")
    working["source_name"] = working["source_name"].fillna("unknown")
    working["source_version"] = working["source_version"].fillna("unknown")
    return working.sort_values(["symbol", "industry_system"]).reset_index(drop=True)


def _normalize_universe_membership(
    frame: pd.DataFrame,
    *,
    session_date: pd.Timestamp,
    universe_name: str | None,
) -> pd.DataFrame:
    if frame.empty:
        return _empty_universe_membership_frame()

    working = frame.copy()
    working["session_date"] = pd.to_datetime(working["session_date"], errors="coerce")
    working = working.reindex(columns=_EMPTY_UNIVERSE_MEMBERSHIP_COLUMNS)
    working = working.dropna(subset=["session_date", "symbol"]).copy()
    if working.empty:
        return _empty_universe_membership_frame()

    if universe_name:
        working = working.loc[working["universe_name"].astype(str) == universe_name].copy()
    if working.empty:
        return _empty_universe_membership_frame()

    if "is_member" not in working.columns:
        working["is_member"] = True
    working["is_member"] = working["is_member"].fillna(True).astype(bool)
    if "entry_date" in working.columns:
        working["entry_date"] = pd.to_datetime(working["entry_date"], errors="coerce")
    if "exit_date" in working.columns:
        working["exit_date"] = pd.to_datetime(working["exit_date"], errors="coerce")

    active_mask = working["is_member"]
    if "entry_date" in working.columns:
        active_mask &= working["entry_date"].isna() | (working["entry_date"] <= session_date)
    if "exit_date" in working.columns:
        active_mask &= working["exit_date"].isna() | (working["exit_date"] >= session_date)
    working = working.loc[active_mask].copy()
    if working.empty:
        return _empty_universe_membership_frame()

    working["membership_source"] = working["membership_source"].fillna("unknown")
    working["source_name"] = working["source_name"].fillna("unknown")
    working["source_version"] = working["source_version"].fillna("unknown")
    return working.sort_values(["symbol", "universe_name"]).reset_index(drop=True)


def _collapse_industry_membership(
    frame: pd.DataFrame,
    *,
    allowed_symbols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return _empty_industry_map_frame()

    collapsed = frame.rename(
        columns={
            "sector_name": "sector",
            "industry_group_name": "industry_group",
            "industry_name": "industry",
            "subindustry_name": "subindustry",
        }
    ).copy()
    collapsed = collapsed.reindex(columns=_EMPTY_INDUSTRY_MAP_COLUMNS)

    if allowed_symbols is not None:
        allowed = set(allowed_symbols)
        if not allowed:
            return _empty_industry_map_frame()
        collapsed = collapsed.loc[collapsed["symbol"].isin(allowed)].copy()
        if collapsed.empty:
            return _empty_industry_map_frame()

    sort_columns = [column for column in ("symbol", "industry_system", "source_name", "load_time_utc") if column in collapsed.columns]
    if sort_columns:
        collapsed = collapsed.sort_values(sort_columns, kind="stable", na_position="last")

    collapsed = collapsed.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)
    for column in ("sector", "industry_group", "industry", "subindustry", "industry_system"):
        collapsed[column] = collapsed[column].fillna("Unknown")
    collapsed["source_name"] = collapsed["source_name"].fillna("unknown")
    collapsed["source_version"] = collapsed["source_version"].fillna("unknown")
    return collapsed


def _empty_metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_METADATA_COLUMNS)


def _empty_universe_membership_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_UNIVERSE_MEMBERSHIP_COLUMNS)


def _empty_industry_membership_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_INDUSTRY_COLUMNS)


def _empty_industry_map_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_INDUSTRY_MAP_COLUMNS)


def _empty_research_metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "symbol", "company_name", "quote_type", "exchange", "currency", "country", "sector", "industry"]
    )


def _build_exact_snapshot_metadata_history(
    session_dates: list[pd.Timestamp],
    *,
    universe_membership_frame: pd.DataFrame | None,
    symbol_master_frame: pd.DataFrame,
    industry_membership_frame: pd.DataFrame,
    active_only: bool,
    universe_name: str | None,
) -> pd.DataFrame | None:
    if symbol_master_frame.empty:
        return None

    symbol_master = symbol_master_frame.copy()
    symbol_master["as_of_date"] = pd.to_datetime(symbol_master["as_of_date"], errors="coerce").dt.normalize()
    symbol_master = symbol_master.dropna(subset=["as_of_date"]).copy()
    requested_dates = pd.Index(pd.to_datetime(session_dates).normalize())
    available_symbol_dates = pd.Index(symbol_master["as_of_date"].drop_duplicates().sort_values())
    if not requested_dates.isin(available_symbol_dates).all():
        return None

    symbol_master = symbol_master.loc[symbol_master["as_of_date"].isin(requested_dates)].copy()
    if active_only and "is_active" in symbol_master.columns:
        symbol_master = symbol_master.loc[symbol_master["is_active"].fillna(False).astype(bool)].copy()
    if symbol_master.empty:
        return _empty_research_metadata_frame()

    standardized = symbol_master.rename(
        columns={
            "asset_class": "quote_type",
            "exchange_mic": "exchange",
            "country_of_listing": "country",
            "as_of_date": "date",
        }
    )
    standardized["date"] = pd.to_datetime(standardized["date"], errors="coerce").dt.normalize()
    standardized["company_name"] = standardized["company_name"].fillna(standardized["symbol"])
    standardized["quote_type"] = standardized["quote_type"].fillna("Unknown")
    standardized["exchange"] = standardized["exchange"].fillna("Unknown")
    standardized["currency"] = standardized["currency"].fillna("USD")
    standardized["country"] = standardized["country"].fillna("Unknown")
    standardized["sector"] = standardized.get("sector", pd.Series(index=standardized.index, dtype="object")).fillna("Unknown")
    standardized["industry"] = standardized.get("industry", pd.Series(index=standardized.index, dtype="object")).fillna("Unknown")

    if universe_membership_frame is not None and not universe_membership_frame.empty:
        membership = universe_membership_frame.copy()
        membership["session_date"] = pd.to_datetime(membership["session_date"], errors="coerce").dt.normalize()
        membership = membership.dropna(subset=["session_date"]).copy()
        if universe_name:
            membership = membership.loc[membership["universe_name"].astype(str) == universe_name].copy()
        if not membership.empty:
            membership = membership.reindex(columns=_EMPTY_UNIVERSE_MEMBERSHIP_COLUMNS)
            membership["is_member"] = membership["is_member"].fillna(True).astype(bool)
            membership = membership.loc[membership["is_member"]].copy()
            if "entry_date" in membership.columns:
                membership["entry_date"] = pd.to_datetime(membership["entry_date"], errors="coerce")
            if "exit_date" in membership.columns:
                membership["exit_date"] = pd.to_datetime(membership["exit_date"], errors="coerce")
            membership = membership.loc[membership["session_date"].isin(requested_dates)].copy()
            if not membership.empty:
                standardized = standardized.merge(
                    membership[["session_date", "symbol"]].rename(columns={"session_date": "date"}),
                    on=["date", "symbol"],
                    how="inner",
                )

    if not industry_membership_frame.empty:
        industry_map = industry_membership_frame.copy()
        industry_map["as_of_date"] = pd.to_datetime(industry_map["as_of_date"], errors="coerce").dt.normalize()
        industry_map = industry_map.dropna(subset=["as_of_date"]).copy()
        industry_map = industry_map.loc[industry_map["as_of_date"].isin(requested_dates)].copy()
        if not industry_map.empty:
            sort_columns = [column for column in ("symbol", "as_of_date", "industry_system", "load_time_utc") if column in industry_map.columns]
            if sort_columns:
                industry_map = industry_map.sort_values(sort_columns, kind="stable", na_position="last")
            industry_map = industry_map.drop_duplicates(subset=["as_of_date", "symbol"], keep="last")
            industry_map = industry_map.rename(
                columns={
                    "as_of_date": "date",
                    "sector_name": "industry_sector_override",
                    "industry_name": "industry_name_override",
                }
            )
            standardized = standardized.merge(
                industry_map[["date", "symbol", "industry_sector_override", "industry_name_override"]],
                on=["date", "symbol"],
                how="left",
            )
            standardized["sector"] = standardized["industry_sector_override"].fillna(standardized["sector"])
            standardized["industry"] = standardized["industry_name_override"].fillna(standardized["industry"])
            standardized = standardized.drop(columns=["industry_sector_override", "industry_name_override"])

    standardized["sector"] = standardized["sector"].fillna("Unknown")
    standardized["industry"] = standardized["industry"].fillna("Unknown")
    standardized = standardized[
        ["date", "symbol", "company_name", "quote_type", "exchange", "currency", "country", "sector", "industry"]
    ]
    return standardized.sort_values(["date", "symbol"]).reset_index(drop=True)


def _placeholder_metadata_from_membership(frame: pd.DataFrame, *, session_date: pd.Timestamp) -> pd.DataFrame:
    working = frame.copy()
    if working.empty:
        return _empty_research_metadata_frame()
    return pd.DataFrame(
        {
            "date": pd.Timestamp(session_date).normalize(),
            "symbol": working["symbol"].astype(str),
            "company_name": working["symbol"].astype(str),
            "quote_type": "Unknown",
            "exchange": "Unknown",
            "currency": "USD",
            "country": "Unknown",
            "sector": "Unknown",
            "industry": "Unknown",
        }
    ).sort_values(["date", "symbol"]).reset_index(drop=True)


def _standardize_research_metadata(snapshot: PointInTimeUniverse, *, session_date: pd.Timestamp) -> pd.DataFrame:
    metadata = snapshot.metadata.copy()
    if metadata.empty:
        return _empty_research_metadata_frame()

    standardized = metadata.rename(
        columns={
            "asset_class": "quote_type",
            "exchange_mic": "exchange",
            "country_of_listing": "country",
        }
    )
    standardized = standardized[
        ["symbol", "company_name", "quote_type", "exchange", "currency", "country", "sector", "industry"]
    ].copy()

    if not snapshot.industry_map.empty:
        industry_map = snapshot.industry_map[["symbol", "sector", "industry"]].copy()
        industry_map = industry_map.rename(
            columns={
                "sector": "industry_sector_override",
                "industry": "industry_name_override",
            }
        )
        standardized = standardized.merge(industry_map, on="symbol", how="left")
        standardized["sector"] = standardized["industry_sector_override"].fillna(standardized["sector"])
        standardized["industry"] = standardized["industry_name_override"].fillna(standardized["industry"])
        standardized = standardized.drop(columns=["industry_sector_override", "industry_name_override"])

    standardized["date"] = pd.Timestamp(session_date).normalize()
    standardized["company_name"] = standardized["company_name"].fillna(standardized["symbol"])
    standardized["quote_type"] = standardized["quote_type"].fillna("Unknown")
    standardized["exchange"] = standardized["exchange"].fillna("Unknown")
    standardized["currency"] = standardized["currency"].fillna("USD")
    standardized["country"] = standardized["country"].fillna("Unknown")
    standardized["sector"] = standardized["sector"].fillna("Unknown")
    standardized["industry"] = standardized["industry"].fillna("Unknown")
    standardized = standardized[
        ["date", "symbol", "company_name", "quote_type", "exchange", "currency", "country", "sector", "industry"]
    ]
    return standardized.sort_values(["date", "symbol"]).reset_index(drop=True)
