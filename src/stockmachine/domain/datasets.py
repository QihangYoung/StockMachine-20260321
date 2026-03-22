from __future__ import annotations

from dataclasses import dataclass

from .enums import DataLayer, DataType


@dataclass(slots=True, frozen=True)
class ColumnSpec:
    """A single canonical column definition."""

    name: str
    data_type: DataType
    nullable: bool
    description: str


@dataclass(slots=True, frozen=True)
class TableSpec:
    """A normalized internal table contract."""

    name: str
    layer: DataLayer
    description: str
    primary_key: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    partition_by: tuple[str, ...] = ()

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def has_column(self, column_name: str) -> bool:
        return column_name in self.column_names


def _column(
    name: str,
    data_type: DataType,
    description: str,
    *,
    nullable: bool = False,
) -> ColumnSpec:
    return ColumnSpec(
        name=name,
        data_type=data_type,
        nullable=nullable,
        description=description,
    )


CANONICAL_TABLES: dict[str, TableSpec] = {
    "universe_membership": TableSpec(
        name="universe_membership",
        layer=DataLayer.SILVER,
        description="Point-in-time research-universe membership by session date.",
        primary_key=("session_date", "universe_name", "symbol"),
        partition_by=("session_date",),
        columns=(
            _column("session_date", DataType.DATE, "Trading session date."),
            _column("universe_name", DataType.STRING, "Named research universe."),
            _column("symbol", DataType.STRING, "Ticker symbol."),
            _column("is_member", DataType.BOOLEAN, "Whether the symbol is in the universe."),
            _column(
                "membership_source",
                DataType.STRING,
                "How the membership row was produced or sourced.",
                nullable=True,
            ),
            _column(
                "entry_date",
                DataType.DATE,
                "Optional first effective session date for the constituent.",
                nullable=True,
            ),
            _column(
                "exit_date",
                DataType.DATE,
                "Optional last effective session date for the constituent.",
                nullable=True,
            ),
            _column("source_name", DataType.STRING, "Upstream source identifier."),
            _column("load_time_utc", DataType.DATETIME_UTC, "Load timestamp in UTC."),
            _column(
                "source_version",
                DataType.STRING,
                "Optional upstream schema or batch version.",
                nullable=True,
            ),
        ),
    ),
    "symbol_master": TableSpec(
        name="symbol_master",
        layer=DataLayer.SILVER,
        description="Point-in-time security master for tradable instruments.",
        primary_key=("as_of_date", "symbol"),
        partition_by=("as_of_date",),
        columns=(
            _column("as_of_date", DataType.DATE, "As-of date for the security snapshot."),
            _column("symbol", DataType.STRING, "Ticker symbol used inside the project."),
            _column("security_id", DataType.STRING, "Stable upstream security identifier."),
            _column("company_name", DataType.STRING, "Issuer display name."),
            _column("exchange_mic", DataType.STRING, "MIC code for the listing venue."),
            _column("currency", DataType.STRING, "Trading currency."),
            _column("security_type", DataType.STRING, "Normalized security type."),
            _column("asset_class", DataType.STRING, "Normalized asset class."),
            _column("is_active", DataType.BOOLEAN, "Whether the security is active."),
            _column("list_date", DataType.DATE, "Listing date.", nullable=True),
            _column("delist_date", DataType.DATE, "Delisting date.", nullable=True),
            _column("sector", DataType.STRING, "Point-in-time sector name.", nullable=True),
            _column("industry", DataType.STRING, "Point-in-time industry name.", nullable=True),
            _column(
                "country_of_listing",
                DataType.STRING,
                "Country where the instrument is listed.",
                nullable=True,
            ),
            _column(
                "primary_share_class",
                DataType.BOOLEAN,
                "Whether the symbol is the primary share class.",
                nullable=True,
            ),
            _column("source_name", DataType.STRING, "Upstream source identifier."),
            _column("load_time_utc", DataType.DATETIME_UTC, "Load timestamp in UTC."),
            _column(
                "source_version",
                DataType.STRING,
                "Optional upstream schema or batch version.",
                nullable=True,
            ),
        ),
    ),
    "trading_calendar": TableSpec(
        name="trading_calendar",
        layer=DataLayer.SILVER,
        description="Exchange trading sessions and session boundaries.",
        primary_key=("calendar_name", "session_date"),
        partition_by=("calendar_name",),
        columns=(
            _column("calendar_name", DataType.STRING, "Internal calendar name."),
            _column("session_date", DataType.DATE, "Trading session date."),
            _column("is_open", DataType.BOOLEAN, "Whether the session is open."),
            _column(
                "session_open_utc",
                DataType.DATETIME_UTC,
                "Scheduled open timestamp in UTC.",
                nullable=True,
            ),
            _column(
                "session_close_utc",
                DataType.DATETIME_UTC,
                "Scheduled close timestamp in UTC.",
                nullable=True,
            ),
            _column(
                "previous_open_session",
                DataType.DATE,
                "Previous open trading session.",
                nullable=True,
            ),
            _column(
                "next_open_session",
                DataType.DATE,
                "Next open trading session.",
                nullable=True,
            ),
            _column("source_name", DataType.STRING, "Upstream source identifier."),
            _column("load_time_utc", DataType.DATETIME_UTC, "Load timestamp in UTC."),
            _column(
                "source_version",
                DataType.STRING,
                "Optional upstream schema or batch version.",
                nullable=True,
            ),
        ),
    ),
    "daily_bar": TableSpec(
        name="daily_bar",
        layer=DataLayer.SILVER,
        description="Daily OHLCV bars and daily liquidity summaries.",
        primary_key=("session_date", "symbol"),
        partition_by=("session_date",),
        columns=(
            _column("session_date", DataType.DATE, "Trading session date."),
            _column("symbol", DataType.STRING, "Ticker symbol."),
            _column("open", DataType.FLOAT, "Session open price."),
            _column("high", DataType.FLOAT, "Session high price."),
            _column("low", DataType.FLOAT, "Session low price."),
            _column("close", DataType.FLOAT, "Session close price."),
            _column("volume", DataType.FLOAT, "Traded share volume."),
            _column("vwap", DataType.FLOAT, "Volume-weighted average price.", nullable=True),
            _column("dollar_volume", DataType.FLOAT, "Daily dollar volume.", nullable=True),
            _column("trade_count", DataType.INTEGER, "Number of trades.", nullable=True),
            _column("source_name", DataType.STRING, "Upstream source identifier."),
            _column("load_time_utc", DataType.DATETIME_UTC, "Load timestamp in UTC."),
            _column(
                "effective_time_utc",
                DataType.DATETIME_UTC,
                "Effective time of the upstream record.",
                nullable=True,
            ),
            _column(
                "source_version",
                DataType.STRING,
                "Optional upstream schema or batch version.",
                nullable=True,
            ),
        ),
    ),
    "adj_factor": TableSpec(
        name="adj_factor",
        layer=DataLayer.SILVER,
        description="Point-in-time adjustment factors and cash dividends.",
        primary_key=("session_date", "symbol"),
        partition_by=("session_date",),
        columns=(
            _column("session_date", DataType.DATE, "Trading session date."),
            _column("symbol", DataType.STRING, "Ticker symbol."),
            _column("split_factor", DataType.FLOAT, "Split factor effective on the date."),
            _column("cash_dividend", DataType.FLOAT, "Cash dividend amount.", nullable=True),
            _column(
                "price_adjust_factor",
                DataType.FLOAT,
                "Cumulative adjustment factor for research labels.",
            ),
            _column("source_name", DataType.STRING, "Upstream source identifier."),
            _column("load_time_utc", DataType.DATETIME_UTC, "Load timestamp in UTC."),
            _column(
                "effective_time_utc",
                DataType.DATETIME_UTC,
                "Effective time of the upstream record.",
                nullable=True,
            ),
            _column(
                "source_version",
                DataType.STRING,
                "Optional upstream schema or batch version.",
                nullable=True,
            ),
        ),
    ),
    "benchmark_index": TableSpec(
        name="benchmark_index",
        layer=DataLayer.SILVER,
        description="Reference index or ETF bars used for excess-return labels.",
        primary_key=("session_date", "symbol"),
        partition_by=("session_date",),
        columns=(
            _column("session_date", DataType.DATE, "Trading session date."),
            _column("symbol", DataType.STRING, "Benchmark symbol."),
            _column("open", DataType.FLOAT, "Session open price."),
            _column("high", DataType.FLOAT, "Session high price."),
            _column("low", DataType.FLOAT, "Session low price."),
            _column("close", DataType.FLOAT, "Session close price."),
            _column("volume", DataType.FLOAT, "Traded share volume.", nullable=True),
            _column("return_1d", DataType.FLOAT, "One-session benchmark return.", nullable=True),
            _column("source_name", DataType.STRING, "Upstream source identifier."),
            _column("load_time_utc", DataType.DATETIME_UTC, "Load timestamp in UTC."),
            _column(
                "source_version",
                DataType.STRING,
                "Optional upstream schema or batch version.",
                nullable=True,
            ),
        ),
    ),
    "daily_basic": TableSpec(
        name="daily_basic",
        layer=DataLayer.SILVER,
        description="Daily basic fundamentals and liquidity aggregates.",
        primary_key=("session_date", "symbol"),
        partition_by=("session_date",),
        columns=(
            _column("session_date", DataType.DATE, "Trading session date."),
            _column("symbol", DataType.STRING, "Ticker symbol."),
            _column(
                "shares_outstanding",
                DataType.FLOAT,
                "Total shares outstanding.",
                nullable=True,
            ),
            _column(
                "free_float_shares",
                DataType.FLOAT,
                "Float shares outstanding.",
                nullable=True,
            ),
            _column("market_cap", DataType.FLOAT, "Total market capitalization.", nullable=True),
            _column(
                "free_float_market_cap",
                DataType.FLOAT,
                "Float market capitalization.",
                nullable=True,
            ),
            _column("turnover_rate", DataType.FLOAT, "Daily turnover rate.", nullable=True),
            _column(
                "median_dollar_volume_20d",
                DataType.FLOAT,
                "Median dollar volume over the last 20 sessions.",
                nullable=True,
            ),
            _column(
                "median_dollar_volume_60d",
                DataType.FLOAT,
                "Median dollar volume over the last 60 sessions.",
                nullable=True,
            ),
            _column("source_name", DataType.STRING, "Upstream source identifier."),
            _column("load_time_utc", DataType.DATETIME_UTC, "Load timestamp in UTC."),
            _column(
                "source_version",
                DataType.STRING,
                "Optional upstream schema or batch version.",
                nullable=True,
            ),
        ),
    ),
    "industry_membership": TableSpec(
        name="industry_membership",
        layer=DataLayer.SILVER,
        description="Point-in-time sector and industry mapping.",
        primary_key=("as_of_date", "symbol", "industry_system"),
        partition_by=("as_of_date",),
        columns=(
            _column("as_of_date", DataType.DATE, "As-of date for the mapping."),
            _column("symbol", DataType.STRING, "Ticker symbol."),
            _column("industry_system", DataType.STRING, "Classification system name."),
            _column("sector_name", DataType.STRING, "Sector name.", nullable=True),
            _column(
                "industry_group_name",
                DataType.STRING,
                "Industry group name.",
                nullable=True,
            ),
            _column("industry_name", DataType.STRING, "Industry name.", nullable=True),
            _column("subindustry_name", DataType.STRING, "Subindustry name.", nullable=True),
            _column("source_name", DataType.STRING, "Upstream source identifier."),
            _column("load_time_utc", DataType.DATETIME_UTC, "Load timestamp in UTC."),
            _column(
                "source_version",
                DataType.STRING,
                "Optional upstream schema or batch version.",
                nullable=True,
            ),
        ),
    ),
    "suspend_resume": TableSpec(
        name="suspend_resume",
        layer=DataLayer.SILVER,
        description="Trading halt and resume status windows.",
        primary_key=("symbol", "status_start_time_utc"),
        partition_by=("symbol",),
        columns=(
            _column("symbol", DataType.STRING, "Ticker symbol."),
            _column(
                "status_start_time_utc",
                DataType.DATETIME_UTC,
                "Start time of the status window.",
            ),
            _column(
                "status_end_time_utc",
                DataType.DATETIME_UTC,
                "End time of the status window.",
                nullable=True,
            ),
            _column("status_code", DataType.STRING, "Normalized halt status code."),
            _column("status_reason", DataType.STRING, "Status reason text.", nullable=True),
            _column("is_halted", DataType.BOOLEAN, "Whether the instrument is halted."),
            _column("source_name", DataType.STRING, "Upstream source identifier."),
            _column("load_time_utc", DataType.DATETIME_UTC, "Load timestamp in UTC."),
            _column(
                "source_version",
                DataType.STRING,
                "Optional upstream schema or batch version.",
                nullable=True,
            ),
        ),
    ),
    "price_limit": TableSpec(
        name="price_limit",
        layer=DataLayer.SILVER,
        description="Point-in-time price guardrail or LULD band table.",
        primary_key=("session_date", "symbol"),
        partition_by=("session_date",),
        columns=(
            _column("session_date", DataType.DATE, "Trading session date."),
            _column("symbol", DataType.STRING, "Ticker symbol."),
            _column("limit_type", DataType.STRING, "Normalized limit or band type."),
            _column("lower_band", DataType.FLOAT, "Lower price band.", nullable=True),
            _column("upper_band", DataType.FLOAT, "Upper price band.", nullable=True),
            _column("source_name", DataType.STRING, "Upstream source identifier."),
            _column("load_time_utc", DataType.DATETIME_UTC, "Load timestamp in UTC."),
            _column(
                "source_version",
                DataType.STRING,
                "Optional upstream schema or batch version.",
                nullable=True,
            ),
        ),
    ),
}


def get_table_spec(name: str) -> TableSpec:
    """Return the canonical table specification by name."""

    try:
        return CANONICAL_TABLES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown canonical table: {name}") from exc
