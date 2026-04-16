"""Scheduled and batch ingestion jobs."""

from .bootstrap_yahoo_us_equities import (
    backfill_static_metadata_history_from_silver,
    bootstrap_fmf_validation_etfs_yahoo_to_silver,
    bootstrap_market_symbols_yahoo_to_silver,
    bootstrap_multi_asset_etfs_yahoo_to_silver,
    bootstrap_multi_asset_proxy_etfs_yahoo_to_silver,
    bootstrap_us_equities_yahoo_to_silver,
)
from .us_equities_v1 import (
    collect_adj_factors,
    collect_daily_bars,
    collect_research_seed,
    collect_symbol_master_snapshot,
)

__all__ = [
    "backfill_static_metadata_history_from_silver",
    "bootstrap_fmf_validation_etfs_yahoo_to_silver",
    "bootstrap_market_symbols_yahoo_to_silver",
    "bootstrap_multi_asset_etfs_yahoo_to_silver",
    "bootstrap_multi_asset_proxy_etfs_yahoo_to_silver",
    "bootstrap_us_equities_yahoo_to_silver",
    "collect_adj_factors",
    "collect_daily_bars",
    "collect_research_seed",
    "collect_symbol_master_snapshot",
]
