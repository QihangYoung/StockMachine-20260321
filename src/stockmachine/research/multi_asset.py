from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


_MIN_NUMERIC_EPSILON = 1e-12


@dataclass(slots=True, frozen=True)
class BucketDefinition:
    """A named multi-asset risk bucket backed by one or more tradable symbols."""

    name: str
    symbols: tuple[str, ...]
    symbol_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        symbols = tuple(str(symbol) for symbol in self.symbols)
        if not symbols:
            raise ValueError("BucketDefinition requires at least one symbol.")
        if len(set(symbols)) != len(symbols):
            raise ValueError(f"BucketDefinition '{self.name}' contains duplicate symbols.")
        object.__setattr__(self, "symbols", symbols)

        if self.symbol_weights is None:
            return

        weights = tuple(float(weight) for weight in self.symbol_weights)
        if len(weights) != len(symbols):
            raise ValueError(
                f"BucketDefinition '{self.name}' must provide one symbol weight per symbol."
            )
        if any(weight < 0.0 for weight in weights):
            raise ValueError(f"BucketDefinition '{self.name}' weights must be non-negative.")
        if sum(weights) <= 0.0:
            raise ValueError(f"BucketDefinition '{self.name}' weights must sum to a positive value.")
        object.__setattr__(self, "symbol_weights", weights)

    def normalized_symbol_weights(self) -> pd.Series:
        """Return bucket member weights normalized to sum to one."""

        if self.symbol_weights is None:
            weights = np.full(len(self.symbols), 1.0 / len(self.symbols), dtype=float)
        else:
            weights = np.asarray(self.symbol_weights, dtype=float)
            weights = weights / weights.sum()
        return pd.Series(weights, index=pd.Index(self.symbols, name="symbol"), dtype=float)


@dataclass(slots=True, frozen=True)
class CovarianceConfig:
    """Configuration for the default blended covariance estimator."""

    long_lookback: int = 252
    short_lookback: int = 63
    ewma_lambda: float = 0.97
    long_weight: float = 0.70
    short_weight: float = 0.30
    min_observations: int | None = None
    min_eigenvalue: float = 1e-10

    def __post_init__(self) -> None:
        if self.long_lookback < 2:
            raise ValueError("CovarianceConfig.long_lookback must be at least 2.")
        if self.short_lookback < 2:
            raise ValueError("CovarianceConfig.short_lookback must be at least 2.")
        if not 0.0 < self.ewma_lambda < 1.0:
            raise ValueError("CovarianceConfig.ewma_lambda must be strictly between 0 and 1.")
        if self.long_weight < 0.0 or self.short_weight < 0.0:
            raise ValueError("CovarianceConfig blend weights must be non-negative.")
        if self.long_weight + self.short_weight <= 0.0:
            raise ValueError("CovarianceConfig blend weights must sum to a positive value.")
        if self.min_observations is not None and self.min_observations < 2:
            raise ValueError("CovarianceConfig.min_observations must be at least 2.")
        if self.min_eigenvalue < 0.0:
            raise ValueError("CovarianceConfig.min_eigenvalue must be non-negative.")


@dataclass(slots=True, frozen=True)
class BlendedCovarianceEstimate:
    """A covariance estimate with access to its long and short components."""

    covariance: pd.DataFrame
    long_covariance: pd.DataFrame
    short_covariance: pd.DataFrame
    observation_count: int


@dataclass(slots=True, frozen=True)
class AffineProxySegment:
    """One proxy segment that maps a proxy symbol into a target symbol via affine fit."""

    target_symbol: str
    proxy_symbol: str
    segment_name: str
    apply_start: str | pd.Timestamp | None = None
    apply_end: str | pd.Timestamp | None = None
    fit_start: str | pd.Timestamp | None = None
    fit_end: str | pd.Timestamp | None = None
    min_observations: int = 20

    def __post_init__(self) -> None:
        if self.min_observations < 2:
            raise ValueError("AffineProxySegment.min_observations must be at least 2.")


@dataclass(slots=True, frozen=True)
class ProxyChainResult:
    """Proxy-extended symbol returns plus per-segment diagnostics."""

    symbol_returns: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass(slots=True, frozen=True)
class RiskBudgetResult:
    """Resolved weights and risk-allocation diagnostics for a covariance matrix."""

    weights: pd.Series
    risk_budgets: pd.Series
    portfolio_volatility: float
    marginal_risk_contributions: pd.Series
    total_risk_contributions: pd.Series
    risk_shares: pd.Series
    iterations: int
    converged: bool


@dataclass(slots=True, frozen=True)
class RiskBudgetPolicyConfig:
    """Policy-level risk-budget settings for a strategic multi-asset core."""

    name: str
    strategic_buckets: tuple[str, ...]
    reserve_buckets: tuple[str, ...] = ()
    risk_budgets: Mapping[str, float] | None = None
    reserve_capital_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        strategic_buckets = tuple(str(bucket) for bucket in self.strategic_buckets)
        reserve_buckets = tuple(str(bucket) for bucket in self.reserve_buckets)
        if not strategic_buckets:
            raise ValueError("RiskBudgetPolicyConfig requires at least one strategic bucket.")
        if len(set(strategic_buckets)) != len(strategic_buckets):
            raise ValueError("RiskBudgetPolicyConfig strategic buckets must be unique.")
        if len(set(reserve_buckets)) != len(reserve_buckets):
            raise ValueError("RiskBudgetPolicyConfig reserve buckets must be unique.")
        if set(strategic_buckets).intersection(reserve_buckets):
            raise ValueError("Strategic buckets and reserve buckets must be disjoint.")
        object.__setattr__(self, "strategic_buckets", strategic_buckets)
        object.__setattr__(self, "reserve_buckets", reserve_buckets)

        if self.risk_budgets is not None:
            budget_keys = {str(key) for key in self.risk_budgets}
            if budget_keys != set(strategic_buckets):
                raise ValueError(
                    "RiskBudgetPolicyConfig risk_budgets must match strategic buckets exactly."
                )
            if any(float(value) <= 0.0 for value in self.risk_budgets.values()):
                raise ValueError("RiskBudgetPolicyConfig risk budgets must be strictly positive.")

        if self.reserve_capital_weights is not None:
            reserve_keys = {str(key) for key in self.reserve_capital_weights}
            if not reserve_keys.issubset(set(reserve_buckets)):
                raise ValueError(
                    "RiskBudgetPolicyConfig reserve_capital_weights must be a subset of reserve buckets."
                )
            if any(float(value) < 0.0 for value in self.reserve_capital_weights.values()):
                raise ValueError(
                    "RiskBudgetPolicyConfig reserve capital weights must be non-negative."
                )
            if sum(float(value) for value in self.reserve_capital_weights.values()) >= 1.0:
                raise ValueError(
                    "RiskBudgetPolicyConfig reserve capital weights must sum to less than 1.0."
                )

    def normalized_risk_budgets(self) -> pd.Series:
        """Return strategic risk budgets normalized to sum to one."""

        if self.risk_budgets is None:
            return pd.Series(
                np.full(len(self.strategic_buckets), 1.0 / len(self.strategic_buckets), dtype=float),
                index=pd.Index(self.strategic_buckets, name="bucket"),
                dtype=float,
            )
        budgets = pd.Series(
            {str(key): float(value) for key, value in self.risk_budgets.items()},
            dtype=float,
        ).reindex(self.strategic_buckets)
        return budgets / budgets.sum()

    def reserve_capital_series(self) -> pd.Series:
        """Return absolute reserve-bucket capital weights."""

        reserve = pd.Series(
            0.0,
            index=pd.Index(self.reserve_buckets, name="bucket"),
            dtype=float,
        )
        if self.reserve_capital_weights is None:
            return reserve
        for bucket_name, value in self.reserve_capital_weights.items():
            reserve.loc[str(bucket_name)] = float(value)
        return reserve

    @property
    def strategic_capital_fraction(self) -> float:
        """Capital fraction left for strategic sleeves after reserve capital."""

        return float(1.0 - self.reserve_capital_series().sum())


@dataclass(slots=True, frozen=True)
class ConfiguredRiskBudgetAllocation:
    """Resolved strategic allocation plus any reserved capital sleeves."""

    config: RiskBudgetPolicyConfig
    strategic_result: RiskBudgetResult
    full_weights: pd.Series
    reserve_capital_weights: pd.Series


@dataclass(slots=True, frozen=True)
class RollingAllocationConfig:
    """Rolling schedule assumptions for multi-asset re-optimization."""

    rebalance_frequency: int = 21
    effective_lag: int = 1
    min_history: int | None = None
    start_date: str | pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if self.rebalance_frequency < 1:
            raise ValueError("RollingAllocationConfig.rebalance_frequency must be at least 1.")
        if self.effective_lag < 1:
            raise ValueError("RollingAllocationConfig.effective_lag must be at least 1.")
        if self.min_history is not None and self.min_history < 2:
            raise ValueError("RollingAllocationConfig.min_history must be at least 2.")


@dataclass(slots=True, frozen=True)
class SharpeTargetVolConfig:
    """Rolling target-vol settings for a Sharpe-score-driven long-only allocator."""

    strategic_buckets: tuple[str, ...]
    cash_bucket: str
    score_lookback: int = 252
    target_volatility: float = 0.10

    def __post_init__(self) -> None:
        strategic_buckets = tuple(str(bucket) for bucket in self.strategic_buckets)
        if not strategic_buckets:
            raise ValueError("SharpeTargetVolConfig requires at least one strategic bucket.")
        if len(set(strategic_buckets)) != len(strategic_buckets):
            raise ValueError("SharpeTargetVolConfig strategic buckets must be unique.")
        if str(self.cash_bucket) in set(strategic_buckets):
            raise ValueError("cash_bucket must not overlap strategic buckets.")
        object.__setattr__(self, "strategic_buckets", strategic_buckets)
        object.__setattr__(self, "cash_bucket", str(self.cash_bucket))
        if self.score_lookback < 2:
            raise ValueError("SharpeTargetVolConfig.score_lookback must be at least 2.")
        if self.target_volatility <= 0.0:
            raise ValueError("SharpeTargetVolConfig.target_volatility must be positive.")


@dataclass(slots=True, frozen=True)
class ThresholdRebalanceConfig:
    """Band-based rebalance settings around a target allocation path."""

    drift_threshold: float
    stale_time_cap: int | None = None

    def __post_init__(self) -> None:
        if self.drift_threshold <= 0.0:
            raise ValueError("ThresholdRebalanceConfig.drift_threshold must be positive.")
        if self.stale_time_cap is not None and self.stale_time_cap < 1:
            raise ValueError("ThresholdRebalanceConfig.stale_time_cap must be at least 1.")


@dataclass(slots=True, frozen=True)
class RollingAllocationResult:
    """Sparse rebalance schedule plus per-step diagnostics."""

    weight_schedule: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass(slots=True, frozen=True)
class ThresholdRebalanceResult:
    """Executed schedule plus daily diagnostics for a threshold-aware rebalance policy."""

    weight_schedule: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass(slots=True, frozen=True)
class MultiAssetBacktestResult:
    """Backtest records and summary for a bucket-level allocation schedule."""

    records: pd.DataFrame
    summary: dict[str, float]
    weight_schedule: pd.DataFrame


@dataclass(slots=True, frozen=True)
class StrategyRunArtifacts:
    """Compact bundle for one named multi-asset strategy run."""

    strategy_name: str
    records: pd.DataFrame
    summary: dict[str, float]
    weight_schedule: pd.DataFrame
    diagnostics: pd.DataFrame | None = None


DEFAULT_MULTI_ASSET_BUCKETS: tuple[BucketDefinition, ...] = (
    BucketDefinition(name="equity_us", symbols=("SPY",)),
    BucketDefinition(name="equity_ex_us", symbols=("VXUS",)),
    BucketDefinition(name="duration", symbols=("IEF",)),
    BucketDefinition(name="credit", symbols=("LQD",)),
    BucketDefinition(name="inflation_hedge", symbols=("GLDM",)),
    BucketDefinition(name="trend", symbols=("CTA",)),
    BucketDefinition(name="cash", symbols=("SGOV",)),
)
DEFAULT_STRATEGIC_BUCKET_NAMES: tuple[str, ...] = (
    "equity_us",
    "equity_ex_us",
    "duration",
    "credit",
    "inflation_hedge",
    "trend",
)
DEFAULT_RESERVE_BUCKET_NAMES: tuple[str, ...] = ("cash",)
DEFAULT_ERC_POLICY_CONFIG = RiskBudgetPolicyConfig(
    name="erc_non_cash",
    strategic_buckets=DEFAULT_STRATEGIC_BUCKET_NAMES,
    reserve_buckets=DEFAULT_RESERVE_BUCKET_NAMES,
    reserve_capital_weights={"cash": 0.0},
)
DEFAULT_C2_V0_POLICY_CONFIG = RiskBudgetPolicyConfig(
    name="c2_v0_balanced_defensive",
    strategic_buckets=DEFAULT_STRATEGIC_BUCKET_NAMES,
    reserve_buckets=DEFAULT_RESERVE_BUCKET_NAMES,
    risk_budgets={
        "equity_us": 0.24,
        "equity_ex_us": 0.11,
        "duration": 0.20,
        "credit": 0.10,
        "inflation_hedge": 0.15,
        "trend": 0.20,
    },
    reserve_capital_weights={"cash": 0.05},
)
DEFAULT_CURRENT_C_POLICY_CORE_WEIGHTS = pd.Series(
    {
        "SPY": 0.10,
        "VXUS": 0.10,
        "AGG": 0.25,
        "CTA": 0.35,
        "GLDM": 0.20,
    },
    dtype=float,
)
DEFAULT_MULTI_ASSET_PROXY_SYMBOLS: tuple[str, ...] = ("FMF", "DBMF", "BIL", "GLD")
DEFAULT_MULTI_ASSET_PROXY_CHAIN: tuple[AffineProxySegment, ...] = (
    AffineProxySegment(
        target_symbol="CTA",
        proxy_symbol="FMF",
        segment_name="cta_pre_dbmf_fmf",
        apply_end="2019-05-08",
    ),
    AffineProxySegment(
        target_symbol="CTA",
        proxy_symbol="DBMF",
        segment_name="cta_dbmf_to_live",
        apply_start="2019-01-01",
    ),
    AffineProxySegment(
        target_symbol="GLDM",
        proxy_symbol="GLD",
        segment_name="gldm_pre_live_gld",
        apply_end="2018-06-26",
    ),
    AffineProxySegment(
        target_symbol="SGOV",
        proxy_symbol="BIL",
        segment_name="sgov_pre_live_bil",
        apply_end="2020-06-01",
    ),
)

FMF_VALIDATION_BUCKETS: tuple[BucketDefinition, ...] = (
    BucketDefinition(name="equity_us", symbols=("SPY",)),
    BucketDefinition(name="equity_ex_us", symbols=("VXUS",)),
    BucketDefinition(name="duration", symbols=("IEF",)),
    BucketDefinition(name="credit", symbols=("LQD",)),
    BucketDefinition(name="inflation_hedge", symbols=("GLD",)),
    BucketDefinition(name="trend", symbols=("FMF",)),
    BucketDefinition(name="cash", symbols=("BIL",)),
)
FMF_VALIDATION_STRATEGIC_BUCKET_NAMES: tuple[str, ...] = (
    "equity_us",
    "equity_ex_us",
    "duration",
    "credit",
    "inflation_hedge",
    "trend",
)
FMF_VALIDATION_RESERVE_BUCKET_NAMES: tuple[str, ...] = ("cash",)
FMF_VALIDATION_ERC_POLICY_CONFIG = RiskBudgetPolicyConfig(
    name="fmf_validation_erc_non_cash",
    strategic_buckets=FMF_VALIDATION_STRATEGIC_BUCKET_NAMES,
    reserve_buckets=FMF_VALIDATION_RESERVE_BUCKET_NAMES,
    reserve_capital_weights={"cash": 0.0},
)
FMF_VALIDATION_C2_V0_POLICY_CONFIG = RiskBudgetPolicyConfig(
    name="fmf_validation_c2_v0_seed",
    strategic_buckets=FMF_VALIDATION_STRATEGIC_BUCKET_NAMES,
    reserve_buckets=FMF_VALIDATION_RESERVE_BUCKET_NAMES,
    risk_budgets={
        "equity_us": 0.24,
        "equity_ex_us": 0.11,
        "duration": 0.20,
        "credit": 0.10,
        "inflation_hedge": 0.15,
        "trend": 0.20,
    },
    reserve_capital_weights={"cash": 0.05},
)


def build_adjusted_price_frame(
    daily_bar: pd.DataFrame,
    *,
    adj_factor: pd.DataFrame | None = None,
    date_column: str = "session_date",
    symbol_column: str = "symbol",
    close_column: str = "close",
    price_adjust_factor_column: str = "price_adjust_factor",
) -> pd.DataFrame:
    """Pivot long-form price history into a wide adjusted-close frame."""

    required_columns = {date_column, symbol_column, close_column}
    missing_columns = required_columns.difference(daily_bar.columns)
    if missing_columns:
        raise KeyError(f"daily_bar is missing required columns: {sorted(missing_columns)}.")

    normalized = daily_bar.loc[:, [date_column, symbol_column, close_column]].copy()
    normalized[date_column] = pd.to_datetime(normalized[date_column], utc=False)
    normalized[symbol_column] = normalized[symbol_column].astype(str)
    normalized[close_column] = pd.to_numeric(normalized[close_column], errors="coerce")

    if adj_factor is not None:
        factor_required_columns = {date_column, symbol_column, price_adjust_factor_column}
        missing_factor_columns = factor_required_columns.difference(adj_factor.columns)
        if missing_factor_columns:
            raise KeyError(
                "adj_factor is missing required columns: "
                f"{sorted(missing_factor_columns)}."
            )
        factor_frame = adj_factor.loc[
            :, [date_column, symbol_column, price_adjust_factor_column]
        ].copy()
        factor_frame[date_column] = pd.to_datetime(factor_frame[date_column], utc=False)
        factor_frame[symbol_column] = factor_frame[symbol_column].astype(str)
        factor_frame[price_adjust_factor_column] = pd.to_numeric(
            factor_frame[price_adjust_factor_column],
            errors="coerce",
        )
        normalized = normalized.merge(
            factor_frame,
            how="left",
            on=[date_column, symbol_column],
            validate="one_to_one",
        )
    else:
        factor_values = daily_bar.get(price_adjust_factor_column)
        normalized[price_adjust_factor_column] = (
            pd.to_numeric(factor_values, errors="coerce")
            if factor_values is not None
            else np.nan
        )

    duplicated = normalized.duplicated(subset=[date_column, symbol_column], keep=False)
    if duplicated.any():
        duplicates = normalized.loc[duplicated, [date_column, symbol_column]]
        preview = duplicates.head(5).to_dict(orient="records")
        raise ValueError(f"Found duplicate date-symbol rows while building prices: {preview}.")

    normalized[price_adjust_factor_column] = normalized[price_adjust_factor_column].fillna(1.0)
    normalized["adjusted_close"] = (
        normalized[close_column].astype(float)
        * normalized[price_adjust_factor_column].astype(float)
    )
    price_frame = (
        normalized.pivot(index=date_column, columns=symbol_column, values="adjusted_close")
        .sort_index()
        .sort_index(axis=1)
    )
    price_frame.columns.name = "symbol"
    return price_frame.astype(float)


def compute_return_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide price frame into close-to-close percentage returns."""

    normalized = _normalize_numeric_frame(price_frame, frame_name="price_frame")
    returns = normalized.sort_index().pct_change(fill_method=None)
    returns = returns.iloc[1:].dropna(how="all")
    return returns.astype(float)


def build_symbol_return_frame(
    daily_bar: pd.DataFrame,
    *,
    adj_factor: pd.DataFrame | None = None,
    date_column: str = "session_date",
    symbol_column: str = "symbol",
    close_column: str = "close",
    price_adjust_factor_column: str = "price_adjust_factor",
) -> pd.DataFrame:
    """Build a wide daily-return frame from canonical daily bars and adjustment factors."""

    adjusted_prices = build_adjusted_price_frame(
        daily_bar,
        adj_factor=adj_factor,
        date_column=date_column,
        symbol_column=symbol_column,
        close_column=close_column,
        price_adjust_factor_column=price_adjust_factor_column,
    )
    return compute_return_frame(adjusted_prices)


def apply_affine_proxy_chain(
    symbol_returns: pd.DataFrame,
    proxy_segments: Sequence[AffineProxySegment],
) -> ProxyChainResult:
    """Fill pre-live history using affine-fitted proxy return segments."""

    normalized = _normalize_numeric_frame(symbol_returns, frame_name="symbol_returns")
    if not proxy_segments:
        return ProxyChainResult(symbol_returns=normalized, diagnostics=pd.DataFrame())

    required_symbols = {
        str(segment.target_symbol)
        for segment in proxy_segments
    }.union({str(segment.proxy_symbol) for segment in proxy_segments})
    missing_symbols = [symbol for symbol in sorted(required_symbols) if symbol not in normalized.columns]
    if missing_symbols:
        raise KeyError(f"symbol_returns is missing required proxy-chain symbols: {missing_symbols}.")

    chained = normalized.copy()
    diagnostics_rows: list[dict[str, object]] = []
    grouped_segments: dict[str, list[AffineProxySegment]] = {}
    for segment in proxy_segments:
        grouped_segments.setdefault(str(segment.target_symbol), []).append(segment)

    for target_symbol, target_segments in grouped_segments.items():
        target_series = chained[target_symbol].astype(float).copy()
        live_series = normalized[target_symbol].astype(float)
        for segment in target_segments:
            proxy_series = normalized[str(segment.proxy_symbol)].astype(float)
            overlap = pd.concat(
                [live_series.rename("target"), proxy_series.rename("proxy")],
                axis=1,
            ).dropna(how="any")
            fit_start = _normalize_timestamp(segment.fit_start)
            fit_end = _normalize_timestamp(segment.fit_end)
            if fit_start is not None:
                overlap = overlap.loc[overlap.index >= fit_start]
            if fit_end is not None:
                overlap = overlap.loc[overlap.index <= fit_end]
            if len(overlap) < segment.min_observations:
                raise ValueError(
                    f"Proxy segment '{segment.segment_name}' has only {len(overlap)} overlap observations; "
                    f"need at least {segment.min_observations}."
                )

            intercept, slope, r_squared = fit_affine_proxy_model(
                target_returns=overlap["target"],
                proxy_returns=overlap["proxy"],
            )
            fitted_proxy = intercept + slope * proxy_series
            apply_mask = target_series.isna()
            apply_start = _normalize_timestamp(segment.apply_start)
            apply_end = _normalize_timestamp(segment.apply_end)
            if apply_start is not None:
                apply_mask = apply_mask & (target_series.index >= apply_start)
            if apply_end is not None:
                apply_mask = apply_mask & (target_series.index <= apply_end)

            applied_values = fitted_proxy.loc[apply_mask]
            target_series.loc[apply_mask] = applied_values
            diagnostics_rows.append(
                {
                    "segment_name": segment.segment_name,
                    "target_symbol": target_symbol,
                    "proxy_symbol": str(segment.proxy_symbol),
                    "fit_start": overlap.index.min(),
                    "fit_end": overlap.index.max(),
                    "overlap_observations": int(len(overlap)),
                    "intercept": intercept,
                    "slope": slope,
                    "r_squared": r_squared,
                    "apply_start": applied_values.index.min() if not applied_values.empty else None,
                    "apply_end": applied_values.index.max() if not applied_values.empty else None,
                    "applied_observations": int(len(applied_values)),
                }
            )
        chained[target_symbol] = target_series.astype(float)

    diagnostics = pd.DataFrame(diagnostics_rows)
    return ProxyChainResult(symbol_returns=chained, diagnostics=diagnostics)


def fit_affine_proxy_model(
    *,
    target_returns: pd.Series,
    proxy_returns: pd.Series,
) -> tuple[float, float, float]:
    """Fit target = intercept + slope * proxy on overlap returns."""

    aligned = pd.concat(
        [target_returns.rename("target"), proxy_returns.rename("proxy")],
        axis=1,
    ).dropna(how="any")
    if len(aligned) < 2:
        raise ValueError("Need at least two aligned observations to fit affine proxy model.")

    x = aligned["proxy"].to_numpy(dtype=float)
    y = aligned["target"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(aligned), dtype=float), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(coefficients[0])
    slope = float(coefficients[1])
    fitted = intercept + slope * x
    residual = y - fitted
    centered = y - y.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= _MIN_NUMERIC_EPSILON:
        r_squared = 1.0
    else:
        r_squared = float(1.0 - np.dot(residual, residual) / denominator)
    return intercept, slope, r_squared


def build_bucket_return_frame(
    symbol_returns: pd.DataFrame,
    buckets: Sequence[BucketDefinition],
) -> pd.DataFrame:
    """Aggregate symbol-level returns into bucket-level return series."""

    if not buckets:
        raise ValueError("At least one bucket is required to build bucket returns.")

    bucket_names = [bucket.name for bucket in buckets]
    if len(set(bucket_names)) != len(bucket_names):
        raise ValueError("Bucket names must be unique.")

    normalized = _normalize_numeric_frame(symbol_returns, frame_name="symbol_returns")
    required_symbols = tuple(
        dict.fromkeys(symbol for bucket in buckets for symbol in bucket.symbols)
    )
    missing_symbols = [symbol for symbol in required_symbols if symbol not in normalized.columns]
    if missing_symbols:
        raise KeyError(f"symbol_returns is missing required symbols: {missing_symbols}.")

    aligned_returns = normalized.loc[:, list(required_symbols)].dropna(how="any")
    if aligned_returns.empty:
        raise ValueError(
            "No overlapping symbol return history remains after enforcing common intersection."
        )

    bucket_series: dict[str, pd.Series] = {}
    for bucket in buckets:
        weights = bucket.normalized_symbol_weights()
        bucket_returns = aligned_returns.loc[:, list(weights.index)].mul(weights, axis=1).sum(axis=1)
        bucket_series[bucket.name] = bucket_returns.astype(float)

    return pd.DataFrame(bucket_series, index=aligned_returns.index, dtype=float)


def estimate_sample_covariance(
    bucket_returns: pd.DataFrame,
    *,
    lookback: int,
    min_eigenvalue: float = 1e-10,
) -> pd.DataFrame:
    """Estimate a rolling sample covariance matrix from recent bucket returns."""

    window = _select_return_window(bucket_returns, lookback=lookback)
    covariance = window.cov().astype(float)
    return repair_covariance_matrix(covariance, min_eigenvalue=min_eigenvalue)


def estimate_ewma_covariance(
    bucket_returns: pd.DataFrame,
    *,
    lookback: int,
    ewma_lambda: float,
    min_eigenvalue: float = 1e-10,
) -> pd.DataFrame:
    """Estimate an exponentially weighted covariance matrix from recent bucket returns."""

    if not 0.0 < ewma_lambda < 1.0:
        raise ValueError("ewma_lambda must be strictly between 0 and 1.")

    window = _select_return_window(bucket_returns, lookback=lookback)
    values = window.to_numpy(dtype=float)
    weights = np.power(ewma_lambda, np.arange(len(window) - 1, -1, -1, dtype=float))
    weights = weights / weights.sum()
    weighted_mean = np.average(values, axis=0, weights=weights)
    centered = values - weighted_mean
    covariance = (centered * weights[:, None]).T @ centered
    covariance_frame = pd.DataFrame(
        covariance,
        index=window.columns.copy(),
        columns=window.columns.copy(),
        dtype=float,
    )
    return repair_covariance_matrix(covariance_frame, min_eigenvalue=min_eigenvalue)


def estimate_blended_covariance(
    bucket_returns: pd.DataFrame,
    *,
    config: CovarianceConfig | None = None,
) -> BlendedCovarianceEstimate:
    """Blend a long-horizon sample covariance with a short-horizon EWMA covariance."""

    resolved_config = config or CovarianceConfig()
    cleaned = _normalize_numeric_frame(bucket_returns, frame_name="bucket_returns").dropna(how="any")
    min_observations = resolved_config.min_observations or 2
    if len(cleaned) < min_observations:
        raise ValueError(
            f"Need at least {min_observations} aligned observations for covariance estimation."
        )

    long_covariance = estimate_sample_covariance(
        cleaned,
        lookback=resolved_config.long_lookback,
        min_eigenvalue=resolved_config.min_eigenvalue,
    )
    short_covariance = estimate_ewma_covariance(
        cleaned,
        lookback=resolved_config.short_lookback,
        ewma_lambda=resolved_config.ewma_lambda,
        min_eigenvalue=resolved_config.min_eigenvalue,
    )

    blend_weight_sum = resolved_config.long_weight + resolved_config.short_weight
    blended_covariance = (
        resolved_config.long_weight * long_covariance
        + resolved_config.short_weight * short_covariance
    ) / blend_weight_sum
    blended_covariance = repair_covariance_matrix(
        blended_covariance,
        min_eigenvalue=resolved_config.min_eigenvalue,
    )
    return BlendedCovarianceEstimate(
        covariance=blended_covariance,
        long_covariance=long_covariance,
        short_covariance=short_covariance,
        observation_count=int(len(cleaned)),
    )


def repair_covariance_matrix(
    covariance: pd.DataFrame | np.ndarray,
    *,
    min_eigenvalue: float = 1e-10,
) -> pd.DataFrame:
    """Project a covariance matrix onto the positive semidefinite cone."""

    if min_eigenvalue < 0.0:
        raise ValueError("min_eigenvalue must be non-negative.")

    covariance_frame = _coerce_covariance_frame(covariance)
    symmetrized = (covariance_frame.to_numpy(dtype=float) + covariance_frame.to_numpy(dtype=float).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrized)
    clipped_eigenvalues = np.clip(eigenvalues, min_eigenvalue, None)
    repaired = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T
    repaired = (repaired + repaired.T) / 2.0
    return pd.DataFrame(
        repaired,
        index=covariance_frame.index.copy(),
        columns=covariance_frame.columns.copy(),
        dtype=float,
    )


def portfolio_volatility(
    covariance: pd.DataFrame | np.ndarray,
    weights: pd.Series | Sequence[float] | Mapping[str, float],
) -> float:
    """Return portfolio volatility implied by weights and covariance."""

    covariance_frame = _coerce_covariance_frame(covariance)
    weight_vector = _coerce_weight_vector(weights, asset_names=covariance_frame.columns)
    variance = float(weight_vector @ covariance_frame.to_numpy(dtype=float) @ weight_vector)
    return float(np.sqrt(max(variance, 0.0)))


def estimate_trailing_asset_sharpe_scores(
    bucket_returns: pd.DataFrame,
    *,
    asset_names: Sequence[str] | None = None,
    lookback: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate annualized trailing Sharpe scores for the requested assets."""

    cleaned = _normalize_numeric_frame(bucket_returns, frame_name="bucket_returns").dropna(how="any")
    if asset_names is None:
        selected = cleaned
    else:
        normalized_names = [str(asset_name) for asset_name in asset_names]
        missing_assets = [asset_name for asset_name in normalized_names if asset_name not in cleaned.columns]
        if missing_assets:
            raise KeyError(f"bucket_returns is missing requested assets: {missing_assets}.")
        selected = cleaned.loc[:, normalized_names]

    window = _select_return_window(selected, lookback=lookback)
    score_rows: list[dict[str, float | str]] = []
    score_values: dict[str, float] = {}
    for asset_name in window.columns:
        series = window[asset_name].astype(float)
        annualized_return = _annualize_total_return(
            float((1.0 + series).prod() - 1.0),
            periods=len(series),
            horizon=1,
        )
        annualized_volatility = (
            float(series.std(ddof=1) * np.sqrt(252))
            if len(series) > 1
            else np.nan
        )
        sharpe = (
            float((series.mean() / series.std(ddof=1)) * np.sqrt(252))
            if len(series) > 1 and series.std(ddof=1) > 0.0
            else 0.0
        )
        score_values[str(asset_name)] = float(sharpe)
        score_rows.append(
            {
                "asset": str(asset_name),
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe_score": float(sharpe),
                "observation_count": int(len(series)),
            }
        )

    score_series = pd.Series(score_values, dtype=float)
    score_frame = pd.DataFrame(score_rows)
    return score_series, score_frame


def solve_long_only_sharpe_score_weights(
    covariance: pd.DataFrame | np.ndarray,
    *,
    sharpe_scores: pd.Series | Sequence[float] | Mapping[str, float],
) -> pd.Series:
    """Solve a long-only tangency-style portfolio using Sharpe-score proxies."""

    covariance_frame = repair_covariance_matrix(covariance)
    asset_names = covariance_frame.columns
    if isinstance(sharpe_scores, pd.Series):
        score_series = sharpe_scores.astype(float).copy()
        score_series.index = pd.Index([str(label) for label in score_series.index], dtype=object)
    elif isinstance(sharpe_scores, Mapping):
        score_series = pd.Series(
            {str(key): float(value) for key, value in sharpe_scores.items()},
            dtype=float,
        )
    else:
        vector = np.asarray(tuple(sharpe_scores), dtype=float)
        if len(vector) != len(asset_names):
            raise ValueError("sharpe_scores sequence length must match covariance dimension.")
        score_series = pd.Series(vector, index=asset_names, dtype=float)

    if set(score_series.index) != set(asset_names):
        raise ValueError("sharpe_scores labels must match covariance asset names exactly.")
    score_series = score_series.reindex(asset_names).fillna(0.0).astype(float)
    if not np.isfinite(score_series.to_numpy(dtype=float)).all():
        raise ValueError("sharpe_scores must be finite.")

    if float(score_series.max()) <= 0.0:
        best_asset = str(score_series.idxmax())
        return pd.Series(
            [1.0 if asset_name == best_asset else 0.0 for asset_name in asset_names],
            index=asset_names,
            dtype=float,
        )

    active_assets = [
        str(asset_name) for asset_name, score in score_series.items() if float(score) > 0.0
    ]
    if not active_assets:
        active_assets = [str(score_series.idxmax())]

    while active_assets:
        active_covariance = covariance_frame.loc[active_assets, active_assets].to_numpy(dtype=float)
        active_scores = score_series.loc[active_assets].to_numpy(dtype=float)
        try:
            raw_weights = np.linalg.solve(active_covariance, active_scores)
        except np.linalg.LinAlgError:
            raw_weights = np.linalg.pinv(active_covariance) @ active_scores
        positive_positions = raw_weights > _MIN_NUMERIC_EPSILON
        if positive_positions.all():
            normalized = raw_weights / raw_weights.sum()
            resolved = pd.Series(0.0, index=asset_names, dtype=float)
            resolved.loc[active_assets] = normalized
            return resolved
        next_active_assets = [
            active_assets[position]
            for position, positive in enumerate(positive_positions)
            if positive
        ]
        if not next_active_assets:
            break
        if next_active_assets == active_assets:
            break
        active_assets = next_active_assets

    best_asset = str(score_series.idxmax())
    return pd.Series(
        [1.0 if asset_name == best_asset else 0.0 for asset_name in asset_names],
        index=asset_names,
        dtype=float,
    )


def allocate_vol_capped_portfolio_into_cash(
    strategic_weights: pd.Series | Sequence[float] | Mapping[str, float],
    *,
    covariance: pd.DataFrame | np.ndarray,
    strategic_buckets: Sequence[str],
    cash_bucket: str,
    target_volatility: float,
) -> tuple[pd.Series, float, float, float]:
    """Scale a risky sleeve into cash until annualized portfolio volatility is at target."""

    if target_volatility <= 0.0:
        raise ValueError("target_volatility must be positive.")

    covariance_frame = repair_covariance_matrix(covariance)
    strategic_index = pd.Index([str(bucket) for bucket in strategic_buckets], dtype=object)
    if str(cash_bucket) not in covariance_frame.columns:
        raise KeyError(f"covariance is missing cash bucket '{cash_bucket}'.")
    missing_buckets = [bucket for bucket in strategic_index if bucket not in covariance_frame.columns]
    if missing_buckets:
        raise KeyError(f"covariance is missing strategic buckets: {missing_buckets}.")

    strategic_vector = _coerce_weight_vector(strategic_weights, asset_names=strategic_index)
    if np.any(strategic_vector < 0.0):
        raise ValueError("strategic_weights must be non-negative.")
    if strategic_vector.sum() <= 0.0:
        raise ValueError("strategic_weights must sum to a positive value.")
    strategic_series = pd.Series(
        strategic_vector / strategic_vector.sum(),
        index=strategic_index,
        dtype=float,
    )

    full_index = pd.Index(list(strategic_index) + [str(cash_bucket)], dtype=object)
    risky_full_weights = pd.Series(0.0, index=full_index, dtype=float)
    risky_full_weights.loc[strategic_index] = strategic_series
    cash_only_weights = pd.Series(0.0, index=full_index, dtype=float)
    cash_only_weights.loc[str(cash_bucket)] = 1.0

    full_covariance = covariance_frame.loc[full_index, full_index]
    risky_daily_volatility = portfolio_volatility(full_covariance, risky_full_weights)
    risky_annualized_volatility = float(risky_daily_volatility * np.sqrt(252.0))
    if risky_annualized_volatility <= target_volatility + _MIN_NUMERIC_EPSILON:
        return (
            risky_full_weights,
            1.0,
            float(risky_annualized_volatility),
            float(risky_annualized_volatility),
        )

    low, high = 0.0, 1.0
    for _ in range(60):
        midpoint = (low + high) / 2.0
        candidate_weights = midpoint * risky_full_weights + (1.0 - midpoint) * cash_only_weights
        candidate_volatility = float(portfolio_volatility(full_covariance, candidate_weights) * np.sqrt(252.0))
        if candidate_volatility > target_volatility:
            high = midpoint
        else:
            low = midpoint

    applied_scale = float(low)
    final_weights = applied_scale * risky_full_weights + (1.0 - applied_scale) * cash_only_weights
    final_volatility = float(portfolio_volatility(full_covariance, final_weights) * np.sqrt(252.0))
    return (
        final_weights.astype(float),
        applied_scale,
        float(risky_annualized_volatility),
        float(final_volatility),
    )


def evaluate_risk_budget(
    covariance: pd.DataFrame | np.ndarray,
    weights: pd.Series | Sequence[float] | Mapping[str, float],
    *,
    risk_budgets: pd.Series | Sequence[float] | Mapping[str, float] | None = None,
    iterations: int = 0,
    converged: bool = True,
) -> RiskBudgetResult:
    """Compute volatility, marginal risk, total risk, and risk-share diagnostics."""

    covariance_frame = _coerce_covariance_frame(covariance)
    weight_vector = _coerce_weight_vector(weights, asset_names=covariance_frame.columns)
    resolved_budgets = _coerce_budget_vector(risk_budgets, asset_names=covariance_frame.columns)
    covariance_values = covariance_frame.to_numpy(dtype=float)

    portfolio_sigma = portfolio_volatility(covariance_frame, weight_vector)
    if portfolio_sigma <= 0.0:
        raise ValueError("Portfolio volatility must be positive to compute risk contributions.")

    marginal = covariance_values @ weight_vector / portfolio_sigma
    total = weight_vector * marginal
    shares = total / portfolio_sigma

    asset_index = covariance_frame.columns.copy()
    return RiskBudgetResult(
        weights=pd.Series(weight_vector, index=asset_index, dtype=float),
        risk_budgets=pd.Series(resolved_budgets, index=asset_index, dtype=float),
        portfolio_volatility=float(portfolio_sigma),
        marginal_risk_contributions=pd.Series(marginal, index=asset_index, dtype=float),
        total_risk_contributions=pd.Series(total, index=asset_index, dtype=float),
        risk_shares=pd.Series(shares, index=asset_index, dtype=float),
        iterations=int(iterations),
        converged=bool(converged),
    )


def solve_equal_risk_contribution_weights(
    covariance: pd.DataFrame | np.ndarray,
    *,
    initial_weights: pd.Series | Sequence[float] | Mapping[str, float] | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 1_000,
) -> RiskBudgetResult:
    """Solve for long-only weights whose risk shares are as equal as possible."""

    return solve_risk_budget_weights(
        covariance,
        risk_budgets=None,
        initial_weights=initial_weights,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def solve_risk_budget_weights(
    covariance: pd.DataFrame | np.ndarray,
    *,
    risk_budgets: pd.Series | Sequence[float] | Mapping[str, float] | None = None,
    initial_weights: pd.Series | Sequence[float] | Mapping[str, float] | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 1_000,
) -> RiskBudgetResult:
    """Solve a long-only risk-budget portfolio via coordinate updates."""

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    covariance_frame = repair_covariance_matrix(covariance)
    asset_names = covariance_frame.columns
    budgets = _coerce_budget_vector(risk_budgets, asset_names=asset_names)
    covariance_values = covariance_frame.to_numpy(dtype=float)

    if initial_weights is None:
        initial = budgets / np.sqrt(np.clip(np.diag(covariance_values), _MIN_NUMERIC_EPSILON, None))
    else:
        initial = _coerce_weight_vector(initial_weights, asset_names=asset_names)
    if np.any(initial <= 0.0):
        raise ValueError("initial_weights must be strictly positive for risk-budget solving.")

    x = np.asarray(initial, dtype=float)
    x = x / x.sum()

    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        previous_weights = x / x.sum()
        sigma_x = covariance_values @ x
        for position in range(len(asset_names)):
            diagonal = max(covariance_values[position, position], _MIN_NUMERIC_EPSILON)
            cross_term = sigma_x[position] - diagonal * x[position]
            discriminant = cross_term * cross_term + 4.0 * diagonal * budgets[position]
            x[position] = max(
                (-cross_term + np.sqrt(max(discriminant, 0.0))) / (2.0 * diagonal),
                _MIN_NUMERIC_EPSILON,
            )
            sigma_x = covariance_values @ x

        current_weights = x / x.sum()
        result = evaluate_risk_budget(
            covariance_frame,
            current_weights,
            risk_budgets=budgets,
            iterations=iteration,
            converged=False,
        )
        share_error = float(np.max(np.abs(result.risk_shares.to_numpy(dtype=float) - budgets)))
        weight_step = float(np.max(np.abs(current_weights - previous_weights)))
        if share_error <= tolerance and weight_step <= tolerance:
            converged = True
            break

    final_weights = x / x.sum()
    return evaluate_risk_budget(
        covariance_frame,
        final_weights,
        risk_budgets=budgets,
        iterations=iteration,
        converged=converged,
    )


def build_full_portfolio_weights(
    strategic_weights: pd.Series | Sequence[float] | Mapping[str, float],
    *,
    config: RiskBudgetPolicyConfig,
) -> pd.Series:
    """Scale strategic weights and append reserve-capital sleeves."""

    strategic_index = pd.Index(config.strategic_buckets, dtype=object)
    strategic_vector = _coerce_weight_vector(strategic_weights, asset_names=strategic_index)
    if np.any(strategic_vector < 0.0):
        raise ValueError("strategic_weights must be non-negative.")
    if strategic_vector.sum() <= 0.0:
        raise ValueError("strategic_weights must sum to a positive value.")

    normalized_strategic = strategic_vector / strategic_vector.sum()
    scaled_strategic = normalized_strategic * config.strategic_capital_fraction
    strategic_series = pd.Series(scaled_strategic, index=strategic_index, dtype=float)
    reserve_series = config.reserve_capital_series()

    combined_index = list(config.strategic_buckets) + [
        bucket_name for bucket_name in config.reserve_buckets if bucket_name not in config.strategic_buckets
    ]
    combined = pd.concat([strategic_series, reserve_series]).reindex(combined_index, fill_value=0.0)
    return combined.astype(float)


def build_equity_duration_shifted_policy_config(
    base_config: RiskBudgetPolicyConfig,
    *,
    name: str,
    equity_duration_shift: float,
) -> RiskBudgetPolicyConfig:
    """Shift risk budget between total equity and duration while keeping other sleeves fixed."""

    base_budgets = base_config.normalized_risk_budgets()
    equity_total = float(base_budgets["equity_us"] + base_budgets["equity_ex_us"])
    duration_budget = float(base_budgets["duration"])
    equity_us_share = float(base_budgets["equity_us"] / equity_total)
    equity_ex_us_share = float(base_budgets["equity_ex_us"] / equity_total)

    shifted_equity_total = equity_total + float(equity_duration_shift)
    shifted_duration_budget = duration_budget - float(equity_duration_shift)
    if shifted_equity_total <= 0.0:
        raise ValueError("equity_duration_shift pushes total equity risk budget non-positive.")
    if shifted_duration_budget <= 0.0:
        raise ValueError("equity_duration_shift pushes duration risk budget non-positive.")

    shifted_budgets = base_budgets.copy()
    shifted_budgets.loc["equity_us"] = shifted_equity_total * equity_us_share
    shifted_budgets.loc["equity_ex_us"] = shifted_equity_total * equity_ex_us_share
    shifted_budgets.loc["duration"] = shifted_duration_budget

    if not np.isclose(float(shifted_budgets.sum()), 1.0, atol=1e-10):
        raise ValueError("Shifted strategic risk budgets must still sum to one.")

    return RiskBudgetPolicyConfig(
        name=name,
        strategic_buckets=base_config.strategic_buckets,
        reserve_buckets=base_config.reserve_buckets,
        risk_budgets=shifted_budgets.to_dict(),
        reserve_capital_weights=base_config.reserve_capital_series().to_dict(),
    )


def solve_configured_risk_budget_allocation(
    covariance: pd.DataFrame | np.ndarray,
    *,
    config: RiskBudgetPolicyConfig,
    tolerance: float = 1e-8,
    max_iterations: int = 1_000,
) -> ConfiguredRiskBudgetAllocation:
    """Solve strategic risk budgets, then append any fixed reserve-capital sleeves."""

    covariance_frame = repair_covariance_matrix(covariance)
    required_buckets = set(config.strategic_buckets).union(config.reserve_buckets)
    missing_buckets = [bucket for bucket in required_buckets if bucket not in covariance_frame.columns]
    if missing_buckets:
        raise KeyError(f"covariance is missing required buckets for config '{config.name}': {missing_buckets}.")

    strategic_covariance = covariance_frame.loc[
        list(config.strategic_buckets),
        list(config.strategic_buckets),
    ]
    strategic_result = solve_risk_budget_weights(
        strategic_covariance,
        risk_budgets=config.normalized_risk_budgets(),
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    full_weights = build_full_portfolio_weights(strategic_result.weights, config=config)
    return ConfiguredRiskBudgetAllocation(
        config=config,
        strategic_result=strategic_result,
        full_weights=full_weights,
        reserve_capital_weights=config.reserve_capital_series(),
    )


def generate_rolling_configured_allocations(
    bucket_returns: pd.DataFrame,
    *,
    covariance_config: CovarianceConfig | None = None,
    policy_config: RiskBudgetPolicyConfig = DEFAULT_ERC_POLICY_CONFIG,
    rolling_config: RollingAllocationConfig | None = None,
) -> RollingAllocationResult:
    """Generate a sparse rebalance schedule from rolling covariance estimates."""

    resolved_covariance_config = covariance_config or CovarianceConfig()
    resolved_rolling_config = rolling_config or RollingAllocationConfig()
    cleaned_returns = _normalize_numeric_frame(bucket_returns, frame_name="bucket_returns").dropna(how="any")
    if cleaned_returns.empty:
        raise ValueError("bucket_returns must contain at least one fully aligned row.")

    required_buckets = set(policy_config.strategic_buckets).union(policy_config.reserve_buckets)
    missing_buckets = [bucket for bucket in required_buckets if bucket not in cleaned_returns.columns]
    if missing_buckets:
        raise KeyError(f"bucket_returns is missing required buckets for config '{policy_config.name}': {missing_buckets}.")

    min_history = resolved_rolling_config.min_history or max(
        resolved_covariance_config.long_lookback,
        resolved_covariance_config.short_lookback,
    )
    if len(cleaned_returns) < min_history + resolved_rolling_config.effective_lag:
        raise ValueError(
            "Not enough history to build at least one effective rolling allocation."
        )

    first_decision_position = min_history - 1
    if resolved_rolling_config.start_date is not None:
        start_timestamp = _normalize_timestamp(resolved_rolling_config.start_date)
        if start_timestamp is not None:
            first_decision_position = max(
                first_decision_position,
                int(cleaned_returns.index.searchsorted(start_timestamp, side="left")),
            )

    schedule_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    last_effective_position = len(cleaned_returns) - resolved_rolling_config.effective_lag
    for decision_position in range(
        first_decision_position,
        last_effective_position,
        resolved_rolling_config.rebalance_frequency,
    ):
        decision_date = cleaned_returns.index[decision_position]
        effective_date = cleaned_returns.index[decision_position + resolved_rolling_config.effective_lag]
        covariance_estimate = estimate_blended_covariance(
            cleaned_returns.iloc[: decision_position + 1],
            config=resolved_covariance_config,
        )
        allocation = solve_configured_risk_budget_allocation(
            covariance_estimate.covariance,
            config=policy_config,
        )
        schedule_rows.append(
            {
                "effective_date": effective_date,
                **allocation.full_weights.to_dict(),
            }
        )
        diagnostics_rows.append(
            {
                "decision_date": decision_date,
                "effective_date": effective_date,
                "config_name": policy_config.name,
                "strategic_portfolio_volatility": allocation.strategic_result.portfolio_volatility,
                "iterations": allocation.strategic_result.iterations,
                "converged": allocation.strategic_result.converged,
                "observation_count": covariance_estimate.observation_count,
                "reserve_capital_fraction": float(allocation.reserve_capital_weights.sum()),
                **{
                    f"risk_share_{bucket_name}": float(risk_share)
                    for bucket_name, risk_share in allocation.strategic_result.risk_shares.items()
                },
            }
        )

    if not schedule_rows:
        raise ValueError("No rolling allocations were generated from the provided history.")

    weight_schedule = (
        pd.DataFrame(schedule_rows)
        .assign(effective_date=lambda frame: pd.to_datetime(frame["effective_date"], utc=False))
        .set_index("effective_date")
        .sort_index()
    )
    diagnostics = (
        pd.DataFrame(diagnostics_rows)
        .assign(
            decision_date=lambda frame: pd.to_datetime(frame["decision_date"], utc=False),
            effective_date=lambda frame: pd.to_datetime(frame["effective_date"], utc=False),
        )
        .sort_values("effective_date")
        .reset_index(drop=True)
    )
    return RollingAllocationResult(
        weight_schedule=weight_schedule.astype(float),
        diagnostics=diagnostics,
    )


def generate_state_conditioned_rolling_allocations(
    bucket_returns: pd.DataFrame,
    *,
    state_by_date: pd.Series | Mapping[pd.Timestamp | str, str],
    state_to_policy_config: Mapping[str, RiskBudgetPolicyConfig],
    default_state: str,
    covariance_config: CovarianceConfig | None = None,
    rolling_config: RollingAllocationConfig | None = None,
) -> RollingAllocationResult:
    """Generate a sparse rebalance schedule that switches policy configs by lagged state."""

    resolved_covariance_config = covariance_config or CovarianceConfig()
    resolved_rolling_config = rolling_config or RollingAllocationConfig()
    cleaned_returns = _normalize_numeric_frame(bucket_returns, frame_name="bucket_returns").dropna(how="any")
    if cleaned_returns.empty:
        raise ValueError("bucket_returns must contain at least one fully aligned row.")
    if default_state not in state_to_policy_config:
        raise KeyError(f"default_state '{default_state}' is missing from state_to_policy_config.")

    normalized_state = pd.Series(state_by_date, dtype="object")
    normalized_state.index = pd.to_datetime(normalized_state.index, utc=False)
    normalized_state = normalized_state.sort_index()
    if normalized_state.empty:
        raise ValueError("state_by_date must contain at least one state observation.")

    required_buckets: set[str] = set()
    for policy_config in state_to_policy_config.values():
        required_buckets.update(policy_config.strategic_buckets)
        required_buckets.update(policy_config.reserve_buckets)
    missing_buckets = [bucket for bucket in sorted(required_buckets) if bucket not in cleaned_returns.columns]
    if missing_buckets:
        raise KeyError(f"bucket_returns is missing required buckets for state-conditioned configs: {missing_buckets}.")

    min_history = resolved_rolling_config.min_history or max(
        resolved_covariance_config.long_lookback,
        resolved_covariance_config.short_lookback,
    )
    if len(cleaned_returns) < min_history + resolved_rolling_config.effective_lag:
        raise ValueError("Not enough history to build at least one effective rolling allocation.")

    first_decision_position = min_history - 1
    if resolved_rolling_config.start_date is not None:
        start_timestamp = _normalize_timestamp(resolved_rolling_config.start_date)
        if start_timestamp is not None:
            first_decision_position = max(
                first_decision_position,
                int(cleaned_returns.index.searchsorted(start_timestamp, side="left")),
            )

    schedule_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    last_effective_position = len(cleaned_returns) - resolved_rolling_config.effective_lag
    for decision_position in range(
        first_decision_position,
        last_effective_position,
        resolved_rolling_config.rebalance_frequency,
    ):
        decision_date = cleaned_returns.index[decision_position]
        effective_date = cleaned_returns.index[decision_position + resolved_rolling_config.effective_lag]
        lagged_states = normalized_state.loc[normalized_state.index <= decision_date]
        if lagged_states.empty:
            state_name = default_state
        else:
            state_name = str(lagged_states.iloc[-1])
        policy_config = state_to_policy_config.get(state_name, state_to_policy_config[default_state])

        covariance_estimate = estimate_blended_covariance(
            cleaned_returns.iloc[: decision_position + 1],
            config=resolved_covariance_config,
        )
        allocation = solve_configured_risk_budget_allocation(
            covariance_estimate.covariance,
            config=policy_config,
        )
        schedule_rows.append(
            {
                "effective_date": effective_date,
                **allocation.full_weights.to_dict(),
            }
        )
        diagnostics_rows.append(
            {
                "decision_date": decision_date,
                "effective_date": effective_date,
                "state_name": state_name,
                "config_name": policy_config.name,
                "strategic_portfolio_volatility": allocation.strategic_result.portfolio_volatility,
                "iterations": allocation.strategic_result.iterations,
                "converged": allocation.strategic_result.converged,
                "observation_count": covariance_estimate.observation_count,
                "reserve_capital_fraction": float(allocation.reserve_capital_weights.sum()),
                **{
                    f"risk_share_{bucket_name}": float(risk_share)
                    for bucket_name, risk_share in allocation.strategic_result.risk_shares.items()
                },
            }
        )

    if not schedule_rows:
        raise ValueError("No rolling allocations were generated from the provided history.")

    weight_schedule = (
        pd.DataFrame(schedule_rows)
        .assign(effective_date=lambda frame: pd.to_datetime(frame["effective_date"], utc=False))
        .set_index("effective_date")
        .sort_index()
    )
    diagnostics = (
        pd.DataFrame(diagnostics_rows)
        .assign(
            decision_date=lambda frame: pd.to_datetime(frame["decision_date"], utc=False),
            effective_date=lambda frame: pd.to_datetime(frame["effective_date"], utc=False),
        )
        .sort_values("effective_date")
        .reset_index(drop=True)
    )
    return RollingAllocationResult(
        weight_schedule=weight_schedule.astype(float),
        diagnostics=diagnostics,
    )


def backtest_weight_schedule(
    bucket_returns: pd.DataFrame,
    weight_schedule: pd.DataFrame,
    *,
    benchmark_column: str = "equity_us",
    cost_bps_per_side: float = 0.0,
) -> MultiAssetBacktestResult:
    """Backtest a sparse effective-date weight schedule on daily bucket returns."""

    if cost_bps_per_side < 0.0:
        raise ValueError("cost_bps_per_side must be non-negative.")

    cleaned_returns = _normalize_numeric_frame(bucket_returns, frame_name="bucket_returns").dropna(how="any")
    cleaned_schedule = _normalize_numeric_frame(weight_schedule, frame_name="weight_schedule")
    if benchmark_column not in cleaned_returns.columns:
        raise KeyError(f"bucket_returns is missing benchmark column '{benchmark_column}'.")
    missing_weight_columns = [column for column in cleaned_schedule.columns if column not in cleaned_returns.columns]
    if missing_weight_columns:
        raise KeyError(f"bucket_returns is missing weight columns: {missing_weight_columns}.")

    cleaned_schedule = cleaned_schedule.loc[~cleaned_schedule.index.duplicated(keep="last")]
    active_dates = cleaned_returns.index[cleaned_returns.index >= cleaned_schedule.index.min()]
    if active_dates.empty:
        raise ValueError("No bucket return rows remain on or after the first effective allocation date.")

    schedule_lookup = {
        timestamp: cleaned_schedule.loc[timestamp].astype(float)
        for timestamp in cleaned_schedule.index
    }
    current_post_weights = pd.Series(0.0, index=cleaned_schedule.columns, dtype=float)
    active_decision_date: pd.Timestamp | None = None
    records: list[dict[str, object]] = []

    for current_date in active_dates:
        current_returns = cleaned_returns.loc[current_date, cleaned_schedule.columns].astype(float)
        if current_date in schedule_lookup:
            applied_weights = schedule_lookup[current_date].astype(float)
            turnover = float((applied_weights - current_post_weights).abs().sum())
            rebalanced = True
            active_decision_date = current_date
        else:
            applied_weights = current_post_weights.copy()
            turnover = 0.0
            rebalanced = False

        gross_return = float((applied_weights * current_returns).sum())
        cost = turnover * cost_bps_per_side / 10000.0
        net_return = gross_return - cost
        benchmark_return = float(cleaned_returns.loc[current_date, benchmark_column])

        gross_growth_vector = applied_weights * (1.0 + current_returns)
        gross_portfolio_growth = 1.0 + gross_return
        if gross_portfolio_growth <= 0.0:
            current_post_weights = pd.Series(0.0, index=applied_weights.index, dtype=float)
        else:
            current_post_weights = (gross_growth_vector / gross_portfolio_growth).astype(float)

        record = {
            "signal_date": active_decision_date.isoformat() if active_decision_date is not None else None,
            "entry_date": current_date.isoformat(),
            "exit_date": current_date.isoformat(),
            "gross_return": gross_return,
            "net_return": net_return,
            "benchmark_return": benchmark_return,
            "turnover": turnover,
            "cost_bps": cost * 10000.0,
            "positions": int((applied_weights > _MIN_NUMERIC_EPSILON).sum()),
            "gross_exposure": float(applied_weights.sum()),
            "rebalanced": rebalanced,
            "applied_weights": applied_weights.to_dict(),
            "post_return_weights": current_post_weights.to_dict(),
        }
        records.append(record)

    record_frame = pd.DataFrame(records)
    summary = summarize_multi_asset_backtest_records(record_frame, horizon=1)
    return MultiAssetBacktestResult(
        records=record_frame,
        summary=summary,
        weight_schedule=cleaned_schedule,
    )


def build_threshold_aware_weight_schedule(
    bucket_returns: pd.DataFrame,
    target_weight_schedule: pd.DataFrame,
    *,
    config: ThresholdRebalanceConfig,
) -> ThresholdRebalanceResult:
    """Execute a target schedule only when drift bands or stale caps require it."""

    cleaned_returns = _normalize_numeric_frame(bucket_returns, frame_name="bucket_returns").dropna(how="any")
    cleaned_schedule = _normalize_numeric_frame(
        target_weight_schedule,
        frame_name="target_weight_schedule",
    )
    missing_weight_columns = [column for column in cleaned_schedule.columns if column not in cleaned_returns.columns]
    if missing_weight_columns:
        raise KeyError(f"bucket_returns is missing weight columns: {missing_weight_columns}.")

    cleaned_schedule = cleaned_schedule.loc[~cleaned_schedule.index.duplicated(keep="last")]
    active_dates = cleaned_returns.index[cleaned_returns.index >= cleaned_schedule.index.min()]
    if active_dates.empty:
        raise ValueError("No bucket return rows remain on or after the first target allocation date.")

    target_lookup = {
        timestamp: cleaned_schedule.loc[timestamp].astype(float)
        for timestamp in cleaned_schedule.index
    }
    current_target_weights: pd.Series | None = None
    current_post_weights = pd.Series(0.0, index=cleaned_schedule.columns, dtype=float)
    last_rebalance_position: int | None = None
    last_target_effective_date: pd.Timestamp | None = None
    schedule_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []

    for current_position, current_date in enumerate(active_dates):
        if current_date in target_lookup:
            current_target_weights = target_lookup[current_date].astype(float)
            last_target_effective_date = current_date
        if current_target_weights is None:
            continue

        max_abs_drift = float((current_target_weights - current_post_weights).abs().max())
        total_abs_drift = float((current_target_weights - current_post_weights).abs().sum())
        turnover_to_target = total_abs_drift
        target_updated_today = current_date in target_lookup
        sessions_since_last_rebalance = (
            None if last_rebalance_position is None else int(current_position - last_rebalance_position)
        )
        initial_rebalance = last_rebalance_position is None
        drift_triggered = max_abs_drift >= config.drift_threshold - _MIN_NUMERIC_EPSILON
        stale_triggered = (
            config.stale_time_cap is not None
            and sessions_since_last_rebalance is not None
            and sessions_since_last_rebalance >= config.stale_time_cap
        )
        should_rebalance = bool(initial_rebalance or drift_triggered or stale_triggered)

        if should_rebalance:
            applied_weights = current_target_weights.astype(float)
            schedule_rows.append(
                {
                    "effective_date": current_date,
                    **applied_weights.to_dict(),
                }
            )
            last_rebalance_position = current_position
        else:
            applied_weights = current_post_weights.copy()

        current_returns = cleaned_returns.loc[current_date, cleaned_schedule.columns].astype(float)
        gross_return = float((applied_weights * current_returns).sum())
        gross_growth_vector = applied_weights * (1.0 + current_returns)
        gross_portfolio_growth = 1.0 + gross_return
        if gross_portfolio_growth <= 0.0:
            current_post_weights = pd.Series(0.0, index=applied_weights.index, dtype=float)
        else:
            current_post_weights = (gross_growth_vector / gross_portfolio_growth).astype(float)

        diagnostics_rows.append(
            {
                "date": current_date,
                "target_effective_date": last_target_effective_date,
                "target_updated_today": bool(target_updated_today),
                "rebalanced": bool(should_rebalance),
                "initial_rebalance": bool(initial_rebalance),
                "drift_threshold": float(config.drift_threshold),
                "max_abs_drift": max_abs_drift,
                "total_abs_drift": total_abs_drift,
                "turnover_to_target": turnover_to_target,
                "sessions_since_last_rebalance": sessions_since_last_rebalance,
                "stale_time_cap": config.stale_time_cap,
                "drift_triggered": bool(drift_triggered),
                "stale_triggered": bool(stale_triggered),
            }
        )

    if not schedule_rows:
        raise ValueError("Threshold-aware rebalance logic produced no executed allocation dates.")

    executed_schedule = (
        pd.DataFrame(schedule_rows)
        .assign(effective_date=lambda frame: pd.to_datetime(frame["effective_date"], utc=False))
        .set_index("effective_date")
        .sort_index()
    )
    diagnostics = pd.DataFrame(diagnostics_rows)
    diagnostics["date"] = pd.to_datetime(diagnostics["date"], utc=False)
    diagnostics["target_effective_date"] = pd.to_datetime(diagnostics["target_effective_date"], utc=False)
    diagnostics = diagnostics.sort_values("date").reset_index(drop=True)
    return ThresholdRebalanceResult(
        weight_schedule=executed_schedule.astype(float),
        diagnostics=diagnostics,
    )


def run_threshold_aware_weight_schedule_backtest(
    bucket_returns: pd.DataFrame,
    target_weight_schedule: pd.DataFrame,
    *,
    config: ThresholdRebalanceConfig,
    benchmark_column: str = "equity_us",
    cost_bps_per_side: float = 0.0,
) -> tuple[ThresholdRebalanceResult, MultiAssetBacktestResult]:
    """Filter a target schedule through rebalance bands, then backtest the executed path."""

    threshold_result = build_threshold_aware_weight_schedule(
        bucket_returns,
        target_weight_schedule,
        config=config,
    )
    backtest_result = backtest_weight_schedule(
        bucket_returns,
        threshold_result.weight_schedule,
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )
    return threshold_result, backtest_result


def run_rolling_risk_budget_backtest(
    bucket_returns: pd.DataFrame,
    *,
    covariance_config: CovarianceConfig | None = None,
    policy_config: RiskBudgetPolicyConfig = DEFAULT_ERC_POLICY_CONFIG,
    rolling_config: RollingAllocationConfig | None = None,
    benchmark_column: str = "equity_us",
    cost_bps_per_side: float = 0.0,
) -> tuple[RollingAllocationResult, MultiAssetBacktestResult]:
    """Convenience wrapper for rolling allocation generation plus schedule backtest."""

    rolling_allocations = generate_rolling_configured_allocations(
        bucket_returns,
        covariance_config=covariance_config,
        policy_config=policy_config,
        rolling_config=rolling_config,
    )
    backtest_result = backtest_weight_schedule(
        bucket_returns,
        rolling_allocations.weight_schedule,
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )
    return rolling_allocations, backtest_result


def run_state_conditioned_rolling_risk_budget_backtest(
    bucket_returns: pd.DataFrame,
    *,
    state_by_date: pd.Series | Mapping[pd.Timestamp | str, str],
    state_to_policy_config: Mapping[str, RiskBudgetPolicyConfig],
    default_state: str,
    covariance_config: CovarianceConfig | None = None,
    rolling_config: RollingAllocationConfig | None = None,
    benchmark_column: str = "equity_us",
    cost_bps_per_side: float = 0.0,
) -> tuple[RollingAllocationResult, MultiAssetBacktestResult]:
    """Run a rolling risk-budget strategy that switches policy config by lagged state."""

    rolling_allocations = generate_state_conditioned_rolling_allocations(
        bucket_returns,
        state_by_date=state_by_date,
        state_to_policy_config=state_to_policy_config,
        default_state=default_state,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
    )
    backtest_result = backtest_weight_schedule(
        bucket_returns,
        rolling_allocations.weight_schedule,
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )
    return rolling_allocations, backtest_result


def generate_rolling_sharpe_target_vol_allocations(
    bucket_returns: pd.DataFrame,
    *,
    covariance_config: CovarianceConfig | None = None,
    rolling_config: RollingAllocationConfig | None = None,
    target_vol_config: SharpeTargetVolConfig,
) -> RollingAllocationResult:
    """Generate rolling allocations by maximizing trailing Sharpe-score efficiency."""

    resolved_covariance_config = covariance_config or CovarianceConfig()
    resolved_rolling_config = rolling_config or RollingAllocationConfig()
    cleaned_returns = _normalize_numeric_frame(bucket_returns, frame_name="bucket_returns").dropna(how="any")
    if cleaned_returns.empty:
        raise ValueError("bucket_returns must contain at least one fully aligned row.")

    required_buckets = set(target_vol_config.strategic_buckets).union({target_vol_config.cash_bucket})
    missing_buckets = [bucket for bucket in sorted(required_buckets) if bucket not in cleaned_returns.columns]
    if missing_buckets:
        raise KeyError(
            f"bucket_returns is missing required buckets for target-vol config: {missing_buckets}."
        )

    min_history = max(
        resolved_rolling_config.min_history or 2,
        resolved_covariance_config.long_lookback,
        resolved_covariance_config.short_lookback,
        target_vol_config.score_lookback,
    )
    if len(cleaned_returns) < min_history + resolved_rolling_config.effective_lag:
        raise ValueError("Not enough history to build at least one effective rolling allocation.")

    first_decision_position = min_history - 1
    if resolved_rolling_config.start_date is not None:
        start_timestamp = _normalize_timestamp(resolved_rolling_config.start_date)
        if start_timestamp is not None:
            first_decision_position = max(
                first_decision_position,
                int(cleaned_returns.index.searchsorted(start_timestamp, side="left")),
            )

    schedule_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    last_effective_position = len(cleaned_returns) - resolved_rolling_config.effective_lag
    strategic_buckets = list(target_vol_config.strategic_buckets)
    full_bucket_names = strategic_buckets + [target_vol_config.cash_bucket]
    for decision_position in range(
        first_decision_position,
        last_effective_position,
        resolved_rolling_config.rebalance_frequency,
    ):
        decision_date = cleaned_returns.index[decision_position]
        effective_date = cleaned_returns.index[decision_position + resolved_rolling_config.effective_lag]
        history = cleaned_returns.iloc[: decision_position + 1]
        covariance_estimate = estimate_blended_covariance(
            history.loc[:, full_bucket_names],
            config=resolved_covariance_config,
        )
        sharpe_scores, score_summary = estimate_trailing_asset_sharpe_scores(
            history.loc[:, strategic_buckets],
            asset_names=strategic_buckets,
            lookback=target_vol_config.score_lookback,
        )
        strategic_weights = solve_long_only_sharpe_score_weights(
            covariance_estimate.covariance.loc[strategic_buckets, strategic_buckets],
            sharpe_scores=sharpe_scores,
        )
        full_weights, risk_scale, risky_volatility, final_volatility = allocate_vol_capped_portfolio_into_cash(
            strategic_weights,
            covariance=covariance_estimate.covariance.loc[full_bucket_names, full_bucket_names],
            strategic_buckets=strategic_buckets,
            cash_bucket=target_vol_config.cash_bucket,
            target_volatility=target_vol_config.target_volatility,
        )

        schedule_rows.append(
            {
                "effective_date": effective_date,
                **full_weights.to_dict(),
            }
        )
        diagnostics_rows.append(
            {
                "decision_date": decision_date,
                "effective_date": effective_date,
                "target_volatility": float(target_vol_config.target_volatility),
                "score_lookback": int(target_vol_config.score_lookback),
                "observation_count": covariance_estimate.observation_count,
                "risk_scale": float(risk_scale),
                "risky_portfolio_volatility": float(risky_volatility),
                "final_portfolio_volatility": float(final_volatility),
                "cash_weight": float(full_weights.loc[target_vol_config.cash_bucket]),
                **{
                    f"score_{bucket_name}": float(sharpe_scores.loc[bucket_name])
                    for bucket_name in strategic_buckets
                },
                **{
                    f"strategic_weight_{bucket_name}": float(strategic_weights.loc[bucket_name])
                    for bucket_name in strategic_buckets
                },
                **{
                    f"annualized_return_{row.asset}": float(row.annualized_return)
                    for row in score_summary.itertuples(index=False)
                },
                **{
                    f"annualized_volatility_{row.asset}": float(row.annualized_volatility)
                    for row in score_summary.itertuples(index=False)
                },
            }
        )

    if not schedule_rows:
        raise ValueError("No rolling allocations were generated from the provided history.")

    weight_schedule = (
        pd.DataFrame(schedule_rows)
        .assign(effective_date=lambda frame: pd.to_datetime(frame["effective_date"], utc=False))
        .set_index("effective_date")
        .sort_index()
    )
    diagnostics = (
        pd.DataFrame(diagnostics_rows)
        .assign(
            decision_date=lambda frame: pd.to_datetime(frame["decision_date"], utc=False),
            effective_date=lambda frame: pd.to_datetime(frame["effective_date"], utc=False),
        )
        .sort_values("effective_date")
        .reset_index(drop=True)
    )
    return RollingAllocationResult(
        weight_schedule=weight_schedule.astype(float),
        diagnostics=diagnostics,
    )


def run_rolling_sharpe_target_vol_backtest(
    bucket_returns: pd.DataFrame,
    *,
    covariance_config: CovarianceConfig | None = None,
    rolling_config: RollingAllocationConfig | None = None,
    target_vol_config: SharpeTargetVolConfig,
    benchmark_column: str = "equity_us",
    cost_bps_per_side: float = 0.0,
) -> tuple[RollingAllocationResult, MultiAssetBacktestResult]:
    """Run a rolling long-only Sharpe-score allocator with a hard volatility cap."""

    rolling_allocations = generate_rolling_sharpe_target_vol_allocations(
        bucket_returns,
        covariance_config=covariance_config,
        rolling_config=rolling_config,
        target_vol_config=target_vol_config,
    )
    backtest_result = backtest_weight_schedule(
        bucket_returns,
        rolling_allocations.weight_schedule,
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )
    return rolling_allocations, backtest_result


def build_constant_weight_schedule(
    returns_index: pd.Index | Sequence[pd.Timestamp],
    weights: pd.Series | Sequence[float] | Mapping[str, float],
    *,
    start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a one-row effective-date schedule for a static allocation."""

    index = pd.DatetimeIndex(pd.to_datetime(pd.Index(returns_index), utc=False)).sort_values()
    if index.empty:
        raise ValueError("returns_index must contain at least one timestamp.")
    effective_date = index[0]
    if start_date is not None:
        start_timestamp = _normalize_timestamp(start_date)
        if start_timestamp is not None:
            position = int(index.searchsorted(start_timestamp, side="left"))
            if position >= len(index):
                raise ValueError("start_date falls after the available return history.")
            effective_date = index[position]

    if isinstance(weights, pd.Series):
        weight_series = weights.astype(float).copy()
    elif isinstance(weights, Mapping):
        weight_series = pd.Series({str(key): float(value) for key, value in weights.items()}, dtype=float)
    else:
        vector = np.asarray(tuple(weights), dtype=float)
        labels = [str(position) for position in range(len(vector))]
        weight_series = pd.Series(vector, index=labels, dtype=float)
    if (weight_series < 0.0).any():
        raise ValueError("weights must be non-negative.")
    if weight_series.sum() <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    normalized = weight_series / weight_series.sum()
    return pd.DataFrame([normalized.to_dict()], index=pd.DatetimeIndex([effective_date]))


def build_periodic_weight_schedule(
    returns_index: pd.Index | Sequence[pd.Timestamp],
    weights: pd.Series | Sequence[float] | Mapping[str, float],
    *,
    rebalance_frequency: int,
    start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a repeated effective-date schedule for a static target allocation."""

    if rebalance_frequency <= 0:
        raise ValueError("rebalance_frequency must be a positive integer.")

    initial_schedule = build_constant_weight_schedule(
        returns_index,
        weights,
        start_date=start_date,
    )
    index = pd.DatetimeIndex(pd.to_datetime(pd.Index(returns_index), utc=False)).sort_values()
    initial_date = pd.Timestamp(initial_schedule.index[0])
    start_position = int(index.get_indexer([initial_date])[0])
    if start_position < 0:
        raise ValueError("Initial effective date is missing from the return index.")

    effective_dates = index[start_position::rebalance_frequency]
    repeated_schedule = pd.DataFrame(
        [initial_schedule.iloc[0].to_dict() for _ in range(len(effective_dates))],
        index=effective_dates,
    )
    return repeated_schedule.astype(float)


def run_constant_weight_backtest(
    bucket_returns: pd.DataFrame,
    *,
    weights: pd.Series | Sequence[float] | Mapping[str, float],
    benchmark_column: str = "equity_us",
    cost_bps_per_side: float = 0.0,
    start_date: str | pd.Timestamp | None = None,
) -> MultiAssetBacktestResult:
    """Backtest a static weight vector on bucket returns."""

    weight_schedule = build_constant_weight_schedule(
        bucket_returns.index,
        weights,
        start_date=start_date,
    )
    return backtest_weight_schedule(
        bucket_returns,
        weight_schedule,
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )


def run_periodic_constant_weight_backtest(
    bucket_returns: pd.DataFrame,
    *,
    weights: pd.Series | Sequence[float] | Mapping[str, float],
    rebalance_frequency: int,
    benchmark_column: str = "equity_us",
    cost_bps_per_side: float = 0.0,
    start_date: str | pd.Timestamp | None = None,
) -> MultiAssetBacktestResult:
    """Backtest a static target vector that is periodically reset to target weights."""

    weight_schedule = build_periodic_weight_schedule(
        bucket_returns.index,
        weights,
        rebalance_frequency=rebalance_frequency,
        start_date=start_date,
    )
    return backtest_weight_schedule(
        bucket_returns,
        weight_schedule,
        benchmark_column=benchmark_column,
        cost_bps_per_side=cost_bps_per_side,
    )


def build_calendar_year_return_summary(
    records: pd.DataFrame,
    *,
    strategy_name: str,
    date_column: str = "entry_date",
    return_column: str = "net_return",
) -> pd.DataFrame:
    """Summarize one backtest record frame into calendar-year return rows."""

    if records.empty:
        return pd.DataFrame(columns=["strategy_name", "calendar_year", "total_return", "sessions"])

    frame = records.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], utc=False)
    frame["calendar_year"] = frame[date_column].dt.year.astype(int)
    rows: list[dict[str, object]] = []
    for calendar_year, group in frame.groupby("calendar_year", sort=True):
        total_return = float((1.0 + group[return_column].astype(float)).prod() - 1.0)
        rows.append(
            {
                "strategy_name": strategy_name,
                "calendar_year": int(calendar_year),
                "total_return": total_return,
                "sessions": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def build_weight_summary_frame(
    weight_schedule: pd.DataFrame,
    *,
    strategy_name: str,
) -> pd.DataFrame:
    """Summarize average, min, max, and last weights for a schedule."""

    cleaned = _normalize_numeric_frame(weight_schedule, frame_name="weight_schedule")
    rows: list[dict[str, object]] = []
    for bucket_name in cleaned.columns:
        series = cleaned[bucket_name].astype(float)
        rows.append(
            {
                "strategy_name": strategy_name,
                "bucket": bucket_name,
                "mean_weight": float(series.mean()),
                "min_weight": float(series.min()),
                "max_weight": float(series.max()),
                "last_weight": float(series.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def build_risk_share_summary_frame(
    diagnostics: pd.DataFrame,
    *,
    strategy_name: str,
) -> pd.DataFrame:
    """Summarize rolling risk-share diagnostics for one strategy."""

    if diagnostics.empty:
        return pd.DataFrame(
            columns=["strategy_name", "bucket", "mean_risk_share", "min_risk_share", "max_risk_share", "last_risk_share"]
        )

    risk_share_columns = [column for column in diagnostics.columns if column.startswith("risk_share_")]
    rows: list[dict[str, object]] = []
    for column_name in risk_share_columns:
        bucket_name = column_name.removeprefix("risk_share_")
        series = diagnostics[column_name].astype(float)
        rows.append(
            {
                "strategy_name": strategy_name,
                "bucket": bucket_name,
                "mean_risk_share": float(series.mean()),
                "min_risk_share": float(series.min()),
                "max_risk_share": float(series.max()),
                "last_risk_share": float(series.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def summarize_multi_asset_backtest_records(
    records: pd.DataFrame,
    *,
    horizon: int,
) -> dict[str, float]:
    """Summarize daily multi-asset records using the same headline metrics as other research paths."""

    if records.empty:
        return {
            "sessions": 0,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "benchmark_total_return": np.nan,
            "mean_turnover": np.nan,
            "mean_cost_bps": np.nan,
        }

    frame = records.copy()
    net_returns = frame["net_return"].astype(float)
    benchmark_returns = frame["benchmark_return"].astype(float)
    total_return = float((1.0 + net_returns).prod() - 1.0)
    benchmark_total_return = float((1.0 + benchmark_returns).prod() - 1.0)
    annualized_return = _annualize_total_return(total_return, len(frame), horizon)
    annualized_volatility = float(net_returns.std(ddof=1) * np.sqrt(252 / horizon)) if len(frame) > 1 else np.nan
    sharpe = (
        float((net_returns.mean() / net_returns.std(ddof=1)) * np.sqrt(252 / horizon))
        if len(frame) > 1 and net_returns.std(ddof=1) > 0
        else np.nan
    )
    equity_curve = (1.0 + net_returns).cumprod()
    peaks = equity_curve.cummax()
    max_drawdown = float((equity_curve / peaks - 1.0).min()) if not equity_curve.empty else np.nan
    return {
        "sessions": int(len(frame)),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "benchmark_total_return": benchmark_total_return,
        "mean_turnover": float(frame["turnover"].astype(float).mean()),
        "mean_cost_bps": float(frame["cost_bps"].astype(float).mean()),
    }


def _select_return_window(bucket_returns: pd.DataFrame, *, lookback: int) -> pd.DataFrame:
    normalized = _normalize_numeric_frame(bucket_returns, frame_name="bucket_returns").dropna(how="any")
    if len(normalized) < 2:
        raise ValueError("Need at least 2 aligned observations to estimate covariance.")
    return normalized.tail(int(lookback))


def _normalize_numeric_frame(frame: pd.DataFrame, *, frame_name: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"{frame_name} must not be empty.")
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index, utc=False)
    normalized = normalized.sort_index()
    normalized.columns = pd.Index([str(column) for column in normalized.columns], name=frame.columns.name)
    for column in normalized.columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.astype(float)


def _normalize_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert(None)
    return timestamp


def _annualize_total_return(total_return: float, periods: int, horizon: int) -> float:
    if periods <= 0 or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (252 / (periods * horizon)) - 1.0)


def _coerce_covariance_frame(covariance: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(covariance, pd.DataFrame):
        covariance_frame = covariance.copy()
        covariance_frame.index = pd.Index([str(label) for label in covariance_frame.index], dtype=object)
        covariance_frame.columns = pd.Index([str(label) for label in covariance_frame.columns], dtype=object)
        if tuple(covariance_frame.index) != tuple(covariance_frame.columns):
            raise ValueError("Covariance DataFrame index and columns must match.")
        return covariance_frame.astype(float)

    covariance_array = np.asarray(covariance, dtype=float)
    if covariance_array.ndim != 2 or covariance_array.shape[0] != covariance_array.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    asset_names = pd.Index([str(index) for index in range(covariance_array.shape[0])], dtype=object)
    return pd.DataFrame(covariance_array, index=asset_names, columns=asset_names, dtype=float)


def _coerce_weight_vector(
    weights: pd.Series | Sequence[float] | Mapping[str, float],
    *,
    asset_names: pd.Index,
) -> np.ndarray:
    if isinstance(weights, pd.Series):
        series = weights.astype(float).copy()
        series.index = pd.Index([str(label) for label in series.index], dtype=object)
        if set(series.index) != set(asset_names):
            raise ValueError("Weight labels must match covariance asset names exactly.")
        return series.reindex(asset_names).to_numpy(dtype=float)
    if isinstance(weights, Mapping):
        series = pd.Series({str(key): float(value) for key, value in weights.items()}, dtype=float)
        if set(series.index) != set(asset_names):
            raise ValueError("Weight labels must match covariance asset names exactly.")
        return series.reindex(asset_names).to_numpy(dtype=float)

    vector = np.asarray(tuple(weights), dtype=float)
    if vector.ndim != 1 or len(vector) != len(asset_names):
        raise ValueError("Weight vector length must match covariance dimension.")
    return vector


def _coerce_budget_vector(
    risk_budgets: pd.Series | Sequence[float] | Mapping[str, float] | None,
    *,
    asset_names: pd.Index,
) -> np.ndarray:
    if risk_budgets is None:
        return np.full(len(asset_names), 1.0 / len(asset_names), dtype=float)

    budgets = _coerce_weight_vector(risk_budgets, asset_names=asset_names)
    if np.any(budgets <= 0.0):
        raise ValueError("risk_budgets must be strictly positive.")
    return budgets / budgets.sum()
