from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np
import pandas as pd

REGIME_LABELS: tuple[str, ...] = ("warmup", "bull", "correction", "bear", "rebound")
FORWARD_REGIME_LABELS: tuple[str, ...] = ("bull", "correction", "bear", "rebound", "unlabeled")


class RegimeDetector(Protocol):
    def label_frame(
        self,
        frame: pd.DataFrame,
        *,
        return_column: str = "benchmark_return",
        date_column: str = "entry_date",
    ) -> pd.DataFrame:
        ...


@dataclass(slots=True, frozen=True)
class BenchmarkTrendDrawdownRegimeDetector:
    """Classify market state using only lagged benchmark returns."""

    trend_lookback_windows: int = 12
    drawdown_threshold: float = -0.10

    def label_frame(
        self,
        frame: pd.DataFrame,
        *,
        return_column: str = "benchmark_return",
        date_column: str = "entry_date",
    ) -> pd.DataFrame:
        if return_column not in frame.columns:
            raise KeyError(f"Missing return column: {return_column}")
        if date_column not in frame.columns:
            raise KeyError(f"Missing date column: {date_column}")
        if self.trend_lookback_windows <= 0:
            raise ValueError("trend_lookback_windows must be positive.")
        if self.drawdown_threshold >= 0.0:
            raise ValueError("drawdown_threshold must be negative.")

        returns = pd.Series(frame[return_column], dtype=float).reset_index(drop=True)
        dates = pd.to_datetime(frame[date_column], errors="coerce").reset_index(drop=True)

        lagged_equity = (1.0 + returns).cumprod().shift(1).fillna(1.0)
        lagged_peaks = lagged_equity.cummax()
        lagged_drawdown = lagged_equity / lagged_peaks - 1.0

        trailing_return = (
            (1.0 + returns)
            .rolling(self.trend_lookback_windows, min_periods=self.trend_lookback_windows)
            .apply(np.prod, raw=True)
            .sub(1.0)
            .shift(1)
        )

        history_windows = pd.Series(np.arange(len(frame)), dtype=int)
        enough_history = history_windows >= self.trend_lookback_windows
        deep_drawdown = lagged_drawdown <= float(self.drawdown_threshold)
        positive_trend = trailing_return > 0.0

        labels = np.full(len(frame), "warmup", dtype=object)
        labels[enough_history & ~deep_drawdown & positive_trend] = "bull"
        labels[enough_history & ~deep_drawdown & ~positive_trend] = "correction"
        labels[enough_history & deep_drawdown & ~positive_trend] = "bear"
        labels[enough_history & deep_drawdown & positive_trend] = "rebound"

        return pd.DataFrame(
            {
                date_column: dates,
                "regime_label": labels,
                "regime_drawdown": lagged_drawdown.astype(float),
                "regime_trailing_return": trailing_return.astype(float),
                "regime_history_windows": history_windows.astype(int),
            }
        )


@dataclass(slots=True, frozen=True)
class BenchmarkTrendDrawdownVolRegimeDetector:
    """White-box regime detector that adds lagged realized volatility."""

    trend_lookback_windows: int = 12
    drawdown_threshold: float = -0.10
    vol_lookback_windows: int = 12
    high_vol_annualized_threshold: float = 0.18
    horizon_sessions: int = 5

    def label_frame(
        self,
        frame: pd.DataFrame,
        *,
        return_column: str = "benchmark_return",
        date_column: str = "entry_date",
    ) -> pd.DataFrame:
        if return_column not in frame.columns:
            raise KeyError(f"Missing return column: {return_column}")
        if date_column not in frame.columns:
            raise KeyError(f"Missing date column: {date_column}")
        if self.trend_lookback_windows <= 0:
            raise ValueError("trend_lookback_windows must be positive.")
        if self.vol_lookback_windows <= 1:
            raise ValueError("vol_lookback_windows must be greater than 1.")
        if self.drawdown_threshold >= 0.0:
            raise ValueError("drawdown_threshold must be negative.")
        if self.high_vol_annualized_threshold <= 0.0:
            raise ValueError("high_vol_annualized_threshold must be positive.")
        if self.horizon_sessions <= 0:
            raise ValueError("horizon_sessions must be positive.")

        returns = pd.Series(frame[return_column], dtype=float).reset_index(drop=True)
        dates = pd.to_datetime(frame[date_column], errors="coerce").reset_index(drop=True)

        lagged_equity = (1.0 + returns).cumprod().shift(1).fillna(1.0)
        lagged_peaks = lagged_equity.cummax()
        lagged_drawdown = lagged_equity / lagged_peaks - 1.0

        trailing_return = (
            (1.0 + returns)
            .rolling(self.trend_lookback_windows, min_periods=self.trend_lookback_windows)
            .apply(np.prod, raw=True)
            .sub(1.0)
            .shift(1)
        )

        annualization = np.sqrt(252.0 / float(self.horizon_sessions))
        lagged_realized_vol = (
            returns.rolling(self.vol_lookback_windows, min_periods=self.vol_lookback_windows).std(ddof=1).shift(1)
            * annualization
        )

        history_windows = pd.Series(np.arange(len(frame)), dtype=int)
        enough_trend = history_windows >= self.trend_lookback_windows
        enough_vol = history_windows >= self.vol_lookback_windows
        enough_history = enough_trend & enough_vol
        deep_drawdown = lagged_drawdown <= float(self.drawdown_threshold)
        positive_trend = trailing_return > 0.0
        high_vol = lagged_realized_vol >= float(self.high_vol_annualized_threshold)

        labels = np.full(len(frame), "warmup", dtype=object)
        labels[enough_history & positive_trend & ~deep_drawdown & ~high_vol] = "bull"
        labels[enough_history & ~deep_drawdown & (~positive_trend | high_vol)] = "correction"
        labels[enough_history & deep_drawdown & positive_trend & ~high_vol] = "rebound"
        labels[enough_history & deep_drawdown & (~positive_trend | high_vol)] = "bear"

        return pd.DataFrame(
            {
                date_column: dates,
                "regime_label": labels,
                "regime_drawdown": lagged_drawdown.astype(float),
                "regime_trailing_return": trailing_return.astype(float),
                "regime_realized_vol_annualized": lagged_realized_vol.astype(float),
                "regime_high_vol": high_vol.astype(bool),
                "regime_history_windows": history_windows.astype(int),
            }
        )


@dataclass(slots=True, frozen=True)
class BenchmarkTrendDrawdownVolCrossAssetRegimeDetector:
    """White-box regime detector using lagged trend, drawdown, vol, and cross-asset spreads."""

    trend_lookback_windows: int = 12
    drawdown_threshold: float = -0.10
    vol_lookback_windows: int = 12
    high_vol_annualized_threshold: float = 0.18
    horizon_sessions: int = 5
    cross_asset_lookback_windows: int = 12
    cross_asset_risk_on_threshold: float = 0.0
    equity_columns: tuple[str, ...] = ("spy_ret", "vxus_ret")
    defensive_columns: tuple[str, ...] = ("agg_ret", "cta_ret", "gldm_ret", "sgov_ret")

    def label_frame(
        self,
        frame: pd.DataFrame,
        *,
        return_column: str = "benchmark_return",
        date_column: str = "entry_date",
    ) -> pd.DataFrame:
        if return_column not in frame.columns:
            raise KeyError(f"Missing return column: {return_column}")
        if date_column not in frame.columns:
            raise KeyError(f"Missing date column: {date_column}")
        missing_equity = [column for column in self.equity_columns if column not in frame.columns]
        missing_defensive = [column for column in self.defensive_columns if column not in frame.columns]
        if missing_equity or missing_defensive:
            missing = sorted(missing_equity + missing_defensive)
            raise KeyError(f"Missing cross-asset columns: {missing}")
        if self.trend_lookback_windows <= 0:
            raise ValueError("trend_lookback_windows must be positive.")
        if self.vol_lookback_windows <= 1:
            raise ValueError("vol_lookback_windows must be greater than 1.")
        if self.cross_asset_lookback_windows <= 0:
            raise ValueError("cross_asset_lookback_windows must be positive.")
        if self.drawdown_threshold >= 0.0:
            raise ValueError("drawdown_threshold must be negative.")
        if self.high_vol_annualized_threshold <= 0.0:
            raise ValueError("high_vol_annualized_threshold must be positive.")
        if self.horizon_sessions <= 0:
            raise ValueError("horizon_sessions must be positive.")

        returns = pd.Series(frame[return_column], dtype=float).reset_index(drop=True)
        dates = pd.to_datetime(frame[date_column], errors="coerce").reset_index(drop=True)

        lagged_equity = (1.0 + returns).cumprod().shift(1).fillna(1.0)
        lagged_peaks = lagged_equity.cummax()
        lagged_drawdown = lagged_equity / lagged_peaks - 1.0

        trailing_return = (
            (1.0 + returns)
            .rolling(self.trend_lookback_windows, min_periods=self.trend_lookback_windows)
            .apply(np.prod, raw=True)
            .sub(1.0)
            .shift(1)
        )

        annualization = np.sqrt(252.0 / float(self.horizon_sessions))
        lagged_realized_vol = (
            returns.rolling(self.vol_lookback_windows, min_periods=self.vol_lookback_windows).std(ddof=1).shift(1)
            * annualization
        )

        equity_returns = pd.DataFrame(
            {column: pd.Series(frame[column], dtype=float).reset_index(drop=True) for column in self.equity_columns}
        )
        defensive_returns = pd.DataFrame(
            {column: pd.Series(frame[column], dtype=float).reset_index(drop=True) for column in self.defensive_columns}
        )

        lagged_equity_signal = _lagged_cross_asset_signal(equity_returns, self.cross_asset_lookback_windows)
        lagged_defensive_signal = _lagged_cross_asset_signal(defensive_returns, self.cross_asset_lookback_windows)
        lagged_cross_asset_spread = lagged_equity_signal - lagged_defensive_signal

        history_windows = pd.Series(np.arange(len(frame)), dtype=int)
        enough_history = (
            (history_windows >= self.trend_lookback_windows)
            & (history_windows >= self.vol_lookback_windows)
            & (history_windows >= self.cross_asset_lookback_windows)
        )
        deep_drawdown = lagged_drawdown <= float(self.drawdown_threshold)
        positive_trend = trailing_return > 0.0
        high_vol = lagged_realized_vol >= float(self.high_vol_annualized_threshold)
        risk_on_supportive = lagged_cross_asset_spread > float(self.cross_asset_risk_on_threshold)

        labels = np.full(len(frame), "warmup", dtype=object)
        labels[enough_history & positive_trend & ~deep_drawdown & ~high_vol & risk_on_supportive] = "bull"
        labels[enough_history & deep_drawdown & positive_trend & ~high_vol & risk_on_supportive] = "rebound"
        labels[enough_history & deep_drawdown & (~positive_trend | high_vol | ~risk_on_supportive)] = "bear"
        labels[enough_history & ~deep_drawdown & (~positive_trend | high_vol | ~risk_on_supportive)] = "correction"

        return pd.DataFrame(
            {
                date_column: dates,
                "regime_label": labels,
                "regime_drawdown": lagged_drawdown.astype(float),
                "regime_trailing_return": trailing_return.astype(float),
                "regime_realized_vol_annualized": lagged_realized_vol.astype(float),
                "regime_cross_asset_spread": lagged_cross_asset_spread.astype(float),
                "regime_risk_on_supportive": risk_on_supportive.astype(bool),
                "regime_high_vol": high_vol.astype(bool),
                "regime_history_windows": history_windows.astype(int),
            }
        )


@dataclass(slots=True, frozen=True)
class MultiSignalBucketScoreRegimeDetector:
    """Probability-style white-box detector for a bucketed multi-asset universe."""

    benchmark_column: str = "equity_us"
    duration_column: str = "duration"
    credit_column: str = "credit"
    inflation_column: str = "inflation_hedge"
    trend_column: str = "trend"
    trend_lookback_windows: int = 84
    relative_lookback_windows: int = 63
    vol_lookback_windows: int = 42
    correlation_lookback_windows: int = 63
    normalization_lookback_windows: int = 126
    score_smoothing_halflife: float = 10.0
    horizon_sessions: int = 1
    risk_on_min_score: float = 0.30
    defensive_min_score: float = 0.30
    risk_on_net_threshold: float = 0.02
    defensive_net_threshold: float = -0.02

    def label_frame(
        self,
        frame: pd.DataFrame,
        *,
        return_column: str = "benchmark_return",
        date_column: str = "entry_date",
    ) -> pd.DataFrame:
        del return_column
        required_columns = {
            date_column,
            self.benchmark_column,
            self.duration_column,
            self.credit_column,
            self.inflation_column,
            self.trend_column,
        }
        missing_columns = sorted(required_columns.difference(frame.columns))
        if missing_columns:
            raise KeyError(f"Missing multi-signal regime columns: {missing_columns}")
        if self.trend_lookback_windows <= 1:
            raise ValueError("trend_lookback_windows must be greater than 1.")
        if self.relative_lookback_windows <= 1:
            raise ValueError("relative_lookback_windows must be greater than 1.")
        if self.vol_lookback_windows <= 1:
            raise ValueError("vol_lookback_windows must be greater than 1.")
        if self.correlation_lookback_windows <= 1:
            raise ValueError("correlation_lookback_windows must be greater than 1.")
        if self.normalization_lookback_windows <= 5:
            raise ValueError("normalization_lookback_windows must be greater than 5.")
        if self.score_smoothing_halflife <= 0.0:
            raise ValueError("score_smoothing_halflife must be positive.")
        if self.horizon_sessions <= 0:
            raise ValueError("horizon_sessions must be positive.")

        dates = pd.to_datetime(frame[date_column], errors="coerce").reset_index(drop=True)
        equity_returns = pd.Series(frame[self.benchmark_column], dtype=float).reset_index(drop=True)
        duration_returns = pd.Series(frame[self.duration_column], dtype=float).reset_index(drop=True)
        credit_returns = pd.Series(frame[self.credit_column], dtype=float).reset_index(drop=True)
        inflation_returns = pd.Series(frame[self.inflation_column], dtype=float).reset_index(drop=True)
        trend_returns = pd.Series(frame[self.trend_column], dtype=float).reset_index(drop=True)

        lagged_equity_trailing_return = _lagged_compounded_return(
            equity_returns,
            self.trend_lookback_windows,
        )
        lagged_duration_trailing_return = _lagged_compounded_return(
            duration_returns,
            self.relative_lookback_windows,
        )
        lagged_credit_trailing_return = _lagged_compounded_return(
            credit_returns,
            self.relative_lookback_windows,
        )
        lagged_inflation_trailing_return = _lagged_compounded_return(
            inflation_returns,
            self.relative_lookback_windows,
        )
        lagged_trend_trailing_return = _lagged_compounded_return(
            trend_returns,
            self.relative_lookback_windows,
        )
        lagged_equity_duration_relative = (
            _lagged_compounded_return(equity_returns, self.relative_lookback_windows)
            - lagged_duration_trailing_return
        )
        lagged_credit_duration_relative = lagged_credit_trailing_return - lagged_duration_trailing_return

        lagged_equity_drawdown = _lagged_drawdown(equity_returns)
        lagged_realized_vol = _lagged_realized_vol_annualized(
            equity_returns,
            lookback_windows=self.vol_lookback_windows,
            horizon_sessions=self.horizon_sessions,
        )
        lagged_equity_duration_correlation = (
            equity_returns.rolling(
                self.correlation_lookback_windows,
                min_periods=self.correlation_lookback_windows,
            )
            .corr(duration_returns)
            .shift(1)
        )

        equity_trend_score = _bounded_lagged_zscore(
            lagged_equity_trailing_return,
            self.normalization_lookback_windows,
        )
        equity_duration_relative_score = _bounded_lagged_zscore(
            lagged_equity_duration_relative,
            self.normalization_lookback_windows,
        )
        credit_duration_relative_score = _bounded_lagged_zscore(
            lagged_credit_duration_relative,
            self.normalization_lookback_windows,
        )
        inflation_pressure_score = _bounded_lagged_zscore(
            lagged_inflation_trailing_return,
            self.normalization_lookback_windows,
        )
        trend_support_score = _bounded_lagged_zscore(
            lagged_trend_trailing_return,
            self.normalization_lookback_windows,
        )
        drawdown_pressure_score = _bounded_lagged_zscore(
            (-lagged_equity_drawdown).clip(lower=0.0),
            self.normalization_lookback_windows,
        )
        volatility_pressure_score = _bounded_lagged_zscore(
            lagged_realized_vol,
            self.normalization_lookback_windows,
        )
        correlation_pressure_score = _bounded_lagged_zscore(
            lagged_equity_duration_correlation,
            self.normalization_lookback_windows,
        )

        risk_on_raw = (
            0.30 * equity_trend_score.clip(lower=0.0)
            + 0.25 * equity_duration_relative_score.clip(lower=0.0)
            + 0.20 * credit_duration_relative_score.clip(lower=0.0)
            + 0.15 * trend_support_score.clip(lower=0.0)
            + 0.10 * (-correlation_pressure_score).clip(lower=0.0)
        )
        defensive_raw = (
            0.25 * drawdown_pressure_score.clip(lower=0.0)
            + 0.20 * volatility_pressure_score.clip(lower=0.0)
            + 0.20 * correlation_pressure_score.clip(lower=0.0)
            + 0.15 * (-equity_trend_score).clip(lower=0.0)
            + 0.10 * (-credit_duration_relative_score).clip(lower=0.0)
            + 0.10 * inflation_pressure_score.clip(lower=0.0)
        )

        risk_on_score = risk_on_raw.ewm(
            halflife=float(self.score_smoothing_halflife),
            adjust=False,
            min_periods=1,
        ).mean()
        defensive_score = defensive_raw.ewm(
            halflife=float(self.score_smoothing_halflife),
            adjust=False,
            min_periods=1,
        ).mean()
        net_score = risk_on_score - defensive_score

        history_windows = pd.Series(np.arange(len(frame)), dtype=int)
        enough_history = history_windows >= max(
            self.trend_lookback_windows,
            self.relative_lookback_windows,
            self.vol_lookback_windows,
            self.correlation_lookback_windows,
            self.normalization_lookback_windows,
        )

        labels = np.full(len(frame), "warmup", dtype=object)
        labels[
            enough_history
            & (defensive_score >= float(self.defensive_min_score))
            & (net_score <= float(self.defensive_net_threshold))
        ] = "defensive"
        labels[
            enough_history
            & (risk_on_score >= float(self.risk_on_min_score))
            & (net_score >= float(self.risk_on_net_threshold))
        ] = "risk_on"
        labels[enough_history & (labels == "warmup")] = "neutral"

        return pd.DataFrame(
            {
                date_column: dates,
                "regime_label": labels,
                "regime_equity_trailing_return": lagged_equity_trailing_return.astype(float),
                "regime_equity_duration_relative": lagged_equity_duration_relative.astype(float),
                "regime_credit_duration_relative": lagged_credit_duration_relative.astype(float),
                "regime_inflation_trailing_return": lagged_inflation_trailing_return.astype(float),
                "regime_trend_trailing_return": lagged_trend_trailing_return.astype(float),
                "regime_drawdown": lagged_equity_drawdown.astype(float),
                "regime_realized_vol_annualized": lagged_realized_vol.astype(float),
                "regime_equity_duration_correlation": lagged_equity_duration_correlation.astype(float),
                "regime_equity_trend_score": equity_trend_score.astype(float),
                "regime_equity_duration_relative_score": equity_duration_relative_score.astype(float),
                "regime_credit_duration_relative_score": credit_duration_relative_score.astype(float),
                "regime_inflation_pressure_score": inflation_pressure_score.astype(float),
                "regime_trend_support_score": trend_support_score.astype(float),
                "regime_drawdown_pressure_score": drawdown_pressure_score.astype(float),
                "regime_volatility_pressure_score": volatility_pressure_score.astype(float),
                "regime_correlation_pressure_score": correlation_pressure_score.astype(float),
                "regime_risk_on_score": risk_on_score.astype(float),
                "regime_defensive_score": defensive_score.astype(float),
                "regime_net_score": net_score.astype(float),
                "regime_history_windows": history_windows.astype(int),
            }
        )


@dataclass(slots=True, frozen=True)
class BenchmarkForwardRegimeLabeler:
    """Create strategy-independent market labels from future benchmark paths."""

    forward_windows: int = 12
    drawdown_threshold: float = -0.10

    def label_frame(
        self,
        frame: pd.DataFrame,
        *,
        return_column: str = "benchmark_return",
        date_column: str = "entry_date",
    ) -> pd.DataFrame:
        if return_column not in frame.columns:
            raise KeyError(f"Missing return column: {return_column}")
        if date_column not in frame.columns:
            raise KeyError(f"Missing date column: {date_column}")
        if self.forward_windows <= 0:
            raise ValueError("forward_windows must be positive.")
        if self.drawdown_threshold >= 0.0:
            raise ValueError("drawdown_threshold must be negative.")

        returns = pd.Series(frame[return_column], dtype=float).reset_index(drop=True)
        dates = pd.to_datetime(frame[date_column], errors="coerce").reset_index(drop=True)

        labels: list[str] = []
        forward_total_returns: list[float] = []
        forward_max_drawdowns: list[float] = []
        forward_positive_rates: list[float] = []

        for index in range(len(frame)):
            window = returns.iloc[index : index + self.forward_windows]
            if len(window) < self.forward_windows:
                labels.append("unlabeled")
                forward_total_returns.append(np.nan)
                forward_max_drawdowns.append(np.nan)
                forward_positive_rates.append(np.nan)
                continue

            equity = (1.0 + window).cumprod()
            total_return = float(equity.iloc[-1] - 1.0)
            drawdown = float((equity / equity.cummax() - 1.0).min())
            positive_rate = float((window > 0.0).mean())

            if total_return > 0.0 and drawdown > float(self.drawdown_threshold):
                label = "bull"
            elif total_return <= 0.0 and drawdown > float(self.drawdown_threshold):
                label = "correction"
            elif total_return <= 0.0:
                label = "bear"
            else:
                label = "rebound"

            labels.append(label)
            forward_total_returns.append(total_return)
            forward_max_drawdowns.append(drawdown)
            forward_positive_rates.append(positive_rate)

        return pd.DataFrame(
            {
                date_column: dates,
                "forward_regime_label": labels,
                "forward_regime_total_return": pd.Series(forward_total_returns, dtype=float),
                "forward_regime_max_drawdown": pd.Series(forward_max_drawdowns, dtype=float),
                "forward_regime_positive_rate": pd.Series(forward_positive_rates, dtype=float),
                "forward_regime_windows": int(self.forward_windows),
            }
        )


@dataclass(slots=True, frozen=True)
class RegimeGatePolicy:
    multipliers: Mapping[str, float]

    def __post_init__(self) -> None:
        missing = set(REGIME_LABELS).difference(self.multipliers)
        if missing:
            raise ValueError(f"Missing regime multipliers for: {sorted(missing)}")
        unknown = set(self.multipliers).difference(REGIME_LABELS)
        if unknown:
            raise ValueError(f"Unknown regime labels: {sorted(unknown)}")
        for label, multiplier in self.multipliers.items():
            value = float(multiplier)
            if value < 0.0:
                raise ValueError(f"Regime multiplier must be non-negative for label '{label}'.")
            if value > 1.0:
                raise ValueError(f"Regime multiplier must be <= 1.0 for label '{label}'.")

    def multiplier_for(self, label: str) -> float:
        if label not in self.multipliers:
            raise KeyError(f"Unknown regime label: {label}")
        return float(self.multipliers[label])

    def apply(self, labels: pd.Series) -> pd.Series:
        mapped = labels.map(self.multipliers)
        if mapped.isna().any():
            missing = sorted(set(labels[mapped.isna()].astype(str)))
            raise KeyError(f"Missing regime multiplier for labels: {missing}")
        return mapped.astype(float)


def apply_regime_gate(
    frame: pd.DataFrame,
    *,
    base_sleeve_weight: float,
    detector: RegimeDetector,
    gate_policy: RegimeGatePolicy,
    return_column: str = "benchmark_return",
    date_column: str = "entry_date",
) -> pd.DataFrame:
    weight = float(base_sleeve_weight)
    if weight < 0.0 or weight > 1.0:
        raise ValueError("base_sleeve_weight must be between 0 and 1.")

    labeled = detector.label_frame(frame, return_column=return_column, date_column=date_column)
    gated = frame.reset_index(drop=True).copy()
    for column in ("regime_label", "regime_drawdown", "regime_trailing_return", "regime_history_windows"):
        gated[column] = labeled[column]
    gated["regime_multiplier"] = gate_policy.apply(gated["regime_label"])
    gated["base_sleeve_weight"] = weight
    gated["effective_sleeve_weight"] = gated["regime_multiplier"] * weight
    gated["effective_core_weight"] = 1.0 - gated["effective_sleeve_weight"]
    return gated


def build_regime_confusion_matrix(
    labeled_frame: pd.DataFrame,
    *,
    predicted_column: str = "regime_label",
    target_column: str = "forward_regime_label",
) -> pd.DataFrame:
    if predicted_column not in labeled_frame.columns:
        raise KeyError(f"Missing predicted column: {predicted_column}")
    if target_column not in labeled_frame.columns:
        raise KeyError(f"Missing target column: {target_column}")

    frame = labeled_frame.loc[
        labeled_frame[predicted_column].notna() & labeled_frame[target_column].notna()
    ].copy()
    return pd.crosstab(frame[predicted_column], frame[target_column], dropna=False).sort_index().sort_index(axis=1)


def _lagged_cross_asset_signal(returns: pd.DataFrame, lookback_windows: int) -> pd.Series:
    compounded = (1.0 + returns).rolling(lookback_windows, min_periods=lookback_windows).apply(np.prod, raw=True) - 1.0
    mean_signal = compounded.mean(axis=1)
    return mean_signal.shift(1).astype(float)


def _lagged_compounded_return(returns: pd.Series, lookback_windows: int) -> pd.Series:
    return (
        (1.0 + pd.Series(returns, dtype=float))
        .rolling(lookback_windows, min_periods=lookback_windows)
        .apply(np.prod, raw=True)
        .sub(1.0)
        .shift(1)
        .astype(float)
    )


def _lagged_drawdown(returns: pd.Series) -> pd.Series:
    lagged_equity = (1.0 + pd.Series(returns, dtype=float)).cumprod().shift(1).fillna(1.0)
    lagged_peaks = lagged_equity.cummax()
    return (lagged_equity / lagged_peaks - 1.0).astype(float)


def _lagged_realized_vol_annualized(
    returns: pd.Series,
    *,
    lookback_windows: int,
    horizon_sessions: int,
) -> pd.Series:
    annualization = np.sqrt(252.0 / float(horizon_sessions))
    return (
        pd.Series(returns, dtype=float)
        .rolling(lookback_windows, min_periods=lookback_windows)
        .std(ddof=1)
        .shift(1)
        .mul(annualization)
        .astype(float)
    )


def _bounded_lagged_zscore(series: pd.Series, lookback_windows: int) -> pd.Series:
    numeric = pd.Series(series, dtype=float)
    rolling_mean = numeric.rolling(lookback_windows, min_periods=lookback_windows).mean().shift(1)
    rolling_std = numeric.rolling(lookback_windows, min_periods=lookback_windows).std(ddof=1).shift(1)
    zscore = (numeric - rolling_mean) / rolling_std.replace(0.0, np.nan)
    bounded = np.tanh(zscore / 2.0)
    return pd.Series(bounded, index=numeric.index, dtype=float)
