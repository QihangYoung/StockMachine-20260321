from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from stockmachine.backtest.protocols import (
    AccountSnapshot,
    BacktestResult,
    ExecutionPolicy,
    MarketBar,
    PortfolioPolicy,
    PositionSnapshot,
    SignalModel,
)


@dataclass(slots=True)
class DailyOpenHoldBacktestEngine:
    """Minimal long-only engine for next-open entry and fixed-horizon hold."""

    predictions: pd.DataFrame
    daily_bar: pd.DataFrame
    benchmark_index: pd.DataFrame
    signal_model: SignalModel
    portfolio_policy: PortfolioPolicy
    execution_policy: ExecutionPolicy
    horizon_bars: int = 5
    cost_bps_per_side: float = 10.0
    initial_equity: float = 1_000_000.0

    def run(self, start_date: date, end_date: date) -> BacktestResult:
        prediction_dates = sorted(
            current_date
            for current_date in self.predictions["date"].dt.date.unique()
            if start_date <= current_date <= end_date
        )
        daily_pivot = self._build_bar_lookup(self.daily_bar)
        benchmark_open = self._build_open_lookup(self.benchmark_index)

        equity = self.initial_equity
        equity_curve = []
        benchmark_curve = []
        periodic_returns = []
        previous_weights: dict[str, float] = {}
        records = []

        for idx in range(0, len(prediction_dates), self.horizon_bars):
            signal_date = prediction_dates[idx]
            entry_index = idx + 1
            exit_index = idx + self.horizon_bars + 1
            if entry_index >= len(prediction_dates) or exit_index >= len(prediction_dates):
                break

            entry_date = prediction_dates[entry_index]
            exit_date = prediction_dates[exit_index]
            universe = self.predictions.loc[
                self.predictions["date"].dt.date == signal_date, "symbol"
            ].tolist()

            account = self._make_account_snapshot(signal_date, equity, previous_weights, daily_pivot)
            signals = self.signal_model.predict(signal_date, universe)
            targets = self.portfolio_policy.build_targets(signal_date, signals, account)
            orders = self.execution_policy.generate_orders(
                entry_date,
                targets,
                self._bars_for_date(entry_date, daily_pivot),
                account,
            )
            if not orders:
                continue

            new_weights = {target.symbol: target.target_weight for target in targets}
            gross_return = self._portfolio_return(new_weights, entry_date, exit_date, daily_pivot)
            benchmark_return = self._benchmark_return(entry_date, exit_date, benchmark_open)
            turnover = self._weight_turnover(previous_weights, new_weights)
            cost = turnover * self.cost_bps_per_side / 10000.0
            net_return = gross_return - cost
            equity *= 1.0 + net_return

            equity_curve.append(equity)
            benchmark_curve.append(benchmark_return)
            periodic_returns.append(net_return)
            previous_weights = new_weights
            records.append(
                {
                    "signal_date": signal_date.isoformat(),
                    "entry_date": entry_date.isoformat(),
                    "exit_date": exit_date.isoformat(),
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "benchmark_return": benchmark_return,
                    "turnover": turnover,
                    "cost_bps": cost * 10000.0,
                    "positions": len(targets),
                }
            )

        periodic_series = pd.Series(periodic_returns, dtype=float)
        benchmark_total = float((1.0 + pd.Series(benchmark_curve, dtype=float)).prod() - 1.0) if benchmark_curve else np.nan
        total_return = equity / self.initial_equity - 1.0
        annualized_return = _annualize_total_return(total_return, len(periodic_returns), self.horizon_bars)
        annualized_volatility = float(periodic_series.std(ddof=1) * np.sqrt(252 / self.horizon_bars)) if len(periodic_series) > 1 else np.nan
        sharpe = (
            float((periodic_series.mean() / periodic_series.std(ddof=1)) * np.sqrt(252 / self.horizon_bars))
            if len(periodic_series) > 1 and periodic_series.std(ddof=1) > 0
            else np.nan
        )
        max_drawdown = _max_drawdown(pd.Series(equity_curve, dtype=float))

        return BacktestResult(
            sessions=len(periodic_returns),
            total_return=float(total_return),
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            meta={
                "benchmark_total_return": benchmark_total,
                "mean_turnover": float(pd.Series([r["turnover"] for r in records]).mean()) if records else np.nan,
                "mean_cost_bps": float(pd.Series([r["cost_bps"] for r in records]).mean()) if records else np.nan,
                "records": records,
            },
        )

    def _build_bar_lookup(self, frame: pd.DataFrame) -> dict[date, dict[str, MarketBar]]:
        lookup: dict[date, dict[str, MarketBar]] = {}
        for row in frame.itertuples(index=False):
            session_date = pd.Timestamp(row.session_date).date()
            adj_open = getattr(row, "adj_open", None)
            lookup.setdefault(session_date, {})[row.symbol] = MarketBar(
                session_date=session_date,
                symbol=row.symbol,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                vwap=float(row.vwap) if getattr(row, "vwap", None) is not None else None,
                adj_open=float(adj_open) if pd.notna(adj_open) else None,
            )
        return lookup

    def _build_open_lookup(self, frame: pd.DataFrame) -> dict[date, float]:
        lookup: dict[date, float] = {}
        for row in frame.itertuples(index=False):
            adj_open = getattr(row, "adj_open", None)
            bar = MarketBar(
                session_date=pd.Timestamp(row.session_date).date(),
                symbol=row.symbol,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume) if getattr(row, "volume", None) is not None else 0.0,
                adj_open=float(adj_open) if pd.notna(adj_open) else None,
            )
            lookup[pd.Timestamp(row.session_date).date()] = self._bar_open_value(bar)
        return lookup

    def _bars_for_date(
        self,
        session_date: date,
        lookup: dict[date, dict[str, MarketBar]],
    ) -> dict[str, MarketBar]:
        return lookup.get(session_date, {})

    def _portfolio_return(
        self,
        weights: dict[str, float],
        entry_date: date,
        exit_date: date,
        lookup: dict[date, dict[str, MarketBar]],
    ) -> float:
        returns = []
        for symbol, weight in weights.items():
            entry_bar = lookup.get(entry_date, {}).get(symbol)
            exit_bar = lookup.get(exit_date, {}).get(symbol)
            entry_open = self._bar_open_value(entry_bar) if entry_bar is not None else np.nan
            exit_open = self._bar_open_value(exit_bar) if exit_bar is not None else np.nan
            if entry_bar is None or exit_bar is None or not np.isfinite(entry_open) or not np.isfinite(exit_open) or entry_open <= 0:
                continue
            realized = exit_open / entry_open - 1.0
            returns.append(weight * realized)
        return float(sum(returns))

    def _benchmark_return(self, entry_date: date, exit_date: date, lookup: dict[date, float]) -> float:
        entry_open = lookup.get(entry_date)
        exit_open = lookup.get(exit_date)
        if entry_open is None or exit_open is None or not np.isfinite(entry_open) or not np.isfinite(exit_open) or entry_open <= 0:
            return np.nan
        return float(exit_open / entry_open - 1.0)

    def _make_account_snapshot(
        self,
        session_date: date,
        equity: float,
        weights: dict[str, float],
        lookup: dict[date, dict[str, MarketBar]],
    ) -> AccountSnapshot:
        positions = []
        bars = lookup.get(session_date, {})
        for symbol, weight in weights.items():
            bar = bars.get(symbol)
            market_value = equity * weight
            quantity = int(market_value / bar.close) if bar and bar.close > 0 else 0
            positions.append(
                PositionSnapshot(
                    symbol=symbol,
                    quantity=quantity,
                    market_value=market_value,
                    weight=weight,
                )
            )
        return AccountSnapshot(
            session_date=session_date,
            cash=max(0.0, equity * (1.0 - sum(weights.values()))),
            equity=equity,
            gross_exposure=float(sum(weights.values())),
            positions=tuple(positions),
        )

    def _weight_turnover(self, previous: dict[str, float], new: dict[str, float]) -> float:
        symbols = set(previous) | set(new)
        return float(sum(abs(new.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols))

    def _bar_open_value(self, bar: MarketBar) -> float:
        return float(bar.adj_open) if bar.adj_open is not None else float(bar.open)


def _annualize_total_return(total_return: float, periods: int, horizon: int) -> float:
    if periods <= 0 or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (252 / (periods * horizon)) - 1.0)


def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return np.nan
    peaks = equity_curve.cummax()
    drawdowns = equity_curve / peaks - 1.0
    return float(drawdowns.min())
