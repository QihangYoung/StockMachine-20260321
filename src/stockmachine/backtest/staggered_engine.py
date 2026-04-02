from __future__ import annotations

from collections import defaultdict
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
from stockmachine.domain.models import TargetPosition


@dataclass(slots=True)
class _ActiveCohort:
    signal_date: date
    entry_date: date
    exit_date: date
    position_values: dict[str, float]
    target_count: int
    order_count: int


@dataclass(slots=True)
class DailyStaggeredOpenHoldBacktestEngine:
    """Daily staggered next-open engine for overlapping fixed-horizon cohorts."""

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
        if self.horizon_bars <= 0:
            raise ValueError("DailyStaggeredOpenHoldBacktestEngine requires horizon_bars >= 1.")

        prediction_dates = sorted(
            current_date
            for current_date in self.predictions["date"].dt.date.unique()
            if start_date <= current_date <= end_date
        )
        if len(prediction_dates) < 3:
            return BacktestResult(
                sessions=0,
                total_return=0.0,
                annualized_return=np.nan,
                annualized_volatility=np.nan,
                sharpe=np.nan,
                max_drawdown=np.nan,
                meta={"records": []},
            )

        daily_pivot = self._build_bar_lookup(self.daily_bar)
        benchmark_open = self._build_open_lookup(self.benchmark_index)

        equity = self.initial_equity
        cash = self.initial_equity
        active_cohorts: list[_ActiveCohort] = []
        equity_curve: list[float] = []
        benchmark_curve: list[float] = []
        daily_returns: list[float] = []
        records: list[dict[str, object]] = []

        for current_idx in range(1, len(prediction_dates) - 1):
            signal_date = prediction_dates[current_idx - 1]
            trade_date = prediction_dates[current_idx]
            next_date = prediction_dates[current_idx + 1]
            equity_before_trade = equity

            aggregate_values_before = self._aggregate_position_values(active_cohorts)
            pre_trade_weights = self._weights_from_values(aggregate_values_before, equity_before_trade)

            exiting_cohorts = [cohort for cohort in active_cohorts if cohort.exit_date == trade_date]
            if exiting_cohorts:
                cash += sum(self._cohort_value(cohort) for cohort in exiting_cohorts)
                active_cohorts = [cohort for cohort in active_cohorts if cohort.exit_date != trade_date]

            account = self._make_account_snapshot(signal_date, equity, pre_trade_weights, daily_pivot)
            new_cohort: _ActiveCohort | None = None
            if current_idx + self.horizon_bars < len(prediction_dates):
                universe = self.predictions.loc[
                    self.predictions["date"].dt.date == signal_date, "symbol"
                ].tolist()
                signals = self.signal_model.predict(signal_date, universe)
                targets = self.portfolio_policy.build_targets(signal_date, signals, account)
                scaled_targets = self._scale_targets(targets, scale=1.0 / self.horizon_bars)
                orders = self.execution_policy.generate_orders(
                    trade_date,
                    scaled_targets,
                    self._bars_for_date(trade_date, daily_pivot),
                    account,
                )
                new_cohort = self._build_cohort(
                    signal_date=signal_date,
                    entry_date=trade_date,
                    exit_date=prediction_dates[current_idx + self.horizon_bars],
                    targets=scaled_targets,
                    account_equity=equity_before_trade,
                    order_count=len(orders),
                )
                if new_cohort is not None:
                    active_cohorts.append(new_cohort)
                    cash -= self._cohort_value(new_cohort)

            aggregate_values_after_trade = self._aggregate_position_values(active_cohorts)
            post_trade_weights = self._weights_from_values(aggregate_values_after_trade, equity_before_trade)
            turnover = self._weight_turnover(pre_trade_weights, post_trade_weights)
            cost = turnover * self.cost_bps_per_side / 10000.0
            cost_amount = equity_before_trade * cost
            cash -= cost_amount

            current_position_value = float(sum(aggregate_values_after_trade.values()))
            updated_position_values = self._advance_cohorts(
                active_cohorts,
                current_date=trade_date,
                next_date=next_date,
                lookup=daily_pivot,
            )
            next_position_value = float(sum(updated_position_values.values()))
            gross_return = (
                (next_position_value - current_position_value) / equity_before_trade
                if equity_before_trade > 0
                else 0.0
            )
            benchmark_return = self._benchmark_return(trade_date, next_date, benchmark_open)
            equity = cash + next_position_value
            net_return = (
                (equity - equity_before_trade) / equity_before_trade
                if equity_before_trade > 0
                else 0.0
            )

            equity_curve.append(equity)
            benchmark_curve.append(benchmark_return)
            daily_returns.append(net_return)
            records.append(
                {
                    "signal_date": signal_date.isoformat(),
                    "trade_date": trade_date.isoformat(),
                    "next_date": next_date.isoformat(),
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "benchmark_return": benchmark_return,
                    "turnover": turnover,
                    "cost_bps": cost * 10000.0,
                    "positions": len(post_trade_weights),
                    "active_cohorts": len(active_cohorts),
                    "gross_exposure": float(sum(post_trade_weights.values())),
                    "cash_weight": float(max(0.0, 1.0 - sum(post_trade_weights.values()))),
                    "exiting_cohorts": len(exiting_cohorts),
                    "new_targets": 0 if new_cohort is None else new_cohort.target_count,
                    "new_orders": 0 if new_cohort is None else new_cohort.order_count,
                }
            )

        periodic_series = pd.Series(daily_returns, dtype=float)
        benchmark_total = (
            float((1.0 + pd.Series(benchmark_curve, dtype=float)).prod() - 1.0)
            if benchmark_curve
            else np.nan
        )
        total_return = equity / self.initial_equity - 1.0
        annualized_return = _annualize_total_return(total_return, len(daily_returns), 1)
        annualized_volatility = (
            float(periodic_series.std(ddof=1) * np.sqrt(252))
            if len(periodic_series) > 1
            else np.nan
        )
        sharpe = (
            float((periodic_series.mean() / periodic_series.std(ddof=1)) * np.sqrt(252))
            if len(periodic_series) > 1 and periodic_series.std(ddof=1) > 0
            else np.nan
        )
        max_drawdown = _max_drawdown(pd.Series(equity_curve, dtype=float))

        return BacktestResult(
            sessions=len(daily_returns),
            total_return=float(total_return),
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            meta={
                "benchmark_total_return": benchmark_total,
                "mean_turnover": float(pd.Series([r["turnover"] for r in records]).mean()) if records else np.nan,
                "mean_cost_bps": float(pd.Series([r["cost_bps"] for r in records]).mean()) if records else np.nan,
                "mean_gross_exposure": float(pd.Series([r["gross_exposure"] for r in records]).mean()) if records else np.nan,
                "mean_cash_weight": float(pd.Series([r["cash_weight"] for r in records]).mean()) if records else np.nan,
                "mean_positions": float(pd.Series([r["positions"] for r in records]).mean()) if records else np.nan,
                "mean_active_cohorts": float(pd.Series([r["active_cohorts"] for r in records]).mean()) if records else np.nan,
                "records": records,
            },
        )

    def _build_cohort(
        self,
        *,
        signal_date: date,
        entry_date: date,
        exit_date: date,
        targets: list[TargetPosition],
        account_equity: float,
        order_count: int,
    ) -> _ActiveCohort | None:
        if not targets or account_equity <= 0:
            return None
        position_values = {
            target.symbol: float(account_equity * target.target_weight)
            for target in targets
            if abs(float(target.target_weight)) > 1e-12
        }
        if not position_values:
            return None
        return _ActiveCohort(
            signal_date=signal_date,
            entry_date=entry_date,
            exit_date=exit_date,
            position_values=position_values,
            target_count=len(targets),
            order_count=order_count,
        )

    def _scale_targets(self, targets: list[TargetPosition], *, scale: float) -> list[TargetPosition]:
        if scale == 1.0:
            return list(targets)
        return [
            TargetPosition(
                symbol=target.symbol,
                target_weight=float(target.target_weight * scale),
                max_weight=float(target.max_weight * scale),
                reason=target.reason,
                timestamp=target.timestamp,
                meta=target.meta,
            )
            for target in targets
        ]

    def _advance_cohorts(
        self,
        cohorts: list[_ActiveCohort],
        *,
        current_date: date,
        next_date: date,
        lookup: dict[date, dict[str, MarketBar]],
    ) -> dict[str, float]:
        updated: dict[str, float] = {}
        for cohort in cohorts:
            for symbol, value in list(cohort.position_values.items()):
                current_bar = lookup.get(current_date, {}).get(symbol)
                next_bar = lookup.get(next_date, {}).get(symbol)
                current_open = self._bar_open_value(current_bar) if current_bar is not None else np.nan
                next_open = self._bar_open_value(next_bar) if next_bar is not None else np.nan
                if (
                    current_bar is not None
                    and next_bar is not None
                    and np.isfinite(current_open)
                    and np.isfinite(next_open)
                    and current_open > 0
                ):
                    cohort.position_values[symbol] = float(value * (next_open / current_open))
                updated[symbol] = updated.get(symbol, 0.0) + cohort.position_values[symbol]
        return updated

    def _aggregate_position_values(self, cohorts: list[_ActiveCohort]) -> dict[str, float]:
        aggregate: dict[str, float] = defaultdict(float)
        for cohort in cohorts:
            for symbol, value in cohort.position_values.items():
                aggregate[symbol] += float(value)
        return dict(aggregate)

    def _cohort_value(self, cohort: _ActiveCohort) -> float:
        return float(sum(cohort.position_values.values()))

    def _weights_from_values(self, values: dict[str, float], equity: float) -> dict[str, float]:
        if equity <= 0:
            return {}
        return {
            symbol: float(value / equity)
            for symbol, value in values.items()
            if abs(float(value)) > 1e-12
        }

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

    def _benchmark_return(self, entry_date: date, exit_date: date, lookup: dict[date, float]) -> float:
        entry_open = lookup.get(entry_date)
        exit_open = lookup.get(exit_date)
        if (
            entry_open is None
            or exit_open is None
            or not np.isfinite(entry_open)
            or not np.isfinite(exit_open)
            or entry_open <= 0
        ):
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


def _annualize_total_return(total_return: float, periods: int, period_bars: int) -> float:
    if periods <= 0 or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (252 / (periods * period_bars)) - 1.0)


def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return np.nan
    peaks = equity_curve.cummax()
    drawdowns = equity_curve / peaks - 1.0
    return float(drawdowns.min())
