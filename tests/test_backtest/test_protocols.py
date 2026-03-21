from datetime import date

from stockmachine.backtest import AccountSnapshot, BacktestResult, MarketBar, PositionSnapshot


def test_backtest_primitives_can_be_constructed() -> None:
    bar = MarketBar(
        session_date=date(2025, 1, 2),
        symbol="AAPL",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000000,
    )
    position = PositionSnapshot(symbol="AAPL", quantity=10, market_value=1005.0, weight=0.1)
    account = AccountSnapshot(
        session_date=bar.session_date,
        cash=9000.0,
        equity=10005.0,
        gross_exposure=0.1,
        positions=(position,),
    )
    result = BacktestResult(
        sessions=10,
        total_return=0.02,
        annualized_return=0.12,
        annualized_volatility=0.15,
        sharpe=0.8,
        max_drawdown=-0.04,
    )

    assert account.positions[0].symbol == "AAPL"
    assert result.sessions == 10
    assert bar.close == 100.5
