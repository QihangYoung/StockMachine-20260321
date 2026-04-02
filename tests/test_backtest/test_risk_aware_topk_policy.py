from datetime import date, datetime

from stockmachine.backtest.protocols import AccountSnapshot, PositionSnapshot
from stockmachine.domain.models import Signal
from stockmachine.portfolio import RiskAwareTopKPortfolioPolicy


def _signal(symbol: str, score: float, sector: str) -> Signal:
    return Signal(
        symbol=symbol,
        side="LONG",
        score=score,
        confidence=score,
        horizon_bars=1,
        timestamp=datetime(2025, 1, 2, 16, 0),
        meta={
            "close": 100.0,
            "median_dollar_volume_20": 100_000_000.0,
            "vol_20": 0.02,
            "sector": sector,
            "industry": sector,
        },
    )


def test_policy_rank_buffer_keeps_buffered_incumbent_but_admits_strong_new_entry() -> None:
    policy = RiskAwareTopKPortfolioPolicy(
        top_k=2,
        max_positions_per_sector=2,
        hold_rank_buffer=2,
        entry_rank_buffer=1,
        max_new_names_per_rebalance=1,
    )
    account = AccountSnapshot(
        session_date=date(2025, 1, 2),
        cash=0.0,
        equity=1_000_000.0,
        gross_exposure=1.0,
        positions=(
            PositionSnapshot(symbol="BBB", quantity=100, market_value=500_000.0, weight=0.5),
            PositionSnapshot(symbol="CCC", quantity=100, market_value=500_000.0, weight=0.5),
        ),
    )
    signals = [
        _signal("AAA", 1.00, "Tech"),
        _signal("BBB", 0.80, "Health"),
        _signal("DDD", 0.70, "Industrials"),
        _signal("CCC", 0.60, "Utilities"),
    ]

    targets = policy.build_targets(date(2025, 1, 2), signals, account)

    assert [target.symbol for target in targets] == ["AAA", "BBB"]
    assert targets[0].meta["selection_source"] == "strong_entry"
    assert targets[1].meta["selection_source"] == "retain_buffer"


def test_policy_max_new_names_caps_replacements_per_rebalance() -> None:
    policy = RiskAwareTopKPortfolioPolicy(
        top_k=3,
        max_positions_per_sector=3,
        hold_rank_buffer=2,
        entry_rank_buffer=1,
        max_new_names_per_rebalance=1,
    )
    account = AccountSnapshot(
        session_date=date(2025, 1, 2),
        cash=0.0,
        equity=1_000_000.0,
        gross_exposure=1.0,
        positions=(
            PositionSnapshot(symbol="BBB", quantity=100, market_value=333_333.0, weight=1 / 3),
            PositionSnapshot(symbol="CCC", quantity=100, market_value=333_333.0, weight=1 / 3),
            PositionSnapshot(symbol="EEE", quantity=100, market_value=333_333.0, weight=1 / 3),
        ),
    )
    signals = [
        _signal("AAA", 1.00, "Tech"),
        _signal("DDD", 0.95, "Tech"),
        _signal("BBB", 0.90, "Health"),
        _signal("CCC", 0.85, "Industrials"),
        _signal("EEE", 0.80, "Utilities"),
    ]

    targets = policy.build_targets(date(2025, 1, 2), signals, account)

    assert [target.symbol for target in targets] == ["AAA", "BBB", "CCC"]
    assert sum(1 for target in targets if not target.meta["is_incumbent"]) == 1
