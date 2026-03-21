from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time

from stockmachine.domain.models import Signal, TargetPosition
from stockmachine.backtest.protocols import AccountSnapshot


@dataclass(slots=True)
class RiskAwareTopKPortfolioPolicy:
    """Long-only top-k selector with simple risk filters and sector caps."""

    top_k: int = 10
    min_close: float = 10.0
    min_median_dollar_volume_20: float = 50_000_000.0
    max_vol_20: float = 0.04
    max_positions_per_sector: int = 2
    sector_neutral: bool = True

    def build_targets(
        self,
        session_date: date,
        signals: list[Signal],
        account: AccountSnapshot,
    ) -> list[TargetPosition]:
        if not signals:
            return []

        prepared = []
        sector_scores: dict[str, list[float]] = defaultdict(list)
        for signal in signals:
            close = float(signal.meta.get("close", 0.0))
            median_dollar_volume_20 = float(signal.meta.get("median_dollar_volume_20", 0.0))
            vol_20 = float(signal.meta.get("vol_20", 1.0))
            sector = str(signal.meta.get("sector", "Unknown"))
            if close < self.min_close:
                continue
            if median_dollar_volume_20 < self.min_median_dollar_volume_20:
                continue
            if vol_20 > self.max_vol_20:
                continue
            prepared.append((signal, sector))
            sector_scores[sector].append(signal.score)

        if not prepared:
            return []

        adjusted = []
        for signal, sector in prepared:
            score = signal.score
            if self.sector_neutral and sector_scores[sector]:
                score -= sum(sector_scores[sector]) / len(sector_scores[sector])
            adjusted.append((signal, sector, score))

        adjusted.sort(key=lambda item: item[2], reverse=True)
        sector_counts: dict[str, int] = defaultdict(int)
        selected: list[tuple[Signal, float]] = []
        for signal, sector, score in adjusted:
            if sector_counts[sector] >= self.max_positions_per_sector:
                continue
            selected.append((signal, score))
            sector_counts[sector] += 1
            if len(selected) >= self.top_k:
                break

        if not selected:
            return []

        weight = 1.0 / len(selected)
        timestamp = datetime.combine(session_date, time(16, 0))
        return [
            TargetPosition(
                symbol=signal.symbol,
                target_weight=weight,
                max_weight=weight,
                reason="risk_aware_top_k",
                timestamp=timestamp,
                meta={
                    **signal.meta,
                    "adjusted_score": adjusted_score,
                },
            )
            for signal, adjusted_score in selected
        ]
