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
    hold_rank_buffer: int = 0
    entry_rank_buffer: int = 0
    max_new_names_per_rebalance: int | None = None
    entry_score_threshold: float | None = None
    hold_score_threshold: float | None = None

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
        incumbent_symbols = {
            str(position.symbol).upper()
            for position in account.positions
            if str(position.symbol).strip() and abs(float(position.weight)) > 1e-12
        }
        selected = self._select_candidates(
            adjusted=adjusted,
            incumbent_symbols=incumbent_symbols,
        )

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
                    "adjusted_rank": adjusted_rank,
                    "selection_source": selection_source,
                    "is_incumbent": signal.symbol.upper() in incumbent_symbols,
                },
            )
            for signal, adjusted_score, adjusted_rank, selection_source in selected
        ]

    def _select_candidates(
        self,
        *,
        adjusted: list[tuple[Signal, str, float]],
        incumbent_symbols: set[str],
    ) -> list[tuple[Signal, float, int, str]]:
        ranked = [
            (rank, signal, sector, score)
            for rank, (signal, sector, score) in enumerate(adjusted, start=1)
        ]
        if not incumbent_symbols or (
            self.hold_rank_buffer <= 0
            and self.entry_rank_buffer <= 0
            and self.max_new_names_per_rebalance is None
        ):
            return self._select_plain_top_k(ranked)

        hold_rank_limit = max(self.top_k, self.top_k + max(self.hold_rank_buffer, 0))
        entry_rank_limit = max(1, self.top_k - max(self.entry_rank_buffer, 0))
        sector_counts: dict[str, int] = defaultdict(int)
        selected_symbols: set[str] = set()
        selected: list[tuple[Signal, float, int, str]] = []
        new_entries = 0

        def maybe_add(
            *,
            rank: int,
            signal: Signal,
            sector: str,
            score: float,
            source: str,
            is_new_entry: bool,
        ) -> bool:
            nonlocal new_entries
            symbol = signal.symbol.upper()
            if symbol in selected_symbols:
                return False
            if sector_counts[sector] >= self.max_positions_per_sector:
                return False
            if is_new_entry and self.max_new_names_per_rebalance is not None and new_entries >= self.max_new_names_per_rebalance:
                return False
            selected.append((signal, score, rank, source))
            selected_symbols.add(symbol)
            sector_counts[sector] += 1
            if is_new_entry:
                new_entries += 1
            return len(selected) >= self.top_k

        passes = (
            (
                "strong_entry",
                lambda rank, symbol, signal: (
                    symbol not in incumbent_symbols
                    and rank <= entry_rank_limit
                    and self._passes_entry_threshold(signal)
                ),
            ),
            (
                "retain_buffer",
                lambda rank, symbol, signal: (
                    symbol in incumbent_symbols
                    and rank <= hold_rank_limit
                    and self._passes_hold_threshold(signal)
                ),
            ),
            (
                "retain_fill",
                lambda rank, symbol, signal: symbol in incumbent_symbols and self._passes_hold_threshold(signal),
            ),
            (
                "entry_fill",
                lambda rank, symbol, signal: (
                    symbol not in incumbent_symbols and self._passes_entry_threshold(signal)
                ),
            ),
        )
        for source, predicate in passes:
            for rank, signal, sector, score in ranked:
                symbol = signal.symbol.upper()
                if not predicate(rank, symbol, signal):
                    continue
                if maybe_add(
                    rank=rank,
                    signal=signal,
                    sector=sector,
                    score=score,
                    source=source,
                    is_new_entry=symbol not in incumbent_symbols,
                ):
                    return selected
        return selected

    def _select_plain_top_k(
        self,
        ranked: list[tuple[int, Signal, str, float]],
    ) -> list[tuple[Signal, float, int, str]]:
        sector_counts: dict[str, int] = defaultdict(int)
        selected: list[tuple[Signal, float, int, str]] = []
        for rank, signal, sector, score in ranked:
            if not self._passes_entry_threshold(signal):
                continue
            if sector_counts[sector] >= self.max_positions_per_sector:
                continue
            selected.append((signal, score, rank, "plain_top_k"))
            sector_counts[sector] += 1
            if len(selected) >= self.top_k:
                break
        return selected

    def _passes_entry_threshold(self, signal: Signal) -> bool:
        if self.entry_score_threshold is None:
            return True
        return float(signal.score) >= float(self.entry_score_threshold)

    def _passes_hold_threshold(self, signal: Signal) -> bool:
        threshold = self.hold_score_threshold
        if threshold is None:
            threshold = self.entry_score_threshold
        if threshold is None:
            return True
        return float(signal.score) >= float(threshold)
