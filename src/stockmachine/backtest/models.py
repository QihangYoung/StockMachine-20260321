from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time

import pandas as pd

from stockmachine.domain.models import Signal


@dataclass(slots=True)
class DataFrameSignalModel:
    """Signal model backed by a prediction DataFrame."""

    predictions: pd.DataFrame
    horizon_bars: int = 5

    def predict(self, session_date: date, universe: list[str]) -> list[Signal]:
        current = self.predictions[
            (self.predictions["date"].dt.date == session_date)
            & (self.predictions["symbol"].isin(universe))
        ].copy()
        if current.empty:
            return []

        timestamp = datetime.combine(session_date, time(16, 0))
        return [
            Signal(
                symbol=row.symbol,
                side="LONG",
                score=float(row.score),
                confidence=float(row.confidence),
                horizon_bars=self.horizon_bars,
                timestamp=timestamp,
                meta={
                    "sector": row.sector,
                    "industry": row.industry,
                    "close": float(row.close),
                    "vol_20": float(row.vol_20),
                    "median_dollar_volume_20": float(row.median_dollar_volume_20),
                },
            )
            for row in current.itertuples(index=False)
        ]
