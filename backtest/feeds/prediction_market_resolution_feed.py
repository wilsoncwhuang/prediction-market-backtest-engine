from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from core.events import Resolution


class PredictionMarketResolutionFeed:
    """
    Loads a prediction market resolutions parquet and emits Resolution events.

    Works with any binary prediction market (Kalshi, Polymarket, PredictIt, etc.)
    as long as the parquet follows the schema below.

    Parquet columns: ticker, close_time, result ("yes" or "no").
    """

    def __init__(self, parquet_path: Path | str) -> None:
        df = pd.read_parquet(parquet_path)
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
        self._df = df

    def load(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        tickers: list[str] | None = None,
    ) -> list[Resolution]:
        df = self._df

        if start is not None:
            df = df[df["close_time"] >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df["close_time"] <= pd.Timestamp(end, tz="UTC")]
        if tickers is not None:
            df = df[df["ticker"].isin(tickers)]

        return sorted(
            Resolution(
                timestamp=row.close_time.to_pydatetime(),
                ticker=row.ticker,
                result=(row.result == "yes"),
            )
            for row in df.itertuples(index=False)
        )
