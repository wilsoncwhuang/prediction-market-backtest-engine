from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.constants import Side
from core.events import OHLCVSnapshot

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"timestamp", "open", "high", "low", "close"}


class GenericOHLCVFeed:
    """
    Loads OHLCV bar data from a CSV or Parquet file and emits OHLCVSnapshot events.

    Required columns: timestamp, open, high, low, close
    Optional columns: volume, bid, ask
      - If bid/ask are absent, they default to close inside OHLCVSnapshot.
      - Pass bid=low / ask=high for realistic limit-order fill simulation.

    timestamp column must be parseable by pd.to_datetime (ISO-8601 recommended).
    Timestamps are coerced to UTC-aware datetimes.
    """

    def __init__(
        self,
        path: str,
        ticker: str,
        side: Side = Side.YES,
        use_high_low_for_limits: bool = False,
    ) -> None:
        self.path = path
        self.ticker = ticker
        self.side = side
        self.use_high_low_for_limits = use_high_low_for_limits

    def load(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[OHLCVSnapshot]:
        p = Path(self.path)
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)

        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"OHLCV feed {self.path!r} missing columns: {missing}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")

        if start is not None:
            ts = pd.Timestamp(start, tz="UTC") if start.tzinfo is None else pd.Timestamp(start)
            df = df[df["timestamp"] >= ts]
        if end is not None:
            ts = pd.Timestamp(end, tz="UTC") if end.tzinfo is None else pd.Timestamp(end)
            df = df[df["timestamp"] <= ts]

        events: list[OHLCVSnapshot] = []
        for row in df.itertuples(index=False):
            ts = row.timestamp.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            bid = float(row.bid) if hasattr(row, "bid") and pd.notna(row.bid) else None
            ask = float(row.ask) if hasattr(row, "ask") and pd.notna(row.ask) else None

            if self.use_high_low_for_limits and bid is None and ask is None:
                bid = float(row.low)
                ask = float(row.high)

            events.append(OHLCVSnapshot(
                timestamp=ts,
                ticker=self.ticker,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume) if hasattr(row, "volume") and pd.notna(row.volume) else 0.0,
                side=self.side,
                bid=bid,
                ask=ask,
            ))

        logger.info("GenericOHLCVFeed loaded %d bars for %s", len(events), self.ticker)
        return events
