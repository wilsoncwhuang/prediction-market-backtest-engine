from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from core.constants import Side
from core.events import PredictionMarketSnapshot


class PredictionMarketPriceFeed:
    """
    Loads a prediction market price history parquet and converts rows to
    PredictionMarketSnapshot events.

    Works with any binary prediction market (Kalshi, Polymarket, PredictIt, etc.)
    as long as the parquet follows the schema below.

    Parquet columns: ts, ticker, bid_close, ask_close, price_close, volume.
    Prices are in dollars (0–1); this feed converts them to cents (0–100).

    Pass side=Side.YES (default) to emit YES snapshots directly.
    Pass side=Side.NO to derive NO prices: no_bid = 1 - yes_ask, no_ask = 1 - yes_bid.
    """

    def __init__(self, parquet_path: Path | str, side: Side = Side.YES) -> None:
        df = pd.read_parquet(parquet_path)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        self._df   = df.sort_values(["ts", "ticker"]).reset_index(drop=True)
        self._side = side

    def load(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        tickers: list[str] | None = None,
    ) -> list[PredictionMarketSnapshot]:
        df = self._df

        if start is not None:
            df = df[df["ts"] >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df["ts"] <= pd.Timestamp(end, tz="UTC")]
        if tickers is not None:
            df = df[df["ticker"].isin(tickers)]

        df = df.dropna(subset=["bid_close", "ask_close"])
        if df.empty:
            return []

        if self._side == Side.YES:
            bids = (df["bid_close"] * 100).round(2)
            asks = (df["ask_close"] * 100).round(2)
        else:
            bids = ((1 - df["ask_close"]) * 100).round(2)
            asks = ((1 - df["bid_close"]) * 100).round(2)

        has_lp      = df["price_close"].notna()
        last_prices = (df["price_close"] * 100).round(2).where(has_lp, other=None)
        volumes     = df["volume"].fillna(0).astype(int)
        side        = self._side

        events = [
            PredictionMarketSnapshot(
                timestamp=ts,
                ticker=ticker,
                side=side,
                bid=bid,
                ask=ask,
                last_price=lp,
                volume=vol,
            )
            for ts, ticker, bid, ask, lp, vol in zip(
                df["ts"].dt.to_pydatetime(),
                df["ticker"],
                bids,
                asks,
                last_prices,
                volumes,
            )
        ]
        return sorted(events)
