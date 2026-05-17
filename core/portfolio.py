from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from typing import Union

from core.constants import Action, Side
from core.events import Expiry, Fill, OHLCVSnapshot, PredictionMarketSnapshot, Resolution

MarketSnapshot = Union[PredictionMarketSnapshot, OHLCVSnapshot]

logger = logging.getLogger(__name__)


@dataclass
class Position:
    ticker: str
    quantity: int = 0           # always non-negative; side tracked by the portfolio key
    avg_cost: float = 0.0       # cents
    realized_pnl: float = 0.0  # cents

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    def market_value(self, mid: float) -> float:
        """Current mark-to-market value in cents."""
        return self.quantity * mid

    def unrealized_pnl(self, mid: float) -> float:
        return self.quantity * (mid - self.avg_cost)


@dataclass
class EquityPoint:
    timestamp: datetime
    cash: float
    position_value: float
    realized_pnl: float
    unrealized_pnl: float

    @property
    def equity(self) -> float:
        return self.cash + self.position_value


class Portfolio:
    """
    Tracks cash, positions, and PnL throughout a backtest.

    Positions are keyed by (ticker, side) so YES and NO on the same
    ticker are tracked independently.

    Subscribes to:
        FILL        — update cash and positions on execution
        MARKET_DATA — mark-to-market positions
        RESOLUTION  — settle both YES and NO positions for a ticker

    Equity curve is recorded on every MARKET_DATA event.

    avg_cost update rules (BUY fills):
        - Flat → opening: set avg_cost = fill_price
        - Adding to existing: weighted average

    SELL fills:
        - Realized PnL booked immediately at fill time
        - avg_cost of remaining position unchanged
        - avg_cost reset to 0 when position goes flat
    """

    def __init__(self, initial_cash: float = 10_000.0) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self._positions: dict[tuple[str, Side, str], Position] = {}
        self._mids: dict[tuple[str, Side], float] = {}
        self.equity_curve: list[EquityPoint] = []
        self.trade_log: list[dict] = []

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_fill(self, event: Fill) -> None:
        key = (event.ticker, event.side, event.strategy_id)
        pos = self._positions.setdefault(key, Position(ticker=event.ticker))

        realized_pnl_this_fill = 0.0

        if event.action == Action.BUY:
            cost = event.fill_price * event.quantity + event.fees
            if pos.quantity == 0:
                pos.avg_cost = event.fill_price
            else:
                # Adding to existing position — weighted average
                total_qty = pos.quantity + event.quantity
                pos.avg_cost = (
                    pos.avg_cost * pos.quantity + event.fill_price * event.quantity
                ) / total_qty
            pos.quantity += event.quantity
            self.cash -= cost
        else:
            # SELL: close part or all of position
            proceeds = event.fill_price * event.quantity - event.fees
            realized_pnl_this_fill = (event.fill_price - pos.avg_cost) * event.quantity
            pos.realized_pnl += realized_pnl_this_fill
            pos.quantity -= event.quantity
            if pos.quantity == 0:
                pos.avg_cost = 0.0
            self.cash += proceeds

        self.trade_log.append({
            "timestamp":    event.timestamp,
            "ticker":       event.ticker,
            "side":         event.side.value,
            "action":       event.action.value,
            "quantity":     event.quantity,
            "fill_price":   event.fill_price,
            "fees":         event.fees,
            "realized_pnl": realized_pnl_this_fill,
            **event.metadata,
        })

        logger.debug(
            "%s %s fill: %s qty=%d avg_cost=%.1f cash=%.1f",
            event.action.value, event.side.value, event.ticker,
            pos.quantity, pos.avg_cost, self.cash,
        )

    def on_market_data(self, event: MarketSnapshot) -> None:
        self._mids[(event.ticker, event.side)] = event.mid
        self._record_equity(event.timestamp)

    def on_resolution(self, event: Resolution) -> None:
        for (ticker, side, strategy_id), pos in list(self._positions.items()):
            if ticker != event.ticker or pos.is_flat:
                continue
            key = (ticker, side, strategy_id)

            # YES wins (result=True): YES → 100, NO → 0
            # NO wins (result=False): YES → 0,   NO → 100
            settle_price = 100.0 if (event.result == (side == Side.YES)) else 0.0
            settlement_cash = settle_price * pos.quantity
            pnl = settlement_cash - pos.avg_cost * pos.quantity
            pos.realized_pnl += pnl
            self.cash += settlement_cash
            event.pnl += pnl

            self.trade_log.append({
                "timestamp":    event.timestamp,
                "ticker":       event.ticker,
                "side":         side.value,
                "action":       "settlement",
                "quantity":     pos.quantity,
                "fill_price":   settle_price,
                "fees":         0.0,
                "realized_pnl": pnl,
            })

            logger.info(
                "[%s] Resolution: %s -> %s | settlement=%.1f pnl=%.1f cents",
                event.ticker, side.value, "YES" if event.result else "NO",
                settlement_cash, pnl,
            )

            pos.quantity = 0
            pos.avg_cost = 0.0
            self._mids.pop(key, None)

    def on_expiry(self, event: Expiry) -> None:
        """Settle all open positions for ticker at settle_price."""
        for (ticker, side, strategy_id), pos in list(self._positions.items()):
            if ticker != event.ticker or pos.is_flat:
                continue
            key = (ticker, side, strategy_id)

            settlement_cash = event.settle_price * pos.quantity
            pnl = settlement_cash - pos.avg_cost * pos.quantity
            pos.realized_pnl += pnl
            self.cash += settlement_cash

            self.trade_log.append({
                "timestamp":    event.timestamp,
                "ticker":       event.ticker,
                "side":         side.value,
                "action":       "expiry",
                "quantity":     pos.quantity,
                "fill_price":   event.settle_price,
                "fees":         0.0,
                "realized_pnl": pnl,
            })

            logger.info(
                "[%s] Expiry: %s settle=%.4f | cash_in=%.4f pnl=%.4f",
                event.ticker, side.value, event.settle_price, settlement_cash, pnl,
            )

            pos.quantity = 0
            pos.avg_cost = 0.0
            self._mids.pop(key, None)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def positions(self) -> dict[tuple[str, Side, str], Position]:
        return {k: p for k, p in self._positions.items() if not p.is_flat}

    @property
    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self._positions.values())

    @property
    def unrealized_pnl(self) -> float:
        return sum(
            p.unrealized_pnl(self._mids[k])
            for k, p in self._positions.items()
            if not p.is_flat and k in self._mids
        )

    @property
    def position_value(self) -> float:
        return sum(
            p.market_value(self._mids[k])
            for k, p in self._positions.items()
            if not p.is_flat and k in self._mids
        )

    @property
    def equity(self) -> float:
        return self.cash + self.position_value

    @property
    def total_pnl(self) -> float:
        return self.equity - self.initial_cash

    def _record_equity(self, timestamp: datetime) -> None:
        self.equity_curve.append(EquityPoint(
            timestamp=timestamp,
            cash=self.cash,
            position_value=self.position_value,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
        ))
