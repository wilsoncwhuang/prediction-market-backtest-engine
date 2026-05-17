"""
Example strategy: buy YES when ask is in range and factor confidence is high enough.

This is a minimal template demonstrating the Strategy interface, including
how to consume factor data to gate order entry.
Replace the signal logic with your own alpha.
"""
from __future__ import annotations

import logging
from datetime import date

from core.constants import Action, OrderType, Side
from core.events import Expiry, Factor, Fill, PredictionMarketSnapshot, Resolution, Signal
from strategies.base import Strategy

logger = logging.getLogger(__name__)


class ExampleStrategy(Strategy):
    """
    Buys YES on any ticker whose ask falls within [min_ask_cents, max_ask_cents],
    gated by a daily factor (e.g. model confidence score).

    Places at most one order per ticker per day.
    """

    def __init__(
        self,
        min_ask_cents:    float = 30.0,
        max_ask_cents:    float = 70.0,
        quantity:         int   = 10,
        factor_threshold: float = 0.0,   # skip trading days where factor < threshold
        strategy_id:      str   = "",
    ) -> None:
        super().__init__()
        self.min_ask_cents    = min_ask_cents
        self.max_ask_cents    = max_ask_cents
        self.quantity         = quantity
        self.factor_threshold = factor_threshold
        self.strategy_id      = strategy_id
        self._ordered_today: set[tuple[str, date]] = set()
        self._latest_factor: float = 1.0   # default: always trade if no factor feed

    def on_market_data(self, event: PredictionMarketSnapshot) -> None:
        if event.side != Side.YES:
            return

        today = event.timestamp.date()
        if (event.ticker, today) in self._ordered_today:
            return

        if self._latest_factor < self.factor_threshold:
            return

        if not (self.min_ask_cents <= event.ask <= self.max_ask_cents):
            return

        self._ordered_today.add((event.ticker, today))

        if self._publish:
            self._publish(Signal(
                timestamp   = event.timestamp,
                ticker      = event.ticker,
                signal_type = "ask_in_range",
                value       = event.ask,
                strategy_id = self.strategy_id,
                metadata    = {
                    "side":        Side.YES,
                    "action":      Action.BUY,
                    "quantity":    self.quantity,
                    "order_type":  OrderType.LIMIT,
                    "limit_price": event.ask,
                },
            ))
        logger.info("[%s] Signal: %s ask=%.1fc qty=%d factor=%.2f",
                    self.strategy_id, event.ticker, event.ask, self.quantity, self._latest_factor)

    def on_factor(self, event: Factor) -> None:
        self._latest_factor = event.value
        logger.debug("[%s] Factor update: %.4f", self.strategy_id, event.value)

    def on_fill(self, event: Fill) -> None:
        pass

    def on_resolution(self, event: Resolution) -> None:
        pass

    def on_expiry(self, event: Expiry) -> None:
        pass
