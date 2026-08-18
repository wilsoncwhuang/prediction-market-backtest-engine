"""
Example strategy: tennis win-probability vs. market-implied probability.

ILLUSTRATIVE DEMO — NOT a profit claim. This shows how to feed an external
per-match signal through the engine's existing Factor feed and compare it to a
binary market's implied probability. Replace the sample signal and data with
your own before drawing any conclusion.

Idea
----
A binary "will PLAYER win this match?" market prices an implied probability in
its YES ask (cents -> ask/100). We treat the daily Factor value as an
independent live-tennis win-probability signal for that YES side (e.g. a
ranking/Elo/H2H-derived prior, or the paid model win-probability). When the
signal is higher than the market-implied probability by at least `edge_cents`,
the market looks cheap on YES and we buy one lot.

The engine owns fills, resolution and settlement; this strategy only emits a
Signal from the two inputs it is given. Disclosure: contributed by the Live
Tennis API team (https://livetennisapi.com) — the tennis signal is a data
input, not the market or the executor.
"""
from __future__ import annotations

import logging
from datetime import date

from core.constants import Action, OrderType, Side
from core.events import Factor, Fill, PredictionMarketSnapshot, Resolution, Signal
from strategies.base import Strategy

logger = logging.getLogger(__name__)


class TennisWinProbStrategy(Strategy):
    """
    Buys YES on a tennis match market when a live-tennis win-probability signal
    exceeds the market-implied probability (YES ask) by at least `edge_cents`.

    The signal arrives as a daily Factor value in [0, 1] (win-prob of the YES
    side); the implied probability is the YES ask in cents / 100. Places at most
    one order per ticker per day, mirroring the ExampleStrategy contract.
    """

    def __init__(
        self,
        edge_cents:    float = 5.0,    # required edge (signal% - implied%) in cents
        quantity:      int   = 10,
        max_ask_cents: float = 95.0,   # skip near-certain YES (little to gain)
        min_ask_cents: float = 5.0,    # skip near-zero YES (illiquid tails)
        strategy_id:   str   = "",
    ) -> None:
        super().__init__()
        self.edge_cents    = edge_cents
        self.quantity      = quantity
        self.max_ask_cents = max_ask_cents
        self.min_ask_cents = min_ask_cents
        self.strategy_id   = strategy_id
        self._ordered_today: set[tuple[str, date]] = set()
        # win-prob signal for the YES side, 0..1; None until the first Factor
        self._signal: float | None = None

    def on_market_data(self, event: PredictionMarketSnapshot) -> None:
        if event.side != Side.YES:
            return
        if self._signal is None:
            return   # no tennis signal yet for this day

        today = event.timestamp.date()
        if (event.ticker, today) in self._ordered_today:
            return

        if not (self.min_ask_cents <= event.ask <= self.max_ask_cents):
            return

        implied_cents = event.ask                 # YES ask already in cents
        signal_cents  = self._signal * 100.0      # win-prob -> cents
        edge          = signal_cents - implied_cents
        if edge < self.edge_cents:
            return

        self._ordered_today.add((event.ticker, today))

        if self._publish:
            self._publish(Signal(
                timestamp   = event.timestamp,
                ticker      = event.ticker,
                signal_type = "winprob_edge",
                value       = edge,
                strategy_id = self.strategy_id,
                confidence  = self._signal,
                metadata    = {
                    "side":        Side.YES,
                    "action":      Action.BUY,
                    "quantity":    self.quantity,
                    "order_type":  OrderType.LIMIT,
                    "limit_price": event.ask,
                },
            ))
        logger.info(
            "[%s] Signal: %s implied=%.1fc signal=%.1fc edge=%.1fc qty=%d",
            self.strategy_id, event.ticker, implied_cents, signal_cents,
            edge, self.quantity,
        )

    def on_factor(self, event: Factor) -> None:
        # Factor value is the live-tennis win-probability of the YES side (0..1).
        self._signal = event.value
        logger.debug("[%s] Tennis signal update: %.4f", self.strategy_id, event.value)

    def on_fill(self, event: Fill) -> None:
        pass

    def on_resolution(self, event: Resolution) -> None:
        pass
