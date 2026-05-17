from __future__ import annotations

import logging
from datetime import datetime

from typing import Protocol, runtime_checkable

from backtest.broker import PredictionSimulatedBroker
from core.event_bus import EventBus
from core.portfolio import Portfolio
from core.risk_manager import PassthroughRiskManager
from strategies.base import Strategy
from core.constants import EventType


@runtime_checkable
class Feed(Protocol):
    def load(self, start: datetime | None = None, end: datetime | None = None) -> list: ...

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Wires together the event bus, broker, portfolio, feeds, and strategy,
    then runs the backtest by draining the event queue.
    """

    def __init__(
        self,
        bus: EventBus,
        broker: PredictionSimulatedBroker,
        portfolio: Portfolio,
        strategy: Strategy,
        feeds: list[Feed],
        risk_manager: PassthroughRiskManager | None = None,
    ) -> None:
        self.bus = bus
        self.broker = broker
        self.portfolio = portfolio
        self.strategy = strategy
        self.feeds = feeds
        self.risk_manager = risk_manager or PassthroughRiskManager()
        self._wired = False

    def _wire(self) -> None:
        if self._wired:
            return
        self.broker.set_publish(self.bus.publish)
        self.bus.subscribe(EventType.MARKET_DATA, self.broker.on_market_data)
        self.bus.subscribe(EventType.ORDER,       self.broker.on_order)
        self.bus.subscribe(EventType.RESOLUTION,  self.broker.on_resolution)
        self.bus.subscribe(EventType.EXPIRY,      self.broker.on_expiry)

        self.bus.subscribe(EventType.MARKET_DATA, self.portfolio.on_market_data)
        self.bus.subscribe(EventType.FILL,        self.portfolio.on_fill)
        self.bus.subscribe(EventType.RESOLUTION,  self.portfolio.on_resolution)
        self.bus.subscribe(EventType.EXPIRY,      self.portfolio.on_expiry)

        self.strategy.set_publish(self.bus.publish)
        self.bus.subscribe(EventType.MARKET_DATA, self.strategy.on_market_data)
        self.bus.subscribe(EventType.FILL,        self.strategy.on_fill)
        self.bus.subscribe(EventType.RESOLUTION,  self.strategy.on_resolution)
        self.bus.subscribe(EventType.EXPIRY,      self.strategy.on_expiry)

        self.risk_manager.set_publish(self.bus.publish)
        self.bus.subscribe(EventType.SIGNAL,  self.risk_manager.on_signal)
        self.bus.subscribe(EventType.FACTOR,  self.strategy.on_factor)
        self._wired = True

    def run(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> None:
        self._wire()

        # Load events from all feeds
        total = 0
        for feed in self.feeds:
            events = feed.load(start=start, end=end)
            self.bus.publish_many(events)
            total += len(events)
        logger.info("Loaded %d events from %d feed(s)", total, len(self.feeds))

        # Run
        self.bus.run()
        logger.info(
            "Backtest complete — %d events processed | equity=%.2f total_pnl=%.2f",
            self.bus.processed, self.portfolio.equity, self.portfolio.total_pnl,
        )
