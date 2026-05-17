from __future__ import annotations

from typing import Callable, Optional

from core.events import Expiry, Factor, Fill, PredictionMarketSnapshot, Resolution


class Strategy:
    """
    Abstract base for all strategies.

    Subclasses implement the on_* handlers and emit Signal events
    via self._publish to trigger order placement.
    """

    def __init__(self) -> None:
        self._publish: Optional[Callable] = None
        self._equity_callback: Optional[Callable[[], float]] = None

    def set_publish(self, fn: Callable) -> None:
        self._publish = fn

    def set_equity_callback(self, fn: Callable[[], float]) -> None:
        """Called by the backtest engine to allow equity-based position sizing."""
        self._equity_callback = fn

    @property
    def equity(self) -> float:
        return self._equity_callback() if self._equity_callback else 0.0

    def on_market_data(self, event: PredictionMarketSnapshot) -> None:
        pass

    def on_fill(self, event: Fill) -> None:
        pass

    def on_resolution(self, event: Resolution) -> None:
        pass

    def on_expiry(self, event: Expiry) -> None:
        pass

    def on_factor(self, event: Factor) -> None:
        pass
