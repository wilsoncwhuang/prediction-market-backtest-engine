from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from core.constants import Action, Direction, EventType, OrderStatus, OrderType, Side


# ---------------------------------------------------------------------------
# Base event — everything flowing through the event bus is an Event
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Event:
    timestamp: datetime
    event_type: Optional[EventType] = field(init=False, default=None, compare=False)

    def __post_init__(self):
        if not isinstance(self.timestamp, datetime):
            raise TypeError(f"timestamp must be datetime, got {type(self.timestamp)}")


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

@dataclass(order=True)
class PredictionMarketSnapshot(Event):
    ticker: str = field(compare=False)
    side: Side = field(compare=False)
    bid: float = field(compare=False)           # best bid (cents)
    ask: float = field(compare=False)           # best ask (cents)
    last_price: Optional[float] = field(compare=False, default=None)
    volume: int = field(compare=False, default=0)
    is_seed: bool = field(compare=False, default=False)

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.MARKET_DATA

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


# ---------------------------------------------------------------------------
# OHLCV market data — generic financial instruments
# ---------------------------------------------------------------------------

@dataclass(order=True)
class OHLCVSnapshot(Event):
    """
    OHLCV bar for general financial instruments (equities, futures, forex).

    bid/ask default to close if not provided, so the existing broker fill
    logic works out-of-the-box. For more realistic limit fills, pass
    bid=low and ask=high to represent the bar's range.
    """
    ticker: str = field(compare=False)
    open: float = field(compare=False)
    high: float = field(compare=False)
    low: float = field(compare=False)
    close: float = field(compare=False)
    volume: float = field(compare=False, default=0.0)
    side: Side = field(compare=False, default=Side.YES)   # YES = long position tracking
    bid: Optional[float] = field(compare=False, default=None)
    ask: Optional[float] = field(compare=False, default=None)

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.MARKET_DATA
        if self.bid is None:
            self.bid = self.close
        if self.ask is None:
            self.ask = self.close

    @property
    def mid(self) -> float:
        return self.close

    @property
    def spread(self) -> float:
        return self.ask - self.bid


# ---------------------------------------------------------------------------
# Factor — external data consumed by strategy (forecast, observation, etc.)
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Factor(Event):
    series: str = field(compare=False)          # e.g. "KXHIGHNY"
    target_date: date = field(compare=False)    # date this factor applies to
    value: float = field(compare=False)         # factor value (e.g. predicted high temp °F)
    model_id: str = field(compare=False, default="")

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.FACTOR


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Signal(Event):
    ticker: str = field(compare=False)
    signal_type: str = field(compare=False)      # e.g. "prob_yes", "edge", "z_score"
    value: float = field(compare=False)          # signal value
    strategy_id: str = field(compare=False, default="")
    confidence: Optional[float] = field(compare=False, default=None)
    metadata: dict = field(compare=False, default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.SIGNAL


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Order(Event):
    order_id: str = field(compare=False)
    ticker: str = field(compare=False)
    side: Side = field(compare=False)
    action: Action = field(compare=False)
    quantity: int = field(compare=False)        # number of contracts
    order_type: OrderType = field(compare=False, default=OrderType.LIMIT)
    limit_price: Optional[float] = field(compare=False, default=None)  # cents
    status: OrderStatus = field(compare=False, default=OrderStatus.PENDING)
    strategy_id: str = field(compare=False, default="")
    metadata: dict = field(compare=False, default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.ORDER
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price required for LIMIT orders")


# ---------------------------------------------------------------------------
# Fill — confirmed execution
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Fill(Event):
    order_id: str = field(compare=False)
    ticker: str = field(compare=False)
    side: Side = field(compare=False)
    action: Action = field(compare=False)
    quantity: int = field(compare=False)
    fill_price: float = field(compare=False)    # cents (1–99)
    fees: float = field(compare=False, default=0.0)
    strategy_id: str = field(compare=False, default="")
    metadata: dict = field(compare=False, default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.FILL

    @property
    def notional(self) -> float:
        """Gross notional value in cents (fill_price × quantity), excluding fees."""
        return self.fill_price * self.quantity


# ---------------------------------------------------------------------------
# Trade — closed round-trip (entry Fill + exit Fill collapsed)
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    ticker: str
    side: Side
    quantity: int
    entry_price: float      # cents
    exit_price: float       # cents
    entry_time: datetime
    exit_time: datetime
    fees: float = 0.0
    strategy_id: str = ""

    @property
    def pnl(self) -> float:
        """PnL in cents per contract × quantity, minus fees."""
        if self.side == Side.YES:
            raw = (self.exit_price - self.entry_price) * self.quantity
        else:
            raw = (self.entry_price - self.exit_price) * self.quantity
        return raw - self.fees

    @property
    def duration_seconds(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds()


# ---------------------------------------------------------------------------
# Resolution — binary prediction market settlement
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Resolution(Event):
    ticker: str = field(compare=False)
    result: bool = field(compare=False)         # True = YES wins
    pnl: float = field(compare=False, default=0.0)  # total settlement PnL (cents), set by portfolio

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.RESOLUTION


# ---------------------------------------------------------------------------
# Expiry — general instrument settlement at a specific price
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Expiry(Event):
    """
    Settlement event for general financial instruments.

    settle_price is the final settlement price per unit (e.g. futures expiry,
    option exercise, or end-of-backtest mark). The portfolio settles all open
    positions for the ticker at this price.
    """
    ticker: str = field(compare=False)
    settle_price: float = field(compare=False)

    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.EXPIRY
