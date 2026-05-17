from enum import Enum, auto


class Side(Enum):
    YES = "yes"
    NO = "no"


class Direction(Enum):
    """Direction for general financial instruments (non-binary markets)."""
    LONG = "long"
    SHORT = "short"


class EventType(Enum):
    MARKET_DATA = auto()   # price/orderbook snapshot
    FACTOR = auto()        # external factor consumed by strategy (forecast, observation, etc.)
    SIGNAL = auto()        # strategy trade decision → risk manager
    ORDER = auto()         # strategy → broker
    FILL = auto()          # broker → strategy
    RESOLUTION = auto()    # prediction market settled (binary YES/NO)
    EXPIRY = auto()        # general instrument expiry/settlement


class OrderStatus(Enum):
    PENDING = auto()
    OPEN = auto()
    FILLED = auto()
    PARTIALLY_FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()


class Action(Enum):
    BUY = "buy"
    SELL = "sell"
