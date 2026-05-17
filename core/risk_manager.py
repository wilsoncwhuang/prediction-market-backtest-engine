from __future__ import annotations

import logging
import uuid
from typing import Callable, Optional

from core.constants import EventType, OrderType
from core.events import Order, Signal

logger = logging.getLogger(__name__)


class PassthroughRiskManager:
    """
    Converts Signal events into Order events with no sizing or risk logic.

    Reads order parameters from Signal.metadata:
        side        : Side
        action      : Action
        quantity    : int
        order_type  : OrderType  (default LIMIT)
        limit_price : float | None

    strategy_id is taken from Signal.strategy_id if not in metadata.

    This is a placeholder — replace with real position sizing and risk
    checks when ready.
    """

    def __init__(self) -> None:
        self._publish: Optional[Callable] = None

    def set_publish(self, publish_fn: Callable) -> None:
        self._publish = publish_fn

    def on_signal(self, event: Signal) -> None:
        meta = event.metadata
        try:
            order = Order(
                timestamp=event.timestamp,
                order_id=str(uuid.uuid4()),
                ticker=event.ticker,
                side=meta["side"],
                action=meta["action"],
                quantity=meta["quantity"],
                order_type=meta.get("order_type", OrderType.LIMIT),
                limit_price=meta.get("limit_price"),
                strategy_id=meta.get("strategy_id", event.strategy_id),
                metadata={k: v for k, v in meta.items()
                          if k not in {"side", "action", "quantity", "order_type",
                                       "limit_price", "strategy_id"}},
            )
        except KeyError as exc:
            logger.warning("Signal %s missing metadata key %s — skipped", event, exc)
            return

        self._publish(order)
        logger.debug(
            "Signal → Order: %s %s %s x%d @ %s",
            order.action.value, order.side.value, order.ticker,
            order.quantity, order.limit_price,
        )
