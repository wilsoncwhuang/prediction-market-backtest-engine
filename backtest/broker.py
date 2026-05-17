from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from core.constants import Action, OrderStatus, OrderType, Side
from core.events import Expiry, Fill, Order, OHLCVSnapshot, Resolution, PredictionMarketSnapshot

MarketSnapshot = Union[PredictionMarketSnapshot, OHLCVSnapshot]

logger = logging.getLogger(__name__)


@dataclass
class BrokerConfig:
    fee_per_contract: float = 0.0        # cents per contract (ignored when use_platform_fee_formula=True)
    slippage_cents: float = 0.0          # added to buy fill price, subtracted from sell fill price
    fill_at_limit: bool = True           # True = fill at limit price, False = fill at ask/bid
    price_floor: float = 1.0             # minimum fill price (1.0 for prediction markets, 0.0 for OHLCV)
    price_ceiling: float = 99.0          # maximum fill price (99.0 for prediction markets, inf for OHLCV)
    use_platform_fee_formula: bool = False  # use taker fee formula: ceil(rate * C * P * (1-P))
    platform_fee_rate: float = 0.07      # fee rate for use_platform_fee_formula (default: Kalshi 7%)


class PredictionSimulatedBroker:
    """
    Simulates order execution for binary prediction markets.

    Side-agnostic by default — handles YES and NO orders for any ticker.
    Snapshots are keyed by (ticker, side) so YES and NO prices are tracked
    independently. The strategy embeds the desired side in each Order.

    Pass market_side to restrict to a single side (useful for single-side
    strategies and backward-compatible unit tests).

    Fill logic:
      BUY:
        MARKET → fill at ask
        LIMIT  → fill if ask <= limit_price
                 fill price = limit_price (fill_at_limit=True, conservative)
                            = ask         (fill_at_limit=False, realistic)
      SELL:
        MARKET → fill at bid
        LIMIT  → fill if bid >= limit_price
                 fill price = limit_price (fill_at_limit=True, conservative)
                            = bid         (fill_at_limit=False, realistic)
    """

    def __init__(
        self,
        market_side: Optional[Side] = None,
        config: Optional[BrokerConfig] = None,
    ) -> None:
        self.market_side = market_side          # None = accept all sides
        self.config = config or BrokerConfig()
        self._snapshots: dict[tuple[str, Side], PredictionMarketSnapshot] = {}
        self._open_orders: dict[str, Order] = {}
        self._publish: Optional[Callable] = None

    def set_publish(self, publish_fn: Callable) -> None:
        """Inject the EventBus.publish function."""
        self._publish = publish_fn

    # ------------------------------------------------------------------
    # Event handlers (subscribe these to EventBus)
    # ------------------------------------------------------------------

    def on_market_data(self, event: MarketSnapshot) -> None:
        """Update latest snapshot and attempt to fill resting limit orders."""
        if self.market_side is not None and event.side != self.market_side:
            return
        self._snapshots[(event.ticker, event.side)] = event
        self._try_fill_open_orders(event.ticker, event.side, event)

    def on_order(self, event: Order) -> None:
        """Reject filtered-side orders; attempt to fill or rest valid orders."""
        if self.market_side is not None and event.side != self.market_side:
            logger.warning(
                "Broker restricted to %s — rejecting %s order %s",
                self.market_side.value, event.side.value, event.order_id,
            )
            event.status = OrderStatus.REJECTED
            return

        snapshot = self._snapshots.get((event.ticker, event.side))
        if snapshot is None:
            logger.warning(
                "No snapshot for (%s, %s) — rejecting order %s",
                event.ticker, event.side.value, event.order_id,
            )
            event.status = OrderStatus.REJECTED
            return

        filled = self._try_fill(event, snapshot)
        if not filled:
            event.status = OrderStatus.OPEN
            self._open_orders[event.order_id] = event
            logger.debug("Order %s resting at %.1f", event.order_id, event.limit_price)

    def on_resolution(self, event: Resolution) -> None:
        """Cancel all open orders for a resolved ticker (any side)."""
        self._cancel_open_orders(event.ticker)

    def on_expiry(self, event: Expiry) -> None:
        """Cancel all open orders for an expired ticker."""
        self._cancel_open_orders(event.ticker)

    # ------------------------------------------------------------------
    # Internal fill logic
    # ------------------------------------------------------------------

    def _cancel_open_orders(self, ticker: str) -> None:
        to_cancel = [oid for oid, o in self._open_orders.items() if o.ticker == ticker]
        for oid in to_cancel:
            order = self._open_orders.pop(oid)
            order.status = OrderStatus.CANCELLED
            logger.debug("Cancelled order %s on settlement of %s", oid, ticker)

    def _try_fill(self, order: Order, snapshot: MarketSnapshot) -> bool:
        """Attempt to fill an order. Returns True if filled."""
        fill_price = self._match_price(order, snapshot)
        if fill_price is None:
            return False

        if order.action == Action.BUY:
            fill_price = min(fill_price + self.config.slippage_cents, self.config.price_ceiling)
        else:
            fill_price = max(fill_price - self.config.slippage_cents, self.config.price_floor)

        if self.config.use_platform_fee_formula:
            # Taker fee formula: ceil(rate * C * P * (1-P)) cents, where P is in dollars (0–1).
            # At rate=0.07 (Kalshi default), max fee is 1.75c per contract at P=0.5.
            p = fill_price / 100.0
            fees = math.ceil(self.config.platform_fee_rate * order.quantity * p * (1 - p) * 100)
        else:
            fees = self.config.fee_per_contract * order.quantity

        fill = Fill(
            timestamp=snapshot.timestamp,
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side,
            action=order.action,
            quantity=order.quantity,
            fill_price=fill_price,
            fees=fees,
            strategy_id=order.strategy_id,
            metadata=order.metadata,
        )

        order.status = OrderStatus.FILLED
        self._publish(fill)
        logger.debug(
            "%s %s %s x%d @ %.1f (fees=%.2f)",
            order.action.value, order.side.value, order.ticker,
            order.quantity, fill_price, fees,
        )
        return True

    def _match_price(
        self, order: Order, snapshot: MarketSnapshot
    ) -> Optional[float]:
        """
        Return fill price if order can be filled, else None.

        BUY:
          MARKET → fill at ask
          LIMIT  → fill if ask <= limit_price
                   fill_at_limit=True → limit_price, False → ask

        SELL:
          MARKET → fill at bid
          LIMIT  → fill if bid >= limit_price
                   fill_at_limit=True → limit_price, False → bid
        """
        if order.action == Action.BUY:
            ask = snapshot.ask
            if order.order_type == OrderType.MARKET:
                return ask
            if ask <= order.limit_price:
                return order.limit_price if self.config.fill_at_limit else ask
        else:
            bid = snapshot.bid
            if order.order_type == OrderType.MARKET:
                return bid
            if bid >= order.limit_price:
                return order.limit_price if self.config.fill_at_limit else bid

        return None

    def _try_fill_open_orders(
        self, ticker: str, side: Side, snapshot: MarketSnapshot
    ) -> None:
        """Check resting limit orders for (ticker, side) against new snapshot."""
        to_fill = [
            oid for oid, o in self._open_orders.items()
            if o.ticker == ticker and o.side == side
        ]
        for oid in to_fill:
            order = self._open_orders[oid]
            if self._try_fill(order, snapshot):
                del self._open_orders[oid]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def open_orders(self) -> dict[str, Order]:
        return dict(self._open_orders)

    @property
    def open_order_count(self) -> int:
        return len(self._open_orders)
