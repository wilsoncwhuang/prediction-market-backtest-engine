from __future__ import annotations

import heapq
import logging
from collections import defaultdict
from typing import Callable

from core.events import Event
from core.constants import EventType

logger = logging.getLogger(__name__)

Handler = Callable[[Event], None]


class EventBus:
    """
    Priority-queue-based event bus for the backtest engine.

    Events are processed in timestamp order (earliest first).
    Handlers subscribe to specific EventTypes and are called synchronously
    when their event type is published.
    """

    def __init__(self) -> None:
        self._queue: list[tuple] = []   # (timestamp, counter, event)
        self._counter: int = 0
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._processed: int = 0

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Register a handler for a given EventType."""
        self._handlers[event_type].append(handler)
        logger.debug("Subscribed %s to %s", handler, event_type)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        """Remove a previously registered handler."""
        handlers = self._handlers[event_type]
        if handler in handlers:
            handlers.remove(handler)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, event: Event) -> None:
        """Push an event onto the priority queue (ordered by timestamp)."""
        heapq.heappush(self._queue, (event.timestamp, self._counter, event))
        self._counter += 1

    def publish_many(self, events: list[Event]) -> None:
        """Push multiple events at once."""
        for event in events:
            heapq.heappush(self._queue, (event.timestamp, self._counter, event))
            self._counter += 1

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Drain the queue, dispatching each event to its handlers."""
        while self._queue:
            _, _, event = heapq.heappop(self._queue)
            self._dispatch(event)
            self._processed += 1

    def step(self) -> bool:
        """
        Process a single event.
        Returns True if an event was processed, False if queue is empty.
        """
        if not self._queue:
            return False
        _, _, event = heapq.heappop(self._queue)
        self._dispatch(event)
        self._processed += 1
        return True

    def _dispatch(self, event: Event) -> None:
        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            logger.debug("No handlers for %s", event.event_type)
            return
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Handler %s raised while processing %s", handler, event
                )
                raise

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def pending(self) -> int:
        """Number of events still in the queue."""
        return len(self._queue)

    @property
    def processed(self) -> int:
        """Total events dispatched so far."""
        return self._processed

    def clear(self) -> None:
        """Discard all queued events (does not reset processed count)."""
        self._queue.clear()

    def reset(self) -> None:
        """Full reset — clear queue and processed counter."""
        self._queue.clear()
        self._counter = 0
        self._processed = 0
