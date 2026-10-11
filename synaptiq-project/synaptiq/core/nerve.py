"""The central nervous system - async event bus for inter-component communication."""
import asyncio
from typing import Any, Dict, List, Callable, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class EventBus:
    """The central nervous system of Synaptiq."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def subscribe(self, event_type: str, callback: Callable):
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        if event_type in self._subscribers:
            try: self._subscribers[event_type].remove(callback)
            except ValueError: pass

    async def publish(self, event_type: str, data: Dict[str, Any]):
        await self._queue.put({"type": event_type, "data": data})

    async def start(self):
        self._running = True
        while self._running:
            event = await self._queue.get()
            if event is None: break
            event_type = event["type"]
            data = event["data"]
            for callback in self._subscribers.get(event_type, []):
                try:
                    if asyncio.iscoroutinefunction(callback): await callback(data)
                    else: callback(data)
                except Exception as e:
                    logger.error(f"Error processing event {event_type}: {e}")
            self._queue.task_done()

    async def stop(self):
        self._running = False
        await self._queue.put(None)
        if self._task: await self._task

    async def wait_until_empty(self):
        await self._queue.join()
