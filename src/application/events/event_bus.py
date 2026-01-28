"""
Event Bus - Central Event Dispatcher
Implements Observer pattern for loose coupling
"""
from typing import Callable, Dict, List, Type
import asyncio
import logging

from .base_event import Event


class EventBus:
    """
    Central event dispatcher for loose coupling.
    Implements Observer pattern for extensibility.
    """
    
    def __init__(self):
        self._handlers: Dict[Type[Event], List[Callable]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._logger = logging.getLogger(__name__)
        self._task = None
    
    def subscribe(self, event_type: Type[Event], handler: Callable) -> None:
        """
        Subscribe a handler to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Callable to handle the event
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        self._logger.debug(f"Handler {handler.__name__} subscribed to {event_type.__name__}")
    
    def unsubscribe(self, event_type: Type[Event], handler: Callable) -> None:
        """
        Unsubscribe a handler from an event type.
        
        Args:
            event_type: Type of event
            handler: Handler to remove
        """
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                self._logger.debug(f"Handler {handler.__name__} unsubscribed from {event_type.__name__}")
            except ValueError:
                pass
    
    async def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        await self._queue.put(event)
        self._logger.debug(f"Event published: {type(event).__name__}")
    
    def publish_sync(self, event: Event) -> None:
        """
        Synchronous publish (creates task if event loop exists).
        
        Args:
            event: Event to publish
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.publish(event))
            else:
                loop.run_until_complete(self.publish(event))
        except RuntimeError:
            # No event loop, create one
            asyncio.run(self.publish(event))
    
    async def start(self) -> None:
        """Start processing events"""
        self._running = True
        self._logger.info("Event bus started")
        
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                await self._dispatch(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self._logger.error(f"Event bus error: {e}")
    
    async def _dispatch(self, event: Event) -> None:
        """
        Dispatch event to registered handlers.
        
        Args:
            event: Event to dispatch
        """
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])
        
        self._logger.debug(f"Dispatching {event_type.__name__} to {len(handlers)} handlers")
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self._logger.error(f"Handler error in {handler.__name__}: {e}")
    
    def stop(self) -> None:
        """Stop the event bus"""
        self._running = False
        self._logger.info("Event bus stopped")
    
    def clear(self) -> None:
        """Clear all handlers"""
        self._handlers.clear()
        self._logger.info("All handlers cleared")
