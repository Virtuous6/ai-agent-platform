"""
Events package for the AI Agent Platform.

This package provides event-driven communication between agents and components,
enabling loose coupling and better scalability.
"""

from .event_bus import EventBus, Event, EventHandler, EventType

__all__ = [
    'EventBus',
    'Event', 
    'EventHandler',
    'EventType'
] 