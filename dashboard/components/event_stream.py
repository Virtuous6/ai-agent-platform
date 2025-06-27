"""
Event Stream Viewer Component

Displays real-time event streams from the event bus.
"""

import logging

logger = logging.getLogger(__name__)

class EventStreamViewer:
    """Views real-time event streams."""
    
    def __init__(self, event_bus=None):
        """Initialize the event stream viewer."""
        self.event_bus = event_bus
        logger.info("Event Stream Viewer initialized") 