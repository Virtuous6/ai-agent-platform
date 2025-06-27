"""
Agent Monitor Component

Monitors active agents and their performance metrics.
"""

import logging

logger = logging.getLogger(__name__)

class AgentMonitor:
    """Monitors agent performance and activity."""
    
    def __init__(self, db_logger=None, orchestrator=None):
        """Initialize the agent monitor."""
        self.db_logger = db_logger
        self.orchestrator = orchestrator
        logger.info("Agent Monitor initialized") 