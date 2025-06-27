"""
Logs Viewer Component

Displays real-time logs from Supabase and system components.
"""

import logging

logger = logging.getLogger(__name__)

class LogsViewer:
    """Views real-time system logs."""
    
    def __init__(self, db_logger=None):
        """Initialize the logs viewer."""
        self.db_logger = db_logger
        logger.info("Logs Viewer initialized") 