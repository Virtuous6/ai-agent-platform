"""MCP Connection Modal - Slack modal for creating connections."""

import logging

logger = logging.getLogger(__name__)

class MCPConnectionModal:
    """Slack modal for MCP connection creation."""
    
    def __init__(self):
        self.modal_type = "mcp_connection"
        
    def get_modal_config(self):
        """Get modal configuration."""
        return {"title": "Create MCP Connection", "fields": []} 