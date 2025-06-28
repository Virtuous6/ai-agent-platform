"""MCP Tool Browser - Slack interface for browsing available tools."""

import logging

logger = logging.getLogger(__name__)

class MCPToolBrowser:
    """Slack interface for browsing MCP tools."""
    
    def __init__(self):
        self.browser_type = "mcp_tools"
        
    async def get_tool_list(self):
        """Get list of available tools."""
        return {"tools": []} 