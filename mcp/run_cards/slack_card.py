"""Slack MCP Run Card - Quick setup for Slack connections."""

import logging

logger = logging.getLogger(__name__)

class SlackRunCard:
    """Pre-configured template for Slack MCP connections."""
    
    def __init__(self):
        self.card_type = "slack"
        
    def get_config_template(self):
        """Get configuration template for Slack connection."""
        return {
            "mcp_type": "slack",
            "display_name": "Slack Workspace",
            "description": "Connect to Slack workspace",
            "config": {
                "bot_token": "xoxb-your-bot-token",
                "workspace": "your-workspace"
            }
        } 