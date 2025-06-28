"""Custom MCP Run Card - Template for custom API connections."""

import logging

logger = logging.getLogger(__name__)

class CustomRunCard:
    """Pre-configured template for custom MCP connections."""
    
    def __init__(self):
        self.card_type = "custom"
        
    def get_config_template(self):
        """Get configuration template for custom connection."""
        return {
            "mcp_type": "custom_api",
            "display_name": "Custom API",
            "description": "Connect to custom API service",
            "config": {
                "base_url": "https://api.example.com",
                "api_key": "your-api-key"
            }
        } 