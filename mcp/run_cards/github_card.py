"""GitHub MCP Run Card - Quick setup for GitHub connections."""

import logging

logger = logging.getLogger(__name__)

class GitHubRunCard:
    """Pre-configured template for GitHub MCP connections."""
    
    def __init__(self):
        self.card_type = "github"
        
    def get_config_template(self):
        """Get configuration template for GitHub connection."""
        return {
            "mcp_type": "github",
            "display_name": "GitHub Repository",
            "description": "Connect to GitHub repositories",
            "config": {
                "access_token": "your-github-token",
                "organization": "your-org"
            }
        } 