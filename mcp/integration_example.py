"""
MCP Integration Example

Demonstrates how to integrate MCP functionality into the existing Slack bot.
This file shows the changes needed in slack_bot.py to add MCP support.
"""

import logging
from typing import Dict, Any

from database.supabase_logger import SupabaseLogger
from mcp.slack_interface.mcp_commands import MCPSlackCommands

logger = logging.getLogger(__name__)

class MCPIntegratedSlackBot:
    """
    Example of how to integrate MCP into the existing AIAgentSlackBot.
    
    This demonstrates the minimal changes needed to add MCP functionality
    to the current slack_bot.py implementation.
    """
    
    def __init__(self, supabase_logger: SupabaseLogger):
        """Initialize the MCP-integrated Slack bot."""
        self.supabase_logger = supabase_logger
        
        # Initialize MCP commands handler
        self.mcp_commands = MCPSlackCommands(supabase_logger)
        
        logger.info("🔌 MCP Integration initialized")
    
    def register_mcp_handlers(self, slack_app):
        """
        Register MCP handlers with the Slack app.
        
        Add this method call to the existing AIAgentSlackBot.__init__():
        
        ```python
        # In AIAgentSlackBot.__init__(), after existing handler registration:
        
        # Initialize MCP integration
        self.mcp_integration = MCPIntegratedSlackBot(self.supabase_logger)
        self.mcp_integration.register_mcp_handlers(self.app)
        ```
        """
        # Register all MCP command handlers
        self.mcp_commands.register_handlers(slack_app)
        
        logger.info("✅ MCP handlers registered with Slack app")
    
    async def enhance_agent_with_mcp_tools(self, agent, user_id: str, context: Dict[str, Any]):
        """
        Enhance an agent with user's MCP tools.
        
        Add this to the existing message processing flow:
        
        ```python
        # In _handle_message(), before calling orchestrator:
        
        # Enhance context with MCP tools
        if hasattr(self, 'mcp_integration'):
            context = await self.mcp_integration.enhance_agent_with_mcp_tools(
                agent, user_id, context
            )
        ```
        """
        try:
            # Get user's MCP connections
            connections = await self.mcp_commands.connection_manager.get_user_connections(user_id)
            
            if not connections:
                return context
            
            # Collect all available tools
            available_tools = []
            for connection in connections:
                if connection.status == 'active' and connection.tools_available:
                    for tool_name in connection.tools_available:
                        available_tools.append({
                            "name": tool_name,
                            "connection_id": connection.id,
                            "connection_name": connection.connection_name,
                            "mcp_type": connection.mcp_type,
                            "description": self._get_tool_description(tool_name, connection.mcp_type)
                        })
            
            # Add tools to context
            context["mcp_tools_available"] = available_tools
            context["mcp_connections_count"] = len(connections)
            
            logger.info(f"🔧 Enhanced agent with {len(available_tools)} MCP tools from {len(connections)} connections")
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Failed to enhance agent with MCP tools: {str(e)}")
            return context
    
    def _get_tool_description(self, tool_name: str, mcp_type: str) -> str:
        """Get description for an MCP tool."""
        descriptions = {
            'supabase': {
                'list_tables': 'Get all tables in the database',
                'execute_sql': 'Run custom SQL queries',
                'get_schema': 'Get table schemas and relationships',
                'insert_data': 'Insert new records',
                'update_data': 'Update existing records'
            },
            'github': {
                'search_repos': 'Search repositories',
                'get_issues': 'Get repository issues', 
                'create_issue': 'Create new issues',
                'get_pull_requests': 'Get pull requests'
            },
            'slack': {
                'send_message': 'Send messages to channels',
                'get_channels': 'List workspace channels',
                'get_users': 'Get workspace members',
                'search_messages': 'Search conversation history'
            }
        }
        
        return descriptions.get(mcp_type, {}).get(tool_name, 'External service tool')


# Example updated app home with MCP status
def get_enhanced_app_home_blocks(mcp_connections_count: int = 0) -> list:
    """
    Enhanced app home blocks that include MCP status.
    
    Replace the existing home_view blocks in _handle_app_home():
    """
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Welcome to the AI Agent Platform!* 🤖\n\nI can help you with various tasks through specialized agents:"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "• *General Agent* ✅: Everyday conversations and assistance\n• *Technical Agent* 🔧: Technical support and development help\n• *Research Agent* 🔍: Research and analysis tasks"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Just mention me in any channel or send me a direct message to get started!\n\n💡 *Tip*: I automatically route your requests to the best agent based on content analysis."
            }
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*🔌 MCP Connections*\nYou have {mcp_connections_count} external service connections.\n Use `/mcp connect` to add more tools to your agents!"
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "🔗 Manage"},
                "action_id": "open_mcp_manager"
            }
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*System Status*\n🔗 Orchestrator: ✅ Active\n📊 Logging: ✅ Connected\n🔌 MCP: {'✅ Active' if mcp_connections_count > 0 else '💤 Ready'}"
            }
        }
    ]
    
    return blocks


# Example usage in existing Slack bot
"""
To integrate MCP into the existing slack_bot.py, make these changes:

1. **In AIAgentSlackBot.__init__()**, after the existing initialization:

```python
# Add MCP integration
try:
    from mcp.integration_example import MCPIntegratedSlackBot
    self.mcp_integration = MCPIntegratedSlackBot(self.supabase_logger)
    self.mcp_integration.register_mcp_handlers(self.app)
    logger.info("🔌 MCP integration enabled")
except Exception as e:
    logger.warning(f"MCP integration disabled: {str(e)}")
    self.mcp_integration = None
```

2. **In _handle_message()**, before calling the orchestrator:

```python
# Enhance context with MCP tools if available
if self.mcp_integration:
    context = await self.mcp_integration.enhance_agent_with_mcp_tools(
        None, user_id, context
    )
```

3. **In _handle_app_home()**, use the enhanced home view:

```python
# Get user's MCP connections count
mcp_count = 0
if self.mcp_integration:
    try:
        connections = await self.mcp_integration.mcp_commands.connection_manager.get_user_connections(user_id)
        mcp_count = len([c for c in connections if c.status == 'active'])
    except:
        pass

home_view = {
    "type": "home",
    "blocks": get_enhanced_app_home_blocks(mcp_count)
}
```

4. **Add to requirements.txt**:
```
# MCP integration dependencies (if using cloud storage)
google-cloud-secret-manager>=2.16.0  # For GCP
boto3>=1.26.0  # For AWS
```

That's it! The MCP functionality will be seamlessly integrated into your existing bot.
""" 