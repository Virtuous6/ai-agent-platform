"""
MCP Slack Commands

Integrates MCP functionality into the existing Slack bot with new commands:
- /mcp connect [service] - Quick connect with run cards
- /mcp list - Show user's connections  
- /mcp tools [connection] - Browse available tools
- /mcp disconnect [name] - Remove connection
- /mcp test [connection] - Test connection health
"""

import logging
import json
from typing import Dict, Any, Optional, List

from mcp.connection_manager import MCPConnectionManager
from mcp.run_cards.supabase_card import SupabaseRunCard
from mcp.run_cards.github_card import GitHubRunCard
from mcp.run_cards.slack_card import SlackRunCard
from database.supabase_logger import SupabaseLogger

logger = logging.getLogger(__name__)

class MCPSlackCommands:
    """
    Handles MCP-related Slack commands and integrates with existing bot.
    """
    
    def __init__(self, db_logger: SupabaseLogger):
        """Initialize MCP Slack commands."""
        self.db_logger = db_logger
        self.connection_manager = MCPConnectionManager(db_logger)
        
        # Initialize run cards
        self.run_cards = {
            'supabase': SupabaseRunCard(),
            'github': GitHubRunCard(),
            'slack': SlackRunCard()
        }
        
        logger.info("🤖 MCP Slack Commands initialized")
    
    def register_handlers(self, slack_app):
        """Register MCP command handlers with the Slack app."""
        
        @slack_app.command("/mcp")
        async def handle_mcp_command(ack, command, client, say):
            await ack()
            await self._handle_mcp_command(command, client, say)
        
        # Handle button interactions for connection modals
        @slack_app.action("connect_supabase")
        async def handle_connect_supabase(ack, body, client):
            await ack()
            await self._show_supabase_modal(body, client)
        
        @slack_app.action("connect_github")
        async def handle_connect_github(ack, body, client):
            await ack()
            await self._show_github_modal(body, client)
        
        @slack_app.action("connect_slack")
        async def handle_connect_slack(ack, body, client):
            await ack()
            await self._show_slack_modal(body, client)
        
        # Handle modal submissions
        @slack_app.view("mcp_connection_modal")
        async def handle_connection_modal_submission(ack, body, client, view):
            await ack()
            await self._handle_connection_submission(body, client, view)
        
        logger.info("✅ MCP command handlers registered")
    
    async def _handle_mcp_command(self, command: Dict[str, Any], client, say):
        """Handle the main /mcp command with subcommands."""
        try:
            user_id = command.get("user_id")
            channel_id = command.get("channel_id")
            text = command.get("text", "").strip()
            
            # Parse subcommand
            parts = text.split() if text else []
            subcommand = parts[0] if parts else "help"
            
            if subcommand == "connect":
                service = parts[1] if len(parts) > 1 else None
                await self._handle_connect_command(user_id, service, client)
                
            elif subcommand == "list":
                await self._handle_list_command(user_id, say)
                
            elif subcommand == "tools":
                connection_name = parts[1] if len(parts) > 1 else None
                await self._handle_tools_command(user_id, connection_name, say)
                
            elif subcommand == "disconnect":
                connection_name = parts[1] if len(parts) > 1 else None
                await self._handle_disconnect_command(user_id, connection_name, say)
                
            elif subcommand == "test":
                connection_name = parts[1] if len(parts) > 1 else None
                await self._handle_test_command(user_id, connection_name, say)
                
            elif subcommand == "analytics":
                await self._handle_analytics_command(user_id, say)
                
            else:
                await self._show_help(say)
                
        except Exception as e:
            logger.error(f"❌ Error handling MCP command: {str(e)}")
            await say({
                "text": f"❌ Error processing MCP command: {str(e)}",
                "response_type": "ephemeral"
            })
    
    async def _handle_connect_command(self, user_id: str, service: Optional[str], client):
        """Handle /mcp connect [service] command."""
        if service:
            # Direct connection to specific service
            if service.lower() in self.run_cards:
                await self._show_service_modal(user_id, service.lower(), client)
            else:
                await client.chat_postEphemeral(
                    channel=user_id,
                    user=user_id,
                    text=f"❌ Unknown service: {service}. Available services: {', '.join(self.run_cards.keys())}"
                )
        else:
            # Show service selection modal
            await self._show_connection_selection_modal(user_id, client)
    
    async def _handle_list_command(self, user_id: str, say):
        """Handle /mcp list command."""
        try:
            connections = await self.connection_manager.get_user_connections(user_id)
            
            if not connections:
                await say({
                    "text": "📭 You don't have any MCP connections yet. Use `/mcp connect` to get started!",
                    "response_type": "ephemeral"
                })
                return
            
            # Format connections list
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*🔗 Your MCP Connections ({len(connections)})*"
                    }
                },
                {"type": "divider"}
            ]
            
            for conn in connections:
                status_emoji = "✅" if conn.status == "active" else "⚠️"
                tools_count = len(conn.tools_available)
                
                connection_text = (
                    f"{status_emoji} *{conn.display_name or conn.connection_name}*\n"
                    f"Type: {conn.mcp_type} | Tools: {tools_count} | "
                    f"Last used: {conn.last_used.strftime('%Y-%m-%d') if conn.last_used else 'Never'}"
                )
                
                if conn.description:
                    connection_text += f"\n_{conn.description}_"
                
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": connection_text
                    },
                    "accessory": {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🔧 Tools"},
                        "action_id": f"show_tools_{conn.id}"
                    }
                })
            
            await say({
                "blocks": blocks,
                "response_type": "ephemeral"
            })
            
        except Exception as e:
            logger.error(f"❌ Error listing connections: {str(e)}")
            await say({
                "text": f"❌ Error retrieving your connections: {str(e)}",
                "response_type": "ephemeral"
            })
    
    async def _handle_tools_command(self, user_id: str, connection_name: Optional[str], say):
        """Handle /mcp tools [connection] command."""
        try:
            connections = await self.connection_manager.get_user_connections(user_id)
            
            if connection_name:
                # Show tools for specific connection
                connection = next((c for c in connections if c.connection_name == connection_name), None)
                if not connection:
                    await say({
                        "text": f"❌ Connection '{connection_name}' not found.",
                        "response_type": "ephemeral"
                    })
                    return
                
                await self._show_connection_tools(connection, say)
            else:
                # Show all available tools across all connections
                await self._show_all_tools(connections, say)
                
        except Exception as e:
            logger.error(f"❌ Error showing tools: {str(e)}")
            await say({
                "text": f"❌ Error retrieving tools: {str(e)}",
                "response_type": "ephemeral"
            })
    
    async def _handle_disconnect_command(self, user_id: str, connection_name: Optional[str], say):
        """Handle /mcp disconnect [name] command."""
        if not connection_name:
            await say({
                "text": "❌ Please specify a connection name: `/mcp disconnect connection_name`",
                "response_type": "ephemeral"
            })
            return
        
        try:
            connections = await self.connection_manager.get_user_connections(user_id)
            connection = next((c for c in connections if c.connection_name == connection_name), None)
            
            if not connection:
                await say({
                    "text": f"❌ Connection '{connection_name}' not found.",
                    "response_type": "ephemeral"
                })
                return
            
            # Delete the connection
            success = await self.connection_manager.delete_connection(connection.id, user_id)
            
            if success:
                await say({
                    "text": f"✅ Successfully disconnected '{connection_name}' ({connection.mcp_type})",
                    "response_type": "ephemeral"
                })
            else:
                await say({
                    "text": f"❌ Failed to disconnect '{connection_name}'",
                    "response_type": "ephemeral"
                })
                
        except Exception as e:
            logger.error(f"❌ Error disconnecting: {str(e)}")
            await say({
                "text": f"❌ Error disconnecting: {str(e)}",
                "response_type": "ephemeral"
            })
    
    async def _handle_test_command(self, user_id: str, connection_name: Optional[str], say):
        """Handle /mcp test [connection] command."""
        if not connection_name:
            await say({
                "text": "❌ Please specify a connection name: `/mcp test connection_name`",
                "response_type": "ephemeral"
            })
            return
        
        try:
            connections = await self.connection_manager.get_user_connections(user_id)
            connection = next((c for c in connections if c.connection_name == connection_name), None)
            
            if not connection:
                await say({
                    "text": f"❌ Connection '{connection_name}' not found.",
                    "response_type": "ephemeral"
                })
                return
            
            # Test the connection health
            health_result = await self.connection_manager.test_connection_health(connection.id, user_id)
            
            status_emoji = "✅" if health_result.get("status") == "healthy" else "❌"
            
            await say({
                "text": f"{status_emoji} Connection test for '{connection_name}': {health_result.get('status', 'unknown')}",
                "response_type": "ephemeral"
            })
            
        except Exception as e:
            logger.error(f"❌ Error testing connection: {str(e)}")
            await say({
                "text": f"❌ Error testing connection: {str(e)}",
                "response_type": "ephemeral"
            })
    
    async def _handle_analytics_command(self, user_id: str, say):
        """Handle /mcp analytics command."""
        try:
            analytics = await self.connection_manager.get_connection_analytics(user_id)
            
            summary = analytics.get("summary", {})
            
            text = (
                f"*📊 Your MCP Analytics*\n\n"
                f"• **Total Connections**: {summary.get('total_connections', 0)}\n"
                f"• **Total Tool Executions**: {summary.get('total_executions', 0)}\n"
                f"• **Average Success Rate**: {summary.get('avg_success_rate', 0):.1f}%\n"
                f"• **Total Tokens Saved**: {summary.get('total_tokens_saved', 0)}\n"
            )
            
            if analytics.get("by_type"):
                text += "\n*By Connection Type:*\n"
                for type_data in analytics["by_type"]:
                    text += f"• {type_data['mcp_type']}: {type_data['total_executions']} executions\n"
            
            await say({
                "text": text,
                "response_type": "ephemeral"
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting analytics: {str(e)}")
            await say({
                "text": f"❌ Error retrieving analytics: {str(e)}",
                "response_type": "ephemeral"
            })
    
    async def _show_help(self, say):
        """Show MCP command help."""
        help_text = """
*🔌 MCP (Model Context Protocol) Commands*

• `/mcp connect [service]` - Connect to external services
• `/mcp list` - Show your connections
• `/mcp tools [connection]` - Browse available tools
• `/mcp disconnect [name]` - Remove a connection
• `/mcp test [connection]` - Test connection health
• `/mcp analytics` - View usage analytics

*Available Services:*
• `supabase` - Database operations
• `github` - Repository management
• `slack` - Workspace integration

*Examples:*
• `/mcp connect supabase` - Quick Supabase setup
• `/mcp list` - Show all your connections
• `/mcp tools my_database` - Show tools for specific connection
"""
        
        await say({
            "text": help_text,
            "response_type": "ephemeral"
        })
    
    async def _show_connection_selection_modal(self, user_id: str, client):
        """Show modal for selecting connection type."""
        modal = {
            "type": "modal",
            "callback_id": "mcp_connection_selection",
            "title": {"type": "plain_text", "text": "Connect MCP Service"},
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Choose a service to connect:*\n\nSelect from our popular run cards for quick setup:"
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "🗄️ Supabase"},
                            "action_id": "connect_supabase",
                            "style": "primary"
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "🐙 GitHub"},
                            "action_id": "connect_github"
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "💬 Slack"},
                            "action_id": "connect_slack"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "_More connection types coming soon! Each service provides powerful tools for your AI agents._"
                    }
                }
            ]
        }
        
        await client.views_open(
            trigger_id=None,  # This would be provided in real implementation
            view=modal
        )
    
    async def _show_connection_tools(self, connection, say):
        """Show tools available for a specific connection."""
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*🔧 Tools for {connection.display_name or connection.connection_name}*"
                }
            },
            {"type": "divider"}
        ]
        
        if not connection.tools_available:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "❌ No tools available for this connection."
                }
            })
        else:
            for tool in connection.tools_available:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"• **{tool}** - {self._get_tool_description(tool, connection.mcp_type)}"
                    }
                })
        
        await say({
            "blocks": blocks,
            "response_type": "ephemeral"
        })
    
    async def _show_all_tools(self, connections: List, say):
        """Show all available tools across all connections."""
        if not connections:
            await say({
                "text": "📭 No connections found. Use `/mcp connect` to add your first connection!",
                "response_type": "ephemeral"
            })
            return
        
        blocks = [
            {
                "type": "section", 
                "text": {
                    "type": "mrkdwn",
                    "text": "*🔧 All Available Tools*"
                }
            },
            {"type": "divider"}
        ]
        
        for connection in connections:
            if connection.tools_available:
                conn_text = f"*{connection.display_name or connection.connection_name}* ({connection.mcp_type})\n"
                for tool in connection.tools_available[:3]:  # Show first 3 tools
                    conn_text += f"• {tool}\n"
                
                if len(connection.tools_available) > 3:
                    conn_text += f"• ... and {len(connection.tools_available) - 3} more"
                
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": conn_text
                    }
                })
        
        await say({
            "blocks": blocks,
            "response_type": "ephemeral"
        })
    
    def _get_tool_description(self, tool_name: str, mcp_type: str) -> str:
        """Get description for a tool."""
        descriptions = {
            'supabase': {
                'list_tables': 'Get all tables in your database',
                'execute_sql': 'Run custom SQL queries',
                'get_schema': 'Get table schemas and relationships',
                'insert_data': 'Insert new records',
                'update_data': 'Update existing records'
            },
            'github': {
                'search_repos': 'Search repositories',
                'get_issues': 'Get repository issues',
                'create_issue': 'Create new issues',
                'get_pull_requests': 'Get pull requests',
                'get_file_content': 'Read file contents'
            },
            'slack': {
                'send_message': 'Send messages to channels',
                'get_channels': 'List workspace channels',
                'get_users': 'Get workspace members',
                'search_messages': 'Search conversation history'
            }
        }
        
        return descriptions.get(mcp_type, {}).get(tool_name, 'Tool for external service integration')
    
    # Modal handling methods would be implemented here
    async def _show_supabase_modal(self, body, client):
        """Show Supabase connection modal."""
        # Implementation would show modal for Supabase credentials
        pass
    
    async def _show_github_modal(self, body, client):
        """Show GitHub connection modal."""
        # Implementation would show modal for GitHub credentials
        pass
    
    async def _show_slack_modal(self, body, client):
        """Show Slack connection modal."""
        # Implementation would show modal for Slack credentials
        pass
    
    async def _handle_connection_submission(self, body, client, view):
        """Handle connection modal submission."""
        # Implementation would process the submitted credentials
        pass 