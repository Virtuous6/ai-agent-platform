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
        
        self.commands = ["mcp-list", "mcp-connect", "mcp-status"]
        
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
                if len(parts) >= 3 and parts[1] == "custom":
                    # /mcp connect custom [name] [url]
                    connection_name = parts[2] if len(parts) > 2 else None
                    server_url = parts[3] if len(parts) > 3 else None
                    await self._handle_custom_connect_command(user_id, connection_name, server_url, client, say)
                else:
                    service = parts[1] if len(parts) > 1 else None
                    await self._handle_connect_command(user_id, service, client, say)
                
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
    
    async def _handle_connect_command(self, user_id: str, service: Optional[str], client, say):
        """Handle /mcp connect [service] command."""
        if service:
            # Direct connection to specific service
            if service.lower() in self.run_cards:
                await say({
                    "text": f"🔌 **Connect to {service.title()}**\n\nConnection setup for {service} would be initiated here. This feature is under development!",
                    "response_type": "ephemeral"
                })
            else:
                await say({
                    "text": f"❌ Unknown service: {service}. Available services: {', '.join(self.run_cards.keys())}\n\n💡 **Want to connect a custom MCP server?**\nUse: `/mcp connect custom [name] [server_url]`\n\nExample: `/mcp connect custom google-ads https://your-server.com/mcp`",
                    "response_type": "ephemeral"
                })
        else:
            # Show service selection via text instead of modal for compatibility
            available_services = ', '.join(self.run_cards.keys())
            await say({
                "text": f"🔌 **MCP Service Connection**\n\n**Built-in Services:** {available_services}\n\nUse `/mcp connect [service]` to connect to a specific service.\n\n**Custom MCP Server:**\nUse `/mcp connect custom [name] [server_url]` for your own MCP servers.\n\n**Examples:**\n• `/mcp connect supabase` - Built-in service\n• `/mcp connect custom google-ads https://your-server.com/mcp` - Custom server",
                "response_type": "ephemeral"
            })
    
    async def _handle_custom_connect_command(self, user_id: str, connection_name: Optional[str], server_url: Optional[str], client, say):
        """Handle /mcp connect custom [name] [url] command."""
        if not connection_name or not server_url:
            await say({
                "text": "❌ **Missing parameters for custom MCP connection**\n\nUsage: `/mcp connect custom [name] [server_url]`\n\nExample: `/mcp connect custom google-ads https://n8n.soulnav.co/mcp/b3c575cf-5b9b-49d7-81c3-ab4de7dca451/sse`",
                "response_type": "ephemeral"
            })
            return
        
        try:
            # Validate URL format
            if not (server_url.startswith('http://') or server_url.startswith('https://')):
                await say({
                    "text": f"❌ **Invalid URL format**\n\nURL must start with http:// or https://\n\nProvided: `{server_url}`",
                    "response_type": "ephemeral"
                })
                return
            
            # Create custom MCP connection
            success = await self._create_custom_mcp_connection(
                user_id=user_id,
                connection_name=connection_name,
                server_url=server_url
            )
            
            if success:
                await say({
                    "text": f"✅ **Custom MCP connection created successfully!**\n\n🔗 **Name:** {connection_name}\n🌐 **URL:** {server_url}\n📊 **Status:** Active\n\n💡 Use `/mcp test {connection_name}` to verify the connection\n💡 Use `/mcp tools {connection_name}` to see available tools",
                    "response_type": "ephemeral"
                })
            else:
                await say({
                    "text": f"❌ **Failed to create MCP connection**\n\nThere was an error storing the connection. Please try again or check if a connection with name '{connection_name}' already exists.",
                    "response_type": "ephemeral"
                })
                
        except Exception as e:
            logger.error(f"❌ Error creating custom MCP connection: {str(e)}")
            await say({
                "text": f"❌ **Error creating custom MCP connection**\n\nError: {str(e)}\n\nPlease try again or contact support.",
                "response_type": "ephemeral"
            })
    
    async def _create_custom_mcp_connection(self, user_id: str, connection_name: str, server_url: str) -> bool:
        """Create a custom MCP connection in the database."""
        try:
            # Store in Supabase mcp_connections table using the actual schema
            connection_data = {
                "user_id": user_id,
                "service_name": "custom",  # Use service_name field that exists
                "connection_name": connection_name,
                "mcp_server_url": server_url,  # Use mcp_server_url field that exists
                "credentials_encrypted": {},  # Empty for now, can be enhanced later
                "credential_storage_type": "local_env",  # Use allowed value from constraint
                "status": "active",
                "health_status": "unknown",  # Initial health status
                "total_tool_calls": 0,
                "last_used": None
            }
            
            result = self.db_logger.client.table("mcp_connections").insert(connection_data).execute()
            
            if result.data:
                logger.info(f"✅ Created custom MCP connection: {connection_name} for user {user_id}")
                return True
            else:
                logger.error(f"❌ No data returned when creating MCP connection")
                return False
                
        except Exception as e:
            logger.error(f"❌ Database error creating MCP connection: {str(e)}")
            return False
    
    async def _handle_list_command(self, user_id: str, say):
        """Handle /mcp list command."""
        try:
            # Query the database directly for user connections using actual schema
            result = self.db_logger.client.table("mcp_connections").select(
                "id, connection_name, service_name, mcp_server_url, status, health_status, total_tool_calls, last_used, created_at"
            ).eq("user_id", user_id).execute()
            
            connections = result.data if result.data else []
            
            if not connections:
                await say({
                    "text": "📭 You don't have any MCP connections yet. Use `/mcp connect custom [name] [url]` to get started!",
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
                status_emoji = "✅" if conn.get("status") == "active" else "⚠️"
                health_status = conn.get("health_status", "unknown")
                if health_status == "healthy":
                    health_emoji = "🟢"
                elif health_status == "unhealthy":
                    health_emoji = "🔴"
                elif health_status == "timeout":
                    health_emoji = "⏰"
                else:  # unknown
                    health_emoji = "🟡"
                
                connection_text = (
                    f"{status_emoji} *{conn.get('connection_name', 'Unknown')}*\n"
                    f"Service: {conn.get('service_name', 'custom')} | "
                    f"Health: {health_emoji} {conn.get('health_status', 'unknown')} | "
                    f"Tool calls: {conn.get('total_tool_calls', 0)}\n"
                    f"URL: `{conn.get('mcp_server_url', 'Not set')}`\n"
                    f"Last used: {conn.get('last_used', 'Never')[:10] if conn.get('last_used') else 'Never'}"
                )
                
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": connection_text
                    },
                    "accessory": {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🔧 Test"},
                        "action_id": f"test_connection_{conn.get('id')}"
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
            # Find the connection in the database
            result = self.db_logger.client.table("mcp_connections").select(
                "id, connection_name, service_name, mcp_server_url"
            ).eq("user_id", user_id).eq("connection_name", connection_name).execute()
            
            if not result.data:
                await say({
                    "text": f"❌ Connection '{connection_name}' not found.",
                    "response_type": "ephemeral"
                })
                return
            
            connection = result.data[0]
            
            # Delete the connection
            delete_result = self.db_logger.client.table("mcp_connections").delete().eq(
                "id", connection["id"]
            ).eq("user_id", user_id).execute()
            
            if delete_result.data:
                await say({
                    "text": f"✅ Successfully disconnected '{connection_name}' ({connection.get('service_name', 'custom')})\n🗑️ Server URL: `{connection.get('mcp_server_url', 'Unknown')}`",
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
            # Find the connection in the database
            result = self.db_logger.client.table("mcp_connections").select(
                "id, connection_name, service_name, mcp_server_url, status, health_status"
            ).eq("user_id", user_id).eq("connection_name", connection_name).execute()
            
            if not result.data:
                await say({
                    "text": f"❌ Connection '{connection_name}' not found.",
                    "response_type": "ephemeral"
                })
                return
            
            connection = result.data[0]
            server_url = connection.get("mcp_server_url")
            
            # Simple connection test - try to reach the URL
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(server_url)
                    is_reachable = response.status_code < 500
                    test_status = "healthy" if is_reachable else "unhealthy"
                    
                # Update health status in database
                self.db_logger.client.table("mcp_connections").update({
                    "health_status": test_status,
                    "last_health_check": "now()"
                }).eq("id", connection["id"]).execute()
                
                status_emoji = "✅" if test_status == "healthy" else "❌"
                
                await say({
                    "text": f"{status_emoji} **Connection test for '{connection_name}'**\n\n🌐 **URL:** `{server_url}`\n📊 **Status:** {test_status}\n🔍 **Response:** {response.status_code if 'response' in locals() else 'No response'}\n\n💡 Connection {'is working!' if test_status == 'healthy' else 'has issues - check the server URL'}",
                    "response_type": "ephemeral"
                })
                
            except Exception as test_error:
                # Update health status to unhealthy
                self.db_logger.client.table("mcp_connections").update({
                    "health_status": "unhealthy",
                    "last_health_check": "now()"
                }).eq("id", connection["id"]).execute()
                
                await say({
                    "text": f"❌ **Connection test failed for '{connection_name}'**\n\n🌐 **URL:** `{server_url}`\n📊 **Status:** unhealthy\n🔍 **Error:** {str(test_error)[:200]}...\n\n💡 Check if the server is running and accessible",
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
            # Get analytics directly from the database using actual schema
            result = self.db_logger.client.table("mcp_connections").select(
                "id, connection_name, service_name, total_tool_calls, status, health_status, created_at, last_used"
            ).eq("user_id", user_id).execute()
            
            connections = result.data if result.data else []
            
            if not connections:
                await say({
                    "text": "📊 **MCP Analytics**\n\n📭 No connections found. Use `/mcp connect custom [name] [url]` to create your first connection!",
                    "response_type": "ephemeral"
                })
                return
            
            # Calculate analytics
            total_connections = len(connections)
            total_tool_calls = sum(conn.get('total_tool_calls', 0) for conn in connections)
            active_connections = len([c for c in connections if c.get('status') == 'active'])
            healthy_connections = len([c for c in connections if c.get('health_status') == 'healthy'])
            unhealthy_connections = len([c for c in connections if c.get('health_status') == 'unhealthy'])
            unknown_connections = len([c for c in connections if c.get('health_status') == 'unknown'])
            
            # Group by service type
            by_service = {}
            for conn in connections:
                service = conn.get('service_name', 'unknown')
                if service not in by_service:
                    by_service[service] = {'count': 0, 'tool_calls': 0}
                by_service[service]['count'] += 1
                by_service[service]['tool_calls'] += conn.get('total_tool_calls', 0)
            
            text = (
                f"*📊 Your MCP Analytics*\n\n"
                f"• **Total Connections**: {total_connections}\n"
                f"• **Active Connections**: {active_connections}\n"
                f"• **Health Status**: 🟢 {healthy_connections} healthy, 🔴 {unhealthy_connections} unhealthy, 🟡 {unknown_connections} unknown\n"
                f"• **Total Tool Calls**: {total_tool_calls:,}\n"
                f"• **Success Rate**: {(healthy_connections/max(total_connections,1)*100):.1f}%\n"
            )
            
            if by_service:
                text += "\n*📈 By Service Type:*\n"
                for service, data in by_service.items():
                    text += f"• **{service}**: {data['count']} connections, {data['tool_calls']} calls\n"
            
            # Recent activity
            recent_connections = [c for c in connections if c.get('last_used')]
            if recent_connections:
                recent_connections.sort(key=lambda x: x.get('last_used', ''), reverse=True)
                text += f"\n*🕐 Most Recent Activity:*\n"
                for conn in recent_connections[:3]:
                    last_used = conn.get('last_used', '')[:10] if conn.get('last_used') else 'Never'
                    text += f"• **{conn.get('connection_name')}**: {last_used}\n"
            
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

**Built-in Services:**
• `/mcp connect [service]` - Connect to built-in services
• `/mcp list` - Show your connections
• `/mcp tools [connection]` - Browse available tools
• `/mcp disconnect [name]` - Remove a connection
• `/mcp test [connection]` - Test connection health
• `/mcp analytics` - View usage analytics

**Custom MCP Servers:**
• `/mcp connect custom [name] [url]` - Connect to your own MCP server

*Available Built-in Services:*
• `supabase` - Database operations
• `github` - Repository management
• `slack` - Workspace integration

*Examples:*
• `/mcp connect supabase` - Quick Supabase setup
• `/mcp connect custom google-ads https://your-server.com/mcp` - Custom server
• `/mcp list` - Show all your connections
• `/mcp tools google-ads` - Show tools for specific connection
• `/mcp test google-ads` - Test custom connection
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

    async def handle_command(self, command: str, parameters: dict):
        """Handle MCP-related Slack commands."""
        return {"response": f"Executed {command} with parameters {parameters}"} 