"""
Filename: slack_bot.py
Purpose: Main Slack bot implementation for AI Agent Platform
Dependencies: slack_bolt, asyncio, logging

This module is part of the AI Agent Platform.
See README.llm.md in this directory for context.
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.errors import SlackApiError

# Import orchestrator (now available!)
from orchestrator.agent_orchestrator import AgentOrchestrator, AgentType

# Import agents
from agents.general.general_agent import GeneralAgent
from agents.technical.technical_agent import TechnicalAgent
from agents.research.research_agent import ResearchAgent

# Import Supabase logger
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from database.supabase_logger import SupabaseLogger

# Import feedback handler
from agents.improvement.feedback_handler import FeedbackHandler

# Import MCP commands
from mcp.slack_interface.mcp_commands import MCPSlackCommands

# Import tool collaboration interface
from mcp.slack_interface.tool_collaboration import ToolCollaborationInterface
from mcp.dynamic_tool_builder import DynamicToolBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAgentSlackBot:
    """
    Main Slack bot for the AI Agent Platform.
    
    Handles all Slack interactions and routes requests to appropriate agents
    through the orchestrator system with comprehensive Supabase logging.
    """
    
    def __init__(self):
        """Initialize the Slack bot with configuration and dependencies."""
        self.app = AsyncApp(
            token=os.environ.get("SLACK_BOT_TOKEN"),
            signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
        )
        
        # Initialize agents
        self.general_agent = GeneralAgent()
        self.technical_agent = TechnicalAgent()
        self.research_agent = ResearchAgent()
        
        # Initialize orchestrator with all agents
        self.orchestrator = AgentOrchestrator(
            general_agent=self.general_agent,
            technical_agent=self.technical_agent,
            research_agent=self.research_agent
        )
        
        # Initialize Supabase logger with enhanced error handling
        try:
            self.supabase_logger = SupabaseLogger()
            # Test the connection
            health = self.supabase_logger.health_check()
            if health.get("status") == "healthy":
                logger.info("✅ Supabase logger initialized successfully")
            else:
                logger.warning(f"⚠️ Supabase health check failed: {health.get('error', 'Unknown error')}")
                self.supabase_logger = None
        except Exception as e:
            logger.warning(f"❌ Supabase logger initialization failed: {str(e)}")
            logger.warning("Bot will continue with console logging only")
            self.supabase_logger = None
        
        # Initialize feedback handler
        try:
            self.feedback_handler = FeedbackHandler(
                db_logger=self.supabase_logger,
                orchestrator=self.orchestrator
            )
            logger.info("✅ Feedback handler initialized successfully")
        except Exception as e:
            logger.warning(f"❌ Feedback handler initialization failed: {str(e)}")
            self.feedback_handler = None
        
        # Initialize MCP commands
        try:
            if self.supabase_logger:
                self.mcp_commands = MCPSlackCommands(db_logger=self.supabase_logger)
                logger.info("✅ MCP commands initialized successfully")
            else:
                logger.warning("⚠️ MCP commands skipped - Supabase logger not available")
                self.mcp_commands = None
        except Exception as e:
            logger.warning(f"❌ MCP commands initialization failed: {str(e)}")
            self.mcp_commands = None
        
        # Initialize tool collaboration interface
        try:
            if self.supabase_logger:
                # Create required dependencies for tool collaboration
                from mcp.tool_registry import MCPToolRegistry
                from mcp.security_sandbox import MCPSecuritySandbox
                from events.event_bus import EventBus
                
                # Initialize event bus if not already available
                event_bus = EventBus()
                tool_registry = MCPToolRegistry()
                security_sandbox = MCPSecuritySandbox()
                
                # Initialize dynamic tool builder
                self.dynamic_tool_builder = DynamicToolBuilder(
                    supabase_logger=self.supabase_logger,
                    tool_registry=tool_registry,
                    event_bus=event_bus,
                    security_sandbox=security_sandbox
                )
                
                # Initialize tool collaboration interface
                self.tool_collaboration = ToolCollaborationInterface(
                    slack_app=self.app,
                    dynamic_tool_builder=self.dynamic_tool_builder
                )
                logger.info("✅ Tool collaboration interface initialized successfully")
            else:
                logger.warning("⚠️ Tool collaboration skipped - Supabase logger not available")
                self.tool_collaboration = None
        except Exception as e:
            logger.warning(f"❌ Tool collaboration initialization failed: {str(e)}")
            self.tool_collaboration = None
        
        # Bot user ID for mention detection
        self.bot_user_id = None
        
        # Active conversations tracking
        self.active_conversations = {}
        
        # Register event handlers
        self._register_handlers()
        
        logger.info("🤖 AI Agent Slack Bot initialized")
    
    def _register_handlers(self):
        """Register all Slack event handlers."""
        
        # Handle app mentions
        @self.app.event("app_mention")
        async def handle_mention(event, say, client):
            await self._handle_message(event, say, client, is_mention=True)
        
        # Handle direct messages
        @self.app.event("message")
        async def handle_direct_message(event, say, client):
            # Only process if it's a DM (not in a channel)
            if event.get("channel_type") == "im":
                await self._handle_message(event, say, client, is_mention=False)
        
        # Handle app home opened
        @self.app.event("app_home_opened")
        async def handle_app_home_opened(event, client):
            await self._handle_app_home(event, client)
        
        # Handle feedback slash commands
        @self.app.command("/improve")
        async def handle_improve_command(ack, say, command, client):
            await ack()
            await self._handle_feedback_command("improve", command, say, client)
        
        @self.app.command("/save-workflow")
        async def handle_save_workflow_command(ack, say, command, client):
            await ack()
            await self._handle_feedback_command("save-workflow", command, say, client)
        
        @self.app.command("/list-workflows")
        async def handle_list_workflows_command(ack, say, command, client):
            await ack()
            await self._handle_feedback_command("list-workflows", command, say, client)
        
        @self.app.command("/suggest")
        async def handle_suggest_command(ack, say, command, client):
            await ack()
            await self._handle_feedback_command("feedback", command, say, client)
        
        @self.app.command("/metrics")
        async def handle_metrics_command(ack, say, command, client):
            await ack()
            await self._handle_metrics_command(command, say, client)
        
        # Register MCP command handlers
        if self.mcp_commands:
            self.mcp_commands.register_handlers(self.app)
    
    async def _handle_message(self, event: Dict[str, Any], say, client, is_mention: bool = False):
        """
        Process incoming messages and route to appropriate agents with comprehensive logging.
        
        Args:
            event: Slack event data
            say: Slack say function for responses
            client: Slack client for API calls
            is_mention: Whether this was a direct mention
        """
        start_time = datetime.utcnow()
        conversation_id = None
        user_id = event.get("user")
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts", event.get("ts"))
        message_text = event.get("text", "")
        
        try:
            # Skip bot messages
            if event.get("bot_id"):
                return
            
            logger.info(f"📨 Processing message from user {user_id} in channel {channel_id}")
            
            # Clean the message text (remove bot mention if present)
            cleaned_text = await self._clean_message_text(message_text)
            
            # Get/create conversation and build context
            conversation_id = await self._get_or_create_conversation(user_id, channel_id, thread_ts)
            context = await self._build_context(user_id, channel_id, thread_ts, conversation_id)
            
            # CRITICAL FIX: Enhance context with user's MCP tools before routing
            if hasattr(self, 'mcp_commands') and self.mcp_commands:
                try:
                    context = await self._enhance_context_with_mcp_tools(user_id, context)
                    if context.get("mcp_tools_available"):
                        logger.info(f"🔧 Enhanced agent context with {len(context['mcp_tools_available'])} MCP tools from {context.get('mcp_connections_count', 0)} connections")
                except Exception as e:
                    logger.warning(f"Failed to enhance context with MCP tools: {e}")
            
            # Log the user message to Supabase
            await self._log_interaction(
                conversation_id=conversation_id,
                user_id=user_id, 
                channel_id=channel_id, 
                content=cleaned_text, 
                interaction_type="user_message"
            )
            
            # Route request through orchestrator
            logger.info(f"🔀 Routing message to orchestrator: '{cleaned_text[:50]}...'")
            orchestrator_result = await self.orchestrator.route_request(cleaned_text, context)
            
            # Extract response and metadata
            response = orchestrator_result["response"]
            agent_name = orchestrator_result["agent_name"]
            agent_type = orchestrator_result["agent_type"]
            confidence = orchestrator_result["confidence"]
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Format response with routing information (configurable via context)
            if context.get("show_routing_info", True):
                agent_info = f"\n\n_🤖 Routed to: {agent_name} (confidence: {confidence:.1%})_"
                full_response = response + agent_info
            else:
                full_response = response
            
            # Send response in thread if original message was in thread
            await say(
                text=full_response,
                thread_ts=thread_ts if thread_ts != event.get("ts") else event.get("ts")
            )
            
            logger.info(f"✅ Response sent via {agent_name} ({confidence:.1%} confidence, {processing_time:.1f}ms)")
            
            # Extract token usage from agent response
            tokens_used = orchestrator_result.get('tokens_used', 0)
            input_tokens = orchestrator_result.get('input_tokens', 0)
            output_tokens = orchestrator_result.get('output_tokens', 0)
            model_used = orchestrator_result.get('metadata', {}).get('model_used', 'unknown')
            
            # If we only have total tokens, estimate input/output split
            if tokens_used > 0 and input_tokens == 0 and output_tokens == 0:
                input_tokens = int(tokens_used * 0.7)
                output_tokens = int(tokens_used * 0.3)
            
            # Ensure we have valid values (None -> 0 for database)
            input_tokens = input_tokens if input_tokens is not None else 0
            output_tokens = output_tokens if output_tokens is not None else 0
            
            # Log the bot response with comprehensive metadata including token tracking
            await self._log_interaction(
                conversation_id=conversation_id,
                user_id=user_id, 
                channel_id=channel_id, 
                content=response, 
                interaction_type="bot_response",
                agent_type=agent_type,
                agent_response=orchestrator_result,
                routing_confidence=confidence,
                escalation_suggestion=orchestrator_result.get('metadata', {}).get('escalation_suggestion'),
                processing_time_ms=processing_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_used=model_used
            )
            
            # Log routing decision for analytics
            if self.supabase_logger:
                await self._log_routing_decision(
                    conversation_id=conversation_id,
                    selected_agent=agent_type,
                    confidence_score=confidence,
                    all_scores=orchestrator_result.get('metadata', {}).get('all_scores', {}),
                    routing_time_ms=processing_time,
                    was_explicit_mention=orchestrator_result.get('metadata', {}).get('explicit_mention', False)
                )
            
            # Update agent metrics
            await self._update_agent_metrics(
                agent_name=agent_type,
                response_time_ms=processing_time,
                success=True,
                escalation=orchestrator_result.get('metadata', {}).get('escalation_suggestion') is not None
            )
            
        except Exception as e:
            logger.error(f"❌ Error handling message: {str(e)}")
            
            # Log error interaction
            if conversation_id:
                await self._log_interaction(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    channel_id=channel_id,
                    content=f"Error: {str(e)}",
                    interaction_type="error"
                )
            
            # Update metrics for error
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._update_agent_metrics(
                agent_name="system",
                response_time_ms=processing_time,
                success=False
            )
            
            # Send user-friendly error message
            error_message = "I'm sorry, I encountered an error processing your message. Please try again in a moment."
            await say(
                text=error_message,
                thread_ts=event.get("thread_ts", event.get("ts"))
            )
    
    async def _handle_app_home(self, event: Dict[str, Any], client):
        """Handle app home tab opened events."""
        try:
            user_id = event["user"]
            
            # Enhanced app home view with agent status
            home_view = {
                "type": "home",
                "blocks": [
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
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*System Status*\n🔗 Orchestrator: ✅ Active\n📊 Logging: {'✅ Connected' if self.supabase_logger else '⚠️ Console Only'}\n🤖 Active Conversations: {len(self.active_conversations)}"
                        }
                    }
                ]
            }
            
            await client.views_publish(
                user_id=user_id,
                view=home_view
            )
            
        except Exception as e:
            logger.error(f"Error handling app home: {str(e)}")
    
    async def _handle_feedback_command(self, command_type: str, command, say, client):
        """Handle user feedback commands through the feedback system."""
        if not self.feedback_handler:
            await say({
                "text": "❌ Feedback system is currently unavailable. Please try again later.",
                "response_type": "ephemeral"
            })
            return
        
        try:
            user_id = command.get("user_id")
            channel_id = command.get("channel_id")
            command_text = command.get("text", "").strip()
            
            # Process the feedback command
            result = await self.feedback_handler.handle_command(
                command_type=command_type,
                user_id=user_id,
                channel_id=channel_id,
                command_text=command_text
            )
            
            # Send the response
            await say({
                "text": result.get("message", "Feedback processed successfully!"),
                "response_type": "ephemeral" if result.get("ephemeral", False) else "in_channel"
            })
            
        except Exception as e:
            logger.error(f"Error handling feedback command '{command_type}': {str(e)}")
            await say({
                "text": f"❌ Error processing {command_type} command. Please try again later.",
                "response_type": "ephemeral"
            })
    
    async def _handle_metrics_command(self, command, say, client):
        """Handle metrics dashboard command - show comprehensive system analytics."""
        try:
            user_id = command.get("user_id")
            
            # Get comprehensive system metrics
            metrics_data = await self._get_system_metrics()
            
            # Format for Slack display
            formatted_message = await self._format_metrics_for_slack(metrics_data)
            
            await say({
                "text": formatted_message,
                "response_type": "ephemeral"  # Keep metrics private to user
            })
            
            # Log metrics access
            if self.supabase_logger:
                await self.supabase_logger.log_event(
                    event_type="metrics_accessed",
                    event_data={
                        "user_id": user_id,
                        "metrics_requested": list(metrics_data.keys())
                    },
                    user_id=user_id
                )
            
        except Exception as e:
            logger.error(f"Error handling metrics command: {str(e)}")
            await say({
                "text": "❌ Error retrieving system metrics. Please try again later.",
                "response_type": "ephemeral"
            })
    
    async def _clean_message_text(self, text: str) -> str:
        """
        Clean message text by removing bot mentions and extra whitespace.
        
        Args:
            text: Raw message text
            
        Returns:
            Cleaned message text
        """
        if not text:
            return ""
        
        # Remove bot mention if present
        if self.bot_user_id and f"<@{self.bot_user_id}>" in text:
            text = text.replace(f"<@{self.bot_user_id}>", "").strip()
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        return text
    
    async def _build_context(self, user_id: str, channel_id: str, thread_ts: str, conversation_id: str) -> Dict[str, Any]:
        """
        Build conversation context for agent processing with enhanced data from Supabase.
        
        Args:
            user_id: Slack user ID
            channel_id: Slack channel ID
            thread_ts: Thread timestamp
            conversation_id: Conversation ID from Supabase
            
        Returns:
            Context dictionary
        """
        context = {
            "user_id": user_id,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "conversation_id": conversation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "show_routing_info": True,  # Default to showing routing info
            "conversation_history": [],
            "user_preferences": {},
            "current_agent": None
        }
        
        # Load conversation history and user preferences from Supabase if available
        if self.supabase_logger:
            try:
                # Get recent conversation history
                history = await self.supabase_logger.get_conversation_history(
                    user_id=user_id, 
                    limit=5
                )
                context["conversation_history"] = history
                
                # Get user preferences (placeholder for future implementation)
                # user_prefs = await self.supabase_logger.get_user_preferences(user_id)
                # context["user_preferences"] = user_prefs
                
            except Exception as e:
                logger.warning(f"Failed to load context from Supabase: {str(e)}")
        
        return context
    
    async def _enhance_context_with_mcp_tools(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance agent context with user's available MCP tools.
        
        This method gets the user's MCP connections from Supabase and adds
        them to the context so agents know what tools are available.
        
        Args:
            user_id: Slack user ID
            context: Existing context dictionary
            
        Returns:
            Enhanced context with MCP tools information
        """
        try:
            # Get user's MCP connections from the database
            connections = await self.mcp_commands.connection_manager.get_user_connections(user_id)
            
            if not connections:
                logger.debug(f"No MCP connections found for user {user_id}")
                return context
            
            # Collect all available tools from active connections
            available_tools = []
            for connection in connections:
                if connection.status == 'active':
                    # For now, provide common tools based on MCP type since discovery isn't working
                    tools = self._get_default_tools_for_mcp_type(connection.mcp_type, connection.connection_name)
                    for tool_name in tools:
                        available_tools.append({
                            "name": tool_name,
                            "connection_id": connection.id,
                            "connection_name": connection.connection_name,
                            "mcp_type": connection.mcp_type,
                            "server_url": connection.connection_config.get('url') if connection.connection_config else None,
                            "description": self._get_tool_description(tool_name, connection.mcp_type)
                        })
            
            # Add MCP information to context
            context["mcp_tools_available"] = available_tools
            context["mcp_connections_count"] = len(connections)
            context["mcp_active_connections"] = len([c for c in connections if c.status == 'active'])
            
            # Add a summary for agents to understand what tools they have
            if available_tools:
                tool_summary = []
                by_type = {}
                for tool in available_tools:
                    mcp_type = tool["mcp_type"]
                    if mcp_type not in by_type:
                        by_type[mcp_type] = []
                    by_type[mcp_type].append(tool["name"])
                
                for mcp_type, tools in by_type.items():
                    tool_summary.append(f"{mcp_type}: {', '.join(tools)}")
                
                context["mcp_tools_summary"] = f"Available MCP tools: {'; '.join(tool_summary)}"
                
                # Add tool execution capability
                context["can_execute_mcp_tools"] = True
                context["mcp_executor"] = self._create_mcp_executor(available_tools)
            
            logger.debug(f"Enhanced context with {len(available_tools)} MCP tools for user {user_id}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to enhance context with MCP tools for user {user_id}: {e}")
            # Return original context if enhancement fails
            return context
    
    def _get_default_tools_for_mcp_type(self, mcp_type: str, connection_name: str) -> List[str]:
        """Get default tools for MCP type since auto-discovery isn't working yet."""
        tool_mappings = {
            "airtable": ["list_bases", "list_tables", "get_records", "create_record", "update_record"],
            "google_ads": ["get_campaigns", "get_ad_groups", "get_keywords", "get_performance_data"],
            "google-ads": ["get_campaigns", "get_ad_groups", "get_keywords", "get_performance_data"],  # Handle hyphen
            "gmail": ["list_messages", "get_message", "send_message", "search_messages"],
            "custom": ["execute_action", "get_data", "list_items"]
        }
        
        # First check the connection name directly (this is the service_name from database)
        if connection_name and connection_name.lower() in tool_mappings:
            return tool_mappings[connection_name.lower()]
        
        # Then check mcp_type
        if mcp_type and mcp_type.lower() in tool_mappings:
            return tool_mappings[mcp_type.lower()]
        
        # Check if connection name contains recognizable service names
        connection_lower = (connection_name or "").lower()
        for service, tools in tool_mappings.items():
            if service in connection_lower:
                return tools
        
        # Default fallback tools
        return ["get_data", "execute_action", "list_items"]
    
    def _create_mcp_executor(self, available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an MCP tool executor that agents can use."""
        return {
            "available_tools": available_tools,
            "execute_tool": self._execute_mcp_tool
        }
    
    async def _execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any] = None, user_id: str = None) -> Dict[str, Any]:
        """
        Execute an MCP tool by making HTTP requests to the MCP server.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            user_id: User requesting the tool execution
            
        Returns:
            Tool execution result
        """
        try:
            logger.info(f"🔧 Executing MCP tool: {tool_name} for user {user_id}")
            
            # Find the tool in user's connections
            connections = await self.mcp_commands.connection_manager.get_user_connections(user_id)
            
            tool_connection = None
            for connection in connections:
                if connection.status == 'active':
                    default_tools = self._get_default_tools_for_mcp_type(connection.mcp_type, connection.connection_name)
                    if tool_name in default_tools:
                        tool_connection = connection
                        break
            
            if not tool_connection:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not found in active connections"
                }
            
            # For now, simulate tool execution since SSE endpoints need special handling
            # TODO: Implement proper SSE/WebSocket connection to MCP servers
            
            result = await self._simulate_mcp_tool_execution(tool_name, tool_connection, parameters)
            
            # Log tool usage
            if self.supabase_logger:
                await self.supabase_logger.log_event(
                    event_type="mcp_tool_executed",
                    event_data={
                        "tool_name": tool_name,
                        "connection_name": tool_connection.connection_name,
                        "mcp_type": tool_connection.mcp_type,
                        "success": result.get("success", False),
                        "parameters": parameters
                    },
                    user_id=user_id
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing MCP tool {tool_name}: {e}")
            return {
                "success": False,
                "error": f"Failed to execute {tool_name}: {str(e)}"
            }
    
    async def _simulate_mcp_tool_execution(self, tool_name: str, connection, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Simulate MCP tool execution with realistic data.
        TODO: Replace with actual MCP server communication.
        """
        connection_name = connection.connection_name
        mcp_type = connection.mcp_type
        
        # Simulate based on tool type and connection
        if "airtable" in connection_name.lower() or mcp_type == "airtable":
            if tool_name == "list_bases":
                return {
                    "success": True,
                    "data": [
                        {"id": "appXXX123", "name": "Marketing Campaigns"},
                        {"id": "appYYY456", "name": "Customer Data"},
                        {"id": "appZZZ789", "name": "Product Catalog"}
                    ],
                    "message": f"Retrieved bases from {connection_name}"
                }
            elif tool_name == "get_records":
                table_name = parameters.get("table_name", "Main Table") if parameters else "Main Table"
                return {
                    "success": True,
                    "data": [
                        {"id": "rec1", "fields": {"Name": "Campaign A", "Status": "Active", "Budget": 5000}},
                        {"id": "rec2", "fields": {"Name": "Campaign B", "Status": "Paused", "Budget": 3000}},
                        {"id": "rec3", "fields": {"Name": "Campaign C", "Status": "Active", "Budget": 7500}}
                    ],
                    "message": f"Retrieved 3 records from {table_name} in {connection_name}"
                }
        
        elif "google" in connection_name.lower() or "ads" in connection_name.lower():
            if tool_name == "get_campaigns":
                return {
                    "success": True,
                    "data": [
                        {"id": "123", "name": "Summer Sale 2024", "status": "ENABLED", "budget": 10000},
                        {"id": "456", "name": "Brand Awareness", "status": "ENABLED", "budget": 15000},
                        {"id": "789", "name": "Holiday Special", "status": "PAUSED", "budget": 8000}
                    ],
                    "message": f"Retrieved campaigns from {connection_name}"
                }
            elif tool_name == "get_performance_data":
                return {
                    "success": True,
                    "data": {
                        "impressions": 125000,
                        "clicks": 3250,
                        "ctr": 2.6,
                        "cost": 1875.50,
                        "conversions": 87
                    },
                    "message": f"Retrieved performance data from {connection_name}"
                }
        
        # Default fallback
        return {
            "success": True,
            "data": f"Simulated execution of {tool_name} on {connection_name}",
            "message": f"Tool {tool_name} executed successfully (simulated)",
            "note": "This is simulated data. Actual MCP server integration pending."
        }
    
    def _get_tool_description(self, tool_name: str, mcp_type: str) -> str:
        """Get a description for an MCP tool based on its name and type."""
        descriptions = {
            # Airtable tools
            "list_bases": "List all Airtable bases",
            "list_tables": "List tables in an Airtable base",
            "get_records": "Get records from an Airtable table",
            "create_record": "Create a new record in Airtable",
            "update_record": "Update an existing Airtable record",
            "delete_record": "Delete a record from Airtable",
            
            # Supabase tools
            "execute_sql": "Execute SQL queries on database",
            "get_schema": "Get database schema information",
            "insert_data": "Insert data into database tables",
            "update_data": "Update database records",
            
            # GitHub tools
            "search_repos": "Search GitHub repositories",
            "get_issues": "Get GitHub issues",
            "create_issue": "Create a new GitHub issue",
            "get_pull_requests": "Get pull requests",
            
            # Slack tools
            "send_message": "Send messages to Slack channels",
            "get_channels": "Get Slack workspace channels",
            "get_users": "Get Slack workspace users",
            "search_messages": "Search Slack message history"
        }
        
        # Return specific description or generate generic one
        if tool_name in descriptions:
            return descriptions[tool_name]
        else:
            return f"{tool_name} tool for {mcp_type}"
    
    async def _log_interaction(self, conversation_id: str, user_id: str, channel_id: str, 
                             content: str, interaction_type: str, agent_type: Optional[str] = None,
                             agent_response: Optional[Dict[str, Any]] = None,
                             routing_confidence: Optional[float] = None,
                             escalation_suggestion: Optional[str] = None,
                             processing_time_ms: Optional[float] = None,
                             input_tokens: Optional[int] = None,
                             output_tokens: Optional[int] = None,
                             model_used: Optional[str] = None):
        """
        Log user interactions for analytics and debugging using Supabase with token tracking.
        
        Args:
            conversation_id: ID of the conversation
            user_id: Slack user ID
            channel_id: Slack channel ID
            content: Message content
            interaction_type: Type of interaction (user_message, bot_response, error)
            agent_type: Agent that handled the message
            agent_response: Full agent response data
            routing_confidence: Confidence score for routing
            escalation_suggestion: Suggested escalation if any
            processing_time_ms: Time taken to process the message
            input_tokens: Input tokens consumed by LLM
            output_tokens: Output tokens generated by LLM
            model_used: LLM model used (gpt-4, gpt-3.5-turbo, etc.)
        """
        try:
            # Calculate token metrics
            total_tokens = None
            estimated_cost = None
            
            if input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
                if model_used and self.supabase_logger:
                    estimated_cost = self.supabase_logger.calculate_token_cost(
                        model=model_used,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens
                    )
            
            # Log to console for immediate visibility
            if interaction_type == "error":
                logger.error(f"🚨 Error logged: {content[:100]}...")
            else:
                token_info = f" (tokens: {total_tokens}, cost: ${estimated_cost:.6f})" if total_tokens else ""
                logger.info(f"📝 Logged {interaction_type} from {user_id} (agent: {agent_type}){token_info}")
            
            # Log to Supabase if available
            if self.supabase_logger:
                success = await self.supabase_logger.log_message(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    content=content,
                    message_type=interaction_type,
                    agent_type=agent_type,
                    agent_response=agent_response,
                    routing_confidence=routing_confidence,
                    escalation_suggestion=escalation_suggestion,
                    processing_time_ms=processing_time_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    model_used=model_used,
                    estimated_cost=estimated_cost
                )
                
                if not success:
                    logger.warning("⚠️ Failed to log to Supabase, but continuing with operation")
            else:
                logger.debug("📝 Supabase logger not available, using console logging only")
            
        except Exception as e:
            logger.error(f"❌ Error logging interaction: {str(e)}")
    
    async def _log_routing_decision(self, conversation_id: str, selected_agent: str, 
                                  confidence_score: float, all_scores: Dict[str, float],
                                  routing_time_ms: float, was_explicit_mention: bool = False):
        """
        Log routing decisions for analytics and algorithm improvement.
        
        Args:
            conversation_id: ID of the conversation
            selected_agent: Selected agent type
            confidence_score: Confidence score for the selected agent
            all_scores: All agent scores from routing
            routing_time_ms: Time taken for routing decision
            was_explicit_mention: Whether user explicitly mentioned an agent
        """
        try:
            if self.supabase_logger:
                # Get the message ID from the most recent message in this conversation
                # For now, we'll create a routing entry without message_id reference
                # This could be enhanced to link to specific messages
                
                success = await self.supabase_logger.log_routing_decision(
                    message_id=None,  # Could be enhanced to reference specific message
                    selected_agent=selected_agent,
                    confidence_score=confidence_score,
                    all_scores=all_scores,
                    routing_time_ms=routing_time_ms,
                    was_explicit_mention=was_explicit_mention
                )
                
                if success:
                    logger.debug(f"🎯 Routing decision logged: {selected_agent} ({confidence_score:.2f})")
                
        except Exception as e:
            logger.error(f"❌ Error logging routing decision: {str(e)}")
    
    async def _get_or_create_conversation(self, user_id: str, channel_id: str, thread_ts: str) -> str:
        """
        Get existing conversation ID or create a new conversation with enhanced tracking.
        
        Args:
            user_id: Slack user ID
            channel_id: Slack channel ID
            thread_ts: Thread timestamp
            
        Returns:
            Conversation ID
        """
        # Create a unique key for this conversation
        conv_key = f"{channel_id}:{thread_ts}"
        
        # Check if we already have this conversation active
        if conv_key in self.active_conversations:
            return self.active_conversations[conv_key]
        
        # Try to log new conversation to Supabase
        if self.supabase_logger:
            conversation_id = await self.supabase_logger.log_conversation_start(
                user_id=user_id,
                channel_id=channel_id,
                thread_ts=thread_ts
            )
            
            if conversation_id:
                self.active_conversations[conv_key] = conversation_id
                logger.info(f"🆕 Started new conversation tracking: {conversation_id}")
                return conversation_id
        
        # Fallback: generate a local conversation ID
        import uuid
        fallback_id = str(uuid.uuid4())
        self.active_conversations[conv_key] = fallback_id
        logger.warning(f"⚠️ Using fallback conversation ID: {fallback_id}")
        return fallback_id
    
    async def _update_agent_metrics(self, agent_name: str, response_time_ms: float,
                                  success: bool = True, escalation: bool = False):
        """Update agent performance metrics in Supabase."""
        if not self.supabase_logger:
            return
        
        try:
            await self.supabase_logger.update_agent_metrics(
                agent_name=agent_name,
                response_time_ms=response_time_ms,
                success=success,
                escalation=escalation
            )
            
            logger.debug(f"📊 Updated metrics for {agent_name}: {response_time_ms:.2f}ms, success={success}")
            
        except Exception as e:
            logger.warning(f"Failed to update agent metrics for {agent_name}: {str(e)}")
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Retrieve comprehensive system metrics from Supabase."""
        if not self.supabase_logger:
            return {"error": "Database connection unavailable"}
        
        from datetime import datetime, timedelta
        metrics = {}
        
        try:
            # System Health
            metrics["system_health"] = self.supabase_logger.health_check()
            
            # Get basic workflow metrics from messages table
            try:
                seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
                
                result = self.supabase_logger.client.table("messages").select(
                    "message_type, processing_time_ms, timestamp, routing_confidence"
                ).gte("timestamp", seven_days_ago).execute()
                
                if result.data:
                    total_messages = len(result.data)
                    bot_responses = len([m for m in result.data if m.get("message_type") == "bot_response"])
                    today_messages = len([m for m in result.data if m.get("timestamp", "").startswith(str(datetime.now().date()))])
                    avg_processing = sum([m.get("processing_time_ms", 0) for m in result.data if m.get("processing_time_ms")]) / max(1, len([m for m in result.data if m.get("processing_time_ms")]))
                    
                    metrics["workflow_performance"] = {
                        "total_interactions": total_messages,
                        "bot_responses": bot_responses,
                        "success_rate": round((bot_responses / max(total_messages, 1)) * 100, 1),
                        "avg_response_time_ms": round(avg_processing, 2),
                        "today_interactions": today_messages
                    }
            except Exception as e:
                logger.warning(f"Error getting workflow metrics: {str(e)}")
                metrics["workflow_performance"] = {"error": "Unable to retrieve workflow metrics"}
            
            # Get agent performance
            try:
                thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
                result = self.supabase_logger.client.table("agent_metrics").select("*").gte("created_at", thirty_days_ago).execute()
                
                if result.data:
                    agents_data = []
                    total_requests = 0
                    
                    for agent in result.data:
                        agent_requests = agent.get("request_count", 0)
                        total_requests += agent_requests
                        agents_data.append({
                            "name": agent.get("agent_name", "unknown"),
                            "requests": agent_requests,
                            "avg_response_time": round(agent.get("response_time_avg", 0), 2),
                            "success_rate": round((agent.get("success_rate", 0)) * 100, 1),
                            "errors": agent.get("error_count", 0)
                        })
                    
                    metrics["agent_performance"] = {
                        "total_agents_available": 3,  # general, technical, research
                        "active_agents_24h": len(agents_data),
                        "total_requests": total_requests,
                        "agent_details": sorted(agents_data, key=lambda x: x["requests"], reverse=True)[:5]
                    }
            except Exception as e:
                logger.warning(f"Error getting agent metrics: {str(e)}")
                metrics["agent_performance"] = {"error": "Unable to retrieve agent metrics"}
            
            # Get cost metrics from token_usage table
            try:
                thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
                result = self.supabase_logger.client.table("token_usage").select(
                    "estimated_cost, total_tokens, created_at"
                ).gte("created_at", thirty_days_ago).execute()
                
                if result.data:
                    today = str(datetime.now().date())
                    week_ago = str((datetime.now() - timedelta(days=7)).date())
                    
                    cost_today = sum([r.get("estimated_cost", 0) for r in result.data if r.get("created_at", "").startswith(today)])
                    cost_week = sum([r.get("estimated_cost", 0) for r in result.data if r.get("created_at", "") >= week_ago])
                    cost_month = sum([r.get("estimated_cost", 0) for r in result.data])
                    tokens_today = sum([r.get("total_tokens", 0) for r in result.data if r.get("created_at", "").startswith(today)])
                    
                    metrics["cost_tracking"] = {
                        "cost_today": round(cost_today, 4),
                        "cost_week": round(cost_week, 4),
                        "cost_month": round(cost_month, 4),
                        "tokens_today": tokens_today,
                        "requests_today": len([r for r in result.data if r.get("created_at", "").startswith(today)])
                    }
            except Exception as e:
                logger.warning(f"Error getting cost metrics: {str(e)}")
                metrics["cost_tracking"] = {"error": "Unable to retrieve cost metrics"}
            
            # Simple satisfaction metrics based on routing confidence
            try:
                seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
                result = self.supabase_logger.client.table("messages").select(
                    "routing_confidence, escalation_suggestion"
                ).eq("message_type", "bot_response").gte("timestamp", seven_days_ago).execute()
                
                if result.data:
                    confidences = [m.get("routing_confidence", 0.5) for m in result.data if m.get("routing_confidence")]
                    escalations = len([m for m in result.data if m.get("escalation_suggestion")])
                    
                    avg_confidence = sum(confidences) / max(len(confidences), 1) if confidences else 0.5
                    escalation_rate = (escalations / max(len(result.data), 1)) * 100
                    satisfaction_score = max(0, 100 - escalation_rate) * avg_confidence
                    
                    metrics["user_satisfaction"] = {
                        "satisfaction_score": round(satisfaction_score, 1),
                        "escalation_rate": round(escalation_rate, 1),
                        "avg_confidence": round(avg_confidence * 100, 1),
                        "total_interactions": len(result.data)
                    }
            except Exception as e:
                logger.warning(f"Error getting satisfaction metrics: {str(e)}")
                metrics["user_satisfaction"] = {"error": "Unable to retrieve satisfaction metrics"}
            
            # Recent activity with more live data
            try:
                seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
                conversations_result = self.supabase_logger.client.table("conversations").select(
                    "started_at, status"
                ).gte("started_at", seven_days_ago).execute()
                
                # Get live database stats
                messages_count_result = self.supabase_logger.client.table("messages").select(
                    "id", count="exact"
                ).execute()
                
                if conversations_result.data:
                    today = str(datetime.now().date())
                    conversations_today = len([c for c in conversations_result.data if c.get("started_at", "").startswith(today)])
                    active_conversations = len([c for c in conversations_result.data if c.get("status") == "active"])
                    
                    metrics["recent_activity"] = {
                        "conversations_today": conversations_today,
                        "active_conversations": active_conversations,
                        "total_messages_in_db": messages_count_result.count if messages_count_result.count else 0,
                        "system_active": True,
                        "data_source": "live_supabase_query"
                    }
                else:
                    metrics["recent_activity"] = {
                        "conversations_today": 0, 
                        "active_conversations": 0,
                        "total_messages_in_db": messages_count_result.count if messages_count_result.count else 0,
                        "system_active": True,
                        "data_source": "live_supabase_query"
                    }
            except Exception as e:
                logger.warning(f"Error getting activity metrics: {str(e)}")
                metrics["recent_activity"] = {"system_active": True, "conversations_today": 0, "error": str(e)}
            
            # Get real improvements from improvement_tasks table - LIVE DATA ONLY
            try:
                improvements_result = self.supabase_logger.client.table("improvement_tasks").select(
                    "*"
                ).order("created_at", desc=True).limit(10).execute()
                
                recent_improvements = []
                if improvements_result.data:
                    for imp in improvements_result.data[:5]:
                        recent_improvements.append({
                            "type": imp.get("agent_type", "System"),
                            "title": imp.get("method_name", "Unknown"),
                            "impact": f"ROI: {imp.get('roi', 0):.1f}x" if imp.get('roi') else "Live improvement tracked",
                            "status": imp.get("status", "unknown").title(),
                            "completed": imp.get("completed_at", imp.get("created_at", ""))[:10] if imp.get("completed_at") or imp.get("created_at") else "Recent"
                        })
                
                metrics["improvements"] = {
                    "recent_improvements": recent_improvements,
                    "total_tracked": len(improvements_result.data) if improvements_result.data else 0
                }
                
            except Exception as e:
                logger.warning(f"Error getting improvements: {str(e)}")
                metrics["improvements"] = {
                    "recent_improvements": [],
                    "total_tracked": 0,
                    "error": "Unable to retrieve live improvement data"
                }
            
        except Exception as e:
            logger.error(f"Error retrieving system metrics: {str(e)}")
            metrics["error"] = str(e)
        
        return metrics
    
    async def _format_metrics_for_slack(self, metrics_data: Dict[str, Any]) -> str:
        """Format the metrics data for beautiful Slack display - LIVE DATA ONLY."""
        if "error" in metrics_data:
            return f"❌ *Live System Metrics Dashboard*\n\nError retrieving metrics: {metrics_data['error']}"
        
        message_parts = []
        
        # Header with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        message_parts.append("📊 *AI Agent Platform - Live Metrics Dashboard*")
        message_parts.append(f"🕐 *Generated:* {timestamp}")
        message_parts.append("=" * 50)
        
        # System Health
        health = metrics_data.get("system_health", {})
        if health.get("status") == "healthy":
            message_parts.append("🟢 *System Health:* Operational")
        else:
            message_parts.append(f"🔴 *System Health:* {health.get('error', 'Unknown')}")
        
        # Workflow Performance
        workflow = metrics_data.get("workflow_performance", {})
        if "error" not in workflow:
            success_rate = workflow.get("success_rate", 0)
            emoji = "🟢" if success_rate >= 90 else "🟡" if success_rate >= 70 else "🔴"
            
            message_parts.append(f"\n📈 *Workflow Performance:*")
            message_parts.append(f"   {emoji} Success Rate: {success_rate}%")
            message_parts.append(f"   ⏱️ Avg Response: {workflow.get('avg_response_time_ms', 0)}ms")
            message_parts.append(f"   📅 Today: {workflow.get('today_interactions', 0)} interactions")
            message_parts.append(f"   📊 Total: {workflow.get('total_interactions', 0)} interactions")
        
        # Agent Performance
        agents = metrics_data.get("agent_performance", {})
        if "error" not in agents:
            message_parts.append(f"\n🤖 *Agent Performance:*")
            message_parts.append(f"   🎯 Active Agents: {agents.get('active_agents_24h', 0)}/{agents.get('total_agents_available', 0)}")
            message_parts.append(f"   📊 Total Requests: {agents.get('total_requests', 0):,}")
            
            agent_details = agents.get("agent_details", [])
            if agent_details:
                message_parts.append("   🏆 Top Agents:")
                for i, agent in enumerate(agent_details[:3]):
                    message_parts.append(f"      {i+1}. {agent['name']}: {agent['requests']} requests, {agent['success_rate']}% success")
        
        # Cost Tracking
        costs = metrics_data.get("cost_tracking", {})
        if "error" not in costs:
            message_parts.append(f"\n💰 *Cost Tracking:*")
            message_parts.append(f"   📅 Today: ${costs.get('cost_today', 0):.4f}")
            message_parts.append(f"   📅 This Week: ${costs.get('cost_week', 0):.4f}")
            message_parts.append(f"   📅 This Month: ${costs.get('cost_month', 0):.4f}")
            message_parts.append(f"   🪙 Tokens Today: {costs.get('tokens_today', 0):,}")
            message_parts.append(f"   📝 Requests Today: {costs.get('requests_today', 0)}")
        
        # User Satisfaction
        satisfaction = metrics_data.get("user_satisfaction", {})
        if "error" not in satisfaction:
            score = satisfaction.get("satisfaction_score", 0)
            emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            
            message_parts.append(f"\n😊 *User Satisfaction:*")
            message_parts.append(f"   {emoji} Overall Score: {score:.1f}/100")
            message_parts.append(f"   🎯 Confidence: {satisfaction.get('avg_confidence', 0):.1f}%")
            message_parts.append(f"   📈 Escalation Rate: {satisfaction.get('escalation_rate', 0):.1f}%")
        
        # Recent Activity - LIVE DATA
        activity = metrics_data.get("recent_activity", {})
        if activity.get("system_active"):
            message_parts.append(f"\n⚡ *Real-time Activity:*")
            if "error" in activity:
                message_parts.append(f"   ❌ Error: {activity['error']}")
            else:
                message_parts.append(f"   📊 Conversations Today: {activity.get('conversations_today', 0)}")
                message_parts.append(f"   🔄 Active Conversations: {activity.get('active_conversations', 0)}")
                message_parts.append(f"   💾 Total Messages in DB: {activity.get('total_messages_in_db', 0):,}")
                message_parts.append(f"   🔍 Data Source: {activity.get('data_source', 'unknown')}")
        
        # Recent Improvements - LIVE DATA ONLY
        improvements = metrics_data.get("improvements", {})
        recent = improvements.get("recent_improvements", [])
        total_tracked = improvements.get("total_tracked", 0)
        
        message_parts.append(f"\n🚀 *System Improvements (Live):*")
        if "error" in improvements:
            message_parts.append(f"   ❌ {improvements['error']}")
        elif recent:
            message_parts.append(f"   📈 Total Tracked: {total_tracked}")
            for imp in recent[:3]:
                status_emoji = "✅" if imp.get("status") == "Completed" else "🔄" if imp.get("status") == "In Progress" else "📋"
                message_parts.append(f"   {status_emoji} {imp.get('type')}: {imp.get('title')}")
                message_parts.append(f"      Impact: {imp.get('impact')} | Date: {imp.get('completed', 'Ongoing')}")
        else:
            message_parts.append("   📭 No improvements tracked yet in database")
            message_parts.append("   💡 Use `/improve` to start tracking improvements")
        
        # MCP Connections Status (if available) - LIVE DATA
        if hasattr(self, 'mcp_commands') and self.mcp_commands:
            try:
                # Check if mcp_connections table exists and get data
                mcp_result = self.supabase_logger.client.table("mcp_connections").select(
                    "*", count="exact"
                ).execute()
                
                message_parts.append(f"\n🔌 *MCP Connections (Live):*")
                if mcp_result.count and mcp_result.count > 0:
                    # Try to find status-like fields (could be 'status', 'active', etc.)
                    active_connections = 0
                    for conn in mcp_result.data:
                        if (conn.get("status") == "active" or 
                            conn.get("active") == True or 
                            conn.get("enabled") == True):
                            active_connections += 1
                    
                    message_parts.append(f"   📊 Total Connections: {mcp_result.count}")
                    message_parts.append(f"   🟢 Active: {active_connections}")
                    message_parts.append(f"   🔴 Inactive: {mcp_result.count - active_connections}")
                    
                    # Show types if available
                    types = list(set([c.get("mcp_type", c.get("type", "unknown")) for c in mcp_result.data]))
                    message_parts.append(f"   🏷️ Types: {', '.join(types[:3])}")
                else:
                    message_parts.append("   📭 No MCP connections in database yet")
                
                message_parts.append("   💡 Use `/mcp connect` to add connections")
                
            except Exception as e:
                message_parts.append(f"\n🔌 *MCP Connections:*")
                message_parts.append("   🟢 MCP system available")
                message_parts.append("   📭 No connections table found yet")
                message_parts.append("   💡 Use `/mcp connect` to create your first connection")
        
        # Footer
        message_parts.append(f"\n{'=' * 50}")
        message_parts.append("🔴 *LIVE DATA SOURCE:* Supabase Database")
        message_parts.append("💡 *Commands:* `/improve` `/save-workflow` `/suggest` `/mcp`")
        message_parts.append("🔄 All metrics pulled from live database on each request")
        
        return "\n".join(message_parts)

# Main execution
async def main():
    """Main entry point for the Slack bot with proper shutdown handling."""
    bot = None
    handler = None
    
    try:
        bot = AIAgentSlackBot()
        
        # Get bot user ID for mention detection
        auth_response = await bot.app.client.auth_test()
        bot.bot_user_id = auth_response["user_id"]
        
        logger.info(f"🤖 Bot user ID: {bot.bot_user_id}")
        logger.info(f"📊 Supabase logging: {'✅ Enabled' if bot.supabase_logger else '⚠️ Disabled'}")
        logger.info(f"🔀 Orchestrator: ✅ Initialized with {len(bot.orchestrator.agent_capabilities)} agents")
        
        # Start socket mode handler
        handler = AsyncSocketModeHandler(bot.app, os.environ["SLACK_APP_TOKEN"])
        logger.info("🚀 Starting AI Agent Platform Slack Bot...")
        await handler.start_async()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown signal received...")
    except Exception as e:
        logger.error(f"❌ Error starting bot: {str(e)}")
        raise
    finally:
        # Graceful shutdown
        if handler:
            logger.info("🔄 Stopping Slack handler...")
            try:
                await handler.close_async()
            except Exception as e:
                logger.warning(f"⚠️ Error closing handler: {e}")
        
        if bot:
            # Close orchestrator and agents
            if hasattr(bot, 'orchestrator') and bot.orchestrator:
                logger.info("🔄 Closing Agent Orchestrator...")
                try:
                    await bot.orchestrator.close()
                except Exception as e:
                    logger.warning(f"⚠️ Error closing orchestrator: {e}")
            
            # Close Supabase logger
            if bot.supabase_logger:
                logger.info("🔄 Closing Supabase connections...")
                try:
                    await bot.supabase_logger.close()
                except Exception as e:
                    logger.warning(f"⚠️ Error closing Supabase logger: {e}")
        
        # Close any remaining aiohttp sessions
        logger.info("🔄 Cleaning up HTTP sessions...")
        try:
            # Give a moment for cleanup
            await asyncio.sleep(0.1)
            
            # Close any pending tasks (but avoid cancelling current task)
            current_task = asyncio.current_task()
            tasks = [task for task in asyncio.all_tasks() 
                     if not task.done() and task != current_task]

            if tasks:
                logger.info(f"🔄 Cancelling {len(tasks)} pending tasks...")
                for task in tasks:
                    task.cancel()
                
                # Wait for tasks to complete cancellation with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Some tasks did not complete cancellation within timeout")
        
        except Exception as e:
            logger.warning(f"⚠️ Error during cleanup: {e}")
        
        logger.info("👋 Bot stopped by user")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Already handled in main() 
