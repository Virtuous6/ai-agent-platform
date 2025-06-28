"""
Simplified Slack Bot Adapter for AI Agent Platform
Works with the new refactored architecture using dynamic agent spawning
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)

class SlackBot:
    """
    Simplified Slack bot for the refactored AI Agent Platform.
    
    Features:
    - Uses new Orchestrator for dynamic agent spawning
    - Event-driven architecture integration
    - Simplified command handling
    - Self-improvement feedback loop
    """
    
    def __init__(self, orchestrator, event_bus, storage):
        """Initialize the Slack bot with platform components."""
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self.storage = storage
        
        # Initialize Slack app
        self.app = AsyncApp(
            token=os.environ.get("SLACK_BOT_TOKEN"),
            signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
        )
        
        # Bot user ID for mention detection
        self.bot_user_id = None
        
        # Active conversations tracking
        self.active_conversations = {}
        
        # Handler for cleanup
        self.handler = None
        
        # Register event handlers
        self._register_handlers()
        
        logger.info("🤖 Simplified Slack Bot initialized")
    
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
        
        # Handle slash commands
        @self.app.command("/improve")
        async def handle_improve_command(ack, say, command, client):
            await ack()
            await self._handle_command("improve", command, say, client)
        
        @self.app.command("/save-workflow")
        async def handle_save_workflow_command(ack, say, command, client):
            await ack()
            await self._handle_command("save-workflow", command, say, client)
        
        @self.app.command("/status")
        async def handle_status_command(ack, say, command, client):
            await ack()
            await self._handle_command("status", command, say, client)
        
        @self.app.command("/agents")
        async def handle_agents_command(ack, say, command, client):
            await ack()
            await self._handle_command("agents", command, say, client)
    
    async def _handle_message(self, event: Dict[str, Any], say, client, is_mention: bool = False):
        """Handle incoming messages from Slack."""
        try:
            # Extract message details
            user_id = event.get("user")
            channel_id = event.get("channel")
            text = event.get("text", "")
            ts = event.get("ts")
            
            # Skip bot messages
            if user_id == self.bot_user_id:
                return
            
            # Clean message text (remove bot mention if present)
            if is_mention and self.bot_user_id:
                text = text.replace(f"<@{self.bot_user_id}>", "").strip()
            
            if not text:
                await say("Hi! How can I help you today?")
                return
            
            # Show typing indicator
            try:
                await client.chat_postMessage(
                    channel=channel_id,
                    text="🤔 Thinking...",
                    thread_ts=ts
                )
            except:
                pass  # Ignore if can't show typing
            
            # Build context
            context = {
                "user_id": user_id,
                "channel_id": channel_id,
                "is_mention": is_mention,
                "thread_ts": ts,
                "slack_client": client
            }
            
            # Process with orchestrator
            start_time = datetime.utcnow()
            response = await self.orchestrator.process(text, context)
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Send response
            await say(response, thread_ts=ts if is_mention else None)
            
            # Log interaction
            await self._log_interaction(user_id, channel_id, text, response, processing_time)
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await say(f"Sorry, I encountered an error: {str(e)}")
    
    async def _handle_command(self, command_type: str, command, say, client):
        """Handle slash commands."""
        try:
            user_id = command.get("user_id")
            channel_id = command.get("channel_id")
            text = command.get("text", "")
            
            # Build command message
            message = f"/{command_type} {text}".strip()
            
            # Build context
            context = {
                "user_id": user_id,
                "channel_id": channel_id,
                "is_command": True,
                "command_type": command_type,
                "slack_client": client
            }
            
            # Process with orchestrator
            response = await self.orchestrator.process(message, context)
            
            # Send response
            await say(response)
            
        except Exception as e:
            logger.error(f"Error handling command {command_type}: {e}")
            await say(f"Sorry, I encountered an error processing that command: {str(e)}")
    
    async def _log_interaction(self, user_id: str, channel_id: str, 
                              message: str, response: str, processing_time_ms: float):
        """Log interaction to Supabase."""
        if not self.storage:
            return
        
        try:
            interaction_data = {
                "user_id": user_id,
                "channel_id": channel_id,
                "message": message,
                "response": response,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.utcnow().isoformat(),
                "platform": "slack"
            }
            
            self.storage.client.table("messages").insert(interaction_data).execute()
            
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")
    
    async def _get_bot_user_id(self, client):
        """Get bot user ID for mention detection."""
        try:
            auth_result = await client.auth_test()
            self.bot_user_id = auth_result["user_id"]
            logger.info(f"Bot user ID: {self.bot_user_id}")
        except Exception as e:
            logger.error(f"Failed to get bot user ID: {e}")
    
    async def start(self):
        """Start the Slack bot."""
        try:
            # Get bot user ID
            await self._get_bot_user_id(self.app.client)
            
            # Start socket mode handler
            self.handler = AsyncSocketModeHandler(self.app, os.environ.get("SLACK_APP_TOKEN"))
            
            logger.info("🚀 Slack bot starting...")
            
            # Start the handler in a way that can be cancelled
            try:
                await self.handler.start_async()
            except asyncio.CancelledError:
                logger.info("🔄 Bot start cancelled")
                raise
            except Exception as e:
                logger.error(f"❌ Handler error: {e}")
                raise
            
        except asyncio.CancelledError:
            # Clean cancellation
            logger.info("🔄 Bot startup cancelled")
            raise
        except Exception as e:
            logger.error(f"Failed to start Slack bot: {e}")
            raise
    
    async def stop(self):
        """Stop the Slack bot and cleanup resources."""
        try:
            if self.handler:
                logger.info("🔄 Stopping Slack handler...")
                await self.handler.close_async()
                
            # Close any remaining aiohttp sessions
            logger.info("🔄 Cleaning up HTTP sessions...")
            
            # Give a moment for cleanup
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error during bot shutdown: {e}")
        
        logger.info("👋 Slack bot stopped")
