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
from typing import Dict, Any, Optional
from datetime import datetime

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
        """
        Update agent performance metrics with enhanced error handling.
        
        Args:
            agent_name: Name of the agent
            response_time_ms: Response time in milliseconds
            success: Whether the request was successful
            escalation: Whether this resulted in an escalation
        """
        try:
            if self.supabase_logger:
                await self.supabase_logger.update_agent_metrics(
                    agent_name=agent_name,
                    response_time_ms=response_time_ms,
                    success=success,
                    escalation=escalation
                )
                logger.debug(f"📊 Updated metrics for {agent_name}: {response_time_ms:.1f}ms (success: {success})")
            else:
                logger.debug(f"📊 Metrics update skipped (no Supabase): {agent_name} - {response_time_ms:.1f}ms")
                
        except Exception as e:
            logger.error(f"❌ Error updating agent metrics: {str(e)}")
    


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
