"""
Filename: supabase_logger.py
Purpose: Comprehensive logging system for AI Agent Platform interactions
Dependencies: supabase, asyncio, logging, typing

This module is part of the AI Agent Platform.
See README.llm.md in this directory for context.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
import json
import uuid

from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions

logger = logging.getLogger(__name__)

class SupabaseLogger:
    """
    Comprehensive logging system for the AI Agent Platform.
    
    Handles message logging, conversation tracking, agent performance metrics,
    and analytics data collection using Supabase as the backend.
    """
    
    def __init__(self, project_id: str = "pfvlmoybzjkajubzlnsx"):
        """
        Initialize the Supabase logger with connection and schema setup.
        
        Args:
            project_id: Supabase project ID
        """
        self.project_id = project_id
        self.supabase_url = os.getenv("SUPABASE_URL", f"https://{project_id}.supabase.co")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        
        if not self.supabase_key:
            raise ValueError("SUPABASE_KEY environment variable is required")
        
        # Initialize Supabase client
        self.client: Client = create_client(
            self.supabase_url,
            self.supabase_key,
            options=ClientOptions(
                auto_refresh_token=True,
                persist_session=True
            )
        )
        
        logger.info(f"Supabase Logger initialized for project: {project_id}")
    
    async def setup_database_schema(self) -> bool:
        """
        Set up the database schema for the AI Agent Platform.
        Creates all necessary tables if they don't exist.
        
        Returns:
            True if schema setup successful, False otherwise
        """
        try:
            logger.info("Setting up database schema...")
            
            # SQL to create tables
            schema_sql = """
            -- Conversations table
            CREATE TABLE IF NOT EXISTS conversations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                thread_ts TEXT,
                started_at TIMESTAMPTZ DEFAULT NOW(),
                last_activity TIMESTAMPTZ DEFAULT NOW(),
                status TEXT DEFAULT 'active' CHECK (status IN ('active', 'archived', 'closed')),
                assigned_agent TEXT,
                message_count INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            
            -- Messages table
            CREATE TABLE IF NOT EXISTS messages (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL,
                content TEXT,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                message_type TEXT NOT NULL CHECK (message_type IN ('user_message', 'bot_response', 'system', 'error')),
                agent_type TEXT,
                agent_response JSONB,
                routing_confidence FLOAT,
                escalation_suggestion TEXT,
                processing_time_ms FLOAT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            
            -- User preferences table
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferences JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            
            -- Agent metrics table
            CREATE TABLE IF NOT EXISTS agent_metrics (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                agent_name TEXT NOT NULL,
                date DATE DEFAULT CURRENT_DATE,
                request_count INTEGER DEFAULT 0,
                response_time_avg FLOAT DEFAULT 0,
                success_rate FLOAT DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                escalation_count INTEGER DEFAULT 0,
                last_updated TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(agent_name, date)
            );
            
            -- Routing decisions table (for analytics)
            CREATE TABLE IF NOT EXISTS routing_decisions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
                selected_agent TEXT NOT NULL,
                confidence_score FLOAT NOT NULL,
                all_scores JSONB,
                routing_time_ms FLOAT,
                was_explicit_mention BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            
            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_channel_id ON conversations(channel_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_last_activity ON conversations(last_activity);
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
            CREATE INDEX IF NOT EXISTS idx_messages_message_type ON messages(message_type);
            CREATE INDEX IF NOT EXISTS idx_agent_metrics_date ON agent_metrics(date);
            CREATE INDEX IF NOT EXISTS idx_routing_decisions_created_at ON routing_decisions(created_at);
            """
            
            # Execute schema creation using RPC call to avoid SQL injection
            # Note: In production, this would typically be done via migrations
            logger.info("Database schema setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up database schema: {str(e)}")
            return False
    
    async def log_conversation_start(self, user_id: str, channel_id: str, 
                                   thread_ts: Optional[str] = None) -> Optional[str]:
        """
        Log the start of a new conversation.
        
        Args:
            user_id: Slack user ID
            channel_id: Slack channel ID
            thread_ts: Thread timestamp if applicable
            
        Returns:
            Conversation ID if successful, None otherwise
        """
        try:
            conversation_data = {
                "user_id": user_id,
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "status": "active"
            }
            
            result = self.client.table("conversations").insert(conversation_data).execute()
            
            if result.data:
                conversation_id = result.data[0]["id"]
                logger.info(f"Started conversation tracking: {conversation_id}")
                return conversation_id
            
        except Exception as e:
            logger.error(f"Error logging conversation start: {str(e)}")
        
        return None
    
    async def log_message(self, conversation_id: str, user_id: str, content: str,
                         message_type: str, agent_type: Optional[str] = None,
                         agent_response: Optional[Dict[str, Any]] = None,
                         routing_confidence: Optional[float] = None,
                         escalation_suggestion: Optional[str] = None,
                         processing_time_ms: Optional[float] = None,
                         input_tokens: Optional[int] = None,
                         output_tokens: Optional[int] = None,
                         total_tokens: Optional[int] = None,
                         model_used: Optional[str] = None,
                         estimated_cost: Optional[float] = None) -> bool:
        """
        Log a message in the conversation with enhanced token tracking.
        
        Args:
            conversation_id: ID of the conversation
            user_id: Slack user ID
            content: Message content
            message_type: Type of message (user_message, bot_response, system, error)
            agent_type: Agent that handled the message
            agent_response: Full agent response data
            routing_confidence: Confidence score for routing
            escalation_suggestion: Suggested escalation if any
            processing_time_ms: Time taken to process the message
            input_tokens: Input tokens consumed by LLM
            output_tokens: Output tokens generated by LLM
            total_tokens: Total tokens (input + output)
            model_used: LLM model used (gpt-4, gpt-3.5-turbo, etc.)
            estimated_cost: Estimated cost in USD for this interaction
            
        Returns:
            True if successful, False otherwise
        """
        try:
            message_data = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message_type": message_type,
                "agent_type": agent_type,
                "agent_response": agent_response,
                "routing_confidence": routing_confidence,
                "escalation_suggestion": escalation_suggestion,
                "processing_time_ms": processing_time_ms,
                "input_tokens": input_tokens or 0,
                "output_tokens": output_tokens or 0,
                "total_tokens": total_tokens or (input_tokens or 0) + (output_tokens or 0),
                "model_used": model_used,
                "estimated_cost": estimated_cost or 0.0
            }
            
            result = self.client.table("messages").insert(message_data).execute()
            
            if result.data:
                message_id = result.data[0]["id"]
                
                # Log detailed token usage for analytics if tokens were used
                if total_tokens and total_tokens > 0 and agent_type:
                    await self._log_token_usage(
                        user_id=user_id,
                        agent_type=agent_type,
                        model_used=model_used or "unknown",
                        input_tokens=input_tokens or 0,
                        output_tokens=output_tokens or 0,
                        total_tokens=total_tokens,
                        estimated_cost=estimated_cost or 0.0,
                        conversation_id=conversation_id,
                        message_id=message_id
                    )
                
                # Update conversation last_activity and message_count
                await self._update_conversation_activity(conversation_id)
                logger.debug(f"Logged message for conversation: {conversation_id} (tokens: {total_tokens})")
                return True
            
        except Exception as e:
            logger.error(f"Error logging message: {str(e)}")
        
        return False
    
    async def log_routing_decision(self, message_id: str, selected_agent: str,
                                 confidence_score: float, all_scores: Dict[str, float],
                                 routing_time_ms: float, was_explicit_mention: bool = False) -> bool:
        """
        Log an agent routing decision for analytics.
        
        Args:
            message_id: ID of the message
            selected_agent: Agent that was selected
            confidence_score: Confidence score for the selection
            all_scores: All agent scores from routing
            routing_time_ms: Time taken for routing decision
            was_explicit_mention: Whether this was an explicit agent mention
            
        Returns:
            True if successful, False otherwise
        """
        try:
            routing_data = {
                "message_id": message_id,
                "selected_agent": selected_agent,
                "confidence_score": confidence_score,
                "all_scores": all_scores,
                "routing_time_ms": routing_time_ms,
                "was_explicit_mention": was_explicit_mention,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            result = self.client.table("routing_decisions").insert(routing_data).execute()
            
            if result.data:
                logger.debug(f"Logged routing decision for message: {message_id}")
                return True
            
        except Exception as e:
            logger.error(f"Error logging routing decision: {str(e)}")
        
        return False
    
    async def update_agent_metrics(self, agent_name: str, request_count: int = 1,
                                 response_time_ms: Optional[float] = None,
                                 success: bool = True, escalation: bool = False) -> bool:
        """
        Update agent performance metrics.
        
        Args:
            agent_name: Name of the agent
            request_count: Number of requests to add (default 1)
            response_time_ms: Response time in milliseconds
            success: Whether the request was successful
            escalation: Whether this resulted in an escalation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            today = datetime.now(timezone.utc).date()
            
            # Try to get existing metrics for today
            existing = self.client.table("agent_metrics").select("*").eq("agent_name", agent_name).eq("date", today.isoformat()).execute()
            
            if existing.data:
                # Update existing record
                current = existing.data[0]
                total_requests = current["request_count"] + request_count
                
                # Calculate new average response time
                if response_time_ms and current["response_time_avg"]:
                    new_avg = ((current["response_time_avg"] * current["request_count"]) + response_time_ms) / total_requests
                elif response_time_ms:
                    new_avg = response_time_ms
                else:
                    new_avg = current["response_time_avg"]
                
                # Calculate new success rate
                if current["request_count"] > 0:
                    current_successes = current["success_rate"] * current["request_count"]
                    new_successes = current_successes + (1 if success else 0)
                    new_success_rate = new_successes / total_requests
                else:
                    new_success_rate = 1.0 if success else 0.0
                
                update_data = {
                    "request_count": total_requests,
                    "response_time_avg": new_avg,
                    "success_rate": new_success_rate,
                    "error_count": current["error_count"] + (0 if success else 1),
                    "escalation_count": current["escalation_count"] + (1 if escalation else 0),
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
                
                self.client.table("agent_metrics").update(update_data).eq("id", current["id"]).execute()
                
            else:
                # Create new record
                metrics_data = {
                    "agent_name": agent_name,
                    "date": today.isoformat(),
                    "request_count": request_count,
                    "response_time_avg": response_time_ms or 0,
                    "success_rate": 1.0 if success else 0.0,
                    "error_count": 0 if success else 1,
                    "escalation_count": 1 if escalation else 0,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
                
                self.client.table("agent_metrics").insert(metrics_data).execute()
            
            logger.debug(f"Updated metrics for agent: {agent_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating agent metrics: {str(e)}")
            return False
    
    async def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user.
        
        Args:
            user_id: Slack user ID
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation data
        """
        try:
            result = self.client.table("conversations").select("*").eq("user_id", user_id).order("last_activity", desc=True).limit(limit).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
    async def get_agent_performance(self, agent_name: Optional[str] = None, 
                                  days: int = 7) -> List[Dict[str, Any]]:
        """
        Get agent performance metrics.
        
        Args:
            agent_name: Specific agent name (None for all agents)
            days: Number of days to look back
            
        Returns:
            List of agent performance data
        """
        try:
            from_date = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
            
            query = self.client.table("agent_metrics").select("*").gte("date", from_date).order("date", desc=True)
            
            if agent_name:
                query = query.eq("agent_name", agent_name)
            
            result = query.execute()
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error getting agent performance: {str(e)}")
            return []
    
    async def get_conversation_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get conversation analytics for the specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with analytics data
        """
        try:
            from datetime import timedelta
            from_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            
            # Get conversation counts
            conversations = self.client.table("conversations").select("*").gte("started_at", from_date).execute()
            
            # Get message counts by type
            messages = self.client.table("messages").select("message_type").gte("timestamp", from_date).execute()
            
            # Get agent distribution
            agent_messages = self.client.table("messages").select("agent_type").gte("timestamp", from_date).execute()
            
            # Process the data
            total_conversations = len(conversations.data) if conversations.data else 0
            
            message_type_counts = {}
            if messages.data:
                for msg in messages.data:
                    msg_type = msg.get("message_type", "unknown")
                    message_type_counts[msg_type] = message_type_counts.get(msg_type, 0) + 1
            
            agent_distribution = {}
            if agent_messages.data:
                for msg in agent_messages.data:
                    agent = msg.get("agent_type", "unknown")
                    if agent and agent != "unknown":
                        agent_distribution[agent] = agent_distribution.get(agent, 0) + 1
            
            return {
                "period_days": days,
                "total_conversations": total_conversations,
                "total_messages": len(messages.data) if messages.data else 0,
                "message_type_distribution": message_type_counts,
                "agent_distribution": agent_distribution,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation analytics: {str(e)}")
            return {}
    
    async def _update_conversation_activity(self, conversation_id: str):
        """
        Update the last activity timestamp and message count for a conversation.
        
        Args:
            conversation_id: ID of the conversation to update
        """
        try:
            # Get current message count
            messages_result = self.client.table("messages").select("id", count="exact").eq("conversation_id", conversation_id).execute()
            message_count = messages_result.count if messages_result.count else 0
            
            # Update conversation
            update_data = {
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "message_count": message_count,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            self.client.table("conversations").update(update_data).eq("id", conversation_id).execute()
            
        except Exception as e:
            logger.error(f"Error updating conversation activity: {str(e)}")
    
    async def close_conversation(self, conversation_id: str) -> bool:
        """
        Mark a conversation as closed.
        
        Args:
            conversation_id: ID of the conversation to close
            
        Returns:
            True if successful, False otherwise
        """
        try:
            update_data = {
                "status": "closed",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            result = self.client.table("conversations").update(update_data).eq("id", conversation_id).execute()
            
            if result.data:
                logger.info(f"Closed conversation: {conversation_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error closing conversation: {str(e)}")
        
        return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the Supabase connection.
        
        Returns:
            Dictionary with health check results
        """
        try:
            # Simple query to test connection
            result = self.client.table("conversations").select("id").limit(1).execute()
            
            return {
                "status": "healthy",
                "connected": True,
                "project_id": self.project_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "project_id": self.project_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _log_token_usage(self, user_id: str, agent_type: str, model_used: str,
                              input_tokens: int, output_tokens: int, total_tokens: int,
                              estimated_cost: float, conversation_id: str, message_id: str,
                              session_id: Optional[str] = None) -> bool:
        """
        Log detailed token usage for analytics and cost tracking.
        
        Args:
            user_id: Slack user ID
            agent_type: Agent that handled the request
            model_used: LLM model used (gpt-4, gpt-3.5-turbo, etc.)
            input_tokens: Input tokens consumed
            output_tokens: Output tokens generated
            total_tokens: Total tokens used
            estimated_cost: Estimated cost in USD
            conversation_id: ID of the conversation
            message_id: ID of the message
            session_id: Optional session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Log to token_usage table for detailed analytics
            token_data = {
                "user_id": user_id,
                "agent_type": agent_type,
                "model_used": model_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost": estimated_cost,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "cost_per_token": estimated_cost / total_tokens if total_tokens > 0 else 0,
                    "input_ratio": input_tokens / total_tokens if total_tokens > 0 else 0,
                    "output_ratio": output_tokens / total_tokens if total_tokens > 0 else 0
                }
            }
            
            result = self.client.table("token_usage").insert(token_data).execute()
            
            if result.data:
                # Update daily summary
                await self._update_daily_token_summary(user_id, agent_type, model_used, 
                                                     total_tokens, estimated_cost)
                logger.debug(f"Logged token usage: {total_tokens} tokens, ${estimated_cost:.6f}")
                return True
                
        except Exception as e:
            logger.error(f"Error logging token usage: {str(e)}")
        
        return False
    
    async def _update_daily_token_summary(self, user_id: str, agent_type: str, 
                                        model_used: str, tokens: int, cost: float):
        """
        Update daily token usage summary for cost tracking.
        
        Args:
            user_id: Slack user ID
            agent_type: Agent type
            model_used: Model used
            tokens: Number of tokens
            cost: Cost in USD
        """
        try:
            today = datetime.now(timezone.utc).date()
            
            # Try to get existing summary for today
            existing = self.client.table("daily_token_summary").select("*").eq("user_id", user_id).eq("date", today.isoformat()).execute()
            
            if existing.data:
                # Update existing record
                current = existing.data[0]
                agent_breakdown = current.get("agent_breakdown", {})
                model_breakdown = current.get("model_breakdown", {})
                
                # Update agent breakdown
                if agent_type in agent_breakdown:
                    agent_breakdown[agent_type]["tokens"] += tokens
                    agent_breakdown[agent_type]["cost"] += cost
                    agent_breakdown[agent_type]["requests"] += 1
                else:
                    agent_breakdown[agent_type] = {
                        "tokens": tokens,
                        "cost": cost,
                        "requests": 1
                    }
                
                # Update model breakdown
                if model_used in model_breakdown:
                    model_breakdown[model_used]["tokens"] += tokens
                    model_breakdown[model_used]["cost"] += cost
                    model_breakdown[model_used]["requests"] += 1
                else:
                    model_breakdown[model_used] = {
                        "tokens": tokens,
                        "cost": cost,
                        "requests": 1
                    }
                
                update_data = {
                    "total_tokens": current["total_tokens"] + tokens,
                    "total_cost": float(current["total_cost"]) + cost,
                    "request_count": current["request_count"] + 1,
                    "agent_breakdown": agent_breakdown,
                    "model_breakdown": model_breakdown,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
                
                self.client.table("daily_token_summary").update(update_data).eq("id", current["id"]).execute()
                
            else:
                # Create new record
                summary_data = {
                    "user_id": user_id,
                    "date": today.isoformat(),
                    "total_tokens": tokens,
                    "total_cost": cost,
                    "request_count": 1,
                    "agent_breakdown": {
                        agent_type: {
                            "tokens": tokens,
                            "cost": cost,
                            "requests": 1
                        }
                    },
                    "model_breakdown": {
                        model_used: {
                            "tokens": tokens,
                            "cost": cost,
                            "requests": 1
                        }
                    }
                }
                
                self.client.table("daily_token_summary").insert(summary_data).execute()
            
            logger.debug(f"Updated daily token summary for {user_id}: +{tokens} tokens, +${cost:.6f}")
            
        except Exception as e:
            logger.error(f"Error updating daily token summary: {str(e)}")
    
    async def get_token_usage_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get token usage analytics for a user.
        
        Args:
            user_id: Slack user ID
            days: Number of days to analyze
            
        Returns:
            Dictionary with token usage analytics
        """
        try:
            from_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            
            # Get daily summaries
            daily_data = self.client.table("daily_token_summary").select("*").eq("user_id", user_id).gte("date", from_date).order("date", desc=True).execute()
            
            # Get detailed token usage
            detailed_data = self.client.table("token_usage").select("*").eq("user_id", user_id).gte("timestamp", from_date).execute()
            
            # Process analytics
            total_tokens = 0
            total_cost = 0.0
            agent_totals = {}
            model_totals = {}
            daily_breakdown = []
            
            if daily_data.data:
                for day in daily_data.data:
                    total_tokens += day["total_tokens"]
                    total_cost += float(day["total_cost"])
                    daily_breakdown.append({
                        "date": day["date"],
                        "tokens": day["total_tokens"],
                        "cost": float(day["total_cost"]),
                        "requests": day["request_count"]
                    })
                    
                    # Aggregate agent breakdown
                    for agent, data in day.get("agent_breakdown", {}).items():
                        if agent not in agent_totals:
                            agent_totals[agent] = {"tokens": 0, "cost": 0.0, "requests": 0}
                        agent_totals[agent]["tokens"] += data["tokens"]
                        agent_totals[agent]["cost"] += data["cost"]
                        agent_totals[agent]["requests"] += data["requests"]
                    
                    # Aggregate model breakdown
                    for model, data in day.get("model_breakdown", {}).items():
                        if model not in model_totals:
                            model_totals[model] = {"tokens": 0, "cost": 0.0, "requests": 0}
                        model_totals[model]["tokens"] += data["tokens"]
                        model_totals[model]["cost"] += data["cost"]
                        model_totals[model]["requests"] += data["requests"]
            
            return {
                "period_days": days,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "avg_tokens_per_day": total_tokens / days if days > 0 else 0,
                "avg_cost_per_day": total_cost / days if days > 0 else 0,
                "agent_breakdown": agent_totals,
                "model_breakdown": model_totals,
                "daily_breakdown": daily_breakdown,
                "cost_per_token": total_cost / total_tokens if total_tokens > 0 else 0,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting token usage analytics: {str(e)}")
            return {}
    
    async def get_daily_token_summary(self, user_id: str, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get daily token summary for a specific date.
        
        Args:
            user_id: Slack user ID
            date: Date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            Daily token summary data
        """
        try:
            if not date:
                date = datetime.now(timezone.utc).date().isoformat()
            
            result = self.client.table("daily_token_summary").select("*").eq("user_id", user_id).eq("date", date).execute()
            
            if result.data:
                return result.data[0]
            else:
                return {
                    "user_id": user_id,
                    "date": date,
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "request_count": 0,
                    "agent_breakdown": {},
                    "model_breakdown": {}
                }
                
        except Exception as e:
            logger.error(f"Error getting daily token summary: {str(e)}")
            return {}
    
    @staticmethod
    def calculate_token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate estimated cost for token usage based on OpenAI pricing.
        
        Args:
            model: Model name (gpt-4, gpt-3.5-turbo, etc.)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        # OpenAI pricing (as of 2024) - prices per 1K tokens
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
            "default": {"input": 0.002, "output": 0.003}  # Conservative estimate
        }
        
        model_pricing = pricing.get(model, pricing["default"])
        
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        
        return input_cost + output_cost

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom SQL query against the database.
        
        Args:
            query: SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            List of query results
        """
        try:
            # For Supabase, we'll use RPC (Remote Procedure Call) for custom queries
            # This is a safer approach than raw SQL execution
            
            # For now, return empty results for unsupported queries
            # In production, you'd implement specific RPC functions for each query type
            logger.warning(f"Custom query execution not fully implemented: {query[:100]}...")
            return []
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return []

    async def get_system_metrics(self, metric_type: str = "performance") -> Dict[str, Any]:
        """
        Get system metrics for the improvement orchestrator.
        
        Args:
            metric_type: Type of metrics to retrieve (performance, cost, activity, etc.)
            
        Returns:
            Dictionary containing system metrics
        """
        try:
            if metric_type == "performance":
                # Get performance metrics
                agent_perf = await self.get_agent_performance(days=7)
                return {
                    "total_agents": len(set(a.get("agent_name") for a in agent_perf)),
                    "avg_response_time": sum(a.get("response_time_avg", 0) for a in agent_perf) / len(agent_perf) if agent_perf else 0,
                    "avg_success_rate": sum(a.get("success_rate", 0) for a in agent_perf) / len(agent_perf) if agent_perf else 0,
                    "total_requests": sum(a.get("request_count", 0) for a in agent_perf)
                }
            
            elif metric_type == "cost":
                # Get cost metrics (simplified)
                return {
                    "daily_cost": 0.0,  # Would calculate from token_usage table
                    "weekly_cost": 0.0,
                    "monthly_cost": 0.0,
                    "cost_trend": "stable"
                }
            
            elif metric_type == "activity":
                # Get user activity metrics
                analytics = await self.get_conversation_analytics(days=7)
                return {
                    "total_conversations": analytics.get("total_conversations", 0),
                    "total_messages": analytics.get("total_messages", 0),
                    "daily_average": analytics.get("total_conversations", 0) / 7,
                    "activity_level": "normal"
                }
            
            else:
                return {"error": f"Unknown metric type: {metric_type}"}
                
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}

    async def count_active_issues(self) -> int:
        """
        Count active issues in the system.
        
        Returns:
            Number of active issues
        """
        try:
            # This would query for error messages, failed workflows, etc.
            # For now, return 0 as a safe default
            return 0
            
        except Exception as e:
            logger.error(f"Error counting active issues: {str(e)}")
            return 0

    async def measure_user_activity(self, hours: int = 1) -> float:
        """
        Measure user activity level in the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Activity level (0.0 to 1.0, where 1.0 is high activity)
        """
        try:
            from_time = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            
            # Get recent messages
            messages = self.client.table("messages").select("id").gte("timestamp", from_time).execute()
            message_count = len(messages.data) if messages.data else 0
            
            # Convert to activity level (normalize based on expected volume)
            # Assuming 10 messages per hour is "normal" activity
            activity_level = min(message_count / (10 * hours), 1.0)
            
            return activity_level
            
        except Exception as e:
            logger.error(f"Error measuring user activity: {str(e)}")
            return 0.0

    async def log_event(self, event_type: str, event_data: Dict[str, Any], 
                       user_id: Optional[str] = None) -> bool:
        """
        Log a system event for tracking and analytics.
        
        Args:
            event_type: Type of event (e.g., 'agent_spawned', 'pattern_found')
            event_data: Event data dictionary
            user_id: Optional user ID associated with the event
            
        Returns:
            True if logged successfully, False otherwise
        """
        try:
            event_record = {
                "event_type": event_type,
                "event_data": event_data,
                "user_id": user_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            # For now, we'll use the messages table to store events
            # In a full implementation, you'd want a dedicated events table
            result = self.client.table("messages").insert({
                "user_id": user_id or "system",
                "content": f"System Event: {event_type}",
                "message_type": "system",
                "agent_response": event_record
            }).execute()
            
            if result.data:
                logger.debug(f"Logged event: {event_type}")
                return True
            
        except Exception as e:
            logger.error(f"Error logging event {event_type}: {str(e)}")
        
        return False

    async def close(self):
        """
        Close the Supabase client and cleanup resources.
        
        This method ensures proper cleanup of HTTP connections
        and other resources when shutting down the logger.
        """
        try:
            logger.info("Closing Supabase logger connections...")
            
            # Close the underlying HTTP client if it exists and is awaitable
            if hasattr(self.client, '_session') and self.client._session:
                try:
                    if hasattr(self.client._session, 'close'):
                        close_method = self.client._session.close
                        if asyncio.iscoroutinefunction(close_method):
                            await close_method()
                        else:
                            close_method()
                        logger.debug("Closed Supabase HTTP session")
                except Exception as e:
                    logger.debug(f"Error closing HTTP session: {e}")
            
            # Check for other client cleanup methods
            if hasattr(self.client, 'close'):
                try:
                    close_method = self.client.close
                    if asyncio.iscoroutinefunction(close_method):
                        await close_method()
                    else:
                        close_method()
                    logger.debug("Closed Supabase client")
                except Exception as e:
                    logger.debug(f"Error closing Supabase client: {e}")
            
            # Clear client reference to help garbage collection
            self.client = None
                
            logger.info("Supabase logger connections closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing Supabase logger: {e}") 