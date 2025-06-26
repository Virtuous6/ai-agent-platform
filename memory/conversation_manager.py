"""
Conversation Memory Manager

Manages conversation context, memory retrieval, and intelligent context injection
for LangGraph workflows and agent interactions.
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque

from .vector_store import VectorMemoryStore
from database.supabase_logger import SupabaseLogger

logger = logging.getLogger(__name__)

class ConversationMemoryManager:
    """
    Manages conversation memory and context for intelligent agent interactions.
    
    Provides:
    - Context tracking and retrieval
    - Memory-aware agent responses  
    - Conversation state management
    - Intelligent memory summarization
    """
    
    def __init__(self, vector_store: Optional[VectorMemoryStore] = None,
                 supabase_logger: Optional[SupabaseLogger] = None,
                 context_window_size: int = 10,
                 memory_retention_days: int = 90):
        """
        Initialize conversation memory manager.
        
        Args:
            vector_store: Vector memory store for semantic search
            supabase_logger: Database logger for conversation storage
            context_window_size: Number of recent messages to keep in context
            memory_retention_days: Days to retain conversation memories
        """
        self.vector_store = vector_store or VectorMemoryStore()
        self.supabase_logger = supabase_logger or SupabaseLogger()
        self.context_window_size = context_window_size
        self.memory_retention_days = memory_retention_days
        
        # In-memory conversation context cache
        self.conversation_contexts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=context_window_size))
        self.conversation_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Memory summary cache
        self.memory_summaries: Dict[str, Dict[str, Any]] = {}
        self.summary_cache_ttl = timedelta(hours=1)
        
        logger.info(f"Conversation Memory Manager initialized")
        logger.info(f"  Context window: {context_window_size} messages")
        logger.info(f"  Memory retention: {memory_retention_days} days")
        logger.info(f"  Vector store available: {self.vector_store.is_available()}")
    
    async def start_conversation(self, user_id: str, 
                               conversation_type: str = "general",
                               initial_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new conversation and return conversation ID.
        
        Args:
            user_id: User identifier
            conversation_type: Type of conversation (general, support, technical, etc.)
            initial_context: Initial conversation context
            
        Returns:
            Generated conversation ID
        """
        conversation_id = str(uuid.uuid4())
        
        try:
            # Create conversation record
            metadata = {
                "user_id": user_id,
                "conversation_type": conversation_type,
                "started_at": datetime.utcnow().isoformat(),
                "message_count": 0,
                "last_activity": datetime.utcnow().isoformat(),
                "context": initial_context or {},
                "agents_involved": [],
                "escalation_count": 0,
                "satisfaction_rating": None
            }
            
            # Log conversation start and get the database-generated conversation ID
            db_conversation_id = await self.supabase_logger.log_conversation_start(
                user_id=user_id,
                channel_id="memory_system",  # Use a default channel ID for memory system
                thread_ts=None
            )
            
            # Use database ID if available, otherwise fallback to generated ID
            if db_conversation_id:
                conversation_id = db_conversation_id
            
            self.conversation_metadata[conversation_id] = metadata
            
            logger.info(f"Started conversation {conversation_id} for user {user_id}")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            return conversation_id  # Return ID even if logging fails
    
    async def add_message(self, conversation_id: str, content: str, 
                         sender_type: str, sender_id: str,
                         message_type: str = "text",
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a message to conversation and store in memory.
        
        Args:
            conversation_id: Conversation identifier
            content: Message content
            sender_type: Type of sender (user, agent, system)
            sender_id: Identifier of sender
            message_type: Type of message (text, command, response, etc.)
            metadata: Additional message metadata
            
        Returns:
            Generated message ID
        """
        message_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        try:
            # Create message record
            message = {
                "id": message_id,
                "conversation_id": conversation_id,
                "content": content,
                "sender_type": sender_type,
                "sender_id": sender_id,
                "message_type": message_type,
                "timestamp": timestamp.isoformat(),
                "metadata": metadata or {}
            }
            
            # Add to conversation context
            self.conversation_contexts[conversation_id].append(message)
            
            # Update conversation metadata
            if conversation_id in self.conversation_metadata:
                conv_meta = self.conversation_metadata[conversation_id]
                conv_meta["message_count"] += 1
                conv_meta["last_activity"] = timestamp.isoformat()
                
                # Track agent involvement
                if sender_type == "agent" and sender_id not in conv_meta["agents_involved"]:
                    conv_meta["agents_involved"].append(sender_id)
            
            # Determine content type for memory storage
            if sender_type == "user":
                content_type_message = "user_message"  # For messages table
                content_type_memory = "message"  # For conversation_embeddings table
            elif sender_type == "agent":
                content_type_message = "bot_response"  # For messages table
                content_type_memory = "response"  # For conversation_embeddings table
            else:
                content_type_message = "system"  # For messages table
                content_type_memory = "message"  # For conversation_embeddings table
            
            # Log message to database first and get the actual message ID
            db_message_result = await self.supabase_logger.log_message(
                conversation_id=conversation_id,
                user_id=user_id,
                content=content,
                message_type=content_type_message,
                agent_type=sender_id if sender_type == "agent" else None,
                agent_response=metadata if sender_type == "agent" else None
            )
            
            # Store in vector memory if significant content (use database message ID if available)
            if len(content.strip()) > 10:  # Only store substantial messages
                conversation_user_id = self.conversation_metadata.get(conversation_id, {}).get("user_id", "unknown")
                
                # Use the original message_id since we don't get the DB ID back from log_message
                # The embedding will use our generated ID, but for future improvement we should
                # modify the logger to return the database message ID
                await self.vector_store.store_conversation_memory(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    content=content,
                    user_id=conversation_user_id,
                    content_type=content_type_memory,  # Use the correct type for memory storage
                    metadata={
                        "sender_type": sender_type,
                        "sender_id": sender_id,
                        "message_type": message_type,
                        **(metadata or {})
                    }
                )
            
            logger.debug(f"Added message {message_id} to conversation {conversation_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            return message_id
    
    async def get_conversation_context(self, conversation_id: str,
                                     include_memory: bool = True,
                                     memory_query: Optional[str] = None,
                                     max_memory_results: int = 3) -> Dict[str, Any]:
        """
        Get complete conversation context including recent messages and relevant memories.
        
        Args:
            conversation_id: Conversation identifier
            include_memory: Whether to include relevant memories
            memory_query: Query for semantic memory search
            max_memory_results: Maximum memory results to include
            
        Returns:
            Complete conversation context
        """
        try:
            # Get recent messages from context window
            recent_messages = list(self.conversation_contexts.get(conversation_id, []))
            
            # Get conversation metadata
            conversation_meta = self.conversation_metadata.get(conversation_id, {})
            user_id = conversation_meta.get("user_id")
            
            context = {
                "conversation_id": conversation_id,
                "conversation_metadata": conversation_meta,
                "recent_messages": recent_messages,
                "message_count": len(recent_messages),
                "relevant_memories": []
            }
            
            # Add relevant memories if requested
            if include_memory and user_id and self.vector_store.is_available():
                if memory_query:
                    # Use provided query for memory search
                    search_query = memory_query
                elif recent_messages:
                    # Use latest message as query
                    latest_message = recent_messages[-1]
                    search_query = latest_message.get("content", "")
                else:
                    search_query = ""
                
                if search_query:
                    memories = await self.vector_store.search_similar_memories(
                        query=search_query,
                        user_id=user_id,
                        limit=max_memory_results
                    )
                    
                    # Filter out memories from current conversation to avoid redundancy
                    relevant_memories = [
                        memory for memory in memories
                        if memory.get("conversation_id") != conversation_id
                    ]
                    
                    context["relevant_memories"] = relevant_memories
            
            logger.debug(f"Retrieved context for conversation {conversation_id}: "
                        f"{len(recent_messages)} messages, {len(context['relevant_memories'])} memories")
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return {
                "conversation_id": conversation_id,
                "recent_messages": [],
                "relevant_memories": [],
                "error": str(e)
            }
    
    async def generate_context_summary(self, conversation_id: str,
                                     summary_type: str = "current_state") -> Optional[str]:
        """
        Generate intelligent summary of conversation state.
        
        Args:
            conversation_id: Conversation identifier
            summary_type: Type of summary (current_state, key_points, action_items)
            
        Returns:
            Generated summary text
        """
        try:
            context = await self.get_conversation_context(conversation_id, include_memory=False)
            recent_messages = context.get("recent_messages", [])
            
            if not recent_messages:
                return "No conversation history available."
            
            if summary_type == "current_state":
                return await self._generate_current_state_summary(recent_messages)
            elif summary_type == "key_points":
                return await self._generate_key_points_summary(recent_messages)
            elif summary_type == "action_items":
                return await self._generate_action_items_summary(recent_messages)
            else:
                return await self._generate_current_state_summary(recent_messages)
                
        except Exception as e:
            logger.error(f"Failed to generate context summary: {e}")
            return None
    
    async def get_user_memory_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get insights about user's conversation patterns and preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            User memory insights and patterns
        """
        try:
            # Check cache first
            cache_key = f"insights_{user_id}"
            if cache_key in self.memory_summaries:
                cached_data = self.memory_summaries[cache_key]
                cache_time = datetime.fromisoformat(cached_data["generated_at"])
                if datetime.utcnow() - cache_time < self.summary_cache_ttl:
                    return cached_data["insights"]
            
            # Generate fresh insights
            insights = {}
            
            # Get memory summary from vector store
            if self.vector_store.is_available():
                memory_summary = await self.vector_store.get_user_memory_summary(user_id)
                insights["memory_summary"] = memory_summary
            
            # Get conversation patterns from database
            conversation_patterns = await self._analyze_conversation_patterns(user_id)
            insights["conversation_patterns"] = conversation_patterns
            
            # Get agent interaction preferences
            agent_preferences = await self._analyze_agent_preferences(user_id)
            insights["agent_preferences"] = agent_preferences
            
            # Cache the results
            self.memory_summaries[cache_key] = {
                "insights": insights,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get user memory insights: {e}")
            return {}
    
    async def escalate_conversation(self, conversation_id: str, escalation_reason: str,
                                  target_agent: Optional[str] = None) -> bool:
        """
        Escalate conversation and update context appropriately.
        
        Args:
            conversation_id: Conversation identifier
            escalation_reason: Reason for escalation
            target_agent: Optional target agent for escalation
            
        Returns:
            True if escalation was successful
        """
        try:
            # Update conversation metadata
            if conversation_id in self.conversation_metadata:
                conv_meta = self.conversation_metadata[conversation_id]
                conv_meta["escalation_count"] += 1
                conv_meta["last_escalation"] = datetime.utcnow().isoformat()
                conv_meta["escalation_reason"] = escalation_reason
                if target_agent:
                    conv_meta["escalated_to"] = target_agent
            
            # Add escalation message to context
            await self.add_message(
                conversation_id=conversation_id,
                content=f"Conversation escalated: {escalation_reason}",
                sender_type="system",
                sender_id="escalation_manager",
                message_type="escalation",
                metadata={
                    "escalation_reason": escalation_reason,
                    "target_agent": target_agent,
                    "is_escalation": True
                }
            )
            
            logger.info(f"Escalated conversation {conversation_id}: {escalation_reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to escalate conversation: {e}")
            return False
    
    async def _generate_current_state_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate current state summary from messages."""
        if not messages:
            return "No conversation activity."
        
        # Simple rule-based summary generation
        user_messages = [msg for msg in messages if msg.get("sender_type") == "user"]
        agent_messages = [msg for msg in messages if msg.get("sender_type") == "agent"]
        
        latest_user_msg = user_messages[-1] if user_messages else None
        latest_agent_msg = agent_messages[-1] if agent_messages else None
        
        summary_parts = []
        
        if latest_user_msg:
            content = latest_user_msg.get("content", "")[:100]
            summary_parts.append(f"User's latest query: {content}")
        
        if latest_agent_msg:
            content = latest_agent_msg.get("content", "")[:100]
            summary_parts.append(f"Agent's latest response: {content}")
        
        summary_parts.append(f"Total messages exchanged: {len(messages)}")
        
        return " | ".join(summary_parts)
    
    async def _generate_key_points_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate key points summary from messages."""
        # Extract questions and important statements
        key_points = []
        
        for message in messages:
            content = message.get("content", "")
            if "?" in content:  # Questions
                key_points.append(f"Q: {content[:80]}...")
            elif any(word in content.lower() for word in ["problem", "issue", "error", "help"]):
                key_points.append(f"Issue: {content[:80]}...")
        
        return "\n".join(key_points[:5]) if key_points else "No key points identified."
    
    async def _generate_action_items_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate action items summary from messages."""
        # Look for action-oriented language
        action_items = []
        
        action_keywords = ["need to", "should", "will", "must", "have to", "please"]
        
        for message in messages:
            content = message.get("content", "")
            if any(keyword in content.lower() for keyword in action_keywords):
                action_items.append(f"Action: {content[:80]}...")
        
        return "\n".join(action_items[:3]) if action_items else "No action items identified."
    
    async def _analyze_conversation_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's conversation patterns."""
        # This would typically query the database for historical patterns
        # For now, return placeholder data
        return {
            "avg_conversation_length": 8.5,
            "common_conversation_types": ["technical", "support"],
            "peak_activity_hours": [9, 14, 16],
            "preferred_response_style": "detailed"
        }
    
    async def _analyze_agent_preferences(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's agent interaction preferences."""
        # This would analyze which agents user interacts with most
        return {
            "most_used_agents": ["general", "technical"],
            "satisfaction_ratings": {"general": 4.2, "technical": 4.5},
            "escalation_triggers": ["complex_technical", "urgent_issues"]
        }
    
    def get_active_conversation_count(self) -> int:
        """Get number of active conversations."""
        return len(self.conversation_contexts)
    
    def cleanup_expired_conversations(self, max_age_hours: int = 24):
        """Clean up expired conversation contexts from memory."""
        current_time = datetime.utcnow()
        expired_conversations = []
        
        for conv_id, metadata in self.conversation_metadata.items():
            last_activity = datetime.fromisoformat(metadata.get("last_activity", current_time.isoformat()))
            age_hours = (current_time - last_activity).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                expired_conversations.append(conv_id)
        
        # Remove expired conversations
        for conv_id in expired_conversations:
            if conv_id in self.conversation_contexts:
                del self.conversation_contexts[conv_id]
            if conv_id in self.conversation_metadata:
                del self.conversation_metadata[conv_id]
        
        if expired_conversations:
            logger.info(f"Cleaned up {len(expired_conversations)} expired conversations")
    
    async def close(self):
        """Clean up conversation memory manager resources."""
        # Clear caches
        self.conversation_contexts.clear()
        self.conversation_metadata.clear()
        self.memory_summaries.clear()
        
        # Close vector store
        if self.vector_store:
            await self.vector_store.close()
        
        logger.info("Conversation Memory Manager closed") 