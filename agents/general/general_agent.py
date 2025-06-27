"""
Filename: general_agent.py
Purpose: LLM-powered general conversation agent with full platform integration
Dependencies: langchain, openai, asyncio, logging, typing, platform integrations

This module is part of the AI Agent Platform.
See README.llm.md in this directory for context.
"""

import asyncio
import logging
import os
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel

# Platform integrations
from database.supabase_logger import SupabaseLogger
from memory.vector_store import VectorMemoryStore
from events.event_bus import EventBus, EventType
from orchestrator.workflow_tracker import WorkflowTracker

logger = logging.getLogger(__name__)

class ConversationType(Enum):
    """Types of conversations the general agent handles."""
    GREETING = "greeting"
    QUESTION = "question"
    GRATITUDE = "gratitude"
    SOCIAL = "social"
    HELP_REQUEST = "help_request"
    ESCALATION_NEEDED = "escalation_needed"
    GENERAL = "general"

class EscalationSuggestion(BaseModel):
    """Structured escalation suggestion from the LLM."""
    should_escalate: bool
    recommended_agent: Optional[str] = None
    confidence: float
    reasoning: str

class GeneralAgent:
    """
    LLM-powered general conversation agent with full platform integration.
    
    Features:
    - Supabase logging for all interactions
    - Event-driven communication
    - Vector memory integration
    - Workflow tracking
    - Self-improvement capabilities
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo-0125", 
                 temperature: float = 0.7,
                 # Platform integrations
                 supabase_logger: Optional[SupabaseLogger] = None,
                 vector_store: Optional[VectorMemoryStore] = None,
                 event_bus: Optional[EventBus] = None,
                 workflow_tracker: Optional[WorkflowTracker] = None):
        """Initialize the LLM-powered General Agent with platform integrations."""
        
        # Core configuration
        self.model_name = model_name
        self.temperature = temperature
        self.agent_id = "general_agent"
        
        # Initialize LLMs
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=500,
        )
        
        self.escalation_llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=200,
        )
        
        # Platform integrations
        self.supabase_logger = supabase_logger or SupabaseLogger()
        self.vector_store = vector_store or VectorMemoryStore()
        self.event_bus = event_bus
        self.workflow_tracker = workflow_tracker
        self.tool_registry = {}
        
        # Initialize prompt templates
        self.main_prompt = self._create_main_prompt()
        self.escalation_prompt = self._create_escalation_prompt()
        
        # Performance tracking
        self.conversation_history = []
        self.performance_metrics = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
            "escalations_suggested": 0,
            "user_feedback_score": 0.0
        }
        
        # Register event handlers if event bus available
        if self.event_bus:
            asyncio.create_task(self._register_event_handlers())
        
        logger.info(f"General Agent initialized with full platform integration")
    
    def _create_main_prompt(self) -> ChatPromptTemplate:
        """Create the main conversation prompt template."""
        
        system_template = """You are the General Agent for an AI Agent Platform - a friendly, helpful assistant integrated into the self-improving platform. Your personality and behavior:

**Your Role:**
- Primary interface for general conversations, questions, and everyday assistance
- Bridge between users and specialized agents when needed
- Maintain warm, professional tone while being genuinely helpful
- Learn from every interaction to continuously improve
- Contribute to the platform's knowledge and capabilities

**Platform Integration:**
- All interactions are logged for learning and improvement
- Your responses contribute to platform knowledge
- You can access relevant context from previous conversations
- Your insights help train other agents and improve workflows

**Your Personality:**
- Warm, friendly, and approachable 😊
- Professional but not stiff - like a helpful colleague
- Enthusiastic about helping users succeed
- Clear communicator who explains things simply
- Patient and understanding
- Continuously learning and improving

**Your Capabilities:**
- Answer general questions using your knowledge
- Provide explanations, definitions, and guidance
- Help with basic problem-solving and decision-making
- Engage in natural conversation
- Recognize when specialized help is needed
- Access relevant context from previous conversations

**Important Guidelines:**
1. Be conversational and natural while leveraging platform capabilities
2. Use emojis appropriately to convey warmth (but don't overdo it)
3. If you're uncertain about something, be honest about limitations
4. For technical coding issues, research questions, or specialized tasks, suggest escalation
5. Keep responses concise but complete and contextually aware
6. Reference previous conversation context when relevant
7. Learn from each interaction to improve future responses

**When to Suggest Escalation:**
- Technical/programming questions → Technical Agent
- Research, analysis, or data gathering → Research Agent
- Complex specialized topics outside general knowledge

Current conversation context: {context}
Recent conversation history: {history}
Relevant context from memory: {memory_context}"""

        human_template = """User message: {message}

Please respond as the General Agent, being helpful, warm, and professional. If this requires specialized assistance, you can mention it, but still provide what help you can."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_escalation_prompt(self) -> ChatPromptTemplate:
        """Create the escalation assessment prompt template."""
        
        system_template = """You are an escalation classifier for an AI Agent Platform. Analyze user messages to determine if they need specialized agent assistance.

**Available Specialized Agents:**
1. **Technical Agent** - Handles: programming, debugging, technical issues, system administration, DevOps, infrastructure
2. **Research Agent** - Handles: research tasks, data analysis, market research, competitive intelligence, information gathering

**Your Task:**
Analyze the user message and determine:
1. Should this be escalated to a specialist? (yes/no)
2. If yes, which agent? (technical or research)
3. Confidence level (0.0-1.0)
4. Brief reasoning

**Guidelines:**
- Only escalate if the user specifically needs specialized expertise
- General questions about topics can often be handled by General Agent
- Consider the user's intent and specificity
- Be conservative - don't over-escalate simple questions

Return your analysis in this exact JSON format:
{{
    "should_escalate": boolean,
    "recommended_agent": "technical|research|null",
    "confidence": float,
    "reasoning": "brief explanation"
}}"""

        human_template = """User message: "{message}"

Context: {context}

Analyze this message for escalation needs:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user message with full platform integration.
        
        Args:
            message: User message content
            context: Conversation context
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = datetime.utcnow()
        run_id = str(uuid.uuid4())
        
        try:
            logger.info(f"General Agent processing: '{message[:50]}...'")
            
            # Start workflow tracking
            if self.workflow_tracker:
                await self.workflow_tracker.start_workflow(run_id, {
                    "agent_id": self.agent_id,
                    "agent_type": "general",
                    "message_preview": message[:100]
                })
            
            # Retrieve relevant memory context
            memory_context = await self._get_memory_context(message, context)
            
            # Prepare conversation history context
            history_context = self._format_conversation_history(context.get("conversation_history", []))
            
            # Check if escalation is needed using focused LLM call
            escalation_suggestion = await self._assess_escalation_needs(message, context)
            
            # Generate main response
            try:
                with get_openai_callback() as cb:
                    response = await self._generate_response(message, context, history_context, memory_context, escalation_suggestion)
                tokens_used = cb.total_tokens
                input_tokens = cb.prompt_tokens
                output_tokens = cb.completion_tokens
                cost = cb.total_cost
            except Exception as e:
                logger.warning(f"OpenAI callback failed, proceeding without tracking: {e}")
                response = await self._generate_response(message, context, history_context, memory_context, escalation_suggestion)
                tokens_used = 0
                input_tokens = 0
                output_tokens = 0
                cost = 0.0
            
            # Calculate processing time
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Log to Supabase
            conversation_id = context.get("conversation_id")
            message_id = await self._log_to_supabase(
                conversation_id, message, response, context, 
                tokens_used, input_tokens, output_tokens, cost, processing_time_ms
            )
            
            # Store in memory
            await self._store_memory(conversation_id, message_id, message, response, context)
            
            # Publish events
            await self._publish_events(run_id, message, response, context, processing_time_ms)
            
            # Update performance metrics
            self._update_performance_metrics(tokens_used, cost, processing_time_ms, True, escalation_suggestion)
            
            # Complete workflow tracking
            if self.workflow_tracker:
                await self.workflow_tracker.complete_workflow(run_id, {
                    "success": True,
                    "response_length": len(response),
                    "tokens_used": tokens_used,
                    "cost": cost,
                    "escalation_suggested": bool(escalation_suggestion)
                })
            
            # Log the interaction
            interaction_log = {
                "run_id": run_id,
                "timestamp": start_time.isoformat(),
                "message_preview": message[:100],
                "escalation_suggestion": escalation_suggestion.dict() if escalation_suggestion else None,
                "user_id": context.get("user_id"),
                "channel_id": context.get("channel_id"),
                "conversation_id": conversation_id,
                "message_id": message_id,
                "tokens_used": tokens_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "processing_time_ms": processing_time_ms,
                "memory_context_used": bool(memory_context)
            }
            self.conversation_history.append(interaction_log)
            
            return {
                "response": response,
                "agent_id": self.agent_id,
                "escalation_suggestion": escalation_suggestion.dict() if escalation_suggestion else None,
                "conversation_type": "general",
                "confidence": 0.8,
                "tokens_used": tokens_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "processing_cost": cost,
                "processing_time_ms": processing_time_ms,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "run_id": run_id,
                "memory_context_used": bool(memory_context),
                "events_published": ["agent_task_completed"],
                "metadata": {
                    "model_used": self.llm.model_name,
                    "temperature": self.llm.temperature,
                    "agent_id": self.agent_id,
                    "platform_integrated": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing message in General Agent: {str(e)}")
            
            # Update performance metrics for failure
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_performance_metrics(0, 0.0, processing_time_ms, False, None)
            
            # Log failure event
            if self.event_bus:
                await self.event_bus.publish(
                    EventType.AGENT_ERROR,
                    {"agent_id": self.agent_id, "error": str(e), "agent_type": "general"},
                    source=self.agent_id
                )
            
            # Complete workflow tracking with failure
            if self.workflow_tracker:
                await self.workflow_tracker.complete_workflow(run_id, {
                    "success": False,
                    "error": str(e)
                })
            
            return {
                "response": "I apologize, but I'm having trouble processing your message right now. Please try again or contact support if this continues.",
                "agent_id": self.agent_id,
                "escalation_suggestion": None,
                "conversation_type": "error",
                "confidence": 0.0,
                "error": str(e),
                "tokens_used": 0,
                "processing_time_ms": processing_time_ms,
                "run_id": run_id,
                "platform_integrated": True
            }
    
    async def _assess_escalation_needs(self, message: str, context: Dict[str, Any]) -> Optional[EscalationSuggestion]:
        """Assess if the message needs escalation using LLM analysis."""
        
        try:
            escalation_chain = self.escalation_prompt | self.escalation_llm
            
            response = await escalation_chain.ainvoke({
                "message": message,
                "context": context.get("channel_type", "unknown")
            })
            
            # Extract JSON from response (handle potential markdown formatting)
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text.strip("```json").strip("```").strip()
            elif response_text.startswith("```"):
                response_text = response_text.strip("```").strip()
            
            escalation_data = json.loads(response_text)
            
            if escalation_data["should_escalate"]:
                return EscalationSuggestion(**escalation_data)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error assessing escalation needs: {str(e)}")
            # Fall back to simple keyword detection
            return self._fallback_escalation_check(message)
    
    def _fallback_escalation_check(self, message: str) -> Optional[EscalationSuggestion]:
        """Fallback escalation check using simple keyword matching."""
        
        message_lower = message.lower()
        
        technical_keywords = ["code", "programming", "debug", "error", "bug", "api", "technical", "server"]
        research_keywords = ["research", "analyze", "data", "study", "market", "competitor"]
        
        technical_matches = sum(1 for keyword in technical_keywords if keyword in message_lower)
        research_matches = sum(1 for keyword in research_keywords if keyword in message_lower)
        
        if technical_matches >= 2:
            return EscalationSuggestion(
                should_escalate=True,
                recommended_agent="technical",
                confidence=0.6,
                reasoning="Multiple technical keywords detected"
            )
        elif research_matches >= 2:
            return EscalationSuggestion(
                should_escalate=True,
                recommended_agent="research",
                confidence=0.6,
                reasoning="Multiple research keywords detected"
            )
        
        return None
    
    async def _generate_response(self, message: str, context: Dict[str, Any], 
                               history_context: str, memory_context: str, escalation_suggestion: Optional[EscalationSuggestion]) -> str:
        """Generate the main response using the LLM with memory context."""
        
        main_chain = self.main_prompt | self.llm
        
        response = await main_chain.ainvoke({
            "message": message,
            "context": self._format_context(context),
            "history": history_context,
            "memory_context": memory_context
        })
        
        response_text = response.content
        
        # Add escalation note if needed
        if escalation_suggestion and escalation_suggestion.should_escalate:
            agent_name = "Technical Agent" if escalation_suggestion.recommended_agent == "technical" else "Research Agent"
            response_text += f"\n\n💡 *For more specialized help with this, you might want to mention the {agent_name} specifically.*"
        
        return response_text
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for context."""
        if not history:
            return "No previous conversation history."
        
        formatted_history = []
        for item in history[-3:]:  # Last 3 messages for context
            role = "User" if item.get("message_type") == "user_message" else "Assistant"
            content = item.get("content", "")[:100]  # Truncate for context
            formatted_history.append(f"{role}: {content}")
        
        return "\n".join(formatted_history)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information for the prompt."""
        context_parts = []
        
        if context.get("channel_type"):
            context_parts.append(f"Channel type: {context['channel_type']}")
        
        if context.get("user_id"):
            context_parts.append(f"User: {context['user_id']}")
        
        if context.get("is_thread"):
            context_parts.append("In thread conversation")
        
        return ", ".join(context_parts) if context_parts else "Standard conversation"
    
    async def _register_event_handlers(self):
        """Register event handlers for platform communication."""
        if not self.event_bus:
            return
            
        try:
            # Subscribe to relevant events
            await self.event_bus.subscribe(
                self.agent_id,
                [
                    EventType.IMPROVEMENT_APPLIED.value,
                    EventType.PATTERN_DISCOVERED.value,
                    EventType.FEEDBACK_RECEIVED.value
                ],
                self._handle_platform_event
            )
            logger.info("General Agent subscribed to platform events")
        except Exception as e:
            logger.error(f"Failed to register event handlers: {e}")
    
    async def _handle_platform_event(self, event: Dict[str, Any]):
        """Handle incoming platform events."""
        try:
            event_type = event.get('type')
            event_data = event.get('data', {})
            
            if event_type == EventType.IMPROVEMENT_APPLIED.value:
                await self._apply_improvement(event_data)
            elif event_type == EventType.PATTERN_DISCOVERED.value:
                await self._learn_from_pattern(event_data)
            elif event_type == EventType.FEEDBACK_RECEIVED.value:
                await self._process_feedback(event_data)
                
        except Exception as e:
            logger.error(f"Error handling platform event: {e}")
    
    async def _get_memory_context(self, message: str, context: Dict[str, Any]) -> str:
        """Retrieve relevant context from vector memory."""
        if not self.vector_store:
            return ""
        
        try:
            user_id = context.get("user_id")
            similar_memories = await self.vector_store.search_similar_memories(
                message, user_id=user_id, limit=3, similarity_threshold=0.7
            )
            
            if similar_memories:
                context_parts = []
                for memory in similar_memories:
                    context_parts.append(f"- {memory.get('content_summary', '')[:100]}")
                return f"Relevant context from previous conversations:\n" + "\n".join(context_parts)
            
        except Exception as e:
            logger.warning(f"Failed to retrieve memory context: {e}")
        
        return ""
    
    async def _log_to_supabase(self, conversation_id: str, message: str, response: str, 
                             context: Dict[str, Any], tokens_used: int, input_tokens: int,
                             output_tokens: int, cost: float, processing_time_ms: float) -> str:
        """Log interaction to Supabase database."""
        if not self.supabase_logger:
            return ""
        
        try:
            message_id = await self.supabase_logger.log_message(
                conversation_id=conversation_id or str(uuid.uuid4()),
                user_id=context.get("user_id", "unknown"),
                content=message,
                message_type="user_message",
                agent_type="general",
                agent_response={"response": response, "agent_id": self.agent_id},
                processing_time_ms=processing_time_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=tokens_used,
                model_used=self.model_name,
                estimated_cost=cost
            )
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to log to Supabase: {e}")
            return ""
    
    async def _store_memory(self, conversation_id: str, message_id: str, 
                          message: str, response: str, context: Dict[str, Any]):
        """Store interaction in vector memory."""
        if not self.vector_store:
            return
        
        try:
            # Store user message
            await self.vector_store.store_conversation_memory(
                conversation_id=conversation_id or str(uuid.uuid4()),
                message_id=message_id or str(uuid.uuid4()),
                content=message,
                user_id=context.get("user_id", "unknown"),
                content_type="message",
                metadata={"agent_type": "general"}
            )
            
            # Store agent response
            await self.vector_store.store_conversation_memory(
                conversation_id=conversation_id or str(uuid.uuid4()),
                message_id=str(uuid.uuid4()),
                content=response,
                user_id=context.get("user_id", "unknown"),
                content_type="agent_response",
                metadata={"agent_id": self.agent_id, "agent_type": "general"}
            )
            
        except Exception as e:
            logger.warning(f"Failed to store memory: {e}")
    
    async def _publish_events(self, run_id: str, message: str, response: str, 
                            context: Dict[str, Any], processing_time_ms: float):
        """Publish events about the interaction."""
        if not self.event_bus:
            return
        
        try:
            # Publish task completion event
            await self.event_bus.publish(
                EventType.AGENT_ACTIVATED,
                {
                    "agent_id": self.agent_id,
                    "agent_type": "general",
                    "run_id": run_id,
                    "user_id": context.get("user_id"),
                    "success": True,
                    "duration_ms": processing_time_ms,
                    "message_length": len(message),
                    "response_length": len(response)
                },
                source=self.agent_id
            )
            
        except Exception as e:
            logger.warning(f"Failed to publish events: {e}")
    
    def _update_performance_metrics(self, tokens: int, cost: float, processing_time: float, 
                                  success: bool, escalation_suggestion: Optional[EscalationSuggestion]):
        """Update performance tracking metrics."""
        self.performance_metrics["total_interactions"] += 1
        
        if success:
            self.performance_metrics["successful_interactions"] += 1
        
        if escalation_suggestion:
            self.performance_metrics["escalations_suggested"] += 1
        
        self.performance_metrics["total_tokens"] += tokens
        self.performance_metrics["total_cost"] += cost
        
        # Update average response time
        current_avg = self.performance_metrics["average_response_time"]
        total_interactions = self.performance_metrics["total_interactions"]
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total_interactions - 1) + processing_time) / total_interactions
        )
    
    async def _apply_improvement(self, improvement_data: Dict[str, Any]):
        """Apply platform improvement suggestions."""
        try:
            improvement_type = improvement_data.get("type")
            if improvement_type == "prompt_optimization":
                # Update system prompt based on improvement
                if "new_prompt" in improvement_data:
                    # Recreate prompts with improvement
                    self.main_prompt = self._create_main_prompt()
                    logger.info("Applied prompt improvement for General Agent")
                    
        except Exception as e:
            logger.error(f"Failed to apply improvement: {e}")
    
    async def _learn_from_pattern(self, pattern_data: Dict[str, Any]):
        """Learn from discovered patterns."""
        try:
            pattern_type = pattern_data.get("pattern_type")
            if pattern_type == "successful_interaction":
                # Store successful interaction patterns for escalation optimization
                pass  # Can be enhanced with pattern-based escalation improvements
                
        except Exception as e:
            logger.error(f"Failed to learn from pattern: {e}")
    
    async def _process_feedback(self, feedback_data: Dict[str, Any]):
        """Process user feedback for continuous improvement."""
        try:
            feedback_score = feedback_data.get("score", 0.0)
            feedback_text = feedback_data.get("text", "")
            
            # Update feedback metrics
            total_interactions = self.performance_metrics["total_interactions"]
            current_score = self.performance_metrics["user_feedback_score"]
            
            if total_interactions > 0:
                self.performance_metrics["user_feedback_score"] = (
                    (current_score * (total_interactions - 1) + feedback_score) / total_interactions
                )
            
        except Exception as e:
            logger.error(f"Failed to process feedback: {e}")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about conversations handled."""
        if not self.conversation_history:
            return {"total_conversations": 0}
        
        total_tokens = sum(log.get("tokens_used", 0) for log in self.conversation_history)
        total_cost = sum(log.get("cost", 0) for log in self.conversation_history)
        escalations = sum(1 for log in self.conversation_history if log.get("escalation_suggestion"))
        
        return {
            "total_conversations": len(self.conversation_history),
            "total_tokens_used": total_tokens,
            "total_cost": total_cost,
            "escalation_suggestions": escalations,
            "escalation_rate": escalations / len(self.conversation_history) if self.conversation_history else 0
        }
    
    async def close(self):
        """Close the agent and cleanup platform resources."""
        try:
            logger.info("Closing General Agent connections...")
            
            # Close LLM clients
            if hasattr(self.llm, 'client') and hasattr(self.llm.client, 'close'):
                await self.llm.client.close()
            
            if hasattr(self.escalation_llm, 'client') and hasattr(self.escalation_llm.client, 'close'):
                await self.escalation_llm.client.close()
            
            # Close platform integrations
            if self.vector_store:
                await self.vector_store.close()
            
            if self.supabase_logger:
                await self.supabase_logger.close()
            
            # Unsubscribe from events
            if self.event_bus:
                await self.event_bus.unsubscribe(self.agent_id)
                
            logger.info("General Agent closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing General Agent: {e}") 