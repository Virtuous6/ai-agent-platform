"""
Filename: universal_agent.py
Purpose: Configuration-driven universal agent with full platform integration
Dependencies: langchain, openai, asyncio, logging, typing, platform integrations

This module is part of the AI Agent Platform self-improvement system.
Creates specialist agents dynamically based on configuration with full platform integration.
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

class SpecialtyType(Enum):
    """Types of specialist configurations supported."""
    TECHNICAL = "technical"
    RESEARCH = "research"
    BUSINESS = "business"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    CUSTOM = "custom"

class ToolCapability(BaseModel):
    """Represents a tool capability for the universal agent."""
    name: str
    description: str
    function: Optional[Any] = None  # Callable function
    enabled: bool = True

class UniversalAgent:
    """
    Configuration-driven universal agent with full platform integration.
    
    Features:
    - Configuration-driven specialist creation
    - Supabase logging for all interactions
    - Event-driven communication
    - Vector memory integration
    - Workflow tracking
    - Tools registry integration
    - Self-improvement capabilities
    """
    
    def __init__(self, 
                 specialty: str,
                 system_prompt: str,
                 temperature: float = 0.4,
                 tools: Optional[List[ToolCapability]] = None,
                 model_name: str = "gpt-3.5-turbo-0125",
                 max_tokens: int = 500,
                 agent_id: Optional[str] = None,
                 # Platform integrations
                 supabase_logger: Optional[SupabaseLogger] = None,
                 vector_store: Optional[VectorMemoryStore] = None,
                 event_bus: Optional[EventBus] = None,
                 workflow_tracker: Optional[WorkflowTracker] = None):
        """
        Initialize the Universal Agent with platform integrations.
        
        Args:
            specialty: The specialty area (e.g., "Python Optimization", "Data Analysis")
            system_prompt: Custom system prompt for this specialist
            temperature: LLM temperature for response creativity
            tools: List of tool capabilities available to this agent
            model_name: OpenAI model to use
            max_tokens: Maximum tokens for responses
            agent_id: Unique identifier for this agent instance
            
            # Platform integrations
            supabase_logger: Database logging integration
            vector_store: Memory/context integration
            event_bus: Event-driven communication
            workflow_tracker: Workflow analysis integration
        """
        
        # Core configuration
        self.specialty = specialty
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.agent_id = agent_id or f"universal_{specialty.lower().replace(' ', '_')}"
        self.tools = tools or []
        
        # Initialize LLMs
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=max_tokens,
        )
        
        self.analysis_llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=200,
        )
        
        # Platform integrations
        self.supabase_logger = supabase_logger or SupabaseLogger()
        self.vector_store = vector_store or VectorMemoryStore()
        self.event_bus = event_bus
        self.workflow_tracker = workflow_tracker
        self.tool_registry = {}  # For tools registry integration
        
        # Initialize prompt templates
        self.main_prompt = self._create_main_prompt()
        self.improvement_prompt = self._create_improvement_prompt()
        
        # Performance tracking
        self.conversation_history = []
        self.performance_metrics = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
            "user_feedback_score": 0.0,
            "improvement_suggestions": []
        }
        
        # Register event handlers if event bus available
        if self.event_bus:
            asyncio.create_task(self._register_event_handlers())
        
        logger.info(f"Universal Agent '{self.specialty}' initialized with full platform integration")
    
    def _create_main_prompt(self) -> ChatPromptTemplate:
        """Create the main conversation prompt template based on specialty."""
        
        # Create dynamic system template based on configuration
        tools_description = ""
        if self.tools:
            tool_list = [f"- {tool.name}: {tool.description}" for tool in self.tools if tool.enabled]
            tools_description = f"\n\n**Available Tools:**\n" + "\n".join(tool_list)
        
        system_template = f"""{self.system_prompt}

**Your Specialty:** {self.specialty}

**Your Role in the AI Agent Platform:**
- You are a specialized expert integrated into the self-improving platform
- Provide expert-level assistance in your specialty area
- Maintain professional yet approachable communication
- Share best practices and industry standards
- Suggest optimizations and improvements
- Learn from every interaction to continuously improve
- Collaborate with other agents through the platform

**Platform Integration:**
- All interactions are logged for learning and improvement
- Your responses contribute to platform knowledge
- You can access relevant context from previous conversations
- Your insights help train other agents and improve workflows

**Communication Guidelines:**
1. Be precise and technically accurate in your specialty
2. Explain complex concepts clearly for the user's level
3. Provide actionable recommendations
4. Use examples and practical applications when helpful
5. Be proactive in suggesting improvements or alternatives
6. Keep responses concise but comprehensive
7. Learn from each interaction to improve future responses{tools_description}

**Quality Standards:**
- Always prioritize accuracy and helpfulness
- If uncertain about something outside your expertise, be honest
- Suggest collaboration with other agents when beneficial
- Focus on providing value that leverages your specialized knowledge
- Contribute to the platform's continuous learning

Current conversation context: {{context}}
Recent conversation history: {{history}}
Relevant context from memory: {{memory_context}}"""

        human_template = f"""User message: {{message}}

As the {self.specialty} specialist, provide expert assistance. Focus on delivering value through your specialized knowledge while maintaining clear communication."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_improvement_prompt(self) -> ChatPromptTemplate:
        """Create prompt for self-improvement analysis."""
        
        system_template = f"""You are a self-improvement analyzer for the {self.specialty} specialist agent. 

Your task is to analyze the agent's performance and suggest improvements.

**Analysis Framework:**
1. **Response Quality**: Was the specialist knowledge effectively applied?
2. **User Satisfaction**: Did the response meet the user's needs?
3. **Efficiency**: Could the response be more concise or clear?
4. **Completeness**: Was anything important missed?
5. **Proactivity**: Were valuable suggestions or improvements offered?

**Improvement Areas:**
- Prompt optimization for better responses
- Knowledge gaps to address
- Communication style adjustments
- Tool utilization improvements
- Collaboration opportunities

Return analysis in JSON format:
{{
    "quality_score": float (0.0-1.0),
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "knowledge_gaps": ["gap1", "gap2"],
    "strengths": ["strength1", "strength2"],
    "next_optimization": "specific improvement to implement"
}}"""

        human_template = f"""Analyze this interaction:

**User Request:** {{user_message}}
**Agent Response:** {{agent_response}}
**User Feedback:** {{feedback}}
**Performance Metrics:** {{metrics}}

Provide improvement analysis for the {self.specialty} specialist:"""

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
            logger.info(f"Universal Agent '{self.specialty}' processing: '{message[:50]}...'")
            
            # Start workflow tracking
            if self.workflow_tracker:
                await self.workflow_tracker.start_workflow(run_id, {
                    "agent_id": self.agent_id,
                    "specialty": self.specialty,
                    "message_preview": message[:100]
                })
            
            # Retrieve relevant memory context
            memory_context = await self._get_memory_context(message, context)
            
            # Prepare conversation history context
            history_context = self._format_conversation_history(context.get("conversation_history", []))
            
            # Execute tools if available
            tool_results = await self._execute_tools(message, context)
            
            # Generate specialist response
            try:
                with get_openai_callback() as cb:
                    response = await self._generate_specialist_response(
                        message, context, history_context, memory_context, tool_results
                    )
                tokens_used = cb.total_tokens
                input_tokens = cb.prompt_tokens
                output_tokens = cb.completion_tokens
                cost = cb.total_cost
            except Exception as e:
                logger.warning(f"OpenAI callback failed, proceeding without tracking: {e}")
                response = await self._generate_specialist_response(
                    message, context, history_context, memory_context, tool_results
                )
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
            self._update_performance_metrics(tokens_used, cost, processing_time_ms, True)
            
            # Complete workflow tracking
            if self.workflow_tracker:
                await self.workflow_tracker.complete_workflow(run_id, {
                    "success": True,
                    "response_length": len(response),
                    "tokens_used": tokens_used,
                    "cost": cost
                })
            
            # Log the interaction
            interaction_log = {
                "run_id": run_id,
                "timestamp": start_time.isoformat(),
                "message_preview": message[:100],
                "specialty": self.specialty,
                "user_id": context.get("user_id"),
                "channel_id": context.get("channel_id"),
                "conversation_id": conversation_id,
                "message_id": message_id,
                "tokens_used": tokens_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "processing_time_ms": processing_time_ms,
                "tool_results": tool_results,
                "memory_context_used": bool(memory_context)
            }
            self.conversation_history.append(interaction_log)
            
            return {
                "response": response,
                "agent_id": self.agent_id,
                "specialty": self.specialty,
                "conversation_type": "specialist",
                "confidence": 0.9,
                "tokens_used": tokens_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "processing_cost": cost,
                "processing_time_ms": processing_time_ms,
                "tool_results": tool_results,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "run_id": run_id,
                "memory_context_used": bool(memory_context),
                "events_published": ["agent_task_completed"],
                "metadata": {
                    "model_used": self.llm.model_name,
                    "temperature": self.llm.temperature,
                    "agent_id": self.agent_id,
                    "specialty": self.specialty,
                    "tools_used": [tool.name for tool in self.tools if tool.enabled],
                    "performance_score": self.performance_metrics.get("average_quality_score", 0.0),
                    "platform_integrated": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing message in Universal Agent '{self.specialty}': {str(e)}")
            
            # Update performance metrics for failure
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_performance_metrics(0, 0.0, processing_time_ms, False)
            
            # Log failure event
            if self.event_bus:
                await self.event_bus.publish(
                    EventType.AGENT_ERROR,
                    {"agent_id": self.agent_id, "error": str(e), "specialty": self.specialty},
                    source=self.agent_id
                )
            
            # Complete workflow tracking with failure
            if self.workflow_tracker:
                await self.workflow_tracker.complete_workflow(run_id, {
                    "success": False,
                    "error": str(e)
                })
            
            return {
                "response": f"I apologize, but I'm having trouble processing your {self.specialty} request right now. Please try again or contact support if this continues.",
                "agent_id": self.agent_id,
                "specialty": self.specialty,
                "conversation_type": "error",
                "confidence": 0.0,
                "error": str(e),
                "tokens_used": 0,
                "processing_time_ms": processing_time_ms,
                "run_id": run_id,
                "platform_integrated": True
            }
    
    async def _execute_tools(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute available tools based on the message content."""
        tool_results = {}
        
        for tool in self.tools:
            if not tool.enabled or not tool.function:
                continue
                
            try:
                # Simple tool execution - can be enhanced with more sophisticated matching
                if tool.name.lower() in message.lower():
                    result = await tool.function(message, context) if asyncio.iscoroutinefunction(tool.function) else tool.function(message, context)
                    tool_results[tool.name] = result
                    logger.debug(f"Executed tool '{tool.name}' for Universal Agent '{self.specialty}'")
            except Exception as e:
                logger.warning(f"Error executing tool '{tool.name}': {str(e)}")
                tool_results[tool.name] = {"error": str(e)}
        
        return tool_results
    
    async def _generate_specialist_response(self, message: str, context: Dict[str, Any], 
                                          history_context: str, memory_context: str, 
                                          tool_results: Dict[str, Any]) -> str:
        """Generate the specialist response using the LLM."""
        
        main_chain = self.main_prompt | self.llm
        
        # Prepare context including tool results and memory
        enhanced_context = self._format_context(context)
        if tool_results:
            enhanced_context += f"\nTool Results: {json.dumps(tool_results, indent=2)}"
        
        response = await main_chain.ainvoke({
            "message": message,
            "context": enhanced_context,
            "history": history_context,
            "memory_context": memory_context
        })
        
        return response.content
    
    async def analyze_performance(self, user_feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze agent performance and suggest improvements.
        
        Args:
            user_feedback: Optional user feedback on recent interactions
            
        Returns:
            Analysis results with improvement suggestions
        """
        try:
            if not self.conversation_history:
                return {"message": "No interactions to analyze yet"}
            
            # Get recent interaction for analysis
            recent_interaction = self.conversation_history[-1]
            
            analysis_chain = self.improvement_prompt | self.analysis_llm
            
            analysis_response = await analysis_chain.ainvoke({
                "user_message": recent_interaction.get("message_preview", ""),
                "agent_response": "Recent specialist response",  # Would need to store actual response
                "feedback": user_feedback or "No feedback provided",
                "metrics": json.dumps(self.performance_metrics, indent=2),
                "specialty": self.specialty
            })
            
            # Parse JSON response
            analysis_text = analysis_response.content.strip()
            if analysis_text.startswith("```json"):
                analysis_text = analysis_text.strip("```json").strip("```").strip()
            elif analysis_text.startswith("```"):
                analysis_text = analysis_text.strip("```").strip()
            
            analysis = json.loads(analysis_text)
            
            # Store improvement suggestions
            if "improvement_suggestions" in analysis:
                self.performance_metrics["improvement_suggestions"].extend(analysis["improvement_suggestions"])
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing performance for Universal Agent '{self.specialty}': {str(e)}")
            return {"error": str(e), "message": "Performance analysis failed"}
    
    def _update_performance_metrics(self, tokens: int, cost: float, processing_time: float, success: bool):
        """Update performance tracking metrics."""
        self.performance_metrics["total_interactions"] += 1
        
        if success:
            self.performance_metrics["successful_interactions"] += 1
        
        self.performance_metrics["total_tokens"] += tokens
        self.performance_metrics["total_cost"] += cost
        
        # Update average response time
        current_avg = self.performance_metrics["average_response_time"]
        total_interactions = self.performance_metrics["total_interactions"]
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total_interactions - 1) + processing_time) / total_interactions
        )
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for context."""
        if not history:
            return f"No previous conversation history in {self.specialty}."
        
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
        
        context_parts.append(f"Specialty context: {self.specialty}")
        
        return ", ".join(context_parts) if context_parts else f"Standard {self.specialty} conversation"
    
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
            logger.info(f"Universal Agent '{self.specialty}' subscribed to platform events")
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
                agent_type=self.specialty,
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
                metadata={
                    "agent_specialty": self.specialty,
                    "channel_id": context.get("channel_id")
                }
            )
            
            # Store agent response
            await self.vector_store.store_conversation_memory(
                conversation_id=conversation_id or str(uuid.uuid4()),
                message_id=str(uuid.uuid4()),
                content=response,
                user_id=context.get("user_id", "unknown"),
                content_type="response",
                metadata={
                    "agent_id": self.agent_id,
                    "specialty": self.specialty,
                    "channel_id": context.get("channel_id")
                }
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
                    "specialty": self.specialty,
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
    
    async def _apply_improvement(self, improvement_data: Dict[str, Any]):
        """Apply platform improvement suggestions."""
        try:
            improvement_type = improvement_data.get("type")
            if improvement_type == "prompt_optimization":
                # Update system prompt based on improvement
                if "new_prompt" in improvement_data:
                    self.system_prompt = improvement_data["new_prompt"]
                    self.main_prompt = self._create_main_prompt()
                    logger.info(f"Applied prompt improvement for {self.specialty}")
                    
        except Exception as e:
            logger.error(f"Failed to apply improvement: {e}")
    
    async def _learn_from_pattern(self, pattern_data: Dict[str, Any]):
        """Learn from discovered patterns."""
        try:
            pattern_type = pattern_data.get("pattern_type")
            if pattern_type == "successful_interaction":
                # Store successful interaction patterns
                self.performance_metrics["improvement_suggestions"].append({
                    "type": "pattern_learning",
                    "pattern": pattern_data,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
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
            
            self.performance_metrics["user_feedback_score"] = (
                (current_score * (total_interactions - 1) + feedback_score) / total_interactions
            )
            
            # Store feedback for analysis
            self.performance_metrics["improvement_suggestions"].append({
                "type": "user_feedback",
                "score": feedback_score,
                "text": feedback_text,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to process feedback: {e}")
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about this agent instance."""
        success_rate = 0.0
        if self.performance_metrics["total_interactions"] > 0:
            success_rate = self.performance_metrics["successful_interactions"] / self.performance_metrics["total_interactions"]
        
        return {
            "agent_id": self.agent_id,
            "specialty": self.specialty,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "total_interactions": self.performance_metrics["total_interactions"],
            "success_rate": success_rate,
            "total_tokens_used": self.performance_metrics["total_tokens"],
            "total_cost": self.performance_metrics["total_cost"],
            "average_response_time_ms": self.performance_metrics["average_response_time"],
            "cost_per_interaction": (
                self.performance_metrics["total_cost"] / self.performance_metrics["total_interactions"]
                if self.performance_metrics["total_interactions"] > 0 else 0.0
            ),
            "available_tools": [tool.name for tool in self.tools],
            "enabled_tools": [tool.name for tool in self.tools if tool.enabled],
            "recent_improvement_suggestions": self.performance_metrics["improvement_suggestions"][-5:],
            "platform_integrations": {
                "supabase_logging": bool(self.supabase_logger),
                "vector_memory": bool(self.vector_store),
                "event_communication": bool(self.event_bus),
                "workflow_tracking": bool(self.workflow_tracker),
                "tools_registry": bool(self.tool_registry)
            },
            "created_at": datetime.utcnow().isoformat()
        }
    
    def update_configuration(self, **kwargs):
        """
        Update agent configuration dynamically.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        updatable_params = {
            "temperature": "temperature",
            "max_tokens": "max_tokens", 
            "system_prompt": "system_prompt",
            "tools": "tools"
        }
        
        for param, attr in updatable_params.items():
            if param in kwargs:
                setattr(self, attr, kwargs[param])
                logger.info(f"Updated {param} for Universal Agent '{self.specialty}'")
        
        # Recreate LLM if temperature or max_tokens changed
        if "temperature" in kwargs or "max_tokens" in kwargs:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=self.max_tokens,
            )
            logger.info(f"Recreated LLM for Universal Agent '{self.specialty}' with new parameters")
        
        # Recreate prompts if system_prompt changed
        if "system_prompt" in kwargs:
            self.main_prompt = self._create_main_prompt()
            logger.info(f"Recreated prompts for Universal Agent '{self.specialty}'")
    
    async def close(self):
        """Close the agent and cleanup platform resources."""
        try:
            logger.info(f"Closing Universal Agent '{self.specialty}' connections...")
            
            # Close LLM clients
            if hasattr(self.llm, 'client') and hasattr(self.llm.client, 'close'):
                await self.llm.client.close()
            
            if hasattr(self.analysis_llm, 'client') and hasattr(self.analysis_llm.client, 'close'):
                await self.analysis_llm.client.close()
            
            # Close platform integrations
            if self.vector_store:
                await self.vector_store.close()
            
            if self.supabase_logger:
                await self.supabase_logger.close()
            
            # Unsubscribe from events
            if self.event_bus:
                await self.event_bus.unsubscribe(self.agent_id)
            
            # Close tool resources
            for tool in self.tools:
                if hasattr(tool, 'cleanup') and callable(tool.cleanup):
                    try:
                        await tool.cleanup() if asyncio.iscoroutinefunction(tool.cleanup) else tool.cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up tool '{tool.name}': {e}")
                
            logger.info(f"Universal Agent '{self.specialty}' closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing Universal Agent '{self.specialty}': {e}") 