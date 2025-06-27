"""
Filename: technical_agent.py
Purpose: LLM-powered technical agent with full platform integration
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

class TechnicalDomain(Enum):
    """Technical domains the technical agent handles."""
    PROGRAMMING = "programming"
    DEBUGGING = "debugging"
    INFRASTRUCTURE = "infrastructure"
    DEVOPS = "devops"
    DATABASE = "database"
    API_DEVELOPMENT = "api_development"
    SYSTEM_ADMIN = "system_admin"
    PERFORMANCE = "performance"

class ToolSuggestion(BaseModel):
    """Structured tool suggestion from the Technical Agent."""
    should_use_tool: bool
    recommended_tool: Optional[str] = None
    confidence: float
    reasoning: str

class TechnicalAgent:
    """
    LLM-powered technical agent with full platform integration.
    
    Features:
    - Supabase logging for all interactions
    - Event-driven communication
    - Vector memory integration
    - Workflow tracking
    - Self-improvement capabilities
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo-0125", 
                 temperature: float = 0.3,
                 # Platform integrations
                 supabase_logger: Optional[SupabaseLogger] = None,
                 vector_store: Optional[VectorMemoryStore] = None,
                 event_bus: Optional[EventBus] = None,
                 workflow_tracker: Optional[WorkflowTracker] = None):
        """Initialize the LLM-powered Technical Agent with platform integrations."""
        
        # Core configuration
        self.model_name = model_name
        self.temperature = temperature
        self.agent_id = "technical_agent"
        
        # Initialize LLMs
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,  # Lower temperature for technical precision
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=800,  # More tokens for detailed technical responses
        )
        
        self.tool_llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,  # Very focused for tool recommendations
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
        self.tool_prompt = self._create_tool_prompt()
        
        # Performance tracking
        self.interaction_history = []
        self.performance_metrics = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
            "tools_recommended": 0,
            "user_feedback_score": 0.0
        }
        
        # Register event handlers if event bus available
        if self.event_bus:
            asyncio.create_task(self._register_event_handlers())
        
        logger.info(f"Technical Agent initialized with full platform integration")
    
    def _create_main_prompt(self) -> ChatPromptTemplate:
        """Create the main technical conversation prompt template."""
        
        system_template = """You are the Technical Agent for an AI Agent Platform - a specialized expert in programming, debugging, and technical systems. Your expertise and approach:

**Your Role:**
- Expert technical support for programming and development issues
- Advanced debugging and problem-solving assistance  
- Infrastructure, DevOps, and system administration guidance
- Code review and optimization recommendations

**Your Expertise Areas:**
- **Programming Languages**: Python, JavaScript, Java, C++, Go, Rust, TypeScript, etc.
- **Web Development**: React, Vue, Angular, Node.js, Express, FastAPI, Django
- **Databases**: PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch
- **Infrastructure**: Docker, Kubernetes, AWS, GCP, Azure, Terraform
- **DevOps**: CI/CD, Jenkins, GitHub Actions, monitoring, logging
- **System Administration**: Linux, networking, security, performance tuning

**Your Personality:**
- Methodical and precise in technical explanations
- Patient teacher who breaks down complex concepts
- Solution-oriented problem solver
- Detail-focused but understands big picture
- Uses relevant emojis for clarity (🐛 for bugs, ⚡ for performance, etc.)

**Your Technical Approach:**
1. **Understand the Problem**: Ask clarifying questions if needed
2. **Analyze Root Causes**: Look beyond symptoms to underlying issues
3. **Provide Step-by-Step Solutions**: Clear, actionable instructions
4. **Include Best Practices**: Share relevant coding standards and patterns
5. **Suggest Testing**: Recommend verification and testing approaches
6. **Optimize**: Mention performance or security improvements where relevant

**Code Formatting Guidelines:**
- Use proper markdown code blocks with language specification
- Include comments explaining complex logic
- Show before/after examples when appropriate
- Provide complete, runnable examples when possible

**When to Use Tools:**
- Web search for latest documentation, framework updates, or recent solutions
- Database queries for system diagnostics
- File system operations for configuration checks

Current conversation context: {context}
Recent conversation history: {history}
Relevant context from memory: {memory_context}
User's technical level: {user_level}"""

        human_template = """Technical request: {message}

Please provide expert technical assistance. Be thorough, precise, and include practical solutions with code examples where appropriate."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_tool_prompt(self) -> ChatPromptTemplate:
        """Create the tool assessment prompt template."""
        
        system_template = """You are a tool recommendation system for the Technical Agent. Analyze technical requests to determine if external tools would be helpful.

**Available Tools:**
1. **Web Search** - For finding current documentation, recent solutions, latest framework versions
2. **Database Query** - For system diagnostics, log analysis, performance metrics
3. **File System** - For configuration checks, log file analysis, system status

**Your Task:**
Analyze the technical request and determine:
1. Would external tools significantly improve the response? (yes/no)
2. If yes, which tool would be most helpful?
3. Confidence level (0.0-1.0)
4. Brief reasoning

**Guidelines:**
- Recommend tools only when they add significant value
- Web search for: latest documentation, recent framework updates, current best practices
- Database query for: system diagnostics, performance analysis, error log investigation
- File system for: configuration validation, log analysis, system health checks

Return your analysis in this exact JSON format:
{{
    "should_use_tool": boolean,
    "recommended_tool": "web_search|database_query|file_system|null",
    "confidence": float,
    "reasoning": "brief explanation"
}}"""

        human_template = """Technical request: "{message}"

Context: {context}

Analyze if tools would enhance the technical response:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a technical request with full platform integration.
        
        Args:
            message: Technical request content
            context: Conversation and technical context
            
        Returns:
            Dictionary containing technical response and metadata
        """
        start_time = datetime.utcnow()
        run_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Technical Agent processing: '{message[:50]}...'")
            
            # Start workflow tracking
            if self.workflow_tracker:
                await self.workflow_tracker.start_workflow(run_id, {
                    "agent_id": self.agent_id,
                    "agent_type": "technical",
                    "message_preview": message[:100]
                })
            
            # Retrieve relevant memory context
            memory_context = await self._get_memory_context(message, context)
            
            # Classify technical domain
            domain = self._classify_technical_domain(message)
            
            # Assess user's technical level from context
            user_level = self._assess_user_level(context, message)
            
            # Prepare conversation history context
            history_context = self._format_conversation_history(context.get("conversation_history", []))
            
            # Check if tools would enhance the response
            tool_suggestion = await self._assess_tool_needs(message, context)
            
            # Generate technical response
            try:
                with get_openai_callback() as cb:
                    response = await self._generate_technical_response(
                        message, context, history_context, memory_context, domain, user_level, tool_suggestion
                    )
                tokens_used = cb.total_tokens
                input_tokens = cb.prompt_tokens
                output_tokens = cb.completion_tokens
                cost = cb.total_cost
            except Exception as e:
                logger.warning(f"OpenAI callback failed, proceeding without tracking: {e}")
                response = await self._generate_technical_response(
                    message, context, history_context, memory_context, domain, user_level, tool_suggestion
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
            self._update_performance_metrics(tokens_used, cost, processing_time_ms, True, tool_suggestion)
            
            # Complete workflow tracking
            if self.workflow_tracker:
                await self.workflow_tracker.complete_workflow(run_id, {
                    "success": True,
                    "technical_domain": domain.value,
                    "user_level": user_level,
                    "tool_recommended": bool(tool_suggestion),
                    "tokens_used": tokens_used,
                    "cost": cost
                })
            
            # Log the interaction
            interaction_log = {
                "run_id": run_id,
                "timestamp": start_time.isoformat(),
                "message_preview": message[:100],
                "technical_domain": domain.value,
                "user_level": user_level,
                "tool_suggestion": tool_suggestion.dict() if tool_suggestion else None,
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
            self.interaction_history.append(interaction_log)
            
            return {
                "response": response,
                "agent_id": self.agent_id,
                "technical_domain": domain.value,
                "user_level": user_level,
                "tool_suggestion": tool_suggestion.dict() if tool_suggestion else None,
                "confidence": 0.9,
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
                    "agent_type": "technical",
                    "model_used": self.llm.model_name,
                    "temperature": self.llm.temperature,
                    "specialization": "programming_and_systems",
                    "agent_id": self.agent_id,
                    "platform_integrated": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing technical request in Technical Agent: {str(e)}")
            
            # Update performance metrics for failure
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_performance_metrics(0, 0.0, processing_time_ms, False, None)
            
            # Log failure event
            if self.event_bus:
                await self.event_bus.publish(
                    EventType.AGENT_ERROR,
                    {"agent_id": self.agent_id, "error": str(e), "agent_type": "technical"},
                    source=self.agent_id
                )
            
            # Complete workflow tracking with failure
            if self.workflow_tracker:
                await self.workflow_tracker.complete_workflow(run_id, {
                    "success": False,
                    "error": str(e)
                })
            
            return {
                "response": "I apologize, but I'm experiencing technical difficulties processing your request. Please try rephrasing your technical question, and I'll do my best to help with programming, debugging, or system issues.",
                "agent_id": self.agent_id,
                "technical_domain": "error",
                "confidence": 0.0,
                "error": str(e),
                "tokens_used": 0,
                "processing_time_ms": processing_time_ms,
                "run_id": run_id,
                "platform_integrated": True
            }
    
    def _classify_technical_domain(self, message: str) -> TechnicalDomain:
        """Classify the technical domain of the request."""
        message_lower = message.lower()
        
        # Programming keywords
        programming_keywords = ["code", "function", "variable", "class", "method", "algorithm", "syntax"]
        debugging_keywords = ["debug", "error", "bug", "exception", "crash", "fix", "broken"]
        infrastructure_keywords = ["server", "deployment", "docker", "kubernetes", "cloud", "aws"]
        devops_keywords = ["ci/cd", "pipeline", "jenkins", "github actions", "deployment"]
        database_keywords = ["database", "sql", "query", "postgresql", "mysql", "mongodb"]
        api_keywords = ["api", "endpoint", "rest", "graphql", "web service"]
        sysadmin_keywords = ["linux", "unix", "terminal", "command", "shell", "configuration"]
        performance_keywords = ["performance", "optimize", "slow", "memory", "cpu", "bottleneck"]
        
        # Count matches for each domain
        domain_scores = {
            TechnicalDomain.PROGRAMMING: sum(1 for kw in programming_keywords if kw in message_lower),
            TechnicalDomain.DEBUGGING: sum(1 for kw in debugging_keywords if kw in message_lower),
            TechnicalDomain.INFRASTRUCTURE: sum(1 for kw in infrastructure_keywords if kw in message_lower),
            TechnicalDomain.DEVOPS: sum(1 for kw in devops_keywords if kw in message_lower),
            TechnicalDomain.DATABASE: sum(1 for kw in database_keywords if kw in message_lower),
            TechnicalDomain.API_DEVELOPMENT: sum(1 for kw in api_keywords if kw in message_lower),
            TechnicalDomain.SYSTEM_ADMIN: sum(1 for kw in sysadmin_keywords if kw in message_lower),
            TechnicalDomain.PERFORMANCE: sum(1 for kw in performance_keywords if kw in message_lower),
        }
        
        # Return domain with highest score, default to programming
        best_domain = max(domain_scores.keys(), key=lambda k: domain_scores[k])
        return best_domain if domain_scores[best_domain] > 0 else TechnicalDomain.PROGRAMMING
    
    def _assess_user_level(self, context: Dict[str, Any], message: str) -> str:
        """Assess user's technical level based on context and message complexity."""
        message_lower = message.lower()
        
        # Advanced indicators
        advanced_indicators = ["optimization", "architecture", "scalability", "microservices", 
                             "design patterns", "refactor", "performance tuning"]
        
        # Beginner indicators
        beginner_indicators = ["how to start", "tutorial", "basic", "simple", "beginner", 
                             "getting started", "what is"]
        
        advanced_count = sum(1 for indicator in advanced_indicators if indicator in message_lower)
        beginner_count = sum(1 for indicator in beginner_indicators if indicator in message_lower)
        
        if advanced_count > beginner_count:
            return "advanced"
        elif beginner_count > 0:
            return "beginner"
        else:
            return "intermediate"
    
    async def _assess_tool_needs(self, message: str, context: Dict[str, Any]) -> Optional[ToolSuggestion]:
        """Assess if external tools would enhance the technical response."""
        
        try:
            tool_chain = self.tool_prompt | self.tool_llm
            
            response = await tool_chain.ainvoke({
                "message": message,
                "context": context.get("channel_type", "unknown")
            })
            
            # Extract JSON from response
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text.strip("```json").strip("```").strip()
            elif response_text.startswith("```"):
                response_text = response_text.strip("```").strip()
            
            tool_data = json.loads(response_text)
            
            if tool_data["should_use_tool"]:
                return ToolSuggestion(**tool_data)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error assessing tool needs: {str(e)}")
            return None
    
    async def _generate_technical_response(self, message: str, context: Dict[str, Any], 
                                         history_context: str, memory_context: str, domain: TechnicalDomain,
                                         user_level: str, tool_suggestion: Optional[ToolSuggestion]) -> str:
        """Generate the main technical response using the LLM with memory context."""
        
        main_chain = self.main_prompt | self.llm
        
        response = await main_chain.ainvoke({
            "message": message,
            "context": self._format_context(context),
            "history": history_context,
            "memory_context": memory_context,
            "user_level": user_level
        })
        
        response_text = response.content
        
        # Add domain-specific footer
        domain_emoji = {
            TechnicalDomain.PROGRAMMING: "💻",
            TechnicalDomain.DEBUGGING: "🐛",
            TechnicalDomain.INFRASTRUCTURE: "🏗️",
            TechnicalDomain.DEVOPS: "⚙️",
            TechnicalDomain.DATABASE: "🗄️",
            TechnicalDomain.API_DEVELOPMENT: "🔌",
            TechnicalDomain.SYSTEM_ADMIN: "🖥️",
            TechnicalDomain.PERFORMANCE: "⚡"
        }
        
        emoji = domain_emoji.get(domain, "🔧")
        response_text += f"\n\n{emoji} *Technical Agent - {domain.value.title()} Specialist*"
        
        # Add tool suggestion if available
        if tool_suggestion and tool_suggestion.should_use_tool:
            response_text += f"\n💡 *Consider using {tool_suggestion.recommended_tool} for enhanced analysis*"
        
        return response_text
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for context."""
        if not history:
            return "No previous technical conversation history."
        
        formatted_history = []
        for item in history[-3:]:  # Last 3 messages for context
            role = "User" if item.get("message_type") == "user_message" else "Technical Agent"
            content = item.get("content", "")[:150]  # More context for technical discussions
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
            context_parts.append("Technical discussion thread")
        
        return ", ".join(context_parts) if context_parts else "Technical support session"
    
    # Platform Integration Methods (same as other agents)
    async def _register_event_handlers(self):
        """Register event handlers for platform communication."""
        if not self.event_bus:
            return
        try:
            await self.event_bus.subscribe(self.agent_id, [EventType.IMPROVEMENT_APPLIED.value, EventType.PATTERN_DISCOVERED.value, EventType.FEEDBACK_RECEIVED.value], self._handle_platform_event)
            logger.info("Technical Agent subscribed to platform events")
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
            similar_memories = await self.vector_store.search_similar_memories(message, user_id=user_id, limit=3, similarity_threshold=0.7)
            if similar_memories:
                context_parts = []
                for memory in similar_memories:
                    context_parts.append(f"- {memory.get('content_summary', '')[:100]}")
                return f"Relevant context from previous conversations:\n" + "\n".join(context_parts)
        except Exception as e:
            logger.warning(f"Failed to retrieve memory context: {e}")
        return ""
    
    async def _log_to_supabase(self, conversation_id: str, message: str, response: str, context: Dict[str, Any], tokens_used: int, input_tokens: int, output_tokens: int, cost: float, processing_time_ms: float) -> str:
        """Log interaction to Supabase database."""
        if not self.supabase_logger:
            return ""
        try:
            message_id = await self.supabase_logger.log_message(conversation_id=conversation_id or str(uuid.uuid4()), user_id=context.get("user_id", "unknown"), content=message, message_type="user_message", agent_type="technical", agent_response={"response": response, "agent_id": self.agent_id}, processing_time_ms=processing_time_ms, input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=tokens_used, model_used=self.model_name, estimated_cost=cost)
            return message_id
        except Exception as e:
            logger.error(f"Failed to log to Supabase: {e}")
            return ""
    
    async def _store_memory(self, conversation_id: str, message_id: str, message: str, response: str, context: Dict[str, Any]):
        """Store interaction in vector memory."""
        if not self.vector_store:
            return
        try:
            await self.vector_store.store_conversation_memory(conversation_id=conversation_id or str(uuid.uuid4()), message_id=message_id or str(uuid.uuid4()), content=message, user_id=context.get("user_id", "unknown"), content_type="message", metadata={"agent_type": "technical"})
            await self.vector_store.store_conversation_memory(conversation_id=conversation_id or str(uuid.uuid4()), message_id=str(uuid.uuid4()), content=response, user_id=context.get("user_id", "unknown"), content_type="response", metadata={"agent_id": self.agent_id, "agent_type": "technical"})
        except Exception as e:
            logger.warning(f"Failed to store memory: {e}")
    
    async def _publish_events(self, run_id: str, message: str, response: str, context: Dict[str, Any], processing_time_ms: float):
        """Publish events about the interaction."""
        if not self.event_bus:
            return
        try:
            await self.event_bus.publish(EventType.AGENT_ACTIVATED, {"agent_id": self.agent_id, "agent_type": "technical", "run_id": run_id, "user_id": context.get("user_id"), "success": True, "duration_ms": processing_time_ms, "message_length": len(message), "response_length": len(response)}, source=self.agent_id)
        except Exception as e:
            logger.warning(f"Failed to publish events: {e}")
    
    def _update_performance_metrics(self, tokens: int, cost: float, processing_time: float, success: bool, tool_suggestion: Optional[ToolSuggestion]):
        """Update performance tracking metrics."""
        self.performance_metrics["total_interactions"] += 1
        if success:
            self.performance_metrics["successful_interactions"] += 1
        if tool_suggestion:
            self.performance_metrics["tools_recommended"] += 1
        self.performance_metrics["total_tokens"] += tokens
        self.performance_metrics["total_cost"] += cost
        current_avg = self.performance_metrics["average_response_time"]
        total_interactions = self.performance_metrics["total_interactions"]
        self.performance_metrics["average_response_time"] = ((current_avg * (total_interactions - 1) + processing_time) / total_interactions)
    
    async def _apply_improvement(self, improvement_data: Dict[str, Any]):
        """Apply platform improvement suggestions."""
        try:
            improvement_type = improvement_data.get("type")
            if improvement_type == "prompt_optimization":
                if "new_prompt" in improvement_data:
                    self.main_prompt = self._create_main_prompt()
                    logger.info("Applied prompt improvement for Technical Agent")
        except Exception as e:
            logger.error(f"Failed to apply improvement: {e}")
    
    async def _learn_from_pattern(self, pattern_data: Dict[str, Any]):
        """Learn from discovered patterns."""
        try:
            pattern_type = pattern_data.get("pattern_type")
            if pattern_type == "successful_interaction":
                pass  # Can be enhanced with technical solution improvements
        except Exception as e:
            logger.error(f"Failed to learn from pattern: {e}")
    
    async def _process_feedback(self, feedback_data: Dict[str, Any]):
        """Process user feedback for continuous improvement."""
        try:
            feedback_score = feedback_data.get("score", 0.0)
            total_interactions = self.performance_metrics["total_interactions"]
            current_score = self.performance_metrics["user_feedback_score"]
            if total_interactions > 0:
                self.performance_metrics["user_feedback_score"] = ((current_score * (total_interactions - 1) + feedback_score) / total_interactions)
        except Exception as e:
            logger.error(f"Failed to process feedback: {e}")
    
    def get_technical_stats(self) -> Dict[str, Any]:
        """Get statistics about technical interactions handled."""
        if not self.interaction_history:
            return {"total_interactions": 0}
        
        total_tokens = sum(log.get("tokens_used", 0) for log in self.interaction_history)
        total_cost = sum(log.get("cost", 0) for log in self.interaction_history)
        
        # Domain distribution
        domain_counts = {}
        user_level_counts = {}
        tool_suggestions = sum(1 for log in self.interaction_history if log.get("tool_suggestion"))
        
        for log in self.interaction_history:
            domain = log.get("technical_domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            level = log.get("user_level", "unknown")
            user_level_counts[level] = user_level_counts.get(level, 0) + 1
        
        return {
            "total_interactions": len(self.interaction_history),
            "total_tokens_used": total_tokens,
            "total_cost": total_cost,
            "domain_distribution": domain_counts,
            "user_level_distribution": user_level_counts,
            "tool_suggestions": tool_suggestions,
            "tool_suggestion_rate": tool_suggestions / len(self.interaction_history) if self.interaction_history else 0
        }
    
    async def close(self):
        """Close the agent and cleanup platform resources."""
        try:
            logger.info("Closing Technical Agent connections...")
            
            # Close LLM clients
            if hasattr(self.llm, 'client') and hasattr(self.llm.client, 'close'):
                await self.llm.client.close()
            
            if hasattr(self.tool_llm, 'client') and hasattr(self.tool_llm.client, 'close'):
                await self.tool_llm.client.close()
            
            # Close platform integrations
            if self.vector_store:
                await self.vector_store.close()
            
            if self.supabase_logger:
                await self.supabase_logger.close()
            
            # Unsubscribe from events
            if self.event_bus:
                await self.event_bus.unsubscribe(self.agent_id)
                
            logger.info("Technical Agent closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing Technical Agent: {e}") 