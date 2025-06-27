"""
Filename: research_agent.py
Purpose: LLM-powered research agent with full platform integration
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

class ResearchType(Enum):
    """Types of research the research agent handles."""
    MARKET_RESEARCH = "market_research"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    DATA_ANALYSIS = "data_analysis"
    ACADEMIC_RESEARCH = "academic_research"
    INDUSTRY_TRENDS = "industry_trends"
    CUSTOMER_INSIGHTS = "customer_insights"
    TECHNOLOGY_ASSESSMENT = "technology_assessment"
    STRATEGIC_PLANNING = "strategic_planning"

class ResearchSuggestion(BaseModel):
    """Structured research methodology suggestion."""
    research_approach: str
    data_sources: List[str]
    key_questions: List[str]
    deliverables: List[str]
    timeline_estimate: str
    confidence: float

class ResearchAgent:
    """
    LLM-powered research agent with full platform integration.
    
    Features:
    - Supabase logging for all interactions
    - Event-driven communication
    - Vector memory integration
    - Workflow tracking
    - Self-improvement capabilities
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo-0125", 
                 temperature: float = 0.4,
                 # Platform integrations
                 supabase_logger: Optional[SupabaseLogger] = None,
                 vector_store: Optional[VectorMemoryStore] = None,
                 event_bus: Optional[EventBus] = None,
                 workflow_tracker: Optional[WorkflowTracker] = None):
        """Initialize the LLM-powered Research Agent with platform integrations."""
        
        # Core configuration
        self.model_name = model_name
        self.temperature = temperature
        self.agent_id = "research_agent"
        
        # Initialize LLMs
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,  # Balanced for analytical creativity
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=800,  # More tokens for comprehensive research responses
        )
        
        self.methodology_llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,  # Lower temperature for structured methodology
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=300,
        )
        
        # Platform integrations
        self.supabase_logger = supabase_logger or SupabaseLogger()
        self.vector_store = vector_store or VectorMemoryStore()
        self.event_bus = event_bus
        self.workflow_tracker = workflow_tracker
        self.tool_registry = {}
        
        # Initialize prompt templates
        self.main_prompt = self._create_main_prompt()
        self.methodology_prompt = self._create_methodology_prompt()
        
        # Performance tracking
        self.research_history = []
        self.performance_metrics = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
            "research_methodologies_provided": 0,
            "user_feedback_score": 0.0
        }
        
        # Register event handlers if event bus available
        if self.event_bus:
            asyncio.create_task(self._register_event_handlers())
        
        logger.info(f"Research Agent initialized with full platform integration")
    
    def _create_main_prompt(self) -> ChatPromptTemplate:
        """Create the main research conversation prompt template."""
        
        system_template = """You are the Research Agent for an AI Agent Platform - a specialized expert in research methodology, data analysis, and strategic insights. Your expertise and approach:

**Your Role:**
- Expert research guidance and methodology design
- Comprehensive analysis and insights synthesis
- Market research and competitive intelligence
- Data-driven decision support and strategic planning

**Your Research Expertise:**
- **Market Research**: Consumer behavior, market sizing, segmentation, surveys
- **Competitive Analysis**: Competitor profiling, positioning, SWOT analysis
- **Data Analysis**: Statistical analysis, trend identification, pattern recognition
- **Academic Research**: Literature reviews, methodology design, hypothesis testing
- **Industry Analysis**: Market trends, disruption analysis, technology assessment
- **Strategic Planning**: Business intelligence, opportunity assessment, risk analysis

**Your Personality:**
- Methodical and thorough in research approach
- Analytical thinker who sees patterns and connections
- Objective and evidence-based in conclusions
- Curious investigator who asks the right questions
- Clear communicator of complex insights 📊

**Your Research Methodology:**
1. **Define Objectives**: Clarify research questions and success criteria
2. **Design Approach**: Select appropriate research methods and data sources
3. **Gather Information**: Systematic data collection and source validation
4. **Analyze Findings**: Statistical analysis, pattern identification, insight synthesis
5. **Present Insights**: Clear, actionable recommendations with supporting evidence
6. **Validate Results**: Cross-reference findings and assess reliability

**Research Deliverables You Provide:**
- Executive summaries with key findings
- Detailed analysis with supporting data
- Actionable recommendations and next steps
- Research methodology and data source transparency
- Confidence assessments and limitations

**When to Recommend External Research:**
- Primary data collection (surveys, interviews)
- Real-time market data and industry reports
- Proprietary databases and specialized sources
- Statistical analysis of large datasets

Current research context: {context}
Previous research history: {history}
Relevant context from memory: {memory_context}
Research complexity level: {complexity}"""

        human_template = """Research request: {message}

Please provide expert research guidance. Include methodology recommendations, key questions to investigate, potential data sources, and expected deliverables."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_methodology_prompt(self) -> ChatPromptTemplate:
        """Create the research methodology assessment prompt template."""
        
        system_template = """You are a research methodology advisor. Analyze research requests to recommend optimal research approaches and frameworks.

**Your Task:**
For each research request, provide structured methodology recommendations including:
1. **Research Approach**: Qualitative, quantitative, or mixed methods
2. **Data Sources**: Primary and secondary sources to investigate
3. **Key Questions**: Critical research questions to address
4. **Deliverables**: Expected outputs and formats
5. **Timeline**: Realistic time estimate for completion

**Research Frameworks:**
- **Exploratory**: For new topics or hypothesis generation
- **Descriptive**: For characterizing markets, trends, or phenomena
- **Explanatory**: For understanding relationships and causation
- **Evaluative**: For assessing effectiveness or performance

Return your recommendations in this exact JSON format:
{{
    "research_approach": "exploratory|descriptive|explanatory|evaluative",
    "data_sources": ["source1", "source2", "source3"],
    "key_questions": ["question1", "question2", "question3"],
    "deliverables": ["deliverable1", "deliverable2"],
    "timeline_estimate": "timeframe description",
    "confidence": float
}}"""

        human_template = """Research request: "{message}"

Context: {context}

Provide structured research methodology recommendations:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a research request with full platform integration.
        
        Args:
            message: Research request content
            context: Conversation and research context
            
        Returns:
            Dictionary containing research response and methodology
        """
        start_time = datetime.utcnow()
        run_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Research Agent processing: '{message[:50]}...'")
            
            # Start workflow tracking
            if self.workflow_tracker:
                await self.workflow_tracker.start_workflow(run_id, {
                    "agent_id": self.agent_id,
                    "agent_type": "research",
                    "message_preview": message[:100]
                })
            
            # Retrieve relevant memory context
            memory_context = await self._get_memory_context(message, context)
            
            # Classify research type
            research_type = self._classify_research_type(message)
            
            # Assess research complexity
            complexity = self._assess_research_complexity(message, context)
            
            # Prepare research history context
            history_context = self._format_research_history(context.get("conversation_history", []))
            
            # Generate research methodology recommendations
            methodology = await self._generate_methodology(message, context)
            
            # Generate research response
            try:
                with get_openai_callback() as cb:
                    response = await self._generate_research_response(
                        message, context, history_context, memory_context, research_type, complexity, methodology
                    )
                tokens_used = cb.total_tokens
                input_tokens = cb.prompt_tokens
                output_tokens = cb.completion_tokens
                cost = cb.total_cost
            except Exception as e:
                logger.warning(f"OpenAI callback failed, proceeding without tracking: {e}")
                response = await self._generate_research_response(
                    message, context, history_context, memory_context, research_type, complexity, methodology
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
            self._update_performance_metrics(tokens_used, cost, processing_time_ms, True, methodology)
            
            # Complete workflow tracking
            if self.workflow_tracker:
                await self.workflow_tracker.complete_workflow(run_id, {
                    "success": True,
                    "research_type": research_type.value,
                    "complexity": complexity,
                    "methodology_provided": bool(methodology),
                    "tokens_used": tokens_used,
                    "cost": cost
                })
            
            # Log the research interaction
            research_log = {
                "run_id": run_id,
                "timestamp": start_time.isoformat(),
                "message_preview": message[:100],
                "research_type": research_type.value,
                "complexity": complexity,
                "methodology": methodology.dict() if methodology else None,
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
            self.research_history.append(research_log)
            
            return {
                "response": response,
                "agent_id": self.agent_id,
                "research_type": research_type.value,
                "complexity": complexity,
                "methodology": methodology.dict() if methodology else None,
                "confidence": 0.85,
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
                    "agent_type": "research",
                    "model_used": self.llm.model_name,
                    "temperature": self.llm.temperature,
                    "specialization": "research_and_analysis",
                    "agent_id": self.agent_id,
                    "platform_integrated": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing research request in Research Agent: {str(e)}")
            
            # Update performance metrics for failure
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_performance_metrics(0, 0.0, processing_time_ms, False, None)
            
            # Log failure event
            if self.event_bus:
                await self.event_bus.publish(
                    EventType.AGENT_ERROR,
                    {"agent_id": self.agent_id, "error": str(e), "agent_type": "research"},
                    source=self.agent_id
                )
            
            # Complete workflow tracking with failure
            if self.workflow_tracker:
                await self.workflow_tracker.complete_workflow(run_id, {
                    "success": False,
                    "error": str(e)
                })
            
            return {
                "response": "I apologize, but I'm experiencing difficulties processing your research request. Please try rephrasing your question about market research, analysis, or data gathering, and I'll provide comprehensive research guidance.",
                "agent_id": self.agent_id,
                "research_type": "error",
                "confidence": 0.0,
                "error": str(e),
                "tokens_used": 0,
                "processing_time_ms": processing_time_ms,
                "run_id": run_id,
                "platform_integrated": True
            }
    
    def _classify_research_type(self, message: str) -> ResearchType:
        """Classify the type of research being requested."""
        message_lower = message.lower()
        
        # Research type keywords
        market_keywords = ["market", "customer", "consumer", "buyer", "demand", "market size"]
        competitive_keywords = ["competitor", "competition", "competitive", "rival", "benchmark"]
        data_keywords = ["data", "analytics", "statistics", "metrics", "trends", "patterns"]
        academic_keywords = ["research", "study", "academic", "literature", "paper", "methodology"]
        industry_keywords = ["industry", "sector", "vertical", "market trends", "disruption"]
        customer_keywords = ["customer", "user", "feedback", "satisfaction", "behavior"]
        technology_keywords = ["technology", "tech", "innovation", "emerging", "assessment"]
        strategic_keywords = ["strategy", "planning", "opportunity", "business case", "feasibility"]
        
        # Count matches for each research type
        type_scores = {
            ResearchType.MARKET_RESEARCH: sum(1 for kw in market_keywords if kw in message_lower),
            ResearchType.COMPETITIVE_ANALYSIS: sum(1 for kw in competitive_keywords if kw in message_lower),
            ResearchType.DATA_ANALYSIS: sum(1 for kw in data_keywords if kw in message_lower),
            ResearchType.ACADEMIC_RESEARCH: sum(1 for kw in academic_keywords if kw in message_lower),
            ResearchType.INDUSTRY_TRENDS: sum(1 for kw in industry_keywords if kw in message_lower),
            ResearchType.CUSTOMER_INSIGHTS: sum(1 for kw in customer_keywords if kw in message_lower),
            ResearchType.TECHNOLOGY_ASSESSMENT: sum(1 for kw in technology_keywords if kw in message_lower),
            ResearchType.STRATEGIC_PLANNING: sum(1 for kw in strategic_keywords if kw in message_lower),
        }
        
        # Return type with highest score, default to market research
        best_type = max(type_scores.keys(), key=lambda k: type_scores[k])
        return best_type if type_scores[best_type] > 0 else ResearchType.MARKET_RESEARCH
    
    def _assess_research_complexity(self, message: str, context: Dict[str, Any]) -> str:
        """Assess the complexity level of the research request."""
        message_lower = message.lower()
        
        # Complexity indicators
        high_complexity = ["comprehensive", "detailed", "in-depth", "extensive", "complete analysis"]
        medium_complexity = ["analyze", "compare", "evaluate", "assess", "investigate"]
        low_complexity = ["overview", "summary", "quick", "basic", "simple"]
        
        high_count = sum(1 for indicator in high_complexity if indicator in message_lower)
        medium_count = sum(1 for indicator in medium_complexity if indicator in message_lower)
        low_count = sum(1 for indicator in low_complexity if indicator in message_lower)
        
        if high_count > 0:
            return "high"
        elif medium_count > low_count:
            return "medium"
        else:
            return "low"
    
    async def _generate_methodology(self, message: str, context: Dict[str, Any]) -> Optional[ResearchSuggestion]:
        """Generate research methodology recommendations."""
        
        try:
            methodology_chain = self.methodology_prompt | self.methodology_llm
            
            response = await methodology_chain.ainvoke({
                "message": message,
                "context": context.get("channel_type", "research_request")
            })
            
            # Extract JSON from response
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text.strip("```json").strip("```").strip()
            elif response_text.startswith("```"):
                response_text = response_text.strip("```").strip()
            
            methodology_data = json.loads(response_text)
            
            return ResearchSuggestion(**methodology_data)
            
        except Exception as e:
            logger.warning(f"Error generating methodology: {str(e)}")
            return None
    
    async def _generate_research_response(self, message: str, context: Dict[str, Any], 
                                        history_context: str, memory_context: str, research_type: ResearchType,
                                        complexity: str, methodology: Optional[ResearchSuggestion]) -> str:
        """Generate the main research response using the LLM with memory context."""
        
        main_chain = self.main_prompt | self.llm
        
        response = await main_chain.ainvoke({
            "message": message,
            "context": self._format_context(context),
            "history": history_context,
            "memory_context": memory_context,
            "complexity": complexity
        })
        
        response_text = response.content
        
        # Add research type-specific footer
        type_emoji = {
            ResearchType.MARKET_RESEARCH: "📊",
            ResearchType.COMPETITIVE_ANALYSIS: "🔍",
            ResearchType.DATA_ANALYSIS: "📈",
            ResearchType.ACADEMIC_RESEARCH: "📚",
            ResearchType.INDUSTRY_TRENDS: "🌐",
            ResearchType.CUSTOMER_INSIGHTS: "👥",
            ResearchType.TECHNOLOGY_ASSESSMENT: "🔬",
            ResearchType.STRATEGIC_PLANNING: "🎯"
        }
        
        emoji = type_emoji.get(research_type, "🔍")
        response_text += f"\n\n{emoji} *Research Agent - {research_type.value.replace('_', ' ').title()} Specialist*"
        
        # Add methodology summary if available
        if methodology:
            response_text += f"\n\n**Recommended Approach**: {methodology.research_approach.title()}"
            response_text += f"\n**Timeline Estimate**: {methodology.timeline_estimate}"
            if methodology.confidence > 0.7:
                response_text += f"\n**Confidence**: High ({methodology.confidence:.1%})"
        
        return response_text
    
    def _format_research_history(self, history: List[Dict[str, Any]]) -> str:
        """Format research conversation history for context."""
        if not history:
            return "No previous research conversation history."
        
        formatted_history = []
        for item in history[-3:]:  # Last 3 messages for context
            role = "User" if item.get("message_type") == "user_message" else "Research Agent"
            content = item.get("content", "")[:150]  # More context for research discussions
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
            context_parts.append("Research discussion thread")
        
        if context.get("organization"):
            context_parts.append(f"Organization: {context['organization']}")
        
        return ", ".join(context_parts) if context_parts else "Research consultation session"
    
    # Platform Integration Methods (same as other agents)
    async def _register_event_handlers(self):
        """Register event handlers for platform communication."""
        if not self.event_bus:
            return
        try:
            await self.event_bus.subscribe(self.agent_id, [EventType.IMPROVEMENT_APPLIED.value, EventType.PATTERN_DISCOVERED.value, EventType.FEEDBACK_RECEIVED.value], self._handle_platform_event)
            logger.info("Research Agent subscribed to platform events")
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
            message_id = await self.supabase_logger.log_message(conversation_id=conversation_id or str(uuid.uuid4()), user_id=context.get("user_id", "unknown"), content=message, message_type="user_message", agent_type="research", agent_response={"response": response, "agent_id": self.agent_id}, processing_time_ms=processing_time_ms, input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=tokens_used, model_used=self.model_name, estimated_cost=cost)
            return message_id
        except Exception as e:
            logger.error(f"Failed to log to Supabase: {e}")
            return ""
    
    async def _store_memory(self, conversation_id: str, message_id: str, message: str, response: str, context: Dict[str, Any]):
        """Store interaction in vector memory."""
        if not self.vector_store:
            return
        try:
            await self.vector_store.store_conversation_memory(conversation_id=conversation_id or str(uuid.uuid4()), message_id=message_id or str(uuid.uuid4()), content=message, user_id=context.get("user_id", "unknown"), content_type="message", metadata={"agent_type": "research"})
            await self.vector_store.store_conversation_memory(conversation_id=conversation_id or str(uuid.uuid4()), message_id=str(uuid.uuid4()), content=response, user_id=context.get("user_id", "unknown"), content_type="response", metadata={"agent_id": self.agent_id, "agent_type": "research"})
        except Exception as e:
            logger.warning(f"Failed to store memory: {e}")
    
    async def _publish_events(self, run_id: str, message: str, response: str, context: Dict[str, Any], processing_time_ms: float):
        """Publish events about the interaction."""
        if not self.event_bus:
            return
        try:
            await self.event_bus.publish(EventType.AGENT_ACTIVATED, {"agent_id": self.agent_id, "agent_type": "research", "run_id": run_id, "user_id": context.get("user_id"), "success": True, "duration_ms": processing_time_ms, "message_length": len(message), "response_length": len(response)}, source=self.agent_id)
        except Exception as e:
            logger.warning(f"Failed to publish events: {e}")
    
    def _update_performance_metrics(self, tokens: int, cost: float, processing_time: float, success: bool, methodology: Optional[ResearchSuggestion]):
        """Update performance tracking metrics."""
        self.performance_metrics["total_interactions"] += 1
        if success:
            self.performance_metrics["successful_interactions"] += 1
        if methodology:
            self.performance_metrics["research_methodologies_provided"] += 1
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
                    logger.info("Applied prompt improvement for Research Agent")
        except Exception as e:
            logger.error(f"Failed to apply improvement: {e}")
    
    async def _learn_from_pattern(self, pattern_data: Dict[str, Any]):
        """Learn from discovered patterns."""
        try:
            pattern_type = pattern_data.get("pattern_type")
            if pattern_type == "successful_interaction":
                pass  # Can be enhanced with research methodology improvements
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
    
    def get_research_stats(self) -> Dict[str, Any]:
        """Get statistics about research interactions handled."""
        if not self.research_history:
            return {"total_research_requests": 0}
        
        total_tokens = sum(log.get("tokens_used", 0) for log in self.research_history)
        total_cost = sum(log.get("cost", 0) for log in self.research_history)
        
        # Research type distribution
        type_counts = {}
        complexity_counts = {}
        methodology_requests = sum(1 for log in self.research_history if log.get("methodology"))
        
        for log in self.research_history:
            research_type = log.get("research_type", "unknown")
            type_counts[research_type] = type_counts.get(research_type, 0) + 1
            
            complexity = log.get("complexity", "unknown")
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        return {
            "total_research_requests": len(self.research_history),
            "total_tokens_used": total_tokens,
            "total_cost": total_cost,
            "research_type_distribution": type_counts,
            "complexity_distribution": complexity_counts,
            "methodology_requests": methodology_requests,
            "methodology_request_rate": methodology_requests / len(self.research_history) if self.research_history else 0
        }
    
    async def close(self):
        """Close the agent and cleanup platform resources."""
        try:
            logger.info("Closing Research Agent connections...")
            
            # Close LLM clients
            if hasattr(self.llm, 'client') and hasattr(self.llm.client, 'close'):
                await self.llm.client.close()
            
            if hasattr(self.methodology_llm, 'client') and hasattr(self.methodology_llm.client, 'close'):
                await self.methodology_llm.client.close()
            
            # Close platform integrations
            if self.vector_store:
                await self.vector_store.close()
            
            if self.supabase_logger:
                await self.supabase_logger.close()
            
            # Unsubscribe from events
            if self.event_bus:
                await self.event_bus.unsubscribe(self.agent_id)
                
            logger.info("Research Agent closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing Research Agent: {e}") 