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
from datetime import datetime, timezone
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel

# Platform integrations
from storage.supabase import SupabaseLogger
from evolution.memory import VectorMemoryStore
from core.events import EventBus, EventType
from evolution.tracker import WorkflowTracker

# MCP integrations (will be imported conditionally to avoid circular imports)
# from mcp.mcp_discovery_engine import MCPDiscoveryEngine
# from mcp.dynamic_tool_builder import DynamicToolBuilder

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
                 workflow_tracker: Optional[WorkflowTracker] = None,
                 # MCP integrations
                 mcp_discovery_engine = None,
                 dynamic_tool_builder = None,
                 mcp_tool_registry = None):
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
            
            # MCP integrations (Model Context Protocol)
            mcp_discovery_engine: Engine to find MCP solutions for missing capabilities
            dynamic_tool_builder: Builder for creating new tools when MCPs aren't available
            mcp_tool_registry: Registry of available MCP tools
        """
        
        # Core configuration
        self.specialty = specialty
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.agent_id = agent_id or f"universal_{specialty.lower().replace(' ', '_')}"
        self.tools = tools or []
        self.adaptive_mode = False  # Enable dynamic self-tuning by default
        
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
        
        # MCP integrations for tool discovery and building
        self.mcp_discovery_engine = mcp_discovery_engine
        self.dynamic_tool_builder = dynamic_tool_builder
        self.mcp_tool_registry = mcp_tool_registry
        
        # Track MCP tool requests and gaps
        self.mcp_tool_requests: Dict[str, str] = {}  # gap_id -> request_id
        self.pending_mcp_tasks: List[Dict[str, Any]] = []
        
        # Token management and cost optimization
        self.token_management = {
            'max_memory_tokens': 500,          # Budget for memory context
            'max_working_memory_tokens': 300,  # Budget for working memory  
            'max_history_tokens': 200,         # Budget for conversation history
            'max_tool_results_tokens': 200,    # Budget for tool outputs
            'similarity_threshold': 0.8,       # Higher threshold = more relevant memories
            'max_memories_to_load': 2,         # Limit number of memories loaded
            'enable_token_optimization': True,  # Enable smart token management
            'context_compression': True,        # Enable context compression
            'adaptive_budget': True             # Adjust budgets based on task complexity
        }
        
        # Context window limits by model
        self.context_windows = {
            'gpt-3.5-turbo': 4096,
            'gpt-3.5-turbo-0125': 4096,
            'gpt-4': 8192,
            'gpt-4-0125-preview': 8192,
            'gpt-4-32k': 32768
        }
        
        # Initialize prompt templates
        self.main_prompt = self._create_main_prompt()
        self.improvement_prompt = self._create_improvement_prompt()
        
        # Track current prompt template for performance logging
        self.current_prompt_template_id: Optional[str] = None
        self.enable_dynamic_prompts = True  # Feature flag for dynamic prompts
        
        # Initialize working memory with required keys
        self._working_memory = {
            'conversation_flow': [],
            'learned_approaches': {},
            'last_task': '',
            'current_focus': '',
            'pattern_insights': []
        }
        
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
        
        # Adaptive processing configuration - TEMPORARILY DISABLED to prevent loops
        self.adaptive_mode = False  # TODO: Re-enable after testing basic functionality
        
        # Register event handlers if event bus available
        if self.event_bus:
            asyncio.create_task(self._register_event_handlers())
        
        mcp_status = "with MCP-first tool discovery" if self.mcp_discovery_engine else "without MCP integration"
        logger.info(f"Universal Agent '{self.specialty}' initialized with full platform integration {mcp_status}")
    
    def _create_main_prompt(self) -> ChatPromptTemplate:
        """Create the main conversation prompt template based on specialty."""
        
        # TODO: This will be replaced with dynamic prompt loading from Supabase
        # For now, keeping existing logic but making it configurable
        
        # Create dynamic system template based on configuration
        tools_description = ""
        if self.tools:
            tool_list = [f"- {tool.name}: {tool.description}" for tool in self.tools if tool.enabled]
            tools_description = f"\n\n**Available Tools:**\n" + "\n".join(tool_list)
        
        # Use configured system prompt or fallback
        base_system_prompt = self.system_prompt or f"You are a {self.specialty} specialist."
        
        system_template = f"""{base_system_prompt}

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

**TOOL DECISION FRAMEWORK:**
1. **Assess Need**: Does this require real-time data, external services, or specialized processing?
2. **MCP-First**: Always check for existing MCP tools before creating custom solutions
3. **Gap Detection**: If confidence < 0.7, identify what tools are missing
4. **User Guidance**: Clearly explain what tools you're using and why

**Tool Selection Priorities:**
- Existing MCPs > MCP Library > Custom Tool Creation > Inform user of limitations

**Communication Guidelines:**
1. Be precise and technically accurate in your specialty
2. Explain complex concepts clearly for the user's level
3. Provide actionable recommendations
4. Use examples and practical applications when helpful
5. Be proactive in suggesting improvements or alternatives
6. Keep responses concise but comprehensive
7. Learn from each interaction to improve future responses
8. **Be transparent about tool usage** - explain what tools you're using and why{tools_description}

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

    async def _create_dynamic_prompt(self, complexity_level: str = "medium") -> ChatPromptTemplate:
        """
        Create prompt using dynamic prompt manager from Supabase.
        
        This is the NEW way - prompts are loaded from Supabase and optimized based on performance.
        """
        try:
            from core.dynamic_prompt_manager import get_prompt_manager, ComplexityLevel
            
            # Map string to enum
            complexity_map = {
                'simple': ComplexityLevel.SIMPLE,
                'medium': ComplexityLevel.MEDIUM,
                'complex': ComplexityLevel.COMPLEX
            }
            complexity_enum = complexity_map.get(complexity_level, ComplexityLevel.MEDIUM)
            
            # Get optimal prompt from Supabase
            prompt_manager = get_prompt_manager()
            prompt_template = await prompt_manager.get_optimal_prompt(
                agent_type='universal',
                specialty=self.specialty.lower() if self.specialty else None,
                complexity_level=complexity_enum,
                context={'agent_id': self.agent_id}
            )
            
            # Store template ID for performance tracking
            self.current_prompt_template_id = prompt_template.template_id
            
            # Build system template from dynamic components
            tools_description = ""
            if self.tools:
                tool_list = [f"- {tool.name}: {tool.description}" for tool in self.tools if tool.enabled]
                tools_description = f"\n\n**Available Tools:**\n" + "\n".join(tool_list)
            
            # Combine dynamic prompt components
            system_template = f"""{prompt_template.system_prompt}

**Your Specialty:** {self.specialty}

**Tool Decision Guidance:**
{prompt_template.tool_decision_guidance}

**Communication Style:**
{prompt_template.communication_style}

**Tool Selection Criteria:**
{prompt_template.tool_selection_criteria}{tools_description}

Current conversation context: {{context}}
Recent conversation history: {{history}}
Relevant context from memory: {{memory_context}}"""

            human_template = f"""User message: {{message}}

As the {self.specialty} specialist, provide expert assistance using the tool decision framework above."""

            logger.info(f"🎯 Using dynamic prompt template {prompt_template.template_id} "
                       f"(performance: {prompt_template.performance_score:.2f})")
            
            return ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
            
        except Exception as e:
            logger.warning(f"Failed to load dynamic prompt, falling back to static: {e}")
            # Fallback to existing static prompt
            return self._create_main_prompt()
    
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
    
    async def process(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process_message to match orchestrator's expectations."""
        return await self.process_message(message, context)
    
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
            # Restore working memory
            await self.restore_working_memory()
            
            logger.info(f"Universal Agent '{self.specialty}' processing: '{message[:50]}...'")
            
            # 🚀 STEP 1: Assess complexity and load dynamic prompt
            complexity_level = await self._assess_message_complexity(message, context)
            context['complexity_level'] = complexity_level  # Ensure it's available
            
            # 🎯 STEP 2: Always try to load dynamic prompt (with fallback)
            if self.enable_dynamic_prompts:
                try:
                    self.main_prompt = await self._create_dynamic_prompt(complexity_level)
                    logger.info(f"🎯 Using dynamic prompt for {complexity_level} complexity")
                except Exception as e:
                    logger.warning(f"Failed to load dynamic prompt, using static: {e}")
            
            # Calculate dynamic context budget if optimization enabled
            if self.token_management['enable_token_optimization']:
                context_budget = self._calculate_context_budget(message)
                logger.debug(f"Context budget: {sum(context_budget.values())} tokens allocated")
                
                # Update token management limits dynamically
                self.token_management['max_memory_tokens'] = context_budget['memory_context']
                self.token_management['max_working_memory_tokens'] = context_budget['working_memory']
                self.token_management['max_history_tokens'] = context_budget['history']
                self.token_management['max_tool_results_tokens'] = context_budget['tool_results']
            
            # Start workflow tracking
            if self.workflow_tracker:
                await self.workflow_tracker.start_workflow(run_id, {
                    "agent_id": self.agent_id,
                    "specialty": self.specialty,
                    "message_preview": message[:100],
                    "complexity_level": complexity_level,
                    "token_optimization": self.token_management['enable_token_optimization']
                })
            
            # Retrieve relevant memory context
            memory_context = await self._get_memory_context(message, context)
            
            # Prepare conversation history context
            history_context = self._format_conversation_history(context.get("conversation_history", []))
            
            # Execute tools if available
            tool_results = await self._execute_tools(message, context)
            
            # Generate specialist response with optional adaptive processing
            if self.adaptive_mode:
                try:
                    # Use adaptive processing for better performance
                    adaptive_result = await self.adapt_to_task(message, context)
                    response = adaptive_result.get('response', '')
                    tokens_used = adaptive_result.get('tokens_used', 0)
                    input_tokens = 0  # Would need to extract from adaptive result
                    output_tokens = 0  # Would need to extract from adaptive result
                    cost = adaptive_result.get('cost', 0.0)
                    adaptive_metadata = {
                        'adaptation_used': adaptive_result.get('adaptation_used', False),
                        'optimal_params': adaptive_result.get('optimal_params', {}),
                        'task_analysis': adaptive_result.get('task_analysis', {}),
                        'quality_score': adaptive_result.get('quality_score', 0.0)
                    }
                except Exception as e:
                    logger.warning(f"Adaptive processing failed, falling back to standard: {e}")
                    # Fallback to standard processing
                    try:
                        with get_openai_callback() as cb:
                            response = await self._generate_specialist_response(
                                message, context, history_context, memory_context, tool_results
                            )
                        tokens_used = cb.total_tokens
                        input_tokens = cb.prompt_tokens
                        output_tokens = cb.completion_tokens
                        cost = cb.total_cost
                        adaptive_metadata = {'adaptation_used': False, 'fallback_reason': str(e)}
                    except Exception as e2:
                        logger.warning(f"OpenAI callback failed: {e2}")
                        response = await self._generate_specialist_response(
                            message, context, history_context, memory_context, tool_results
                        )
                        tokens_used = 0
                        input_tokens = 0
                        output_tokens = 0
                        cost = 0.0
                        adaptive_metadata = {'adaptation_used': False, 'fallback_reason': str(e)}
            else:
                # Standard processing without adaptation
                try:
                    with get_openai_callback() as cb:
                        response = await self._generate_specialist_response(
                            message, context, history_context, memory_context, tool_results
                        )
                    tokens_used = cb.total_tokens
                    input_tokens = cb.prompt_tokens
                    output_tokens = cb.completion_tokens
                    cost = cb.total_cost
                    adaptive_metadata = {'adaptation_used': False}
                except Exception as e:
                    logger.warning(f"OpenAI callback failed, proceeding without tracking: {e}")
                    response = await self._generate_specialist_response(
                        message, context, history_context, memory_context, tool_results
                    )
                    tokens_used = 0
                    input_tokens = 0
                    output_tokens = 0
                    cost = 0.0
                    adaptive_metadata = {'adaptation_used': False}
            
            # Calculate processing time
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # 📊 STEP 3: Log prompt performance for learning
            if self.enable_dynamic_prompts and self.current_prompt_template_id:
                await self._log_prompt_performance(
                    message, response, context, cost, processing_time_ms
                )
            
            # Token usage monitoring and alerting
            if self.token_management['enable_token_optimization']:
                context_size = len(memory_context + history_context) // 4
                total_context_tokens = context_size + tokens_used
                
                # Log detailed token breakdown
                logger.info(f"Token usage breakdown for {self.specialty}:")
                logger.info(f"  Input tokens: {input_tokens}")
                logger.info(f"  Output tokens: {output_tokens}")
                logger.info(f"  Memory context: ~{len(memory_context) // 4} tokens")
                logger.info(f"  History context: ~{len(history_context) // 4} tokens")
                logger.info(f"  Total tokens: {tokens_used}")
                logger.info(f"  Cost: ${cost:.4f}")
                
                # Alert on high usage
                context_window = self.context_windows.get(self.model_name, 4096)
                if tokens_used > context_window * 0.8:
                    logger.warning(f"🚨 HIGH TOKEN USAGE: {tokens_used}/{context_window} tokens ({tokens_used/context_window:.1%})")
                    logger.warning(f"   Consider enabling more aggressive optimization for {self.specialty}")
                
                # Alert on high cost
                cost_threshold = 0.01  # $0.01 per request
                if cost > cost_threshold:
                    logger.warning(f"💰 HIGH COST ALERT: ${cost:.4f} for single request")
                
                # Update performance metrics with cost efficiency
                cost_per_token = cost / max(1, tokens_used)
                if hasattr(self, 'cost_efficiency_history'):
                    self.cost_efficiency_history.append(cost_per_token)
                else:
                    self.cost_efficiency_history = [cost_per_token]
                
                # Keep only recent cost history
                self.cost_efficiency_history = self.cost_efficiency_history[-10:]
            
            # Log to Supabase
            conversation_id = context.get("conversation_id")
            if not conversation_id:
                logger.warning(f"🚨 Agent {self.agent_id} received NO conversation_id in context from user {context.get('user_id', 'unknown')}")
                logger.debug(f"Context keys available: {list(context.keys())}")
                logger.warning(f"🚨 SKIPPING Supabase logging and memory storage - no conversation_id provided")
                # Don't log or store memory without conversation_id to prevent false warnings
                message_id = ""
            else:
                logger.debug(f"✅ Agent {self.agent_id} using conversation_id: {conversation_id}")
                
                message_id = await self._log_to_supabase(
                    conversation_id, message, response, context, 
                    tokens_used, input_tokens, output_tokens, cost, processing_time_ms
                )
                
                # Store in memory only when we have a valid conversation_id
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
                    "cost": cost,
                    "complexity_level": complexity_level,
                    "dynamic_prompt_used": self.enable_dynamic_prompts and self.current_prompt_template_id is not None,
                    "adaptive_params": adaptive_metadata,
                    "task_analysis": adaptive_metadata.get('task_analysis') if 'adaptive_metadata' in locals() else None,
                    "quality_score": adaptive_metadata.get('quality_score') if 'adaptive_metadata' in locals() else None
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
            
            # Check for MCP tool gaps if confidence is low
            confidence = 0.9  # Default confidence
            mcp_gap_detected = False
            mcp_request_info = {}
            
            # MCP-FIRST APPROACH: Check for tool gaps when confidence is low
            if confidence < 0.7 and self.dynamic_tool_builder:
                logger.info(f"🔍 Low confidence ({confidence:.2f}) - checking for MCP tool gaps...")
                
                mcp_gap = await self._detect_mcp_tool_gap(message, response, context, confidence)
                if mcp_gap:
                    mcp_gap_detected = True
                    mcp_request_info = await self._handle_mcp_tool_gap(mcp_gap, message, context)
                    
                    # Update response to inform user about MCP tool request
                    if mcp_request_info.get('mcp_solutions_found', 0) > 0:
                        response = mcp_request_info.get('enhanced_response', response)
                        confidence = 0.8  # Higher confidence when we have MCP solutions
                    else:
                        response = mcp_request_info.get('tool_building_response', response)
                        confidence = 0.6  # Moderate confidence for custom tool building
            
            # Save working memory for next interaction
            await self.save_working_memory()
            
            # Log prompt performance if using dynamic prompts
            await self._log_prompt_performance(message, response, context, cost, processing_time_ms)
            
            return {
                "response": response,
                "agent_id": self.agent_id,
                "specialty": self.specialty,
                "conversation_type": "specialist",
                "confidence": confidence,
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
                    "platform_integrated": True,
                    "adaptive_processing": adaptive_metadata,
                    "mcp_integration": {
                        "mcp_discovery_available": bool(self.mcp_discovery_engine),
                        "tool_builder_available": bool(self.dynamic_tool_builder),
                        "gap_detected": mcp_gap_detected,
                        "pending_mcp_tasks": len(self.pending_mcp_tasks),
                        "mcp_request_info": mcp_request_info
                    },
                    "prompt_template_id": getattr(self, 'current_prompt_template_id', None)
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
    
    async def _detect_mcp_tool_gap(self, message: str, response: str, 
                                 context: Dict[str, Any], confidence: float) -> Optional[Any]:
        """
        Detect if the agent needs an MCP tool it doesn't have.
        
        MCP-FIRST APPROACH: Prioritizes finding MCP solutions before building custom tools.
        
        Returns ToolGap object if gap detected, None otherwise.
        """
        if confidence >= 0.7:  # Good confidence, no gap
            return None
        
        try:
            # Check if we have MCP tool builder
            if not self.dynamic_tool_builder:
                return None
            
            # Use dynamic tool builder to detect gap (it has MCP-first logic)
            gap = await self.dynamic_tool_builder.detect_tool_gap(
                agent_id=self.agent_id,
                message=message,
                context={
                    **context,
                    "low_confidence_response": response,
                    "confidence_score": confidence,
                    "agent_specialty": self.specialty,
                    "detection_source": "universal_agent"
                }
            )
            
            if gap:
                logger.info(f"🔍 MCP tool gap detected by {self.specialty}: {gap.capability_needed}")
                if gap.mcp_solutions:
                    logger.info(f"📦 Found {len(gap.mcp_solutions)} potential MCP solutions")
                    for i, mcp in enumerate(gap.mcp_solutions[:3]):
                        logger.info(f"   {i+1}. {mcp.capability.name} ({mcp.match_type}, score: {mcp.match_score:.2f})")
                else:
                    logger.info(f"🔧 No MCP solutions found - will attempt custom tool creation")
            
            return gap
            
        except Exception as e:
            logger.error(f"Error detecting MCP tool gap: {e}")
            return None
    
    async def _handle_mcp_tool_gap(self, mcp_gap, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a detected MCP tool gap by requesting tool creation.
        
        Returns information about the MCP tool request.
        """
        try:
            user_id = context.get('user_id', 'unknown')
            
            # Check if we already have a pending request for this capability
            existing_request = self._find_existing_mcp_request(mcp_gap.capability_needed)
            if existing_request:
                return await self._check_mcp_request_status(existing_request, message, context)
            
            # Request tool creation (will use MCP-first approach)
            request_id = await self.dynamic_tool_builder.request_tool_creation(
                gap=mcp_gap, 
                user_id=user_id
            )
            
            # Track the request
            self.mcp_tool_requests[mcp_gap.gap_id] = request_id
            
            # Store the pending task for later completion
            self.pending_mcp_tasks.append({
                'message': message,
                'context': context,
                'gap_id': mcp_gap.gap_id,
                'request_id': request_id,
                'created_at': datetime.utcnow(),
                'mcp_solutions_count': len(mcp_gap.mcp_solutions) if mcp_gap.mcp_solutions else 0
            })
            
            # Generate enhanced response based on available solutions
            if mcp_gap.mcp_solutions:
                enhanced_response = await self._generate_mcp_solution_response(mcp_gap, request_id)
                return {
                    'mcp_solutions_found': len(mcp_gap.mcp_solutions),
                    'enhanced_response': enhanced_response,
                    'request_id': request_id,
                    'best_solution': mcp_gap.mcp_solutions[0].capability.name
                }
            else:
                tool_building_response = await self._generate_tool_building_response(mcp_gap, request_id)
                return {
                    'mcp_solutions_found': 0,
                    'tool_building_response': tool_building_response,
                    'request_id': request_id,
                    'custom_tool_needed': True
                }
                
        except Exception as e:
            logger.error(f"Error handling MCP tool gap: {e}")
            return {
                'error': str(e),
                'mcp_solutions_found': 0
            }
    
    async def _generate_mcp_solution_response(self, mcp_gap, request_id: str) -> str:
        """Generate response when MCP solutions are available."""
        best_mcp = mcp_gap.mcp_solutions[0]
        
        response = f"""I found a way to help you! 🚀

**What you need:** {mcp_gap.capability_needed}
**Solution:** I can use the **{best_mcp.capability.name}** MCP

📦 **Available MCP Solutions:**"""
        
        for i, mcp in enumerate(mcp_gap.mcp_solutions[:3]):
            match_emoji = "✅" if mcp.match_type == "exact_existing" else "🔌" if mcp.match_type == "exact" else "🔗"
            response += f"\n{i+1}. {match_emoji} **{mcp.capability.name}** ({mcp.match_score:.0%} match)"
            response += f"\n   {mcp.capability.description}"
        
        if best_mcp.match_type == "exact_existing":
            response += f"\n\n✅ **Great news!** The {best_mcp.capability.name} is already connected. Let me set this up for you..."
        else:
            response += f"\n\n🔌 **Setup needed:** I'll help you connect to {best_mcp.capability.name}. You may need to provide some credentials."
        
        response += f"\n\n**Request ID:** `{request_id}`\nI'll handle the technical details - you focus on your task! 🎯"
        
        return response
    
    async def _generate_tool_building_response(self, mcp_gap, request_id: str) -> str:
        """Generate response when custom tool building is needed."""
        
        response = f"""I need to build a custom tool to help you! 🛠️

**What I need:** {mcp_gap.capability_needed}
**Why:** {mcp_gap.description}

🔧 **Here's my plan:**
1. I've analyzed your request and found I need a new capability
2. No existing MCP (Model Context Protocol) tools match your needs
3. I'll build a custom tool specifically for this task
4. You might need to provide some information (like API keys)

**Suggested approaches:**"""
        
        for solution in mcp_gap.suggested_solutions:
            response += f"\n• {solution}"
        
        response += f"""

**Request ID:** `{request_id}`

I'll notify you when I need your input to complete the tool. This is how I continuously learn and improve! 🚀"""
        
        return response
    
    def _find_existing_mcp_request(self, capability: str) -> Optional[str]:
        """Find if we already have a pending MCP request for this capability."""
        capability_lower = capability.lower()
        
        for task in self.pending_mcp_tasks:
            gap_id = task['gap_id']
            gap = self.dynamic_tool_builder.active_gaps.get(gap_id)
            if gap and capability_lower in gap.capability_needed.lower():
                return task['request_id']
        
        return None
    
    async def _check_mcp_request_status(self, request_id: str, message: str, 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Check status of existing MCP tool request."""
        request = self.dynamic_tool_builder.active_requests.get(request_id)
        
        if not request:
            return {
                'response': "I'm working on getting the MCP tools I need. Please try again in a moment.",
                'mcp_solutions_found': 0
            }
        
        status_messages = {
            "analyzing": "🔍 I'm analyzing what MCP tool I need...",
            "building": "🔧 I'm setting up the MCP connection automatically...",
            "testing": "🧪 I'm testing the new MCP tool...",
            "user_input_needed": f"👤 I need your help to set up an MCP! Check for collaboration request: {request_id}",
            "completed": "✅ The MCP tool is ready! Let me retry your request...",
            "failed": "❌ MCP setup failed. Let me try a different approach..."
        }
        
        status_message = status_messages.get(request.status.value, "🔄 Working on your MCP tool request...")
        
        return {
            'response': f"**MCP Tool Status:** {status_message}",
            'request_id': request_id,
            'status': request.status.value,
            'mcp_solutions_found': 1  # Assume at least one if we have a request
        }

    async def _log_prompt_performance(self, message: str, response: str, context: Dict[str, Any], cost: float, processing_time_ms: float):
        """Log prompt performance for self-improvement."""
        if not self.current_prompt_template_id or self.current_prompt_template_id == 'fallback':
            return
        
        try:
            from core.dynamic_prompt_manager import get_prompt_manager
            
            # Calculate performance metrics
            complexity_score = context.get('complexity_score', 0.5)
            
            # Estimate tool decision accuracy (simplified)
            tools_used = context.get('tools_invoked', [])
            tool_decision_accuracy = 0.8 if tools_used else 0.6  # Placeholder logic
            
            # Estimate response quality (simplified - could use LLM evaluation)
            response_quality = min(1.0, max(0.3, (len(response) / 1000) * 0.5 + 0.5))  # Placeholder
            
            # Get user satisfaction if available
            user_satisfaction = context.get('user_feedback_score')
            
            prompt_manager = get_prompt_manager()
            await prompt_manager.log_prompt_performance(
                template_id=self.current_prompt_template_id,
                user_message=message,
                complexity_score=complexity_score,
                tool_decision_accuracy=tool_decision_accuracy,
                response_quality_score=response_quality,
                tools_invoked=tools_used,
                mcp_solutions_found=context.get('mcp_solutions_found', 0),
                user_satisfaction=user_satisfaction
            )
            
        except Exception as e:
            logger.error(f"Failed to log prompt performance: {e}")

    async def _assess_message_complexity(self, message: str, context: Dict[str, Any]) -> str:
        """
        Assess the complexity level of a user message for dynamic prompt selection.
        
        Args:
            message: User message to analyze
            context: Conversation context
            
        Returns:
            Complexity level: 'simple', 'medium', or 'complex'
        """
        try:
            message_lower = message.lower()
            
            # Simple heuristics for complexity assessment
            complexity_indicators = {
                'simple': [
                    len(message) < 50,  # Short messages
                    any(word in message_lower for word in ['hello', 'hi', 'thanks', 'help', 'what', 'who']),
                    message.count('?') <= 1,  # Single question
                    len(message.split()) < 10  # Few words
                ],
                'complex': [
                    len(message) > 200,  # Long messages
                    message.count('?') > 2,  # Multiple questions
                    any(word in message_lower for word in ['integrate', 'optimize', 'analyze', 'implement', 'compare', 'complex']),
                    len(message.split()) > 40,  # Many words
                    any(word in message_lower for word in ['architecture', 'system', 'algorithm', 'framework', 'strategy']),
                    len(context.get('conversation_history', [])) > 5  # Long conversation - fixed to return boolean
                ]
            }
            
            # Count indicators
            simple_score = sum(complexity_indicators['simple'])
            complex_score = sum(complexity_indicators['complex'])
            
            # Determine complexity level
            if complex_score >= 2:
                complexity_level = 'complex'
            elif simple_score >= 2:
                complexity_level = 'simple'
            else:
                complexity_level = 'medium'
            
            logger.debug(f"Message complexity assessment: {complexity_level} "
                        f"(simple: {simple_score}, complex: {complex_score})")
            
            return complexity_level
            
        except Exception as e:
            logger.warning(f"Failed to assess message complexity: {e}")
            return 'medium'  # Default fallback

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
        """Format conversation history for context with token budget management."""
        if not history or not self.token_management['enable_token_optimization']:
            return f"No previous conversation history in {self.specialty}."
        
        max_history_tokens = self.token_management['max_history_tokens']
        formatted_history = []
        total_tokens = 0
        
        # Process recent messages first (reverse order to prioritize recent)
        for item in reversed(history[-5:]):  # Look at last 5 messages
            role = "User" if item.get("message_type") == "user_message" else "Assistant"
            content = item.get("content", "")
            
            # Smart truncation to fit budget
            estimated_tokens = len(content) // 4
            if total_tokens + estimated_tokens > max_history_tokens:
                # Try to fit a truncated version
                remaining_tokens = max_history_tokens - total_tokens
                if remaining_tokens > 20:  # Only if we have meaningful space
                    content = content[:remaining_tokens * 4] + "..."
                    formatted_history.insert(0, f"{role}: {content}")
                    total_tokens += remaining_tokens
                break
            
            # Smart content truncation for readability
            if len(content) > 200:  # Longer messages get truncated regardless
                content = content[:200] + "..."
            
            formatted_history.insert(0, f"{role}: {content}")
            total_tokens += len(content) // 4
            
            # Stop if we're approaching budget
            if total_tokens >= max_history_tokens * 0.9:
                break
        
        result = "\n".join(formatted_history) if formatted_history else f"No recent history in {self.specialty}."
        logger.debug(f"History context: {len(result) // 4} tokens (budget: {max_history_tokens})")
        return result
    
    def _calculate_context_budget(self, message: str) -> Dict[str, int]:
        """Calculate dynamic token budget allocation based on task and model."""
        if not self.token_management['adaptive_budget']:
            # Use fixed budgets
            return {
                'system_prompt': 800,
                'user_message': len(message) // 4,
                'memory_context': self.token_management['max_memory_tokens'],
                'working_memory': self.token_management['max_working_memory_tokens'],
                'history': self.token_management['max_history_tokens'],
                'tool_results': self.token_management['max_tool_results_tokens'],
                'response_reserve': 1000
            }
        
        # Dynamic budget allocation
        total_window = self.context_windows.get(self.model_name, 4096)
        response_reserve = 1000  # Reserve for response
        available_context = total_window - response_reserve
        
        # Estimate base requirements
        system_prompt_tokens = 800  # Roughly fixed
        user_message_tokens = len(message) // 4
        
        # Calculate remaining budget for context
        remaining_budget = available_context - system_prompt_tokens - user_message_tokens
        
        # Allocate remaining budget proportionally
        if remaining_budget > 0:
            # Prioritize based on task complexity
            message_lower = message.lower()
            is_complex = any(word in message_lower for word in ['analyze', 'explain', 'complex', 'detailed', 'comprehensive'])
            
            if is_complex:
                # Complex tasks get more memory context
                return {
                    'system_prompt': system_prompt_tokens,
                    'user_message': user_message_tokens,
                    'memory_context': min(800, int(remaining_budget * 0.4)),
                    'working_memory': min(400, int(remaining_budget * 0.2)),
                    'history': min(300, int(remaining_budget * 0.2)),
                    'tool_results': min(300, int(remaining_budget * 0.2)),
                    'response_reserve': response_reserve
                }
            else:
                # Simple tasks use less context
                return {
                    'system_prompt': system_prompt_tokens,
                    'user_message': user_message_tokens,
                    'memory_context': min(400, int(remaining_budget * 0.3)),
                    'working_memory': min(200, int(remaining_budget * 0.2)),
                    'history': min(200, int(remaining_budget * 0.2)),
                    'tool_results': min(200, int(remaining_budget * 0.3)),
                    'response_reserve': response_reserve
                }
        else:
            # Minimal budget fallback
            return {
                'system_prompt': 600,  # Reduced system prompt
                'user_message': user_message_tokens,
                'memory_context': 200,
                'working_memory': 100,
                'history': 100,
                'tool_results': 100,
                'response_reserve': response_reserve
            }
    
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
        """Retrieve relevant context from vector memory with smart token management."""
        if not self.vector_store or not self.token_management['enable_token_optimization']:
            return ""
        
        try:
            user_id = context.get("user_id")
            
            # Use optimized search parameters
            similar_memories = await self.vector_store.search_similar_memories(
                message, 
                user_id=user_id, 
                limit=self.token_management['max_memories_to_load'],
                similarity_threshold=self.token_management['similarity_threshold']
            )
            
            if not similar_memories:
                return ""
            
            # Smart token-budgeted context building
            context_parts = []
            total_tokens = 0
            max_memory_tokens = self.token_management['max_memory_tokens']
            
            for memory in similar_memories:
                content = memory.get('content_summary', '')
                if not content:
                    continue
                
                # Estimate tokens (rough: 1 token ≈ 4 characters)
                estimated_tokens = len(content) // 4
                
                # Check if this memory fits in our budget
                if total_tokens + estimated_tokens > max_memory_tokens:
                    # Try to fit a truncated version
                    remaining_tokens = max_memory_tokens - total_tokens
                    if remaining_tokens > 50:  # Only if we have meaningful space left
                        truncated_content = content[:remaining_tokens * 4]
                        # Find last complete sentence
                        last_period = truncated_content.rfind('.')
                        if last_period > len(truncated_content) * 0.7:  # If we found a good break point
                            content = truncated_content[:last_period + 1]
                        else:
                            content = truncated_content + "..."
                        context_parts.append(f"- {content}")
                        total_tokens += len(content) // 4
                    break
                
                context_parts.append(f"- {content}")
                total_tokens += estimated_tokens
                
                # Stop if we're approaching our budget
                if total_tokens >= max_memory_tokens * 0.9:
                    break
            
            if context_parts:
                result = f"Relevant context from previous conversations:\n" + "\n".join(context_parts)
                # Log token usage for monitoring
                actual_tokens = len(result) // 4
                logger.debug(f"Memory context: {actual_tokens} tokens (budget: {max_memory_tokens})")
                return result
            
        except Exception as e:
            logger.warning(f"Failed to retrieve memory context: {e}")
        
        return ""
    
    async def _log_to_supabase(self, conversation_id: str, message: str, response: str, 
                             context: Dict[str, Any], tokens_used: int, input_tokens: int,
                             output_tokens: int, cost: float, processing_time_ms: float) -> str:
        """Log interaction to Supabase database."""
        if not self.supabase_logger or not conversation_id:
            logger.warning(f"Skipping Supabase logging - no logger or conversation_id")
            return ""
        
        try:
            message_id = await self.supabase_logger.log_message(
                conversation_id=conversation_id,
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
        """Store interaction in vector memory with circuit breaker."""
        if not self.vector_store or not conversation_id:
            logger.warning(f"Skipping memory storage - no vector store or conversation_id")
            return
        
        # Circuit breaker for memory storage
        if not hasattr(self, '_memory_storage_failures'):
            self._memory_storage_failures = 0
            
        if self._memory_storage_failures > 5:
            logger.warning(f"Memory storage circuit breaker active ({self._memory_storage_failures} failures)")
            return
        
        try:
            # Store user message
            await self.vector_store.store_conversation_memory(
                conversation_id=conversation_id,
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
                conversation_id=conversation_id,
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
            
            # Reset failure count on success
            self._memory_storage_failures = max(0, self._memory_storage_failures - 1)
            
        except Exception as e:
            self._memory_storage_failures += 1
            logger.warning(f"Failed to store memory (failure count: {self._memory_storage_failures}): {e}")
    
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
    
    async def persist_learning(self, pattern_type: str, pattern_data: Dict):
        """Persist learned patterns to vector store for all agents to access."""
        if self.vector_store and self.supabase_logger:
            try:
                # Create system learning conversation ID  
                conversation_id = f"learning_{pattern_type}_{datetime.utcnow().timestamp()}"
                message_id = str(uuid.uuid4())
                
                # First, create the conversation record to prevent memory warnings
                if hasattr(self.supabase_logger, 'client'):
                    try:
                        self.supabase_logger.client.table("conversations").insert({
                            "id": conversation_id,
                            "user_id": "system",
                            "channel_id": "system_learning",
                            "status": "active"
                        }).execute()
                    except Exception as e:
                        # Conversation might already exist, that's fine
                        logger.debug(f"System conversation already exists: {e}")
                
                # Now store the learning pattern
                await self.vector_store.store_conversation_memory(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    content=json.dumps(pattern_data),
                    user_id="system",  # System-level learning
                    content_type="learned_pattern",
                    metadata={
                        'pattern_type': pattern_type,
                        'agent_specialty': self.specialty,
                        'confidence': pattern_data.get('confidence', 0.5),
                        'agent_id': self.agent_id
                    }
                )
                
                # Publish learning event for other agents
                if self.event_bus:
                    await self.event_bus.publish(
                        EventType.PATTERN_DISCOVERED,
                        {
                            "agent_id": self.agent_id,
                            "pattern_type": pattern_type,
                            "specialty": self.specialty,
                            "confidence": pattern_data.get('confidence', 0.5),
                            "pattern_data": pattern_data
                        },
                        source=self.agent_id
                    )
                    
                logger.info(f"Persisted learning pattern: {pattern_type} from {self.specialty}")
                
            except Exception as e:
                logger.error(f"Failed to persist learning pattern: {e}")

    async def recall_relevant_patterns(self, context: str) -> List[Dict]:
        """Recall patterns learned by any agent relevant to current context."""
        if not self.vector_store:
            return []
        
        try:
            # Search for learned patterns using correct method parameters
            results = await self.vector_store.search_similar_memories(
                query=context,
                user_id="system",  # System-level patterns
                limit=5,
                similarity_threshold=0.7,
                content_types=["learned_pattern"]  # Filter for learned patterns
            )
            return results or []
        except Exception as e:
            logger.warning(f"Failed to recall patterns: {e}")
            return []
    
    async def save_working_memory(self):
        """Save compressed working memory with token budget management."""
        if not hasattr(self, '_working_memory'):
            self._working_memory = {
                'last_task': None,
                'reasoning_chain': [],
                'learned_approaches': {},
                'conversation_flow': []
            }
        
        # Update with current interaction (compressed)
        if self.conversation_history:
            recent = self.conversation_history[-1]
            task_preview = recent.get('message_preview', '')
            
            # Compress task preview to fit budget
            max_task_tokens = 50  # Small budget for last task
            if len(task_preview) > max_task_tokens * 4:
                task_preview = task_preview[:max_task_tokens * 4] + "..."
            
            self._working_memory['last_task'] = task_preview
            
            # Keep only recent conversation flow (token-efficient)
            flow_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'task_type': recent.get('specialty', self.specialty),
                'tokens_used': recent.get('tokens_used', 0),
                'success': recent.get('cost', 0) > 0  # Simple success indicator
            }
            # Ensure conversation_flow exists (defensive programming)
            if 'conversation_flow' not in self._working_memory:
                self._working_memory['conversation_flow'] = []
            self._working_memory['conversation_flow'].append(flow_entry)
        
        # Compress memory to fit token budget
        compressed_memory = self._compress_working_memory()
        
        # Save compressed version to vector store
        try:
            await self.persist_learning('working_memory', {
                'agent_id': self.agent_id,
                'specialty': self.specialty,
                'memory_state': compressed_memory,
                'last_updated': datetime.utcnow().isoformat(),
                'memory_size_estimate': len(json.dumps(compressed_memory)) // 4  # Token estimate
            })
        except Exception as e:
            logger.warning(f"Failed to save working memory: {e}")
    
    def _compress_working_memory(self) -> Dict[str, Any]:
        """Compress working memory to fit within token budget."""
        max_working_memory_tokens = self.token_management['max_working_memory_tokens']
        
        # Keep only essential recent data
        # Ensure conversation_flow exists (defensive programming)
        if 'conversation_flow' not in self._working_memory:
            self._working_memory['conversation_flow'] = []
            
        compressed = {
            'last_task': self._working_memory.get('last_task', '')[:200],  # Limit to 50 tokens
            'recent_patterns_count': len(self._working_memory.get('learned_approaches', {})),
            'conversation_flow': self._working_memory.get('conversation_flow', [])[-3:],  # Last 3 only
            'key_insights': self._extract_key_insights(),
            'performance_summary': self._get_performance_summary()
        }
        
        # Estimate and trim if needed
        estimated_tokens = len(json.dumps(compressed)) // 4
        if estimated_tokens > max_working_memory_tokens:
            # Further compression - keep only most essential
            compressed = {
                'last_task': compressed['last_task'][:100],
                'recent_patterns_count': compressed['recent_patterns_count'],
                'last_interaction': compressed['conversation_flow'][-1] if compressed['conversation_flow'] else {},
                'performance_summary': compressed['performance_summary']
            }
        
        return compressed
    
    def _extract_key_insights(self) -> List[str]:
        """Extract key insights from working memory (token-efficient)."""
        insights = []
        
        # Add performance insights
        if self.performance_metrics['total_interactions'] > 0:
            success_rate = self.performance_metrics['successful_interactions'] / self.performance_metrics['total_interactions']
            if success_rate < 0.8:
                insights.append("low_success_rate")
            if self.performance_metrics['total_cost'] > 0.1:  # Arbitrary threshold
                insights.append("high_cost_usage")
        
        # Add recent pattern insights
        if len(self._working_memory.get('learned_approaches', {})) > 5:
            insights.append("rich_pattern_knowledge")
        
        return insights[:3]  # Limit to 3 insights
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get compressed performance summary."""
        return {
            'total_interactions': self.performance_metrics['total_interactions'],
            'success_rate': round(
                self.performance_metrics['successful_interactions'] / max(1, self.performance_metrics['total_interactions']), 2
            ),
            'avg_cost': round(
                self.performance_metrics['total_cost'] / max(1, self.performance_metrics['total_interactions']), 4
            )
        }
    
    async def restore_working_memory(self):
        """Restore my previous reasoning state."""
        memories = await self.recall_relevant_patterns(
            f"working_memory agent_id:{self.agent_id}"
        )
        if memories and len(memories) > 0:
            try:
                memory_content = json.loads(memories[0].get('content', '{}'))
                self._working_memory = memory_content.get('memory_state', {})
                logger.info(f"Restored working memory for {self.specialty}")
            except:
                self._working_memory = {}
    
    async def adapt_to_task(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the task and adapt parameters dynamically for optimal performance."""
        # Circuit breaker to prevent infinite loops
        if hasattr(self, '_adaptive_failure_count'):
            self._adaptive_failure_count += 1
            if self._adaptive_failure_count > 2:
                logger.warning(f"Too many adaptive failures ({self._adaptive_failure_count}), using standard processing")
                return await self._generate_fallback_response(message, context)
        else:
            self._adaptive_failure_count = 0
            
        try:
            logger.info(f"Universal Agent '{self.specialty}' adapting to task: '{message[:50]}...'")
            
            # Quick analysis of what this task needs
            task_analysis = await self._analyze_task_requirements(message, context)
            
            # Recall what worked for similar tasks
            similar_patterns = await self.recall_relevant_patterns(
                f"task_parameters: {task_analysis.get('task_type', 'general')}"
            )
            
            # Start with base parameters
            optimal_params = {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'model': self.model_name
            }
            
            # Adjust based on task analysis
            if task_analysis.get('creativity_needed', 0.0) > 0.7:
                optimal_params['temperature'] = min(0.9, task_analysis['creativity_needed'])
            elif task_analysis.get('precision_needed', 0.0) > 0.7:
                optimal_params['temperature'] = 0.1
            
            if task_analysis.get('complexity_score', 0.0) > 0.8:
                optimal_params['max_tokens'] = min(2000, int(task_analysis['complexity_score'] * 2000))
                optimal_params['model'] = 'gpt-4' if task_analysis.get('reasoning_complexity', 0.0) > 0.8 else 'gpt-3.5-turbo'
            
            # Learn from similar patterns
            if similar_patterns:
                pattern_params = self._extract_optimal_params_from_patterns(similar_patterns)
                if pattern_params:
                    # Weight learned patterns
                    optimal_params.update(pattern_params)
            
            # Generate response with optimal parameters
            result = await self._generate_adaptive_response(message, context, optimal_params)
            
            # Calculate quality score (simplified)
            quality_score = self._calculate_response_quality(result, task_analysis)
            
            # Learn from the outcome if successful
            if result.get('success', True) and quality_score > 0.7:
                await self.persist_learning('task_parameters', {
                    'task_type': task_analysis.get('task_type', 'general'),
                    'parameters': optimal_params,
                    'outcome_quality': quality_score,
                    'message_length': len(message),
                    'response_length': len(result.get('response', '')),
                    'specialty': self.specialty,
                    'confidence': 0.8
                })
                logger.info(f"Learned optimal parameters for {task_analysis.get('task_type', 'general')} tasks")
            
            # Publish events for existing improvement systems
            if self.event_bus:
                # Feed data to cost optimizer
                if 'cost' in result:
                    default_cost = self._estimate_default_cost(message, optimal_params)
                    await self.event_bus.publish(
                        EventType.AGENT_OPTIMIZED,
                        {
                            "agent_id": self.agent_id,
                            "optimization_type": "adaptive_parameters",
                            "cost_before": default_cost,
                            "cost_after": result['cost'],
                            "parameters_used": optimal_params,
                            "savings": max(0, default_cost - result['cost']),
                            "task_type": task_analysis.get('task_type', 'general'),
                            "quality_score": quality_score
                        },
                        source=self.agent_id
                    )
                
                # Feed data to workflow analyst and pattern recognition
                if quality_score > 0.7:
                    await self.event_bus.publish(
                        EventType.WORKFLOW_COMPLETED,
                        {
                            "workflow_id": f"adaptive_task_{datetime.utcnow().timestamp()}",
                            "agent_id": self.agent_id,
                            "pattern_data": {
                                "task_type": task_analysis.get('task_type', 'general'),
                                "parameters": optimal_params,
                                "quality_score": quality_score,
                                "adaptive_success": True
                            },
                            "performance_metrics": {
                                "tokens_used": result.get('tokens_used', 0),
                                "cost": result.get('cost', 0.0),
                                "model_used": result.get('model_used', self.model_name)
                            }
                        },
                        source=self.agent_id
                    )
            
            # Reset failure count on success
            self._adaptive_failure_count = 0
            
            # Add adaptation metadata
            result['adaptation_used'] = True
            result['optimal_params'] = optimal_params
            result['task_analysis'] = task_analysis
            result['quality_score'] = quality_score
            
            return result
            
        except Exception as e:
            logger.error(f"Error in adaptive task processing: {e}")
            
            # Learn from this failure
            await self.persist_learning('task_failure', {
                'error': str(e),
                'error_type': type(e).__name__,
                'task_preview': message[:200],
                'attempted_params': optimal_params if 'optimal_params' in locals() else {},
                'specialty': self.specialty,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Use fallback instead of recursive call
            return await self._generate_fallback_response(message, context)

    async def _generate_fallback_response(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a simple fallback response when adaptive processing fails."""
        try:
            # Use basic specialist response without adaptive features
            memory_context = await self._get_memory_context(message, context)
            history_context = self._format_conversation_history(context.get("conversation_history", []))
            tool_results = await self._execute_tools(message, context)
            
            response = await self._generate_specialist_response(
                message, context, history_context, memory_context, tool_results
            )
            
            return {
                'response': response,
                'success': True,
                'adaptation_used': False,
                'fallback_used': True,
                'tokens_used': 0,  # Would need tracking for accurate count
                'cost': 0.0
            }
            
        except Exception as e:
            logger.error(f"Fallback response generation failed: {e}")
            return {
                'response': f"I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                'success': False,
                'error': str(e),
                'fallback_used': True
            }

    async def _analyze_task_requirements(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task requirements and complexity with improved error handling."""
        try:
            analysis_prompt = f"""
            Analyze this user request and determine optimal approach:
            
            Request: "{message}"
            
            Consider:
            1. Complexity level (simple/medium/complex)
            2. Required expertise domains
            3. Time sensitivity
            4. Resource requirements
            
            Respond with JSON only:
            {{
                "task_type": "general",
                "complexity": "simple|medium|complex",
                "domains": ["domain1", "domain2"],
                "time_sensitive": true/false,
                "estimated_time": "5m|30m|2h",
                "requires_tools": true/false
            }}
            """
            
            # Use invoke instead of agenerate for better error handling
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a task analysis expert. Respond only with valid JSON."),
                ("user", analysis_prompt)
            ])
            
            chain = prompt | self.llm
            response = await chain.ainvoke({})
            
            # Handle different response types
            response_text = ""
            if hasattr(response, 'content'):
                response_text = response.content.strip()
            elif hasattr(response, 'text'):
                response_text = response.text.strip()
            elif isinstance(response, str):
                response_text = response.strip()
            else:
                logger.warning(f"Unexpected response type: {type(response)}")
                return self._get_default_task_analysis()
            
            # Handle empty or invalid responses
            if not response_text:
                logger.warning("Empty response from task analysis LLM")
                return self._get_default_task_analysis()
            
            # Try to extract JSON from response
            try:
                # Look for JSON block if wrapped in markdown
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    if json_end > json_start:
                        response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.rfind("```")
                    if json_end > json_start:
                        response_text = response_text[json_start:json_end].strip()
                
                analysis = json.loads(response_text)
                
                # Validate and add missing required fields
                if not isinstance(analysis, dict):
                    logger.warning("Task analysis response is not a dictionary")
                    return self._get_default_task_analysis()
                
                # Ensure required fields exist
                analysis.setdefault('task_type', 'general')
                analysis.setdefault('complexity', 'medium')
                analysis.setdefault('domains', ['general'])
                
                return analysis
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse task analysis JSON: {e}")
                logger.debug(f"Raw response: {response_text}")
                return self._get_default_task_analysis()
                
        except Exception as e:
            logger.warning(f"Task analysis failed, using defaults: {e}")
            return self._get_default_task_analysis()
    
    def _get_default_task_analysis(self) -> Dict[str, Any]:
        """Return default task analysis when parsing fails."""
        return {
            "task_type": "general",
            "complexity": "medium",
            "domains": ["general"],
            "time_sensitive": False,
            "estimated_time": "15m",
            "requires_tools": False
        }
    
    def _extract_optimal_params_from_patterns(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Extract optimal parameters from learned patterns."""
        if not patterns:
            return {}
        
        # Weight patterns by quality and recency
        weighted_params = {'temperature': [], 'max_tokens': [], 'model': []}
        
        for pattern in patterns:
            try:
                pattern_data = json.loads(pattern.get('content', '{}'))
                if 'parameters' in pattern_data and 'outcome_quality' in pattern_data:
                    quality_weight = pattern_data['outcome_quality']
                    params = pattern_data['parameters']
                    
                    if 'temperature' in params:
                        weighted_params['temperature'].append((params['temperature'], quality_weight))
                    if 'max_tokens' in params:
                        weighted_params['max_tokens'].append((params['max_tokens'], quality_weight))
                    if 'model' in params:
                        weighted_params['model'].append((params['model'], quality_weight))
                        
            except Exception as e:
                logger.warning(f"Error extracting pattern params: {e}")
                continue
        
        # Calculate weighted averages
        optimal = {}
        
        if weighted_params['temperature']:
            total_weight = sum(weight for _, weight in weighted_params['temperature'])
            if total_weight > 0:
                optimal['temperature'] = sum(temp * weight for temp, weight in weighted_params['temperature']) / total_weight
        
        if weighted_params['max_tokens']:
            total_weight = sum(weight for _, weight in weighted_params['max_tokens'])
            if total_weight > 0:
                optimal['max_tokens'] = int(sum(tokens * weight for tokens, weight in weighted_params['max_tokens']) / total_weight)
        
        if weighted_params['model']:
            # Use most frequently successful model
            model_scores = {}
            for model, weight in weighted_params['model']:
                model_scores[model] = model_scores.get(model, 0) + weight
            if model_scores:
                optimal['model'] = max(model_scores.items(), key=lambda x: x[1])[0]
        
        return optimal
    
    async def _generate_adaptive_response(self, message: str, context: Dict[str, Any], 
                                        optimal_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using dynamically optimized parameters."""
        try:
            # Create temporary LLM with optimal parameters
            adaptive_llm = ChatOpenAI(
                model=optimal_params.get('model', self.model_name),
                temperature=optimal_params.get('temperature', self.temperature),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=optimal_params.get('max_tokens', self.max_tokens),
            )
            
            # Use the same context gathering as standard processing
            memory_context = await self._get_memory_context(message, context)
            history_context = self._format_conversation_history(context.get("conversation_history", []))
            tool_results = await self._execute_tools(message, context)
            
            # Generate response with adaptive LLM
            main_chain = self.main_prompt | adaptive_llm
            
            enhanced_context = self._format_context(context)
            if tool_results:
                enhanced_context += f"\nTool Results: {json.dumps(tool_results, indent=2)}"
            
            with get_openai_callback() as cb:
                response = await main_chain.ainvoke({
                    "message": message,
                    "context": enhanced_context,
                    "history": history_context,
                    "memory_context": memory_context
                })
                tokens_used = cb.total_tokens
                cost = cb.total_cost
            
            return {
                'response': response.content,
                'success': True,
                'tokens_used': tokens_used,
                'cost': cost,
                'model_used': optimal_params.get('model', self.model_name),
                'parameters_used': optimal_params
            }
            
        except Exception as e:
            logger.error(f"Adaptive response generation failed: {e}")
            return {
                'response': f"I apologize, but I encountered an issue while adapting to your {self.specialty} request.",
                'success': False,
                'error': str(e)
            }
    
    def _calculate_response_quality(self, result: Dict[str, Any], task_analysis: Dict[str, Any]) -> float:
        """Calculate quality score for the response (simplified heuristic)."""
        try:
            if not result.get('success', False):
                return 0.0
            
            response = result.get('response', '')
            
            # Basic quality heuristics
            quality_score = 0.5  # Base score
            
            # Length appropriateness
            expected_length = task_analysis.get('expected_length', 'medium')
            response_length = len(response)
            
            if expected_length == 'short' and 50 <= response_length <= 300:
                quality_score += 0.2
            elif expected_length == 'medium' and 200 <= response_length <= 800:
                quality_score += 0.2
            elif expected_length == 'long' and response_length >= 500:
                quality_score += 0.2
            
            # Task type specific checks
            task_type = task_analysis.get('task_type', 'general')
            
            if task_type == 'code_generation' and any(marker in response for marker in ['```', 'def ', 'function', 'class ']):
                quality_score += 0.2
            elif task_type == 'analysis' and any(word in response.lower() for word in ['because', 'therefore', 'analysis', 'conclusion']):
                quality_score += 0.2
            elif task_type == 'creative_writing' and response_length > 200:
                quality_score += 0.2
            
            # Cost efficiency
            tokens_used = result.get('tokens_used', 0)
            if tokens_used > 0 and tokens_used < 1000:  # Efficient token usage
                quality_score += 0.1
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.warning(f"Quality calculation failed: {e}")
            return 0.5
    
    def _estimate_default_cost(self, message: str, optimal_params: Dict[str, Any]) -> float:
        """Estimate what the cost would have been with default parameters."""
        try:
            # Rough estimation based on message length and default model
            message_tokens = len(message.split()) * 1.3  # Rough token estimation
            default_tokens = min(self.max_tokens, message_tokens + 200)  # Default response estimate
            
            # Cost per token for default model (rough estimates)
            if self.model_name.startswith('gpt-4'):
                cost_per_token = 0.00003  # $0.03 per 1K tokens
            else:
                cost_per_token = 0.000002  # $0.002 per 1K tokens
            
            return default_tokens * cost_per_token
            
        except Exception as e:
            logger.warning(f"Cost estimation failed: {e}")
            return 0.01  # Default fallback cost
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about this agent instance."""
        success_rate = 0.0
        if self.performance_metrics["total_interactions"] > 0:
            success_rate = self.performance_metrics["successful_interactions"] / self.performance_metrics["total_interactions"]
        
        # Calculate token efficiency metrics
        avg_tokens_per_interaction = 0.0
        avg_cost_per_interaction = 0.0
        if self.performance_metrics["total_interactions"] > 0:
            avg_tokens_per_interaction = self.performance_metrics["total_tokens"] / self.performance_metrics["total_interactions"]
            avg_cost_per_interaction = self.performance_metrics["total_cost"] / self.performance_metrics["total_interactions"]
        
        base_stats = {
            "agent_id": self.agent_id,
            "specialty": self.specialty,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "total_interactions": self.performance_metrics["total_interactions"],
            "success_rate": success_rate,
            "total_tokens_used": self.performance_metrics["total_tokens"],
            "total_cost": self.performance_metrics["total_cost"],
            "average_response_time_ms": self.performance_metrics["average_response_time"],
            "cost_per_interaction": avg_cost_per_interaction,
            "tokens_per_interaction": avg_tokens_per_interaction,
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
        
        # Add token optimization stats if enabled
        if self.token_management['enable_token_optimization']:
            base_stats["token_optimization"] = self.get_token_optimization_stats()
        
        return base_stats
    
    def get_token_optimization_stats(self) -> Dict[str, Any]:
        """Get detailed token optimization and cost efficiency statistics."""
        stats = {
            "optimization_enabled": self.token_management['enable_token_optimization'],
            "token_budgets": {
                "memory_context": self.token_management['max_memory_tokens'],
                "working_memory": self.token_management['max_working_memory_tokens'],
                "history": self.token_management['max_history_tokens'],
                "tool_results": self.token_management['max_tool_results_tokens']
            },
            "search_parameters": {
                "similarity_threshold": self.token_management['similarity_threshold'],
                "max_memories_loaded": self.token_management['max_memories_to_load']
            },
            "features": {
                "context_compression": self.token_management['context_compression'],
                "adaptive_budget": self.token_management['adaptive_budget']
            }
        }
        
        # Add cost efficiency metrics if available
        if hasattr(self, 'cost_efficiency_history') and self.cost_efficiency_history:
            avg_cost_per_token = sum(self.cost_efficiency_history) / len(self.cost_efficiency_history)
            stats["cost_efficiency"] = {
                "avg_cost_per_token": round(avg_cost_per_token, 6),
                "recent_measurements": len(self.cost_efficiency_history),
                "efficiency_trend": "improving" if len(self.cost_efficiency_history) >= 2 and 
                                  self.cost_efficiency_history[-1] < self.cost_efficiency_history[0] else "stable"
            }
        
        # Estimate potential savings
        if self.performance_metrics["total_interactions"] > 0:
            estimated_tokens_without_optimization = self.performance_metrics["total_tokens"] * 2.5  # Rough estimate
            estimated_savings = max(0, estimated_tokens_without_optimization - self.performance_metrics["total_tokens"])
            stats["estimated_savings"] = {
                "tokens_saved": int(estimated_savings),
                "cost_saved": round(estimated_savings * 0.000002, 4),  # Rough cost per token
                "efficiency_gain": f"{round((estimated_savings / estimated_tokens_without_optimization) * 100, 1)}%" if estimated_tokens_without_optimization > 0 else "0%"
            }
        
        return stats
    
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
        
        # Handle token optimization settings
        if "token_optimization" in kwargs:
            self.update_token_optimization_settings(kwargs["token_optimization"])
        
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
    
    def update_token_optimization_settings(self, settings: Dict[str, Any]):
        """
        Update token optimization settings.
        
        Args:
            settings: Dictionary of token management settings to update
        """
        updatable_settings = [
            'max_memory_tokens', 'max_working_memory_tokens', 'max_history_tokens',
            'max_tool_results_tokens', 'similarity_threshold', 'max_memories_to_load',
            'enable_token_optimization', 'context_compression', 'adaptive_budget'
        ]
        
        for setting, value in settings.items():
            if setting in updatable_settings:
                self.token_management[setting] = value
                logger.info(f"Updated token optimization setting '{setting}' to {value}")
            else:
                logger.warning(f"Unknown token optimization setting: {setting}")
        
        # Log current optimization status
        if self.token_management['enable_token_optimization']:
            logger.info(f"Token optimization ENABLED for {self.specialty}")
            logger.info(f"  Memory budget: {self.token_management['max_memory_tokens']} tokens")
            logger.info(f"  Similarity threshold: {self.token_management['similarity_threshold']}")
        else:
            logger.info(f"Token optimization DISABLED for {self.specialty}")
    
    def enable_token_optimization(self, 
                                 memory_budget: int = 500,
                                 similarity_threshold: float = 0.8,
                                 aggressive_mode: bool = False):
        """
        Enable token optimization with specified settings.
        
        Args:
            memory_budget: Maximum tokens for memory context
            similarity_threshold: Minimum similarity for memory retrieval
            aggressive_mode: If True, use more aggressive optimization
        """
        if aggressive_mode:
            settings = {
                'enable_token_optimization': True,
                'max_memory_tokens': min(300, memory_budget),
                'max_working_memory_tokens': 200,
                'max_history_tokens': 150,
                'max_tool_results_tokens': 150,
                'similarity_threshold': max(0.85, similarity_threshold),
                'max_memories_to_load': 1,
                'context_compression': True,
                'adaptive_budget': True
            }
        else:
            settings = {
                'enable_token_optimization': True,
                'max_memory_tokens': memory_budget,
                'max_working_memory_tokens': 300,
                'max_history_tokens': 200,
                'max_tool_results_tokens': 200,
                'similarity_threshold': similarity_threshold,
                'max_memories_to_load': 2,
                'context_compression': True,
                'adaptive_budget': True
            }
        
        self.update_token_optimization_settings(settings)
        logger.info(f"🎯 Token optimization {'AGGRESSIVE' if aggressive_mode else 'STANDARD'} mode enabled for {self.specialty}")
    
    def disable_token_optimization(self):
        """Disable token optimization - use full context."""
        self.token_management['enable_token_optimization'] = False
        logger.info(f"❌ Token optimization disabled for {self.specialty} - using full context")
    
    def get_cost_projection(self, estimated_daily_requests: int) -> Dict[str, Any]:
        """
        Project daily/monthly costs based on current usage patterns.
        
        Args:
            estimated_daily_requests: Expected number of requests per day
            
        Returns:
            Cost projection with optimization impact
        """
        if self.performance_metrics["total_interactions"] == 0:
            return {"error": "No interaction data available for projection"}
        
        avg_cost_per_request = self.performance_metrics["total_cost"] / self.performance_metrics["total_interactions"]
        avg_tokens_per_request = self.performance_metrics["total_tokens"] / self.performance_metrics["total_interactions"]
        
        # Current projections
        daily_cost = avg_cost_per_request * estimated_daily_requests
        monthly_cost = daily_cost * 30
        
        # Estimate cost without optimization (rough 2.5x multiplier)
        unoptimized_cost_per_request = avg_cost_per_request * 2.5
        unoptimized_daily_cost = unoptimized_cost_per_request * estimated_daily_requests
        unoptimized_monthly_cost = unoptimized_daily_cost * 30
        
        return {
            "agent_specialty": self.specialty,
            "optimization_enabled": self.token_management['enable_token_optimization'],
            "current_usage": {
                "avg_cost_per_request": round(avg_cost_per_request, 4),
                "avg_tokens_per_request": round(avg_tokens_per_request, 1),
                "daily_cost": round(daily_cost, 2),
                "monthly_cost": round(monthly_cost, 2)
            },
            "without_optimization": {
                "estimated_daily_cost": round(unoptimized_daily_cost, 2),
                "estimated_monthly_cost": round(unoptimized_monthly_cost, 2)
            },
            "savings": {
                "daily_savings": round(unoptimized_daily_cost - daily_cost, 2),
                "monthly_savings": round(unoptimized_monthly_cost - monthly_cost, 2),
                "cost_reduction_percentage": round(((unoptimized_monthly_cost - monthly_cost) / unoptimized_monthly_cost) * 100, 1) if unoptimized_monthly_cost > 0 else 0
            },
            "projection_based_on": f"{self.performance_metrics['total_interactions']} interactions"
        }
    
    async def check_mcp_tool_requests_status(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Check status of MCP tool creation requests for this user.
        
        This method can be called by Slack interface or other UIs
        to show users what MCP tool requests need their attention.
        """
        if not self.dynamic_tool_builder:
            return None
        
        collaboration_request = await self.dynamic_tool_builder.get_user_collaboration_request(user_id)
        
        if collaboration_request:
            # Find the corresponding pending task
            pending_task = None
            for task in self.pending_mcp_tasks:
                if task['request_id'] == collaboration_request['request_id']:
                    pending_task = task
                    break
            
            if pending_task:
                collaboration_request['original_task'] = {
                    'message': pending_task['message'],
                    'agent': self.specialty,
                    'created_at': pending_task['created_at'].isoformat(),
                    'mcp_solutions_available': pending_task.get('mcp_solutions_count', 0) > 0
                }
        
        return collaboration_request
    
    async def handle_mcp_tool_ready(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Handle notification that a requested MCP tool is ready.
        
        This completes any pending tasks that were waiting for the MCP tool.
        """
        try:
            # Find pending task for this request
            pending_task = None
            for i, task in enumerate(self.pending_mcp_tasks):
                if task['request_id'] == request_id:
                    pending_task = self.pending_mcp_tasks.pop(i)
                    break
            
            if not pending_task:
                logger.warning(f"No pending MCP task found for request {request_id}")
                return None
            
            # Reload MCP tools to include the new tool
            if self.mcp_tool_registry:
                await self._reload_available_mcp_tools()
            
            # Retry the original task with the new MCP tool
            logger.info(f"🔄 Retrying original task with new MCP tool: {pending_task['message'][:50]}...")
            
            result = await self.process_message(
                pending_task['message'],
                pending_task['context']
            )
            
            # Enhanced response indicating MCP tool was used
            if result.get('confidence', 0) >= 0.7:
                result['mcp_tool_creation_success'] = True
                result['original_request_id'] = request_id
                result['response'] = f"✅ **Task completed with new MCP tool!**\n\n{result['response']}\n\n*I successfully set up and used a new MCP tool to complete your original request.*"
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling MCP tool ready notification: {e}")
            return None
    
    async def _reload_available_mcp_tools(self):
        """Reload available MCP tools from the registry."""
        try:
            if self.mcp_tool_registry:
                # Get updated MCP tool list
                available_tools = await self.mcp_tool_registry.get_available_tools()
                
                # Convert to ToolCapability objects and add to agent
                for tool_info in available_tools:
                    tool_capability = ToolCapability(
                        name=tool_info['tool_name'],
                        description=tool_info['description'],
                        function=self._create_mcp_tool_function(tool_info['tool_id']),
                        enabled=True
                    )
                    
                    # Add if not already present
                    if not any(t.name == tool_capability.name for t in self.tools):
                        self.tools.append(tool_capability)
                        logger.info(f"📦 Added new MCP tool: {tool_capability.name}")
                        
        except Exception as e:
            logger.error(f"Error reloading MCP tools: {e}")
    
    def _create_mcp_tool_function(self, tool_id: str) -> callable:
        """Create a function wrapper for MCP tool execution."""
        async def mcp_tool_wrapper(message: str, context: Dict[str, Any]) -> Dict[str, Any]:
            try:
                if self.mcp_tool_registry:
                    # Extract parameters from message (simplified)
                    parameters = {"message": message, "context": context}
                    return await self.mcp_tool_registry.execute_tool(tool_id, parameters)
                else:
                    return {"error": "MCP tool registry not available"}
            except Exception as e:
                return {"error": f"MCP tool execution failed: {str(e)}"}
        
        return mcp_tool_wrapper
    
    async def get_pending_mcp_tasks_count(self) -> int:
        """Get count of pending tasks waiting for MCP tools."""
        return len(self.pending_mcp_tasks)
    
    async def get_mcp_requests_summary(self) -> Dict[str, Any]:
        """Get summary of MCP tool requests for monitoring."""
        return {
            "active_mcp_requests": len(self.mcp_tool_requests),
            "pending_mcp_tasks": len(self.pending_mcp_tasks),
            "capabilities_requested": [
                self.dynamic_tool_builder.active_gaps[gap_id].capability_needed
                for gap_id in self.mcp_tool_requests.keys()
                if self.dynamic_tool_builder and gap_id in self.dynamic_tool_builder.active_gaps
            ],
            "recent_mcp_requests": [
                {
                    "capability": self.dynamic_tool_builder.active_gaps[task['gap_id']].capability_needed if self.dynamic_tool_builder and task['gap_id'] in self.dynamic_tool_builder.active_gaps else "unknown",
                    "created_at": task['created_at'].isoformat(),
                    "mcp_solutions_available": task.get('mcp_solutions_count', 0) > 0,
                    "status": self.dynamic_tool_builder.active_requests.get(task['request_id'], {}).status.value if self.dynamic_tool_builder and hasattr(self.dynamic_tool_builder.active_requests.get(task['request_id'], {}), 'status') else 'unknown'
                }
                for task in self.pending_mcp_tasks[-5:]  # Last 5 requests
            ]
        }
    
    async def _save_pending_mcp_tasks(self):
        """Save pending MCP tasks to vector store for recovery."""
        try:
            if self.pending_mcp_tasks:
                tasks_data = {
                    'agent_id': self.agent_id,
                    'specialty': self.specialty,
                    'pending_tasks': self.pending_mcp_tasks,
                    'saved_at': datetime.utcnow().isoformat()
                }
                
                # Store in vector memory for recovery
                await self.persist_learning('pending_mcp_tasks', tasks_data)
                logger.info(f"💾 Saved {len(self.pending_mcp_tasks)} pending MCP tasks for {self.specialty}")
                
        except Exception as e:
            logger.error(f"Error saving pending MCP tasks: {e}")

    async def close(self):
        """Close the agent and cleanup platform resources."""
        try:
            # Save working memory and MCP state before closing to prevent state loss
            await self.save_working_memory()
            
            # Save pending MCP tasks for later recovery
            if self.pending_mcp_tasks:
                await self._save_pending_mcp_tasks()
            
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
            
            # Close MCP integrations
            if self.mcp_tool_registry:
                try:
                    await self.mcp_tool_registry.close()
                except Exception as e:
                    logger.warning(f"Error closing MCP tool registry: {e}")
            
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