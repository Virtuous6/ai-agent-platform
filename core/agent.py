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
import time
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

# MCP integrations using official SDK
# from mcp import mcp_registry, find_mcp_tools

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
                 # MCP integrations (official SDK)
                 mcp_registry = None):
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
            
            # MCP integrations (Model Context Protocol) - Official SDK
            mcp_registry: Official MCP registry for tool discovery and execution
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
        
        # MCP integration using official SDK
        self.mcp_registry = mcp_registry
        
        # Track MCP tool usage
        self.mcp_tool_usage: Dict[str, int] = {}  # tool_name -> usage_count
        self.last_mcp_sync: Optional[datetime] = None
        
        # Tool execution improvements
        self.tool_cache = {}  # Cache for tool results
        self.cache_ttl = 3600  # 1 hour cache TTL
        
        # Context persistence for tools
        self.conversation_context = {
            'tool_results_history': [],  # All previous tool results
            'user_preferences': {},      # Learned user preferences
            'session_data': {},          # Current session context
            'previous_queries': [],      # Recent queries for context
            'successful_patterns': [],   # Patterns that worked well
            'failed_patterns': []        # Patterns that failed (to avoid)
        }
        
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
        
        # Tool registry
        self._tool_registry = None
        self._tools_initialized = False
        
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
        
        mcp_status = "with official MCP SDK" if self.mcp_registry else "without MCP integration"
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
        Process a user message with layered approach: simple → direct, complex → specialist.
        
        Args:
            message: User message content
            context: Conversation context
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = datetime.utcnow()
        run_id = str(uuid.uuid4())
        
        try:
            # Initialize tool registry if not done
            await self._ensure_tools_initialized()
            
            # Restore working memory
            await self.restore_working_memory()
            
            logger.info(f"Universal Agent '{self.specialty}' processing: '{message[:50]}...'")
            
            # 🚀 STEP 1: Detect intent and complexity
            intent = await self._detect_intent(message, context)
            context['intent'] = intent
            
            # 🎯 STEP 2: Route based on complexity
            if intent['complexity'] == 'simple':
                logger.info(f"✅ Simple intent detected: {intent['type']} - handling directly")
                return await self._handle_simple_intent(message, context, intent, run_id, start_time)
            else:
                logger.info(f"🔄 Complex intent detected: {intent['type']} - spawning specialist")
                return await self._spawn_specialist_for_complex_task(message, context, intent, run_id, start_time)
            
        except Exception as e:
            logger.error(f"Error in Universal Agent '{self.specialty}' message processing: {str(e)}")
            
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
            
            return {
                "response": f"I apologize, but I encountered an error processing your request. Please try rephrasing or try again.",
                "agent_id": self.agent_id,
                "specialty": self.specialty,
                "conversation_type": "error",
                "confidence": 0.0,
                "error": str(e),
                "tokens_used": 0,
                "processing_time_ms": processing_time_ms,
                "run_id": run_id,
                "routing": "error"
            }
    
    async def _ensure_tools_initialized(self):
        """Ensure tool registry is initialized."""
        if not self._tools_initialized:
            try:
                from core.universal_mcp_tools import get_tool_registry
                self._tool_registry = get_tool_registry()
                await self._tool_registry.initialize()
                self._tools_initialized = True
                logger.info(f"Tool registry initialized for {self.agent_id}")
            except Exception as e:
                logger.error(f"Failed to initialize tool registry: {e}")
                self._tool_registry = None

    async def _find_mcp_tools_for_capability(self, capability: str) -> List[Dict[str, Any]]:
        """
        Find MCP tools that provide a specific capability using the official SDK.
        
        Args:
            capability: Description of needed capability
            
        Returns:
            List of matching MCP tools
        """
        try:
            if not self.mcp_registry:
                return []
            
            # Use official MCP registry to find tools
            from mcp import find_mcp_tools
            tools = await find_mcp_tools(capability)
            
            logger.info(f"🔍 Found {len(tools)} MCP tools for capability: {capability}")
            return tools
            
        except Exception as e:
            logger.error(f"Error finding MCP tools for capability '{capability}': {e}")
            return []
    
    async def _use_mcp_tool(self, capability: str, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use an MCP tool to handle a capability need.
        
        Args:
            capability: The capability needed
            message: User message
            context: Request context
            
        Returns:
            Result from MCP tool execution
        """
        try:
            # Find suitable MCP tools
            tools = await self._find_mcp_tools_for_capability(capability)
            
            if not tools:
                return {
                    'mcp_tools_found': 0,
                    'message': f"No MCP tools found for capability: {capability}"
                }
            
            # Use the best tool (first in list, sorted by usage)
            best_tool = tools[0]
            tool_key = f"{best_tool['server']}:{best_tool['name']}"
            
            logger.info(f"🔧 Using MCP tool: {tool_key}")
            
            # Prepare arguments (simplified)
            arguments = {
                "query": message,
                "context": capability
            }
            
            # Execute the tool via official MCP registry
            if self.mcp_registry:
                result = await self.mcp_registry.call_tool(tool_key, arguments)
                
                # Track usage
                self.mcp_tool_usage[tool_key] = self.mcp_tool_usage.get(tool_key, 0) + 1
                
                return {
                    'mcp_tools_found': len(tools),
                    'tool_used': tool_key,
                    'result': result,
                    'success': True
                }
            else:
                return {
                    'mcp_tools_found': len(tools),
                    'error': 'MCP registry not available'
                }
                
        except Exception as e:
            logger.error(f"Error using MCP tool for capability '{capability}': {e}")
            return {
                'mcp_tools_found': 0,
                'error': str(e)
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

    async def _detect_intent(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect user intent and determine if it needs simple or complex handling.
        
        Returns:
            Dict with intent type, complexity, and routing info
        """
        try:
            message_lower = message.lower()
            
            # Simple intents - can be handled directly
            simple_patterns = {
                'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
                'thanks': ['thank', 'thanks', 'appreciate'],
                'help': ['help', 'what can you do', 'how do you work'],
                'status': ['status', 'how are you', 'are you working'],
                'simple_question': len(message.split()) < 10 and '?' in message
            }
            
            # Complex intents - need specialist agents
            complex_patterns = {
                'analysis': ['analyze', 'examine', 'study', 'research', 'investigate'],
                'integration': ['integrate', 'connect', 'setup', 'configure', 'implement'],
                'optimization': ['optimize', 'improve', 'enhance', 'performance', 'speed up'],
                'troubleshooting': ['error', 'problem', 'issue', 'bug', 'not working', 'failed'],
                'multi_step': message.count('?') > 1 or message.count('and') > 2,
                'code_generation': ['code', 'script', 'function', 'algorithm', 'program'],
                'architecture': ['architecture', 'design', 'system', 'structure', 'framework']
            }
            
            # Check for simple intents first
            for intent_type, patterns in simple_patterns.items():
                if intent_type == 'simple_question':
                    if patterns:  # It's a boolean, check if True
                        return {
                            'type': intent_type,
                            'complexity': 'simple',
                            'confidence': 0.8,
                            'reasoning': 'Short question'
                        }
                else:
                    if any(pattern in message_lower for pattern in patterns):
                        return {
                            'type': intent_type,
                            'complexity': 'simple',
                            'confidence': 0.9,
                            'reasoning': f'Matched {intent_type} pattern'
                        }
            
            # Check for complex intents
            for intent_type, patterns in complex_patterns.items():
                if intent_type in ['multi_step']:
                    if patterns:  # It's a boolean, check if True
                        return {
                            'type': intent_type,
                            'complexity': 'complex',
                            'confidence': 0.8,
                            'reasoning': 'Multiple questions or complex structure',
                            'specialist_needed': self._get_specialist_type(intent_type)
                        }
                else:
                    if any(pattern in message_lower for pattern in patterns):
                        return {
                            'type': intent_type,
                            'complexity': 'complex',
                            'confidence': 0.8,
                            'reasoning': f'Matched {intent_type} pattern',
                            'specialist_needed': self._get_specialist_type(intent_type)
                        }
            
            # Default: medium complexity for current agent
            return {
                'type': 'general',
                'complexity': 'simple' if len(message) < 100 else 'complex',
                'confidence': 0.6,
                'reasoning': 'Default classification',
                'specialist_needed': self.specialty
            }
            
        except Exception as e:
            logger.warning(f"Intent detection failed: {e}")
            return {
                'type': 'general',
                'complexity': 'simple',
                'confidence': 0.5,
                'reasoning': 'Fallback due to error'
            }

    def _get_specialist_type(self, intent_type: str) -> str:
        """Map intent types to specialist agent types."""
        specialist_mapping = {
            'analysis': 'Data Analysis Specialist',
            'integration': 'System Integration Specialist', 
            'optimization': 'Performance Optimization Specialist',
            'troubleshooting': 'Technical Support Specialist',
            'code_generation': 'Python Development Specialist',
            'architecture': 'System Architecture Specialist',
            'multi_step': 'Research Specialist'
        }
        return specialist_mapping.get(intent_type, 'General Specialist')

    async def _handle_simple_intent(self, message: str, context: Dict[str, Any], 
                                   intent: Dict[str, Any], run_id: str, start_time: datetime) -> Dict[str, Any]:
        """
        Handle simple intents directly without spawning specialists.
        """
        try:
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Handle specific simple intents
            intent_type = intent['type']
            
            if intent_type == 'greeting':
                response = f"Hello! I'm your {self.specialty} assistant. How can I help you today?"
            elif intent_type == 'thanks':
                response = "You're welcome! Feel free to ask if you need anything else."
            elif intent_type == 'help':
                response = f"I'm a {self.specialty} specialist. I can help with questions in my area of expertise. For complex tasks, I'll connect you with the right specialist agent."
            elif intent_type == 'status':
                response = f"I'm working well! Ready to help with {self.specialty} tasks."
            elif intent_type == 'simple_question':
                # Use lightweight LLM call for simple questions
                response = await self._generate_simple_response(message, context)
            else:
                # General simple response
                response = await self._generate_simple_response(message, context)
            
            # Log simple interaction
            conversation_id = context.get("conversation_id")
            message_id = ""
            if conversation_id and self.supabase_logger:
                message_id = await self._log_to_supabase(
                    conversation_id, message, response, context, 
                    0, 0, 0, 0.0, processing_time_ms  # Minimal tokens/cost for simple responses
                )
            
            return {
                "response": response,
                "agent_id": self.agent_id,
                "specialty": self.specialty,
                "conversation_type": "simple_direct",
                "confidence": intent['confidence'],
                "intent": intent,
                "tokens_used": 0,
                "processing_time_ms": processing_time_ms,
                "run_id": run_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "routing": "direct_response"
            }
            
        except Exception as e:
            logger.error(f"Error handling simple intent: {e}")
            return {
                "response": f"I apologize, but I had trouble with that simple request. Please try again.",
                "error": str(e),
                "intent": intent,
                "routing": "direct_response_failed"
            }

    async def _spawn_specialist_for_complex_task(self, message: str, context: Dict[str, Any], 
                                               intent: Dict[str, Any], run_id: str, start_time: datetime) -> Dict[str, Any]:
        """
        Spawn a specialist agent for complex tasks that need ReAct patterns.
        """
        try:
            specialist_type = intent.get('specialist_needed', 'General Specialist')
            
            # For now, handle in current agent but mark as specialist work
            # TODO: Actually spawn specialist agents when orchestrator is ready
            logger.info(f"🔄 Complex task needs {specialist_type} - handling with enhanced processing")
            
            # Use the existing complex processing but mark it as specialist work
            result = await self._handle_complex_task_with_react(message, context, intent, run_id, start_time)
            
            # Mark that this should have been a specialist
            result['routing'] = 'specialist_needed'
            result['specialist_type'] = specialist_type
            result['note'] = f"This task should be handled by a {specialist_type} agent"
            
            return result
            
        except Exception as e:
            logger.error(f"Error spawning specialist: {e}")
            return {
                "response": f"I need to connect you with a {intent.get('specialist_needed', 'specialist')} for this task, but there was an issue. Please try rephrasing your request.",
                "error": str(e),
                "intent": intent,
                "routing": "specialist_spawn_failed"
            }

    async def _generate_simple_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate a simple, direct response for basic questions."""
        try:
            simple_prompt = ChatPromptTemplate.from_messages([
                ("system", f"You are a helpful {self.specialty} assistant. Give brief, direct answers to simple questions. Keep responses under 100 words."),
                ("user", "{message}")
            ])
            
            chain = simple_prompt | self.llm
            response = await chain.ainvoke({"message": message})
            return response.content
            
        except Exception as e:
            logger.warning(f"Simple response generation failed: {e}")
            return f"I understand you're asking about {self.specialty}. Could you rephrase your question?"

    async def _handle_complex_task_with_react(self, message: str, context: Dict[str, Any], 
                                            intent: Dict[str, Any], run_id: str, start_time: datetime) -> Dict[str, Any]:
        """
        Handle complex tasks using ReAct-like patterns (Think, Act, Observe).
        This is what specialist agents would do.
        """
        try:
            logger.info(f"🧠 Starting ReAct pattern for {intent['type']} task")
            
            # THINK: Analyze what needs to be done
            task_plan = await self._think_about_task(message, context, intent)
            
            # ACT: Execute planned actions
            action_results = []
            for step in task_plan['steps']:
                logger.info(f"🔧 Executing: {step['action']}")
                step_result = await self._execute_react_step(step, message, context)
                action_results.append(step_result)
                
                # OBSERVE: Check if we should continue or adjust
                should_continue = await self._observe_and_decide(step_result, task_plan, len(action_results))
                if not should_continue:
                    break
            
            # Generate final response based on all actions
            final_response = await self._synthesize_react_results(message, task_plan, action_results)
            
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Log complex interaction
            conversation_id = context.get("conversation_id")
            message_id = ""
            if conversation_id and self.supabase_logger:
                message_id = await self._log_to_supabase(
                    conversation_id, message, final_response, context, 
                    100, 50, 50, 0.01, processing_time_ms  # Estimate for complex task
                )
            
            return {
                "response": final_response,
                "agent_id": self.agent_id,
                "specialty": self.specialty,
                "conversation_type": "complex_react",
                "confidence": 0.8,
                "intent": intent,
                "task_plan": task_plan,
                "action_results": action_results,
                "tokens_used": 100,  # Estimate
                "processing_time_ms": processing_time_ms,
                "run_id": run_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "routing": "react_pattern"
            }
            
        except Exception as e:
            logger.error(f"Error in ReAct processing: {e}")
            return {
                "response": f"I started working on your {intent['type']} task but encountered an issue. Please try breaking it into smaller parts.",
                "error": str(e),
                "intent": intent,
                "routing": "react_failed"
            }

    async def _think_about_task(self, message: str, context: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
        """Think step: Plan what actions are needed."""
        return {
            'intent': intent['type'],
            'steps': [
                {'action': 'analyze_requirements', 'description': 'Understand what user needs'},
                {'action': 'gather_information', 'description': 'Collect relevant data'},
                {'action': 'process_and_respond', 'description': 'Generate comprehensive response'}
            ],
            'reasoning': f"This {intent['type']} task needs structured approach"
        }

    async def _execute_react_step(self, step: Dict[str, Any], message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Act step: Execute a planned action."""
        action = step['action']
        
        if action == 'analyze_requirements':
            return {'action': action, 'result': 'Requirements analyzed', 'success': True}
        elif action == 'gather_information':
            # This would use tools, memory, etc.
            return {'action': action, 'result': 'Information gathered', 'success': True}
        elif action == 'process_and_respond':
            # Generate response using current logic
            response = await self._generate_simple_response(message, context)
            return {'action': action, 'result': response, 'success': True}
        else:
            return {'action': action, 'result': 'Action completed', 'success': True}

    async def _observe_and_decide(self, step_result: Dict[str, Any], task_plan: Dict[str, Any], step_count: int) -> bool:
        """Observe step: Decide if we should continue."""
        # Simple logic: continue if step succeeded and we haven't done too many steps
        return step_result.get('success', False) and step_count < 5

    async def _synthesize_react_results(self, message: str, task_plan: Dict[str, Any], action_results: List[Dict[str, Any]]) -> str:
        """Synthesize all ReAct steps into final response."""
        successful_actions = [r for r in action_results if r.get('success')]
        
        if successful_actions:
            last_result = successful_actions[-1].get('result', '')
            return f"After analyzing your {task_plan['intent']} request, here's my response:\n\n{last_result}"
        else:
            return f"I worked on your {task_plan['intent']} request but need more information to provide a complete answer."

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
        """Execute available tools based on intelligent tool detection and planning."""
        
        # Step 1: Check if this should use code generation instead of tools
        if self._should_use_code_generation(message):
            code_result = await self._handle_with_code_generation(message, context)
            if code_result and code_result.get('success'):
                return {'code_execution': code_result}
        
        # Step 2: Fast rule-based tool detection (no LLM calls)
        needed_tools = self._detect_required_tools(message)
        if not needed_tools:
            return {}
        
        # Step 3: If we have tool registry, use it directly
        if self._tool_registry:
            return await self._execute_tools_via_registry(needed_tools, message, context)
        
        # Step 4: Otherwise use old system - Plan tool execution sequence
        tool_plan = self._plan_tool_execution(needed_tools, message)
        
        # Step 5: Execute with failure recovery and caching
        return await self._execute_tool_plan(tool_plan, message, context)

    async def _execute_tools_via_registry(self, needed_tools: List[str], message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tools using the universal tool registry."""
        if not self._tool_registry:
            logger.warning("Tool registry not initialized")
            return {"error": "Tool registry not available"}
            
        results = {}
        
        for tool_type in needed_tools:
            try:
                # Search for matching tools
                matching_tools = self._tool_registry.search_tools(tool_type)
                
                if not matching_tools:
                    logger.warning(f"No tools found for {tool_type}")
                    results[tool_type] = {
                        "success": False,
                        "error": f"No tools available for {tool_type}"
                    }
                    continue
                
                # Use the first matching tool
                tool_info = matching_tools[0]
                tool_name = tool_info['name']
                
                logger.info(f"Executing {tool_name} for {tool_type}")
                
                # Prepare parameters based on tool type
                params = self._prepare_tool_parameters(tool_type, message, context)
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._tool_registry.execute_tool(
                        tool_name, 
                        params,  # Pass as single dict
                        user_id=context.get('user_id')
                    ),
                    timeout=30.0
                )
                
                results[tool_type] = result
                
                # Track usage
                if result.get('success', False):
                    self._track_successful_pattern(tool_type, message, result, context)
                    
            except asyncio.TimeoutError:
                results[tool_type] = {
                    "success": False,
                    "error": "Tool execution timed out"
                }
            except Exception as e:
                logger.error(f"Tool execution failed for {tool_type}: {e}")
                results[tool_type] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def _prepare_tool_parameters(self, tool_type: str, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare parameters for tool execution based on tool type."""
        # Extract parameters from message based on tool type
        if tool_type == 'search' or tool_type == 'web_search':
            return {"query": message, "num_results": 5}
        elif tool_type == 'calculate':
            # Extract mathematical expression
            import re
            math_expr = re.search(r'[\d\s\+\-\*/\(\)\.]+', message)
            if math_expr:
                return {"expression": math_expr.group(0).strip()}
            return {"expression": message}
        elif tool_type == 'file':
            # Extract file path if present
            parts = message.split()
            if len(parts) > 1:
                return {"file_path": parts[-1]}
            return {"file_path": message}
        else:
            # Generic parameters
            return {"input": message, "context": context}

    def _should_use_code_generation(self, message: str) -> bool:
        """Determine if message should be handled with code generation."""
        msg_lower = message.lower()
        
        # Only use code generation for complex calculations or when tools unavailable
        # Check if we have calculate tools available first
        has_calc_tool = any('calculate' in tool.name.lower() for tool in self.tools if tool.enabled)
        
        # For simple math with available tools, use tools instead
        import re
        if re.search(r'^\s*calculate\s+\d+\s*[+\-*/]\s*\d+\s*$', message.lower()) and has_calc_tool:
            return False
        
        # Complex calculations or multi-step math
        if any(word in msg_lower for word in ['percentage', 'tip', 'interest', 'formula', 'compound']):
            return True
        
        # Programming-like requests
        if any(word in msg_lower for word in ['code', 'python', 'script', 'function']):
            return True
        
        # Mathematical expressions only if no calc tool available
        if re.search(r'\d+\s*[+\-*/]\s*\d+', message) and not has_calc_tool:
            return True
        
        return False

    async def _handle_with_code_generation(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle request using code generation inspired by SimpleCodeAgent."""
        try:
            # Create code generation prompt
            code_prompt = self._create_code_generation_prompt()
            
            # Generate code using LLM
            chain = code_prompt | self.llm
            llm_response = await chain.ainvoke({"message": message})
            generated_code = llm_response.content
            
            logger.info(f"LLM generated response: {generated_code[:200]}...")
            
            # Extract Python code from response
            executable_code = self._extract_python_code(generated_code)
            
            if not executable_code:
                logger.warning(f"No executable code extracted from LLM response: {generated_code}")
                return {"success": False, "error": "No executable code generated", "raw_response": generated_code}
            
            logger.info(f"Extracted executable code: {executable_code}")
            
            # Execute code with tools available
            execution_result = await self._execute_generated_code(executable_code, message, context)
            
            return {
                "success": execution_result.get("success", False),
                "code": executable_code,
                "output": execution_result.get("output", ""),
                "execution_result": execution_result
            }
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {"success": False, "error": str(e)}

    def _create_code_generation_prompt(self) -> ChatPromptTemplate:
        """Create prompt for code generation inspired by SimpleCodeAgent."""
        
        # Get available tools for code execution
        tools_info = []
        for tool in self.tools:
            if tool.enabled and tool.function:
                doc = getattr(tool.function, '__doc__', 'No description')
                first_line = doc.split('\n')[0].strip() if doc else "No description"
                tools_info.append(f"- {tool.name}: {first_line}")
        
        tools_list = "\n".join(tools_info) if tools_info else "- No tools available"
        
        system_template = f"""You are a code-generating AI that can write Python code to solve problems.

**Available Tools as Functions:**
{tools_list}

**Instructions:**
1. Write Python code to solve the user's request
2. Use available tool functions when needed
3. Always wrap your code in ```python code blocks
4. Be direct - write executable code, not explanations
5. For calculations, show the result clearly

**Examples:**

User: "What's 15% tip on $50?"
```python
bill = 50
tip_rate = 0.15
tip = bill * tip_rate
total = bill + tip
print(f"Tip: ${{{{tip:.2f}}}}")
print(f"Total: ${{{{total:.2f}}}}")
```

User: "Calculate compound interest on $1000 at 5% for 3 years"
```python
principal = 1000
rate = 0.05
time = 3
amount = principal * (1 + rate) ** time
interest = amount - principal
print(f"Principal: ${{{{principal}}}}")
print(f"Amount after {{{{time}}}} years: ${{{{amount:.2f}}}}")
print(f"Interest earned: ${{{{interest:.2f}}}}")
```

**Important:**
- Always use ```python code blocks
- Write executable Python code only
- No explanations outside code blocks
- Use available tools when appropriate
- Make calculations clear with print statements"""

        human_template = """User request: {message}

Generate Python code to solve this request. Wrap your code in ```python code blocks."""

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])

    def _extract_python_code(self, llm_response: str) -> str:
        """Extract Python code from LLM response."""
        import re
        
        # Clean the response
        response = llm_response.strip()
        
        logger.debug(f"Extracting code from response: {response[:100]}...")
        
        # If response starts with ```python, extract everything after it
        if response.startswith('```python'):
            # Remove ```python and any trailing ```
            code = response[9:].strip()  # Remove ```python
            if code.endswith('```'):
                code = code[:-3].strip()  # Remove trailing ```
            logger.debug(f"Found code with ```python block: {code}")
            return code
        
        # Try to find ```python code blocks
        python_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if python_blocks:
            code = python_blocks[0].strip()
            logger.debug(f"Found code with regex python block: {code}")
            return code
        
        # Try to find any ``` code blocks
        code_blocks = re.findall(r'```\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            code = code_blocks[0].strip()
            logger.debug(f"Found code with generic block: {code}")
            return code
        
        # Try to find python code blocks without newlines
        python_blocks_inline = re.findall(r'```python(.*?)```', response, re.DOTALL)
        if python_blocks_inline:
            code = python_blocks_inline[0].strip()
            logger.debug(f"Found inline python block: {code}")
            return code
        
        # Look for lines that look like executable Python code
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip explanatory text
            if any(phrase in stripped.lower() for phrase in ['to calculate', 'let\'s', 'we can', 'follow these steps']):
                continue
                
            # Look for actual Python code patterns
            if (stripped.startswith(('result =', 'calc =', 'tip =', 'total =', 'print(', 'bill =', 'amount =')) or
                any(pattern in stripped for pattern in ['=', '*', '/', '+', '-']) or
                stripped.startswith(('if ', 'for ', 'while ', 'def ', 'import ', 'from '))):
                in_code = True
                code_lines.append(line)
            elif in_code and (stripped.startswith('    ') or stripped == ''):
                # Continue if we're in a code block (indented lines or empty lines)
                code_lines.append(line)
            elif stripped and not any(phrase in stripped.lower() for phrase in ['calculate', 'step', 'result']):
                # Stop if we hit non-code text
                in_code = False
        
        if code_lines:
            code = '\n'.join(code_lines).strip()
            logger.debug(f"Found code from line analysis: {code}")
            return code
        
        # Fallback: if it looks like simple math, make it a calculation
        if any(op in response for op in ['+', '-', '*', '/', '%']) and len(response.split()) < 10:
            code = f'result = {response.strip()}\nprint(f"Result: {{result}}")'
            logger.debug(f"Generated simple math code: {code}")
            return code
        
        # Last resort: if response contains mathematical expressions, try to extract them
        math_match = re.search(r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)', response)
        if math_match:
            expression = math_match.group(0)
            code = f'result = {expression}\nprint(f"Result: {{result}}")'
            logger.debug(f"Generated math expression code: {code}")
            return code
        
        logger.warning(f"Could not extract any executable code from: {response}")
        return ""

    async def _execute_generated_code(self, code: str, original_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generated code with tool access in safe environment."""
        try:
            # Create execution environment with tools
            exec_globals = {
                "__builtins__": {
                    "print": print, "len": len, "str": str, "int": int, "float": float,
                    "range": range, "enumerate": enumerate, "zip": zip, "abs": abs,
                    "round": round, "min": min, "max": max, "sum": sum
                }
            }
            
            # Add available tools to execution environment
            for tool in self.tools:
                if tool.enabled and tool.function:
                    exec_globals[tool.name] = tool.function
            
            # Capture output
            output_lines = []
            
            def capture_print(*args, **kwargs):
                output_lines.append(" ".join(str(arg) for arg in args))
            
            exec_globals["print"] = capture_print
            
            # Execute the code
            exec(code, exec_globals)
            
            output = "\n".join(output_lines) if output_lines else "Code executed successfully"
            
            return {
                "success": True,
                "output": output,
                "code": code,
                "method": "code_generation"
            }
            
        except Exception as e:
            logger.warning(f"Code execution failed: {e}")
            return {
                "success": False,
                "output": f"Execution error: {str(e)}",
                "error": str(e),
                "code": code,
                "suggestion": "Try simplifying the request or check for syntax errors"
            }

    def _detect_required_tools(self, message: str) -> List[str]:
        """Rule-based tool detection - fast and reliable."""
        needed = []
        msg_lower = message.lower()
        
        # Simple keyword mapping to tool types
        tool_keywords = {
            'search': ['search', 'find', 'lookup', 'google', 'web', 'online'],
            'calculate': ['calculate', 'math', 'compute', '+', '-', '*', '/', 'percentage', 'tip'],
            'file': ['file', 'read', 'write', 'save', 'load', 'download'],
            'api': ['api', 'call', 'request', 'fetch', 'get', 'post'],
            'analyze': ['analyze', 'analysis', 'examine', 'review', 'study'],
            'web_search': ['search', 'find', 'lookup', 'google', 'information about']
        }
        
        # Check each tool category
        for tool_type, keywords in tool_keywords.items():
            if any(kw in msg_lower for kw in keywords):
                # Only add if we actually have this tool available
                available_tool_names = [tool.name for tool in self.tools if tool.enabled]
                if any(tool_type in tool_name.lower() for tool_name in available_tool_names):
                    needed.append(tool_type)
        
        # Check for specific mathematical expressions
        import re
        if re.search(r'\d+\s*[+\-*/]\s*\d+', message):
            needed.append('calculate')
        
        logger.debug(f"Detected required tools: {needed}")
        return needed

    def _plan_tool_execution(self, tools: List[str], message: str) -> List[Dict]:
        """Plan tool execution order with basic sequencing."""
        plan = []
        
        # Simple sequencing rules for common patterns
        if 'search' in tools and 'analyze' in tools:
            # Search first, then analyze results
            plan = [
                {'tool_type': 'search', 'input': message, 'depends_on': None},
                {'tool_type': 'analyze', 'input': 'use search results', 'depends_on': 'search'}
            ]
        elif 'search' in tools and 'calculate' in tools:
            # Search for data, then calculate
            plan = [
                {'tool_type': 'search', 'input': message, 'depends_on': None},
                {'tool_type': 'calculate', 'input': 'use search results', 'depends_on': 'search'}
            ]
        else:
            # Single tool or parallel execution
            for tool in tools:
                plan.append({'tool_type': tool, 'input': message, 'depends_on': None})
        
        logger.debug(f"Tool execution plan: {plan}")
        return plan

    async def _execute_tool_plan(self, plan: List[Dict], message: str, context: Dict) -> Dict:
        """Execute tools in sequence with caching and failure handling."""
        results = {}
        tool_context = context.copy()
        
        # Add conversation context to tool execution
        tool_context.update({
            'conversation_history': self.conversation_context,
            'previous_tool_results': self.conversation_context['tool_results_history'][-5:],  # Last 5 results
            'user_preferences': self.conversation_context['user_preferences'],
            'session_data': self.conversation_context['session_data'],
            'previous_queries': self.conversation_context['previous_queries'][-3:]  # Last 3 queries
        })
        
        for step in plan:
            tool_type = step['tool_type']
            depends_on = step.get('depends_on')
            
            try:
                # Skip if dependency failed
                if depends_on and depends_on in results and 'error' in results[depends_on]:
                    results[tool_type] = {"error": f"Dependency {depends_on} failed"}
                    continue
                
                # Find matching tool
                matching_tool = self._find_matching_tool(tool_type)
                if not matching_tool:
                    results[tool_type] = {"error": f"Tool {tool_type} not available"}
                    continue
                
                # Check cache first (with context-aware caching)
                cache_key = self._get_context_aware_cache_key(matching_tool.name, message, tool_context)
                if cache_key in self.tool_cache:
                    cached_result, timestamp = self.tool_cache[cache_key]
                    if time.time() - timestamp < self.cache_ttl:
                        logger.info(f"Cache hit for {matching_tool.name}")
                        results[tool_type] = cached_result
                        continue
                
                # Prepare enhanced input based on dependencies and context
                tool_input = message
                if depends_on and depends_on in results:
                    tool_context['previous_results'] = results
                    tool_context[f'{depends_on}_result'] = results[depends_on]
                
                # Add context from successful patterns
                if self.conversation_context['successful_patterns']:
                    tool_context['successful_patterns'] = self.conversation_context['successful_patterns']
                
                # Execute tool with timeout
                result = await asyncio.wait_for(
                    matching_tool.function(tool_input, tool_context),
                    timeout=30.0  # 30 second timeout
                )
                
                # Validate result
                if self._validate_tool_result(result):
                    results[tool_type] = result
                    
                    # Cache successful result with context
                    self.tool_cache[cache_key] = (result, time.time())
                    
                    # Track successful execution pattern
                    self._track_successful_pattern(tool_type, message, result, tool_context)
                else:
                    # Try retry with fallback
                    retry_result = await self._retry_tool_with_fallback(matching_tool, tool_input, tool_context)
                    results[tool_type] = retry_result
                    
                    # Track failed pattern if retry also failed
                    if 'error' in retry_result:
                        self._track_failed_pattern(tool_type, message, tool_context)
                
            except asyncio.TimeoutError:
                results[tool_type] = await self._handle_tool_timeout(tool_type, message)
                self._track_failed_pattern(tool_type, message, tool_context, "timeout")
            except Exception as e:
                results[tool_type] = await self._handle_tool_failure(tool_type, e, message)
                self._track_failed_pattern(tool_type, message, tool_context, str(e))
        
        # Update conversation context with results
        self._update_conversation_context(message, results, context)
        
        return results

    def _get_context_aware_cache_key(self, tool_name: str, message: str, context: Dict) -> str:
        """Generate context-aware cache key that considers user preferences and session data."""
        import hashlib
        
        # Include relevant context factors in cache key
        context_factors = {
            'tool_name': tool_name,
            'message': message,
            'user_id': context.get('user_id', 'unknown'),
            'user_preferences': str(self.conversation_context['user_preferences']),
            'session_data': str(self.conversation_context['session_data'])
        }
        
        content = "|".join(f"{k}:{v}" for k, v in context_factors.items())
        return hashlib.md5(content.encode()).hexdigest()

    def _track_successful_pattern(self, tool_type: str, message: str, result: Dict, context: Dict):
        """Track successful tool execution patterns for learning."""
        pattern = {
            'tool_type': tool_type,
            'message_type': self._classify_message_type(message),
            'success_timestamp': time.time(),
            'result_type': type(result).__name__,
            'context_factors': {
                'user_id': context.get('user_id'),
                'message_length': len(message),
                'has_dependencies': bool(context.get('previous_results'))
            }
        }
        
        # Add to successful patterns (keep last 20)
        self.conversation_context['successful_patterns'].append(pattern)
        if len(self.conversation_context['successful_patterns']) > 20:
            self.conversation_context['successful_patterns'] = self.conversation_context['successful_patterns'][-20:]

    def _track_failed_pattern(self, tool_type: str, message: str, context: Dict, error_type: str = "unknown"):
        """Track failed tool execution patterns to avoid repeating mistakes."""
        pattern = {
            'tool_type': tool_type,
            'message_type': self._classify_message_type(message),
            'failure_timestamp': time.time(),
            'error_type': error_type,
            'context_factors': {
                'user_id': context.get('user_id'),
                'message_length': len(message),
                'has_dependencies': bool(context.get('previous_results'))
            }
        }
        
        # Add to failed patterns (keep last 10)
        self.conversation_context['failed_patterns'].append(pattern)
        if len(self.conversation_context['failed_patterns']) > 10:
            self.conversation_context['failed_patterns'] = self.conversation_context['failed_patterns'][-10:]

    def _classify_message_type(self, message: str) -> str:
        """Classify message type for pattern tracking."""
        msg_lower = message.lower()
        
        if any(word in msg_lower for word in ['calculate', 'math', '+', '-', '*', '/']):
            return 'calculation'
        elif any(word in msg_lower for word in ['search', 'find', 'lookup']):
            return 'search'
        elif any(word in msg_lower for word in ['analyze', 'examine', 'study']):
            return 'analysis'
        elif any(word in msg_lower for word in ['file', 'read', 'write', 'save']):
            return 'file_operation'
        else:
            return 'general'

    def _update_conversation_context(self, message: str, results: Dict, context: Dict):
        """Update conversation context with new interaction data."""
        # Add to previous queries
        self.conversation_context['previous_queries'].append({
            'message': message,
            'timestamp': time.time(),
            'user_id': context.get('user_id', 'unknown')
        })
        
        # Keep only last 10 queries
        if len(self.conversation_context['previous_queries']) > 10:
            self.conversation_context['previous_queries'] = self.conversation_context['previous_queries'][-10:]
        
        # Add to tool results history
        if results:
            self.conversation_context['tool_results_history'].append({
                'message': message,
                'results': results,
                'timestamp': time.time(),
                'user_id': context.get('user_id', 'unknown')
            })
            
            # Keep only last 20 tool results
            if len(self.conversation_context['tool_results_history']) > 20:
                self.conversation_context['tool_results_history'] = self.conversation_context['tool_results_history'][-20:]
        
        # Learn user preferences from successful interactions
        self._learn_user_preferences(message, results, context)

    def _learn_user_preferences(self, message: str, results: Dict, context: Dict):
        """Learn user preferences from interaction patterns."""
        user_id = context.get('user_id')
        if not user_id:
            return
        
        # Initialize user preferences if not exists
        if user_id not in self.conversation_context['user_preferences']:
            self.conversation_context['user_preferences'][user_id] = {
                'preferred_tools': {},
                'message_patterns': {},
                'interaction_count': 0
            }
        
        user_prefs = self.conversation_context['user_preferences'][user_id]
        user_prefs['interaction_count'] += 1
        
        # Track tool usage preferences
        for tool_type, result in results.items():
            if 'error' not in result:  # Only track successful tool usage
                if tool_type not in user_prefs['preferred_tools']:
                    user_prefs['preferred_tools'][tool_type] = 0
                user_prefs['preferred_tools'][tool_type] += 1
        
        # Track message pattern preferences
        message_type = self._classify_message_type(message)
        if message_type not in user_prefs['message_patterns']:
            user_prefs['message_patterns'][message_type] = 0
        user_prefs['message_patterns'][message_type] += 1

    def _find_matching_tool(self, tool_type: str) -> Optional[Any]:
        """Find tool that matches the required type."""
        for tool in self.tools:
            if not tool.enabled or not tool.function:
                continue
            
            # Check if tool name contains the type
            if tool_type.lower() in tool.name.lower():
                return tool
            
            # Check common aliases
            aliases = {
                'search': ['web_search', 'google', 'lookup'],
                'calculate': ['calc', 'math', 'compute'],
                'file': ['file_ops', 'filesystem'],
                'analyze': ['analysis', 'examine']
            }
            
            if tool_type in aliases:
                for alias in aliases[tool_type]:
                    if alias in tool.name.lower():
                        return tool
        
        return None

    def _get_tool_cache_key(self, tool_name: str, message: str) -> str:
        """Generate cache key for tool result."""
        import hashlib
        content = f"{tool_name}:{message}"
        return hashlib.md5(content.encode()).hexdigest()

    def _validate_tool_result(self, result: Dict) -> bool:
        """Validate tool result makes sense."""
        if not result:
            return False
        
        # Check for obvious errors
        if isinstance(result, dict) and 'error' in result:
            return False
        
        # Check for empty results
        if isinstance(result, dict) and not result:
            return False
            
        # Check for timeout or failure indicators
        if isinstance(result, str) and any(indicator in result.lower() for indicator in ['timeout', 'failed', 'error']):
            return False
        
        return True

    async def _retry_tool_with_fallback(self, tool, message: str, context: Dict) -> Dict:
        """Retry tool with simplified input."""
        try:
            # Try with simplified message (first sentence only)
            simplified = message.split('.')[0].strip()
            if simplified != message:
                logger.info(f"Retrying {tool.name} with simplified input")
                result = await tool.function(simplified, context)
                if self._validate_tool_result(result):
                    return result
        except Exception as e:
            logger.warning(f"Tool retry failed for {tool.name}: {e}")
        
        return {"error": "Tool retry failed", "original_input": message}

    async def _handle_tool_timeout(self, tool_type: str, message: str) -> Dict:
        """Handle tool timeout with user guidance."""
        return {
            "error": "timeout",
            "message": f"Tool {tool_type} timed out",
            "suggestion": f"Try simplifying your request or try again later",
            "partial_result": f"I was working on {tool_type} for your request but it's taking too long"
        }

    async def _handle_tool_failure(self, tool_type: str, error: Exception, message: str) -> Dict:
        """Handle tool failures with fallback strategies."""
        logger.warning(f"Tool {tool_type} failed: {error}")
        
        # Try alternative tools
        alternatives = self._get_alternative_tools(tool_type)
        for alt_tool in alternatives:
            try:
                logger.info(f"Trying alternative tool: {alt_tool.name}")
                result = await alt_tool.function(message, {})
                if self._validate_tool_result(result):
                    return {
                        "result": result,
                        "used_alternative": alt_tool.name,
                        "original_tool_failed": tool_type
                    }
            except Exception:
                continue
        
        # No alternatives worked
        return {
            "error": str(error),
            "fallback_message": f"Tool {tool_type} is currently unavailable",
            "suggestion": self._get_tool_failure_suggestion(tool_type, message)
        }

    def _get_alternative_tools(self, failed_tool: str) -> List:
        """Get alternative tools for common failures."""
        alternatives = {
            'search': [t for t in self.tools if any(keyword in t.name.lower() for keyword in ['web', 'google', 'lookup'])],
            'calculate': [t for t in self.tools if any(keyword in t.name.lower() for keyword in ['math', 'calc', 'compute'])],
            'file': [t for t in self.tools if any(keyword in t.name.lower() for keyword in ['file', 'read', 'write'])],
        }
        return alternatives.get(failed_tool, [])

    def _get_tool_failure_suggestion(self, tool_type: str, message: str) -> str:
        """Provide helpful suggestion when tool fails."""
        suggestions = {
            'search': "Try rephrasing your search query or check your internet connection",
            'calculate': "Try breaking down complex calculations into simpler steps",
            'file': "Check file permissions and ensure the file path is correct",
            'api': "Check API credentials and network connectivity"
        }
        return suggestions.get(tool_type, "Please try again or rephrase your request")
    
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
    
    async def get_mcp_status(self) -> Dict[str, Any]:
        """
        Get status of MCP integration using the official SDK.
        
        Returns:
            MCP integration status and available tools
        """
        try:
            if not self.mcp_registry:
                return {
                    'mcp_enabled': False,
                    'implementation': 'official_mcp_sdk',
                    'message': 'MCP registry not initialized'
                }
            
            # Use official MCP registry to get status
            from mcp import get_mcp_status
            return await get_mcp_status()
            
        except Exception as e:
            logger.error(f"Error getting MCP status: {e}")
            return {
                'mcp_enabled': False,
                'error': str(e),
                'implementation': 'official_mcp_sdk'
            }
    
    async def sync_mcp_tools(self) -> bool:
        """
        Sync available MCP tools with the official registry.
        
        Returns:
            True if sync successful, False otherwise
        """
        try:
            if not self.mcp_registry:
                return False
            
            # Get tools from official MCP registry
            available_tools = self.mcp_registry.get_available_tools()
            
            # Track usage
            tool_count = len(available_tools)
            self.last_mcp_sync = datetime.utcnow()
            
            logger.info(f"🔄 Synced {tool_count} MCP tools for {self.specialty}")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing MCP tools: {e}")
            return False
    
    def get_mcp_usage_stats(self) -> Dict[str, Any]:
        """Get MCP tool usage statistics."""
        return {
            'mcp_enabled': bool(self.mcp_registry),
            'tool_usage': self.mcp_tool_usage.copy(),
            'last_sync': self.last_mcp_sync.isoformat() if self.last_mcp_sync else None,
            'total_mcp_calls': sum(self.mcp_tool_usage.values()),
            'unique_tools_used': len(self.mcp_tool_usage)
        }

    async def close(self):
        """Close the agent and cleanup platform resources."""
        try:
            # Save working memory before closing to prevent state loss
            await self.save_working_memory()
            
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
            
            # Close MCP integrations (official SDK)
            if self.mcp_registry:
                try:
                    await self.mcp_registry.shutdown()
                except Exception as e:
                    logger.warning(f"Error closing MCP registry: {e}")
            
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

    def get_enhanced_tool_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the enhanced tool execution system."""
        return {
            "tool_execution_system": {
                "version": "enhanced_v2",
                "features": [
                    "rule_based_tool_detection",
                    "execution_planning_with_dependencies", 
                    "context_aware_caching",
                    "failure_recovery_with_alternatives",
                    "conversation_context_persistence",
                    "code_generation_integration",
                    "pattern_learning_and_tracking"
                ]
            },
            "available_tools": [
                {
                    "name": tool.name,
                    "enabled": tool.enabled,
                    "has_function": tool.function is not None,
                    "description": tool.description
                }
                for tool in self.tools
            ] if self.tools else [],
            "cache_status": {
                "enabled": hasattr(self, 'tool_cache'),
                "cache_size": len(getattr(self, 'tool_cache', {})),
                "cache_ttl_hours": getattr(self, 'cache_ttl', 3600) / 3600
            },
            "conversation_context": {
                "tool_results_tracked": len(self.conversation_context.get('tool_results_history', [])),
                "successful_patterns": len(self.conversation_context.get('successful_patterns', [])),
                "failed_patterns": len(self.conversation_context.get('failed_patterns', [])),
                "user_preferences_learned": len(self.conversation_context.get('user_preferences', {})),
                "previous_queries_tracked": len(self.conversation_context.get('previous_queries', []))
            },
            "code_generation": {
                "enabled": True,
                "triggers": ["mathematical_expressions", "calculations", "programming_requests"],
                "execution_environment": "safe_isolated"
            },
            "capabilities": {
                "tool_chaining": True,
                "dependency_handling": True,
                "timeout_protection": True,
                "alternative_fallbacks": True,
                "result_validation": True,
                "context_persistence": True,
                "pattern_learning": True,
                "user_preference_tracking": True
            }
        }

    def clear_conversation_context(self):
        """Clear conversation context (useful for testing or new sessions)."""
        self.conversation_context = {
            'tool_results_history': [],
            'user_preferences': {},
            'session_data': {},
            'previous_queries': [],
            'successful_patterns': [],
            'failed_patterns': []
        }
        logger.info(f"Cleared conversation context for agent {self.agent_id}")

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get learned preferences for a specific user."""
        return self.conversation_context.get('user_preferences', {}).get(user_id, {
            'preferred_tools': {},
            'message_patterns': {},
            'interaction_count': 0
        })

    def export_learned_patterns(self) -> Dict[str, Any]:
        """Export learned patterns for analysis or transfer to other agents."""
        return {
            'successful_patterns': self.conversation_context.get('successful_patterns', []),
            'failed_patterns': self.conversation_context.get('failed_patterns', []),
            'user_preferences': self.conversation_context.get('user_preferences', {}),
            'export_timestamp': time.time(),
            'agent_id': self.agent_id,
            'agent_specialty': self.specialty
        }