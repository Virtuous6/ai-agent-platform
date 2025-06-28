"""
Filename: agent_orchestrator.py
Purpose: Central orchestrator for routing requests to specialized agents
Dependencies: asyncio, logging, typing

This module is part of the AI Agent Platform.
See README.llm.md in this directory for context.
"""

import asyncio
import logging
import re
import os
import uuid
import weakref
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, OrderedDict

# LLM imports for intent classification
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Type checking imports to avoid circular imports
if TYPE_CHECKING:
    from agents.general.general_agent import GeneralAgent

# Import the QueryComplexityRouter for intelligent routing
from .query_complexity_router import (
    QueryComplexityRouter, 
    ComplexityLevel, 
    ResponseStrategy,
    ComplexityAnalysis
)

logger = logging.getLogger(__name__)

class IntentClassification(BaseModel):
    """Schema for LLM intent classification response."""
    agent_type: str = Field(description="Best agent type: 'general', 'technical', or 'research'")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of the classification decision")
    alternative_agent: Optional[str] = Field(description="Alternative agent if confidence is low")

class AgentType(Enum):
    """Available agent types in the system."""
    GENERAL = "general"
    TECHNICAL = "technical" 
    RESEARCH = "research"

# Event Bus for agent communication
class EventBus:
    """Simple event bus for agent communication."""
    
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_queue = asyncio.Queue(maxsize=10000)
        self.handlers = {}
        
    async def publish(self, event_type: str, data: Dict, source: str = None):
        """Publish event without knowing subscribers."""
        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": datetime.utcnow()
        }
        
        try:
            await self.event_queue.put(event)
            logger.debug(f"Published event: {event_type} from {source}")
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event: {event_type}")
        
    async def subscribe(self, subscriber_id: str, event_types: List[str], handler=None):
        """Subscribe to event types."""
        for event_type in event_types:
            self.subscribers[event_type].append(subscriber_id)
            if handler:
                self.handlers[f"{subscriber_id}:{event_type}"] = handler

# Resource Budget Manager
class ResourceBudget:
    """Manages resource budgets to prevent runaway spawning."""
    
    def __init__(self):
        self.max_agents = 1000  # Total agent configs
        self.max_active = 50    # Active in memory
        self.max_spawns_per_hour = 20
        self.max_cost_per_hour = 10.0  # USD
        
        self.spawn_times = []
        self.hourly_cost = 0.0
        self.reset_hour = datetime.utcnow().hour
        
    def can_spawn_agent(self, estimated_cost: float = 0.1) -> tuple[bool, str]:
        """Check if we can spawn a new agent."""
        now = datetime.utcnow()
        
        # Reset hourly counters if new hour
        if now.hour != self.reset_hour:
            self.spawn_times = []
            self.hourly_cost = 0.0
            self.reset_hour = now.hour
        
        # Remove old spawn times (older than 1 hour)
        hour_ago = now - timedelta(hours=1)
        self.spawn_times = [t for t in self.spawn_times if t > hour_ago]
        
        # Check limits
        if len(self.spawn_times) >= self.max_spawns_per_hour:
            return False, f"Spawn limit reached: {self.max_spawns_per_hour}/hour"
        
        if self.hourly_cost + estimated_cost > self.max_cost_per_hour:
            return False, f"Cost limit reached: ${self.hourly_cost:.2f}/{self.max_cost_per_hour}/hour"
        
        return True, "OK"
    
    def record_spawn(self, cost: float = 0.1):
        """Record a successful spawn."""
        self.spawn_times.append(datetime.utcnow())
        self.hourly_cost += cost

class AgentOrchestrator:
    """
    Central orchestrator that routes user requests to appropriate specialized agents.
    
    Enhanced with self-improvement capabilities:
    - Dynamic agent spawning
    - Lazy loading system
    - Resource budget management
    - Event-based communication
    - Automatic cleanup
    """
    
    def __init__(self, general_agent: Optional['GeneralAgent'] = None, 
                 technical_agent: Optional[Any] = None, research_agent: Optional[Any] = None,
                 db_logger: Optional[Any] = None):
        """Initialize the orchestrator with routing rules and agent capabilities."""
        self.routing_rules = self._initialize_routing_rules()
        self.agent_capabilities = self._initialize_agent_capabilities()
        self.request_history = []  # Simple in-memory history for now
        
        # Agent instances (will be set by the main bot)
        self.general_agent = general_agent
        self.technical_agent = technical_agent
        self.research_agent = research_agent
        
        # Self-improvement components
        self.resource_budget = ResourceBudget()
        self.event_bus = EventBus()
        self.db_logger = db_logger
        
        # Initialize Lazy Agent Loader for 1000+ configurations with 50 active
        from orchestrator.lazy_loader import LazyAgentLoader
        self.lazy_loader = LazyAgentLoader(
            max_active_agents=50,
            max_total_configurations=1000,
            preload_threshold=0.7,
            cleanup_interval=3600
        )
        
        # Legacy support - maintain old interface
        self.agent_registry = {}  # Deprecated - use lazy_loader instead
        self.active_agents = OrderedDict()  # Deprecated - use lazy_loader instead
        self.agent_last_used = {}  # Deprecated - use lazy_loader instead
        self.cleanup_task = None
        
        # Initialize Improvement Orchestrator
        self.improvement_orchestrator = None
        
        # Initialize Workflow Tracker
        from orchestrator.workflow_tracker import WorkflowTracker
        self.workflow_tracker = WorkflowTracker(db_logger=db_logger)
        
        # Initialize LLM for intelligent intent classification
        self.intent_llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",  # Fast and cost-effective for classification
            temperature=0.1,  # Low temperature for consistent classification
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=200,  # Short responses for classification
        )
        
        # Create intent classification prompt
        self.intent_prompt = self._create_intent_classification_prompt()
        self.intent_parser = JsonOutputParser(pydantic_object=IntentClassification)
        self.intent_chain = self.intent_prompt | self.intent_llm | self.intent_parser
        
        # Initialize Query Complexity Router for intelligent routing
        self.complexity_router = QueryComplexityRouter(
            llm=ChatOpenAI(
                model="gpt-3.5-turbo-0125",
                temperature=0.2,  # Low temperature for consistent assessment
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=300
            )
        )
        
        # Start cleanup task
        self._start_cleanup_task()
        
        # Initialize improvement orchestrator after everything else is set up
        self._initialize_improvement_orchestrator()
        
        logger.info("Agent Orchestrator initialized with self-improvement capabilities and complexity routing")

    def _initialize_improvement_orchestrator(self):
        """Initialize the Improvement Orchestrator for continuous self-improvement."""
        try:
            from orchestrator.improvement_orchestrator import ImprovementOrchestrator
            
            self.improvement_orchestrator = ImprovementOrchestrator(
                main_orchestrator=self,
                db_logger=self.db_logger
            )
            
            logger.info("Improvement Orchestrator initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Could not initialize Improvement Orchestrator: {e}")
            self.improvement_orchestrator = None
        except Exception as e:
            logger.error(f"Error initializing Improvement Orchestrator: {e}")
            self.improvement_orchestrator = None

    async def spawn_specialist_agent(self, specialty: str, parent_context: Optional[Dict[str, Any]] = None,
                                   temperature: float = 0.4, max_tokens: int = 500) -> str:
        """
        Create a new specialist agent dynamically based on specialty and context.
        
        Args:
            specialty: The specialty area for the new agent
            parent_context: Context from parent request that triggered spawning
            temperature: LLM temperature for the agent
            max_tokens: Max tokens for agent responses
            
        Returns:
            Agent ID if successful, None if spawning failed
        """
        try:
            # Check resource budget
            can_spawn, reason = self.resource_budget.can_spawn_agent()
            if not can_spawn:
                logger.warning(f"Cannot spawn agent: {reason}")
                return None
            
            # Generate unique agent ID
            agent_id = f"{specialty.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
            
            # Create specialized system prompt
            base_prompt = """You are a highly specialized AI assistant with deep expertise in {specialty}.

Your role:
- Provide expert-level assistance in {specialty}
- Offer detailed, accurate, and practical solutions
- Share best practices and industry standards
- Explain complex concepts clearly
- Suggest optimizations and improvements

Always be:
- Precise and technically accurate
- Helpful and solution-oriented
- Clear in your explanations
- Proactive in suggesting improvements"""

            system_prompt = base_prompt.format(specialty=specialty)
            
            # Add context from parent if available
            if parent_context:
                context_info = f"\n\nContext: This agent was created to handle {specialty} requests. "
                if "user_pattern" in parent_context:
                    context_info += f"User often asks about: {parent_context['user_pattern']}"
                system_prompt += context_info
            
            # Create agent configuration for lazy loader
            from orchestrator.lazy_loader import AgentConfiguration
            
            agent_config = AgentConfiguration(
                agent_id=agent_id,
                specialty=specialty,
                system_prompt=system_prompt,
                temperature=temperature,
                model_name="gpt-3.5-turbo-0125",
                max_tokens=max_tokens,
                tools=[],  # No tools by default
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow(),
                usage_count=0,
                success_rate=1.0,
                average_response_time=0.0,
                total_cost=0.0,
                priority_score=0.5  # Default priority
            )
            
            # Add to lazy loader
            success = await self.lazy_loader.add_agent_configuration(agent_config)
            if not success:
                logger.error(f"Failed to add agent configuration to lazy loader: {agent_id}")
                return None
            
            # Legacy support - also store in old registry
            self.agent_registry[agent_id] = {
                "agent_id": agent_id,
                "specialty": specialty,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "created": datetime.utcnow(),
                "last_used": datetime.utcnow(),
                "usage_count": 0,
                "parent_context": parent_context or {},
                "model": "gpt-3.5-turbo-0125"
            }
            
            # Log the spawn event
            if self.db_logger:
                await self.db_logger.log_event("agent_spawned", {
                    "agent_id": agent_id,
                    "specialty": specialty,
                    "parent_context": parent_context
                })
                
                # Save to spawned_agents table
                try:
                    spawn_data = {
                        "agent_id": agent_id,
                        "specialty": specialty,
                        "parent_context": parent_context or {},
                        "system_prompt": system_prompt,
                        "temperature": temperature,
                        "model_name": "gpt-3.5-turbo-0125",
                        "max_tokens": max_tokens,
                        "spawned_by": parent_context.get("user_id", "system") if parent_context else "system",
                        "spawned_at": datetime.utcnow().isoformat(),
                        "status": "active"
                    }
                    self.db_logger.client.table("spawned_agents").insert(spawn_data).execute()
                except Exception as e:
                    logger.warning(f"Failed to save spawned agent to database: {e}")
            
            # Publish spawn event
            await self.event_bus.publish("agent_spawned", {
                "agent_id": agent_id,
                "specialty": specialty,
                "timestamp": datetime.utcnow().isoformat()
            }, source="orchestrator")
            
            # Record spawn in budget
            self.resource_budget.record_spawn()
            
            logger.info(f"Spawned specialist agent: {agent_id} for {specialty}")
            return agent_id
            
        except Exception as e:
            logger.error(f"Error spawning specialist agent: {str(e)}")
            return None
    
    async def get_or_load_agent(self, agent_id: str) -> Optional[Any]:
        """
        Get agent from active cache or load from registry with lazy loading.
        
        Args:
            agent_id: The agent ID to load
            
        Returns:
            Agent instance if successful, None otherwise
        """
        try:
            # Use lazy loader for efficient agent management
            agent = await self.lazy_loader.get_agent(agent_id)
            if agent:
                # Update legacy tracking for compatibility
                self.agent_last_used[agent_id] = datetime.utcnow()
                if agent_id not in self.active_agents:
                    # Manage legacy active agents cache
                    while len(self.active_agents) >= 50:
                        self.active_agents.popitem(last=False)
                    self.active_agents[agent_id] = agent
                return agent
            
            # Fallback to legacy method if not in lazy loader
            # Check if already active
            if agent_id in self.active_agents:
                # Move to end (most recently used)
                agent = self.active_agents.pop(agent_id)
                self.active_agents[agent_id] = agent
                self.agent_last_used[agent_id] = datetime.utcnow()
                return agent
            
            # Check if agent exists in registry
            if agent_id not in self.agent_registry:
                logger.warning(f"Agent {agent_id} not found in registry")
                return None
            
            # Load agent from configuration
            config = self.agent_registry[agent_id]
            
            # Create agent instance using universal pattern
            from langchain_openai import ChatOpenAI
            
            agent_llm = ChatOpenAI(
                model=config["model"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Create simple agent wrapper
            class SpecialistAgent:
                def __init__(self, agent_id, config, llm):
                    self.agent_id = agent_id
                    self.specialty = config["specialty"]
                    self.system_prompt = config["system_prompt"]
                    self.llm = llm
                    self.usage_count = config["usage_count"]
                
                async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
                    """Process message with specialist knowledge."""
                    try:
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", self.system_prompt),
                            ("human", message)
                        ])
                        
                        chain = prompt | self.llm
                        response = await chain.ainvoke({"message": message})
                        
                        self.usage_count += 1
                        
                        return {
                            "response": response.content,
                            "agent_id": self.agent_id,
                            "specialty": self.specialty,
                            "tokens_used": response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
                            "metadata": {
                                "model_used": self.llm.model_name,
                                "usage_count": self.usage_count
                            }
                        }
                    except Exception as e:
                        logger.error(f"Error in specialist agent {self.agent_id}: {str(e)}")
                        return {
                            "response": f"I encountered an error processing your {self.specialty} request. Please try again.",
                            "error": str(e),
                            "agent_id": self.agent_id
                        }
                
                async def close(self):
                    """Cleanup agent resources."""
                    pass
            
            # Create agent instance
            agent = SpecialistAgent(agent_id, config, agent_llm)
            
            # Manage active agent cache
            if len(self.active_agents) >= self.resource_budget.max_active:
                # Remove least recently used agent
                oldest_id, oldest_agent = self.active_agents.popitem(last=False)
                await oldest_agent.close()
                logger.debug(f"Unloaded agent {oldest_id} from memory")
            
            # Add to active cache
            self.active_agents[agent_id] = agent
            self.agent_last_used[agent_id] = datetime.utcnow()
            
            # Update usage in registry
            self.agent_registry[agent_id]["last_used"] = datetime.utcnow()
            self.agent_registry[agent_id]["usage_count"] += 1
            
            logger.debug(f"Loaded specialist agent: {agent_id}")
            return agent
            
        except Exception as e:
            logger.error(f"Error loading agent {agent_id}: {str(e)}")
            return None
    
    async def cleanup_inactive_agents(self):
        """Remove agents inactive for more than 24 hours."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            agents_to_remove = []
            
            for agent_id, config in self.agent_registry.items():
                if config["last_used"] < cutoff_time and config["usage_count"] < 5:
                    agents_to_remove.append(agent_id)
            
            for agent_id in agents_to_remove:
                # Remove from active cache if present
                if agent_id in self.active_agents:
                    agent = self.active_agents.pop(agent_id)
                    await agent.close()
                
                # Remove from registry
                del self.agent_registry[agent_id]
                
                # Remove from tracking
                if agent_id in self.agent_last_used:
                    del self.agent_last_used[agent_id]
                
                logger.info(f"Cleaned up inactive agent: {agent_id}")
            
            if agents_to_remove:
                await self.event_bus.publish("agents_cleaned", {
                    "removed_count": len(agents_to_remove),
                    "timestamp": datetime.utcnow().isoformat()
                }, source="orchestrator")
            
        except Exception as e:
            logger.error(f"Error in agent cleanup: {str(e)}")
    
    def _start_cleanup_task(self):
        """Start the background cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    await self.cleanup_inactive_agents()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cleanup loop: {str(e)}")
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about agent spawning and usage."""
        # Get lazy loader metrics
        lazy_metrics = self.lazy_loader.get_cache_metrics()
        
        return {
            # Legacy stats for compatibility
            "total_agents": len(self.agent_registry),
            "active_agents": len(self.active_agents),
            "max_capacity": self.resource_budget.max_agents,
            "spawns_this_hour": len(self.resource_budget.spawn_times),
            "hourly_cost": self.resource_budget.hourly_cost,
            
            # Lazy loader advanced metrics
            "lazy_loader": lazy_metrics,
            
            # Cache performance
            "cache_hit_rate": lazy_metrics["cache_performance"]["hit_rate"],
            "cache_utilization": lazy_metrics["agent_status"]["cache_utilization"],
            
            # Recent spawns from legacy registry
            "recent_spawns": [
                {
                    "agent_id": aid,
                    "specialty": config["specialty"],
                    "usage_count": config["usage_count"],
                    "created": config["created"].isoformat()
                }
                for aid, config in sorted(
                    self.agent_registry.items(),
                    key=lambda x: x[1]["created"],
                    reverse=True
                )[:10]
            ]
        }
    
    def _initialize_routing_rules(self) -> Dict[AgentType, Dict[str, Any]]:
        """
        Initialize keyword-based routing rules for each agent.
        
        Returns:
            Dictionary mapping agents to their routing criteria
        """
        return {
            AgentType.TECHNICAL: {
                "keywords": [
                    # Programming and development
                    "code", "programming", "debug", "error", "bug", "api", "function",
                    "variable", "class", "method", "algorithm", "database", "sql",
                    "python", "javascript", "java", "html", "css", "react", "node",
                    
                    # Infrastructure and systems
                    "server", "deployment", "docker", "kubernetes", "aws", "cloud",
                    "linux", "unix", "terminal", "command", "shell", "bash",
                    
                    # Technical issues
                    "broken", "not working", "failing", "crash", "exception",
                    "timeout", "connection", "network", "performance", "slow",
                    
                    # Development tools
                    "git", "github", "repository", "commit", "branch", "merge",
                    "testing", "unit test", "integration", "ci/cd", "pipeline"
                ],
                "patterns": [
                    r"error.*code",
                    r"how.*implement",
                    r"fix.*bug",
                    r"optimize.*performance",
                    r"deploy.*application"
                ],
                "priority": 1
            },
            
            AgentType.RESEARCH: {
                "keywords": [
                    # Research activities
                    "research", "analyze", "analysis", "study", "investigate",
                    "compare", "evaluate", "assess", "review", "examine",
                    
                    # Data and information
                    "data", "statistics", "metrics", "trends", "patterns",
                    "report", "summary", "findings", "insights", "evidence",
                    
                    # Academic and scientific
                    "paper", "article", "publication", "journal", "academic",
                    "methodology", "hypothesis", "experiment", "survey",
                    
                    # Business intelligence
                    "market", "competitor", "industry", "benchmark", "forecast",
                    "strategy", "business case", "requirements", "feasibility"
                ],
                "patterns": [
                    r"what.*research.*shows",
                    r"analyze.*data",
                    r"find.*information.*about",
                    r"compare.*options",
                    r"research.*topic"
                ],
                "priority": 2
            },
            
            AgentType.GENERAL: {
                "keywords": [
                    # General conversation
                    "hello", "hi", "hey", "thanks", "thank you", "please",
                    "help", "question", "ask", "tell me", "explain", "what",
                    
                    # General requests
                    "information", "details", "overview", "introduction",
                    "basics", "general", "simple", "easy", "quick",
                    
                    # Conversational
                    "how are you", "good morning", "good afternoon", "goodbye",
                    "nice", "great", "awesome", "cool", "interesting"
                ],
                "patterns": [
                    r"^(hi|hello|hey)",
                    r"how.*are.*you",
                    r"what.*is",
                    r"can.*you.*help",
                    r"tell.*me.*about"
                ],
                "priority": 3  # Lowest priority - fallback
            }
        }
    
    def _initialize_agent_capabilities(self) -> Dict[AgentType, Dict[str, Any]]:
        """
        Initialize agent capabilities and metadata.
        
        Returns:
            Dictionary describing each agent's capabilities
        """
        return {
            AgentType.TECHNICAL: {
                "name": "Technical Agent",
                "description": "Handles technical support, coding help, and system troubleshooting",
                "specialties": ["programming", "debugging", "infrastructure", "DevOps"],
                "confidence_threshold": 0.3,  # Lowered threshold
                "max_concurrent": 5
            },
            
            AgentType.RESEARCH: {
                "name": "Research Agent", 
                "description": "Conducts research, analysis, and data gathering",
                "specialties": ["market research", "data analysis", "competitive intelligence"],
                "confidence_threshold": 0.3,  # Lowered threshold
                "max_concurrent": 3
            },
            
            AgentType.GENERAL: {
                "name": "General Agent",
                "description": "Handles general conversations and non-specialized requests",
                "specialties": ["conversation", "general assistance", "information"],
                "confidence_threshold": 0.1,  # Very low threshold for fallback
                "max_concurrent": 10
            }
        }
    
    def _create_intent_classification_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for LLM-based intent classification."""
        return ChatPromptTemplate.from_template("""
You are an expert assistant routing system. Analyze the user's message and determine the BEST agent to handle it. Be decisive and confident in your classification.

**Agent Capabilities:**

🤖 **Technical Agent** - Use for:
- Programming, coding, debugging (Python, JavaScript, etc.)
- Software development, APIs, frameworks
- Infrastructure, servers, DevOps, cloud
- Technical troubleshooting, errors, performance
- System administration, databases
- Development tools (Git, Docker, etc.)

🔬 **Research Agent** - Use for:
- Research requests and information gathering
- Data analysis, statistics, metrics
- Market research, competitive analysis
- Academic or scientific inquiries  
- In-depth investigations, reports
- Business intelligence, strategy

😊 **General Agent** - Use ONLY for:
- Simple factual questions (capitals, dates, basic info)
- Casual conversation, greetings
- General help without technical/research needs
- Basic explanations of everyday topics

**Classification Rules:**
- If the message mentions coding, debugging, technical issues → **Technical Agent**
- If the message asks for research, analysis, data gathering → **Research Agent**  
- If it's a simple question or conversation → **General Agent**
- Be confident! Don't default to General unless it truly doesn't fit the others.

**User Message:** "{message}"

**Context:** {context}

Classify this request with high confidence. Look for keywords that clearly indicate technical or research needs.

{format_instructions}

Respond with JSON only.
""")
    
    async def route_request(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a user request using complexity-aware intelligent routing.
        
        This enhanced routing system:
        1. Assesses query complexity to determine response strategy
        2. Routes simple queries to direct agent responses
        3. Routes medium queries to agents with MCP tools
        4. Routes complex queries to goal-oriented workflows
        
        Args:
            message: User message content
            context: Conversation context
            
        Returns:
            Dictionary containing routing decision and agent response
        """
        # Start workflow tracking
        run_id = None
        if self.workflow_tracker:
            run_id = await self.workflow_tracker.start_workflow(
                workflow_type="complexity_aware_routing",
                user_id=context.get("user_id", "unknown"),
                trigger_message=message,
                conversation_id=context.get("conversation_id"),
                goal_id=context.get("goal_id"),
                parent_goal_id=context.get("parent_goal_id")
            )
        
        try:
            logger.info(f"Routing request with complexity assessment: '{message[:50]}...'")
            
            # Step 1: Assess query complexity for intelligent routing
            complexity_analysis = await self.complexity_router.assess_complexity(message, context)
            
            logger.info(f"Complexity assessment: {complexity_analysis.level.value} "
                       f"({complexity_analysis.score:.2f}) - {complexity_analysis.strategy.value}")
            
            # Step 2: Check for explicit agent mentions (overrides complexity)
            explicit_agent = self._check_explicit_mentions(message)
            if explicit_agent:
                logger.info(f"Explicit agent mention detected: {explicit_agent.value} "
                           f"(overriding complexity-based routing)")
                result = await self._route_to_agent(explicit_agent, message, context, confidence=1.0)
                
                # Complete workflow tracking with complexity metadata
                if run_id and self.workflow_tracker:
                    await self.workflow_tracker.track_agent_used(run_id, explicit_agent.value)
                    await self.workflow_tracker.complete_workflow(
                        run_id=run_id,
                        result={
                            "success": True,
                            "response": result.get("response", ""),
                            "tokens_used": result.get("tokens_used", 0),
                            "estimated_cost": result.get("processing_cost", 0.0),
                            "confidence_score": 1.0,
                            "pattern_signature": f"explicit_{explicit_agent.value}",
                            "automation_potential": 0.9,
                            "metadata": {
                                "complexity_score": complexity_analysis.score,
                                "complexity_level": complexity_analysis.level.value,
                                "routing_strategy": "explicit_override",
                                "complexity_indicators": complexity_analysis.indicators
                            }
                        }
                    )
                
                return result
            
            # Step 3: Route based on complexity analysis
            if complexity_analysis.strategy == ResponseStrategy.DIRECT_RESPONSE:
                # Simple query - direct agent response
                result = await self._handle_simple_query(message, context, complexity_analysis)
                
            elif complexity_analysis.strategy == ResponseStrategy.AGENT_WITH_TOOLS:
                # Medium query - agent with MCP tools
                result = await self._handle_medium_query(message, context, complexity_analysis)
                
            elif complexity_analysis.strategy == ResponseStrategy.GOAL_WORKFLOW:
                # Complex query - goal-oriented workflow
                result = await self._handle_complex_query(message, context, complexity_analysis)
            
            else:
                # Fallback to standard routing
                logger.warning(f"Unknown strategy {complexity_analysis.strategy}, falling back to standard routing")
                result = await self._handle_fallback_routing(message, context, complexity_analysis)
            
            # Complete workflow tracking with full complexity metadata
            if run_id and self.workflow_tracker:
                await self.workflow_tracker.complete_workflow(
                    run_id=run_id,
                    result={
                        "success": result.get("success", True),
                        "response": result.get("response", ""),
                        "tokens_used": result.get("tokens_used", 0),
                        "estimated_cost": result.get("processing_cost", complexity_analysis.estimated_cost),
                        "confidence_score": result.get("confidence", 0.8),
                        "pattern_signature": f"complexity_{complexity_analysis.level.value}_{int(complexity_analysis.score*10)}",
                        "automation_potential": 0.9 if complexity_analysis.level == ComplexityLevel.SIMPLE else 0.6,
                        "metadata": {
                            "complexity_score": complexity_analysis.score,
                            "complexity_level": complexity_analysis.level.value,
                            "routing_strategy": complexity_analysis.strategy.value,
                            "complexity_indicators": complexity_analysis.indicators,
                            "estimated_steps": complexity_analysis.estimated_steps,
                            "requires_tools": complexity_analysis.requires_tools,
                            "requires_multiple_agents": complexity_analysis.requires_multiple_agents
                        }
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in complexity-aware routing: {str(e)}")
            
            # Track failure
            if run_id and self.workflow_tracker:
                await self.workflow_tracker.fail_workflow(
                    run_id=run_id,
                    error_message=str(e),
                    error_details={"routing_stage": "complexity_assessment"}
                )
            
            # Fallback to general agent on error
            return await self._route_to_agent(AgentType.GENERAL, message, context, confidence=0.0, error=str(e))
    
    async def _handle_simple_query(self, message: str, context: Dict[str, Any], 
                                 complexity_analysis: ComplexityAnalysis) -> Dict[str, Any]:
        """
        Handle simple queries with direct agent response.
        
        Simple queries get routed to the most appropriate agent for a direct answer
        without additional tool usage or multi-agent coordination.
        """
        logger.info(f"Handling simple query: {complexity_analysis.reasoning}")
        
        # Use standard intent classification for simple queries
        classification = await self._classify_intent_with_llm(message, context)
        selected_agent, confidence = self._process_llm_classification(classification)
        
        # Track the agent used
        if hasattr(self, 'workflow_tracker') and self.workflow_tracker:
            for run_id in self.workflow_tracker.active_runs:
                await self.workflow_tracker.track_agent_used(run_id, selected_agent.value)
        
        # Route to the selected agent
        result = await self._route_to_agent(selected_agent, message, context, confidence)
        
        # Add complexity metadata
        if isinstance(result, dict):
            result["complexity_routing"] = {
                "strategy": "direct_response",
                "complexity_score": complexity_analysis.score,
                "estimated_steps": complexity_analysis.estimated_steps
            }
        
        return result
    
    async def _handle_medium_query(self, message: str, context: Dict[str, Any],
                                 complexity_analysis: ComplexityAnalysis) -> Dict[str, Any]:
        """
        Handle medium complexity queries with agent + MCP tools.
        
        Medium queries require analysis, research, or tool usage but remain single-focus.
        These get enhanced with MCP tool discovery and usage.
        """
        logger.info(f"Handling medium query with tools: {complexity_analysis.reasoning}")
        
        # Use intent classification but prepare for tool usage
        classification = await self._classify_intent_with_llm(message, context)
        selected_agent, confidence = self._process_llm_classification(classification)
        
        # Enhance context with tool requirement
        enhanced_context = context.copy()
        enhanced_context["requires_tools"] = True
        enhanced_context["complexity_indicators"] = complexity_analysis.indicators
        enhanced_context["estimated_steps"] = complexity_analysis.estimated_steps
        
        # Track the agent used
        if hasattr(self, 'workflow_tracker') and self.workflow_tracker:
            for run_id in self.workflow_tracker.active_runs:
                await self.workflow_tracker.track_agent_used(run_id, selected_agent.value)
        
        # Route to the selected agent with enhanced context
        result = await self._route_to_agent(selected_agent, message, enhanced_context, confidence)
        
        # Add complexity metadata
        if isinstance(result, dict):
            result["complexity_routing"] = {
                "strategy": "agent_with_tools",
                "complexity_score": complexity_analysis.score,
                "estimated_steps": complexity_analysis.estimated_steps,
                "tools_enabled": True
            }
        
        return result
    
    async def _handle_complex_query(self, message: str, context: Dict[str, Any],
                                  complexity_analysis: ComplexityAnalysis) -> Dict[str, Any]:
        """
        Handle complex queries with goal-oriented workflow.
        
        Complex queries require multi-step coordination, multiple agents, or
        workflow orchestration. These trigger goal creation and coordinated execution.
        """
        logger.info(f"Handling complex query with goal workflow: {complexity_analysis.reasoning}")
        
        try:
            # Check if we already have a goal context
            if context.get("goal_id"):
                logger.info(f"Complex query is part of existing goal: {context.get('goal_id')}")
                return await self._handle_goal_sub_query(message, context, complexity_analysis)
            
            # Try to use LangGraph workflow if available
            if hasattr(self, '_workflow_engine') or hasattr(self, 'process_with_langgraph'):
                logger.info("Attempting LangGraph workflow for complex query")
                try:
                    return await self.process_with_langgraph(message, context)
                except Exception as e:
                    logger.warning(f"LangGraph workflow failed, falling back to goal creation: {e}")
            
            # Create a new goal for the complex query
            goal_id = await self._create_goal_for_complex_query(message, context, complexity_analysis)
            
            if goal_id:
                logger.info(f"Created goal {goal_id} for complex query")
                
                # Update context with goal information
                goal_context = context.copy()
                goal_context["goal_id"] = goal_id
                goal_context["complexity_analysis"] = complexity_analysis
                
                # Route to goal-aware agent processing
                return await self._execute_goal_workflow(message, goal_context, complexity_analysis)
            else:
                # Fallback to enhanced agent routing if goal creation fails
                logger.warning("Goal creation failed, falling back to enhanced agent routing")
                return await self._handle_fallback_routing(message, context, complexity_analysis)
            
        except Exception as e:
            logger.error(f"Error handling complex query: {e}")
            return await self._handle_fallback_routing(message, context, complexity_analysis)
    
    async def _handle_fallback_routing(self, message: str, context: Dict[str, Any],
                                     complexity_analysis: ComplexityAnalysis) -> Dict[str, Any]:
        """
        Fallback routing when complexity-based routing fails.
        
        Falls back to the original intent classification system.
        """
        logger.info("Using fallback routing with intent classification")
        
        # Use standard intent classification
        classification = await self._classify_intent_with_llm(message, context)
        selected_agent, confidence = self._process_llm_classification(classification)
        
        # Track the agent used
        if hasattr(self, 'workflow_tracker') and self.workflow_tracker:
            for run_id in self.workflow_tracker.active_runs:
                await self.workflow_tracker.track_agent_used(run_id, selected_agent.value)
        
        # Route to the selected agent
        result = await self._route_to_agent(selected_agent, message, context, confidence)
        
        # Add fallback metadata
        if isinstance(result, dict):
            result["complexity_routing"] = {
                "strategy": "fallback_routing",
                "complexity_score": complexity_analysis.score,
                "original_strategy": complexity_analysis.strategy.value,
                "fallback_reason": "complexity_routing_failed"
            }
        
        return result
    
    async def _create_goal_for_complex_query(self, message: str, context: Dict[str, Any],
                                           complexity_analysis: ComplexityAnalysis) -> Optional[str]:
        """
        Create a goal for a complex query that requires multi-agent coordination.
        
        Returns:
            Goal ID if successful, None if goal creation fails
        """
        try:
            # Try to import and use the goal manager
            from core.goal_manager import GoalManager
            
            goal_manager = GoalManager()
            
            # Create goal based on complexity analysis
            goal_data = {
                "title": f"Complex Query: {message[:50]}...",
                "description": message,
                "user_id": context.get("user_id", "unknown"),
                "complexity_score": complexity_analysis.score,
                "estimated_steps": complexity_analysis.estimated_steps,
                "requires_multiple_agents": complexity_analysis.requires_multiple_agents,
                "complexity_indicators": complexity_analysis.indicators
            }
            
            goal_id = await goal_manager.create_goal(goal_data)
            logger.info(f"Created goal {goal_id} for complex query")
            return goal_id
            
        except ImportError:
            logger.warning("GoalManager not available, cannot create goal for complex query")
            return None
        except Exception as e:
            logger.error(f"Failed to create goal for complex query: {e}")
            return None
    
    async def _handle_goal_sub_query(self, message: str, context: Dict[str, Any],
                                   complexity_analysis: ComplexityAnalysis) -> Dict[str, Any]:
        """Handle a query that's part of an existing goal."""
        logger.info(f"Processing sub-query for goal {context.get('goal_id')}")
        
        # Use enhanced routing for goal sub-queries
        classification = await self._classify_intent_with_llm(message, context)
        selected_agent, confidence = self._process_llm_classification(classification)
        
        # Track the agent used
        if hasattr(self, 'workflow_tracker') and self.workflow_tracker:
            for run_id in self.workflow_tracker.active_runs:
                await self.workflow_tracker.track_agent_used(run_id, selected_agent.value)
        
        # Route with goal context
        result = await self._route_to_agent(selected_agent, message, context, confidence)
        
        # Add goal metadata
        if isinstance(result, dict):
            result["complexity_routing"] = {
                "strategy": "goal_sub_query",
                "goal_id": context.get("goal_id"),
                "complexity_score": complexity_analysis.score
            }
        
        return result
    
    async def _execute_goal_workflow(self, message: str, context: Dict[str, Any],
                                   complexity_analysis: ComplexityAnalysis) -> Dict[str, Any]:
        """Execute a goal-oriented workflow for complex queries."""
        logger.info(f"Executing goal workflow for goal {context.get('goal_id')}")
        
        try:
            # Try to use LangGraph workflow if available
            if hasattr(self, 'process_with_langgraph'):
                return await self.process_with_langgraph(message, context)
            
            # Fallback to enhanced agent routing with goal context
            classification = await self._classify_intent_with_llm(message, context)
            selected_agent, confidence = self._process_llm_classification(classification)
            
            # Track the agent used
            if hasattr(self, 'workflow_tracker') and self.workflow_tracker:
                for run_id in self.workflow_tracker.active_runs:
                    await self.workflow_tracker.track_agent_used(run_id, selected_agent.value)
            
            result = await self._route_to_agent(selected_agent, message, context, confidence)
            
            # Add goal workflow metadata
            if isinstance(result, dict):
                result["complexity_routing"] = {
                    "strategy": "goal_workflow",
                    "goal_id": context.get("goal_id"),
                    "complexity_score": complexity_analysis.score,
                    "workflow_type": "agent_based"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing goal workflow: {e}")
            return await self._handle_fallback_routing(message, context, complexity_analysis)
    
    def _check_explicit_mentions(self, message: str) -> Optional[AgentType]:
        """
        Check if the message explicitly mentions an agent.
        
        Args:
            message: User message content
            
        Returns:
            AgentType if explicit mention found, None otherwise
        """
        message_lower = message.lower()
        
        # Check for explicit agent mentions
        if any(word in message_lower for word in ["@technical", "technical agent", "tech agent"]):
            return AgentType.TECHNICAL
        elif any(word in message_lower for word in ["@research", "research agent", "researcher"]):
            return AgentType.RESEARCH
        elif any(word in message_lower for word in ["@general", "general agent", "chat agent"]):
            return AgentType.GENERAL
            
        return None
    
    async def _classify_intent_with_llm(self, message: str, context: Dict[str, Any]) -> IntentClassification:
        """
        Use LLM to classify user intent and determine appropriate agent.
        
        Args:
            message: User message content
            context: Conversation context
            
        Returns:
            IntentClassification object with routing decision
        """
        try:
            # Prepare context string
            context_str = f"User: {context.get('user_id', 'unknown')}, Channel: {context.get('channel_id', 'unknown')}"
            
            # Get classification from LLM
            classification = await self.intent_chain.ainvoke({
                "message": message,
                "context": context_str,
                "format_instructions": self.intent_parser.get_format_instructions()
            })
            
            logger.debug(f"LLM classification: {classification['agent_type']} (confidence: {classification['confidence']:.2f}) - {classification['reasoning']}")
            
            return IntentClassification(**classification)
            
        except Exception as e:
            logger.error(f"Error in LLM intent classification: {str(e)}")
            # Fallback to general agent with low confidence
            return IntentClassification(
                agent_type="general",
                confidence=0.5,
                reasoning=f"Fallback due to classification error: {str(e)}",
                alternative_agent=None
            )
    
    def _process_llm_classification(self, classification: IntentClassification) -> tuple[AgentType, float]:
        """
        Process LLM classification result and return agent selection.
        
        Args:
            classification: LLM classification result
            
        Returns:
            Tuple of (selected_agent, confidence_score)
        """
        try:
            # Convert string to AgentType with flexible matching
            agent_type_lower = classification.agent_type.lower().strip()
            
            # Handle different formats the LLM might return
            if "technical" in agent_type_lower:
                selected_agent = AgentType.TECHNICAL
            elif "research" in agent_type_lower:
                selected_agent = AgentType.RESEARCH
            elif "general" in agent_type_lower:
                selected_agent = AgentType.GENERAL
            else:
                logger.warning(f"Unknown agent type '{classification.agent_type}', defaulting to general")
                selected_agent = AgentType.GENERAL
            confidence = classification.confidence
            
            # Validate confidence and apply thresholds
            if confidence < 0.3:
                logger.info(f"LLM confidence too low for {selected_agent.value} ({confidence:.2f}), falling back to general")
                return AgentType.GENERAL, 0.5
            
            return selected_agent, confidence
            
        except Exception as e:
            logger.error(f"Error processing LLM classification: {str(e)}")
            return AgentType.GENERAL, 0.5
    
    async def _route_to_agent(self, agent_type: AgentType, message: str, context: Dict[str, Any], 
                            confidence: float, error: Optional[str] = None) -> Dict[str, Any]:
        """
        Route request to specific agent and generate response.
        
        Args:
            agent_type: Selected agent type
            message: User message
            context: Conversation context
            confidence: Routing confidence score
            error: Error message if any
            
        Returns:
            Agent response dictionary
        """
        agent_info = self.agent_capabilities[agent_type]
        
        # Get agent response with token tracking
        agent_response = await self._generate_agent_response(agent_type, message, context, confidence, error)
        
        # Extract token information from agent response if available
        tokens_used = 0
        input_tokens = 0
        output_tokens = 0
        model_used = "unknown"
        processing_cost = 0.0
        
        if isinstance(agent_response, dict):
            response_text = agent_response.get("response", agent_response)
            tokens_used = agent_response.get("tokens_used", 0)
            input_tokens = agent_response.get("input_tokens", 0)
            output_tokens = agent_response.get("output_tokens", 0)
            model_used = agent_response.get("metadata", {}).get("model_used", "unknown")
            processing_cost = agent_response.get("processing_cost", 0.0)
            
            # If we have total tokens but not input/output split, estimate
            if tokens_used > 0 and input_tokens == 0 and output_tokens == 0:
                # Estimate based on typical conversation patterns
                input_tokens = int(tokens_used * 0.7)
                output_tokens = int(tokens_used * 0.3)
        else:
            response_text = str(agent_response)
        
        # Log the routing decision
        routing_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": agent_type.value,
            "confidence": confidence,
            "message_preview": message[:100],
            "user_id": context.get("user_id"),
            "channel_id": context.get("channel_id"),
            "tokens_used": tokens_used
        }
        self.request_history.append(routing_log)
        
        return {
            "agent_type": agent_type.value,
            "agent_name": agent_info["name"],
            "confidence": confidence,
            "response": response_text,
            "tokens_used": tokens_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "processing_cost": processing_cost,
            "metadata": {
                "routing_time": datetime.utcnow().isoformat(),
                "specialties": agent_info["specialties"],
                "model_used": model_used,
                "error": error
            }
        }
    
    async def _generate_agent_response(self, agent_type: AgentType, message: str, 
                                     context: Dict[str, Any], confidence: float, 
                                     error: Optional[str] = None) -> str:
        """
        Generate appropriate response for the selected agent.
        
        Args:
            agent_type: Selected agent type
            message: User message
            context: Conversation context
            confidence: Routing confidence
            error: Error message if any
            
        Returns:
            Agent response text
        """
        if error:
            return f"I encountered an issue routing your request ({error}), but I'm here to help! What can I assist you with?"
        
        agent_info = self.agent_capabilities[agent_type]
        
        if agent_type == AgentType.TECHNICAL:
            # Use actual Technical Agent if available
            if self.technical_agent:
                try:
                    agent_response = await self.technical_agent.process_message(message, context)
                    return agent_response  # Return full response dict with token data
                except Exception as e:
                    logger.error(f"Error calling Technical Agent: {str(e)}")
                    # Fallback to placeholder response
            
            return f"👨‍💻 **Technical Agent** here! I can help with {', '.join(agent_info['specialties'])}.\n\nI understand you need technical assistance. While I'm still learning to fully utilize my capabilities, I can help with programming, debugging, infrastructure questions, and more.\n\n*Confidence: {confidence:.1%}*"
            
        elif agent_type == AgentType.RESEARCH:
            # Use actual Research Agent if available
            if self.research_agent:
                try:
                    agent_response = await self.research_agent.process_message(message, context)
                    return agent_response  # Return full response dict with token data
                except Exception as e:
                    logger.error(f"Error calling Research Agent: {str(e)}")
                    # Fallback to placeholder response
            
            return f"🔬 **Research Agent** reporting! I specialize in {', '.join(agent_info['specialties'])}.\n\nI'm ready to help you research, analyze data, or gather information. I can dive deep into topics and provide comprehensive insights.\n\n*Confidence: {confidence:.1%}*"
            
        elif agent_type == AgentType.GENERAL:
            # Use actual General Agent if available
            if self.general_agent:
                try:
                    agent_response = await self.general_agent.process_message(message, context)
                    return agent_response  # Return full response dict with token data
                except Exception as e:
                    logger.error(f"Error calling General Agent: {str(e)}")
                    # Fallback to placeholder response
            
            return f"😊 **General Agent** here! I handle {', '.join(agent_info['specialties'])}.\n\nI'm your friendly general assistant, ready to help with conversations, general questions, or direct you to the right specialist if needed.\n\n*Confidence: {confidence:.1%}*"
        
        return "Hello! I'm processing your request and will respond shortly."
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about routing decisions including complexity assessment.
        
        Returns:
            Dictionary with routing statistics and complexity analytics
        """
        if not self.request_history:
            basic_stats = {"total_requests": 0}
            # Add complexity stats even if no requests yet
            if hasattr(self, 'complexity_router'):
                basic_stats["complexity_assessment"] = self.complexity_router.get_assessment_stats()
            return basic_stats
        
        agent_counts = {}
        total_confidence = 0
        
        for entry in self.request_history:
            agent = entry["agent"]
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            total_confidence += entry["confidence"]
        
        stats = {
            "total_requests": len(self.request_history),
            "agent_distribution": agent_counts,
            "average_confidence": total_confidence / len(self.request_history),
            "last_request": self.request_history[-1]["timestamp"]
        }
        
        # Add complexity assessment statistics
        if hasattr(self, 'complexity_router'):
            stats["complexity_assessment"] = self.complexity_router.get_assessment_stats()
        
        return stats
    
    async def assess_query_complexity(self, message: str, context: Dict[str, Any] = None) -> ComplexityAnalysis:
        """
        Public method to assess query complexity without routing.
        
        Useful for testing, analysis, or external complexity assessment.
        
        Args:
            message: User query to assess
            context: Optional context for assessment
            
        Returns:
            ComplexityAnalysis with detailed assessment results
        """
        if hasattr(self, 'complexity_router'):
            return await self.complexity_router.assess_complexity(message, context or {})
        else:
            # Fallback if complexity router not available
            return ComplexityAnalysis(
                score=0.5,
                level=ComplexityLevel.MEDIUM,
                strategy=ResponseStrategy.AGENT_WITH_TOOLS,
                reasoning="Complexity router not available",
                indicators=["fallback_assessment"],
                estimated_steps=3,
                requires_tools=True,
                requires_multiple_agents=False,
                estimated_cost=0.05
            )
    
    async def process_with_langgraph(self, message: str, context: Dict[str, Any], 
                               runbook_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process message using LangGraph workflow execution.
        
        Falls back to standard routing if LangGraph unavailable.
        """
        try:
            from .langgraph import LangGraphWorkflowEngine
            
            # Initialize workflow engine if not already done
            if not hasattr(self, '_workflow_engine'):
                self._workflow_engine = LangGraphWorkflowEngine(
                    agents={
                        'general': self.general_agent,
                        'technical': self.technical_agent,
                        'research': self.research_agent
                    },
                    tools=getattr(self, 'tools', {}),
                    supabase_logger=getattr(self, 'supabase_logger', None)
                )
            
            # Check if LangGraph is available
            if not self._workflow_engine.is_available():
                logger.info("LangGraph not available, falling back to standard routing")
                return await self.route_request(message, context)
            
            # Determine appropriate runbook
            if not runbook_name:
                runbook_name = await self._select_runbook_for_message(message, context)
            
            # Load and execute runbook workflow from Supabase
            workflow = await self._workflow_engine.load_runbook_from_supabase(runbook_name)
            
            if not workflow:
                logger.warning("Failed to load workflow, falling back to standard routing")
                return await self.route_request(message, context)
            
            initial_state = {
                'user_id': context.get('user_id', 'unknown'),
                'user_message': message,
                'conversation_id': context.get('conversation_id'),
                'channel_id': context.get('channel_id'),
                'current_step': 'start',
                'execution_history': [],
                'error_count': 0,
                'agent_responses': {},
                'tool_results': {},
                'conversation_history': context.get('conversation_history', []),
                'retrieved_memories': [],
                'user_preferences': context.get('user_preferences', {}),
                'routing_confidence': 0.0,
                'confidence_score': 0.0,
                'processing_time_ms': 0.0,
                'tokens_used': 0,
                'estimated_cost': 0.0,
                'needs_escalation': False,
                'retry_count': 0,
                'max_retries': 3
            }
            
            result = await self._workflow_engine.execute_workflow(runbook_name, initial_state)
            
            logger.info(f"LangGraph workflow '{runbook_name}' completed successfully")
            return {
                'response': result.get('final_response', 'Workflow completed'),
                'agent_type': result.get('selected_agent', 'workflow'),
                'agent_name': f"LangGraph Workflow: {runbook_name}",
                'confidence': result.get('confidence_score', 1.0),
                'tokens_used': result.get('tokens_used', 0),
                'metadata': {
                    'workflow_used': runbook_name,
                    'execution_time': result.get('processing_time_ms', 0),
                    'tokens_used': result.get('tokens_used', 0),
                    'estimated_cost': result.get('estimated_cost', 0),
                    'langgraph_enabled': True
                }
            }
            
        except Exception as e:
            logger.warning(f"LangGraph processing failed, falling back to standard routing: {e}")
            return await self.route_request(message, context)

    async def _select_runbook_for_message(self, message: str, context: Dict[str, Any]) -> str:
        """Select appropriate runbook based on message analysis using Supabase."""
        try:
            # Import and get runbook manager
            from runbooks.manager import get_runbook_manager
            runbook_manager = await get_runbook_manager()
            
            # Find matching runbooks using intelligent database queries
            matches = await runbook_manager.find_matching_runbooks(
                message=message,
                agent_type=context.get("preferred_agent"),
                user_context=context
            )
            
            if matches:
                # Return the best matching runbook
                best_runbook, match_score = matches[0]
                logger.info(f"Selected runbook: {best_runbook.name} (score: {match_score:.2f})")
                return best_runbook.name
            
            # Fallback: try to get any active runbook for questions
            question_keywords = ["what", "how", "why", "when", "where", "who", "?"]
            if any(keyword in message.lower() for keyword in question_keywords):
                # Try to get answer-question runbook specifically
                runbook = await runbook_manager.get_runbook("answer-question")
                if runbook:
                    return runbook.name
            
            # Ultimate fallback - get any active runbook
            runbooks = await runbook_manager.list_runbooks()
            if runbooks:
                return runbooks[0].name
            
            logger.warning("No runbooks available - falling back to standard routing")
            return None
            
        except Exception as e:
            logger.error(f"Error selecting runbook: {e}")
            # Fallback to old behavior if runbook manager fails
            message_lower = message.lower()
            
            if any(word in message_lower for word in ['what', 'how', 'why', 'when', 'where', 'who', '?']):
                return 'answer-question'
            elif any(word in message_lower for word in ['code', 'debug', 'error', 'bug', 'technical']):
                return 'technical-support'  # Create this next
            elif any(word in message_lower for word in ['research', 'analyze', 'study', 'investigate']):
                return 'research-task'  # Create this next
            else:
                return 'answer-question'  # Default fallback

    async def get_improvement_status(self) -> Dict[str, Any]:
        """Get current improvement system status."""
        if self.improvement_orchestrator:
            return await self.improvement_orchestrator.get_improvement_status()
        else:
            return {
                "improvement_orchestrator": "not_initialized",
                "message": "Improvement orchestrator is not available"
            }

    async def force_improvement_cycle(self, cycle_type: str) -> Dict[str, Any]:
        """Force execution of a specific improvement cycle."""
        if self.improvement_orchestrator:
            from orchestrator.improvement_orchestrator import ImprovementCycle
            try:
                cycle = ImprovementCycle(cycle_type.lower())
                return await self.improvement_orchestrator.force_improvement_cycle(cycle)
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid cycle type: {cycle_type}. Valid types: real_time, hourly, daily, weekly, monthly"
                }
        else:
            return {
                "success": False,
                "error": "Improvement orchestrator is not available"
            }

    async def pause_improvements(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Temporarily pause improvement activities."""
        if self.improvement_orchestrator:
            await self.improvement_orchestrator.pause_improvements(duration_minutes)
            return {
                "success": True,
                "message": f"Improvements paused for {duration_minutes} minutes"
            }
        else:
            return {
                "success": False,
                "error": "Improvement orchestrator is not available"
            }

    async def close(self):
        """
        Close the orchestrator and cleanup agent resources.
        
        This method ensures proper cleanup of all agent connections
        when shutting down the orchestrator.
        """
        try:
            logger.info("Closing Agent Orchestrator...")
            
            # Close improvement orchestrator first
            if self.improvement_orchestrator:
                try:
                    await self.improvement_orchestrator.close()
                    logger.info("Closed Improvement Orchestrator")
                except Exception as e:
                    logger.error(f"Error closing improvement orchestrator: {e}")
            
            # Close lazy loader (manages all specialist agents efficiently)
            if hasattr(self, 'lazy_loader'):
                try:
                    await self.lazy_loader.close()
                    logger.info("Closed Lazy Agent Loader")
                except Exception as e:
                    logger.error(f"Error closing lazy loader: {e}")
            
            # Cancel cleanup task
            if self.cleanup_task and not self.cleanup_task.done():
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
                logger.info("Cancelled cleanup task")
            
            # Close any remaining legacy active specialist agents
            for agent_id, agent in self.active_agents.items():
                try:
                    await agent.close()
                    logger.debug(f"Closed legacy specialist agent: {agent_id}")
                except Exception as e:
                    logger.warning(f"Error closing legacy specialist agent {agent_id}: {e}")
            
            # Close LangGraph workflow engine if available
            if hasattr(self, '_workflow_engine'):
                try:
                    await self._workflow_engine.close()
                    logger.info("Closed LangGraph Workflow Engine")
                except Exception as e:
                    logger.warning(f"Error closing workflow engine: {e}")
            
            # Close Query Complexity Router if available
            if hasattr(self, 'complexity_router'):
                try:
                    # Note: QueryComplexityRouter doesn't need async cleanup,
                    # but we log the cleanup for completeness
                    logger.info("Closed Query Complexity Router")
                except Exception as e:
                    logger.warning(f"Error closing complexity router: {e}")
            
            # Close all available agents
            agents_to_close = [
                ("General Agent", self.general_agent),
                ("Technical Agent", self.technical_agent),
                ("Research Agent", self.research_agent)
            ]
            
            for agent_name, agent in agents_to_close:
                if agent and hasattr(agent, 'close'):
                    try:
                        await agent.close()
                        logger.info(f"Closed {agent_name}")
                    except Exception as e:
                        logger.warning(f"Error closing {agent_name}: {e}")
            
            logger.info("Agent Orchestrator closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing Agent Orchestrator: {e}") 