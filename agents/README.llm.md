# Agents Directory - Self-Improving LLM-Powered Architecture

## Purpose
Contains the **Self-Improving AI Agent Platform** that learns from every interaction, spawns specialized agents dynamically, and continuously optimizes itself. All agents are **LLM-powered** using OpenAI's ChatGPT with domain-specific prompts and adaptive expertise.

## 🧠 Current Architecture Status

### Implementation Reality vs Vision
The platform is currently in **transition** between legacy static agents and the new dynamic system:

**✅ IMPLEMENTED:**
- **Universal Agent pattern** (`universal_agent.py`) - Configuration-driven agent creation
- **Legacy Agents** - Fully functional general, technical, and research agents
- **Self-Improvement Components** - Pattern recognition, cost optimization, feedback handling
- **Event-Driven Architecture** - Event bus integration for all components
- **Platform Integrations** - Supabase logging, vector memory, workflow tracking

**🚧 IN PROGRESS:**
- Dynamic agent spawning orchestration
- Automated specialist creation based on patterns
- Resource budget enforcement
- User feedback integration via slash commands

**📋 PLANNED:**
- Complete migration from legacy to universal agents
- Automated runbook generation from patterns
- Advanced cost optimization with intelligent caching

### Current Agent Types

#### 🔧 Legacy Agents (Fully Functional)
Located in dedicated directories with full platform integration:

- **`general/general_agent.py`** - Conversational specialist with escalation logic
  - Temperature: 0.7 (natural conversation)
  - Features: Escalation assessment, platform integration, performance tracking
  - Status: Production-ready with event bus and memory integration

- **`technical/technical_agent.py`** - Programming and infrastructure expert
  - Temperature: 0.3 (technical precision)
  - Features: Code analysis, debugging assistance, tool recommendations
  - Status: Production-ready with domain classification and tool integration

- **`research/research_agent.py`** - Research methodology and analysis specialist
  - Temperature: 0.4 (balanced analytical creativity)
  - Features: Research methodology design, data source recommendations
  - Status: Production-ready with research type classification

#### 🆕 Universal Agent (Configuration-Driven)
The `UniversalAgent` class supports dynamic specialist creation:

```python
# Current UniversalAgent initialization
agent = UniversalAgent(
    specialty="Python Optimization",
    system_prompt=custom_prompt,
    temperature=0.3,
    tools=[web_search_tool, code_analyzer],
    model_name="gpt-3.5-turbo-0125",
    agent_id="python_optimizer_001"
)
```

**Key Features:**
- ✅ Configuration-driven behavior (not class inheritance)
- ✅ Full platform integration (Supabase, events, memory, workflow tracking)
- ✅ Tool capability integration
- ✅ Performance metrics and self-improvement analysis
- ✅ Event-driven communication
- ✅ Dynamic prompt creation based on specialty

## 🔄 Self-Improvement Components (Production Ready)

### `improvement/` Directory
**Intelligent systems for continuous learning:**

#### `workflow_analyst.py` ✅
- **Purpose**: Analyzes completed workflows using GPT-4 for pattern extraction
- **LLM Models**: GPT-4 for analysis, GPT-3.5-turbo for optimization suggestions
- **Features**: Pattern discovery, bottleneck identification, runbook generation
- **Status**: Fully implemented with periodic analysis every 6 hours

#### `cost_optimizer.py` ✅
- **Purpose**: Intelligent cost optimization with real-time monitoring
- **LLM Models**: GPT-4 for analysis, GPT-3.5-turbo for compression
- **Features**: Prompt compression, intelligent caching, model selection optimization
- **Status**: Production-ready with similarity-based caching (85% threshold)

#### `pattern_recognition.py` ✅
- **Purpose**: Discovers recurring patterns for automation
- **LLM Integration**: ChatGPT-powered pattern extraction and classification
- **Features**: Frequency analysis, automation triggers, workflow templates
- **Status**: Operational with pattern strength classification

#### `error_recovery.py` ✅
- **Purpose**: Learns from failures to prevent future errors
- **LLM Integration**: GPT-4 analyzes error patterns and generates solutions
- **Features**: Error classification, recovery strategies, prevention measures
- **Status**: Implemented with automatic error pattern learning

#### `feedback_handler.py` ✅
- **Purpose**: Processes user feedback into system improvements
- **LLM Models**: GPT-4 for analysis, GPT-3.5-turbo for response generation
- **Features**: Feedback analysis, improvement generation, impact assessment
- **Status**: Functional with sentiment analysis and improvement tracking

## 📊 Event-Driven Architecture (Implemented)

### Event Bus Integration
All agents communicate through **events** for loose coupling:

```python
# Actual event registration pattern
async def _register_event_handlers(self):
    if not self.event_bus:
        return
        
    await self.event_bus.subscribe(
        self.agent_id,
        [
            EventType.IMPROVEMENT_APPLIED.value,
            EventType.PATTERN_DISCOVERED.value,
            EventType.FEEDBACK_RECEIVED.value
        ],
        self._handle_platform_event
    )
```

### Platform Integration Features
Every agent includes:
- ✅ **Supabase Logging** - All interactions logged with performance metrics
- ✅ **Vector Memory** - Context retrieval from previous conversations
- ✅ **Event Communication** - Async event publishing and subscription
- ✅ **Workflow Tracking** - Run ID tracking for analysis
- ✅ **Performance Metrics** - Token usage, cost tracking, success rates

## 💰 Cost Optimization (Fully Operational)

### Intelligent Caching System
The cost optimizer includes production-ready caching:

```python
# Actual cache implementation
async def check_intelligent_cache(self, query: str, agent_id: str = None):
    # Calculate query hash
    query_hash = self._generate_query_hash(query)
    
    # Check exact match first
    if query_hash in self.query_cache:
        return self._get_cached_response(query_hash)
    
    # Check similarity-based cache (85% threshold)
    similar_entry = await self._find_similar_cached_query(query)
    if similar_entry and similar_entry["similarity"] >= 0.85:
        return similar_entry
    
    return None
```

### Real-Time Cost Tracking
- ✅ **Token Usage Monitoring** - Input/output token tracking per agent
- ✅ **Model Cost Analysis** - Per-model cost optimization recommendations
- ✅ **Prompt Compression** - Automatic prompt optimization for cost reduction
- ✅ **Daily Cost Reports** - Comprehensive cost analysis and projections

## 🔄 Legacy to Universal Migration Status

### Migration Strategy
**Current Approach**: Coexistence during transition

```python
# How legacy agents currently work
class GeneralAgent:
    def __init__(self, 
                 supabase_logger: Optional[SupabaseLogger] = None,
                 vector_store: Optional[VectorMemoryStore] = None,
                 event_bus: Optional[EventBus] = None,
                 workflow_tracker: Optional[WorkflowTracker] = None):
        # Full platform integration like UniversalAgent
        
# How UniversalAgent works  
class UniversalAgent:
    def __init__(self, specialty: str, system_prompt: str, temperature: float = 0.4, ...):
        # Same platform integrations, but configuration-driven
```

**Migration Progress:**
- ✅ Both systems have identical platform integrations
- ✅ UniversalAgent supports all features of legacy agents
- 🚧 Orchestrator integration for dynamic spawning
- 📋 Automated migration of legacy agent workflows

## 🔧 Development Guidelines

### Creating New Agents
**Current Best Practice**: Use UniversalAgent for new specialists

```python
# ✅ Recommended: Universal Agent pattern
specialist = UniversalAgent(
    specialty="Database Performance Optimization",
    system_prompt=await generate_specialist_prompt("database_optimization"),
    temperature=0.3,
    tools=[sql_analyzer, performance_monitor],
    agent_id=f"db_optimizer_{timestamp}"
)

# ✅ Also Valid: Legacy agents for established patterns
technical_agent = TechnicalAgent(
    supabase_logger=logger,
    vector_store=memory,
    event_bus=events
)
```

### Self-Improvement Integration
**Every component implements the learning cycle:**

```python
# Actual pattern used across all agents
async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
    start_time = datetime.utcnow()
    run_id = str(uuid.uuid4())
    
    try:
        # Start workflow tracking
        if self.workflow_tracker:
            await self.workflow_tracker.start_workflow(run_id, context)
        
        # Process with LLM
        response = await self._generate_response(message, context)
        
        # Log everything for learning
        await self._log_to_supabase(conversation_id, message, response, context, ...)
        await self._store_memory(conversation_id, message_id, message, response, context)
        await self._publish_events(run_id, message, response, context, processing_time)
        
        return response
    except Exception as e:
        # Learn from failures
        await self._log_error_for_learning(run_id, e)
        
    finally:
        # Complete workflow tracking
        if self.workflow_tracker:
            await self.workflow_tracker.complete_workflow(run_id, result)
```

## 📈 Testing Organization

### Test Structure
All tests are now organized in `agents/tests/`:

- **`test_agent_lifecycle_protocol.py`** - Agent lifecycle and protocol testing
- **`test_integration_review.py`** - Cross-agent integration testing
- **`test_cost_optimizer.py`** - Cost optimization functionality
- **`test_feedback_handler.py`** - User feedback processing
- **`test_error_recovery.py`** - Error handling and recovery
- **`test_pattern_recognition.py`** - Pattern discovery testing
- **`test_knowledge_graph.py`** - Knowledge graph integration
- **`test_agent_performance_analyst.py`** - Performance analysis

### Running Tests
```bash
cd agents/tests
python -m pytest test_*.py -v
```

## 🚨 Current Status Summary

### ✅ Production Ready
- **Legacy Agents**: General, Technical, Research agents with full platform integration
- **Self-Improvement**: Pattern recognition, cost optimization, error recovery
- **Platform Integration**: Event bus, Supabase logging, vector memory
- **Cost Management**: Real-time tracking, intelligent caching, optimization

### 🚧 In Development
- **Dynamic Spawning**: Orchestrator-driven agent creation
- **User Feedback Integration**: Slash commands and direct improvement workflows
- **Advanced Analytics**: ML-driven pattern recognition and optimization

### 📋 Architecture Goals
- **Configuration-Driven**: All agents use UniversalAgent pattern
- **Self-Improving**: Every interaction drives system learning
- **Cost-Optimized**: Intelligent resource management and optimization
- **Event-Driven**: Loose coupling through comprehensive event system

The platform successfully combines **production-ready legacy agents** with **next-generation self-improving capabilities**, providing a robust foundation for continuous evolution while maintaining operational stability. 