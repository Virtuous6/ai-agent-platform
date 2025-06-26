# Agents Directory - Self-Improving LLM-Powered Architecture

## Purpose
Contains the **Self-Improving AI Agent Platform** that learns from every interaction, spawns specialized agents dynamically, and continuously optimizes itself. All agents are **LLM-powered** using OpenAI's ChatGPT with domain-specific prompts and adaptive expertise.

## 🧠 Self-Improving Agent Architecture

### Evolution from Static to Dynamic
The platform has evolved from static agent classes to a **self-improving dynamic system**:

- **Before**: Static agent classes, fixed behavior, manual optimization
- **After**: Dynamic agent spawning, continuous learning, automatic improvement
- **Benefit**: Adaptive intelligence, resource optimization, user feedback integration

### Dynamic Agent Spawning
Instead of creating agent classes, the platform now **spawns agents on-demand**:

```python
# ✅ NEW WAY: Dynamic spawning
agent_id = await orchestrator.spawn_specialist_agent(
    specialty="data_analysis",
    context={"focus": "sales_metrics", "user_expertise": "intermediate"}
)

# ❌ OLD WAY: Static classes (legacy only)
agent = DataAnalysisAgent()
```

### Universal Agent Pattern
All agents use the same **UniversalAgent** class with configuration-driven behavior:

```python
class UniversalAgent:
    """Configuration-driven agent that can be any specialty."""
    
    def __init__(self, agent_id: str, specialty: str, 
                 system_prompt: str, temperature: float = 0.7):
        self.agent_id = agent_id
        self.specialty = specialty
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=temperature,
            max_tokens=800
        )
        self.system_prompt = system_prompt
        self.tools = tools or []
```

### Self-Improvement Core Loop
Every agent interaction supports the **continuous learning cycle**:

```python
async def process_with_improvement(self, message: str, context: Dict) -> Dict:
    """Process with self-improvement tracking."""
    run_id = str(uuid.uuid4())
    
    try:
        # Execute workflow
        response = await self._execute_workflow(message, context)
        
        # Track success for learning
        await self._track_execution(run_id, {
            "status": "completed",
            "response": response,
            "agents_used": [self.agent_id],
            "user_satisfaction": await self._estimate_satisfaction(response)
        })
        
        return response
        
    except Exception as e:
        # Learn from failures
        await self._track_execution(run_id, {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__
        })
        response = await self._handle_error(e)
    
    finally:
        # Always analyze for improvement
        asyncio.create_task(self._analyze_run(run_id))
        
    return response
```

## 🚀 Agent Types & Specialties

### Dynamic Specialist Categories
Agents are spawned based on **context and need**:

#### Technical Specialists
- **Programming Languages**: Python, JavaScript, Go, Rust, etc.
- **Infrastructure**: Docker, Kubernetes, AWS, DevOps
- **Data & Analytics**: SQL, data science, machine learning
- **Security**: Cybersecurity, penetration testing, compliance

#### Research Specialists  
- **Market Research**: Competition analysis, market trends
- **Academic Research**: Literature review, methodology design
- **Data Analysis**: Statistical analysis, trend identification
- **Strategic Planning**: Business strategy, decision support

#### Domain Specialists
- **Industry Experts**: Healthcare, finance, education, etc.
- **Process Optimization**: Workflow analysis, efficiency improvement
- **Content Creation**: Writing, documentation, creative work
- **Problem Solving**: Root cause analysis, solution design

### Adaptive Temperature Selection
Agents automatically adjust LLM temperature based on **context and learning**:

```python
def _calculate_optimal_temperature(self, specialty: str, context: Dict) -> float:
    """Calculate optimal temperature based on specialty and context."""
    base_temps = {
        "technical": 0.3,     # Precision needed
        "research": 0.4,      # Balanced analysis
        "creative": 0.7,      # Innovation required
        "conversation": 0.6    # Natural interaction
    }
    
    # Adjust based on user feedback and success patterns
    historical_performance = await self.get_performance_data(specialty)
    adjustment = self._calculate_temperature_adjustment(historical_performance)
    
    return max(0.1, min(0.9, base_temps.get(specialty, 0.5) + adjustment))
```

## 🔄 Self-Improvement Components

### `improvement/` Directory
**Specialized agents for continuous learning**:

#### `workflow_analyst.py`
- **Purpose**: Analyzes completed workflows for improvement opportunities
- **LLM Integration**: Uses ChatGPT to identify patterns and bottlenecks
- **Temperature**: 0.4 (analytical precision)
- **Key Features**: Pattern recognition, efficiency analysis, automation suggestions

#### `pattern_recognition.py`
- **Purpose**: Discovers recurring patterns for automation
- **LLM Integration**: ChatGPT-powered pattern extraction and classification
- **Temperature**: 0.3 (pattern precision)
- **Key Features**: Frequency analysis, automation triggers, workflow templates

#### `error_recovery.py`
- **Purpose**: Learns from failures to prevent future errors
- **LLM Integration**: ChatGPT analyzes error patterns and solutions
- **Temperature**: 0.4 (balanced problem-solving)
- **Key Features**: Error classification, recovery strategies, prevention measures

#### `cost_optimizer.py`
- **Purpose**: Optimizes LLM usage and API costs
- **LLM Integration**: Analyzes token usage patterns for optimization
- **Temperature**: 0.2 (precise cost analysis)
- **Key Features**: Usage tracking, cost prediction, efficiency improvements

#### `feedback_handler.py`
- **Purpose**: Processes user feedback into system improvements
- **LLM Integration**: ChatGPT interprets feedback and generates improvements
- **Temperature**: 0.5 (balanced interpretation)
- **Key Features**: Feedback analysis, improvement generation, impact assessment

## 📊 Event-Driven Architecture

### Event Bus Integration
All agents communicate through **events** for loose coupling:

```python
from events import EVENT_BUS

class UniversalAgent:
    async def __init__(self, agent_id: str):
        # Subscribe to relevant events
        await EVENT_BUS.subscribe(
            self.agent_id,
            ["workflow_completed", "pattern_found", "improvement_available"],
            self.handle_event
        )
    
    async def process_message(self, message: str, context: Dict) -> Dict:
        result = await self._process(message, context)
        
        # Publish completion event
        await EVENT_BUS.publish(
            "agent_task_completed",
            {
                "agent_id": self.agent_id,
                "task_type": self.specialty,
                "success": result.get("success", True),
                "duration": result.get("duration")
            },
            source=self.agent_id
        )
        
        return result
```

### Common Event Types
```python
EVENTS = {
    # Workflow events
    "workflow_started": "New workflow execution began",
    "workflow_completed": "Workflow finished successfully", 
    "workflow_failed": "Workflow encountered error",
    
    # Agent events
    "agent_spawned": "New specialist agent created",
    "agent_activated": "Agent loaded into memory",
    "agent_deactivated": "Agent unloaded from memory",
    
    # Improvement events
    "pattern_discovered": "New pattern found",
    "improvement_applied": "System improvement made",
    "feedback_received": "User provided feedback"
}
```

## 💰 Resource Management & Cost Optimization

### Lazy Loading Pattern
Agents are only **loaded when needed** to optimize resources:

```python
async def get_agent(self, agent_id: str) -> UniversalAgent:
    """Load agent only when needed."""
    # Check if already active
    if agent_id in self.active_agents:
        self.last_activity[agent_id] = datetime.utcnow()
        return self.active_agents[agent_id]
    
    # Enforce active limit (max 50 agents)
    if len(self.active_agents) >= self.max_active_agents:
        await self._evict_least_recently_used()
    
    # Create from configuration
    config = self.agent_registry[agent_id]
    agent = UniversalAgent(**config)
    
    self.active_agents[agent_id] = agent
    return agent
```

### Resource Budgets
Every goal/project has **enforced resource limits**:

```python
class ResourceBudget:
    max_agents: int = 100
    max_concurrent: int = 10
    max_daily_cost: float = 10.0
    
async def can_spawn_agent(self, goal_id: str) -> bool:
    budget = self.budgets[goal_id]
    current = await self.get_current_usage(goal_id)
    
    return all([
        current.agents < budget.max_agents,
        current.concurrent < budget.max_concurrent,
        current.cost < budget.max_daily_cost
    ])
```

## 🎯 User Feedback Integration

### Slash Commands for Improvement
Users can **directly improve workflows**:

```python
@app.command("/improve")
async def improve_workflow(ack, command):
    """User directly improves last workflow."""
    await ack()
    
    # Get user's last workflow
    last_run = await get_last_user_run(command["user_id"])
    improvement = command["text"] or "Make it better"
    
    # Store and apply feedback
    result = await apply_improvement({
        "run_id": last_run["run_id"],
        "user_id": command["user_id"],
        "improvement": improvement
    })
    
    return await respond(f"✨ Thanks! I'll {improvement}. {result}")

@app.command("/workflow")
async def workflow_management(ack, command):
    """Manage saved workflows."""
    # save, list, show, improve, delete workflows
    pass
```

### Feedback Processing Pipeline
User feedback becomes **system improvements**:

```python
class FeedbackProcessor:
    async def process_feedback(self, feedback: Dict) -> Dict:
        """Turn user feedback into system improvements."""
        
        # Analyze feedback intent with LLM
        intent = await self._analyze_feedback_intent(feedback)
        
        # Generate improvement using ChatGPT
        improvement = await self._generate_improvement(
            feedback["run_id"], intent, feedback["improvement"]
        )
        
        # Test and apply improvement
        if await self._test_improvement(improvement):
            await self._apply_improvement(improvement)
            return {"status": "applied", "improvement": improvement}
        
        return {"status": "pending_review", "reason": "Needs testing"}
```

## 🔄 Legacy Agent Support

### `legacy/` Directory
**Backward compatibility** for existing static agents:

- **general/**: General Agent (conversational specialist)
- **technical/**: Technical Agent (programming expert)  
- **research/**: Research Agent (analysis specialist)

These agents are **gradually being phased out** in favor of dynamic spawning, but remain available during the transition period.

### Migration Path
```python
# Phase 1: Both systems coexist
if use_dynamic_agents:
    agent_id = await spawn_specialist_agent(specialty, context)
    agent = await get_agent(agent_id)
else:
    agent = self.legacy_agents[agent_type]

# Phase 2: All new features use dynamic agents
# Phase 3: Legacy agents deprecated and removed
```

## 🔧 Development Guidelines

### Creating Dynamic Agents
**No more agent classes** - use the spawning pattern:

```python
# ✅ Correct: Spawn agents dynamically
async def create_specialist(self, specialty: str, context: Dict):
    agent_id = f"{specialty}_{uuid.uuid4().hex[:8]}"
    
    agent_config = {
        "agent_id": agent_id,
        "specialty": specialty,
        "system_prompt": await self._generate_specialist_prompt(specialty, context),
        "temperature": self._calculate_optimal_temperature(specialty),
        "tools": self._determine_required_tools(specialty)
    }
    
    # Store configuration only
    self.agent_registry[agent_id] = agent_config
    await self.db_logger.log_agent_spawn(agent_id, agent_config)
    
    return agent_id

# ❌ Incorrect: Don't create agent classes
class NewSpecialistAgent(BaseAgent):
    pass
```

### Self-Improvement Integration
**Every component must support learning**:

```python
class SelfImprovingComponent:
    async def execute(self, context: Dict) -> Dict:
        run_id = str(uuid.uuid4())
        result = await self._perform_task(context)
        await self._track_execution(run_id, result)
        asyncio.create_task(self._analyze_for_improvement(run_id))
        return result
```

### Testing Self-Improvement
```python
@pytest.mark.asyncio
async def test_self_improvement_cycle():
    # Execute workflow
    run_id = await execute_test_workflow()
    
    # Analyze it
    analysis = await analyst.analyze_run(run_id)
    assert analysis["patterns_found"] > 0
    
    # Test improvement
    improvement = await generate_improvement(analysis)
    assert improvement["expected_benefit"] > 0
```

## 📈 Monitoring & Analytics

### Self-Improvement Metrics
- **Learning Rate**: How quickly the system improves
- **Pattern Recognition**: Automation opportunities identified
- **User Satisfaction**: Feedback-driven improvements
- **Cost Efficiency**: Resource optimization over time

### Performance Tracking
```python
{
    "total_agents_spawned": 1247,
    "active_agents": 23,
    "patterns_discovered": 156,
    "improvements_applied": 89,
    "cost_optimization": "12% reduction",
    "user_satisfaction": "94% positive feedback"
}
```

## 🚨 Critical Requirements

### **MANDATORY Self-Improvement**
- **Every workflow MUST be tracked** with unique run_ids
- **All components MUST analyze for improvement** 
- **User feedback MUST drive system evolution**
- **Resource budgets MUST be enforced**
- **Event-driven patterns MUST be used**

### **Development Checklist**
- [ ] Use dynamic agent spawning (not classes)
- [ ] Implement workflow tracking with run_ids
- [ ] Subscribe to relevant events
- [ ] Enforce resource budgets
- [ ] Support user feedback integration
- [ ] Test self-improvement cycles
- [ ] Monitor cost and performance

This **Self-Improving LLM-Powered Architecture** transforms the platform from static agents into a living system that learns, adapts, and optimizes itself through every interaction while maintaining the robust LLM integration and cost management of the original system. 