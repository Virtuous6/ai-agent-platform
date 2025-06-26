# Orchestrator Directory - Self-Improving LLM-Powered Intelligence

## Purpose
Central coordination system for the **Self-Improving AI Agent Platform** that intelligently spawns agents on-demand, learns from every interaction, and continuously optimizes itself. Features **dynamic agent spawning**, **event-driven communication**, and **intelligent routing** with ChatGPT integration.

## 🧠 Self-Improving Orchestration Features

### Dynamic Agent Spawning
The orchestrator now **spawns specialists on-demand** instead of using static agents:

```python
# ✅ NEW: Dynamic spawning
agent_id = await orchestrator.spawn_specialist_agent(
    specialty="data_analysis",
    context={
        "focus": "sales_metrics",
        "user_expertise": "intermediate",
        "urgency": "high"
    }
)

# ❌ OLD: Static routing (legacy only)
agent = self.agents["technical"]
```

### Intelligent Resource Management
- **Lazy Loading**: Agents loaded only when needed (max 50 active)
- **Auto-Cleanup**: Inactive agents removed after 24 hours
- **Budget Enforcement**: Resource limits per goal/project
- **Cost Optimization**: LLM usage tracked and optimized

### Event-Driven Architecture
All communication happens through **events** for loose coupling:

```python
from events import EVENT_BUS

# Publish agent spawn event
await EVENT_BUS.publish(
    "agent_spawned",
    {
        "agent_id": agent_id,
        "specialty": specialty,
        "context": context
    },
    source="orchestrator"
)
```

## 🏗️ Core Components

### `agent_orchestrator.py` (UPDATED)
**Self-Improving Routing Engine**
- Spawns agents dynamically based on context and need
- Manages agent lifecycle with lazy loading and cleanup
- Tracks all workflows for continuous improvement
- Integrates with event bus for loose coupling

### `improvement_orchestrator.py` (NEW)
**Continuous Learning Coordinator**
- Analyzes workflow patterns for automation opportunities
- Applies improvements from user feedback and system learning
- Coordinates between improvement agents
- Manages improvement deployment and rollback

### `lazy_loader.py` (NEW)
**Resource-Efficient Agent Management**
- Loads agents only when needed
- Enforces active agent limits (max 50)
- Implements LRU eviction for memory management
- Tracks agent usage and performance

### `resource_manager.py` (NEW)
**Budget and Cost Control**
- Enforces resource budgets per goal/project
- Tracks LLM usage and costs across all agents
- Prevents resource exhaustion
- Optimizes cost efficiency

## 🎯 Dynamic Agent Spawning Logic

### Specialty-Based Spawning
**Context-Driven Agent Creation**:

```python
async def spawn_specialist_agent(self, specialty: str, context: Dict) -> str:
    """Spawn agent based on specialty and context."""
    agent_id = f"{specialty}_{uuid.uuid4().hex[:8]}"
    
    # Generate specialized configuration
    agent_config = {
        "agent_id": agent_id,
        "specialty": specialty,
        "system_prompt": await self._generate_specialist_prompt(specialty, context),
        "temperature": self._calculate_optimal_temperature(specialty),
        "tools": self._determine_required_tools(specialty),
        "parent_context": context
    }
    
    # Store configuration (not instance)
    self.agent_registry[agent_id] = agent_config
    
    # Track spawn event
    await self.db_logger.log_agent_spawn(agent_id, agent_config)
    
    return agent_id
```

### Intelligent Routing Algorithm
**Multi-layered routing with learning**:

1. **Pattern Recognition**: Learned patterns from successful workflows
2. **Context Analysis**: Deep understanding of request context
3. **Resource Availability**: Consider current agent load and budgets
4. **User Feedback**: Incorporate past user satisfaction scores
5. **Cost Optimization**: Balance quality with cost efficiency

### Agent Selection Criteria
```python
class AgentSelectionCriteria:
    specialty_match: float = 0.0      # How well specialty matches request
    context_relevance: float = 0.0    # Relevance to current context
    past_performance: float = 0.0     # Historical success rate
    resource_availability: float = 0.0 # Available resources
    cost_efficiency: float = 0.0      # Cost vs. expected quality
    user_preference: float = 0.0      # User feedback patterns
```

## 🔄 Self-Improvement Integration

### Workflow Tracking
**Every interaction teaches the system**:

```python
async def route_request(self, message: str, context: Dict) -> Dict:
    """Route request with self-improvement tracking."""
    run_id = str(uuid.uuid4())
    
    try:
        # Intelligent agent selection
        agent_id = await self._select_optimal_agent(message, context)
        agent = await self.lazy_loader.get_agent(agent_id)
        
        # Execute with tracking
        response = await agent.process_message(message, context)
        
        # Track success
        await self._track_workflow_success(run_id, {
            "agent_id": agent_id,
            "specialty": agent.specialty,
            "response_quality": await self._assess_quality(response),
            "user_satisfaction": context.get("feedback_score")
        })
        
        return response
        
    except Exception as e:
        # Learn from failures
        await self._track_workflow_failure(run_id, e)
        return await self._handle_error(e, context)
    
    finally:
        # Always analyze for improvement
        asyncio.create_task(self._analyze_workflow(run_id))
```

### Pattern Learning
**Automatic workflow optimization**:

```python
class WorkflowPatternLearner:
    async def analyze_patterns(self):
        """Identify automation opportunities."""
        recent_workflows = await self.get_recent_workflows(hours=24)
        
        for workflow in recent_workflows:
            signature = self._extract_signature(workflow)
            
            if self.patterns[signature].frequency >= 3:
                # Create automation for frequent patterns
                automation = await self._create_automation(signature)
                await self._deploy_automation(automation)
```

## 📊 Event-Driven Communication

### Event Bus Integration
**Loose coupling through events**:

```python
class SelfImprovingOrchestrator:
    async def __init__(self):
        # Subscribe to improvement events
        await EVENT_BUS.subscribe(
            "orchestrator",
            ["pattern_discovered", "improvement_available", "feedback_received"],
            self.handle_improvement_event
        )
    
    async def handle_improvement_event(self, event: Event):
        """Handle system improvement events."""
        if event.type == "pattern_discovered":
            await self._evaluate_pattern_for_automation(event.data)
        elif event.type == "improvement_available":
            await self._apply_improvement(event.data)
        elif event.type == "feedback_received":
            await self._process_user_feedback(event.data)
```

### Event Types Published
```python
ORCHESTRATOR_EVENTS = {
    "agent_spawned": "New specialist agent created",
    "workflow_started": "New workflow execution began",
    "workflow_completed": "Workflow finished successfully",
    "workflow_failed": "Workflow encountered error",
    "pattern_recognized": "Recurring pattern identified",
    "automation_created": "New automation deployed"
}
```

## 💰 Resource Management & Cost Optimization

### Budget Enforcement
**Per-goal resource limits**:

```python
class ResourceBudget:
    max_agents: int = 100
    max_concurrent: int = 10
    max_daily_cost: float = 10.0
    
async def can_spawn_agent(self, goal_id: str) -> bool:
    """Check if agent spawning is within budget."""
    budget = self.budgets[goal_id]
    current = await self.get_current_usage(goal_id)
    
    return all([
        current.agents < budget.max_agents,
        current.concurrent < budget.max_concurrent,
        current.cost < budget.max_daily_cost
    ])
```

### Lazy Loading Implementation
**Memory-efficient agent management**:

```python
async def get_agent(self, agent_id: str) -> UniversalAgent:
    """Load agent only when needed."""
    # Check if already active
    if agent_id in self.active_agents:
        self.last_activity[agent_id] = datetime.utcnow()
        return self.active_agents[agent_id]
    
    # Enforce active limit
    if len(self.active_agents) >= self.max_active_agents:
        await self._evict_least_recently_used()
    
    # Create from configuration
    config = self.agent_registry[agent_id]
    agent = UniversalAgent(**config)
    
    self.active_agents[agent_id] = agent
    self.last_activity[agent_id] = datetime.utcnow()
    
    return agent
```

## 🎯 User Feedback Integration

### Direct Improvement Commands
**Users drive system evolution**:

```python
@app.command("/improve")
async def improve_workflow(ack, command):
    """User directly improves last workflow."""
    await ack()
    
    # Get user's last workflow
    last_run = await self.get_last_user_run(command["user_id"])
    improvement = command["text"] or "Make it better"
    
    # Store feedback
    feedback = {
        "run_id": last_run["run_id"],
        "user_id": command["user_id"],
        "improvement": improvement,
        "timestamp": datetime.utcnow()
    }
    
    # Apply improvement
    result = await self.feedback_processor.process_feedback(feedback)
    
    # Publish feedback event
    await EVENT_BUS.publish("feedback_received", feedback, source="user")
    
    return await respond(f"✨ Thanks! {result['status']}")
```

### Workflow Version Control
**Track and manage workflow evolution**:

```python
class WorkflowVersionControl:
    async def save_workflow_version(self, workflow_id: str, 
                                   changes: Dict, 
                                   source: str = "user"):
        """Version control for workflows."""
        current = await self.get_current_version(workflow_id)
        
        new_version = {
            "workflow_id": workflow_id,
            "version": current["version"] + 1,
            "changes": changes,
            "source": source,  # "user", "system", "automated"
            "parent_version": current["version"]
        }
        
        await self.db.save_version(new_version)
        return new_version
```

## 📈 Performance Monitoring & Analytics

### Self-Improvement Metrics
**Track learning and optimization**:

```python
{
    "agents_spawned_today": 47,
    "active_agents": 12,
    "patterns_discovered": 8,
    "automations_created": 3,
    "user_feedback_items": 15,
    "cost_optimization": "8% reduction",
    "avg_response_time": "1.1s",
    "user_satisfaction": "96%"
}
```

### Learning Analytics
**Continuous improvement tracking**:

- **Pattern Recognition Rate**: How quickly recurring patterns are identified
- **Automation Success**: Effectiveness of created automations
- **Cost Efficiency**: Resource optimization over time
- **User Satisfaction**: Feedback-driven improvements
- **Agent Performance**: Success rates by specialty

## 🛡️ Error Handling & Recovery

### Self-Healing Architecture
**Learn from failures**:

```python
async def handle_error(self, error: Exception, context: Dict) -> Dict:
    """Self-healing error handling."""
    
    # Log for learning
    await self.error_tracker.log_error({
        "error_type": type(error).__name__,
        "context": context,
        "timestamp": datetime.utcnow()
    })
    
    # Try alternative approaches
    alternatives = await self._find_alternative_approaches(context)
    
    for alternative in alternatives:
        try:
            return await alternative.execute(context)
        except Exception:
            continue
    
    # Learn from complete failure
    await self._analyze_failure_pattern(error, context)
    
    return {"error": "All approaches failed", "learning": True}
```

### Graceful Degradation
**Robust fallback strategies**:

1. **Primary**: Dynamic agent spawning with full features
2. **Secondary**: Legacy agent routing with reduced features
3. **Tertiary**: Keyword-based responses
4. **Emergency**: Static fallback responses

## 🚀 Future Self-Improvement Features

### Advanced Learning
- **Multi-Agent Collaboration**: Agents working together on complex tasks
- **Predictive Spawning**: Pre-spawn agents based on usage patterns
- **Auto-Optimization**: Self-tuning parameters based on performance
- **Cross-Goal Learning**: Apply learnings across different projects

### Enhanced Intelligence
- **Context Prediction**: Anticipate user needs from conversation patterns
- **Workflow Templates**: Auto-generate workflow templates from patterns
- **Smart Caching**: Intelligent response caching and reuse
- **Performance Forecasting**: Predict and prevent bottlenecks

## 🔧 Development Guidelines

### Adding Self-Improvement Features
**Integration patterns**:

```python
class NewFeature:
    async def execute(self, context: Dict) -> Dict:
        run_id = str(uuid.uuid4())
        
        try:
            result = await self._perform_task(context)
            await self._track_execution(run_id, result)
            return result
        finally:
            asyncio.create_task(self._analyze_for_improvement(run_id))
```

### Event-Driven Development
**Always use events for communication**:

```python
# ✅ Correct: Use events
await EVENT_BUS.publish("task_completed", data, source="component")

# ❌ Incorrect: Direct calls
await other_component.handle_completion(data)
```

### Resource Budget Integration
**Always check budgets before spawning**:

```python
if await self.resource_manager.can_spawn_agent(goal_id):
    agent_id = await self.spawn_specialist_agent(specialty, context)
else:
    return await self._handle_budget_exceeded(goal_id)
```

## 🚨 Critical Requirements

### **MANDATORY Self-Improvement**
- **Every workflow MUST be tracked** with unique run_ids
- **All routing decisions MUST be analyzed** for improvement
- **User feedback MUST drive orchestration evolution**
- **Resource budgets MUST be enforced**
- **Event-driven patterns MUST be used**

### **Development Checklist**
- [ ] Implement workflow tracking with run_ids
- [ ] Use dynamic agent spawning (not static routing)
- [ ] Subscribe to improvement events
- [ ] Enforce resource budgets before spawning
- [ ] Support user feedback integration
- [ ] Test self-improvement cycles
- [ ] Monitor cost and performance

This **Self-Improving Orchestrator** serves as the intelligent brain of the platform, learning from every interaction to spawn better agents, optimize resources, and continuously improve user experience while maintaining cost efficiency and robust error handling. 