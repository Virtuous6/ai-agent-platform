# Event Bus System - AI Agent Platform

## 🚀 Overview

The Event Bus System provides a comprehensive event-driven architecture for the AI Agent Platform, enabling loose coupling between agents and components through a sophisticated publish-subscribe pattern.

## ✨ Features

### Core Event Management
- **19 Standard Event Types** covering workflow, agent, improvement, user, and system events
- **Priority-Based Processing** with configurable queue management
- **Event Correlation** tracking related events across workflows
- **TTL Support** with automatic cleanup of expired events
- **Event Tags** for flexible filtering and categorization

### Advanced Architecture
- **Publish-Subscribe Pattern** for complete component decoupling
- **Async Processing** with priority queues and graceful shutdown
- **Rate Limiting** to prevent event storms and system overload
- **Dead Letter Queue** for handling overflow and system resilience
- **Event Persistence** with Supabase integration for important events

### Monitoring & Analytics
- **Real-Time Metrics** for event throughput and system health
- **Subscription Analytics** tracking event delivery and performance
- **Comprehensive Logging** with configurable persistence rules
- **Event Replay** capability for debugging and analysis

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Publishers    │    │   Event Bus     │    │  Subscribers    │
│                 │    │                 │    │                 │
│ • Orchestrator  │───▶│ • Priority Queue│───▶│ • Agents        │
│ • Agents        │    │ • Rate Limiting │    │ • Orchestrator  │
│ • Improvement   │    │ • Persistence   │    │ • Improvement   │
│   Systems       │    │ • Metrics       │    │   Systems       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎯 Event Types

### Workflow Events
- `WORKFLOW_STARTED` - New workflow execution began
- `WORKFLOW_COMPLETED` - Workflow finished successfully  
- `WORKFLOW_FAILED` - Workflow encountered error
- `WORKFLOW_PAUSED` - Workflow execution paused
- `WORKFLOW_RESUMED` - Workflow execution resumed

### Agent Events
- `AGENT_SPAWNED` - New agent created
- `AGENT_ACTIVATED` - Agent loaded into memory
- `AGENT_DEACTIVATED` - Agent unloaded from memory
- `AGENT_ERROR` - Agent encountered error
- `AGENT_OPTIMIZED` - Agent performance optimized

### Improvement Events
- `PATTERN_DISCOVERED` - New pattern found
- `IMPROVEMENT_APPLIED` - System improvement applied
- `OPTIMIZATION_COMPLETED` - Optimization process finished
- `KNOWLEDGE_UPDATED` - Knowledge base updated

### User Events
- `FEEDBACK_RECEIVED` - User provided feedback
- `WORKFLOW_SAVED` - User saved workflow
- `COMMAND_EXECUTED` - User command executed

### System Events
- `COST_THRESHOLD_REACHED` - Cost limit approaching
- `PERFORMANCE_ALERT` - Performance issue detected
- `ERROR_RECOVERY` - Error recovery completed
- `HEALTH_CHECK` - System health check performed

## 📖 Usage

### Basic Usage

```python
from events import EventBus, EventType

# Initialize event bus
bus = EventBus()
await bus.start()

# Publish an event
await bus.publish(
    event_type=EventType.WORKFLOW_STARTED,
    data={"workflow_id": "wf-123", "user": "user-456"},
    source="orchestrator",
    priority=1
)

# Subscribe to events
async def handle_workflow_event(event):
    print(f"Received: {event.type} - {event.data}")

await bus.subscribe(
    subscriber_id="workflow-handler",
    event_types=[EventType.WORKFLOW_STARTED, EventType.WORKFLOW_COMPLETED],
    handler=handle_workflow_event
)
```

### Advanced Features

```python
# Event with correlation and tags
await bus.publish(
    event_type=EventType.PATTERN_DISCOVERED,
    data={"pattern": "optimization_request"},
    source="pattern-engine",
    correlation_id="corr-123",
    tags={"important", "optimization"},
    ttl=3600  # 1 hour TTL
)

# Filtered subscription
def priority_filter(event):
    return event.priority <= 3

await bus.subscribe(
    subscriber_id="priority-handler",
    event_types=[EventType.CUSTOM],
    handler=handle_priority_event,
    filter_func=priority_filter
)

# Get metrics
metrics = await bus.get_metrics()
print(f"Events processed: {metrics['events_processed']}")
```

### Global Event Bus

```python
from events import get_event_bus, init_event_bus

# Initialize global instance
bus = await init_event_bus(max_queue_size=5000)

# Access anywhere in the application
bus = get_event_bus()
await bus.publish(EventType.AGENT_SPAWNED, {"agent_id": "new-agent"})
```

## 🔧 Configuration

### Rate Limiting

```python
from events import RateLimitConfig

rate_config = RateLimitConfig(
    max_events_per_second=100,
    max_events_per_minute=1000,
    burst_limit=50,
    enable_rate_limiting=True
)

bus = EventBus(rate_limit_config=rate_config)
```

### Persistence

```python
# Enable persistence for important events
bus = EventBus(
    enable_persistence=True,  # Logs to Supabase
    max_queue_size=10000
)
```

## 📊 Monitoring

### Event Metrics

```python
metrics = await bus.get_metrics()
# Returns:
# {
#   'events_published': 1234,
#   'events_processed': 1230,
#   'events_failed': 4,
#   'events_rate_limited': 12,
#   'active_subscriptions': 8,
#   'is_running': True
# }
```

### Subscription Information

```python
info = await bus.get_subscription_info("my-agent")
# Returns:
# {
#   'subscriber_id': 'my-agent',
#   'event_types': ['workflow.started', 'workflow.completed'],
#   'event_count': 45,
#   'last_event_at': '2024-12-01T10:30:00Z'
# }
```

## 🔄 Integration Patterns

### Event-Aware Agent

```python
class EventAwareAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.event_bus = get_event_bus()
    
    async def initialize(self):
        await self.event_bus.subscribe(
            subscriber_id=self.agent_id,
            event_types=[EventType.WORKFLOW_STARTED],
            handler=self.handle_workflow
        )
    
    async def handle_workflow(self, event):
        # Process workflow
        await self.process(event.data)
        
        # Publish completion
        await self.event_bus.publish(
            event_type=EventType.WORKFLOW_COMPLETED,
            data={"result": "success"},
            source=self.agent_id,
            correlation_id=event.correlation_id
        )
```

### Event-Driven Orchestrator

```python
class EventDrivenOrchestrator:
    def __init__(self):
        self.event_bus = get_event_bus()
        self.workflows = {}
    
    async def initialize(self):
        await self.event_bus.subscribe(
            subscriber_id="orchestrator",
            event_types=[EventType.WORKFLOW_COMPLETED],
            handler=self.handle_completion,
            priority=1  # High priority
        )
    
    async def start_workflow(self, workflow_id: str):
        await self.event_bus.publish(
            event_type=EventType.WORKFLOW_STARTED,
            data={"workflow_id": workflow_id},
            source="orchestrator",
            priority=1
        )
```

## 🛠️ Testing

Run the comprehensive test suite:

```bash
cd events
python -c "
import asyncio
from event_bus import EventBus, EventType

async def demo():
    bus = EventBus(enable_persistence=False)
    await bus.start()
    
    # Your test code here
    
    await bus.stop()

asyncio.run(demo())
"
```

Or run the integration example:

```bash
python integration_example.py
```

## 🔒 Error Handling

The Event Bus provides robust error handling:

- **Rate Limiting**: Automatically throttles high-volume publishers
- **Dead Letter Queue**: Captures events when queues are full
- **Graceful Shutdown**: Processes remaining events during shutdown
- **Handler Isolation**: Exceptions in handlers don't affect other subscribers
- **Event TTL**: Automatic cleanup of expired events

## 🚀 Performance

- **Async Processing**: Non-blocking event processing
- **Priority Queues**: High-priority events processed first  
- **Efficient Routing**: O(1) event delivery to subscribers
- **Memory Management**: Automatic cleanup of expired events
- **Configurable Limits**: Adjustable queue sizes and rate limits

## 📝 Best Practices

1. **Use Correlation IDs** for tracking related events across workflows
2. **Set Appropriate Priorities** for time-sensitive events
3. **Filter Events** at subscription level to reduce processing overhead
4. **Monitor Metrics** to detect performance issues early
5. **Use Tags** for flexible event categorization and filtering
6. **Handle Errors Gracefully** in event handlers
7. **Clean Up Subscriptions** when components shut down

## 🔮 Future Enhancements

- Event replay and time-travel debugging
- Event sourcing capabilities  
- Cross-service event distribution
- Event schema validation
- Enhanced analytics and dashboards
- Event transformation pipelines

---

*The Event Bus System is part of the revolutionary AI Agent Platform, enabling true event-driven architecture for scalable, maintainable, and efficient agent communication.* 🎉 