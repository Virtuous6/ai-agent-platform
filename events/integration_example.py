"""
Event Bus Integration Example

This demonstrates how the Event Bus integrates with the existing
AI Agent Platform components for loose coupling and better scalability.

Created: June 2025
"""

import asyncio
import sys
import os
sys.path.append('..')

from event_bus import EventBus, EventType, get_event_bus
from datetime import datetime, timezone
import uuid

class EventAwareAgent:
    """Example of how agents can integrate with the Event Bus."""
    
    def __init__(self, agent_id: str, specialty: str):
        self.agent_id = agent_id
        self.specialty = specialty
        self.event_bus = get_event_bus()
        self.processed_tasks = []
    
    async def initialize(self):
        """Initialize the agent and subscribe to relevant events."""
        # Subscribe to events this agent cares about
        await self.event_bus.subscribe(
            subscriber_id=self.agent_id,
            event_types=[
                EventType.WORKFLOW_STARTED,
                EventType.PATTERN_DISCOVERED,
                EventType.IMPROVEMENT_APPLIED
            ],
            handler=self.handle_event,
            priority=2
        )
        
        print(f"✅ {self.specialty} Agent ({self.agent_id}) initialized and subscribed")
    
    async def handle_event(self, event):
        """Handle incoming events."""
        print(f"🤖 {self.specialty} Agent received: {event.type}")
        
        if event.type == "workflow.started":
            await self.process_workflow_start(event)
        elif event.type == "improvement.pattern_discovered":
            await self.analyze_pattern(event)
        elif event.type == "improvement.applied":
            await self.learn_from_improvement(event)
    
    async def process_workflow_start(self, event):
        """Process workflow start event."""
        workflow_data = event.data
        print(f"   📋 Processing workflow: {workflow_data.get('workflow_id')}")
        
        # Simulate work
        await asyncio.sleep(0.1)
        
        # Publish completion event
        await self.event_bus.publish(
            event_type=EventType.WORKFLOW_COMPLETED,
            data={
                "workflow_id": workflow_data.get('workflow_id'),
                "agent_id": self.agent_id,
                "specialty": self.specialty,
                "result": "success",
                "processing_time": 0.1
            },
            source=self.agent_id,
            correlation_id=event.correlation_id,
            priority=2
        )
        
        self.processed_tasks.append(workflow_data.get('workflow_id'))
    
    async def analyze_pattern(self, event):
        """Analyze discovered patterns."""
        pattern_data = event.data
        print(f"   🧠 Analyzing pattern: {pattern_data.get('pattern')}")
        
        # If this agent specializes in the pattern area, contribute
        if self.specialty.lower() in pattern_data.get('pattern', '').lower():
            await self.event_bus.publish(
                event_type=EventType.OPTIMIZATION_COMPLETED,
                data={
                    "optimizer_id": self.agent_id,
                    "pattern_optimized": pattern_data.get('pattern'),
                    "optimization_type": f"{self.specialty}_enhancement",
                    "estimated_benefit": "25% improvement"
                },
                source=self.agent_id,
                correlation_id=event.correlation_id,
                priority=3
            )
    
    async def learn_from_improvement(self, event):
        """Learn from applied improvements."""
        improvement_data = event.data
        print(f"   📚 Learning from improvement: {improvement_data.get('type')}")
        
        # Update internal knowledge based on improvement
        # This demonstrates how agents can learn from system-wide improvements
    
    async def cleanup(self):
        """Cleanup agent resources."""
        await self.event_bus.unsubscribe(self.agent_id)
        print(f"🧹 {self.specialty} Agent ({self.agent_id}) cleaned up")

class EventAwareOrchestrator:
    """Example orchestrator that uses events for coordination."""
    
    def __init__(self):
        self.event_bus = get_event_bus()
        self.active_workflows = {}
        self.metrics = {
            'workflows_started': 0,
            'workflows_completed': 0,
            'patterns_discovered': 0,
            'improvements_applied': 0
        }
    
    async def initialize(self):
        """Initialize orchestrator."""
        await self.event_bus.subscribe(
            subscriber_id="event-orchestrator",
            event_types=[
                EventType.WORKFLOW_COMPLETED,
                EventType.PATTERN_DISCOVERED,
                EventType.IMPROVEMENT_APPLIED,
                EventType.OPTIMIZATION_COMPLETED
            ],
            handler=self.handle_orchestrator_event,
            priority=1  # High priority for orchestrator
        )
        print("🎯 Event-Aware Orchestrator initialized")
    
    async def handle_orchestrator_event(self, event):
        """Handle orchestrator-level events."""
        if event.type == "workflow.completed":
            self.metrics['workflows_completed'] += 1
            workflow_id = event.data.get('workflow_id')
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]['status'] = 'completed'
                self.active_workflows[workflow_id]['completed_by'] = event.source
            print(f"🎯 Orchestrator: Workflow {workflow_id} completed by {event.source}")
        
        elif event.type == "improvement.pattern_discovered":
            self.metrics['patterns_discovered'] += 1
            print(f"🎯 Orchestrator: Pattern discovered - {event.data.get('pattern')}")
        
        elif event.type == "improvement.applied":
            self.metrics['improvements_applied'] += 1
            print(f"🎯 Orchestrator: Improvement applied - {event.data.get('type')}")
        
        elif event.type == "improvement.optimization_completed":
            print(f"🎯 Orchestrator: Optimization completed by {event.source}")
    
    async def start_workflow(self, workflow_id: str, user_query: str):
        """Start a new workflow."""
        correlation_id = str(uuid.uuid4())
        
        self.active_workflows[workflow_id] = {
            'id': workflow_id,
            'query': user_query,
            'status': 'started',
            'correlation_id': correlation_id,
            'started_at': datetime.now(timezone.utc)
        }
        
        await self.event_bus.publish(
            event_type=EventType.WORKFLOW_STARTED,
            data={
                "workflow_id": workflow_id,
                "user_query": user_query,
                "started_at": datetime.now(timezone.utc).isoformat()
            },
            source="event-orchestrator",
            correlation_id=correlation_id,
            priority=1
        )
        
        self.metrics['workflows_started'] += 1
        print(f"🚀 Started workflow: {workflow_id}")
    
    async def simulate_pattern_discovery(self):
        """Simulate pattern discovery."""
        await self.event_bus.publish(
            event_type=EventType.PATTERN_DISCOVERED,
            data={
                "pattern": "technical_optimization_requests",
                "frequency": 12,
                "strength": 0.85,
                "confidence": "high"
            },
            source="pattern-discovery-engine",
            priority=2
        )
    
    async def apply_improvement(self, improvement_type: str):
        """Apply an improvement."""
        await self.event_bus.publish(
            event_type=EventType.IMPROVEMENT_APPLIED,
            data={
                "type": improvement_type,
                "applied_at": datetime.now(timezone.utc).isoformat(),
                "expected_benefit": "30% faster responses"
            },
            source="improvement-engine",
            priority=2
        )
    
    def get_metrics(self):
        """Get orchestrator metrics."""
        return self.metrics.copy()
    
    async def cleanup(self):
        """Cleanup orchestrator."""
        await self.event_bus.unsubscribe("event-orchestrator")
        print("🧹 Event-Aware Orchestrator cleaned up")

async def run_integration_demo():
    """
    Run comprehensive integration demo showing event-driven architecture.
    """
    print("🚀 Event Bus Integration Demo")
    print("=" * 60)
    
    # Initialize global event bus
    event_bus = EventBus(
        max_queue_size=1000,
        enable_persistence=False,  # Disable for demo
        rate_limit_config=None     # No rate limiting for demo
    )
    await event_bus.start()
    
    # Create event-aware components
    orchestrator = EventAwareOrchestrator()
    technical_agent = EventAwareAgent("tech-agent-001", "Technical")
    research_agent = EventAwareAgent("research-agent-001", "Research")
    optimization_agent = EventAwareAgent("opt-agent-001", "Optimization")
    
    # Initialize all components
    await orchestrator.initialize()
    await technical_agent.initialize()
    await research_agent.initialize()
    await optimization_agent.initialize()
    
    print("\n🎭 All components initialized and connected via Event Bus")
    
    # Simulate realistic workflow scenarios
    print("\n📋 Scenario 1: User Query Processing")
    print("-" * 40)
    
    # Start multiple workflows
    await orchestrator.start_workflow("wf-001", "How to optimize database queries?")
    await orchestrator.start_workflow("wf-002", "Best practices for API design?")
    await orchestrator.start_workflow("wf-003", "Research latest ML optimization techniques")
    
    # Allow workflows to process
    await asyncio.sleep(0.2)
    
    print("\n🧠 Scenario 2: Pattern Discovery and Improvement")
    print("-" * 40)
    
    # Simulate pattern discovery
    await orchestrator.simulate_pattern_discovery()
    
    # Allow pattern analysis
    await asyncio.sleep(0.1)
    
    # Apply improvement
    await orchestrator.apply_improvement("response_optimization")
    
    # Allow improvement learning
    await asyncio.sleep(0.1)
    
    # Display final metrics
    print("\n📊 Final System Metrics")
    print("-" * 40)
    
    # Event bus metrics
    bus_metrics = await event_bus.get_metrics()
    print(f"📈 Event Bus:")
    print(f"   Events Published: {bus_metrics['events_published']}")
    print(f"   Events Processed: {bus_metrics['events_processed']}")
    print(f"   Active Subscriptions: {bus_metrics['active_subscriptions']}")
    print(f"   Events Failed: {bus_metrics['events_failed']}")
    
    # Orchestrator metrics
    orch_metrics = orchestrator.get_metrics()
    print(f"\n🎯 Orchestrator:")
    print(f"   Workflows Started: {orch_metrics['workflows_started']}")
    print(f"   Workflows Completed: {orch_metrics['workflows_completed']}")
    print(f"   Patterns Discovered: {orch_metrics['patterns_discovered']}")
    print(f"   Improvements Applied: {orch_metrics['improvements_applied']}")
    
    # Agent metrics
    print(f"\n🤖 Agents:")
    print(f"   Technical Agent Tasks: {len(technical_agent.processed_tasks)}")
    print(f"   Research Agent Tasks: {len(research_agent.processed_tasks)}")
    print(f"   Optimization Agent Tasks: {len(optimization_agent.processed_tasks)}")
    
    # Show subscription details
    print(f"\n📡 Subscription Details:")
    for agent_id in ["tech-agent-001", "research-agent-001", "opt-agent-001", "event-orchestrator"]:
        info = await event_bus.get_subscription_info(agent_id)
        if info:
            print(f"   {agent_id}: {info['event_count']} events received")
    
    # Cleanup
    await technical_agent.cleanup()
    await research_agent.cleanup()
    await optimization_agent.cleanup()
    await orchestrator.cleanup()
    await event_bus.stop()
    
    print("\n✅ Integration Demo Complete!")
    print("🎉 Event-driven architecture successfully demonstrated!")
    
    return {
        'events_published': bus_metrics['events_published'],
        'events_processed': bus_metrics['events_processed'],
        'workflows_completed': orch_metrics['workflows_completed'],
        'demo_success': True
    }

if __name__ == "__main__":
    # Run the integration demo
    result = asyncio.run(run_integration_demo())
    print(f"\n🏆 Demo Results: {result}") 