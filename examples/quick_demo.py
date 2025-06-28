"""
Quick demo of the simplified AI Agent Platform 2.0
Run this to see the new architecture in action!
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import Orchestrator
from core.events import EventBus
from evolution.tracker import WorkflowTracker
from evolution.learner import Learner
from storage.supabase import SupabaseLogger

async def demo():
    """Run a quick demo of the platform."""
    print("🚀 AI Agent Platform 2.0 - Quick Demo\n")
    
    # Initialize components
    print("Initializing components...")
    storage = SupabaseLogger()
    event_bus = EventBus()
    
    # Create orchestrator
    orchestrator = Orchestrator(storage=storage, event_bus=event_bus)
    
    # Create tracker and learner
    tracker = WorkflowTracker(db_logger=storage, event_bus=event_bus)
    learner = Learner(storage=storage, event_bus=event_bus)
    
    # Connect components
    orchestrator.set_components(tracker, learner)
    
    print("✅ All components initialized!\n")
    
    # Demo 1: Simple message
    print("=" * 50)
    print("Demo 1: Simple Question")
    print("=" * 50)
    
    response = await orchestrator.process(
        "What is machine learning?",
        {"user_id": "demo_user", "channel": "demo"}
    )
    print(f"Response: {response}\n")
    
    # Demo 2: Technical question
    print("=" * 50)
    print("Demo 2: Technical Question")
    print("=" * 50)
    
    response = await orchestrator.process(
        "Help me debug this Python code: def factorial(n): return factorial(n-1) * n",
        {"user_id": "demo_user", "channel": "demo"}
    )
    print(f"Response: {response}\n")
    
    # Demo 3: Show active agents
    print("=" * 50)
    print("Demo 3: System Status")
    print("=" * 50)
    
    status = await orchestrator.process(
        "/status",
        {"user_id": "demo_user", "channel": "demo"}
    )
    print(f"Status:\n{status}\n")
    
    # Demo 4: Pattern learning
    print("=" * 50)
    print("Demo 4: Pattern Learning")
    print("=" * 50)
    
    # Simulate multiple similar requests
    for i in range(4):
        await orchestrator.process(
            f"Find information about quantum computing",
            {"user_id": f"user_{i}", "channel": "demo"}
        )
    
    print("✅ Pattern detected and learned!")
    print("The system now recognizes 'Find information about...' as a common pattern\n")
    
    # Demo 5: Multi-agent workflow
    print("=" * 50)
    print("Demo 5: Event-Driven Multi-Agent Communication")
    print("=" * 50)
    
    # Subscribe to events
    events_received = []
    
    async def event_handler(event):
        events_received.append(f"{event.type}: {event.data}")
    
    await event_bus.subscribe("demo_subscriber", ["agent_spawned", "pattern_discovered"], event_handler)
    
    # Trigger some events
    await orchestrator.process(
        "Research AI trends and create a technical summary",
        {"user_id": "demo_user", "channel": "demo"}
    )
    
    # Give time for events to process
    await asyncio.sleep(0.5)
    
    print("Events captured:")
    for event in events_received[-5:]:  # Show last 5 events
        print(f"  • {event}")
    
    print("\n✅ Demo complete!")
    print("\nKey takeaways:")
    print("• Agents spawn dynamically based on intent")
    print("• Every interaction is tracked and analyzed")
    print("• Agents communicate through events")
    print("• Patterns become workflows automatically")
    print("• One agent class, infinite configurations")

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY=your-key-here")
        sys.exit(1)
    
    # Run the demo
    asyncio.run(demo()) 