"""
Simple test script for the AI Agent Platform
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import Orchestrator
from core.events import EventBus
from evolution.tracker import WorkflowTracker
from evolution.learner import Learner
from storage.supabase import SupabaseLogger

async def test_platform():
    """Test the platform with a simple question."""
    print("🧪 Testing AI Agent Platform...\n")
    
    # Initialize components
    storage = SupabaseLogger()
    event_bus = EventBus()
    orchestrator = Orchestrator(storage=storage, event_bus=event_bus)
    tracker = WorkflowTracker(db_logger=storage)
    learner = Learner(storage=storage, event_bus=event_bus)
    
    # Connect components
    orchestrator.set_components(tracker, learner)
    
    # Test questions
    test_cases = [
        ("What is machine learning?", "general"),
        ("Help me debug this code: def factorial(n): return factorial(n-1) * n", "technical"),
        ("/status", "command"),
    ]
    
    for question, expected_type in test_cases:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"Expected type: {expected_type}")
        print("-" * 60)
        
        try:
            response = await orchestrator.process(
                question,
                {"user_id": "test_user", "channel": "test"}
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Run test
    asyncio.run(test_platform()) 