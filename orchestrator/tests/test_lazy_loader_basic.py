"""
Basic test for Lazy Agent Loader functionality
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from orchestrator.lazy_loader import LazyAgentLoader, AgentConfiguration

async def test_basic_functionality():
    """Test basic lazy loader functionality."""
    print("🚀 Testing Basic Lazy Loader Functionality...")
    
    # Initialize lazy loader
    loader = LazyAgentLoader(
        max_active_agents=5,
        max_total_configurations=20
    )
    
    print(f"✅ Initialized with max_active={loader.max_active_agents}")
    
    # Create test configurations
    configs = []
    for i in range(10):
        config = AgentConfiguration(
            agent_id=f"test_agent_{i}",
            specialty=f"Test Specialist {i}",
            system_prompt=f"You are test specialist {i}.",
            temperature=0.4,
            model_name="gpt-3.5-turbo-0125",
            max_tokens=500,
            tools=[],
            created_at=datetime.utcnow(),
            last_used=datetime.utcnow(),
            usage_count=0,
            success_rate=1.0,
            average_response_time=0.0,
            total_cost=0.0,
            priority_score=0.5
        )
        configs.append(config)
        
        # Add to lazy loader
        success = await loader.add_agent_configuration(config)
        print(f"✅ Added configuration: {config.agent_id} (success: {success})")
    
    # Test metrics
    metrics = loader.get_cache_metrics()
    print(f"\n📊 Cache Metrics:")
    print(f"   • Total configurations: {metrics['agent_status']['total_configurations']}")
    print(f"   • Active agents: {metrics['agent_status']['active_agents']}")
    print(f"   • Cache utilization: {metrics['agent_status']['cache_utilization']:.2%}")
    
    # Test activity report
    activity_report = loader.get_agent_activity_report()
    print(f"\n📈 Activity Report:")
    print(f"   • Agents tracked: {len(activity_report)}")
    
    # Test optimization
    optimization_results = await loader.optimize_cache()
    print(f"\n⚡ Optimization Results:")
    print(f"   • Completed successfully: {optimization_results}")
    
    # Cleanup
    await loader.close()
    print(f"\n✅ Lazy loader test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_basic_functionality()) 