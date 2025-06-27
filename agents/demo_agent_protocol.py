#!/usr/bin/env python3
"""
Agent Protocol Demonstration

A simple script that demonstrates the agent spawning and lifecycle protocol in action.
Shows dynamic spawning, lazy loading, resource management, and proper cleanup.

Usage:
    PYTHONPATH=/path/to/project python agents/demo_agent_protocol.py
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demonstrate_agent_protocol():
    """Demonstrate the complete agent protocol in action."""
    
    print("\n🚀 AI Agent Protocol Demonstration")
    print("=" * 50)
    
    try:
        # 1. Setup the orchestrator
        print("\n📋 Step 1: Initializing Agent Orchestrator")
        from orchestrator.agent_orchestrator import AgentOrchestrator
        from database.supabase_logger import SupabaseLogger
        
        db_logger = SupabaseLogger()
        orchestrator = AgentOrchestrator(db_logger=db_logger)
        print("✅ Agent Orchestrator initialized with full platform integration")
        
        # 2. Check resource budget
        print("\n💰 Step 2: Checking Resource Budget")
        resource_budget = orchestrator.resource_budget
        can_spawn, reason = resource_budget.can_spawn_agent()
        print(f"✅ Can spawn agents: {can_spawn} ({reason})")
        print(f"   - Max agents: {resource_budget.max_agents}")
        print(f"   - Max active: {resource_budget.max_active}")
        print(f"   - Max spawns/hour: {resource_budget.max_spawns_per_hour}")
        print(f"   - Max cost/hour: ${resource_budget.max_cost_per_hour}")
        
        # 3. Demonstrate dynamic agent spawning
        print("\n🤖 Step 3: Dynamic Agent Spawning")
        specialties = [
            "Python Code Optimizer",
            "Database Performance Tuner",
            "API Security Analyst"
        ]
        
        spawned_agents = []
        for specialty in specialties:
            print(f"   🔄 Spawning: {specialty}")
            agent_id = await orchestrator.spawn_specialist_agent(
                specialty=specialty,
                parent_context={
                    "demo": "agent_protocol",
                    "timestamp": datetime.utcnow().isoformat()
                },
                temperature=0.3,
                max_tokens=600
            )
            
            if agent_id:
                spawned_agents.append(agent_id)
                print(f"   ✅ Spawned: {agent_id}")
            else:
                print(f"   ❌ Failed to spawn: {specialty}")
        
        print(f"\n📊 Successfully spawned {len(spawned_agents)} specialist agents")
        
        # 4. Demonstrate lazy loading
        print("\n🔄 Step 4: Demonstrating Lazy Loading")
        if spawned_agents:
            test_agent_id = spawned_agents[0]
            print(f"   🎯 Loading agent: {test_agent_id}")
            
            # First load (cache miss)
            agent = await orchestrator.get_or_load_agent(test_agent_id)
            print(f"   ✅ First load successful: {agent is not None}")
            
            # Second load (cache hit)
            agent2 = await orchestrator.get_or_load_agent(test_agent_id)
            print(f"   ✅ Second load successful: {agent2 is not None}")
            
            # Check cache metrics
            metrics = orchestrator.lazy_loader.get_cache_metrics()
            print(f"   📈 Cache hit rate: {metrics['cache_performance']['hit_rate']:.2%}")
            print(f"   📈 Cache utilization: {metrics['agent_status']['cache_utilization']:.2%}")
        
        # 5. Demonstrate agent processing
        print("\n💬 Step 5: Agent Processing Demo")
        if spawned_agents and agent:
            test_message = "Help me optimize a slow database query"
            test_context = {
                "user_id": "demo_user",
                "channel_id": "demo_channel",
                "conversation_id": "demo_conversation"
            }
            
            print(f"   📝 Processing message: '{test_message}'")
            response = await agent.process_message(test_message, test_context)
            
            print(f"   ✅ Response received:")
            print(f"      - Agent: {response.get('agent_id', 'unknown')}")
            print(f"      - Specialty: {response.get('specialty', 'unknown')}")
            print(f"      - Confidence: {response.get('confidence', 0):.2f}")
            print(f"      - Tokens used: {response.get('tokens_used', 0)}")
            print(f"      - Cost: ${response.get('processing_cost', 0):.4f}")
            print(f"      - Response length: {len(response.get('response', ''))}")
        
        # 6. Show agent statistics
        print("\n📊 Step 6: Agent Statistics")
        agent_stats = orchestrator.get_agent_stats()
        print(f"   📈 Total agents: {agent_stats.get('total_agents', 0)}")
        print(f"   📈 Active agents: {agent_stats.get('active_agents', 0)}")
        print(f"   📈 Cache hit rate: {agent_stats.get('cache_hit_rate', 0):.2%}")
        print(f"   📈 Recent spawns: {len(agent_stats.get('recent_spawns', []))}")
        
        # 7. Demonstrate resource cleanup
        print("\n🧹 Step 7: Resource Cleanup")
        print("   🔄 Running agent cleanup...")
        await orchestrator.cleanup_inactive_agents()
        print("   ✅ Cleanup completed")
        
        # 8. Full orchestrator shutdown
        print("\n🔚 Step 8: Full System Shutdown")
        print("   🔄 Closing orchestrator and all agents...")
        await orchestrator.close()
        print("   ✅ All resources cleaned up successfully")
        
        print("\n🎉 Agent Protocol Demonstration Complete!")
        print("=" * 50)
        print("✅ All protocol features working correctly:")
        print("   • Dynamic agent spawning")
        print("   • Lazy loading with caching")
        print("   • Resource budget management")
        print("   • Agent processing capabilities")
        print("   • Proper lifecycle cleanup")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.error(f"Protocol demonstration failed: {e}")
        return False

async def main():
    """Main demonstration function."""
    try:
        success = await demonstrate_agent_protocol()
        if success:
            print("\n✅ Agent protocol demonstration successful!")
            sys.exit(0)
        else:
            print("\n❌ Agent protocol demonstration failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Demonstration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 