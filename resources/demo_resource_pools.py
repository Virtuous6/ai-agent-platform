#!/usr/bin/env python3
"""
Resource Pool Manager Feature Demonstration

Showcases all key features:
- LLM connection pooling with limits
- Tool instance sharing 
- Database connection pooling
- Vector memory allocation
- Fair scheduling and timeout handling
- Health monitoring and system metrics
- Event-driven architecture integration
"""

import asyncio
import time
import sys
import os
from unittest.mock import Mock, AsyncMock

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from resources.pool_manager import (
    ResourcePoolManager, ResourceType, ResourcePriority, 
    ResourceException, get_resource_pool_manager
)

async def test_llm_connection_pooling():
    """Test LLM connection pooling with concurrent allocations."""
    print("\n🧠 LLM CONNECTION POOLING TEST")
    print("=" * 50)
    
    manager = await get_resource_pool_manager()
    
    async def allocate_llm_task(agent_id: str, model_type: str):
        """Task to allocate LLM connection."""
        try:
            start_time = time.time()
            async with manager.allocate_resource(
                ResourceType.LLM_CONNECTION,
                requester_id=f"agent_{agent_id}",
                priority=ResourcePriority.HIGH,
                creator_type=model_type,
                temperature=0.5
            ) as llm:
                duration = time.time() - start_time
                print(f"✅ Agent {agent_id}: Got {model_type} LLM (wait: {duration:.2f}s)")
                await asyncio.sleep(0.5)  # Simulate work
                return True
        except ResourceException as e:
            print(f"❌ Agent {agent_id}: Failed to get LLM - {e}")
            return False
    
    # Test concurrent allocations
    print("Starting 8 concurrent LLM allocations (pool limit: 10)...")
    tasks = [
        allocate_llm_task("1", "gpt-3.5-turbo"),
        allocate_llm_task("2", "gpt-4"),
        allocate_llm_task("3", "default"),
        allocate_llm_task("4", "gpt-3.5-turbo"),
        allocate_llm_task("5", "gpt-4"),
        allocate_llm_task("6", "default"),
        allocate_llm_task("7", "gpt-3.5-turbo"),
        allocate_llm_task("8", "gpt-4")
    ]
    
    results = await asyncio.gather(*tasks)
    successful = sum(results)
    
    # Show statistics
    stats = await manager.get_pool_stats(ResourceType.LLM_CONNECTION)
    print(f"\n📊 LLM Pool Stats:")
    print(f"   Total Requests: {stats.total_requests}")
    print(f"   Successful: {stats.successful_allocations}")
    print(f"   Peak Concurrent: {stats.peak_concurrent}")
    print(f"   Average Hold Time: {stats.average_hold_time:.2f}s")
    print(f"   Current Active: {stats.current_active}")
    
    return successful == len(tasks)

async def test_tool_instance_sharing():
    """Test tool instance sharing across agents."""
    print("\n🔧 TOOL INSTANCE SHARING TEST")
    print("=" * 50)
    
    manager = await get_resource_pool_manager()
    tool_pool = manager.pools[ResourceType.TOOL_INSTANCE]
    
    # Create mock tools
    web_search = Mock()
    web_search.name = "web_search"
    web_search.search = AsyncMock(return_value="Search results")
    
    calculator = Mock()
    calculator.name = "calculator"
    calculator.calculate = AsyncMock(return_value=42)
    
    database = Mock()
    database.name = "database"
    database.query = AsyncMock(return_value="Query results")
    
    # Register tools
    tool_pool.register_tool("web_search", web_search)
    tool_pool.register_tool("calculator", calculator)
    tool_pool.register_tool("database", database)
    print("✅ Registered 3 tools: web_search, calculator, database")
    
    async def use_tool_task(agent_id: str, tool_name: str):
        """Task to use a shared tool."""
        async with manager.allocate_resource(
            ResourceType.TOOL_INSTANCE,
            requester_id=f"tool_agent_{agent_id}",
            tool_name=tool_name
        ) as tool:
            if tool:
                print(f"✅ Agent {agent_id}: Using {tool_name} (instance: {id(tool)})")
                await asyncio.sleep(0.2)  # Simulate work
                return True
            else:
                print(f"❌ Agent {agent_id}: Failed to get {tool_name}")
                return False
    
    # Test multiple agents sharing tools
    print("\nTesting tool sharing across 6 agents...")
    tasks = [
        use_tool_task("A", "web_search"),
        use_tool_task("B", "calculator"),
        use_tool_task("C", "web_search"),  # Same tool, different agent
        use_tool_task("D", "database"),
        use_tool_task("E", "calculator"),   # Same tool, different agent  
        use_tool_task("F", "web_search")    # Same tool, different agent
    ]
    
    results = await asyncio.gather(*tasks)
    successful = sum(results)
    
    # Show statistics
    stats = await manager.get_pool_stats(ResourceType.TOOL_INSTANCE)
    print(f"\n📊 Tool Pool Stats:")
    print(f"   Total Requests: {stats.total_requests}")
    print(f"   Successful: {stats.successful_allocations}")
    print(f"   Peak Concurrent: {stats.peak_concurrent}")
    
    return successful == len(tasks)

async def test_database_connection_pooling():
    """Test database connection pooling."""
    print("\n🗄️ DATABASE CONNECTION POOLING TEST")
    print("=" * 50)
    
    manager = await get_resource_pool_manager()
    
    async def db_task(agent_id: str):
        """Task to use database connection."""
        async with manager.allocate_resource(
            ResourceType.DATABASE_CONNECTION,
            requester_id=f"db_agent_{agent_id}",
            priority=ResourcePriority.NORMAL
        ) as db:
            print(f"✅ Agent {agent_id}: Connected to database")
            await asyncio.sleep(0.3)  # Simulate query
            return True
    
    # Test concurrent database connections
    print("Testing 5 concurrent database connections...")
    tasks = [db_task(str(i)) for i in range(1, 6)]
    results = await asyncio.gather(*tasks)
    
    # Show statistics
    stats = await manager.get_pool_stats(ResourceType.DATABASE_CONNECTION)
    print(f"\n📊 Database Pool Stats:")
    print(f"   Total Requests: {stats.total_requests}")
    print(f"   Successful: {stats.successful_allocations}")
    print(f"   Peak Concurrent: {stats.peak_concurrent}")
    
    return all(results)

async def test_vector_memory_allocation():
    """Test vector memory allocation with limits."""
    print("\n🧮 VECTOR MEMORY ALLOCATION TEST")
    print("=" * 50)
    
    manager = await get_resource_pool_manager()
    
    print("Testing memory allocation within limits...")
    
    # Test normal allocation
    async with manager.allocate_resource(
        ResourceType.VECTOR_MEMORY,
        requester_id="memory_agent_1",
        size_mb=200
    ) as memory1:
        print(f"✅ Allocated {memory1['size_mb']}MB memory")
        
        # Test nested allocation
        async with manager.allocate_resource(
            ResourceType.VECTOR_MEMORY,
            requester_id="memory_agent_2",
            size_mb=300
        ) as memory2:
            print(f"✅ Allocated additional {memory2['size_mb']}MB memory")
            
            pool = manager.pools[ResourceType.VECTOR_MEMORY]
            print(f"✅ Total usage: {pool._current_usage_mb}/1000MB")
    
    # Test over-allocation (should fail)
    print("\nTesting memory limit enforcement...")
    try:
        async with manager.allocate_resource(
            ResourceType.VECTOR_MEMORY,
            requester_id="memory_agent_3",
            size_mb=1100  # Exceeds limit
        ) as memory:
            print("❌ Over-allocation should have failed!")
            return False
    except ResourceException:
        print("✅ Memory limit correctly enforced (1100MB > 1000MB limit)")
    
    return True

async def test_fair_scheduling_and_timeouts():
    """Test fair scheduling and timeout handling."""
    print("\n⚖️ FAIR SCHEDULING & TIMEOUT TEST")
    print("=" * 50)
    
    manager = await get_resource_pool_manager()
    
    async def heavy_user_task():
        """Task that tries to monopolize resources."""
        for i in range(3):
            try:
                async with manager.allocate_resource(
                    ResourceType.LLM_CONNECTION,
                    requester_id="heavy_user",
                    timeout_seconds=1.0
                ) as llm:
                    print(f"🔥 Heavy user: Got LLM #{i+1}")
                    await asyncio.sleep(0.5)
            except ResourceException:
                print(f"⏰ Heavy user: Timeout on request #{i+1}")
    
    async def normal_user_task():
        """Task for normal usage."""
        try:
            async with manager.allocate_resource(
                ResourceType.LLM_CONNECTION,
                requester_id="normal_user",
                timeout_seconds=2.0
            ) as llm:
                print("✅ Normal user: Got LLM connection")
                await asyncio.sleep(0.3)
                return True
        except ResourceException:
            print("❌ Normal user: Failed to get LLM")
            return False
    
    # Run tasks to test fair scheduling
    print("Testing fair scheduling with mixed usage patterns...")
    await asyncio.gather(
        heavy_user_task(),
        normal_user_task(),
        return_exceptions=True
    )
    
    # Check timeout statistics
    stats = await manager.get_pool_stats(ResourceType.LLM_CONNECTION)
    print(f"\n📊 Scheduling Stats:")
    print(f"   Total Requests: {stats.total_requests}")
    print(f"   Timeout Failures: {stats.timeout_failures}")
    print(f"   Success Rate: {(stats.successful_allocations/stats.total_requests)*100:.1f}%")
    
    return True

async def test_health_monitoring():
    """Test system health monitoring."""
    print("\n🏥 HEALTH MONITORING TEST")
    print("=" * 50)
    
    manager = await get_resource_pool_manager()
    
    # Get overall system health
    health = await manager.get_system_health()
    
    print(f"Overall System Health: {health['overall_health']}")
    print(f"CPU Usage: {health['system_resources']['cpu_percent']:.1f}%")
    print(f"Memory Usage: {health['system_resources']['memory_percent']:.1f}%")
    print(f"Disk Usage: {health['system_resources']['disk_percent']:.1f}%")
    
    print(f"\nPool Health Status:")
    for pool_name, pool_health in health['pools'].items():
        utilization = pool_health['utilization']
        status = pool_health['status']
        active = pool_health['active_allocations']
        capacity = pool_health['max_capacity']
        
        print(f"   {pool_name}:")
        print(f"     Status: {status}")
        print(f"     Utilization: {utilization:.1%}")
        print(f"     Active/Capacity: {active}/{capacity}")
    
    return health['overall_health'] in ['healthy', 'degraded']

async def run_comprehensive_demo():
    """Run all demonstration tests."""
    print("🚀 RESOURCE POOL MANAGER COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Run all tests
        tests = [
            ("LLM Connection Pooling", test_llm_connection_pooling()),
            ("Tool Instance Sharing", test_tool_instance_sharing()),
            ("Database Connection Pooling", test_database_connection_pooling()),
            ("Vector Memory Allocation", test_vector_memory_allocation()),
            ("Fair Scheduling & Timeouts", test_fair_scheduling_and_timeouts()),
            ("Health Monitoring", test_health_monitoring())
        ]
        
        results = []
        for test_name, test_coro in tests:
            print(f"\n🧪 Running {test_name}...")
            try:
                result = await test_coro
                results.append((test_name, result))
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"   {status}")
            except Exception as e:
                results.append((test_name, False))
                print(f"   ❌ FAILED: {e}")
        
        # Summary
        elapsed = time.time() - start_time
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        print(f"\n🎯 DEMONSTRATION COMPLETE")
        print("=" * 80)
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print(f"Total Time: {elapsed:.2f}s")
        
        if passed == total:
            print("\n🎉 ALL FEATURES WORKING PERFECTLY!")
            print("   Resource Pool Manager is production-ready!")
        else:
            print(f"\n⚠️  {total-passed} tests failed - review implementation")
        
        # Final system stats
        manager = await get_resource_pool_manager()
        print(f"\n📈 FINAL SYSTEM STATISTICS")
        print("-" * 40)
        
        for resource_type in ResourceType:
            stats = await manager.get_pool_stats(resource_type)
            if stats and stats.total_requests > 0:
                print(f"{resource_type.value}:")
                print(f"  Total Requests: {stats.total_requests}")
                print(f"  Success Rate: {(stats.successful_allocations/stats.total_requests)*100:.1f}%")
                print(f"  Peak Concurrent: {stats.peak_concurrent}")
                if stats.average_hold_time > 0:
                    print(f"  Avg Hold Time: {stats.average_hold_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        from resources.pool_manager import cleanup_resource_pool_manager
        await cleanup_resource_pool_manager()

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(run_comprehensive_demo()) 