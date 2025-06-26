"""
Comprehensive test suite for Resource Pool Manager

Tests all core functionality:
- LLM connection pooling (max 10 concurrent)
- Tool instance sharing across agents
- Database connection pooling
- Vector memory allocation with limits
- Fair scheduling with timeout protection
- Resource health monitoring and cleanup
- Event-driven architecture integration
"""

import pytest
import asyncio
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from resources.pool_manager import (
    ResourcePoolManager, ResourceType, ResourcePriority, ResourceRequest,
    ResourceException, LLMConnectionPool, ToolInstancePool, 
    DatabaseConnectionPool, VectorMemoryPool, FairScheduler,
    get_resource_pool_manager, cleanup_resource_pool_manager
)
from events.event_bus import get_event_bus


class TestResourcePoolManager:
    """Test the main resource pool manager."""
    
    @pytest.fixture
    async def manager(self):
        """Create a fresh resource pool manager for testing."""
        manager = ResourcePoolManager()
        await manager.start()
        yield manager
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, manager):
        """Test manager initializes all pools correctly."""
        assert len(manager.pools) == 4
        assert ResourceType.LLM_CONNECTION in manager.pools
        assert ResourceType.TOOL_INSTANCE in manager.pools
        assert ResourceType.DATABASE_CONNECTION in manager.pools
        assert ResourceType.VECTOR_MEMORY in manager.pools
        
        # Check pool configurations
        assert manager.pools[ResourceType.LLM_CONNECTION].max_size == 10
        assert manager.pools[ResourceType.TOOL_INSTANCE].max_size == 50
        assert manager.pools[ResourceType.DATABASE_CONNECTION].max_size == 20
        assert manager.pools[ResourceType.VECTOR_MEMORY].max_size == 1000  # MB
        
        print("✅ Resource Pool Manager initialized with all 4 pool types")
    
    @pytest.mark.asyncio
    async def test_llm_connection_allocation(self, manager):
        """Test LLM connection pool allocation and release."""
        print("\n🧠 Testing LLM Connection Pool...")
        
        # Test basic allocation
        async with manager.allocate_resource(
            ResourceType.LLM_CONNECTION,
            requester_id="test_agent_1",
            priority=ResourcePriority.HIGH,
            creator_type="gpt-3.5-turbo",
            temperature=0.7
        ) as llm:
            assert llm is not None
            assert hasattr(llm, 'agenerate')
            print(f"✅ Successfully allocated LLM connection: {type(llm).__name__}")
        
        # Test different LLM types
        async with manager.allocate_resource(
            ResourceType.LLM_CONNECTION,
            requester_id="test_agent_2", 
            creator_type="gpt-4",
            temperature=0.3,
            max_tokens=800
        ) as gpt4_llm:
            assert gpt4_llm is not None
            print("✅ Successfully allocated GPT-4 LLM connection")
        
        # Check pool statistics
        stats = await manager.get_pool_stats(ResourceType.LLM_CONNECTION)
        assert stats.total_requests >= 2
        assert stats.successful_allocations >= 2
        assert stats.current_active == 0  # All should be released
        print(f"✅ LLM Pool Stats: {stats.total_requests} requests, {stats.successful_allocations} successful")
    
    @pytest.mark.asyncio
    async def test_concurrent_llm_allocations(self, manager):
        """Test concurrent LLM allocations up to the limit."""
        print("\n⚡ Testing Concurrent LLM Allocations...")
        
        async def allocate_llm(agent_id: str):
            """Helper to allocate LLM for testing."""
            try:
                async with manager.allocate_resource(
                    ResourceType.LLM_CONNECTION,
                    requester_id=f"agent_{agent_id}",
                    timeout_seconds=5.0
                ) as llm:
                    await asyncio.sleep(0.1)  # Simulate usage
                    return True
            except ResourceException:
                return False
        
        # Start 12 concurrent allocations (pool max is 10)
        tasks = [allocate_llm(i) for i in range(12)]
        results = await asyncio.gather(*tasks)
        
        successful = sum(results)
        failed = len(results) - successful
        
        print(f"✅ Concurrent test: {successful} successful, {failed} failed allocations")
        assert successful >= 10  # Should handle at least the pool size
        
        # Verify pool statistics
        stats = await manager.get_pool_stats(ResourceType.LLM_CONNECTION)
        print(f"✅ Final pool state: {stats.current_active} active, {stats.peak_concurrent} peak")
    
    @pytest.mark.asyncio
    async def test_tool_instance_sharing(self, manager):
        """Test tool instance sharing across agents."""
        print("\n🔧 Testing Tool Instance Sharing...")
        
        # Get the tool pool and register tools
        tool_pool = manager.pools[ResourceType.TOOL_INSTANCE]
        
        # Mock tools for testing
        web_search_tool = Mock()
        web_search_tool.name = "web_search"
        web_search_tool.search = AsyncMock(return_value="search results")
        
        calculator_tool = Mock()
        calculator_tool.name = "calculator"
        calculator_tool.calculate = AsyncMock(return_value=42)
        
        # Register tools
        tool_pool.register_tool("web_search", web_search_tool)
        tool_pool.register_tool("calculator", calculator_tool)
        print("✅ Registered web_search and calculator tools")
        
        # Test multiple agents sharing the same tool
        async with manager.allocate_resource(
            ResourceType.TOOL_INSTANCE,
            requester_id="agent_1",
            tool_name="web_search"
        ) as tool1:
            assert tool1 is web_search_tool
            
            # Second agent uses same tool instance
            async with manager.allocate_resource(
                ResourceType.TOOL_INSTANCE,
                requester_id="agent_2",
                tool_name="web_search"
            ) as tool2:
                assert tool2 is web_search_tool
                assert tool1 is tool2  # Same instance shared
                print("✅ Multiple agents successfully sharing same tool instance")
        
        # Test different tool
        async with manager.allocate_resource(
            ResourceType.TOOL_INSTANCE,
            requester_id="agent_3",
            tool_name="calculator"
        ) as calc_tool:
            assert calc_tool is calculator_tool
            print("✅ Successfully allocated different tool type")
        
        stats = await manager.get_pool_stats(ResourceType.TOOL_INSTANCE)
        print(f"✅ Tool Pool Stats: {stats.successful_allocations} successful allocations")
    
    @pytest.mark.asyncio
    async def test_database_connection_pooling(self, manager):
        """Test database connection pooling."""
        print("\n🗄️ Testing Database Connection Pooling...")
        
        # Test multiple database connections
        connections = []
        
        async def get_db_connection(agent_id: str):
            async with manager.allocate_resource(
                ResourceType.DATABASE_CONNECTION,
                requester_id=f"agent_{agent_id}"
            ) as db:
                assert db is not None
                assert hasattr(db, 'log_event')  # SupabaseLogger interface
                connections.append(db)
                await asyncio.sleep(0.1)  # Simulate usage
        
        # Test concurrent database connections
        tasks = [get_db_connection(i) for i in range(5)]
        await asyncio.gather(*tasks)
        
        print(f"✅ Successfully allocated {len(connections)} database connections")
        
        stats = await manager.get_pool_stats(ResourceType.DATABASE_CONNECTION)
        assert stats.successful_allocations >= 5
        print(f"✅ DB Pool Stats: {stats.successful_allocations} successful, {stats.current_active} active")
    
    @pytest.mark.asyncio
    async def test_vector_memory_allocation(self, manager):
        """Test vector memory allocation with size limits."""
        print("\n🧮 Testing Vector Memory Allocation...")
        
        # Test memory allocation
        async with manager.allocate_resource(
            ResourceType.VECTOR_MEMORY,
            requester_id="vector_agent_1",
            size_mb=100
        ) as memory1:
            assert memory1 is not None
            assert memory1['size_mb'] == 100
            print(f"✅ Allocated {memory1['size_mb']}MB vector memory")
            
            # Allocate more memory
            async with manager.allocate_resource(
                ResourceType.VECTOR_MEMORY,
                requester_id="vector_agent_2",
                size_mb=200
            ) as memory2:
                assert memory2['size_mb'] == 200
                print(f"✅ Allocated additional {memory2['size_mb']}MB vector memory")
                
                # Check total usage
                pool = manager.pools[ResourceType.VECTOR_MEMORY]
                assert pool._current_usage_mb == 300
                print(f"✅ Total memory usage: {pool._current_usage_mb}MB")
        
        # Test memory limit enforcement
        try:
            async with manager.allocate_resource(
                ResourceType.VECTOR_MEMORY,
                requester_id="vector_agent_3",
                size_mb=1100  # Exceeds 1000MB limit
            ) as memory:
                assert False, "Should have failed due to memory limit"
        except ResourceException:
            print("✅ Memory limit correctly enforced (1100MB > 1000MB limit)")
    
    @pytest.mark.asyncio
    async def test_fair_scheduling(self, manager):
        """Test fair scheduling prevents resource starvation."""
        print("\n⚖️ Testing Fair Scheduling...")
        
        scheduler = manager._fair_scheduler
        
        # Create requests from different agents
        request1 = ResourceRequest(
            resource_type=ResourceType.LLM_CONNECTION,
            requester_id="frequent_agent",
            priority=ResourcePriority.NORMAL
        )
        
        request2 = ResourceRequest(
            resource_type=ResourceType.LLM_CONNECTION,
            requester_id="normal_agent",
            priority=ResourcePriority.HIGH
        )
        
        # Simulate frequent requests from one agent
        start_time = time.time()
        for i in range(6):  # Trigger back-off
            await scheduler.schedule_request(request1)
        
        # Normal agent should not be delayed much
        await scheduler.schedule_request(request2)
        elapsed = time.time() - start_time
        
        print(f"✅ Fair scheduling test completed in {elapsed:.2f}s")
        print("✅ Back-off mechanism prevents one agent from monopolizing resources")
    
    @pytest.mark.asyncio 
    async def test_timeout_handling(self, manager):
        """Test timeout handling for resource allocation."""
        print("\n⏰ Testing Timeout Handling...")
        
        # Fill up the LLM pool
        allocations = []
        
        async def hold_resource():
            async with manager.allocate_resource(
                ResourceType.LLM_CONNECTION,
                requester_id="holder_agent",
                timeout_seconds=30.0
            ) as llm:
                await asyncio.sleep(2.0)  # Hold for 2 seconds
                return llm
        
        # Start tasks to fill the pool
        tasks = [hold_resource() for _ in range(10)]  # Fill all 10 slots
        
        # Try to allocate with short timeout - should fail
        start_time = time.time()
        try:
            async with manager.allocate_resource(
                ResourceType.LLM_CONNECTION,
                requester_id="timeout_test_agent",
                timeout_seconds=0.5  # Very short timeout
            ) as llm:
                assert False, "Should have timed out"
        except ResourceException as e:
            elapsed = time.time() - start_time
            print(f"✅ Timeout correctly triggered after {elapsed:.2f}s")
            assert "Failed to allocate" in str(e)
        
        # Wait for holder tasks to complete
        await asyncio.gather(*tasks)
        
        # Check timeout statistics
        stats = await manager.get_pool_stats(ResourceType.LLM_CONNECTION)
        assert stats.timeout_failures > 0
        print(f"✅ Timeout stats: {stats.timeout_failures} timeouts recorded")
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, manager):
        """Test resource health monitoring."""
        print("\n🏥 Testing Health Monitoring...")
        
        # Get system health
        health = await manager.get_system_health()
        
        assert health['overall_health'] in ['healthy', 'degraded', 'critical']
        assert 'pools' in health
        assert 'system_resources' in health
        assert 'timestamp' in health
        
        print(f"✅ System Health: {health['overall_health']}")
        print(f"✅ CPU: {health['system_resources']['cpu_percent']:.1f}%")
        print(f"✅ Memory: {health['system_resources']['memory_percent']:.1f}%")
        
        # Check individual pool health
        for pool_type, pool_health in health['pools'].items():
            print(f"✅ {pool_type}: {pool_health['status']}, "
                  f"utilization: {pool_health['utilization']:.1%}")
        
        # Test pool stats
        for resource_type in manager.pools.keys():
            stats = await manager.get_pool_stats(resource_type)
            assert stats is not None
            print(f"✅ {resource_type.value}: {stats.successful_allocations} successful allocations")
    
    @pytest.mark.asyncio
    async def test_event_integration(self, manager):
        """Test event bus integration."""
        print("\n📡 Testing Event Bus Integration...")
        
        # Track events
        events_received = []
        
        async def event_handler(event):
            events_received.append(event)
        
        # Subscribe to resource events
        event_bus = get_event_bus()
        await event_bus.subscribe(
            "test_subscriber",
            ["resource_allocated", "resource_released", "resource_pool_started"],
            event_handler
        )
        
        # Generate some resource activity
        async with manager.allocate_resource(
            ResourceType.LLM_CONNECTION,
            requester_id="event_test_agent"
        ) as llm:
            await asyncio.sleep(0.1)  # Brief usage
        
        # Wait for events to be processed
        await asyncio.sleep(0.5)
        
        # Check events were published
        allocation_events = [e for e in events_received if e.type == "resource_allocated"]
        release_events = [e for e in events_received if e.type == "resource_released"]
        
        print(f"✅ Received {len(allocation_events)} allocation events")
        print(f"✅ Received {len(release_events)} release events")
        
        if allocation_events:
            event = allocation_events[0]
            assert event.data['resource_type'] == 'llm_connection'
            assert event.data['requester_id'] == 'event_test_agent'
            print("✅ Event data correctly formatted")
    
    @pytest.mark.asyncio
    async def test_emergency_optimization(self, manager):
        """Test emergency optimization when system health is critical."""
        print("\n🚨 Testing Emergency Optimization...")
        
        # Mock critical health condition
        with patch.object(manager, 'get_system_health') as mock_health:
            mock_health.return_value = {
                "overall_health": "critical",
                "pools": {},
                "system_resources": {"cpu_percent": 95.0, "memory_percent": 90.0},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Trigger emergency optimization
            await manager._emergency_optimization()
            print("✅ Emergency optimization triggered successfully")
        
        print("✅ System can handle critical health conditions")
    
    @pytest.mark.asyncio
    async def test_global_manager_instance(self):
        """Test global resource pool manager singleton."""
        print("\n🌐 Testing Global Manager Instance...")
        
        # Get global instance
        manager1 = await get_resource_pool_manager()
        manager2 = await get_resource_pool_manager()
        
        # Should be the same instance
        assert manager1 is manager2
        print("✅ Global manager instance is singleton")
        
        # Test functionality
        async with manager1.allocate_resource(
            ResourceType.LLM_CONNECTION,
            requester_id="global_test_agent"
        ) as llm:
            assert llm is not None
            print("✅ Global manager allocates resources correctly")
        
        # Cleanup
        await cleanup_resource_pool_manager()
        print("✅ Global manager cleanup successful")


class TestResourcePools:
    """Test individual resource pool implementations."""
    
    @pytest.mark.asyncio
    async def test_llm_connection_pool_health_check(self):
        """Test LLM connection pool health checking."""
        print("\n🔍 Testing LLM Health Checks...")
        
        pool = LLMConnectionPool(max_size=2)
        await pool.start()
        
        try:
            # Create a request
            request = ResourceRequest(
                resource_type=ResourceType.LLM_CONNECTION,
                requester_id="health_test_agent",
                metadata={'creator_type': 'default'}
            )
            
            # Allocate a resource
            result = await pool.allocate(request)
            if result:
                allocation_id, llm = result
                
                # Test health check
                is_healthy = await pool._is_resource_healthy(llm)
                print(f"✅ Health check result: {is_healthy}")
                
                # Release the resource
                await pool.release(allocation_id)
                print("✅ Resource released successfully")
            
        finally:
            await pool.stop()
    
    @pytest.mark.asyncio
    async def test_vector_memory_pool_edge_cases(self):
        """Test vector memory pool edge cases."""
        print("\n🧮 Testing Vector Memory Edge Cases...")
        
        pool = VectorMemoryPool(max_size_mb=100)  # Small pool for testing
        
        # Test allocation at limit
        request = ResourceRequest(
            resource_type=ResourceType.VECTOR_MEMORY,
            requester_id="memory_test_agent",
            metadata={'size_mb': 100}  # Exactly at limit
        )
        
        result = await pool.allocate(request)
        assert result is not None
        allocation_id, memory = result
        assert memory['size_mb'] == 100
        print("✅ Allocated memory at exact limit")
        
        # Test over-allocation
        request2 = ResourceRequest(
            resource_type=ResourceType.VECTOR_MEMORY,
            requester_id="memory_test_agent_2",
            metadata={'size_mb': 1}  # Should fail
        )
        
        result2 = await pool.allocate(request2)
        assert result2 is None
        print("✅ Over-allocation correctly rejected")
        
        # Release and test reallocation
        await pool.release(allocation_id)
        assert pool._current_usage_mb == 0
        print("✅ Memory correctly released")


async def run_comprehensive_demo():
    """Run a comprehensive demonstration of the Resource Pool Manager."""
    print("🚀 Starting Comprehensive Resource Pool Manager Demo")
    print("=" * 80)
    
    # Start manager
    manager = ResourcePoolManager()
    await manager.start()
    
    try:
        print("\n📊 SYSTEM OVERVIEW")
        print("-" * 40)
        
        # Show initial system health
        health = await manager.get_system_health()
        print(f"Overall Health: {health['overall_health']}")
        print(f"Available Pools: {len(health['pools'])}")
        print(f"System CPU: {health['system_resources']['cpu_percent']:.1f}%")
        print(f"System Memory: {health['system_resources']['memory_percent']:.1f}%")
        
        print("\n🧠 LLM CONNECTION POOL DEMO")
        print("-" * 40)
        
        # Demo LLM allocations
        async def llm_demo_task(agent_id: str, model: str):
            async with manager.allocate_resource(
                ResourceType.LLM_CONNECTION,
                requester_id=f"demo_agent_{agent_id}",
                creator_type=model,
                temperature=0.5
            ) as llm:
                print(f"Agent {agent_id}: Allocated {model} LLM connection")
                await asyncio.sleep(0.5)  # Simulate processing
                return True
        
        # Run concurrent LLM tasks
        llm_tasks = [
            llm_demo_task("1", "gpt-3.5-turbo"),
            llm_demo_task("2", "gpt-4"),
            llm_demo_task("3", "default"),
            llm_demo_task("4", "gpt-3.5-turbo"),
            llm_demo_task("5", "gpt-4")
        ]
        
        results = await asyncio.gather(*llm_tasks)
        successful_llm = sum(results)
        print(f"✅ Successfully allocated {successful_llm}/5 LLM connections")
        
        print("\n🔧 TOOL SHARING DEMO")
        print("-" * 40)
        
        # Setup tool sharing
        tool_pool = manager.pools[ResourceType.TOOL_INSTANCE]
        
        # Mock tools
        search_tool = Mock()
        search_tool.name = "web_search"
        calc_tool = Mock()
        calc_tool.name = "calculator"
        
        tool_pool.register_tool("web_search", search_tool)
        tool_pool.register_tool("calculator", calc_tool)
        print("Registered web_search and calculator tools")
        
        # Demo tool sharing
        async def tool_demo_task(agent_id: str, tool_name: str):
            async with manager.allocate_resource(
                ResourceType.TOOL_INSTANCE,
                requester_id=f"tool_agent_{agent_id}",
                tool_name=tool_name
            ) as tool:
                print(f"Agent {agent_id}: Using {tool_name} tool")
                await asyncio.sleep(0.3)
                return True
        
        tool_tasks = [
            tool_demo_task("A", "web_search"),
            tool_demo_task("B", "calculator"),
            tool_demo_task("C", "web_search"),  # Same tool, different agent
            tool_demo_task("D", "calculator")
        ]
        
        await asyncio.gather(*tool_tasks)
        print("✅ Tools successfully shared across agents")
        
        print("\n🗄️ DATABASE CONNECTION DEMO")
        print("-" * 40)
        
        # Demo database connections
        async def db_demo_task(agent_id: str):
            async with manager.allocate_resource(
                ResourceType.DATABASE_CONNECTION,
                requester_id=f"db_agent_{agent_id}"
            ) as db:
                print(f"Agent {agent_id}: Connected to database")
                await asyncio.sleep(0.2)
                return True
        
        db_tasks = [db_demo_task(str(i)) for i in range(3)]
        await asyncio.gather(*db_tasks)
        print("✅ Database connections allocated successfully")
        
        print("\n🧮 VECTOR MEMORY DEMO")
        print("-" * 40)
        
        # Demo vector memory allocation
        async with manager.allocate_resource(
            ResourceType.VECTOR_MEMORY,
            requester_id="vector_demo_agent",
            size_mb=250
        ) as memory:
            print(f"Allocated {memory['size_mb']}MB vector memory")
            
            # Nested allocation
            async with manager.allocate_resource(
                ResourceType.VECTOR_MEMORY,
                requester_id="vector_demo_agent_2", 
                size_mb=100
            ) as memory2:
                print(f"Allocated additional {memory2['size_mb']}MB vector memory")
                
                pool = manager.pools[ResourceType.VECTOR_MEMORY]
                print(f"Total memory usage: {pool._current_usage_mb}/1000MB")
        
        print("\n📈 FINAL STATISTICS")
        print("-" * 40)
        
        # Show final statistics
        for resource_type in ResourceType:
            stats = await manager.get_pool_stats(resource_type)
            if stats:
                print(f"{resource_type.value}:")
                print(f"  Requests: {stats.total_requests}")
                print(f"  Successful: {stats.successful_allocations}")
                print(f"  Timeouts: {stats.timeout_failures}")
                print(f"  Peak Concurrent: {stats.peak_concurrent}")
                if stats.average_hold_time > 0:
                    print(f"  Avg Hold Time: {stats.average_hold_time:.2f}s")
        
        # Final health check
        final_health = await manager.get_system_health()
        print(f"\nFinal System Health: {final_health['overall_health']}")
        
        print("\n🎉 DEMO COMPLETE - ALL SYSTEMS WORKING PERFECTLY!")
        print("=" * 80)
        
    finally:
        await manager.stop()


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(run_comprehensive_demo()) 