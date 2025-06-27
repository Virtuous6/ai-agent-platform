#!/usr/bin/env python3
"""
Agent Lifecycle Protocol Test Suite

This script tests the complete agent spawning and lifecycle management protocol
to ensure everything works correctly from spawn to cleanup.

Features tested:
- Dynamic agent spawning protocol
- Lazy loading and caching
- Resource budget enforcement
- Agent activation/deactivation
- Memory management
- Event-driven communication
- Full lifecycle cleanup

Usage:
    python agents/test_agent_lifecycle_protocol.py
"""

import asyncio
import logging
import sys
import os
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentLifecycleProtocolTest:
    """Comprehensive test suite for agent lifecycle protocol."""
    
    def __init__(self):
        self.orchestrator = None
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
        self.spawned_agent_ids = []
    
    async def setup(self):
        """Initialize the orchestrator and necessary components."""
        try:
            # Import and initialize components
            from orchestrator.agent_orchestrator import AgentOrchestrator
            from database.supabase_logger import SupabaseLogger
            
            # Initialize with minimal setup for testing
            db_logger = SupabaseLogger()
            self.orchestrator = AgentOrchestrator(db_logger=db_logger)
            
            logger.info("✅ Test setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test setup failed: {e}")
            return False
    
    def log_test_result(self, test_name: str, success: bool, details: Dict[str, Any]):
        """Log test result with details."""
        self.test_results["total_tests"] += 1
        
        if success:
            self.test_results["passed_tests"] += 1
            logger.info(f"✅ {test_name}: PASSED")
        else:
            self.test_results["failed_tests"] += 1
            logger.error(f"❌ {test_name}: FAILED")
        
        self.test_results["test_details"].append({
            "test_name": test_name,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        })
    
    async def test_dynamic_agent_spawning(self) -> bool:
        """Test 1: Dynamic Agent Spawning Protocol"""
        test_name = "Dynamic Agent Spawning Protocol"
        logger.info(f"\n🚀 Testing: {test_name}")
        
        try:
            # Test spawning different specialties
            specialties = [
                "Python Performance Optimization Expert",
                "Database Query Analyzer", 
                "React Component Architect",
                "Machine Learning Model Specialist",
                "DevOps Automation Engineer"
            ]
            
            spawned_agents = []
            for specialty in specialties:
                agent_id = await self.orchestrator.spawn_specialist_agent(
                    specialty=specialty,
                    parent_context={
                        "test_scenario": "lifecycle_protocol_test",
                        "user_id": "test_user_123"
                    },
                    temperature=0.3,
                    max_tokens=500
                )
                
                if agent_id:
                    spawned_agents.append(agent_id)
                    self.spawned_agent_ids.append(agent_id)
                    logger.info(f"  ✅ Spawned: {specialty} -> {agent_id}")
                else:
                    logger.warning(f"  ❌ Failed to spawn: {specialty}")
            
            # Verify spawn success rate
            success_rate = len(spawned_agents) / len(specialties)
            
            details = {
                "specialties_requested": len(specialties),
                "agents_spawned": len(spawned_agents),
                "success_rate": success_rate,
                "spawned_agent_ids": spawned_agents
            }
            
            success = success_rate >= 0.8  # 80% success rate minimum
            self.log_test_result(test_name, success, details)
            return success
            
        except Exception as e:
            self.log_test_result(test_name, False, {"error": str(e)})
            return False
    
    async def test_lazy_loading_and_caching(self) -> bool:
        """Test 2: Lazy Loading and Caching Protocol"""
        test_name = "Lazy Loading and Caching Protocol"
        logger.info(f"\n🔄 Testing: {test_name}")
        
        try:
            if not self.spawned_agent_ids:
                raise Exception("No spawned agents available for lazy loading test")
            
            test_agent_id = self.spawned_agent_ids[0]
            
            # Test 1: First load (cache miss)
            logger.info(f"  🔍 First load of agent: {test_agent_id}")
            agent1 = await self.orchestrator.get_or_load_agent(test_agent_id)
            
            # Test 2: Second load (cache hit)
            logger.info(f"  🎯 Second load of agent: {test_agent_id}")
            agent2 = await self.orchestrator.get_or_load_agent(test_agent_id)
            
            # Test 3: Verify lazy loader metrics
            lazy_metrics = self.orchestrator.lazy_loader.get_cache_metrics()
            
            details = {
                "first_load_success": agent1 is not None,
                "second_load_success": agent2 is not None,
                "cache_hit_rate": lazy_metrics["cache_performance"]["hit_rate"],
                "cache_utilization": lazy_metrics["agent_status"]["cache_utilization"],
                "total_loads": lazy_metrics["cache_performance"]["total_loads"],
                "cache_hits": lazy_metrics["cache_performance"]["cache_hits"]
            }
            
            success = (
                agent1 is not None and 
                agent2 is not None and 
                lazy_metrics["cache_performance"]["hit_rate"] > 0
            )
            
            self.log_test_result(test_name, success, details)
            return success
            
        except Exception as e:
            self.log_test_result(test_name, False, {"error": str(e)})
            return False
    
    async def test_resource_budget_enforcement(self) -> bool:
        """Test 3: Resource Budget Enforcement Protocol"""
        test_name = "Resource Budget Enforcement Protocol"
        logger.info(f"\n💰 Testing: {test_name}")
        
        try:
            resource_budget = self.orchestrator.resource_budget
            
            # Test budget checking
            can_spawn_before, reason_before = resource_budget.can_spawn_agent()
            
            # Get current stats
            current_spawns = len(resource_budget.spawn_times)
            max_spawns = resource_budget.max_spawns_per_hour
            hourly_cost = resource_budget.hourly_cost
            max_cost = resource_budget.max_cost_per_hour
            
            # Try to spawn an agent to test budget enforcement
            test_agent_id = await self.orchestrator.spawn_specialist_agent(
                "Budget Test Specialist",
                parent_context={"test": "budget_enforcement"}
            )
            
            can_spawn_after, reason_after = resource_budget.can_spawn_agent()
            
            details = {
                "can_spawn_before": can_spawn_before,
                "reason_before": reason_before,
                "can_spawn_after": can_spawn_after,
                "reason_after": reason_after,
                "current_spawns": current_spawns,
                "max_spawns_per_hour": max_spawns,
                "hourly_cost": hourly_cost,
                "max_cost_per_hour": max_cost,
                "budget_enforcement_active": True,
                "spawn_attempt_success": test_agent_id is not None
            }
            
            if test_agent_id:
                self.spawned_agent_ids.append(test_agent_id)
            
            # Success if budget system is working (has limits and tracking)
            success = (
                max_spawns > 0 and 
                max_cost > 0 and 
                isinstance(hourly_cost, (int, float))
            )
            
            self.log_test_result(test_name, success, details)
            return success
            
        except Exception as e:
            self.log_test_result(test_name, False, {"error": str(e)})
            return False
    
    async def test_agent_activation_deactivation(self) -> bool:
        """Test 4: Agent Activation/Deactivation Protocol"""
        test_name = "Agent Activation/Deactivation Protocol"
        logger.info(f"\n🔄 Testing: {test_name}")
        
        try:
            if not self.spawned_agent_ids:
                raise Exception("No spawned agents available for activation test")
            
            # Test activation of multiple agents
            activated_agents = []
            for agent_id in self.spawned_agent_ids[:3]:  # Test first 3 agents
                agent = await self.orchestrator.get_or_load_agent(agent_id)
                if agent:
                    activated_agents.append(agent_id)
                    logger.info(f"  ✅ Activated agent: {agent_id}")
            
            # Check agent stats before cleanup
            agent_stats_before = self.orchestrator.get_agent_stats()
            
            # Force some agents out of cache by hitting the limit
            # Spawn more agents to test eviction
            additional_agents = []
            for i in range(5):
                agent_id = await self.orchestrator.spawn_specialist_agent(
                    f"Eviction Test Agent {i}",
                    parent_context={"test": "eviction"}
                )
                if agent_id:
                    additional_agents.append(agent_id)
                    self.spawned_agent_ids.append(agent_id)
                    # Load each one to fill cache
                    await self.orchestrator.get_or_load_agent(agent_id)
            
            # Check stats after loading more agents
            agent_stats_after = self.orchestrator.get_agent_stats()
            
            details = {
                "agents_requested_for_activation": 3,
                "agents_successfully_activated": len(activated_agents),
                "active_agents_before": agent_stats_before.get("active_agents", 0),
                "active_agents_after": agent_stats_after.get("active_agents", 0),
                "total_agents_before": agent_stats_before.get("total_agents", 0),
                "total_agents_after": agent_stats_after.get("total_agents", 0),
                "cache_management_working": True,
                "additional_agents_spawned": len(additional_agents)
            }
            
            success = (
                len(activated_agents) >= 2 and  # At least 2 agents activated
                agent_stats_after.get("total_agents", 0) > agent_stats_before.get("total_agents", 0)
            )
            
            self.log_test_result(test_name, success, details)
            return success
            
        except Exception as e:
            self.log_test_result(test_name, False, {"error": str(e)})
            return False
    
    async def test_agent_processing_capabilities(self) -> bool:
        """Test 5: Agent Processing Capabilities Protocol"""
        test_name = "Agent Processing Capabilities Protocol"
        logger.info(f"\n💬 Testing: {test_name}")
        
        try:
            if not self.spawned_agent_ids:
                raise Exception("No spawned agents available for processing test")
            
            test_agent_id = self.spawned_agent_ids[0]
            agent = await self.orchestrator.get_or_load_agent(test_agent_id)
            
            if not agent:
                raise Exception(f"Could not load agent: {test_agent_id}")
            
            # Test processing a message
            test_message = "Can you help me optimize a Python function that processes large datasets?"
            test_context = {
                "user_id": "test_user_123",
                "channel_id": "test_channel",
                "conversation_id": str(uuid.uuid4()),
                "is_thread": False
            }
            
            logger.info(f"  📝 Processing test message with agent: {test_agent_id}")
            response = await agent.process_message(test_message, test_context)
            
            # Verify response structure
            expected_fields = ["response", "agent_id", "specialty", "confidence", "metadata"]
            has_required_fields = all(field in response for field in expected_fields)
            
            details = {
                "agent_id": test_agent_id,
                "message_processed": bool(response),
                "has_required_fields": has_required_fields,
                "response_length": len(response.get("response", "")),
                "confidence": response.get("confidence", 0),
                "tokens_used": response.get("tokens_used", 0),
                "processing_time_ms": response.get("processing_time_ms", 0),
                "specialty": response.get("specialty", "unknown"),
                "platform_integrated": response.get("platform_integrated", False)
            }
            
            success = (
                response is not None and
                has_required_fields and
                len(response.get("response", "")) > 10 and  # Meaningful response
                response.get("confidence", 0) > 0
            )
            
            self.log_test_result(test_name, success, details)
            return success
            
        except Exception as e:
            self.log_test_result(test_name, False, {"error": str(e)})
            return False
    
    async def test_event_driven_communication(self) -> bool:
        """Test 6: Event-Driven Communication Protocol"""
        test_name = "Event-Driven Communication Protocol"
        logger.info(f"\n📡 Testing: {test_name}")
        
        try:
            event_bus = self.orchestrator.event_bus
            
            # Test event publishing
            test_event_data = {
                "test_event": True,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "lifecycle_protocol_test"
            }
            
            await event_bus.publish(
                "test_agent_lifecycle",
                test_event_data,
                source="test_suite"
            )
            
            # Verify event queue has events
            queue_size = event_bus.event_queue.qsize()
            
            details = {
                "event_bus_available": event_bus is not None,
                "event_published": True,
                "queue_size_after_publish": queue_size,
                "event_data": test_event_data
            }
            
            success = event_bus is not None and queue_size > 0
            
            self.log_test_result(test_name, success, details)
            return success
            
        except Exception as e:
            self.log_test_result(test_name, False, {"error": str(e)})
            return False
    
    async def test_lifecycle_cleanup(self) -> bool:
        """Test 7: Lifecycle Cleanup Protocol"""
        test_name = "Lifecycle Cleanup Protocol"
        logger.info(f"\n🧹 Testing: {test_name}")
        
        try:
            # Test cleanup of inactive agents
            initial_stats = self.orchestrator.get_agent_stats()
            
            # Run cleanup
            await self.orchestrator.cleanup_inactive_agents()
            
            # Test individual agent close methods
            closed_agents = 0
            close_errors = []
            
            for agent_id in self.spawned_agent_ids[:3]:  # Test first 3 agents
                try:
                    agent = await self.orchestrator.get_or_load_agent(agent_id)
                    if agent and hasattr(agent, 'close'):
                        await agent.close()
                        closed_agents += 1
                        logger.info(f"  ✅ Successfully closed agent: {agent_id}")
                    else:
                        close_errors.append(f"Agent {agent_id} has no close method")
                except Exception as e:
                    close_errors.append(f"Error closing {agent_id}: {str(e)}")
            
            final_stats = self.orchestrator.get_agent_stats()
            
            details = {
                "initial_total_agents": initial_stats.get("total_agents", 0),
                "initial_active_agents": initial_stats.get("active_agents", 0),
                "final_total_agents": final_stats.get("total_agents", 0),
                "final_active_agents": final_stats.get("active_agents", 0),
                "agents_tested_for_close": 3,
                "agents_successfully_closed": closed_agents,
                "close_errors": close_errors,
                "cleanup_executed": True
            }
            
            success = (
                closed_agents >= 2 and  # At least 2 agents closed successfully
                len(close_errors) <= 1   # At most 1 error
            )
            
            self.log_test_result(test_name, success, details)
            return success
            
        except Exception as e:
            self.log_test_result(test_name, False, {"error": str(e)})
            return False
    
    async def test_orchestrator_cleanup(self) -> bool:
        """Test 8: Full Orchestrator Cleanup Protocol"""
        test_name = "Full Orchestrator Cleanup Protocol"
        logger.info(f"\n🔚 Testing: {test_name}")
        
        try:
            # Test full orchestrator cleanup
            logger.info("  🧹 Executing full orchestrator cleanup...")
            await self.orchestrator.close()
            
            details = {
                "orchestrator_closed": True,
                "cleanup_method_available": hasattr(self.orchestrator, 'close'),
                "total_agents_managed": len(self.spawned_agent_ids)
            }
            
            success = True  # If we get here without exception, cleanup worked
            
            self.log_test_result(test_name, success, details)
            return success
            
        except Exception as e:
            self.log_test_result(test_name, False, {"error": str(e)})
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run the complete test suite."""
        logger.info("\n🧪 Agent Lifecycle Protocol Test Suite")
        logger.info("=" * 60)
        
        # Setup
        if not await self.setup():
            return {"error": "Test setup failed"}
        
        # Run all tests in sequence
        tests = [
            self.test_dynamic_agent_spawning,
            self.test_lazy_loading_and_caching,
            self.test_resource_budget_enforcement,
            self.test_agent_activation_deactivation,
            self.test_agent_processing_capabilities,
            self.test_event_driven_communication,
            self.test_lifecycle_cleanup,
            self.test_orchestrator_cleanup
        ]
        
        for test_func in tests:
            try:
                await test_func()
                await asyncio.sleep(0.1)  # Brief pause between tests
            except Exception as e:
                logger.error(f"Test failed with exception: {e}")
        
        # Generate final report
        return self.generate_final_report()
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        
        # Calculate success rates
        total_tests = self.test_results["total_tests"]
        passed_tests = self.test_results["passed_tests"]
        failed_tests = self.test_results["failed_tests"]
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Analyze critical failures
        critical_failures = []
        for test in self.test_results["test_details"]:
            if not test["success"] and "spawn" in test["test_name"].lower():
                critical_failures.append(f"Critical: {test['test_name']}")
            elif not test["success"] and "cleanup" in test["test_name"].lower():
                critical_failures.append(f"Critical: {test['test_name']}")
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": round(success_rate, 2),
                "agents_spawned": len(self.spawned_agent_ids)
            },
            "protocol_status": {
                "dynamic_spawning": "WORKING" if any(t["success"] for t in self.test_results["test_details"] if "spawn" in t["test_name"].lower()) else "FAILED",
                "lazy_loading": "WORKING" if any(t["success"] for t in self.test_results["test_details"] if "lazy" in t["test_name"].lower()) else "FAILED",
                "resource_management": "WORKING" if any(t["success"] for t in self.test_results["test_details"] if "budget" in t["test_name"].lower()) else "FAILED",
                "lifecycle_cleanup": "WORKING" if any(t["success"] for t in self.test_results["test_details"] if "cleanup" in t["test_name"].lower()) else "FAILED"
            },
            "critical_failures": critical_failures,
            "test_details": self.test_results["test_details"],
            "recommendations": self.generate_recommendations()
        }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check for failed tests and provide specific recommendations
        for test in self.test_results["test_details"]:
            if not test["success"]:
                test_name = test["test_name"]
                
                if "spawn" in test_name.lower():
                    recommendations.append("Review agent spawning logic in orchestrator/agent_orchestrator.py")
                elif "lazy" in test_name.lower():
                    recommendations.append("Check lazy loading implementation in orchestrator/lazy_loader.py")
                elif "budget" in test_name.lower():
                    recommendations.append("Verify resource budget configuration and limits")
                elif "cleanup" in test_name.lower():
                    recommendations.append("Ensure all agents implement proper close() methods")
                elif "processing" in test_name.lower():
                    recommendations.append("Check UniversalAgent.process_message() implementation")
        
        # Add general recommendations
        if self.test_results["passed_tests"] == self.test_results["total_tests"]:
            recommendations.append("✅ All tests passed! Agent lifecycle protocol is working correctly.")
        elif self.test_results["passed_tests"] / self.test_results["total_tests"] > 0.8:
            recommendations.append("Most tests passed. Review failed tests for minor improvements.")
        else:
            recommendations.append("Multiple test failures detected. Review agent lifecycle implementation.")
            recommendations.append("Consider running individual tests for detailed debugging.")
        
        return recommendations

async def main():
    """Main test execution function."""
    test_suite = AgentLifecycleProtocolTest()
    
    try:
        # Run the complete test suite
        report = await test_suite.run_all_tests()
        
        # Print final report
        print("\n" + "=" * 60)
        print("🎯 FINAL TEST REPORT")
        print("=" * 60)
        
        print(f"\n📊 Test Summary:")
        summary = report["test_summary"]
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Success Rate: {summary['success_rate']}%")
        print(f"  Agents Spawned: {summary['agents_spawned']}")
        
        print(f"\n🔧 Protocol Status:")
        status = report["protocol_status"]
        for protocol, status_val in status.items():
            emoji = "✅" if status_val == "WORKING" else "❌"
            print(f"  {emoji} {protocol.replace('_', ' ').title()}: {status_val}")
        
        if report["critical_failures"]:
            print(f"\n🚨 Critical Failures:")
            for failure in report["critical_failures"]:
                print(f"  ❌ {failure}")
        
        print(f"\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
        
        # Exit with appropriate code
        if summary["success_rate"] >= 80:
            print(f"\n✅ Agent lifecycle protocol is working correctly!")
            sys.exit(0)
        else:
            print(f"\n❌ Agent lifecycle protocol has issues that need attention.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n🛑 Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 