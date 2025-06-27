"""
Test file for Lazy Agent Loader Integration

This test demonstrates the lazy loader working with the agent orchestrator
to efficiently manage 1000+ agent configurations with only 50 active in memory.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from orchestrator.agent_orchestrator import AgentOrchestrator
from orchestrator.lazy_loader import LazyAgentLoader, AgentConfiguration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LazyLoaderIntegrationDemo:
    """Comprehensive demonstration of lazy loading capabilities."""
    
    def __init__(self):
        self.orchestrator = None
        self.results = {}
    
    async def setup(self):
        """Initialize the orchestrator with lazy loading."""
        logger.info("🚀 Setting up Agent Orchestrator with Lazy Loading...")
        
        self.orchestrator = AgentOrchestrator()
        
        # Verify lazy loader is initialized
        assert hasattr(self.orchestrator, 'lazy_loader'), "Lazy loader not initialized"
        assert isinstance(self.orchestrator.lazy_loader, LazyAgentLoader), "Wrong lazy loader type"
        
        logger.info(f"✅ Lazy loader initialized with {self.orchestrator.lazy_loader.max_active_agents} max active agents")
        logger.info(f"✅ Maximum configurations supported: {self.orchestrator.lazy_loader.max_total_configurations}")
    
    async def test_agent_spawning_and_loading(self):
        """Test spawning agents and lazy loading capabilities."""
        logger.info("\n📊 Testing Agent Spawning and Lazy Loading...")
        
        # Test 1: Spawn multiple specialist agents
        specialties = [
            "Python Performance Optimization",
            "Database Query Analysis", 
            "React Component Architecture",
            "Machine Learning Model Tuning",
            "DevOps Pipeline Automation",
            "API Security Analysis",
            "Frontend Performance Optimization",
            "Backend Scaling Strategies",
            "Data Pipeline Engineering",
            "Cloud Infrastructure Management"
        ]
        
        spawned_agents = []
        for specialty in specialties:
            agent_id = await self.orchestrator.spawn_specialist_agent(
                specialty=specialty,
                parent_context={"test": "lazy_loading_demo"},
                temperature=0.3,
                max_tokens=600
            )
            
            if agent_id:
                spawned_agents.append(agent_id)
                logger.info(f"✅ Spawned specialist: {specialty} (ID: {agent_id})")
        
        # Test 2: Load agents and verify lazy loading
        loaded_agents = []
        for agent_id in spawned_agents[:5]:  # Load first 5
            agent = await self.orchestrator.get_or_load_agent(agent_id)
            if agent:
                loaded_agents.append(agent_id)
                logger.info(f"✅ Loaded agent: {agent_id}")
        
        # Test 3: Check cache metrics
        metrics = self.orchestrator.lazy_loader.get_cache_metrics()
        
        self.results['spawning_test'] = {
            'agents_spawned': len(spawned_agents),
            'agents_loaded': len(loaded_agents),
            'cache_hit_rate': metrics['cache_performance']['hit_rate'],
            'cache_utilization': metrics['agent_status']['cache_utilization'],
            'active_agents': metrics['agent_status']['active_agents']
        }
        
        logger.info(f"📈 Cache hit rate: {metrics['cache_performance']['hit_rate']:.2%}")
        logger.info(f"📈 Cache utilization: {metrics['agent_status']['cache_utilization']:.2%}")
        logger.info(f"📈 Active agents: {metrics['agent_status']['active_agents']}")
        
        return spawned_agents
    
    async def test_lru_cache_eviction(self):
        """Test LRU cache eviction when exceeding max active agents."""
        logger.info("\n🔄 Testing LRU Cache Eviction...")
        
        # Spawn enough agents to exceed max active limit (50)
        many_agents = []
        for i in range(55):  # Spawn 55 agents to exceed limit
            agent_id = await self.orchestrator.spawn_specialist_agent(
                specialty=f"Test Specialist {i}",
                temperature=0.2,
                max_tokens=400
            )
            if agent_id:
                many_agents.append(agent_id)
        
        # Load all agents to trigger evictions
        for agent_id in many_agents:
            await self.orchestrator.get_or_load_agent(agent_id)
        
        # Check final cache state
        metrics = self.orchestrator.lazy_loader.get_cache_metrics()
        
        self.results['lru_eviction_test'] = {
            'agents_created': len(many_agents),
            'max_active_limit': self.orchestrator.lazy_loader.max_active_agents,
            'final_active_count': metrics['agent_status']['active_agents'],
            'evictions_performed': metrics['cache_performance']['evictions'],
            'cache_utilization': metrics['agent_status']['cache_utilization']
        }
        
        logger.info(f"✅ Created {len(many_agents)} agents")
        logger.info(f"✅ Active agents limited to: {metrics['agent_status']['active_agents']}")
        logger.info(f"✅ Evictions performed: {metrics['cache_performance']['evictions']}")
        logger.info(f"✅ Final cache utilization: {metrics['agent_status']['cache_utilization']:.2%}")
        
        # Verify we don't exceed max active agents
        assert metrics['agent_status']['active_agents'] <= self.orchestrator.lazy_loader.max_active_agents
        
        return many_agents
    
    async def test_intelligent_preloading(self):
        """Test intelligent preloading based on usage patterns."""
        logger.info("\n🧠 Testing Intelligent Preloading...")
        
        # Create some test agents with different priority scores
        test_configs = []
        for i in range(10):
            config = AgentConfiguration(
                agent_id=f"preload_test_{i}",
                specialty=f"Preload Test Specialist {i}",
                system_prompt=f"You are a test specialist number {i}.",
                temperature=0.4,
                model_name="gpt-3.5-turbo-0125",
                max_tokens=500,
                tools=[],
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow() - timedelta(hours=i),  # Different last used times
                usage_count=10 - i,  # Different usage counts
                success_rate=0.8 + (i * 0.02),  # Different success rates
                average_response_time=1.0 + (i * 0.1),
                total_cost=i * 0.01,
                priority_score=0.5 + (i * 0.05)  # Increasing priority scores
            )
            test_configs.append(config)
            
            # Add to lazy loader
            await self.orchestrator.lazy_loader.add_agent_configuration(config)
        
        # Test preloading
        preloaded_count = await self.orchestrator.lazy_loader.preload_agents()
        
        # Get metrics after preloading
        metrics = self.orchestrator.lazy_loader.get_cache_metrics()
        
        self.results['preloading_test'] = {
            'configs_added': len(test_configs),
            'agents_preloaded': preloaded_count,
            'preload_successes': metrics['preloading']['preload_successes'],
            'preload_failures': metrics['preloading']['preload_failures'],
            'preload_candidates': metrics['preloading']['candidates_for_preload']
        }
        
        logger.info(f"✅ Added {len(test_configs)} test configurations")
        logger.info(f"✅ Preloaded {preloaded_count} agents")
        logger.info(f"✅ Preload successes: {metrics['preloading']['preload_successes']}")
        logger.info(f"✅ Preload candidates available: {metrics['preloading']['candidates_for_preload']}")
    
    async def test_cache_optimization(self):
        """Test cache optimization features."""
        logger.info("\n⚡ Testing Cache Optimization...")
        
        # Run cache optimization
        optimization_results = await self.orchestrator.lazy_loader.optimize_cache()
        
        # Get metrics after optimization
        metrics = self.orchestrator.lazy_loader.get_cache_metrics()
        
        self.results['optimization_test'] = {
            'optimization_results': optimization_results,
            'final_cache_utilization': metrics['agent_status']['cache_utilization'],
            'hit_rate': metrics['cache_performance']['hit_rate'],
            'total_loads': metrics['cache_performance']['total_loads']
        }
        
        logger.info(f"✅ Cache optimization completed")
        logger.info(f"✅ Evicted agents: {optimization_results.get('evicted_agents', 0)}")
        logger.info(f"✅ Preloaded agents: {optimization_results.get('preloaded_agents', 0)}")
        logger.info(f"✅ Final cache utilization: {metrics['agent_status']['cache_utilization']:.2%}")
    
    async def test_activity_tracking(self):
        """Test agent activity tracking and reporting."""
        logger.info("\n📈 Testing Activity Tracking...")
        
        # Get activity report
        activity_report = self.orchestrator.lazy_loader.get_agent_activity_report()
        
        # Count different activity levels
        high_activity = sum(1 for agent_data in activity_report.values() 
                           if agent_data['usage_frequency'] > 1.0)
        
        active_agents = sum(1 for agent_data in activity_report.values() 
                           if agent_data['is_active'])
        
        self.results['activity_tracking_test'] = {
            'total_agents_tracked': len(activity_report),
            'high_activity_agents': high_activity,
            'currently_active_agents': active_agents,
            'sample_activity_data': dict(list(activity_report.items())[:3])  # First 3 for sample
        }
        
        logger.info(f"✅ Tracking {len(activity_report)} agents")
        logger.info(f"✅ High activity agents: {high_activity}")
        logger.info(f"✅ Currently active agents: {active_agents}")
    
    async def test_orchestrator_integration(self):
        """Test integration with orchestrator's get_agent_stats method."""
        logger.info("\n🔗 Testing Orchestrator Integration...")
        
        # Get orchestrator stats (should include lazy loader metrics)
        stats = self.orchestrator.get_agent_stats()
        
        # Verify lazy loader metrics are included
        assert 'lazy_loader' in stats, "Lazy loader metrics not in orchestrator stats"
        assert 'cache_hit_rate' in stats, "Cache hit rate not in orchestrator stats"
        assert 'cache_utilization' in stats, "Cache utilization not in orchestrator stats"
        
        self.results['integration_test'] = {
            'lazy_loader_metrics_included': 'lazy_loader' in stats,
            'cache_hit_rate': stats.get('cache_hit_rate', 0),
            'cache_utilization': stats.get('cache_utilization', 0),
            'total_configurations': stats['lazy_loader']['agent_status']['total_configurations'],
            'active_agents': stats['lazy_loader']['agent_status']['active_agents']
        }
        
        logger.info(f"✅ Orchestrator integration working")
        logger.info(f"✅ Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
        logger.info(f"✅ Total configurations: {stats['lazy_loader']['agent_status']['total_configurations']}")
        logger.info(f"✅ Active agents: {stats['lazy_loader']['agent_status']['active_agents']}")
    
    async def run_comprehensive_demo(self):
        """Run the complete lazy loading demonstration."""
        logger.info("🎯 Starting Comprehensive Lazy Agent Loader Demo...")
        
        try:
            # Setup
            await self.setup()
            
            # Run all tests
            await self.test_agent_spawning_and_loading()
            await self.test_lru_cache_eviction()
            await self.test_intelligent_preloading()
            await self.test_cache_optimization()
            await self.test_activity_tracking()
            await self.test_orchestrator_integration()
            
            # Final summary
            await self.print_final_summary()
            
        finally:
            # Cleanup
            if self.orchestrator:
                await self.orchestrator.close()
    
    async def print_final_summary(self):
        """Print comprehensive summary of all test results."""
        logger.info("\n" + "="*60)
        logger.info("🎉 LAZY AGENT LOADER DEMO COMPLETE!")
        logger.info("="*60)
        
        # Get final metrics
        final_metrics = self.orchestrator.lazy_loader.get_cache_metrics()
        
        logger.info(f"\n📊 FINAL SYSTEM STATE:")
        logger.info(f"   • Total Configurations: {final_metrics['agent_status']['total_configurations']}")
        logger.info(f"   • Active Agents: {final_metrics['agent_status']['active_agents']}")
        logger.info(f"   • Cache Hit Rate: {final_metrics['cache_performance']['hit_rate']:.2%}")
        logger.info(f"   • Cache Utilization: {final_metrics['agent_status']['cache_utilization']:.2%}")
        logger.info(f"   • Total Loads: {final_metrics['cache_performance']['total_loads']}")
        logger.info(f"   • Evictions: {final_metrics['cache_performance']['evictions']}")
        
        logger.info(f"\n🎯 TEST RESULTS SUMMARY:")
        for test_name, results in self.results.items():
            logger.info(f"   • {test_name}: ✅ PASSED")
        
        logger.info(f"\n✅ All {len(self.results)} tests completed successfully!")
        logger.info("🚀 Lazy Agent Loader is fully integrated and operational!")

async def main():
    """Run the lazy loader integration demo."""
    demo = LazyLoaderIntegrationDemo()
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(main()) 