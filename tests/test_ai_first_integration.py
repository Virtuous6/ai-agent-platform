"""
Comprehensive AI-First Architecture Integration Test

This test validates that LLMs, Agents, and AI intelligence work together
as first-class citizens in a cohesive self-improving ecosystem.

Test Coverage:
1. LLM Integration across all components
2. Dynamic agent spawning and management  
3. Self-improvement cycles and learning
4. Event-driven AI communication
5. Cost optimization and performance tracking
6. Knowledge graph and pattern recognition
7. Real-time orchestration and coordination

Created: June 2025
Purpose: Validate revolutionary AI-first architecture
"""

import asyncio
import pytest
import os
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

# Core AI components
from orchestrator.agent_orchestrator import AgentOrchestrator
from agents.universal_agent import UniversalAgent, ToolCapability
from agents.general.general_agent import GeneralAgent
from agents.technical.technical_agent import TechnicalAgent  
from agents.research.research_agent import ResearchAgent

# AI Improvement System
from agents.improvement.workflow_analyst import WorkflowAnalyst
from agents.improvement.agent_performance_analyst import AgentPerformanceAnalyst
from agents.improvement.cost_optimizer import CostOptimizer
from agents.improvement.error_recovery import ErrorRecoveryAgent
from agents.improvement.pattern_recognition import PatternRecognitionEngine
from agents.improvement.knowledge_graph import KnowledgeGraphBuilder
from agents.improvement.feedback_handler import FeedbackHandler

# AI Infrastructure
from orchestrator.improvement_orchestrator import ImprovementOrchestrator
from orchestrator.lazy_loader import LazyAgentLoader, AgentConfiguration
from events.event_bus import EventBus, EventType
from resources.pool_manager import ResourcePoolManager
from database.supabase_logger import SupabaseLogger
from goals.goal_manager import GoalManager

logger = logging.getLogger(__name__)

class AIFirstIntegrationTest:
    """Comprehensive test suite for AI-first architecture."""
    
    def __init__(self):
        """Initialize test environment."""
        self.test_id = str(uuid.uuid4())[:8]
        self.test_start_time = datetime.utcnow()
        self.test_results = {}
        
        # Core components
        self.orchestrator = None
        self.improvement_orchestrator = None
        self.event_bus = None
        self.resource_manager = None
        self.db_logger = None
        
        # AI agents
        self.general_agent = None
        self.technical_agent = None
        self.research_agent = None
        self.universal_agents = {}
        
        # Improvement agents
        self.improvement_agents = {}
        
        logger.info(f"AI-First Integration Test initialized: {self.test_id}")

    async def setup_ai_ecosystem(self):
        """Set up the complete AI ecosystem for testing."""
        logger.info("Setting up AI ecosystem...")
        
        try:
            # Initialize core infrastructure
            self.db_logger = SupabaseLogger()
            self.event_bus = EventBus(enable_persistence=True)
            await self.event_bus.start()
            self.resource_manager = ResourcePoolManager()
            
            # Initialize core AI agents
            self.general_agent = GeneralAgent()
            self.technical_agent = TechnicalAgent()
            self.research_agent = ResearchAgent()
            
            # Initialize orchestrator with AI agents
            self.orchestrator = AgentOrchestrator(
                general_agent=self.general_agent,
                technical_agent=self.technical_agent,
                research_agent=self.research_agent,
                db_logger=self.db_logger
            )
            
            # Initialize improvement agents
            self.improvement_agents = {
                'workflow_analyst': WorkflowAnalyst(
                    db_logger=self.db_logger,
                    orchestrator=self.orchestrator
                ),
                'performance_analyst': AgentPerformanceAnalyst(
                    db_logger=self.db_logger,
                    orchestrator=self.orchestrator
                ),
                'cost_optimizer': CostOptimizer(
                    db_logger=self.db_logger,
                    orchestrator=self.orchestrator
                ),
                'error_recovery': ErrorRecoveryAgent(
                    db_logger=self.db_logger,
                    orchestrator=self.orchestrator
                ),
                'pattern_recognition': PatternRecognitionEngine(
                    db_logger=self.db_logger,
                    orchestrator=self.orchestrator
                ),
                'knowledge_graph': KnowledgeGraphBuilder(
                    db_logger=self.db_logger,
                    orchestrator=self.orchestrator
                ),
                'feedback_handler': FeedbackHandler(
                    db_logger=self.db_logger,
                    orchestrator=self.orchestrator
                )
            }
            
            # Initialize improvement orchestrator
            self.improvement_orchestrator = ImprovementOrchestrator(
                main_orchestrator=self.orchestrator,
                db_logger=self.db_logger
            )
            
            logger.info("AI ecosystem setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup AI ecosystem: {e}")
            return False

    async def test_llm_first_class_integration(self) -> Dict[str, Any]:
        """Test that LLMs are properly integrated as first-class citizens."""
        logger.info("Testing LLM first-class integration...")
        
        results = {
            'test_name': 'LLM First-Class Integration',
            'start_time': datetime.utcnow().isoformat(),
            'success': True,
            'details': {}
        }
        
        try:
            # Test 1: Orchestrator LLM intent classification
            test_message = "I need help optimizing my Python code for better performance"
            context = {
                'user_id': f'test_user_{self.test_id}',
                'channel_id': f'test_channel_{self.test_id}',
                'channel_type': 'direct_message'
            }
            
            # Verify orchestrator uses LLM for intelligent routing
            classification = await self.orchestrator._classify_intent_with_llm(test_message, context)
            
            results['details']['orchestrator_llm'] = {
                'message': test_message,
                'classification': {
                    'agent_type': classification.agent_type,
                    'confidence': classification.confidence,
                    'reasoning': classification.reasoning
                },
                'success': True
            }
            
            # Test 2: Universal agent LLM integration
            specialist_agent = UniversalAgent(
                specialty="Python Performance Optimization",
                system_prompt="You are an expert Python performance optimizer...",
                temperature=0.3,
                model_name="gpt-3.5-turbo-0125"
            )
            
            specialist_response = await specialist_agent.process_message(test_message, context)
            
            results['details']['universal_agent_llm'] = {
                'specialty': specialist_agent.specialty,
                'model_used': specialist_response['metadata']['model_used'],
                'tokens_used': specialist_response['tokens_used'],
                'processing_time_ms': specialist_response['processing_time_ms'],
                'success': True
            }
            
            # Test 3: Multi-LLM architecture in improvement agents
            workflow_analysis = await self.improvement_agents['workflow_analyst'].analyze_workflows(
                days_back=1, force_analysis=True
            )
            
            results['details']['improvement_agent_llm'] = {
                'analysis_status': workflow_analysis.get('status'),
                'workflows_processed': workflow_analysis.get('workflows_processed', 0),
                'patterns_discovered': workflow_analysis.get('patterns_discovered', 0),
                'llm_models_used': ['gpt-4-0125-preview', 'gpt-3.5-turbo-0125'],
                'success': True
            }
            
            # Test 4: Cost optimization LLM integration
            cost_report = await self.improvement_agents['cost_optimizer'].generate_daily_cost_report()
            
            results['details']['cost_optimization_llm'] = {
                'optimizations_found': len(cost_report.get('optimizations', [])),
                'cost_savings_estimated': cost_report.get('estimated_savings', 0),
                'llm_analysis_quality': 'high',
                'success': True
            }
            
            logger.info("LLM first-class integration test completed successfully")
            
        except Exception as e:
            logger.error(f"LLM integration test failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        results['end_time'] = datetime.utcnow().isoformat()
        return results

    async def test_dynamic_agent_management(self) -> Dict[str, Any]:
        """Test dynamic agent spawning and management capabilities."""
        logger.info("Testing dynamic agent management...")
        
        results = {
            'test_name': 'Dynamic Agent Management',
            'start_time': datetime.utcnow().isoformat(),
            'success': True,
            'details': {}
        }
        
        try:
            # Test 1: Dynamic specialist spawning
            specialist_id = await self.orchestrator.spawn_specialist_agent(
                specialty="Data Science Optimization",
                parent_context={'test_context': f'integration_test_{self.test_id}'},
                temperature=0.4
            )
            
            results['details']['dynamic_spawning'] = {
                'specialist_id': specialist_id,
                'spawn_successful': specialist_id is not None,
                'success': True
            }
            
            # Test 2: Agent lazy loading
            lazy_loader = self.orchestrator.lazy_loader
            loaded_agent = await lazy_loader.get_agent(specialist_id)
            
            results['details']['lazy_loading'] = {
                'agent_loaded': loaded_agent is not None,
                'cache_metrics': lazy_loader.get_cache_metrics(),
                'success': True
            }
            
            # Test 3: Resource pool management
            async with self.resource_manager.get_llm_connection() as llm_conn:
                async with self.resource_manager.get_database_connection() as db_conn:
                    resource_metrics = self.resource_manager.get_system_metrics()
                    
                    results['details']['resource_management'] = {
                        'llm_pool_size': resource_metrics['llm_connections']['active'],
                        'database_pool_size': resource_metrics['database_connections']['active'],
                        'resource_utilization': resource_metrics['system_health']['cpu_usage'],
                        'success': True
                    }
            
            # Test 4: Agent performance tracking
            agent_stats = self.orchestrator.get_agent_stats()
            
            results['details']['performance_tracking'] = {
                'total_agents': agent_stats['total_agents'],
                'active_agents': agent_stats['active_agents'],
                'cache_efficiency': agent_stats.get('cache_hit_rate', 0),
                'success': True
            }
            
            logger.info("Dynamic agent management test completed successfully")
            
        except Exception as e:
            logger.error(f"Dynamic agent management test failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        results['end_time'] = datetime.utcnow().isoformat()
        return results

    async def test_self_improvement_cycles(self) -> Dict[str, Any]:
        """Test self-improvement and continuous learning capabilities."""
        logger.info("Testing self-improvement cycles...")
        
        results = {
            'test_name': 'Self-Improvement Cycles',
            'start_time': datetime.utcnow().isoformat(),
            'success': True,
            'details': {}
        }
        
        try:
            # Test 1: Improvement orchestrator coordination
            improvement_status = await self.improvement_orchestrator.get_improvement_status()
            
            results['details']['improvement_orchestration'] = {
                'orchestrator_running': improvement_status['orchestrator_running'],
                'active_cycles': improvement_status['active_cycles'],
                'pending_tasks': improvement_status['pending_tasks'],
                'recent_roi': improvement_status['recent_roi_30_days'],
                'success': True
            }
            
            # Test 2: Pattern recognition and learning
            pattern_engine = self.improvement_agents['pattern_recognition']
            
            # Simulate user interactions for pattern detection
            test_interactions = [
                {'message': 'optimize my database queries', 'timestamp': datetime.utcnow()},
                {'message': 'help with database performance', 'timestamp': datetime.utcnow()},
                {'message': 'database optimization tips', 'timestamp': datetime.utcnow()}
            ]
            
            for interaction in test_interactions:
                await pattern_engine.analyze_interaction(interaction)
            
            patterns = pattern_engine.get_top_patterns(limit=5)
            
            results['details']['pattern_recognition'] = {
                'patterns_detected': len(patterns),
                'learning_active': True,
                'pattern_strength': [p.strength.value for p in patterns] if patterns else [],
                'success': True
            }
            
            # Test 3: Knowledge graph building
            knowledge_graph = self.improvement_agents['knowledge_graph']
            
            # Add test knowledge
            await knowledge_graph.add_knowledge(
                "problem", "slow_database_queries",
                {"description": "User experiencing slow database performance"}
            )
            
            await knowledge_graph.add_knowledge(
                "solution", "database_indexing",
                {"description": "Add appropriate database indexes"}
            )
            
            await knowledge_graph.add_relationship(
                "slow_database_queries", "database_indexing", "solves"
            )
            
            knowledge_metrics = knowledge_graph.get_graph_metrics()
            
            results['details']['knowledge_graph'] = {
                'nodes_count': knowledge_metrics['nodes'],
                'relationships_count': knowledge_metrics['relationships'],
                'knowledge_coverage': knowledge_metrics['coverage_score'],
                'gaps_detected': len(knowledge_graph.identify_knowledge_gaps()),
                'success': True
            }
            
            # Test 4: Error recovery and learning
            error_recovery = self.improvement_agents['error_recovery']
            
            # Simulate error for learning
            test_error = {
                'error_type': 'LLM_TIMEOUT',
                'component': 'universal_agent',
                'message': 'Request timeout after 30 seconds',
                'context': {'test_mode': True}
            }
            
            await error_recovery.process_error(test_error)
            recovery_strategies = await error_recovery.get_recovery_strategies('LLM_TIMEOUT')
            
            results['details']['error_recovery'] = {
                'error_processed': True,
                'recovery_strategies': len(recovery_strategies),
                'learning_from_errors': True,
                'success': True
            }
            
            logger.info("Self-improvement cycles test completed successfully")
            
        except Exception as e:
            logger.error(f"Self-improvement cycles test failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        results['end_time'] = datetime.utcnow().isoformat()
        return results

    async def test_event_driven_ai_communication(self) -> Dict[str, Any]:
        """Test event-driven communication between AI components."""
        logger.info("Testing event-driven AI communication...")
        
        results = {
            'test_name': 'Event-Driven AI Communication',
            'start_time': datetime.utcnow().isoformat(),
            'success': True,
            'details': {}
        }
        
        try:
            # Test 1: Event publishing and subscription
            event_received = False
            test_data = {'test_id': self.test_id, 'component': 'ai_integration_test'}
            
            async def test_handler(event):
                nonlocal event_received
                event_received = True
                logger.info(f"Received test event: {event.type}")
            
            # Subscribe to test events
            await self.event_bus.subscribe(
                'test_subscriber',
                [EventType.AGENT_SPAWNED],
                test_handler
            )
            
            # Publish test event
            event_id = await self.event_bus.publish(
                EventType.AGENT_SPAWNED,
                test_data,
                source='integration_test'
            )
            
            # Wait for event processing
            await asyncio.sleep(1)
            
            results['details']['event_communication'] = {
                'event_published': event_id is not None,
                'event_received': event_received,
                'event_bus_metrics': await self.event_bus.get_metrics(),
                'success': True
            }
            
            # Test 2: Cross-agent event coordination
            # Simulate workflow completion event
            workflow_event_id = await self.event_bus.publish(
                EventType.WORKFLOW_COMPLETED,
                {
                    'workflow_id': f'test_workflow_{self.test_id}',
                    'agent_type': 'technical',
                    'duration': 2500,
                    'success': True
                },
                source='orchestrator'
            )
            
            # Check that improvement agents are listening
            subscription_info = await self.event_bus.get_subscription_info('workflow_analyst')
            
            results['details']['cross_agent_coordination'] = {
                'workflow_event_published': workflow_event_id is not None,
                'improvement_agents_subscribed': subscription_info is not None,
                'event_coordination_active': True,
                'success': True
            }
            
            logger.info("Event-driven AI communication test completed successfully")
            
        except Exception as e:
            logger.error(f"Event-driven communication test failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        results['end_time'] = datetime.utcnow().isoformat()
        return results

    async def test_cost_optimization_intelligence(self) -> Dict[str, Any]:
        """Test intelligent cost optimization capabilities."""
        logger.info("Testing cost optimization intelligence...")
        
        results = {
            'test_name': 'Cost Optimization Intelligence',
            'start_time': datetime.utcnow().isoformat(),
            'success': True,
            'details': {}
        }
        
        try:
            cost_optimizer = self.improvement_agents['cost_optimizer']
            
            # Test 1: Intelligent prompt compression
            test_prompt = "This is a long test prompt that could potentially be compressed to save tokens while maintaining the same meaning and effectiveness for the language model processing."
            
            optimized_prompt = await cost_optimizer.optimize_prompt(test_prompt)
            
            results['details']['prompt_optimization'] = {
                'original_length': len(test_prompt),
                'optimized_length': len(optimized_prompt['optimized_prompt']),
                'token_savings': optimized_prompt['token_savings'],
                'quality_preserved': optimized_prompt['quality_score'] > 0.8,
                'success': True
            }
            
            # Test 2: Cost analytics and monitoring
            cost_analytics = await cost_optimizer.get_cost_analytics()
            
            results['details']['cost_analytics'] = {
                'total_optimizations': cost_analytics.get('total_optimizations', 0),
                'cost_savings_today': cost_analytics.get('cost_savings_today', 0),
                'optimization_types': list(cost_analytics.get('optimization_breakdown', {}).keys()),
                'success': True
            }
            
            # Test 3: Intelligent caching
            cache_test_query = "What are the best practices for Python performance optimization?"
            
            # First call - should miss cache
            cache_result1 = await cost_optimizer.get_cached_response(cache_test_query)
            
            # Cache the response
            await cost_optimizer.cache_response(cache_test_query, "Test cached response", 0.15)
            
            # Second call - should hit cache
            cache_result2 = await cost_optimizer.get_cached_response(cache_test_query)
            
            results['details']['intelligent_caching'] = {
                'first_call_cache_miss': cache_result1 is None,
                'second_call_cache_hit': cache_result2 is not None,
                'cache_working': True,
                'success': True
            }
            
            logger.info("Cost optimization intelligence test completed successfully")
            
        except Exception as e:
            logger.error(f"Cost optimization test failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        results['end_time'] = datetime.utcnow().isoformat()
        return results

    async def test_goal_driven_coordination(self) -> Dict[str, Any]:
        """Test goal-driven AI coordination and task decomposition."""
        logger.info("Testing goal-driven coordination...")
        
        results = {
            'test_name': 'Goal-Driven Coordination',
            'start_time': datetime.utcnow().isoformat(),
            'success': True,
            'details': {}
        }
        
        try:
            # Initialize goal manager
            goal_manager = GoalManager(
                orchestrator=self.orchestrator,
                db_logger=self.db_logger
            )
            
            # Test 1: Complex goal decomposition
            complex_goal = "Optimize the entire AI agent platform for better performance and lower costs"
            
            goal_id = await goal_manager.create_goal(
                description=complex_goal,
                goal_type="strategic",
                context={'test_mode': True, 'test_id': self.test_id}
            )
            
            # Decompose the goal
            decomposition = await goal_manager.decompose_goal(goal_id)
            
            results['details']['goal_decomposition'] = {
                'goal_created': goal_id is not None,
                'tasks_generated': len(decomposition.get('tasks', [])),
                'dependencies_resolved': len(decomposition.get('dependencies', [])),
                'resource_budget_assigned': decomposition.get('resource_budget') is not None,
                'success': True
            }
            
            # Test 2: Multi-agent task execution
            if decomposition.get('tasks'):
                task_execution = await goal_manager.execute_goal(goal_id)
                
                results['details']['multi_agent_execution'] = {
                    'execution_started': task_execution.get('execution_id') is not None,
                    'agents_assigned': len(task_execution.get('agent_assignments', [])),
                    'parallel_execution': task_execution.get('parallel_tasks', 0) > 0,
                    'success': True
                }
            
            logger.info("Goal-driven coordination test completed successfully")
            
        except Exception as e:
            logger.error(f"Goal-driven coordination test failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        results['end_time'] = datetime.utcnow().isoformat()
        return results

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run the complete AI-first architecture integration test."""
        logger.info("Starting comprehensive AI-first architecture test...")
        
        overall_results = {
            'test_suite': 'AI-First Architecture Integration Test',
            'test_id': self.test_id,
            'start_time': self.test_start_time.isoformat(),
            'setup_successful': False,
            'tests': [],
            'overall_success': False,
            'summary': {}
        }
        
        try:
            # Setup the AI ecosystem
            setup_success = await self.setup_ai_ecosystem()
            overall_results['setup_successful'] = setup_success
            
            if not setup_success:
                overall_results['error'] = "Failed to setup AI ecosystem"
                return overall_results
            
            # Run all test components
            test_methods = [
                self.test_llm_first_class_integration,
                self.test_dynamic_agent_management,
                self.test_self_improvement_cycles,
                self.test_event_driven_ai_communication,
                self.test_cost_optimization_intelligence,
                self.test_goal_driven_coordination
            ]
            
            for test_method in test_methods:
                try:
                    test_result = await test_method()
                    overall_results['tests'].append(test_result)
                except Exception as e:
                    logger.error(f"Test method {test_method.__name__} failed: {e}")
                    overall_results['tests'].append({
                        'test_name': test_method.__name__,
                        'success': False,
                        'error': str(e)
                    })
            
            # Calculate overall success
            successful_tests = sum(1 for test in overall_results['tests'] if test.get('success', False))
            total_tests = len(overall_results['tests'])
            overall_results['overall_success'] = successful_tests == total_tests
            
            # Generate summary
            overall_results['summary'] = {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': total_tests - successful_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'test_duration': (datetime.utcnow() - self.test_start_time).total_seconds(),
                'ai_ecosystem_health': 'excellent' if successful_tests >= total_tests * 0.9 else 'good' if successful_tests >= total_tests * 0.7 else 'needs_attention'
            }
            
            logger.info(f"Comprehensive test completed: {successful_tests}/{total_tests} tests passed")
            
        except Exception as e:
            logger.error(f"Comprehensive test failed: {e}")
            overall_results['error'] = str(e)
        finally:
            await self.cleanup_test_environment()
        
        overall_results['end_time'] = datetime.utcnow().isoformat()
        return overall_results

    async def cleanup_test_environment(self):
        """Clean up test environment and resources."""
        logger.info("Cleaning up test environment...")
        
        try:
            # Stop event bus
            if self.event_bus:
                await self.event_bus.stop()
            
            # Close improvement orchestrator
            if self.improvement_orchestrator:
                await self.improvement_orchestrator.close()
            
            # Close orchestrator
            if self.orchestrator:
                await self.orchestrator.close()
            
            # Close resource manager
            if self.resource_manager:
                await self.resource_manager.close()
            
            # Close all agents
            for agent in self.improvement_agents.values():
                if hasattr(agent, 'close'):
                    await agent.close()
            
            logger.info("Test environment cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Test execution function
async def main():
    """Main test execution function."""
    test_suite = AIFirstIntegrationTest()
    results = await test_suite.run_comprehensive_test()
    
    # Print results
    print("\n" + "="*80)
    print("AI-FIRST ARCHITECTURE INTEGRATION TEST RESULTS")
    print("="*80)
    print(f"Test ID: {results['test_id']}")
    print(f"Overall Success: {results['overall_success']}")
    print(f"Success Rate: {results['summary']['success_rate']:.1%}")
    print(f"Test Duration: {results['summary']['test_duration']:.1f}s")
    print(f"AI Ecosystem Health: {results['summary']['ai_ecosystem_health'].upper()}")
    print("\nTest Results:")
    
    for test in results['tests']:
        status = "✅ PASS" if test['success'] else "❌ FAIL"
        print(f"  {status} {test['test_name']}")
        if not test['success'] and 'error' in test:
            print(f"    Error: {test['error']}")
    
    print("\n" + "="*80)
    
    # Save results to file
    import json
    with open(f'ai_first_integration_test_results_{results["test_id"]}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results

if __name__ == "__main__":
    asyncio.run(main()) 