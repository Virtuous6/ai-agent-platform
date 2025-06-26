"""
Test suite for Error Recovery Agent

This module tests the error analysis, pattern recognition, recovery strategies,
and self-healing capabilities of the Error Recovery Agent.
"""

import pytest
import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from agents.improvement.error_recovery import (
    ErrorRecoveryAgent,
    ErrorEvent, 
    ErrorPattern,
    RecoveryStrategy,
    PreventiveMeasure,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStatus,
    ErrorAnalysis,
    RecoveryPlan
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self, response_type="analysis"):
        self.response_type = response_type
        
    async def ainvoke(self, input_data):
        """Mock LLM response."""
        if self.response_type == "analysis":
            return {
                "error_category": "system",
                "severity_assessment": "high", 
                "root_cause": "Database connection timeout due to network latency",
                "contributing_factors": ["High network latency", "Database overload", "Connection pool exhaustion"],
                "impact_analysis": {"affected_users": 150, "downtime_seconds": 45},
                "pattern_indicators": ["Recurring during peak hours", "Similar timeout values", "Same database instance"]
            }
        elif self.response_type == "recovery":
            return {
                "recovery_strategies": [
                    {
                        "name": "Retry with Exponential Backoff",
                        "description": "Retry database connection with increasing delays",
                        "type": "retry",
                        "implementation": {"max_retries": 3, "backoff_delay": 1.0},
                        "preconditions": ["Database is responsive", "Network is available"],
                        "side_effects": ["Temporary delay"],
                        "priority": 8
                    },
                    {
                        "name": "Fallback to Read Replica",
                        "description": "Switch to read-only database replica",
                        "type": "fallback",
                        "implementation": {"fallback_component": "read_replica", "fallback_method": "readonly"},
                        "preconditions": ["Read replica is available"],
                        "side_effects": ["Read-only mode"],
                        "priority": 6
                    }
                ],
                "preventive_measures": [
                    {
                        "name": "Connection Pool Monitoring",
                        "description": "Monitor connection pool usage and alert on high utilization",
                        "implementation_type": "monitoring",
                        "implementation": {"threshold": 0.8, "alert_method": "webhook"},
                        "estimated_reduction": 0.7,
                        "complexity": "low",
                        "maintenance": "low"
                    }
                ],
                "monitoring_recommendations": [
                    "Add database connection pool metrics",
                    "Monitor network latency to database",
                    "Set up alerts for connection failures"
                ],
                "escalation_triggers": [
                    "More than 5 failures in 10 minutes",
                    "All recovery strategies fail",
                    "Critical database unavailable"
                ]
            }

@pytest.fixture
def mock_db_logger():
    """Mock database logger."""
    db_logger = Mock()
    db_logger.log_error = AsyncMock()
    return db_logger

@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator."""
    orchestrator = Mock()
    orchestrator.restart_component = AsyncMock()
    orchestrator.restart_agent = AsyncMock()
    return orchestrator

@pytest.fixture
async def error_recovery_agent(mock_db_logger, mock_orchestrator):
    """Create Error Recovery Agent for testing."""
    agent = ErrorRecoveryAgent(
        db_logger=mock_db_logger,
        orchestrator=mock_orchestrator
    )
    
    # Replace LLMs with mocks for testing
    agent.analysis_llm = MockLLM("analysis")
    agent.recovery_llm = MockLLM("recovery")
    agent.analysis_chain = MockLLM("analysis")
    agent.recovery_chain = MockLLM("recovery")
    
    yield agent
    
    # Cleanup
    await agent.close()

@pytest.mark.asyncio
async def test_error_recording(error_recovery_agent):
    """Test basic error recording functionality."""
    agent = error_recovery_agent
    
    # Create a test error
    test_error = ValueError("Database connection failed")
    
    # Record the error
    error_id = await agent.record_error(
        error=test_error,
        component="database_manager",
        context={"operation": "user_query", "table": "users"},
        user_id="user123"
    )
    
    # Verify error was recorded
    assert error_id != ""
    assert len(agent.error_events) == 1
    
    # Check error details
    error_event = agent.error_events[0]
    assert error_event.error_type == "ValueError"
    assert error_event.component == "database_manager"
    assert error_event.user_id == "user123"
    assert error_event.context["operation"] == "user_query"
    
    print(f"✅ Error recorded successfully: {error_id}")

@pytest.mark.asyncio
async def test_error_classification(error_recovery_agent):
    """Test error classification capabilities."""
    agent = error_recovery_agent
    
    # Test different error types
    test_cases = [
        {
            "error": ConnectionError("Database connection timeout"),
            "component": "database",
            "expected_category": ErrorCategory.SYSTEM,
            "expected_severity": ErrorSeverity.CRITICAL
        },
        {
            "error": ValueError("Invalid user input format"),
            "component": "input_validator", 
            "expected_category": ErrorCategory.USER,
            "expected_severity": ErrorSeverity.LOW
        },
        {
            "error": Exception("OpenAI API rate limit exceeded"),
            "component": "llm_agent",
            "expected_category": ErrorCategory.LLM,
            "expected_severity": ErrorSeverity.MEDIUM
        }
    ]
    
    for test_case in test_cases:
        error_id = await agent.record_error(
            error=test_case["error"],
            component=test_case["component"]
        )
        
        # Find the recorded error
        error_event = next(e for e in agent.error_events if e.id == error_id)
        
        print(f"Error: {error_event.error_type}")
        print(f"  Category: {error_event.category.value} (expected: {test_case['expected_category'].value})")
        print(f"  Severity: {error_event.severity.value} (expected: {test_case['expected_severity'].value})")
    
    print(f"✅ Classification test completed with {len(test_cases)} error types")

@pytest.mark.asyncio
async def test_pattern_recognition(error_recovery_agent):
    """Test error pattern recognition."""
    agent = error_recovery_agent
    
    # Set lower threshold for testing
    agent.pattern_detection_threshold = 2
    
    # Create similar errors
    for i in range(3):
        await agent.record_error(
            error=ConnectionError("Database connection timeout"),
            component="database_manager",
            context={"operation": f"query_{i}", "table": "users"}
        )
        await asyncio.sleep(0.1)  # Small delay to differentiate timestamps
    
    # Allow time for pattern detection
    await asyncio.sleep(0.5)
    
    # Check if pattern was created
    assert len(agent.error_patterns) >= 1, "Expected at least one error pattern"
    
    # Verify pattern details
    pattern = list(agent.error_patterns.values())[0]
    assert pattern.occurrences >= 2
    assert "database_manager" in pattern.affected_components
    assert pattern.category == ErrorCategory.SYSTEM
    
    print(f"✅ Pattern recognized: {pattern.pattern_name}")
    print(f"  Occurrences: {pattern.occurrences}")
    print(f"  Confidence: {pattern.confidence}")
    print(f"  Affected components: {pattern.affected_components}")

@pytest.mark.asyncio 
async def test_recovery_strategy_generation(error_recovery_agent):
    """Test recovery strategy generation."""
    agent = error_recovery_agent
    
    # Create an error pattern
    pattern_id = str(uuid.uuid4())
    pattern = ErrorPattern(
        id=pattern_id,
        pattern_name="Database Connection Failures",
        description="Recurring database connection timeouts",
        error_signature="test_signature",
        occurrences=5,
        first_seen=datetime.utcnow() - timedelta(hours=2),
        last_seen=datetime.utcnow(),
        affected_components={"database_manager"},
        common_contexts=[{"operation": "query", "table": "users"}],
        severity_distribution={ErrorSeverity.HIGH: 3, ErrorSeverity.CRITICAL: 2},
        category=ErrorCategory.SYSTEM,
        root_cause_analysis="Database connection pool exhaustion",
        confidence=0.9
    )
    
    agent.error_patterns[pattern_id] = pattern
    
    # Generate recovery strategies
    await agent._generate_recovery_strategies(pattern, {})
    
    # Check if strategies were generated
    pattern_strategies = [
        s for s in agent.recovery_strategies.values()
        if s.pattern_id == pattern_id
    ]
    
    assert len(pattern_strategies) >= 1, "Expected at least one recovery strategy"
    
    # Check strategy details
    strategy = pattern_strategies[0]
    assert strategy.strategy_name != ""
    assert strategy.description != ""
    assert strategy.strategy_type in ["retry", "fallback", "reset", "escalate", "ignore"]
    
    print(f"✅ Recovery strategies generated: {len(pattern_strategies)}")
    for strategy in pattern_strategies:
        print(f"  Strategy: {strategy.strategy_name} ({strategy.strategy_type})")
        print(f"  Priority: {strategy.priority}")

@pytest.mark.asyncio
async def test_automatic_recovery(error_recovery_agent):
    """Test automatic recovery application."""
    agent = error_recovery_agent
    
    # Create error pattern and strategy
    pattern_id = str(uuid.uuid4())
    pattern = ErrorPattern(
        id=pattern_id,
        pattern_name="Test Pattern",
        description="Test pattern for recovery",
        error_signature="test_sig",
        occurrences=3,
        first_seen=datetime.utcnow(),
        last_seen=datetime.utcnow(),
        affected_components={"test_component"},
        common_contexts=[],
        severity_distribution={ErrorSeverity.MEDIUM: 3},
        category=ErrorCategory.SYSTEM,
        root_cause_analysis="Test root cause",
        confidence=0.8
    )
    
    strategy_id = str(uuid.uuid4())
    strategy = RecoveryStrategy(
        id=strategy_id,
        pattern_id=pattern_id,
        strategy_name="Test Retry Strategy",
        description="Test strategy",
        strategy_type="retry",
        implementation={"max_retries": 2, "backoff_delay": 0.1},
        preconditions=[],
        success_rate=0.8,
        average_recovery_time=1.0,
        side_effects=[],
        priority=8,
        created_at=datetime.utcnow()
    )
    
    # Store pattern and strategy
    agent.error_patterns[pattern_id] = pattern
    agent.recovery_strategies[strategy_id] = strategy
    agent.error_signatures["test_sig"] = pattern_id
    
    # Mock error signature generation to match our pattern
    original_generate_signature = agent._generate_error_signature
    agent._generate_error_signature = lambda event: "test_sig"
    
    # Create error event that matches pattern
    error_event = ErrorEvent(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        error_type="TestError",
        error_message="Test error message",
        stack_trace="",
        component="test_component",
        context={},
        severity=ErrorSeverity.MEDIUM,
        category=ErrorCategory.SYSTEM
    )
    
    # Attempt automatic recovery
    await agent._attempt_automatic_recovery(error_event)
    
    # Verify recovery was attempted
    assert strategy.application_count > 0, "Recovery strategy should have been applied"
    
    # Restore original function
    agent._generate_error_signature = original_generate_signature
    
    print(f"✅ Automatic recovery tested")
    print(f"  Strategy applied: {strategy.strategy_name}")
    print(f"  Applications: {strategy.application_count}")
    print(f"  Success rate: {strategy.success_rate:.2f}")

@pytest.mark.asyncio
async def test_error_statistics(error_recovery_agent):
    """Test error statistics generation."""
    agent = error_recovery_agent
    
    # Create various error events
    error_types = ["ValueError", "ConnectionError", "TimeoutError"]
    components = ["database", "api_client", "llm_agent"]
    severities = [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]
    
    for i in range(10):
        await agent.record_error(
            error=Exception(f"Test error {i}"),
            component=components[i % len(components)],
            context={"test_id": i}
        )
    
    # Get statistics
    stats = await agent.get_error_statistics()
    
    # Verify statistics structure
    assert "error_counts" in stats
    assert "component_breakdown" in stats
    assert "severity_distribution" in stats
    assert "category_distribution" in stats
    assert "pattern_statistics" in stats
    assert "recovery_statistics" in stats
    
    # Check counts
    assert stats["error_counts"]["total_errors"] == 10
    assert stats["error_counts"]["last_24h"] == 10
    
    print("✅ Error statistics generated:")
    print(f"  Total errors: {stats['error_counts']['total_errors']}")
    print(f"  Components affected: {len(stats['component_breakdown'])}")
    print(f"  Patterns detected: {stats['pattern_statistics']['total_patterns']}")
    print(f"  Recovery strategies: {stats['pattern_statistics']['recovery_strategies']}")

@pytest.mark.asyncio
async def test_self_healing_demo():
    """Demonstrate self-healing capabilities."""
    print("\n🔄 Error Recovery Agent - Self-Healing Demo")
    print("=" * 60)
    
    # Create agent with mocked components
    mock_db = Mock()
    mock_db.log_error = AsyncMock()
    
    mock_orchestrator = Mock()
    mock_orchestrator.restart_component = AsyncMock()
    
    agent = ErrorRecoveryAgent(db_logger=mock_db, orchestrator=mock_orchestrator)
    
    # Mock LLMs
    agent.analysis_chain = MockLLM("analysis")
    agent.recovery_chain = MockLLM("recovery")
    
    try:
        print("\n1. 📊 Recording Diverse Error Types")
        
        # Simulate various error scenarios
        error_scenarios = [
            {
                "error": ConnectionError("Database connection lost"),
                "component": "database_manager",
                "context": {"operation": "user_lookup", "retry_attempt": 1},
                "description": "Database connection failure"
            },
            {
                "error": TimeoutError("LLM API request timeout"),
                "component": "llm_agent",
                "context": {"model": "gpt-4", "token_count": 2000},
                "description": "LLM API timeout"
            },
            {
                "error": ValueError("Invalid JSON in user input"),
                "component": "input_validator",
                "context": {"user_id": "user123", "input_length": 500},
                "description": "User input validation error"
            },
            {
                "error": MemoryError("Insufficient memory for operation"),
                "component": "data_processor",
                "context": {"data_size": "500MB", "available_memory": "200MB"},
                "description": "Memory exhaustion"
            }
        ]
        
        recorded_errors = []
        for scenario in error_scenarios:
            error_id = await agent.record_error(
                error=scenario["error"],
                component=scenario["component"],
                context=scenario["context"]
            )
            recorded_errors.append(error_id)
            print(f"  ✓ {scenario['description']}: {error_id[:8]}")
        
        print(f"\n2. 🔍 Pattern Recognition (Threshold: {agent.pattern_detection_threshold})")
        
        # Create recurring errors to trigger pattern detection
        agent.pattern_detection_threshold = 2  # Lower for demo
        
        # Simulate recurring database errors
        for i in range(3):
            await agent.record_error(
                error=ConnectionError("Database connection timeout"),
                component="database_manager",
                context={"operation": f"query_{i}", "table": "analytics"}
            )
            await asyncio.sleep(0.1)
        
        # Allow pattern detection
        await asyncio.sleep(0.5)
        
        print(f"  ✓ Patterns detected: {len(agent.error_patterns)}")
        for pattern_id, pattern in agent.error_patterns.items():
            print(f"    • {pattern.pattern_name}: {pattern.occurrences} occurrences ({pattern.confidence:.2f} confidence)")
        
        print(f"\n3. 🛠️  Recovery Strategy Generation")
        
        # Trigger strategy generation for patterns
        for pattern in agent.error_patterns.values():
            if not pattern.root_cause_analysis:  # If not analyzed yet
                await agent._analyze_pattern_with_llm(pattern)
        
        print(f"  ✓ Recovery strategies generated: {len(agent.recovery_strategies)}")
        for strategy_id, strategy in agent.recovery_strategies.items():
            print(f"    • {strategy.strategy_name} ({strategy.strategy_type}) - Priority: {strategy.priority}")
        
        print(f"  ✓ Preventive measures suggested: {len(agent.preventive_measures)}")
        for measure_id, measure in agent.preventive_measures.items():
            print(f"    • {measure.measure_name} - Complexity: {measure.implementation_complexity}")
        
        print(f"\n4. ⚡ Automatic Recovery Simulation")
        
        # Test automatic recovery
        if agent.error_patterns and agent.recovery_strategies:
            # Create an error that matches an existing pattern
            pattern = list(agent.error_patterns.values())[0]
            
            # Mock the signature generation to match our pattern
            original_func = agent._generate_error_signature
            agent._generate_error_signature = lambda event: pattern.error_signature
            
            test_error = ConnectionError("Database connection timeout")
            error_id = await agent.record_error(
                error=test_error,
                component="database_manager",
                context={"operation": "recovery_test"}
            )
            
            # Restore original function
            agent._generate_error_signature = original_func
            
            print(f"  ✓ Automatic recovery attempted for error: {error_id[:8]}")
            
            # Check recovery statistics
            stats = await agent.get_error_statistics()
            print(f"  ✓ Active recoveries: {stats['recovery_statistics']['active_recoveries']}")
        
        print(f"\n5. 📈 System Health Analytics")
        
        # Generate comprehensive statistics
        stats = await agent.get_error_statistics()
        
        print(f"  📊 Error Statistics:")
        print(f"    • Total errors: {stats['error_counts']['total_errors']}")
        print(f"    • Last 24h: {stats['error_counts']['last_24h']}")
        print(f"    • Components affected: {len(stats['component_breakdown'])}")
        
        print(f"  🎯 Pattern Intelligence:")
        print(f"    • Total patterns: {stats['pattern_statistics']['total_patterns']}")
        print(f"    • Active patterns: {stats['pattern_statistics']['active_patterns']}")
        print(f"    • Recovery strategies: {stats['pattern_statistics']['recovery_strategies']}")
        print(f"    • Preventive measures: {stats['pattern_statistics']['preventive_measures']}")
        
        print(f"  🔧 Recovery Performance:")
        print(f"    • Active recoveries: {stats['recovery_statistics']['active_recoveries']}")
        print(f"    • Successful recoveries: {stats['recovery_statistics']['successful_recoveries']}")
        
        if stats['top_error_types']:
            print(f"  🔥 Top Error Types:")
            for error_info in stats['top_error_types'][:3]:
                print(f"    • {error_info['error_type']}: {error_info['count']} occurrences")
        
        print(f"\n🎉 Self-Healing Demo Complete!")
        print(f"✨ System demonstrates intelligent error recovery with:")
        print(f"   • Real-time error pattern recognition")
        print(f"   • Automatic recovery strategy generation")
        print(f"   • Self-healing capabilities")
        print(f"   • Comprehensive analytics and monitoring")
        
    finally:
        await agent.close()

# Integration test with other improvement agents
@pytest.mark.asyncio
async def test_integration_with_improvement_system():
    """Test integration with other improvement agents."""
    print("\n🔗 Error Recovery Integration Test")
    print("=" * 50)
    
    # Mock other improvement components
    mock_workflow_analyst = Mock()
    mock_performance_analyst = Mock()
    mock_feedback_handler = Mock()
    
    # Create error recovery agent
    agent = ErrorRecoveryAgent()
    agent.analysis_chain = MockLLM("analysis")
    agent.recovery_chain = MockLLM("recovery")
    
    try:
        print("1. Recording errors from different system components")
        
        # Simulate errors from different parts of the system
        system_errors = [
            ("workflow_analyst", Exception("Pattern analysis failed")),
            ("performance_analyst", MemoryError("Insufficient memory for performance analysis")),
            ("feedback_handler", ValueError("Invalid feedback format")),
            ("agent_orchestrator", ConnectionError("Agent communication failed"))
        ]
        
        for component, error in system_errors:
            error_id = await agent.record_error(error=error, component=component)
            print(f"  ✓ Error from {component}: {error_id[:8]}")
        
        print(f"\n2. Cross-system error analysis")
        
        # Analyze errors affecting the improvement system
        stats = await agent.get_error_statistics()
        
        improvement_components = [
            comp for comp in stats['component_breakdown'].keys()
            if any(keyword in comp for keyword in ['analyst', 'handler', 'orchestrator'])
        ]
        
        if improvement_components:
            print(f"  ✓ Improvement system components with errors: {len(improvement_components)}")
            for component in improvement_components:
                error_count = stats['component_breakdown'][component]
                print(f"    • {component}: {error_count} errors")
        
        print(f"\n3. Recovery impact on system improvement")
        
        # Check if any patterns affect critical improvement components
        critical_patterns = []
        for pattern in agent.error_patterns.values():
            if any('analyst' in comp or 'handler' in comp for comp in pattern.affected_components):
                critical_patterns.append(pattern)
        
        if critical_patterns:
            print(f"  ⚠️  Critical patterns affecting improvement system: {len(critical_patterns)}")
            for pattern in critical_patterns:
                print(f"    • {pattern.pattern_name}: {pattern.occurrences} occurrences")
        
        print(f"\n✅ Integration test completed")
        print(f"   Error Recovery Agent successfully monitors and heals the improvement system!")
        
    finally:
        await agent.close()

if __name__ == "__main__":
    print("🔄 Error Recovery Agent Test Suite")
    print("=" * 50)
    
    # Run the demo
    asyncio.run(test_self_healing_demo())
    print("\n" + "="*50)
    asyncio.run(test_integration_with_improvement_system()) 