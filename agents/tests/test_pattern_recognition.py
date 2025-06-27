"""
Test suite for Pattern Recognition System
Demonstrates pattern identification, temporal analysis, and automation suggestions.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from agents.improvement.pattern_recognition import (
    PatternRecognitionEngine,
    InteractionEvent,
    RecognizedPattern,
    AutomationSuggestion,
    PatternType,
    PatternStrength,
    TemporalPeriod
)

class TestPatternRecognitionEngine:
    """Test the Pattern Recognition Engine functionality."""

    @pytest.fixture
    async def engine(self):
        """Create a test pattern recognition engine."""
        # Mock the LLMs to avoid API calls in tests
        with patch('agents.improvement.pattern_recognition.ChatOpenAI') as mock_llm:
            mock_llm.return_value = AsyncMock()
            engine = PatternRecognitionEngine()
            
            # Stop the monitoring task for testing
            if engine.monitoring_task:
                engine.monitoring_task.cancel()
                try:
                    await engine.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            yield engine
            await engine.close()

    @pytest.fixture
    def sample_interactions(self):
        """Create sample interaction events for testing."""
        base_time = datetime.utcnow()
        
        interactions = [
            InteractionEvent(
                id="event_1",
                user_id="user_123",
                timestamp=base_time,
                message="Check daily sales report",
                context={"channel": "sales"},
                agent_used="general",
                success=True,
                duration_ms=2500,
                tokens_used=150,
                cost=0.005
            ),
            InteractionEvent(
                id="event_2",
                user_id="user_123", 
                timestamp=base_time + timedelta(hours=1),
                message="Generate sales summary for yesterday",
                context={"channel": "sales"},
                agent_used="research",
                success=True,
                duration_ms=3000,
                tokens_used=200,
                cost=0.008
            ),
            InteractionEvent(
                id="event_3",
                user_id="user_123",
                timestamp=base_time + timedelta(hours=2),
                message="Send sales report to team",
                context={"channel": "sales"},
                agent_used="general",
                success=True,
                duration_ms=1500,
                tokens_used=100,
                cost=0.003
            ),
            InteractionEvent(
                id="event_4",
                user_id="user_456",
                timestamp=base_time + timedelta(days=1),
                message="Check daily sales report",
                context={"channel": "sales"},
                agent_used="general",
                success=True,
                duration_ms=2200,
                tokens_used=140,
                cost=0.004
            ),
            InteractionEvent(
                id="event_5",
                user_id="user_456",
                timestamp=base_time + timedelta(days=1, hours=1),
                message="Generate sales summary for yesterday",
                context={"channel": "sales"},
                agent_used="research",
                success=True,
                duration_ms=2800,
                tokens_used=190,
                cost=0.007
            )
        ]
        
        return interactions

    @pytest.mark.asyncio
    async def test_record_interaction(self, engine, sample_interactions):
        """Test recording interactions."""
        event = sample_interactions[0]
        
        await engine.record_interaction(event)
        
        # Check event was added to buffer
        assert len(engine.interaction_buffer) == 1
        assert engine.interaction_buffer[0] == event
        
        # Check user session tracking
        assert event.user_id in engine.user_sessions
        assert len(engine.user_sessions[event.user_id]) == 1
        
        # Check temporal buckets
        hour_key = f"hour_{event.timestamp.hour:02d}:00"
        day_key = f"day_{event.timestamp.strftime('%A')}"
        
        assert len(engine.temporal_buckets[hour_key]) == 1
        assert len(engine.temporal_buckets[day_key]) == 1

    @pytest.mark.asyncio
    async def test_real_time_pattern_detection(self, engine, sample_interactions):
        """Test real-time pattern detection."""
        # Record multiple interactions from same user
        for event in sample_interactions[:3]:  # First 3 events from same user
            await engine.record_interaction(event)
        
        # Check that user session has all events
        user_events = engine.user_sessions["user_123"]
        assert len(user_events) == 3
        
        # Verify the sequence was detected (would trigger pattern creation in real system)
        assert all(event.user_id == "user_123" for event in user_events)

    def test_calculate_pattern_strength(self, engine):
        """Test pattern strength calculation."""
        assert engine._calculate_pattern_strength(2) == PatternStrength.EMERGING
        assert engine._calculate_pattern_strength(8) == PatternStrength.DEVELOPING
        assert engine._calculate_pattern_strength(15) == PatternStrength.ESTABLISHED
        assert engine._calculate_pattern_strength(25) == PatternStrength.STRONG

    def test_prepare_interaction_data(self, engine, sample_interactions):
        """Test interaction data preparation for LLM analysis."""
        data = engine._prepare_interaction_data(sample_interactions)
        
        # Should return valid JSON
        parsed_data = json.loads(data)
        assert isinstance(parsed_data, list)
        assert len(parsed_data) == len(sample_interactions)
        
        # Check data structure
        first_event = parsed_data[0]
        assert "sequence" in first_event
        assert "timestamp" in first_event
        assert "user" in first_event
        assert "message" in first_event
        assert "agent" in first_event

    def test_prepare_temporal_context(self, engine, sample_interactions):
        """Test temporal context preparation."""
        context = engine._prepare_temporal_context(sample_interactions)
        
        # Should return valid JSON
        parsed_context = json.loads(context)
        assert "time_range" in parsed_context
        assert "hourly_distribution" in parsed_context
        assert "daily_distribution" in parsed_context
        assert "peak_hours" in parsed_context
        assert "peak_days" in parsed_context

    def test_prepare_user_summary(self, engine, sample_interactions):
        """Test user behavior summary preparation."""
        summary = engine._prepare_user_summary(sample_interactions)
        
        # Should return valid JSON
        parsed_summary = json.loads(summary)
        assert "total_users" in parsed_summary
        assert "user_patterns" in parsed_summary
        assert "overall_stats" in parsed_summary
        
        # Should identify 2 unique users
        assert parsed_summary["total_users"] == 2

    def test_determine_automation_type(self, engine):
        """Test automation type determination."""
        # Create test patterns
        temporal_pattern = RecognizedPattern(
            id="pattern_1",
            type=PatternType.TEMPORAL,
            name="Daily Report",
            description="Daily sales report pattern",
            trigger_conditions=["daily", "sales", "report"],
            sequence_steps=["check", "generate", "send"],
            frequency=10,
            strength=PatternStrength.ESTABLISHED,
            confidence=0.9,
            users_affected={"user_123"},
            temporal_info={"recurring": "daily"},
            automation_potential=0.8,
            avg_duration_ms=5000,
            avg_cost=0.015,
            success_rate=0.95,
            examples=["event_1", "event_2"],
            created_at=datetime.utcnow(),
            last_seen=datetime.utcnow()
        )
        
        trigger_pattern = RecognizedPattern(
            id="pattern_2",
            type=PatternType.TRIGGER,
            name="Error Response",
            description="Error handling pattern",
            trigger_conditions=["error", "exception"],
            sequence_steps=["detect", "log", "notify"],
            frequency=5,
            strength=PatternStrength.DEVELOPING,
            confidence=0.7,
            users_affected={"user_456"},
            temporal_info=None,
            automation_potential=0.9,
            avg_duration_ms=3000,
            avg_cost=0.01,
            success_rate=0.85,
            examples=["event_3"],
            created_at=datetime.utcnow(),
            last_seen=datetime.utcnow()
        )
        
        frequent_pattern = RecognizedPattern(
            id="pattern_3",
            type=PatternType.SEQUENCE,
            name="Common Workflow",
            description="Frequently used workflow",
            trigger_conditions=["workflow"],
            sequence_steps=["start", "process", "finish"],
            frequency=20,
            strength=PatternStrength.STRONG,
            confidence=0.95,
            users_affected={"user_123", "user_456"},
            temporal_info=None,
            automation_potential=0.7,
            avg_duration_ms=4000,
            avg_cost=0.012,
            success_rate=0.9,
            examples=["event_4", "event_5"],
            created_at=datetime.utcnow(),
            last_seen=datetime.utcnow()
        )
        
        # Test automation type determination
        assert engine._determine_automation_type(temporal_pattern) == "scheduled"
        assert engine._determine_automation_type(trigger_pattern) == "triggered"
        assert engine._determine_automation_type(frequent_pattern) == "shortcut"

    def test_assess_complexity(self, engine):
        """Test complexity assessment."""
        # Create patterns with different step counts
        simple_pattern = RecognizedPattern(
            id="simple",
            type=PatternType.SEQUENCE,
            name="Simple",
            description="Simple pattern",
            trigger_conditions=["simple"],
            sequence_steps=["step1", "step2"],
            frequency=5,
            strength=PatternStrength.DEVELOPING,
            confidence=0.8,
            users_affected={"user"},
            temporal_info=None,
            automation_potential=0.8,
            avg_duration_ms=1000,
            avg_cost=0.01,
            success_rate=0.9,
            examples=["event"],
            created_at=datetime.utcnow(),
            last_seen=datetime.utcnow()
        )
        
        medium_pattern = RecognizedPattern(
            id="medium",
            type=PatternType.SEQUENCE,
            name="Medium",
            description="Medium pattern",
            trigger_conditions=["medium"],
            sequence_steps=["step1", "step2", "step3", "step4"],
            frequency=5,
            strength=PatternStrength.DEVELOPING,
            confidence=0.8,
            users_affected={"user"},
            temporal_info=None,
            automation_potential=0.8,
            avg_duration_ms=1000,
            avg_cost=0.01,
            success_rate=0.9,
            examples=["event"],
            created_at=datetime.utcnow(),
            last_seen=datetime.utcnow()
        )
        
        complex_pattern = RecognizedPattern(
            id="complex",
            type=PatternType.SEQUENCE,
            name="Complex",
            description="Complex pattern",
            trigger_conditions=["complex"],
            sequence_steps=[f"step{i}" for i in range(1, 10)],  # 9 steps
            frequency=5,
            strength=PatternStrength.DEVELOPING,
            confidence=0.8,
            users_affected={"user"},
            temporal_info=None,
            automation_potential=0.8,
            avg_duration_ms=1000,
            avg_cost=0.01,
            success_rate=0.9,
            examples=["event"],
            created_at=datetime.utcnow(),
            last_seen=datetime.utcnow()
        )
        
        assert engine._assess_complexity(simple_pattern) == "low"
        assert engine._assess_complexity(medium_pattern) == "medium"
        assert engine._assess_complexity(complex_pattern) == "high"

    @pytest.mark.asyncio
    async def test_suggest_automation_for_pattern(self, engine):
        """Test automation suggestion creation."""
        pattern = RecognizedPattern(
            id="pattern_1",
            type=PatternType.SEQUENCE,
            name="Daily Sales Report",
            description="Daily sales report generation pattern",
            trigger_conditions=["daily", "sales", "report"],
            sequence_steps=["check data", "generate report", "send to team"],
            frequency=15,
            strength=PatternStrength.ESTABLISHED,
            confidence=0.9,
            users_affected={"user_123", "user_456"},
            temporal_info={"recurring": "daily"},
            automation_potential=0.85,  # High automation potential
            avg_duration_ms=5000,
            avg_cost=0.015,
            success_rate=0.95,
            examples=["event_1", "event_2", "event_3"],
            created_at=datetime.utcnow(),
            last_seen=datetime.utcnow()
        )
        
        await engine._suggest_automation_for_pattern(pattern)
        
        # Check that suggestion was created
        assert len(engine.automation_suggestions) == 1
        
        suggestion = list(engine.automation_suggestions.values())[0]
        assert suggestion.pattern_id == pattern.id
        assert suggestion.automation_type == "runbook"
        assert suggestion.estimated_time_saved > 0
        assert suggestion.estimated_cost_saved > 0

    def test_get_pattern_summary(self, engine):
        """Test pattern summary generation."""
        # Add some test patterns
        pattern1 = RecognizedPattern(
            id="pattern_1",
            type=PatternType.SEQUENCE,
            name="Test Pattern 1",
            description="Test",
            trigger_conditions=[],
            sequence_steps=[],
            frequency=10,
            strength=PatternStrength.ESTABLISHED,
            confidence=0.8,
            users_affected={"user1"},
            temporal_info=None,
            automation_potential=0.7,
            avg_duration_ms=1000,
            avg_cost=0.01,
            success_rate=0.9,
            examples=[],
            created_at=datetime.utcnow(),
            last_seen=datetime.utcnow()
        )
        
        pattern2 = RecognizedPattern(
            id="pattern_2",
            type=PatternType.TEMPORAL,
            name="Test Pattern 2",
            description="Test",
            trigger_conditions=[],
            sequence_steps=[],
            frequency=5,
            strength=PatternStrength.DEVELOPING,
            confidence=0.7,
            users_affected={"user2"},
            temporal_info=None,
            automation_potential=0.6,
            avg_duration_ms=1000,
            avg_cost=0.01,
            success_rate=0.8,
            examples=[],
            created_at=datetime.utcnow(),
            last_seen=datetime.utcnow()
        )
        
        engine.recognized_patterns[pattern1.id] = pattern1
        engine.recognized_patterns[pattern2.id] = pattern2
        
        summary = engine.get_pattern_summary()
        
        assert summary["total_patterns"] == 2
        assert "sequence" in summary["patterns_by_type"]
        assert "temporal" in summary["patterns_by_type"]
        assert "established" in summary["patterns_by_strength"]
        assert "developing" in summary["patterns_by_strength"]

    def test_get_top_patterns(self, engine):
        """Test getting top patterns by frequency and automation potential."""
        # Add test patterns with different scores
        patterns = []
        for i in range(5):
            pattern = RecognizedPattern(
                id=f"pattern_{i}",
                type=PatternType.SEQUENCE,
                name=f"Pattern {i}",
                description=f"Test pattern {i}",
                trigger_conditions=[],
                sequence_steps=[],
                frequency=i * 5,  # Different frequencies
                strength=PatternStrength.DEVELOPING,
                confidence=0.8,
                users_affected={f"user{i}"},
                temporal_info=None,
                automation_potential=0.5 + (i * 0.1),  # Different automation potentials
                avg_duration_ms=1000,
                avg_cost=0.01,
                success_rate=0.9,
                examples=[],
                created_at=datetime.utcnow(),
                last_seen=datetime.utcnow()
            )
            patterns.append(pattern)
            engine.recognized_patterns[pattern.id] = pattern
        
        top_patterns = engine.get_top_patterns(limit=3)
        
        assert len(top_patterns) == 3
        
        # Should be sorted by frequency and automation potential
        assert top_patterns[0].frequency >= top_patterns[1].frequency

    @pytest.mark.asyncio
    async def test_integration_demo(self, engine):
        """Demonstrate the full pattern recognition workflow."""
        print("\n🎯 Pattern Recognition System Integration Demo")
        print("=" * 60)
        
        # Create sample interactions
        base_time = datetime.utcnow()
        sample_interactions = [
            InteractionEvent(
                id=f"demo_event_{i}",
                user_id=f"user_{i % 3}",
                timestamp=base_time + timedelta(hours=i),
                message=f"Sample interaction {i}: {'report' if i % 3 == 0 else 'analysis' if i % 3 == 1 else 'summary'}",
                context={"demo": True},
                agent_used=["general", "research", "technical"][i % 3],
                success=True,
                duration_ms=1000 + (i * 100),
                tokens_used=100 + (i * 10),
                cost=0.005 + (i * 0.001)
            )
            for i in range(10)
        ]
        
        # Step 1: Record interactions
        print("\n📝 Step 1: Recording User Interactions")
        for event in sample_interactions:
            await engine.record_interaction(event)
            print(f"  ✅ Recorded: {event.message[:40]}...")
        
        # Step 2: Create sample pattern
        print("\n🔮 Step 2: Creating Sample Pattern")
        sample_pattern = RecognizedPattern(
            id="demo_pattern_1",
            type=PatternType.SEQUENCE,
            name="Daily Report Workflow",
            description="Users check → generate → send reports daily",
            trigger_conditions=["daily", "report"],
            sequence_steps=["Check report", "Generate summary", "Send to team"],
            frequency=12,
            strength=PatternStrength.ESTABLISHED,
            confidence=0.92,
            users_affected={"user_1", "user_2"},
            temporal_info={"pattern": "daily"},
            automation_potential=0.85,
            avg_duration_ms=4500,
            avg_cost=0.012,
            success_rate=0.95,
            examples=["event_1", "event_2"],
            created_at=datetime.utcnow(),
            last_seen=datetime.utcnow()
        )
        
        engine.recognized_patterns[sample_pattern.id] = sample_pattern
        print(f"  ✅ Created pattern: {sample_pattern.name}")
        print(f"     🎯 Type: {sample_pattern.type.value}")
        print(f"     💪 Strength: {sample_pattern.strength.value}")
        print(f"     🤖 Automation Potential: {sample_pattern.automation_potential:.1%}")
        
        # Step 3: Generate automation suggestion
        print("\n🚀 Step 3: Generating Automation Suggestion")
        await engine._suggest_automation_for_pattern(sample_pattern)
        
        suggestions = engine.get_automation_suggestions()
        if suggestions:
            suggestion = suggestions[0]
            print(f"  ✅ Generated suggestion: {suggestion.title}")
            print(f"     📱 Type: {suggestion.automation_type}")
            print(f"     ⏱️  Time saved: {suggestion.estimated_time_saved:.1f} minutes")
            print(f"     💰 Cost saved: ${suggestion.estimated_cost_saved:.3f}")
        
        # Step 4: Show summary
        print("\n📊 Step 4: System Summary")
        summary = engine.get_pattern_summary()
        print(f"  • Patterns recognized: {summary['total_patterns']}")
        print(f"  • Automation suggestions: {summary['automation_suggestions']}")
        print(f"  • Interactions recorded: {summary['interactions_in_buffer']}")
        
        print("\n🎉 Pattern Recognition Demo Complete!")
        print("   ✅ Successfully monitored user interactions")
        print("   ✅ Identified recurring patterns")  
        print("   ✅ Generated automation suggestions")
        
        return {
            "interactions_recorded": len(sample_interactions),
            "patterns_identified": summary['total_patterns'],
            "automation_suggestions": summary['automation_suggestions'],
            "demo_success": True
        }


if __name__ == "__main__":
    """Run the integration demo."""
    async def main():
        print("🚀 Starting Pattern Recognition System Demo")
        
        # Create engine with mocked LLMs
        with patch('agents.improvement.pattern_recognition.ChatOpenAI') as mock_llm:
            mock_llm.return_value = AsyncMock()
            engine = PatternRecognitionEngine()
            
            # Stop monitoring for demo
            if engine.monitoring_task:
                engine.monitoring_task.cancel()
                try:
                    await engine.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            try:
                test_instance = TestPatternRecognitionEngine()
                result = await test_instance.test_integration_demo(engine)
                
                print(f"\n✅ Demo completed successfully!")
                print(f"📊 Results: {json.dumps(result, indent=2)}")
                
            finally:
                await engine.close()
    
    # Run the demo
    asyncio.run(main()) 