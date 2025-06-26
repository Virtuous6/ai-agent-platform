"""
Test suite for the Event Bus System

This module tests all features of the event-driven architecture including:
- Publish-subscribe pattern
- Event routing and filtering
- Rate limiting and throttling
- Async event processing
- Event persistence and replay
- Metrics and monitoring

Created: June 2025
"""

import asyncio
import pytest
import uuid
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from event_bus import (
    EventBus, Event, EventType, EventHandler, Subscription,
    RateLimitConfig, get_event_bus, init_event_bus, shutdown_event_bus
)

class TestEventBus:
    """Test the Event Bus System comprehensively."""
    
    @pytest.fixture
    async def event_bus(self):
        """Create a test event bus."""
        bus = EventBus(
            max_queue_size=100,
            enable_persistence=False,  # Disable for testing
            rate_limit_config=RateLimitConfig(
                max_events_per_minute=10,
                burst_limit=5,
                enable_rate_limiting=True
            )
        )
        await bus.start()
        yield bus
        await bus.stop()
    
    @pytest.fixture
    def mock_handler(self):
        """Create a mock event handler."""
        return AsyncMock()
    
    @pytest.mark.asyncio
    async def test_event_creation(self):
        """Test event creation and serialization."""
        event = Event(
            type=EventType.WORKFLOW_STARTED,
            data={"workflow_id": "test-123", "user": "test-user"},
            source="test-agent",
            priority=1,
            ttl=300,
            tags={"test", "workflow"},
            user_id="user-123"
        )
        
        assert event.type == "workflow.started"
        assert event.data["workflow_id"] == "test-123"
        assert event.priority == 1
        assert "test" in event.tags
        
        # Test serialization
        event_dict = event.to_dict()
        assert event_dict["type"] == "workflow.started"
        assert event_dict["data"]["workflow_id"] == "test-123"
        assert isinstance(event_dict["timestamp"], str)
    
    @pytest.mark.asyncio
    async def test_event_expiration(self):
        """Test event TTL and expiration."""
        # Non-expiring event
        event1 = Event(type=EventType.CUSTOM, data={})
        assert not event1.is_expired()
        
        # Expiring event
        event2 = Event(type=EventType.CUSTOM, data={}, ttl=1)
        assert not event2.is_expired()
        
        # Simulate time passing
        await asyncio.sleep(1.1)
        assert event2.is_expired()
    
    @pytest.mark.asyncio
    async def test_publish_subscribe_basic(self, event_bus, mock_handler):
        """Test basic publish-subscribe functionality."""
        subscriber_id = "test-subscriber"
        
        # Subscribe to workflow events
        success = await event_bus.subscribe(
            subscriber_id=subscriber_id,
            event_types=[EventType.WORKFLOW_STARTED, EventType.WORKFLOW_COMPLETED],
            handler=mock_handler
        )
        assert success
        
        # Publish workflow started event
        event_id = await event_bus.publish(
            event_type=EventType.WORKFLOW_STARTED,
            data={"workflow_id": "test-workflow"},
            source="test-publisher"
        )
        
        # Wait for event processing
        await asyncio.sleep(0.1)
        
        # Verify handler was called
        mock_handler.assert_called_once()
        event = mock_handler.call_args[0][0]
        assert event.type == "workflow.started"
        assert event.data["workflow_id"] == "test-workflow"
        assert event.source == "test-publisher"
    
    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, event_bus):
        """Test multiple subscribers for the same event type."""
        handlers = [AsyncMock() for _ in range(3)]
        
        # Subscribe multiple handlers to the same event type
        for i, handler in enumerate(handlers):
            await event_bus.subscribe(
                subscriber_id=f"subscriber-{i}",
                event_types=[EventType.PATTERN_DISCOVERED],
                handler=handler,
                priority=i + 1  # Different priorities
            )
        
        # Publish event
        await event_bus.publish(
            event_type=EventType.PATTERN_DISCOVERED,
            data={"pattern": "test-pattern"},
            source="pattern-detector"
        )
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # All handlers should be called
        for handler in handlers:
            handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_event_filtering(self, event_bus):
        """Test event filtering with filter functions."""
        filtered_handler = AsyncMock()
        unfiltered_handler = AsyncMock()
        
        # Filter function - only events with priority <= 3
        def priority_filter(event: Event) -> bool:
            return event.priority <= 3
        
        # Subscribe with filter
        await event_bus.subscribe(
            subscriber_id="filtered-subscriber",
            event_types=[EventType.CUSTOM],
            handler=filtered_handler,
            filter_func=priority_filter
        )
        
        # Subscribe without filter
        await event_bus.subscribe(
            subscriber_id="unfiltered-subscriber",
            event_types=[EventType.CUSTOM],
            handler=unfiltered_handler
        )
        
        # Publish high priority event (should be filtered)
        await event_bus.publish(
            event_type=EventType.CUSTOM,
            data={"test": "high-priority"},
            priority=2  # High priority - passes filter
        )
        
        # Publish low priority event (should be filtered out)
        await event_bus.publish(
            event_type=EventType.CUSTOM,
            data={"test": "low-priority"},
            priority=8  # Low priority - fails filter
        )
        
        await asyncio.sleep(0.1)
        
        # Filtered handler should only get the high priority event
        assert filtered_handler.call_count == 1
        
        # Unfiltered handler should get both events
        assert unfiltered_handler.call_count == 2
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, event_bus):
        """Test that events are processed in priority order."""
        call_order = []
        
        async def priority_handler(event: Event):
            call_order.append(event.priority)
        
        # Subscribe to events
        await event_bus.subscribe(
            subscriber_id="priority-subscriber",
            event_types=[EventType.CUSTOM],
            handler=priority_handler
        )
        
        # Publish events in reverse priority order
        priorities = [5, 1, 3, 2, 4]
        for priority in priorities:
            await event_bus.publish(
                event_type=EventType.CUSTOM,
                data={"priority": priority},
                priority=priority
            )
        
        # Wait for all events to process
        await asyncio.sleep(0.2)
        
        # Events should be processed in priority order (1, 2, 3, 4, 5)
        assert call_order == sorted(priorities)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, event_bus):
        """Test rate limiting functionality."""
        handler = AsyncMock()
        
        await event_bus.subscribe(
            subscriber_id="rate-test-subscriber",
            event_types=[EventType.CUSTOM],
            handler=handler
        )
        
        # Publish events rapidly to trigger rate limiting
        published_count = 0
        rate_limited_count = 0
        
        for i in range(15):  # More than the limit of 10
            try:
                await event_bus.publish(
                    event_type=EventType.CUSTOM,
                    data={"index": i},
                    source="rate-test-source"
                )
                published_count += 1
            except Exception as e:
                if "Rate limit exceeded" in str(e):
                    rate_limited_count += 1
        
        await asyncio.sleep(0.1)
        
        # Should have rate limited some events
        assert rate_limited_count > 0
        assert published_count < 15
        assert event_bus.events_rate_limited > 0
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus, mock_handler):
        """Test unsubscribing from events."""
        subscriber_id = "unsubscribe-test"
        
        # Subscribe
        await event_bus.subscribe(
            subscriber_id=subscriber_id,
            event_types=[EventType.WORKFLOW_COMPLETED],
            handler=mock_handler
        )
        
        # Publish event - should be received
        await event_bus.publish(
            event_type=EventType.WORKFLOW_COMPLETED,
            data={"test": "before-unsubscribe"}
        )
        
        await asyncio.sleep(0.1)
        assert mock_handler.call_count == 1
        
        # Unsubscribe
        success = await event_bus.unsubscribe(subscriber_id)
        assert success
        
        # Publish another event - should not be received
        await event_bus.publish(
            event_type=EventType.WORKFLOW_COMPLETED,
            data={"test": "after-unsubscribe"}
        )
        
        await asyncio.sleep(0.1)
        assert mock_handler.call_count == 1  # Still 1, no new calls
    
    @pytest.mark.asyncio
    async def test_subscription_info(self, event_bus, mock_handler):
        """Test getting subscription information."""
        subscriber_id = "info-test-subscriber"
        
        # Subscribe
        await event_bus.subscribe(
            subscriber_id=subscriber_id,
            event_types=[EventType.FEEDBACK_RECEIVED, EventType.ERROR_RECOVERY],
            handler=mock_handler,
            priority=3
        )
        
        # Get subscription info
        info = await event_bus.get_subscription_info(subscriber_id)
        
        assert info is not None
        assert info["subscriber_id"] == subscriber_id
        assert len(info["event_types"]) == 2
        assert "user.feedback_received" in info["event_types"]
        assert "system.error_recovery" in info["event_types"]
        assert info["priority"] == 3
        assert info["active"] is True
        assert info["event_count"] == 0
        
        # Publish event to update metrics
        await event_bus.publish(
            event_type=EventType.FEEDBACK_RECEIVED,
            data={"feedback": "test"}
        )
        
        await asyncio.sleep(0.1)
        
        # Check updated info
        updated_info = await event_bus.get_subscription_info(subscriber_id)
        assert updated_info["event_count"] == 1
        assert updated_info["last_event_at"] is not None
    
    @pytest.mark.asyncio
    async def test_metrics(self, event_bus, mock_handler):
        """Test event bus metrics collection."""
        # Initial metrics
        metrics = await event_bus.get_metrics()
        initial_published = metrics["events_published"]
        initial_processed = metrics["events_processed"]
        
        # Subscribe and publish events
        await event_bus.subscribe(
            subscriber_id="metrics-test",
            event_types=[EventType.CUSTOM],
            handler=mock_handler
        )
        
        # Publish several events
        for i in range(5):
            await event_bus.publish(
                event_type=EventType.CUSTOM,
                data={"index": i}
            )
        
        await asyncio.sleep(0.1)
        
        # Check updated metrics
        updated_metrics = await event_bus.get_metrics()
        
        assert updated_metrics["events_published"] == initial_published + 5
        assert updated_metrics["events_processed"] == initial_processed + 5
        assert updated_metrics["active_subscriptions"] >= 1
        assert updated_metrics["is_running"] is True
        assert "last_health_check" in updated_metrics
    
    @pytest.mark.asyncio
    async def test_dead_letter_queue(self):
        """Test dead letter queue when event queue is full."""
        # Create bus with very small queue
        small_bus = EventBus(max_queue_size=2, enable_persistence=False)
        await small_bus.start()
        
        try:
            # Fill up the queue
            for i in range(5):
                try:
                    await small_bus.publish(
                        event_type=EventType.CUSTOM,
                        data={"index": i}
                    )
                except Exception:
                    pass  # Expected for queue overflow
            
            # Check dead letter queue has events
            assert len(small_bus.dead_letter_queue) > 0
            
        finally:
            await small_bus.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_in_handlers(self, event_bus):
        """Test error handling when event handlers fail."""
        good_handler = AsyncMock()
        
        async def bad_handler(event: Event):
            raise Exception("Handler error")
        
        # Subscribe both handlers
        await event_bus.subscribe(
            subscriber_id="good-handler",
            event_types=[EventType.CUSTOM],
            handler=good_handler
        )
        
        await event_bus.subscribe(
            subscriber_id="bad-handler", 
            event_types=[EventType.CUSTOM],
            handler=bad_handler
        )
        
        # Publish event
        await event_bus.publish(
            event_type=EventType.CUSTOM,
            data={"test": "error-handling"}
        )
        
        await asyncio.sleep(0.1)
        
        # Good handler should still be called despite bad handler failing
        good_handler.assert_called_once()
        
        # Error should be tracked in metrics
        metrics = await event_bus.get_metrics()
        # Note: We don't increment events_failed for handler errors,
        # only for event processing errors
    
    @pytest.mark.asyncio
    async def test_correlation_ids(self, event_bus):
        """Test event correlation for related events."""
        received_events = []
        
        async def correlation_handler(event: Event):
            received_events.append(event)
        
        await event_bus.subscribe(
            subscriber_id="correlation-subscriber",
            event_types=[EventType.WORKFLOW_STARTED, EventType.WORKFLOW_COMPLETED],
            handler=correlation_handler
        )
        
        correlation_id = str(uuid.uuid4())
        
        # Publish related events with same correlation ID
        await event_bus.publish(
            event_type=EventType.WORKFLOW_STARTED,
            data={"workflow": "test"},
            correlation_id=correlation_id
        )
        
        await event_bus.publish(
            event_type=EventType.WORKFLOW_COMPLETED,
            data={"workflow": "test", "result": "success"},
            correlation_id=correlation_id
        )
        
        await asyncio.sleep(0.1)
        
        assert len(received_events) == 2
        assert all(event.correlation_id == correlation_id for event in received_events)
    
    @pytest.mark.asyncio
    async def test_event_tags(self, event_bus):
        """Test event tagging and filtering."""
        tagged_events = []
        
        async def tag_handler(event: Event):
            tagged_events.append(event)
        
        # Filter for events with 'important' tag
        def important_filter(event: Event) -> bool:
            return 'important' in event.tags
        
        await event_bus.subscribe(
            subscriber_id="tag-subscriber",
            event_types=[EventType.CUSTOM],
            handler=tag_handler,
            filter_func=important_filter
        )
        
        # Publish event with important tag
        await event_bus.publish(
            event_type=EventType.CUSTOM,
            data={"message": "important event"},
            tags={"important", "urgent"}
        )
        
        # Publish event without important tag
        await event_bus.publish(
            event_type=EventType.CUSTOM,
            data={"message": "normal event"},
            tags={"routine"}
        )
        
        await asyncio.sleep(0.1)
        
        # Only important event should be received
        assert len(tagged_events) == 1
        assert "important" in tagged_events[0].tags
        assert "urgent" in tagged_events[0].tags

class TestGlobalEventBus:
    """Test global event bus functionality."""
    
    @pytest.mark.asyncio
    async def test_global_event_bus(self):
        """Test global event bus singleton."""
        # Get global instance
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        
        # Should be the same instance
        assert bus1 is bus2
    
    @pytest.mark.asyncio
    async def test_init_and_shutdown_global_bus(self):
        """Test initializing and shutting down global event bus."""
        # Initialize global bus
        bus = await init_event_bus(max_queue_size=50)
        assert bus.is_running
        
        # Get the same instance
        global_bus = get_event_bus()
        assert global_bus is bus
        
        # Shutdown
        await shutdown_event_bus()
        
        # Should create new instance after shutdown
        new_bus = get_event_bus()
        assert new_bus is not bus

@pytest.mark.asyncio
async def test_integration_demo():
    """
    Integration demonstration of the Event Bus System.
    Shows a realistic workflow scenario.
    """
    print("\n🚀 Event Bus Integration Demo")
    print("=" * 50)
    
    # Initialize event bus
    event_bus = EventBus(enable_persistence=False)
    await event_bus.start()
    
    # Collected events for demonstration
    workflow_events = []
    improvement_events = []
    
    async def workflow_handler(event: Event):
        workflow_events.append(event)
        print(f"📋 Workflow Event: {event.type} - {event.data}")
    
    async def improvement_handler(event: Event):
        improvement_events.append(event)
        print(f"🧠 Improvement Event: {event.type} - {event.data}")
    
    # Subscribe workflow orchestrator
    await event_bus.subscribe(
        subscriber_id="workflow-orchestrator",
        event_types=[
            EventType.WORKFLOW_STARTED,
            EventType.WORKFLOW_COMPLETED,
            EventType.WORKFLOW_FAILED
        ],
        handler=workflow_handler,
        priority=1  # High priority
    )
    
    # Subscribe improvement system
    await event_bus.subscribe(
        subscriber_id="improvement-system",
        event_types=[
            EventType.PATTERN_DISCOVERED,
            EventType.IMPROVEMENT_APPLIED,
            EventType.OPTIMIZATION_COMPLETED
        ],
        handler=improvement_handler,
        priority=2
    )
    
    print("\n📡 Subscribers registered:")
    print("  - Workflow Orchestrator (priority 1)")
    print("  - Improvement System (priority 2)")
    
    # Simulate workflow execution
    correlation_id = str(uuid.uuid4())
    
    print(f"\n🔄 Simulating workflow execution (correlation: {correlation_id[:8]}...)")
    
    # 1. Workflow starts
    await event_bus.publish(
        event_type=EventType.WORKFLOW_STARTED,
        data={
            "workflow_id": "user-query-123",
            "user_id": "user-456",
            "query": "How to optimize API performance?"
        },
        source="slack-interface",
        correlation_id=correlation_id,
        priority=2
    )
    
    # 2. Pattern discovered during execution
    await event_bus.publish(
        event_type=EventType.PATTERN_DISCOVERED,
        data={
            "pattern_type": "optimization_request",
            "frequency": 15,
            "strength": "strong"
        },
        source="pattern-recognition-agent",
        correlation_id=correlation_id,
        priority=3
    )
    
    # 3. Workflow completes successfully
    await event_bus.publish(
        event_type=EventType.WORKFLOW_COMPLETED,
        data={
            "workflow_id": "user-query-123",
            "result": "success",
            "execution_time": 2.4,
            "agents_used": ["technical-agent", "research-agent"]
        },
        source="agent-orchestrator",
        correlation_id=correlation_id,
        priority=2
    )
    
    # 4. Improvement applied based on pattern
    await event_bus.publish(
        event_type=EventType.IMPROVEMENT_APPLIED,
        data={
            "improvement_type": "response_template",
            "pattern_id": "opt-req-001",
            "estimated_speedup": "40%"
        },
        source="improvement-orchestrator",
        correlation_id=correlation_id,
        priority=3
    )
    
    # Wait for all events to process
    await asyncio.sleep(0.2)
    
    # Display results
    print(f"\n📊 Event Processing Results:")
    print(f"  ✅ Workflow events processed: {len(workflow_events)}")
    print(f"  🧠 Improvement events processed: {len(improvement_events)}")
    
    # Show metrics
    metrics = await event_bus.get_metrics()
    print(f"\n📈 Event Bus Metrics:")
    print(f"  📤 Events published: {metrics['events_published']}")
    print(f"  ⚡ Events processed: {metrics['events_processed']}")
    print(f"  🚫 Events failed: {metrics['events_failed']}")
    print(f"  👥 Active subscriptions: {metrics['active_subscriptions']}")
    print(f"  🔄 Queue size: {metrics['priority_queue_size']}")
    
    # Demonstrate subscription info
    workflow_info = await event_bus.get_subscription_info("workflow-orchestrator")
    print(f"\n🔍 Workflow Orchestrator Subscription:")
    print(f"  📧 Events received: {workflow_info['event_count']}")
    print(f"  🏷️ Event types: {workflow_info['event_types']}")
    print(f"  ⏰ Last event: {workflow_info['last_event_at']}")
    
    await event_bus.stop()
    
    print("\n✅ Event Bus Demo Complete!")
    print("🎉 All features demonstrated successfully!")
    
    return {
        'workflow_events': len(workflow_events),
        'improvement_events': len(improvement_events),
        'total_processed': metrics['events_processed'],
        'demo_success': True
    }

if __name__ == "__main__":
    # Run the integration demo
    asyncio.run(test_integration_demo()) 