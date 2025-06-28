"""
Event Bus System for AI Agent Platform

This module implements a comprehensive event-driven architecture that enables
loose coupling between agents and components through a publish-subscribe pattern.

Features:
- Publish-subscribe pattern for agent communication
- Async event processing with queues
- Event persistence for replay and analysis
- Rate limiting to prevent event storms
- Event routing without direct agent dependencies
- Real-time event monitoring and metrics

Created: June 2025
Updated: June 2025
"""

import asyncio
import logging
import uuid
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

from storage.supabase import SupabaseLogger

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Standard event types for the AI Agent Platform."""
    
    # Workflow events
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed" 
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_PAUSED = "workflow.paused"
    WORKFLOW_RESUMED = "workflow.resumed"
    
    # Agent events
    AGENT_SPAWNED = "agent.spawned"
    AGENT_ACTIVATED = "agent.activated"
    AGENT_DEACTIVATED = "agent.deactivated"
    AGENT_ERROR = "agent.error"
    AGENT_OPTIMIZED = "agent.optimized"
    
    # Improvement events
    PATTERN_DISCOVERED = "improvement.pattern_discovered"
    IMPROVEMENT_APPLIED = "improvement.applied"
    OPTIMIZATION_COMPLETED = "improvement.optimization_completed"
    KNOWLEDGE_UPDATED = "improvement.knowledge_updated"
    
    # User events
    FEEDBACK_RECEIVED = "user.feedback_received"
    WORKFLOW_SAVED = "user.workflow_saved"
    COMMAND_EXECUTED = "user.command_executed"
    
    # System events
    COST_THRESHOLD_REACHED = "system.cost_threshold_reached"
    PERFORMANCE_ALERT = "system.performance_alert"
    ERROR_RECOVERY = "system.error_recovery"
    HEALTH_CHECK = "system.health_check"
    
    # Custom event type for extensibility
    CUSTOM = "custom"

@dataclass
class Event:
    """Event data structure."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: Union[EventType, str] = EventType.CUSTOM
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 5  # 1 (highest) to 10 (lowest)
    ttl: Optional[int] = None  # Time to live in seconds
    tags: Set[str] = field(default_factory=set)
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def __post_init__(self):
        """Convert EventType enum to string if needed."""
        if isinstance(self.type, EventType):
            self.type = self.type.value
    
    def is_expired(self) -> bool:
        """Check if event has expired based on TTL."""
        if self.ttl is None:
            return False
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds() > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'id': self.id,
            'type': self.type,
            'data': self.data,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority,
            'ttl': self.ttl,
            'tags': list(self.tags),
            'correlation_id': self.correlation_id,
            'user_id': self.user_id
        }

EventHandler = Callable[[Event], None]

@dataclass
class Subscription:
    """Event subscription data structure."""
    
    subscriber_id: str
    event_types: Set[str]
    handler: EventHandler
    filter_func: Optional[Callable[[Event], bool]] = None
    priority: int = 5
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_event_at: Optional[datetime] = None
    event_count: int = 0

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    
    max_events_per_second: int = 100
    max_events_per_minute: int = 1000
    max_events_per_hour: int = 10000
    burst_limit: int = 50
    enable_rate_limiting: bool = True

class EventBus:
    """
    Comprehensive Event Bus System for AI Agent Platform.
    
    Provides publish-subscribe pattern for loose coupling between agents,
    with async processing, persistence, and rate limiting.
    """
    
    def __init__(self, 
                 max_queue_size: int = 10000,
                 enable_persistence: bool = True,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        """
        Initialize the Event Bus.
        
        Args:
            max_queue_size: Maximum size of event queue
            enable_persistence: Whether to persist events to database
            rate_limit_config: Rate limiting configuration
        """
        self.max_queue_size = max_queue_size
        self.enable_persistence = enable_persistence
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        
        # Core event processing
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.priority_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.dead_letter_queue: List[Event] = []
        
        # Subscriptions management
        self.subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self.subscriber_registry: Dict[str, Subscription] = {}
        
        # Rate limiting
        self.event_timestamps: deque = deque()
        self.event_counts: Dict[str, deque] = defaultdict(deque)
        
        # Event processing state
        self.is_running = False
        self.processor_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Metrics and monitoring
        self.events_published: int = 0
        self.events_processed: int = 0
        self.events_failed: int = 0
        self.events_rate_limited: int = 0
        self.last_health_check: datetime = datetime.now(timezone.utc)
        
        # Database integration
        self.supabase_logger: Optional[SupabaseLogger] = None
        if enable_persistence:
            try:
                self.supabase_logger = SupabaseLogger()
                logger.info("Event Bus initialized with Supabase persistence")
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase persistence: {e}")
                self.enable_persistence = False
        
        logger.info(f"Event Bus initialized (queue_size={max_queue_size}, persistence={enable_persistence})")
    
    async def start(self):
        """Start the event bus processing."""
        if self.is_running:
            logger.warning("Event Bus is already running")
            return
        
        self.is_running = True
        self.processor_task = asyncio.create_task(self._process_events())
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_events())
        
        logger.info("Event Bus started")
    
    async def stop(self):
        """Stop the event bus processing."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel processing tasks
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining events
        await self._drain_queue()
        
        logger.info("Event Bus stopped")
    
    async def publish(self, 
                     event_type: Union[EventType, str],
                     data: Dict[str, Any],
                     source: Optional[str] = None,
                     priority: int = 5,
                     ttl: Optional[int] = None,
                     tags: Optional[Set[str]] = None,
                     correlation_id: Optional[str] = None,
                     user_id: Optional[str] = None) -> str:
        """
        Publish an event to the bus.
        
        Args:
            event_type: Type of event
            data: Event data
            source: Source component/agent
            priority: Event priority (1-10)
            ttl: Time to live in seconds
            tags: Event tags for filtering
            correlation_id: Correlation ID for related events
            user_id: User ID if user-related
            
        Returns:
            Event ID
        """
        # Rate limiting check
        if not self._check_rate_limit(source):
            self.events_rate_limited += 1
            logger.warning(f"Rate limit exceeded for source: {source}")
            raise Exception(f"Rate limit exceeded for source: {source}")
        
        # Create event
        event = Event(
            type=event_type,
            data=data,
            source=source,
            priority=priority,
            ttl=ttl,
            tags=tags or set(),
            correlation_id=correlation_id,
            user_id=user_id
        )
        
        try:
            # Add to priority queue (lower number = higher priority)
            await self.priority_queue.put((priority, time.time(), event))
            
            self.events_published += 1
            
            # Persist important events
            if self.enable_persistence and self._should_persist_event(event):
                await self._persist_event(event)
            
            logger.debug(f"Published event: {event.type} from {source}")
            return event.id
            
        except asyncio.QueueFull:
            logger.error("Event queue is full, adding to dead letter queue")
            self.dead_letter_queue.append(event)
            raise Exception("Event queue is full")
    
    async def subscribe(self,
                       subscriber_id: str,
                       event_types: Union[List[str], List[EventType], str, EventType],
                       handler: EventHandler,
                       filter_func: Optional[Callable[[Event], bool]] = None,
                       priority: int = 5) -> bool:
        """
        Subscribe to event types.
        
        Args:
            subscriber_id: Unique subscriber identifier
            event_types: Event types to subscribe to
            handler: Event handler function
            filter_func: Optional filter function
            priority: Subscription priority
            
        Returns:
            True if subscription successful
        """
        try:
            # Normalize event types
            if isinstance(event_types, (str, EventType)):
                event_types = [event_types]
            
            normalized_types = set()
            for event_type in event_types:
                if isinstance(event_type, EventType):
                    normalized_types.add(event_type.value)
                else:
                    normalized_types.add(event_type)
            
            # Create subscription
            subscription = Subscription(
                subscriber_id=subscriber_id,
                event_types=normalized_types,
                handler=handler,
                filter_func=filter_func,
                priority=priority
            )
            
            # Register subscription
            self.subscriber_registry[subscriber_id] = subscription
            
            # Add to event type mappings
            for event_type in normalized_types:
                self.subscriptions[event_type].append(subscription)
                # Sort by priority
                self.subscriptions[event_type].sort(key=lambda s: s.priority)
            
            logger.info(f"Subscribed {subscriber_id} to {len(normalized_types)} event types")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe {subscriber_id}: {e}")
            return False
    
    async def unsubscribe(self, subscriber_id: str) -> bool:
        """
        Unsubscribe from all events.
        
        Args:
            subscriber_id: Subscriber to unsubscribe
            
        Returns:
            True if unsubscription successful
        """
        try:
            if subscriber_id not in self.subscriber_registry:
                logger.warning(f"Subscriber {subscriber_id} not found")
                return False
            
            subscription = self.subscriber_registry[subscriber_id]
            
            # Remove from event type mappings
            for event_type in subscription.event_types:
                if event_type in self.subscriptions:
                    self.subscriptions[event_type] = [
                        s for s in self.subscriptions[event_type] 
                        if s.subscriber_id != subscriber_id
                    ]
            
            # Remove from registry
            del self.subscriber_registry[subscriber_id]
            
            logger.info(f"Unsubscribed {subscriber_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe {subscriber_id}: {e}")
            return False
    
    async def get_subscription_info(self, subscriber_id: str) -> Optional[Dict[str, Any]]:
        """Get subscription information for a subscriber."""
        if subscriber_id not in self.subscriber_registry:
            return None
        
        subscription = self.subscriber_registry[subscriber_id]
        return {
            'subscriber_id': subscription.subscriber_id,
            'event_types': list(subscription.event_types),
            'priority': subscription.priority,
            'active': subscription.active,
            'created_at': subscription.created_at.isoformat(),
            'last_event_at': subscription.last_event_at.isoformat() if subscription.last_event_at else None,
            'event_count': subscription.event_count
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics."""
        return {
            'events_published': self.events_published,
            'events_processed': self.events_processed,
            'events_failed': self.events_failed,
            'events_rate_limited': self.events_rate_limited,
            'queue_size': self.event_queue.qsize(),
            'priority_queue_size': self.priority_queue.qsize(),
            'dead_letter_queue_size': len(self.dead_letter_queue),
            'active_subscriptions': len(self.subscriber_registry),
            'subscription_types': len(self.subscriptions),
            'is_running': self.is_running,
            'last_health_check': self.last_health_check.isoformat()
        }
    
    async def replay_events(self, 
                           event_types: Optional[List[str]] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           correlation_id: Optional[str] = None) -> List[Event]:
        """
        Replay events from persistence storage.
        
        Args:
            event_types: Filter by event types
            start_time: Start time filter
            end_time: End time filter
            correlation_id: Filter by correlation ID
            
        Returns:
            List of replayed events
        """
        if not self.enable_persistence or not self.supabase_logger:
            logger.warning("Event persistence not enabled, cannot replay")
            return []
        
        try:
            # This would query the events table in Supabase
            # For now, return empty list as the table structure would need to be defined
            logger.info("Event replay requested (implementation pending)")
            return []
            
        except Exception as e:
            logger.error(f"Failed to replay events: {e}")
            return []
    
    def _check_rate_limit(self, source: Optional[str]) -> bool:
        """Check if event is within rate limits."""
        if not self.rate_limit_config.enable_rate_limiting:
            return True
        
        now = time.time()
        
        # Clean old timestamps
        cutoff_time = now - 60  # Keep last minute
        while self.event_timestamps and self.event_timestamps[0] < cutoff_time:
            self.event_timestamps.popleft()
        
        # Check global rate limits
        if len(self.event_timestamps) >= self.rate_limit_config.max_events_per_minute:
            return False
        
        # Check per-source rate limits if source provided
        if source:
            source_timestamps = self.event_counts[source]
            while source_timestamps and source_timestamps[0] < cutoff_time:
                source_timestamps.popleft()
            
            if len(source_timestamps) >= self.rate_limit_config.burst_limit:
                return False
            
            source_timestamps.append(now)
        
        self.event_timestamps.append(now)
        return True
    
    def _should_persist_event(self, event: Event) -> bool:
        """Determine if event should be persisted."""
        # Persist high priority events and important system events
        if event.priority <= 3:
            return True
        
        important_types = {
            EventType.WORKFLOW_COMPLETED.value,
            EventType.WORKFLOW_FAILED.value,
            EventType.PATTERN_DISCOVERED.value,
            EventType.IMPROVEMENT_APPLIED.value,
            EventType.FEEDBACK_RECEIVED.value,
            EventType.ERROR_RECOVERY.value
        }
        
        return event.type in important_types
    
    async def _persist_event(self, event: Event):
        """Persist event to database."""
        if not self.supabase_logger:
            return
        
        try:
            await self.supabase_logger.log_event(
                event_type=event.type,
                event_data={
                    'event_id': event.id,
                    'data': event.data,
                    'source': event.source,
                    'priority': event.priority,
                    'ttl': event.ttl,
                    'tags': list(event.tags),
                    'correlation_id': event.correlation_id
                },
                user_id=event.user_id
            )
            
        except Exception as e:
            logger.error(f"Failed to persist event {event.id}: {e}")
    
    async def _process_events(self):
        """Main event processing loop."""
        logger.info("Event processing started")
        
        while self.is_running:
            try:
                # Get next event from priority queue with timeout
                try:
                    priority, timestamp, event = await asyncio.wait_for(
                        self.priority_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check if event is expired
                if event.is_expired():
                    logger.debug(f"Discarding expired event: {event.id}")
                    continue
                
                # Process event
                await self._handle_event(event)
                self.events_processed += 1
                
            except Exception as e:
                self.events_failed += 1
                logger.error(f"Error processing event: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error
    
    async def _handle_event(self, event: Event):
        """Handle a single event by delivering to subscribers."""
        subscribers = self.subscriptions.get(event.type, [])
        
        if not subscribers:
            logger.debug(f"No subscribers for event type: {event.type}")
            return
        
        # Process subscribers in priority order
        for subscription in subscribers:
            if not subscription.active:
                continue
            
            try:
                # Apply filter if present
                if subscription.filter_func and not subscription.filter_func(event):
                    continue
                
                # Call handler
                if asyncio.iscoroutinefunction(subscription.handler):
                    await subscription.handler(event)
                else:
                    subscription.handler(event)
                
                # Update subscription metrics
                subscription.last_event_at = datetime.now(timezone.utc)
                subscription.event_count += 1
                
            except Exception as e:
                logger.error(f"Error in event handler for {subscription.subscriber_id}: {e}")
    
    async def _cleanup_expired_events(self):
        """Periodic cleanup of expired events."""
        while self.is_running:
            try:
                # Clean dead letter queue
                now = datetime.now(timezone.utc)
                self.dead_letter_queue = [
                    event for event in self.dead_letter_queue 
                    if not event.is_expired()
                ]
                
                # Update health check
                self.last_health_check = now
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    async def _drain_queue(self):
        """Process remaining events in queue during shutdown."""
        logger.info("Draining event queue...")
        
        processed = 0
        while not self.priority_queue.empty():
            try:
                priority, timestamp, event = self.priority_queue.get_nowait()
                if not event.is_expired():
                    await self._handle_event(event)
                    processed += 1
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error(f"Error draining queue: {e}")
        
        logger.info(f"Drained {processed} events from queue")

# Global event bus instance
_global_event_bus: Optional[EventBus] = None

def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus

async def init_event_bus(**kwargs) -> EventBus:
    """Initialize and start the global event bus."""
    global _global_event_bus
    _global_event_bus = EventBus(**kwargs)
    await _global_event_bus.start()
    return _global_event_bus

async def shutdown_event_bus():
    """Shutdown the global event bus."""
    global _global_event_bus
    if _global_event_bus:
        await _global_event_bus.stop()
        _global_event_bus = None 