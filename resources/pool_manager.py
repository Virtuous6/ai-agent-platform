"""
Resource Pool Manager for Virtuous6 AI Platform

Manages shared resources efficiently across all agents:
- LLM connection pools (max 10 concurrent)
- Tool instance sharing across agents
- Database connection pooling
- Vector memory allocation
- Fair scheduling with timeout protection
- Comprehensive resource health monitoring
"""

import asyncio
import time
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import weakref
import psutil
import threading
from contextlib import asynccontextmanager

# Import existing components
from langchain_openai import ChatOpenAI
from database.supabase_logger import SupabaseLogger
from events.event_bus import get_event_bus

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Types of resources managed by the pool."""
    LLM_CONNECTION = "llm_connection"
    TOOL_INSTANCE = "tool_instance"  
    DATABASE_CONNECTION = "database_connection"
    VECTOR_MEMORY = "vector_memory"
    COMPUTE_SLOT = "compute_slot"

class ResourcePriority(Enum):
    """Priority levels for resource allocation."""
    CRITICAL = 1    # System-critical operations
    HIGH = 2        # User-facing operations
    NORMAL = 3      # Background processing
    LOW = 4         # Cleanup and maintenance

@dataclass
class ResourceRequest:
    """Resource allocation request."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: ResourceType = ResourceType.LLM_CONNECTION
    requester_id: str = ""
    priority: ResourcePriority = ResourcePriority.NORMAL
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    allocated_at: Optional[datetime] = None
    released_at: Optional[datetime] = None

@dataclass
class ResourceUsageStats:
    """Resource usage statistics."""
    total_requests: int = 0
    successful_allocations: int = 0
    timeout_failures: int = 0
    current_active: int = 0
    average_hold_time: float = 0.0
    peak_concurrent: int = 0
    last_allocation: Optional[datetime] = None

class ResourcePool:
    """Base class for resource pools."""
    
    def __init__(self, resource_type: ResourceType, max_size: int, 
                 health_check_interval: float = 60.0):
        self.resource_type = resource_type
        self.max_size = max_size
        self.health_check_interval = health_check_interval
        
        # Core synchronization
        self._semaphore = asyncio.Semaphore(max_size)
        self._allocation_lock = asyncio.Lock()
        
        # Resource tracking
        self._available_resources: deque = deque()
        self._allocated_resources: Dict[str, Any] = {}
        self._allocation_times: Dict[str, datetime] = {}
        self._request_queue: List[ResourceRequest] = []
        
        # Statistics
        self.stats = ResourceUsageStats()
        self._resource_creators: Dict[str, Callable] = {}
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._healthy_resources: set = set()
        
    async def start(self):
        """Start the resource pool."""
        logger.info(f"Starting {self.resource_type.value} pool with max_size={self.max_size}")
        
        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_monitor())
        
        # Publish pool started event
        event_bus = get_event_bus()
        await event_bus.publish(
            "resource_pool_started",
            {
                "resource_type": self.resource_type.value,
                "max_size": self.max_size,
                "timestamp": datetime.utcnow().isoformat()
            },
            source="resource_pool_manager"
        )
    
    async def stop(self):
        """Stop the resource pool and cleanup."""
        logger.info(f"Stopping {self.resource_type.value} pool")
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup resources
        async with self._allocation_lock:
            for resource in self._available_resources:
                await self._cleanup_resource(resource)
            self._available_resources.clear()
            
            for resource in self._allocated_resources.values():
                await self._cleanup_resource(resource)
            self._allocated_resources.clear()
    
    async def allocate(self, request: ResourceRequest) -> Optional[Any]:
        """Allocate a resource from the pool."""
        start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            # Wait for available slot with timeout
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=request.timeout_seconds
            )
            
            async with self._allocation_lock:
                # Try to get existing resource
                resource = await self._get_or_create_resource(request)
                
                if resource:
                    # Track allocation
                    allocation_id = str(uuid.uuid4())
                    self._allocated_resources[allocation_id] = resource
                    self._allocation_times[allocation_id] = datetime.utcnow()
                    
                    request.allocated_at = datetime.utcnow()
                    self.stats.successful_allocations += 1
                    self.stats.current_active += 1
                    self.stats.last_allocation = request.allocated_at
                    
                    if self.stats.current_active > self.stats.peak_concurrent:
                        self.stats.peak_concurrent = self.stats.current_active
                    
                    # Publish allocation event
                    event_bus = get_event_bus()
                    await event_bus.publish(
                        "resource_allocated",
                        {
                            "resource_type": self.resource_type.value,
                            "allocation_id": allocation_id,
                            "requester_id": request.requester_id,
                            "priority": request.priority.value,
                            "wait_time": time.time() - start_time
                        },
                        source="resource_pool_manager"
                    )
                    
                    return (allocation_id, resource)
                else:
                    self._semaphore.release()
                    return None
                    
        except asyncio.TimeoutError:
            self.stats.timeout_failures += 1
            logger.warning(
                f"Resource allocation timeout for {request.requester_id} "
                f"after {request.timeout_seconds}s"
            )
            
            event_bus = get_event_bus()
            await event_bus.publish(
                "resource_allocation_timeout",
                {
                    "resource_type": self.resource_type.value,
                    "requester_id": request.requester_id,
                    "timeout_seconds": request.timeout_seconds
                },
                source="resource_pool_manager"
            )
            
            return None
    
    async def release(self, allocation_id: str):
        """Release a resource back to the pool."""
        async with self._allocation_lock:
            if allocation_id in self._allocated_resources:
                resource = self._allocated_resources.pop(allocation_id)
                allocation_time = self._allocation_times.pop(allocation_id, datetime.utcnow())
                
                # Update statistics
                hold_time = (datetime.utcnow() - allocation_time).total_seconds()
                total_time = self.stats.average_hold_time * self.stats.successful_allocations
                self.stats.average_hold_time = (total_time + hold_time) / max(self.stats.successful_allocations, 1)
                self.stats.current_active -= 1
                
                # Check if resource is still healthy
                if await self._is_resource_healthy(resource):
                    self._available_resources.append(resource)
                else:
                    await self._cleanup_resource(resource)
                
                self._semaphore.release()
                
                # Publish release event
                event_bus = get_event_bus()
                await event_bus.publish(
                    "resource_released",
                    {
                        "resource_type": self.resource_type.value,
                        "allocation_id": allocation_id,
                        "hold_time": hold_time
                    },
                    source="resource_pool_manager"
                )
    
    async def _get_or_create_resource(self, request: ResourceRequest) -> Optional[Any]:
        """Get existing resource or create new one."""
        # Try to reuse existing resource
        if self._available_resources:
            resource = self._available_resources.popleft()
            if await self._is_resource_healthy(resource):
                return resource
            else:
                await self._cleanup_resource(resource)
        
        # Create new resource
        return await self._create_resource(request)
    
    async def _create_resource(self, request: ResourceRequest) -> Optional[Any]:
        """Create a new resource. Override in subclasses."""
        creator = self._resource_creators.get(request.metadata.get('creator_type', 'default'))
        if creator:
            return await creator(request)
        return None
    
    async def _is_resource_healthy(self, resource: Any) -> bool:
        """Check if resource is healthy. Override in subclasses."""
        return True
    
    async def _cleanup_resource(self, resource: Any):
        """Cleanup resource. Override in subclasses."""
        pass
    
    async def _health_monitor(self):
        """Monitor pool health."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for {self.resource_type.value}: {e}")
    
    async def _perform_health_check(self):
        """Perform health check on pool."""
        async with self._allocation_lock:
            healthy_count = 0
            total_count = len(self._available_resources)
            
            # Check available resources
            unhealthy_resources = []
            for resource in list(self._available_resources):
                if await self._is_resource_healthy(resource):
                    healthy_count += 1
                else:
                    unhealthy_resources.append(resource)
            
            # Remove unhealthy resources
            for resource in unhealthy_resources:
                self._available_resources.remove(resource)
                await self._cleanup_resource(resource)
            
            health_ratio = healthy_count / max(total_count, 1)
            
            # Publish health status
            event_bus = get_event_bus()
            await event_bus.publish(
                "resource_pool_health_check",
                {
                    "resource_type": self.resource_type.value,
                    "healthy_count": healthy_count,
                    "total_count": total_count,
                    "health_ratio": health_ratio,
                    "active_allocations": self.stats.current_active
                },
                source="resource_pool_manager"
            )

class LLMConnectionPool(ResourcePool):
    """Pool for LLM connections."""
    
    def __init__(self, max_size: int = 10):
        super().__init__(ResourceType.LLM_CONNECTION, max_size)
        
        # Register LLM creators
        self._resource_creators = {
            'default': self._create_default_llm,
            'gpt-4': self._create_gpt4_llm,
            'gpt-3.5-turbo': self._create_gpt35_llm
        }
    
    async def _create_default_llm(self, request: ResourceRequest) -> ChatOpenAI:
        """Create default LLM connection."""
        return ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0.4,
            max_tokens=500,
            request_timeout=30
        )
    
    async def _create_gpt4_llm(self, request: ResourceRequest) -> ChatOpenAI:
        """Create GPT-4 LLM connection."""
        return ChatOpenAI(
            model="gpt-4-0125-preview",
            temperature=request.metadata.get('temperature', 0.3),
            max_tokens=request.metadata.get('max_tokens', 800),
            request_timeout=30
        )
    
    async def _create_gpt35_llm(self, request: ResourceRequest) -> ChatOpenAI:
        """Create GPT-3.5 LLM connection."""
        return ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=request.metadata.get('temperature', 0.4),
            max_tokens=request.metadata.get('max_tokens', 500),
            request_timeout=30
        )
    
    async def _is_resource_healthy(self, resource: ChatOpenAI) -> bool:
        """Check if LLM connection is healthy."""
        try:
            # Simple health check with minimal token usage
            test_response = await resource.agenerate([["Test connection"]])
            return test_response is not None
        except Exception:
            return False

class ToolInstancePool(ResourcePool):
    """Pool for shared tool instances."""
    
    def __init__(self, max_size: int = 50):
        super().__init__(ResourceType.TOOL_INSTANCE, max_size)
        self._tool_registry: Dict[str, Any] = {}
    
    def register_tool(self, tool_name: str, tool_instance: Any):
        """Register a tool instance for sharing."""
        self._tool_registry[tool_name] = tool_instance
    
    async def _create_resource(self, request: ResourceRequest) -> Optional[Any]:
        """Get shared tool instance."""
        tool_name = request.metadata.get('tool_name')
        return self._tool_registry.get(tool_name)

class DatabaseConnectionPool(ResourcePool):
    """Pool for database connections."""
    
    def __init__(self, max_size: int = 20):
        super().__init__(ResourceType.DATABASE_CONNECTION, max_size)
    
    async def _create_resource(self, request: ResourceRequest) -> Optional[Any]:
        """Create database connection."""
        # This would create actual database connections
        # For now, return a mock connection
        return SupabaseLogger()
    
    async def _is_resource_healthy(self, resource: Any) -> bool:
        """Check if database connection is healthy."""
        try:
            # Test database connection
            return hasattr(resource, 'log_event')
        except Exception:
            return False

class VectorMemoryPool(ResourcePool):
    """Pool for vector memory allocation."""
    
    def __init__(self, max_size_mb: int = 1000):
        super().__init__(ResourceType.VECTOR_MEMORY, max_size_mb)
        self._memory_usage: Dict[str, int] = {}
        self._max_memory_mb = max_size_mb
        self._current_usage_mb = 0
    
    async def allocate(self, request: ResourceRequest) -> Optional[Any]:
        """Allocate vector memory."""
        requested_mb = request.metadata.get('size_mb', 50)
        
        async with self._allocation_lock:
            if self._current_usage_mb + requested_mb > self._max_memory_mb:
                return None
            
            allocation_id = str(uuid.uuid4())
            self._memory_usage[allocation_id] = requested_mb
            self._current_usage_mb += requested_mb
            
            return (allocation_id, {'size_mb': requested_mb, 'allocation_id': allocation_id})
    
    async def release(self, allocation_id: str):
        """Release vector memory."""
        async with self._allocation_lock:
            if allocation_id in self._memory_usage:
                released_mb = self._memory_usage.pop(allocation_id)
                self._current_usage_mb -= released_mb

class ResourcePoolManager:
    """Main resource pool manager coordinating all resource types."""
    
    def __init__(self):
        self.pools: Dict[ResourceType, ResourcePool] = {}
        self._fair_scheduler = FairScheduler()
        self._monitor_task: Optional[asyncio.Task] = None
        self._started = False
        
        # Initialize pools
        self.pools[ResourceType.LLM_CONNECTION] = LLMConnectionPool(max_size=10)
        self.pools[ResourceType.TOOL_INSTANCE] = ToolInstancePool(max_size=50)
        self.pools[ResourceType.DATABASE_CONNECTION] = DatabaseConnectionPool(max_size=20)
        self.pools[ResourceType.VECTOR_MEMORY] = VectorMemoryPool(max_size_mb=1000)
    
    async def start(self):
        """Start all resource pools."""
        if self._started:
            return
            
        logger.info("Starting Resource Pool Manager")
        
        # Start all pools
        start_tasks = []
        for pool in self.pools.values():
            start_tasks.append(pool.start())
        
        await asyncio.gather(*start_tasks)
        
        # Start monitoring
        self._monitor_task = asyncio.create_task(self._system_monitor())
        self._started = True
        
        event_bus = get_event_bus()
        await event_bus.publish(
            "resource_pool_manager_started",
            {
                "pools": list(self.pools.keys()),
                "timestamp": datetime.utcnow().isoformat()
            },
            source="resource_pool_manager"
        )
    
    async def stop(self):
        """Stop all resource pools."""
        if not self._started:
            return
            
        logger.info("Stopping Resource Pool Manager")
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop all pools
        stop_tasks = []
        for pool in self.pools.values():
            stop_tasks.append(pool.stop())
        
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        self._started = False
    
    @asynccontextmanager
    async def allocate_resource(self, resource_type: ResourceType, 
                               requester_id: str,
                               priority: ResourcePriority = ResourcePriority.NORMAL,
                               timeout_seconds: float = 30.0,
                               **metadata):
        """Context manager for resource allocation."""
        request = ResourceRequest(
            resource_type=resource_type,
            requester_id=requester_id,
            priority=priority,
            timeout_seconds=timeout_seconds,
            metadata=metadata
        )
        
        # Apply fair scheduling
        await self._fair_scheduler.schedule_request(request)
        
        pool = self.pools.get(resource_type)
        if not pool:
            raise ValueError(f"No pool available for resource type: {resource_type}")
        
        allocation_result = await pool.allocate(request)
        if not allocation_result:
            raise ResourceException(f"Failed to allocate {resource_type.value}")
        
        allocation_id, resource = allocation_result
        
        try:
            yield resource
        finally:
            await pool.release(allocation_id)
    
    async def get_pool_stats(self, resource_type: ResourceType) -> Optional[ResourceUsageStats]:
        """Get statistics for a resource pool."""
        pool = self.pools.get(resource_type)
        return pool.stats if pool else None
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        health_data = {
            "overall_health": "healthy",
            "pools": {},
            "system_resources": await self._get_system_resources(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        unhealthy_pools = 0
        total_pools = len(self.pools)
        
        for resource_type, pool in self.pools.items():
            pool_health = {
                "stats": pool.stats.__dict__,
                "active_allocations": pool.stats.current_active,
                "max_capacity": pool.max_size,
                "utilization": pool.stats.current_active / pool.max_size
            }
            
            # Determine pool health status
            if pool.stats.timeout_failures > pool.stats.successful_allocations * 0.1:
                pool_health["status"] = "degraded"
                unhealthy_pools += 1
            elif pool.stats.current_active / pool.max_size > 0.9:
                pool_health["status"] = "high_utilization"
            else:
                pool_health["status"] = "healthy"
            
            health_data["pools"][resource_type.value] = pool_health
        
        # Overall health determination
        if unhealthy_pools > total_pools * 0.3:
            health_data["overall_health"] = "critical"
        elif unhealthy_pools > 0:
            health_data["overall_health"] = "degraded"
        
        return health_data
    
    async def _get_system_resources(self) -> Dict[str, Any]:
        """Get system resource information."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "process_count": len(psutil.pids())
        }
    
    async def _system_monitor(self):
        """Monitor overall system health."""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                health = await self.get_system_health()
                
                # Publish health status
                event_bus = get_event_bus()
                await event_bus.publish(
                    "resource_manager_health_status",
                    health,
                    source="resource_pool_manager"
                )
                
                # Auto-optimization based on health
                if health["overall_health"] == "critical":
                    await self._emergency_optimization()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System monitor error: {e}")
    
    async def _emergency_optimization(self):
        """Emergency optimization when system health is critical."""
        logger.warning("Triggering emergency resource optimization")
        
        # Release stale allocations
        for pool in self.pools.values():
            await self._cleanup_stale_allocations(pool)
        
        event_bus = get_event_bus()
        await event_bus.publish(
            "emergency_optimization_triggered",
            {"timestamp": datetime.utcnow().isoformat()},
            source="resource_pool_manager"
        )
    
    async def _cleanup_stale_allocations(self, pool: ResourcePool):
        """Cleanup stale resource allocations."""
        current_time = datetime.utcnow()
        stale_threshold = timedelta(minutes=30)
        
        async with pool._allocation_lock:
            stale_allocations = []
            for allocation_id, allocation_time in pool._allocation_times.items():
                if current_time - allocation_time > stale_threshold:
                    stale_allocations.append(allocation_id)
            
            for allocation_id in stale_allocations:
                logger.warning(f"Cleaning up stale allocation: {allocation_id}")
                await pool.release(allocation_id)

class FairScheduler:
    """Fair scheduling algorithm for resource requests."""
    
    def __init__(self):
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._last_allocation: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
    
    async def schedule_request(self, request: ResourceRequest):
        """Apply fair scheduling to resource request."""
        async with self._lock:
            requester_id = request.requester_id
            
            # Track request counts
            self._request_counts[requester_id] += 1
            
            # Calculate delay based on fairness
            current_time = datetime.utcnow()
            last_alloc = self._last_allocation.get(requester_id)
            
            if last_alloc:
                time_since_last = (current_time - last_alloc).total_seconds()
                request_count = self._request_counts[requester_id]
                
                # Apply back-off for frequent requesters
                if time_since_last < 1.0 and request_count > 5:
                    delay = min(request_count * 0.1, 2.0)  # Max 2 second delay
                    await asyncio.sleep(delay)
            
            self._last_allocation[requester_id] = current_time

class ResourceException(Exception):
    """Exception raised for resource allocation failures."""
    pass

# Global resource pool manager instance
_resource_pool_manager: Optional[ResourcePoolManager] = None

async def get_resource_pool_manager() -> ResourcePoolManager:
    """Get or create global resource pool manager."""
    global _resource_pool_manager
    
    if _resource_pool_manager is None:
        _resource_pool_manager = ResourcePoolManager()
        await _resource_pool_manager.start()
    
    return _resource_pool_manager

async def cleanup_resource_pool_manager():
    """Cleanup global resource pool manager."""
    global _resource_pool_manager
    
    if _resource_pool_manager:
        await _resource_pool_manager.stop()
        _resource_pool_manager = None 