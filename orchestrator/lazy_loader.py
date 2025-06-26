"""
Lazy Agent Loader for AI Agent Platform

Purpose: Efficiently manages agent lifecycle with lazy loading, LRU caching, and intelligent preloading.
Supports 1000+ agent configurations while keeping only 50 active in memory.

Features:
- LRU cache for active agents (max 50)
- Agent configuration serialization/deserialization
- Activity tracking for intelligent preloading
- Weak references for memory management
- Cache hit rate metrics
- Smooth activation/deactivation
"""

import asyncio
import logging
import weakref
import pickle
import json
import hashlib
from typing import Dict, Any, Optional, List, Set, TYPE_CHECKING
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile
import os

# Prevent circular imports
if TYPE_CHECKING:
    from agents.universal_agent import UniversalAgent

logger = logging.getLogger(__name__)

@dataclass
class AgentConfiguration:
    """Serializable agent configuration."""
    agent_id: str
    specialty: str
    system_prompt: str
    temperature: float
    model_name: str
    max_tokens: int
    tools: List[Dict[str, Any]]
    created_at: datetime
    last_used: datetime
    usage_count: int
    success_rate: float
    average_response_time: float
    total_cost: float
    priority_score: float  # For intelligent preloading
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_used'] = self.last_used.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfiguration':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)
    
    def get_cache_key(self) -> str:
        """Generate unique cache key for this configuration."""
        config_str = f"{self.specialty}:{self.system_prompt[:100]}:{self.temperature}:{self.model_name}"
        return hashlib.md5(config_str.encode()).hexdigest()

@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    cache_hits: int = 0
    cache_misses: int = 0
    loads: int = 0
    evictions: int = 0
    serializations: int = 0
    deserializations: int = 0
    preload_successes: int = 0
    preload_failures: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def load_efficiency(self) -> float:
        """Calculate load efficiency (successful loads vs attempts)."""
        total_attempts = self.loads + self.preload_failures
        successful = self.loads + self.preload_successes
        return successful / total_attempts if total_attempts > 0 else 0.0

class LazyAgentLoader:
    """
    Lazy loader for AI agents with LRU caching and intelligent preloading.
    
    Features:
    - Maintains LRU cache of active agents (default: 50 max)
    - Serializes inactive agents to disk for memory efficiency
    - Tracks agent activity patterns for intelligent preloading
    - Uses weak references to prevent memory leaks
    - Provides comprehensive cache metrics
    - Handles smooth activation/deactivation
    """
    
    def __init__(self, 
                 max_active_agents: int = 50,
                 max_total_configurations: int = 1000,
                 cache_directory: Optional[str] = None,
                 preload_threshold: float = 0.7,  # Preload agents with priority > 0.7
                 cleanup_interval: int = 3600):   # Cleanup every hour
        """
        Initialize the lazy agent loader.
        
        Args:
            max_active_agents: Maximum number of agents to keep active in memory
            max_total_configurations: Maximum total agent configurations to manage
            cache_directory: Directory for serialized agent cache (temp dir if None)
            preload_threshold: Priority threshold for intelligent preloading
            cleanup_interval: Interval in seconds for cleanup tasks
        """
        self.max_active_agents = max_active_agents
        self.max_total_configurations = max_total_configurations
        self.preload_threshold = preload_threshold
        self.cleanup_interval = cleanup_interval
        
        # LRU cache for active agents using OrderedDict
        self.active_agents: OrderedDict[str, 'UniversalAgent'] = OrderedDict()
        
        # Agent configurations (stored persistently)
        self.agent_configurations: Dict[str, AgentConfiguration] = {}
        
        # Weak references for memory safety
        self.agent_refs: Dict[str, weakref.ReferenceType] = {}
        
        # Activity tracking for intelligent preloading
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.usage_frequencies: Dict[str, float] = {}
        self.co_occurrence_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Cache directory setup
        if cache_directory:
            self.cache_directory = Path(cache_directory)
        else:
            self.cache_directory = Path(tempfile.gettempdir()) / "ai_agent_cache"
        
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.metrics = CacheMetrics()
        
        # Background tasks
        self.cleanup_task = None
        self.preload_task = None
        
        # Thread safety
        self.lock = asyncio.Lock()
        
        # Initialize system
        self._start_background_tasks()
        self._load_configurations_from_disk()
        
        logger.info(f"LazyAgentLoader initialized: max_active={max_active_agents}, "
                   f"max_total={max_total_configurations}, cache_dir={self.cache_directory}")
    
    async def get_agent(self, agent_id: str) -> Optional['UniversalAgent']:
        """
        Get agent instance, loading from cache/disk if necessary.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Agent instance if found, None otherwise
        """
        async with self.lock:
            try:
                # Check if agent is already active (cache hit)
                if agent_id in self.active_agents:
                    agent = self.active_agents.pop(agent_id)
                    self.active_agents[agent_id] = agent  # Move to end (most recent)
                    self.metrics.cache_hits += 1
                    self._record_access(agent_id)
                    logger.debug(f"Cache hit for agent: {agent_id}")
                    return agent
                
                # Cache miss - need to load agent
                self.metrics.cache_misses += 1
                
                # Check if configuration exists
                if agent_id not in self.agent_configurations:
                    logger.warning(f"Agent configuration not found: {agent_id}")
                    return None
                
                # Load agent from configuration
                agent = await self._load_agent_from_config(agent_id)
                if agent:
                    await self._add_to_active_cache(agent_id, agent)
                    self._record_access(agent_id)
                    logger.debug(f"Loaded agent from configuration: {agent_id}")
                
                return agent
                
            except Exception as e:
                logger.error(f"Error getting agent {agent_id}: {e}")
                return None
    
    async def add_agent_configuration(self, config: AgentConfiguration) -> bool:
        """
        Add a new agent configuration to the system.
        
        Args:
            config: Agent configuration to add
            
        Returns:
            True if successful, False otherwise
        """
        async with self.lock:
            try:
                # Check total configuration limit
                if len(self.agent_configurations) >= self.max_total_configurations:
                    # Remove least used configuration
                    await self._evict_least_used_configuration()
                
                # Add configuration
                self.agent_configurations[config.agent_id] = config
                
                # Save to disk
                await self._save_configuration_to_disk(config)
                
                # Initialize tracking
                self.usage_frequencies[config.agent_id] = 0.0
                
                logger.info(f"Added agent configuration: {config.agent_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error adding agent configuration {config.agent_id}: {e}")
                return False
    
    async def remove_agent(self, agent_id: str) -> bool:
        """
        Remove agent from all caches and configurations.
        
        Args:
            agent_id: Agent to remove
            
        Returns:
            True if successful, False otherwise
        """
        async with self.lock:
            try:
                removed = False
                
                # Remove from active cache
                if agent_id in self.active_agents:
                    agent = self.active_agents.pop(agent_id)
                    await agent.close()
                    removed = True
                
                # Remove configuration
                if agent_id in self.agent_configurations:
                    del self.agent_configurations[agent_id]
                    removed = True
                
                # Remove from disk cache
                cache_file = self.cache_directory / f"{agent_id}.cache"
                if cache_file.exists():
                    cache_file.unlink()
                    removed = True
                
                # Clean up tracking data
                if agent_id in self.access_patterns:
                    del self.access_patterns[agent_id]
                if agent_id in self.usage_frequencies:
                    del self.usage_frequencies[agent_id]
                if agent_id in self.agent_refs:
                    del self.agent_refs[agent_id]
                
                # Clean up co-occurrence data
                if agent_id in self.co_occurrence_matrix:
                    del self.co_occurrence_matrix[agent_id]
                for other_agent in self.co_occurrence_matrix:
                    if agent_id in self.co_occurrence_matrix[other_agent]:
                        del self.co_occurrence_matrix[other_agent][agent_id]
                
                if removed:
                    logger.info(f"Removed agent: {agent_id}")
                
                return removed
                
            except Exception as e:
                logger.error(f"Error removing agent {agent_id}: {e}")
                return False
    
    async def preload_agents(self, agent_ids: Optional[List[str]] = None) -> int:
        """
        Intelligently preload agents based on usage patterns.
        
        Args:
            agent_ids: Specific agents to preload (None for intelligent selection)
            
        Returns:
            Number of agents successfully preloaded
        """
        async with self.lock:
            try:
                if agent_ids is None:
                    # Intelligent preloading based on priority scores
                    agent_ids = self._select_agents_for_preloading()
                
                preloaded = 0
                available_slots = self.max_active_agents - len(self.active_agents)
                
                for agent_id in agent_ids[:available_slots]:
                    if agent_id not in self.active_agents:
                        agent = await self._load_agent_from_config(agent_id)
                        if agent:
                            await self._add_to_active_cache(agent_id, agent)
                            preloaded += 1
                            self.metrics.preload_successes += 1
                        else:
                            self.metrics.preload_failures += 1
                
                logger.info(f"Preloaded {preloaded} agents")
                return preloaded
                
            except Exception as e:
                logger.error(f"Error preloading agents: {e}")
                return 0
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache performance metrics."""
        total_accesses = len(self.active_agents) + sum(len(accesses) for accesses in self.access_patterns.values())
        
        return {
            "cache_performance": {
                "hit_rate": self.metrics.hit_rate,
                "load_efficiency": self.metrics.load_efficiency,
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses,
                "total_loads": self.metrics.loads,
                "evictions": self.metrics.evictions
            },
            "agent_status": {
                "active_agents": len(self.active_agents),
                "total_configurations": len(self.agent_configurations),
                "max_active": self.max_active_agents,
                "max_total": self.max_total_configurations,
                "cache_utilization": len(self.active_agents) / self.max_active_agents
            },
            "activity_patterns": {
                "total_accesses": total_accesses,
                "agents_with_activity": len([aid for aid, accesses in self.access_patterns.items() if accesses]),
                "average_usage_frequency": sum(self.usage_frequencies.values()) / len(self.usage_frequencies) if self.usage_frequencies else 0.0
            },
            "preloading": {
                "preload_successes": self.metrics.preload_successes,
                "preload_failures": self.metrics.preload_failures,
                "preload_threshold": self.preload_threshold,
                "candidates_for_preload": len(self._select_agents_for_preloading())
            }
        }
    
    def get_agent_activity_report(self) -> Dict[str, Any]:
        """Get detailed report of agent activity patterns."""
        now = datetime.utcnow()
        
        activity_report = {}
        for agent_id, config in self.agent_configurations.items():
            recent_accesses = [
                access for access in self.access_patterns.get(agent_id, [])
                if (now - access).total_seconds() < 86400  # Last 24 hours
            ]
            
            activity_report[agent_id] = {
                "specialty": config.specialty,
                "total_usage": config.usage_count,
                "success_rate": config.success_rate,
                "last_used": config.last_used.isoformat(),
                "recent_accesses_24h": len(recent_accesses),
                "usage_frequency": self.usage_frequencies.get(agent_id, 0.0),
                "priority_score": config.priority_score,
                "is_active": agent_id in self.active_agents,
                "average_response_time": config.average_response_time,
                "total_cost": config.total_cost
            }
        
        return activity_report
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """
        Perform cache optimization based on usage patterns.
        
        Returns:
            Optimization results and statistics
        """
        async with self.lock:
            try:
                # Update priority scores based on recent activity
                await self._update_priority_scores()
                
                # Identify optimization opportunities
                eviction_candidates = []
                preload_candidates = []
                
                for agent_id, config in self.agent_configurations.items():
                    if agent_id in self.active_agents and config.priority_score < 0.3:
                        eviction_candidates.append(agent_id)
                    elif agent_id not in self.active_agents and config.priority_score > self.preload_threshold:
                        preload_candidates.append(agent_id)
                
                # Perform optimizations
                evicted = 0
                for agent_id in eviction_candidates:
                    if len(self.active_agents) > self.max_active_agents // 2:  # Keep at least half full
                        await self._evict_agent(agent_id)
                        evicted += 1
                
                # Preload high-priority agents
                preloaded = await self.preload_agents(preload_candidates)
                
                optimization_results = {
                    "evicted_agents": evicted,
                    "preloaded_agents": preloaded,
                    "eviction_candidates": len(eviction_candidates),
                    "preload_candidates": len(preload_candidates),
                    "cache_utilization_before": (len(self.active_agents) + evicted) / self.max_active_agents,
                    "cache_utilization_after": len(self.active_agents) / self.max_active_agents
                }
                
                logger.info(f"Cache optimization completed: {optimization_results}")
                return optimization_results
                
            except Exception as e:
                logger.error(f"Error optimizing cache: {e}")
                return {"error": str(e)}
    
    async def close(self):
        """Clean shutdown of the lazy loader."""
        try:
            # Cancel background tasks
            if self.cleanup_task:
                self.cleanup_task.cancel()
            if self.preload_task:
                self.preload_task.cancel()
            
            # Close all active agents
            for agent_id, agent in self.active_agents.items():
                try:
                    await agent.close()
                except Exception as e:
                    logger.warning(f"Error closing agent {agent_id}: {e}")
            
            # Save all configurations to disk
            for config in self.agent_configurations.values():
                await self._save_configuration_to_disk(config)
            
            logger.info("LazyAgentLoader shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during lazy loader shutdown: {e}")
    
    # Private methods
    
    async def _load_agent_from_config(self, agent_id: str) -> Optional['UniversalAgent']:
        """Load agent instance from configuration."""
        try:
            config = self.agent_configurations.get(agent_id)
            if not config:
                return None
            
            # Import here to avoid circular imports
            from agents.universal_agent import UniversalAgent, ToolCapability
            
            # Convert tool configurations to ToolCapability objects
            tools = []
            for tool_config in config.tools:
                tools.append(ToolCapability(**tool_config))
            
            # Create agent instance
            agent = UniversalAgent(
                specialty=config.specialty,
                system_prompt=config.system_prompt,
                temperature=config.temperature,
                tools=tools,
                model_name=config.model_name,
                max_tokens=config.max_tokens,
                agent_id=config.agent_id
            )
            
            # Create weak reference for memory management
            self.agent_refs[agent_id] = weakref.ref(agent, self._agent_cleanup_callback(agent_id))
            
            self.metrics.loads += 1
            logger.debug(f"Loaded agent from config: {agent_id}")
            return agent
            
        except Exception as e:
            logger.error(f"Error loading agent {agent_id} from config: {e}")
            return None
    
    async def _add_to_active_cache(self, agent_id: str, agent: 'UniversalAgent'):
        """Add agent to active cache, managing LRU eviction."""
        # Check if we need to evict
        while len(self.active_agents) >= self.max_active_agents:
            await self._evict_lru_agent()
        
        # Add to cache
        self.active_agents[agent_id] = agent
        
        # Update configuration
        if agent_id in self.agent_configurations:
            self.agent_configurations[agent_id].last_used = datetime.utcnow()
    
    async def _evict_lru_agent(self):
        """Evict least recently used agent from active cache."""
        if not self.active_agents:
            return
        
        # Get least recently used agent (first in OrderedDict)
        lru_agent_id, lru_agent = self.active_agents.popitem(last=False)
        
        try:
            # Serialize agent state if needed
            await self._serialize_agent(lru_agent_id, lru_agent)
            
            # Close agent
            await lru_agent.close()
            
            self.metrics.evictions += 1
            logger.debug(f"Evicted LRU agent: {lru_agent_id}")
            
        except Exception as e:
            logger.error(f"Error evicting agent {lru_agent_id}: {e}")
    
    async def _evict_agent(self, agent_id: str):
        """Evict specific agent from active cache."""
        if agent_id not in self.active_agents:
            return
        
        agent = self.active_agents.pop(agent_id)
        
        try:
            await self._serialize_agent(agent_id, agent)
            await agent.close()
            self.metrics.evictions += 1
            logger.debug(f"Evicted agent: {agent_id}")
        except Exception as e:
            logger.error(f"Error evicting agent {agent_id}: {e}")
    
    async def _serialize_agent(self, agent_id: str, agent: 'UniversalAgent'):
        """Serialize agent state to disk for later restoration."""
        try:
            # Get agent state
            agent_stats = agent.get_agent_stats()
            
            # Update configuration with latest stats
            if agent_id in self.agent_configurations:
                config = self.agent_configurations[agent_id]
                config.usage_count = agent_stats.get('total_interactions', config.usage_count)
                config.success_rate = agent_stats.get('success_rate', config.success_rate)
                config.average_response_time = agent_stats.get('average_response_time', config.average_response_time)
                config.total_cost = agent_stats.get('total_cost', config.total_cost)
                config.last_used = datetime.utcnow()
                
                # Save updated configuration
                await self._save_configuration_to_disk(config)
            
            self.metrics.serializations += 1
            
        except Exception as e:
            logger.error(f"Error serializing agent {agent_id}: {e}")
    
    def _record_access(self, agent_id: str):
        """Record agent access for pattern analysis."""
        self.access_patterns[agent_id].append(datetime.utcnow())
        
        # Keep only recent accesses (last 7 days)
        cutoff = datetime.utcnow() - timedelta(days=7)
        self.access_patterns[agent_id] = [
            access for access in self.access_patterns[agent_id] 
            if access > cutoff
        ]
        
        # Update usage frequency
        recent_accesses = len(self.access_patterns[agent_id])
        self.usage_frequencies[agent_id] = recent_accesses / 7.0  # Accesses per day
    
    def _select_agents_for_preloading(self) -> List[str]:
        """Select agents for intelligent preloading based on priority scores."""
        candidates = []
        
        for agent_id, config in self.agent_configurations.items():
            if (agent_id not in self.active_agents and 
                config.priority_score > self.preload_threshold):
                candidates.append((agent_id, config.priority_score))
        
        # Sort by priority score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [agent_id for agent_id, _ in candidates]
    
    async def _update_priority_scores(self):
        """Update priority scores for all agents based on usage patterns."""
        now = datetime.utcnow()
        
        for agent_id, config in self.agent_configurations.items():
            # Base score from usage frequency
            frequency_score = min(self.usage_frequencies.get(agent_id, 0.0) / 10.0, 1.0)
            
            # Recency score (how recently was it used)
            hours_since_last_use = (now - config.last_used).total_seconds() / 3600
            recency_score = max(0.0, 1.0 - (hours_since_last_use / 168))  # 1 week decay
            
            # Success rate score
            success_score = config.success_rate
            
            # Combined priority score
            config.priority_score = (
                0.4 * frequency_score +
                0.3 * recency_score +
                0.3 * success_score
            )
    
    async def _evict_least_used_configuration(self):
        """Remove least used agent configuration to make room for new one."""
        if not self.agent_configurations:
            return
        
        # Find least used configuration
        least_used_id = min(
            self.agent_configurations.keys(),
            key=lambda aid: (
                self.agent_configurations[aid].usage_count,
                self.agent_configurations[aid].last_used
            )
        )
        
        await self.remove_agent(least_used_id)
        logger.info(f"Evicted least used configuration: {least_used_id}")
    
    async def _save_configuration_to_disk(self, config: AgentConfiguration):
        """Save agent configuration to disk."""
        try:
            config_file = self.cache_directory / f"{config.agent_id}.json"
            with open(config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration {config.agent_id}: {e}")
    
    def _load_configurations_from_disk(self):
        """Load all agent configurations from disk on startup."""
        try:
            for config_file in self.cache_directory.glob("*.json"):
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    config = AgentConfiguration.from_dict(config_data)
                    self.agent_configurations[config.agent_id] = config
                    self.usage_frequencies[config.agent_id] = 0.0
                    
                except Exception as e:
                    logger.warning(f"Error loading configuration from {config_file}: {e}")
            
            logger.info(f"Loaded {len(self.agent_configurations)} configurations from disk")
            
        except Exception as e:
            logger.error(f"Error loading configurations from disk: {e}")
    
    def _agent_cleanup_callback(self, agent_id: str):
        """Callback for when agent is garbage collected."""
        def cleanup():
            if agent_id in self.agent_refs:
                del self.agent_refs[agent_id]
            logger.debug(f"Agent garbage collected: {agent_id}")
        return cleanup
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        async def cleanup_task():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self._cleanup_old_data()
                    await self.optimize_cache()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        async def preload_task():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval // 2)  # Run twice as often as cleanup
                    if len(self.active_agents) < self.max_active_agents * 0.8:  # If cache is not full
                        await self.preload_agents()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in preload task: {e}")
        
        self.cleanup_task = asyncio.create_task(cleanup_task())
        self.preload_task = asyncio.create_task(preload_task())
    
    async def _cleanup_old_data(self):
        """Clean up old access patterns and unused configurations."""
        try:
            now = datetime.utcnow()
            cutoff = now - timedelta(days=30)  # Remove data older than 30 days
            
            # Clean up old access patterns
            for agent_id in list(self.access_patterns.keys()):
                self.access_patterns[agent_id] = [
                    access for access in self.access_patterns[agent_id]
                    if access > cutoff
                ]
                
                if not self.access_patterns[agent_id]:
                    del self.access_patterns[agent_id]
            
            # Clean up unused configurations
            unused_configs = []
            for agent_id, config in self.agent_configurations.items():
                if (config.last_used < cutoff and 
                    config.usage_count < 5 and
                    agent_id not in self.active_agents):
                    unused_configs.append(agent_id)
            
            for agent_id in unused_configs:
                await self.remove_agent(agent_id)
            
            if unused_configs:
                logger.info(f"Cleaned up {len(unused_configs)} unused configurations")
                
        except Exception as e:
            logger.error(f"Error in cleanup_old_data: {e}") 