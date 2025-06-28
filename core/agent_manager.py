"""
Agent Manager

Complete CRUD interface for managing AI agents with profile settings, prompts, 
temperature, configuration, and lifecycle management.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from storage.supabase import SupabaseLogger

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status options."""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    ARCHIVED = "ARCHIVED"
    DRAFT = "DRAFT"

@dataclass
class AgentProfile:
    """Complete agent profile with all configurable settings."""
    id: Optional[str] = None
    name: str = ""
    specialty: str = ""
    description: str = ""
    
    # Core LLM settings
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 500
    
    # Prompt configuration
    system_prompt: str = ""
    tool_decision_guidance: str = ""
    communication_style: str = ""
    tool_selection_criteria: str = ""
    
    # Agent settings
    status: AgentStatus = AgentStatus.DRAFT
    complexity_level: int = 3  # 1=basic, 2=simple, 3=medium, 4=advanced, 5=expert
    
    # Metadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    created_by: str = "user"
    tags: List[str] = None
    
    # Performance tracking
    total_interactions: int = 0
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    user_satisfaction: float = 0.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()

class AgentManager:
    """
    Complete CRUD management system for AI agents.
    
    Features:
    - Full CRUD operations (Create, Read, Update, Delete)
    - Agent profile management
    - Prompt and configuration editing
    - Status management (active/inactive/archived)
    - Performance tracking
    - Bulk operations
    - Search and filtering
    """
    
    def __init__(self, supabase_logger: Optional[SupabaseLogger] = None):
        """Initialize the agent manager."""
        self.supabase_logger = supabase_logger or SupabaseLogger()
        self.cache = {}  # Simple cache for frequently accessed agents
        self.cache_ttl = 300  # 5 minutes
        
        logger.info("✅ AgentManager initialized")
    
    # ================== CREATE OPERATIONS ==================
    
    async def create_agent(self, profile: AgentProfile) -> Dict[str, Any]:
        """
        Create a new agent with the given profile.
        
        Args:
            profile: AgentProfile with agent configuration
            
        Returns:
            Creation result with agent ID and status
        """
        try:
            # Validate profile
            validation_result = self._validate_profile(profile)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': f"Profile validation failed: {validation_result['errors']}",
                    'agent_id': None
                }
            
            # Prepare data for database
            agent_data = {
                'id': profile.id,
                'name': profile.name,
                'specialty': profile.specialty,
                'description': profile.description,
                'model': profile.model,
                'temperature': profile.temperature,
                'max_tokens': profile.max_tokens,
                'system_prompt': profile.system_prompt,
                'tool_decision_guidance': profile.tool_decision_guidance,
                'communication_style': profile.communication_style,
                'tool_selection_criteria': profile.tool_selection_criteria,
                'status': profile.status.value,
                'complexity_level': self._normalize_complexity_level(profile.complexity_level),
                'created_at': profile.created_at,
                'updated_at': datetime.utcnow().isoformat(),
                'created_by': profile.created_by,
                'tags': profile.tags,
                'total_interactions': profile.total_interactions,
                'success_rate': profile.success_rate,
                'avg_response_time': profile.avg_response_time,
                'user_satisfaction': profile.user_satisfaction
            }
            
            # Insert into database
            result = self.supabase_logger.client.table("agents").insert(agent_data).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"✅ Created agent: {profile.name} ({profile.id})")
                
                # Clear cache
                self._clear_cache()
                
                return {
                    'success': True,
                    'agent_id': profile.id,
                    'message': f"Agent '{profile.name}' created successfully",
                    'agent_data': result.data[0]
                }
            else:
                return {
                    'success': False,
                    'error': "Failed to create agent in database",
                    'agent_id': None
                }
                
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_id': None
            }
    
    async def duplicate_agent(self, agent_id: str, new_name: str) -> Dict[str, Any]:
        """
        Duplicate an existing agent with a new name.
        
        Args:
            agent_id: ID of agent to duplicate
            new_name: Name for the new agent
            
        Returns:
            Duplication result
        """
        try:
            # Get existing agent
            existing_agent = await self.get_agent(agent_id)
            if not existing_agent['success']:
                return {
                    'success': False,
                    'error': f"Source agent not found: {agent_id}",
                    'agent_id': None
                }
            
            # Create new profile based on existing
            source_profile = existing_agent['agent']
            new_profile = AgentProfile(
                name=new_name,
                specialty=source_profile['specialty'],
                description=f"Copy of {source_profile['name']}",
                model=source_profile['model'],
                temperature=source_profile['temperature'],
                max_tokens=source_profile['max_tokens'],
                system_prompt=source_profile['system_prompt'],
                tool_decision_guidance=source_profile.get('tool_decision_guidance', ''),
                communication_style=source_profile.get('communication_style', ''),
                tool_selection_criteria=source_profile.get('tool_selection_criteria', ''),
                status=AgentStatus.DRAFT,  # Start as draft
                complexity_level=source_profile.get('complexity_level', 'medium'),
                tags=source_profile.get('tags', []).copy()
            )
            
            # Create the duplicate
            return await self.create_agent(new_profile)
            
        except Exception as e:
            logger.error(f"Failed to duplicate agent {agent_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_id': None
            }
    
    # ================== READ OPERATIONS ==================
    
    async def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Get a single agent by ID.
        
        Args:
            agent_id: Agent ID to retrieve
            
        Returns:
            Agent data or error
        """
        try:
            # Check cache first
            cache_key = f"agent_{agent_id}"
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if (datetime.utcnow() - cache_entry['timestamp']).seconds < self.cache_ttl:
                    return {
                        'success': True,
                        'agent': cache_entry['data'],
                        'source': 'cache'
                    }
            
            # Query database
            result = self.supabase_logger.client.table("agents")\
                .select("*")\
                .eq("id", agent_id)\
                .execute()
            
            if result.data and len(result.data) > 0:
                agent_data = result.data[0]
                
                # Cache the result
                self.cache[cache_key] = {
                    'data': agent_data,
                    'timestamp': datetime.utcnow()
                }
                
                return {
                    'success': True,
                    'agent': agent_data,
                    'source': 'database'
                }
            else:
                return {
                    'success': False,
                    'error': f"Agent {agent_id} not found",
                    'agent': None
                }
                
        except Exception as e:
            logger.error(f"Failed to get agent {agent_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent': None
            }
    
    async def list_agents(self, 
                         status: Optional[str] = None,
                         specialty: Optional[str] = None,
                         search: Optional[str] = None,
                         limit: int = 50,
                         offset: int = 0) -> Dict[str, Any]:
        """
        List agents with filtering and pagination.
        
        Args:
            status: Filter by status (active, inactive, archived, draft)
            specialty: Filter by specialty
            search: Search in name and description
            limit: Number of results to return
            offset: Number of results to skip
            
        Returns:
            List of agents with metadata
        """
        try:
            # Build query
            query = self.supabase_logger.client.table("agents").select("*")
            
            # Apply filters
            if status:
                query = query.eq("status", status)
            if specialty:
                query = query.eq("specialty", specialty)
            if search:
                # Simple text search - could be enhanced with full-text search
                query = query.or_(f"name.ilike.%{search}%,description.ilike.%{search}%")
            
            # Apply pagination and ordering
            query = query.order("updated_at", desc=True)\
                        .range(offset, offset + limit - 1)
            
            result = query.execute()
            
            # Get total count for pagination
            count_query = self.supabase_logger.client.table("agents").select("id", count="exact")
            if status:
                count_query = count_query.eq("status", status)
            if specialty:
                count_query = count_query.eq("specialty", specialty)
            
            count_result = count_query.execute()
            total_count = count_result.count if hasattr(count_result, 'count') else len(result.data or [])
            
            return {
                'success': True,
                'agents': result.data or [],
                'pagination': {
                    'total': total_count,
                    'limit': limit,
                    'offset': offset,
                    'has_more': (offset + limit) < total_count
                },
                'filters': {
                    'status': status,
                    'specialty': specialty,
                    'search': search
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            return {
                'success': False,
                'error': str(e),
                'agents': [],
                'pagination': {'total': 0, 'limit': limit, 'offset': offset, 'has_more': False}
            }
    
    async def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get overall agent statistics.
        
        Returns:
            Statistics summary
        """
        try:
            # Get all agents for stats
            all_agents = await self.list_agents(limit=1000)
            
            if not all_agents['success']:
                return {
                    'success': False,
                    'error': all_agents['error'],
                    'stats': {}
                }
            
            agents = all_agents['agents']
            
            # Calculate statistics
            stats = {
                'total_agents': len(agents),
                'active_agents': len([a for a in agents if a['status'] == 'active']),
                'inactive_agents': len([a for a in agents if a['status'] == 'inactive']),
                'draft_agents': len([a for a in agents if a['status'] == 'draft']),
                'archived_agents': len([a for a in agents if a['status'] == 'archived']),
                'specialties': {},
                'models': {},
                'avg_temperature': 0.0,
                'total_interactions': 0,
                'avg_success_rate': 0.0
            }
            
            # Specialty breakdown
            for agent in agents:
                specialty = agent.get('specialty') or 'unspecified'
                stats['specialties'][specialty] = stats['specialties'].get(specialty, 0) + 1
                
                model = agent.get('model', 'unknown')
                stats['models'][model] = stats['models'].get(model, 0) + 1
                
                stats['total_interactions'] += agent.get('total_interactions', 0)
            
            # Calculate averages
            if agents:
                stats['avg_temperature'] = sum(a.get('temperature', 0.7) for a in agents) / len(agents)
                success_rates = [a.get('success_rate', 0.0) for a in agents if a.get('total_interactions', 0) > 0]
                if success_rates:
                    stats['avg_success_rate'] = sum(success_rates) / len(success_rates)
            
            return {
                'success': True,
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent stats: {e}")
            return {
                'success': False,
                'error': str(e),
                'stats': {}
            }
    
    # ================== UPDATE OPERATIONS ==================
    
    async def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an agent's configuration.
        
        Args:
            agent_id: Agent ID to update
            updates: Dictionary of fields to update
            
        Returns:
            Update result
        """
        try:
            # Validate updates
            validation_result = self._validate_updates(updates)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': f"Update validation failed: {validation_result['errors']}",
                    'updated_fields': []
                }
            
            # Add update timestamp
            updates['updated_at'] = datetime.utcnow().isoformat()
            
            # Update in database
            result = self.supabase_logger.client.table("agents")\
                .update(updates)\
                .eq("id", agent_id)\
                .execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"✅ Updated agent {agent_id}: {list(updates.keys())}")
                
                # Clear cache
                self._clear_cache()
                
                return {
                    'success': True,
                    'updated_fields': list(updates.keys()),
                    'agent': result.data[0],
                    'message': f"Agent updated successfully"
                }
            else:
                return {
                    'success': False,
                    'error': f"Agent {agent_id} not found or update failed",
                    'updated_fields': []
                }
                
        except Exception as e:
            logger.error(f"Failed to update agent {agent_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'updated_fields': []
            }
    
    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> Dict[str, Any]:
        """
        Update an agent's status.
        
        Args:
            agent_id: Agent ID
            status: New status
            
        Returns:
            Update result
        """
        return await self.update_agent(agent_id, {'status': status.value})
    
    async def update_agent_prompt(self, agent_id: str, prompt_updates: Dict[str, str]) -> Dict[str, Any]:
        """
        Update an agent's prompt configuration.
        
        Args:
            agent_id: Agent ID
            prompt_updates: Dictionary of prompt fields to update
            
        Returns:
            Update result
        """
        return await self.update_agent(agent_id, prompt_updates)
    
    async def bulk_update_agents(self, agent_ids: List[str], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update multiple agents at once.
        
        Args:
            agent_ids: List of agent IDs to update
            updates: Dictionary of fields to update
            
        Returns:
            Bulk update result
        """
        try:
            results = []
            errors = []
            
            for agent_id in agent_ids:
                result = await self.update_agent(agent_id, updates.copy())
                if result['success']:
                    results.append(agent_id)
                else:
                    errors.append(f"{agent_id}: {result['error']}")
            
            return {
                'success': len(results) > 0,
                'updated_agents': results,
                'failed_agents': len(errors),
                'errors': errors,
                'message': f"Updated {len(results)} agents, {len(errors)} failed"
            }
            
        except Exception as e:
            logger.error(f"Failed bulk update: {e}")
            return {
                'success': False,
                'error': str(e),
                'updated_agents': [],
                'failed_agents': len(agent_ids)
            }
    
    # ================== DELETE OPERATIONS ==================
    
    async def delete_agent(self, agent_id: str, soft_delete: bool = True) -> Dict[str, Any]:
        """
        Delete an agent (soft delete by default).
        
        Args:
            agent_id: Agent ID to delete
            soft_delete: If True, archive instead of hard delete
            
        Returns:
            Deletion result
        """
        try:
            if soft_delete:
                # Soft delete - change status to archived
                result = await self.update_agent_status(agent_id, AgentStatus.ARCHIVED)
                if result['success']:
                    return {
                        'success': True,
                        'message': f"Agent {agent_id} archived successfully",
                        'action': 'archived'
                    }
                else:
                    return result
            else:
                # Hard delete
                result = self.supabase_logger.client.table("agents")\
                    .delete()\
                    .eq("id", agent_id)\
                    .execute()
                
                if result.data is not None:  # Supabase returns data even for deletes
                    logger.info(f"🗑️ Hard deleted agent {agent_id}")
                    
                    # Clear cache
                    self._clear_cache()
                    
                    return {
                        'success': True,
                        'message': f"Agent {agent_id} deleted permanently",
                        'action': 'deleted'
                    }
                else:
                    return {
                        'success': False,
                        'error': f"Agent {agent_id} not found or delete failed",
                        'action': 'none'
                    }
                    
        except Exception as e:
            logger.error(f"Failed to delete agent {agent_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': 'none'
            }
    
    # ================== HELPER METHODS ==================
    
    def _validate_profile(self, profile: AgentProfile) -> Dict[str, Any]:
        """Validate agent profile data."""
        errors = []
        
        if not profile.name or len(profile.name.strip()) == 0:
            errors.append("Agent name is required")
        
        if not profile.specialty or len(profile.specialty.strip()) == 0:
            errors.append("Agent specialty is required")
        
        if profile.temperature < 0.0 or profile.temperature > 2.0:
            errors.append("Temperature must be between 0.0 and 2.0")
        
        if profile.max_tokens < 1 or profile.max_tokens > 4000:
            errors.append("Max tokens must be between 1 and 4000")
        
        if not profile.system_prompt or len(profile.system_prompt.strip()) == 0:
            errors.append("System prompt is required")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _validate_updates(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Validate update data."""
        errors = []
        
        if 'temperature' in updates:
            temp = updates['temperature']
            if not isinstance(temp, (int, float)) or temp < 0.0 or temp > 2.0:
                errors.append("Temperature must be a number between 0.0 and 2.0")
        
        if 'max_tokens' in updates:
            tokens = updates['max_tokens']
            if not isinstance(tokens, int) or tokens < 1 or tokens > 4000:
                errors.append("Max tokens must be an integer between 1 and 4000")
        
        if 'status' in updates:
            status = updates['status']
            valid_statuses = [s.value for s in AgentStatus]
            if status not in valid_statuses:
                errors.append(f"Status must be one of: {valid_statuses}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _clear_cache(self):
        """Clear the agent cache."""
        self.cache.clear()
        logger.debug("🧹 Agent cache cleared") 
    
    def _normalize_complexity_level(self, complexity_level) -> int:
        """Convert string complexity levels to integers."""
        if isinstance(complexity_level, int):
            return max(1, min(5, complexity_level))  # Ensure 1-5 range
        
        # String to int mapping
        level_map = {
            'basic': 1,
            'simple': 2, 
            'medium': 3,
            'advanced': 4,
            'expert': 5
        }
        
        if isinstance(complexity_level, str):
            return level_map.get(complexity_level.lower(), 3)
        
        return 3  # Default to medium