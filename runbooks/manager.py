"""
Supabase Runbook Manager

Manages global runbooks stored in Supabase database instead of YAML files.
Provides CRUD operations, trigger matching, and performance analytics.
"""

import asyncio
import json
import logging
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

logger = logging.getLogger(__name__)

@dataclass
class RunbookTrigger:
    """Represents a runbook trigger condition."""
    condition_type: str
    parameters: Dict[str, Any]
    priority: int
    is_active: bool = True

@dataclass
class GlobalRunbook:
    """Represents a global runbook from Supabase."""
    id: str
    name: str
    version: str
    description: str
    definition: Dict[str, Any]
    category: str
    tags: List[str]
    priority: int
    status: str
    llm_context: Optional[str] = None
    agent_compatibility: List[str] = None
    usage_count: int = 0
    success_rate: float = 0.0
    triggers: List[RunbookTrigger] = None

class SupabaseRunbookManager:
    """
    Manages global runbooks stored in Supabase database.
    
    Replaces filesystem-based YAML runbook storage with database persistence,
    intelligent querying, and performance analytics.
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize the runbook manager with Supabase connection."""
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.client: Optional[Client] = None
        self._runbook_cache: Dict[str, GlobalRunbook] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_update = None
        
        if not SUPABASE_AVAILABLE:
            logger.warning("Supabase not available - runbook manager in fallback mode")
        
        logger.info("Supabase Runbook Manager initialized")
    
    async def initialize(self):
        """Initialize Supabase connection."""
        if not SUPABASE_AVAILABLE:
            logger.error("Cannot initialize - Supabase library not available")
            return False
        
        if not self.supabase_url or not self.supabase_key:
            logger.error("Missing Supabase credentials for runbook manager")
            return False
        
        try:
            self.client = create_client(self.supabase_url, self.supabase_key)
            
            # Test connection
            result = self.client.table("global_runbooks").select("count").execute()
            logger.info(f"✅ Connected to Supabase runbooks database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            return False
    
    async def get_runbook(self, name: str) -> Optional[GlobalRunbook]:
        """Get a specific runbook by name."""
        if not self.client:
            return None
        
        try:
            # Check cache first
            if name in self._runbook_cache:
                return self._runbook_cache[name]
            
            # Query database
            result = self.client.table("global_runbooks")\
                .select("*, runbook_triggers(*)")\
                .eq("name", name)\
                .eq("status", "active")\
                .execute()
            
            if not result.data:
                logger.warning(f"Runbook '{name}' not found")
                return None
            
            runbook_data = result.data[0]
            runbook = self._convert_to_runbook(runbook_data)
            
            # Cache for future use
            self._runbook_cache[name] = runbook
            
            logger.info(f"Retrieved runbook: {name}")
            return runbook
            
        except Exception as e:
            logger.error(f"Error retrieving runbook {name}: {e}")
            return None
    
    async def list_runbooks(self, category: str = None, 
                          agent_type: str = None) -> List[GlobalRunbook]:
        """List all active runbooks, optionally filtered by category or agent type."""
        if not self.client:
            return []
        
        try:
            query = self.client.table("global_runbooks")\
                .select("*, runbook_triggers(*)")\
                .eq("status", "active")
            
            if category:
                query = query.eq("category", category)
            
            if agent_type:
                query = query.contains("agent_compatibility", [agent_type])
            
            result = query.order("priority", desc=True)\
                         .order("success_rate", desc=True)\
                         .execute()
            
            runbooks = [self._convert_to_runbook(data) for data in result.data]
            
            logger.info(f"Retrieved {len(runbooks)} runbooks")
            return runbooks
            
        except Exception as e:
            logger.error(f"Error listing runbooks: {e}")
            return []
    
    async def find_matching_runbooks(self, message: str, 
                                   agent_type: str = None,
                                   user_context: Dict[str, Any] = None) -> List[Tuple[GlobalRunbook, float]]:
        """
        Find runbooks that match the given message and context.
        Returns list of (runbook, match_score) tuples ordered by relevance.
        """
        if not self.client:
            return []
        
        try:
            # Use database function for intelligent matching
            result = self.client.rpc("find_matching_runbooks", {
                "message_text": message,
                "agent_type": agent_type,
                "user_context": json.dumps(user_context or {})
            }).execute()
            
            matches = []
            for match in result.data:
                if match["match_score"] > 0.1:  # Minimum relevance threshold
                    runbook = await self.get_runbook(match["runbook_name"])
                    if runbook:
                        matches.append((runbook, match["match_score"]))
            
            # Sort by match score descending
            matches.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Found {len(matches)} matching runbooks for message")
            return matches
            
        except Exception as e:
            logger.error(f"Error finding matching runbooks: {e}")
            return []
    
    async def create_runbook(self, name: str, definition: Dict[str, Any],
                           description: str = "", category: str = "general",
                           tags: List[str] = None, agent_compatibility: List[str] = None,
                           triggers: List[RunbookTrigger] = None) -> bool:
        """Create a new global runbook."""
        if not self.client:
            return False
        
        try:
            # Insert runbook
            runbook_data = {
                "name": name,
                "description": description,
                "definition": definition,
                "category": category,
                "tags": tags or [],
                "agent_compatibility": agent_compatibility or [],
                "llm_context": definition.get("metadata", {}).get("llm_context", ""),
                "created_by": "system"
            }
            
            result = self.client.table("global_runbooks")\
                .insert(runbook_data)\
                .execute()
            
            if not result.data:
                logger.error(f"Failed to create runbook: {name}")
                return False
            
            runbook_id = result.data[0]["id"]
            
            # Insert triggers
            if triggers:
                trigger_data = []
                for trigger in triggers:
                    trigger_data.append({
                        "runbook_id": runbook_id,
                        "condition_type": trigger.condition_type,
                        "parameters": trigger.parameters,
                        "priority": trigger.priority,
                        "is_active": trigger.is_active
                    })
                
                self.client.table("runbook_triggers")\
                    .insert(trigger_data)\
                    .execute()
            
            # Clear cache
            self._clear_cache()
            
            logger.info(f"Created runbook: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating runbook {name}: {e}")
            return False
    
    async def update_runbook(self, name: str, definition: Dict[str, Any] = None,
                           description: str = None, status: str = None) -> bool:
        """Update an existing runbook."""
        if not self.client:
            return False
        
        try:
            update_data = {"updated_at": datetime.utcnow().isoformat()}
            
            if definition:
                update_data["definition"] = definition
            if description:
                update_data["description"] = description
            if status:
                update_data["status"] = status
            
            result = self.client.table("global_runbooks")\
                .update(update_data)\
                .eq("name", name)\
                .execute()
            
            if not result.data:
                logger.error(f"Failed to update runbook: {name}")
                return False
            
            # Clear cache
            self._clear_cache()
            
            logger.info(f"Updated runbook: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating runbook {name}: {e}")
            return False
    
    async def record_execution(self, runbook_name: str, success: bool, 
                             execution_time: float) -> None:
        """Record runbook execution for analytics."""
        if not self.client:
            return
        
        try:
            # Use database function to update usage statistics
            self.client.rpc("update_runbook_usage", {
                "runbook_name": runbook_name,
                "execution_success": success,
                "execution_time": execution_time
            }).execute()
            
            logger.debug(f"Recorded execution for {runbook_name}: success={success}")
            
        except Exception as e:
            logger.error(f"Error recording execution for {runbook_name}: {e}")
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get runbook performance analytics."""
        if not self.client:
            return {}
        
        try:
            # Get performance dashboard data
            result = self.client.table("runbook_performance_dashboard")\
                .select("*")\
                .execute()
            
            analytics = {
                "total_runbooks": len(result.data),
                "runbooks": result.data,
                "categories": {},
                "performance_summary": {
                    "avg_success_rate": 0.0,
                    "total_usage": 0,
                    "most_used": None,
                    "best_performing": None
                }
            }
            
            if result.data:
                # Calculate summary statistics
                total_usage = sum(r["usage_count"] for r in result.data)
                avg_success = sum(r["success_rate"] for r in result.data) / len(result.data)
                
                analytics["performance_summary"]["total_usage"] = total_usage
                analytics["performance_summary"]["avg_success_rate"] = round(avg_success, 3)
                
                # Find most used and best performing
                most_used = max(result.data, key=lambda x: x["usage_count"])
                best_performing = max(result.data, key=lambda x: x["success_rate"])
                
                analytics["performance_summary"]["most_used"] = most_used["name"]
                analytics["performance_summary"]["best_performing"] = best_performing["name"]
                
                # Group by category
                for runbook in result.data:
                    category = runbook["category"]
                    if category not in analytics["categories"]:
                        analytics["categories"][category] = []
                    analytics["categories"][category].append(runbook)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {}
    
    async def migrate_yaml_runbook(self, yaml_path: str) -> bool:
        """Migrate a YAML runbook file to Supabase."""
        try:
            with open(yaml_path, 'r') as file:
                runbook_data = yaml.safe_load(file)
            
            # Extract runbook information
            metadata = runbook_data.get("metadata", {})
            name = metadata.get("name", "unknown")
            description = metadata.get("description", "")
            
            # Convert triggers
            triggers = []
            for trigger_data in runbook_data.get("triggers", []):
                trigger = RunbookTrigger(
                    condition_type=trigger_data.get("condition"),
                    parameters=trigger_data.get("parameters", {}),
                    priority=trigger_data.get("priority", 1)
                )
                triggers.append(trigger)
            
            # Determine category and agent compatibility
            category = "user_interaction" if "question" in name else "general"
            agent_compatibility = ["general"]  # Default, can be enhanced
            
            # Create runbook in database
            success = await self.create_runbook(
                name=name,
                definition=runbook_data,
                description=description,
                category=category,
                agent_compatibility=agent_compatibility,
                triggers=triggers
            )
            
            if success:
                logger.info(f"Migrated YAML runbook: {name}")
            else:
                logger.error(f"Failed to migrate YAML runbook: {name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error migrating YAML runbook {yaml_path}: {e}")
            return False
    
    def _convert_to_runbook(self, data: Dict[str, Any]) -> GlobalRunbook:
        """Convert database row to GlobalRunbook object."""
        triggers = []
        if "runbook_triggers" in data:
            for trigger_data in data["runbook_triggers"]:
                trigger = RunbookTrigger(
                    condition_type=trigger_data["condition_type"],
                    parameters=trigger_data["parameters"],
                    priority=trigger_data["priority"],
                    is_active=trigger_data["is_active"]
                )
                triggers.append(trigger)
        
        return GlobalRunbook(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            description=data["description"],
            definition=data["definition"],
            category=data["category"],
            tags=data["tags"],
            priority=data["priority"],
            status=data["status"],
            llm_context=data.get("llm_context"),
            agent_compatibility=data.get("agent_compatibility", []),
            usage_count=data.get("usage_count", 0),
            success_rate=data.get("success_rate", 0.0),
            triggers=triggers
        )
    
    def _clear_cache(self):
        """Clear the runbook cache."""
        self._runbook_cache.clear()
        self._last_cache_update = None
        logger.debug("Cleared runbook cache")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the runbook manager."""
        status = {
            "supabase_available": SUPABASE_AVAILABLE,
            "client_connected": self.client is not None,
            "cache_size": len(self._runbook_cache),
            "last_cache_update": self._last_cache_update
        }
        
        if self.client:
            try:
                result = self.client.table("global_runbooks")\
                    .select("count")\
                    .execute()
                status["database_accessible"] = True
                status["total_runbooks"] = len(result.data) if result.data else 0
            except Exception as e:
                status["database_accessible"] = False
                status["error"] = str(e)
        
        return status

# Global instance
runbook_manager = SupabaseRunbookManager()

async def get_runbook_manager() -> SupabaseRunbookManager:
    """Get the global runbook manager instance."""
    return runbook_manager

async def initialize_runbook_manager(supabase_url: str, supabase_key: str) -> bool:
    """Initialize the global runbook manager with credentials."""
    global runbook_manager
    runbook_manager.supabase_url = supabase_url
    runbook_manager.supabase_key = supabase_key
    return await runbook_manager.initialize() 