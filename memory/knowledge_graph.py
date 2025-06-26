"""
Knowledge Graph Manager for user relationships and contextual insights.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

from database.supabase_logger import SupabaseLogger

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    """Manages knowledge graphs for user relationships and contextual insights."""
    
    def __init__(self, supabase_logger: Optional[SupabaseLogger] = None):
        self.supabase_logger = supabase_logger or SupabaseLogger()
        self.user_graph = None
        self.preference_graph = None
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.relationship_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_ttl = timedelta(hours=2)
        self.last_cache_update: Dict[str, datetime] = {}
        
        self._initialize_graphs()
        logger.info(f"Knowledge Graph Manager initialized (NetworkX: {NETWORKX_AVAILABLE})")
    
    def _initialize_graphs(self):
        """Initialize NetworkX graphs if available."""
        if not NETWORKX_AVAILABLE:
            return
        
        try:
            self.user_graph = nx.DiGraph()
            self.preference_graph = nx.Graph()
            logger.info("NetworkX graphs initialized")
        except Exception as e:
            logger.error(f"Failed to initialize graphs: {e}")
    
    async def add_user_interaction(self, user_id: str, interaction_type: str,
                                 target_entity: str, context: Dict[str, Any],
                                 strength: float = 1.0) -> bool:
        """Add user interaction to knowledge graph."""
        try:
            interaction_data = {
                "user_id": user_id,
                "interaction_type": interaction_type,
                "target_entity": target_entity,
                "context": context,
                "strength": strength,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._store_interaction_in_database(interaction_data)
            
            if NETWORKX_AVAILABLE:
                await self._update_graphs_with_interaction(interaction_data)
            
            await self._update_user_profile(user_id, interaction_data)
            return True
            
        except Exception as e:
            logger.error(f"Failed to add user interaction: {e}")
            return False
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences from interactions."""
        try:
            preferences = {"agents": {}, "topics": {}, "patterns": {}}
            
            result = self.supabase_logger.client.table("user_relationships") \
                .select("*") \
                .eq("user_id", user_id) \
                .in_("interaction_type", ["preference", "feedback", "rating"]) \
                .execute()
            
            if result.data:
                for record in result.data:
                    target = record["target_entity"]
                    strength = record["strength"]
                    
                    if "agent" in target.lower():
                        preferences["agents"][target] = preferences["agents"].get(target, 0) + strength
                    else:
                        preferences["topics"][target] = preferences["topics"].get(target, 0) + strength
            
            return preferences
            
        except Exception as e:
            logger.error(f"Failed to get user preferences: {e}")
            return {}
    
    async def find_similar_users(self, user_id: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Find users with similar patterns."""
        try:
            # Simple database-based similarity for now
            result = self.supabase_logger.client.table("user_relationships") \
                .select("user_id,target_entity") \
                .neq("user_id", user_id) \
                .execute()
            
            user_targets = defaultdict(set)
            for record in result.data:
                user_targets[record["user_id"]].add(record["target_entity"])
            
            # Get current user's targets
            current_result = self.supabase_logger.client.table("user_relationships") \
                .select("target_entity") \
                .eq("user_id", user_id) \
                .execute()
            
            current_targets = set(r["target_entity"] for r in current_result.data)
            
            # Calculate similarities
            similar_users = []
            for other_user, other_targets in user_targets.items():
                if other_targets and current_targets:
                    intersection = len(current_targets.intersection(other_targets))
                    union = len(current_targets.union(other_targets))
                    similarity = intersection / union if union > 0 else 0.0
                    
                    if similarity > 0.1:
                        similar_users.append({
                            "user_id": other_user,
                            "similarity_score": similarity,
                            "common_preferences": list(current_targets.intersection(other_targets))
                        })
            
            similar_users.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similar_users[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to find similar users: {e}")
            return []
    
    async def _store_interaction_in_database(self, interaction_data: Dict[str, Any]):
        """Store interaction in database."""
        try:
            self.supabase_logger.client.table("user_relationships").insert(interaction_data).execute()
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
    
    async def _update_graphs_with_interaction(self, interaction_data: Dict[str, Any]):
        """Update graphs with new interaction."""
        if not NETWORKX_AVAILABLE or not self.user_graph:
            return
        
        try:
            user_id = interaction_data["user_id"]
            target_entity = interaction_data["target_entity"]
            strength = interaction_data["strength"]
            
            user_node = f"user_{user_id}"
            target_node = f"target_{target_entity}"
            
            if not self.user_graph.has_node(user_node):
                self.user_graph.add_node(user_node, type="user")
            
            if not self.user_graph.has_node(target_node):
                self.user_graph.add_node(target_node, type="target")
            
            if self.user_graph.has_edge(user_node, target_node):
                current_weight = self.user_graph[user_node][target_node].get("weight", 0)
                self.user_graph[user_node][target_node]["weight"] = current_weight + strength
            else:
                self.user_graph.add_edge(user_node, target_node, weight=strength)
                
        except Exception as e:
            logger.error(f"Failed to update graphs: {e}")
    
    async def _update_user_profile(self, user_id: str, interaction_data: Dict[str, Any]):
        """Update user profile with interaction."""
        try:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    "interaction_count": 0,
                    "interaction_types": Counter(),
                    "target_entities": Counter()
                }
            
            profile = self.user_profiles[user_id]
            profile["interaction_count"] += 1
            profile["interaction_types"][interaction_data["interaction_type"]] += 1
            profile["target_entities"][interaction_data["target_entity"]] += 1
            
        except Exception as e:
            logger.error(f"Failed to update user profile: {e}")
    
    def is_available(self) -> bool:
        """Check if knowledge graph features are available."""
        return NETWORKX_AVAILABLE
    
    async def close(self):
        """Clean up resources."""
        self.user_profiles.clear()
        self.relationship_cache.clear()
        if self.user_graph:
            self.user_graph.clear()
        if self.preference_graph:
            self.preference_graph.clear()
        logger.info("Knowledge Graph Manager closed") 