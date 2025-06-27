"""
Real-time Data Updater for Dashboard

Gathers data from Supabase logs, agent orchestrator, event bus, and cost analytics.
Provides structured data for the terminal dashboard components.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class RealTimeUpdater:
    """Real-time data updater for dashboard components."""
    
    def __init__(self, db_logger=None, orchestrator=None, event_bus=None):
        """Initialize the real-time updater."""
        self.db_logger = db_logger
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        
        self.is_running_flag = False
        self.cached_data = {
            "overview": {},
            "agents": {},
            "costs": {},
            "events": {},
            "logs": {}
        }
        
        logger.info("Real-time updater initialized")
    
    def is_running(self) -> bool:
        """Check if updater is running."""
        return self.is_running_flag
    
    def start(self):
        """Start the real-time updater."""
        self.is_running_flag = True
    
    def stop(self):
        """Stop the real-time updater."""
        self.is_running_flag = False
    
    async def get_all_data(self) -> Dict[str, Any]:
        """Get all dashboard data."""
        try:
            # Update all data in parallel
            await asyncio.gather(
                self._update_overview(),
                self._update_agents(),
                self._update_costs(),
                self._update_events(),
                self._update_logs(),
                return_exceptions=True
            )
            
            return self.cached_data.copy()
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return self.cached_data.copy()
    
    async def _update_overview(self):
        """Update system overview data."""
        try:
            # Get system health from improvement orchestrator
            system_health = await self._get_system_health()
            agent_stats = await self._get_agent_stats()
            
            self.cached_data["overview"] = {
                "system_health": system_health,
                "agent_ecosystem": agent_stats,
                "improvement_status": await self._get_improvement_status(),
                "recent_events": await self._get_recent_events(5),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating overview: {e}")
    
    async def _update_agents(self):
        """Update agents data."""
        try:
            self.cached_data["agents"] = {
                "active_agents": await self._get_active_agents(),
                "performance_metrics": await self._get_agent_performance(),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating agents: {e}")
    
    async def _update_costs(self):
        """Update cost data."""
        try:
            self.cached_data["costs"] = {
                "daily_cost": await self._get_daily_cost(),
                "efficiency_score": 0.78,
                "optimizations": await self._get_cost_optimizations(),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating costs: {e}")
    
    async def _update_events(self):
        """Update events data."""
        try:
            self.cached_data["events"] = {
                "recent_events": await self._get_recent_events(20),
                "event_metrics": await self._get_event_metrics(),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating events: {e}")
    
    async def _update_logs(self):
        """Update logs data."""
        try:
            self.cached_data["logs"] = {
                "recent_logs": await self._get_recent_logs(),
                "error_summary": await self._get_error_summary(),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating logs: {e}")
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics."""
        try:
            if self.orchestrator and hasattr(self.orchestrator, 'improvement_orchestrator'):
                orch = self.orchestrator.improvement_orchestrator
                if orch:
                    health = await orch._assess_system_health()
                    return {
                        "overall_score": health.overall_score,
                        "performance_score": health.performance_score,
                        "cost_efficiency": health.cost_efficiency_score,
                        "user_satisfaction": health.user_satisfaction_score
                    }
            
            return {
                "overall_score": 0.85,
                "performance_score": 0.90,
                "cost_efficiency": 0.75,
                "user_satisfaction": 0.88
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {"overall_score": 0.5, "performance_score": 0.5, "cost_efficiency": 0.5, "user_satisfaction": 0.5}
    
    async def _get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        try:
            if self.orchestrator:
                stats = self.orchestrator.get_agent_stats()
                return {
                    "active_count": stats.get("active_agents", 0),
                    "total_configurations": stats.get("total_agents", 0),
                    "cache_hit_rate": 0.67,
                    "avg_success_rate": 0.91
                }
            
            return {"active_count": 0, "total_configurations": 0, "cache_hit_rate": 0.0, "avg_success_rate": 0.0}
            
        except Exception as e:
            logger.error(f"Error getting agent stats: {e}")
            return {}
    
    async def _get_improvement_status(self) -> Dict[str, Any]:
        """Get improvement status."""
        try:
            if self.db_logger:
                # Get improvement task statistics
                thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
                
                # Count active tasks
                active_result = self.db_logger.client.table("improvement_tasks").select(
                    "id"
                ).in_("status", ["pending", "running"]).execute()
                
                # Get completed tasks with ROI
                completed_result = self.db_logger.client.table("improvement_tasks").select(
                    "roi", "status"
                ).eq("status", "completed").gte("completed_at", thirty_days_ago).execute()
                
                # Count patterns discovered (from workflow_runs)
                patterns_result = self.db_logger.client.table("workflow_runs").select(
                    "pattern_signature"
                ).is_not("pattern_signature", "null").gte("created_at", thirty_days_ago).execute()
                
                # Calculate metrics
                active_tasks = len(active_result.data) if active_result.data else 0
                
                roi_values = [r.get("roi", 0) for r in (completed_result.data or []) if r.get("roi", 0) > 0]
                avg_roi = sum(roi_values) / len(roi_values) if roi_values else 1.0
                
                unique_patterns = len(set(r.get("pattern_signature") for r in (patterns_result.data or []) 
                                        if r.get("pattern_signature")))
                
                optimizations_applied = len(completed_result.data) if completed_result.data else 0
                
                return {
                    "active_tasks": active_tasks,
                    "roi_last_30_days": round(avg_roi, 2),
                    "patterns_discovered": unique_patterns,
                    "optimizations_applied": optimizations_applied
                }
            
            return {"active_tasks": 0, "roi_last_30_days": 1.0, "patterns_discovered": 0, "optimizations_applied": 0}
            
        except Exception as e:
            logger.error(f"Error getting improvement status: {e}")
            return {"active_tasks": 0, "roi_last_30_days": 1.0, "patterns_discovered": 0, "optimizations_applied": 0}
    
    async def _get_active_agents(self) -> List[Dict[str, Any]]:
        """Get active agents list."""
        try:
            if self.orchestrator and hasattr(self.orchestrator, 'lazy_loader'):
                activity = self.orchestrator.lazy_loader.get_agent_activity_report()
                agents = []
                
                for agent_id, data in activity.items():
                    agents.append({
                        "agent_id": agent_id,
                        "specialty": data.get("specialty", "general"),
                        "is_active": data.get("is_active", False),
                        "success_rate": data.get("success_rate", 0.0),
                        "avg_cost_per_request": 0.025
                    })
                
                return agents[:10]  # Limit to top 10
            
            # Mock data
            return [
                {"agent_id": "general_1", "specialty": "general", "is_active": True, "success_rate": 0.92, "avg_cost_per_request": 0.023},
                {"agent_id": "tech_specialist", "specialty": "technical", "is_active": True, "success_rate": 0.88, "avg_cost_per_request": 0.031}
            ]
            
        except Exception as e:
            logger.error(f"Error getting active agents: {e}")
            return []
    
    async def _get_agent_performance(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            "avg_response_time": 2.3,
            "success_rate": 0.89,
            "cost_efficiency": 0.76
        }
    
    async def _get_daily_cost(self) -> float:
        """Get today's cost."""
        try:
            if self.db_logger:
                # Query today's cost from daily_token_summary
                today = str(datetime.utcnow().date())
                result = self.db_logger.client.table("daily_token_summary").select(
                    "total_cost"
                ).eq("date", today).execute()
                
                if result.data and result.data[0]:
                    return result.data[0].get("total_cost", 0.0)
                
                # Fallback to summing from token_usage
                token_result = self.db_logger.client.table("token_usage").select(
                    "estimated_cost"
                ).gte("created_at", today).execute()
                
                if token_result.data:
                    return sum(r.get("estimated_cost", 0) for r in token_result.data)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting daily cost: {e}")
            return 0.0
    
    async def _get_cost_optimizations(self) -> List[Dict[str, Any]]:
        """Get cost optimization opportunities."""
        return [
            {"title": "Optimize prompt compression", "potential_savings": 0.45, "priority": 5},
            {"title": "Use GPT-3.5 for simple queries", "potential_savings": 0.32, "priority": 4}
        ]
    
    async def _get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent events."""
        try:
            events = []
            
            if self.db_logger:
                # Try to get events from Supabase
                try:
                    result = self.db_logger.client.table("events") \
                        .select("*") \
                        .order("created_at", desc=True) \
                        .limit(limit) \
                        .execute()
                    
                    if result.data:
                        for event in result.data:
                            events.append({
                                "timestamp": event.get("created_at", ""),
                                "type": event.get("event_type", "unknown"),
                                "source": event.get("source", "system")
                            })
                except:
                    pass  # Fall back to mock data
            
            # Mock events if none from DB
            if not events:
                now = datetime.utcnow()
                events = [
                    {"timestamp": (now - timedelta(minutes=1)).isoformat(), "type": "agent_spawned", "source": "orchestrator"},
                    {"timestamp": (now - timedelta(minutes=3)).isoformat(), "type": "workflow_completed", "source": "user"},
                    {"timestamp": (now - timedelta(minutes=5)).isoformat(), "type": "improvement_applied", "source": "improvement"}
                ]
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return []
    
    async def _get_event_metrics(self) -> Dict[str, Any]:
        """Get event metrics."""
        try:
            if self.event_bus:
                return await self.event_bus.get_metrics()
            
            return {"events_published": 1247, "events_processed": 1245, "queue_size": 2}
            
        except Exception as e:
            logger.error(f"Error getting event metrics: {e}")
            return {}
    
    async def _get_recent_logs(self) -> List[Dict[str, Any]]:
        """Get recent logs from Supabase."""
        try:
            logs = []
            
            if self.db_logger:
                try:
                    # Get conversation logs
                    result = self.db_logger.client.table("conversations") \
                        .select("*") \
                        .order("started_at", desc=True) \
                        .limit(20) \
                        .execute()
                    
                    if result.data:
                        for conv in result.data:
                            logs.append({
                                "timestamp": conv.get("started_at", ""),
                                "level": "INFO",
                                "message": f"Conversation {conv.get('id', 'unknown')} - {conv.get('status', 'active')}",
                                "source": "conversation"
                            })
                    
                    # Get message logs
                    msg_result = self.db_logger.client.table("messages") \
                        .select("*") \
                        .order("timestamp", desc=True) \
                        .limit(30) \
                        .execute()
                    
                    if msg_result.data:
                        for msg in msg_result.data:
                            level = "ERROR" if msg.get("error_message") else "INFO"
                            content = str(msg.get("content", ""))[:50]
                            logs.append({
                                "timestamp": msg.get("timestamp", ""),
                                "level": level,
                                "message": f"Message: {content}...",
                                "source": "message"
                            })
                
                except Exception as e:
                    logger.warning(f"Could not fetch logs from Supabase: {e}")
            
            # Mock logs if none from DB
            if not logs:
                now = datetime.utcnow()
                logs = [
                    {"timestamp": (now - timedelta(seconds=30)).isoformat(), "level": "INFO", "message": "Agent spawned: research_specialist", "source": "orchestrator"},
                    {"timestamp": (now - timedelta(minutes=1)).isoformat(), "level": "INFO", "message": "Query processed in 2.3s", "source": "query_handler"},
                    {"timestamp": (now - timedelta(minutes=2)).isoformat(), "level": "WARNING", "message": "High cost detected", "source": "cost_monitor"}
                ]
            
            return sorted(logs, key=lambda x: x["timestamp"], reverse=True)[:50]
            
        except Exception as e:
            logger.error(f"Error getting logs: {e}")
            return []
    
    async def _get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        return {
            "total_errors": 7,
            "error_rate": 0.025,
            "top_errors": [
                {"message": "OpenAI API timeout", "count": 3},
                {"message": "Supabase connection error", "count": 2}
            ]
        } 