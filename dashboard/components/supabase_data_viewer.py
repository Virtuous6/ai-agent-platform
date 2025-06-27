"""
Supabase Data Viewer Component

Displays real data from Supabase tables for monitoring the AI agent platform.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SupabaseDataViewer:
    """Fetches and displays real data from Supabase tables."""
    
    def __init__(self, db_logger=None):
        """Initialize the Supabase data viewer."""
        self.db_logger = db_logger
        logger.info("Supabase Data Viewer initialized")
    
    async def get_messages_data(self, limit: int = 20) -> Dict[str, Any]:
        """Get recent messages from Supabase."""
        try:
            if not self.db_logger:
                return {"messages": [], "total_count": 0, "error": "No database connection"}
            
            # Get recent messages
            result = self.db_logger.client.table("messages") \
                .select("*") \
                .order("timestamp", desc=True) \
                .limit(limit) \
                .execute()
            
            messages = result.data if result.data else []
            
            # Get total count
            count_result = self.db_logger.client.table("messages") \
                .select("id", count="exact") \
                .execute()
            
            total_count = count_result.count if hasattr(count_result, 'count') else len(messages)
            
            # Process messages for display
            processed_messages = []
            for msg in messages:
                processed_msg = {
                    "id": msg.get("id", "unknown")[:8],
                    "content": self._truncate_text(str(msg.get("content", "")), 50),
                    "role": msg.get("role", "unknown"),
                    "timestamp": self._format_timestamp(msg.get("timestamp", "")),
                    "user_id": msg.get("user_id", "unknown")[:12],
                    "conversation_id": msg.get("conversation_id", "")[:8],
                    "error": "ERROR" if msg.get("error_message") else "",
                    "tokens": msg.get("token_count", 0) or 0
                }
                processed_messages.append(processed_msg)
            
            return {
                "messages": processed_messages,
                "total_count": total_count,
                "success_rate": self._calculate_success_rate(messages),
                "avg_tokens": self._calculate_avg_tokens(messages)
            }
            
        except Exception as e:
            logger.error(f"Error fetching messages data: {e}")
            return {"messages": [], "total_count": 0, "error": str(e)}
    
    async def get_workflow_runs_data(self, limit: int = 15) -> Dict[str, Any]:
        """Get workflow runs from runbook_executions table."""
        try:
            if not self.db_logger:
                return {"workflows": [], "total_count": 0, "error": "No database connection"}
            
            # Get recent runbook executions
            result = self.db_logger.client.table("runbook_executions") \
                .select("*") \
                .order("started_at", desc=True) \
                .limit(limit) \
                .execute()
            
            workflows = result.data if result.data else []
            
            # Process workflows for display
            processed_workflows = []
            for wf in workflows:
                duration = self._calculate_duration(
                    wf.get("started_at"), 
                    wf.get("completed_at")
                )
                
                processed_wf = {
                    "id": wf.get("id", "unknown")[:8],
                    "runbook_name": wf.get("runbook_name", "unknown"),
                    "status": wf.get("status", "unknown"),
                    "user_id": wf.get("user_id", "unknown")[:12],
                    "started_at": self._format_timestamp(wf.get("started_at", "")),
                    "duration": duration,
                    "progress": f"{wf.get('progress_percentage', 0):.0f}%",
                    "agents_used": len(wf.get("agents_used", [])),
                    "tools_used": len(wf.get("tools_used", [])),
                    "cost": f"${wf.get('estimated_cost', 0):.4f}",
                    "tokens": wf.get("total_tokens", 0) or 0,
                    "error": wf.get("error_message", "")[:30] if wf.get("error_message") else ""
                }
                processed_workflows.append(processed_wf)
            
            # Calculate statistics
            total_workflows = len(workflows)
            completed = len([w for w in workflows if w.get("status") == "completed"])
            failed = len([w for w in workflows if w.get("status") == "failed"])
            
            return {
                "workflows": processed_workflows,
                "total_count": total_workflows,
                "completed_count": completed,
                "failed_count": failed,
                "success_rate": (completed / total_workflows * 100) if total_workflows > 0 else 0,
                "avg_cost": sum(w.get("estimated_cost", 0) for w in workflows) / total_workflows if total_workflows > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error fetching workflow runs data: {e}")
            return {"workflows": [], "total_count": 0, "error": str(e)}
    
    async def get_mcp_connections_data(self) -> Dict[str, Any]:
        """Get MCP connections from Supabase."""
        try:
            if not self.db_logger:
                return {"connections": [], "total_count": 0, "error": "No database connection"}
            
            # Get MCP connections
            result = self.db_logger.client.table("mcp_connections") \
                .select("*") \
                .order("created_at", desc=True) \
                .execute()
            
            connections = result.data if result.data else []
            
            # Process connections for display
            processed_connections = []
            for conn in connections:
                processed_conn = {
                    "id": conn.get("id", "unknown")[:8],
                    "name": conn.get("connection_name", "unknown"),
                    "type": conn.get("mcp_type", "unknown"),
                    "display_name": conn.get("display_name", conn.get("connection_name", "unknown")),
                    "status": conn.get("status", "unknown"),
                    "user_id": conn.get("user_id", "unknown")[:12],
                    "last_used": self._format_timestamp(conn.get("last_used", "")),
                    "total_executions": conn.get("total_executions", 0),
                    "success_rate": self._calculate_connection_success_rate(conn),
                    "tools_count": len(conn.get("tools_available", [])),
                    "health_status": conn.get("health_status", {}).get("status", "unknown"),
                    "monthly_cost": f"${conn.get('estimated_monthly_cost', 0):.2f}"
                }
                processed_connections.append(processed_conn)
            
            # Calculate statistics
            active_connections = len([c for c in connections if c.get("status") == "active"])
            connection_types = {}
            for conn in connections:
                conn_type = conn.get("mcp_type", "unknown")
                connection_types[conn_type] = connection_types.get(conn_type, 0) + 1
            
            return {
                "connections": processed_connections,
                "total_count": len(connections),
                "active_count": active_connections,
                "connection_types": connection_types,
                "total_executions": sum(c.get("total_executions", 0) for c in connections),
                "avg_success_rate": self._calculate_avg_connection_success_rate(connections)
            }
            
        except Exception as e:
            logger.error(f"Error fetching MCP connections data: {e}")
            return {"connections": [], "total_count": 0, "error": str(e)}
    
    async def get_conversations_data(self, limit: int = 15) -> Dict[str, Any]:
        """Get recent conversations."""
        try:
            if not self.db_logger:
                return {"conversations": [], "total_count": 0, "error": "No database connection"}
            
            result = self.db_logger.client.table("conversations") \
                .select("*") \
                .order("started_at", desc=True) \
                .limit(limit) \
                .execute()
            
            conversations = result.data if result.data else []
            
            processed_conversations = []
            for conv in conversations:
                processed_conv = {
                    "id": conv.get("id", "unknown")[:8],
                    "user_id": conv.get("user_id", "unknown")[:12],
                    "channel_id": conv.get("channel_id", "unknown")[:12],
                    "status": conv.get("status", "unknown"),
                    "started_at": self._format_timestamp(conv.get("started_at", "")),
                    "ended_at": self._format_timestamp(conv.get("ended_at", "")),
                    "total_messages": conv.get("total_messages", 0),
                    "total_tokens": conv.get("total_tokens", 0),
                    "total_cost": f"${conv.get('total_cost', 0):.4f}" if conv.get("total_cost") else "$0.0000"
                }
                processed_conversations.append(processed_conv)
            
            return {
                "conversations": processed_conversations,
                "total_count": len(conversations),
                "active_count": len([c for c in conversations if c.get("status") == "active"]),
                "avg_messages": sum(c.get("total_messages", 0) for c in conversations) / len(conversations) if conversations else 0
            }
            
        except Exception as e:
            logger.error(f"Error fetching conversations data: {e}")
            return {"conversations": [], "total_count": 0, "error": str(e)}
    
    async def get_cost_analytics_data(self) -> Dict[str, Any]:
        """Get cost analytics from various tables."""
        try:
            if not self.db_logger:
                return {"error": "No database connection"}
            
            # Get today's cost from daily_token_summary
            today = str(datetime.utcnow().date())
            daily_result = self.db_logger.client.table("daily_token_summary") \
                .select("*") \
                .eq("date", today) \
                .execute()
            
            today_cost = 0.0
            today_tokens = 0
            if daily_result.data and daily_result.data[0]:
                today_cost = daily_result.data[0].get("total_cost", 0.0)
                today_tokens = daily_result.data[0].get("total_tokens", 0)
            
            # Get last 7 days cost
            week_ago = str((datetime.utcnow() - timedelta(days=7)).date())
            weekly_result = self.db_logger.client.table("daily_token_summary") \
                .select("*") \
                .gte("date", week_ago) \
                .execute()
            
            weekly_cost = sum(r.get("total_cost", 0) for r in (weekly_result.data or []))
            weekly_tokens = sum(r.get("total_tokens", 0) for r in (weekly_result.data or []))
            
            # Get cost optimizations
            optimizations_result = self.db_logger.client.table("cost_optimizations") \
                .select("*") \
                .eq("status", "identified") \
                .order("potential_savings", desc=True) \
                .limit(5) \
                .execute()
            
            optimizations = []
            for opt in (optimizations_result.data or []):
                optimizations.append({
                    "title": opt.get("title", "Unknown")[:40],
                    "savings": f"${opt.get('potential_savings', 0):.2f}",
                    "complexity": opt.get("implementation_complexity", "unknown"),
                    "priority": opt.get("priority", 3)
                })
            
            return {
                "today_cost": today_cost,
                "today_tokens": today_tokens,
                "weekly_cost": weekly_cost,
                "weekly_tokens": weekly_tokens,
                "monthly_projection": weekly_cost * 4.3,
                "cost_per_token": weekly_cost / weekly_tokens if weekly_tokens > 0 else 0,
                "optimizations": optimizations,
                "efficiency_score": 0.78,  # Could calculate from actual data
                "cost_trend": "stable"  # Could calculate from week-over-week comparison
            }
            
        except Exception as e:
            logger.error(f"Error fetching cost analytics: {e}")
            return {"error": str(e)}
    
    async def add_mcp_connection(self, connection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new MCP connection."""
        try:
            if not self.db_logger:
                return {"success": False, "error": "No database connection"}
            
            # Validate required fields
            required_fields = ["connection_name", "mcp_type", "user_id"]
            for field in required_fields:
                if field not in connection_data:
                    return {"success": False, "error": f"Missing required field: {field}"}
            
            # Insert new connection
            result = self.db_logger.client.table("mcp_connections") \
                .insert(connection_data) \
                .execute()
            
            if result.data:
                return {
                    "success": True, 
                    "connection_id": result.data[0].get("id"),
                    "message": f"MCP connection '{connection_data['connection_name']}' created successfully"
                }
            else:
                return {"success": False, "error": "Failed to create connection"}
            
        except Exception as e:
            logger.error(f"Error creating MCP connection: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_available_mcp_types(self) -> List[Dict[str, Any]]:
        """Get available MCP types from run cards."""
        try:
            if not self.db_logger:
                return []
            
            result = self.db_logger.client.table("mcp_run_cards") \
                .select("*") \
                .eq("is_public", True) \
                .order("popularity_score", desc=True) \
                .execute()
            
            cards = []
            for card in (result.data or []):
                cards.append({
                    "card_name": card.get("card_name", ""),
                    "display_name": card.get("display_name", ""),
                    "mcp_type": card.get("mcp_type", ""),
                    "description": card.get("description", ""),
                    "required_credentials": card.get("required_credentials", []),
                    "available_tools": card.get("available_tools", []),
                    "popularity": card.get("popularity_score", 0)
                })
            
            return cards
            
        except Exception as e:
            logger.error(f"Error fetching MCP types: {e}")
            return []
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to max length."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def _format_timestamp(self, timestamp: str) -> str:
        """Format timestamp for display."""
        if not timestamp:
            return "N/A"
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime("%m/%d %H:%M")
        except:
            return str(timestamp)[:16]
    
    def _calculate_success_rate(self, messages: List[Dict]) -> float:
        """Calculate success rate for messages."""
        if not messages:
            return 0.0
        errors = len([m for m in messages if m.get("error_message")])
        return ((len(messages) - errors) / len(messages)) * 100
    
    def _calculate_avg_tokens(self, messages: List[Dict]) -> float:
        """Calculate average tokens per message."""
        if not messages:
            return 0.0
        total_tokens = sum(m.get("token_count", 0) or 0 for m in messages)
        return total_tokens / len(messages)
    
    def _calculate_duration(self, start_time: str, end_time: str) -> str:
        """Calculate duration between timestamps."""
        if not start_time or not end_time:
            return "N/A"
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            duration = end - start
            total_seconds = int(duration.total_seconds())
            
            if total_seconds < 60:
                return f"{total_seconds}s"
            elif total_seconds < 3600:
                return f"{total_seconds // 60}m {total_seconds % 60}s"
            else:
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                return f"{hours}h {minutes}m"
        except:
            return "N/A"
    
    def _calculate_connection_success_rate(self, connection: Dict) -> str:
        """Calculate success rate for a connection."""
        total = connection.get("total_executions", 0)
        successful = connection.get("successful_executions", 0)
        if total == 0:
            return "N/A"
        return f"{(successful / total * 100):.1f}%"
    
    def _calculate_avg_connection_success_rate(self, connections: List[Dict]) -> float:
        """Calculate average success rate across connections."""
        if not connections:
            return 0.0
        
        rates = []
        for conn in connections:
            total = conn.get("total_executions", 0)
            successful = conn.get("successful_executions", 0)
            if total > 0:
                rates.append(successful / total)
        
        return (sum(rates) / len(rates) * 100) if rates else 0.0 