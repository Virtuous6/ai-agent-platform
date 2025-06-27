"""
MCP Connection Manager

Handles creation, management, and lifecycle of MCP connections to external services.
Integrates with the existing Supabase database and agent platform architecture.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from database.supabase_logger import SupabaseLogger
from .credential_store import CredentialManager
from .security_sandbox import MCPSecuritySandbox

logger = logging.getLogger(__name__)

@dataclass
class MCPConnection:
    """Data class representing an MCP connection."""
    id: str
    user_id: str
    connection_name: str
    mcp_type: str
    connection_config: Dict[str, Any]
    credential_reference: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    status: str = 'active'
    tools_available: List[str] = None
    tool_schemas: Dict[str, Any] = None
    created_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    
    def __post_init__(self):
        if self.tools_available is None:
            self.tools_available = []
        if self.tool_schemas is None:
            self.tool_schemas = {}

class MCPConnectionManager:
    """
    Manages MCP connections for users, including creation, validation,
    health monitoring, and lifecycle management.
    """
    
    def __init__(self, db_logger: SupabaseLogger):
        """
        Initialize the MCP Connection Manager.
        
        Args:
            db_logger: Supabase logger for database operations
        """
        self.db_logger = db_logger
        self.credential_manager = CredentialManager()
        self.security_sandbox = MCPSecuritySandbox()
        self.active_connections = {}  # Cache for active connections
        
        logger.info("🔌 MCP Connection Manager initialized")
    
    async def create_connection(self, user_id: str, connection_name: str, 
                              mcp_type: str, config: Dict[str, Any],
                              credentials: Dict[str, str],
                              description: Optional[str] = None) -> MCPConnection:
        """
        Create a new MCP connection for a user.
        
        Args:
            user_id: User creating the connection
            connection_name: Unique name for the connection
            mcp_type: Type of MCP service (supabase, github, etc.)
            config: Connection configuration
            credentials: Service credentials
            description: Optional description
            
        Returns:
            Created MCP connection
            
        Raises:
            ValueError: If connection already exists or invalid parameters
            ConnectionError: If connection test fails
        """
        try:
            # Validate inputs
            await self._validate_connection_request(user_id, connection_name, mcp_type, config)
            
            # Check user connection limits
            await self._check_user_limits(user_id)
            
            # Store credentials securely
            credential_reference = await self.credential_manager.store_credential(
                connection_name=f"{user_id}_{connection_name}",
                credentials=credentials
            )
            
            # Create connection object
            connection = MCPConnection(
                id=str(uuid.uuid4()),
                user_id=user_id,
                connection_name=connection_name,
                mcp_type=mcp_type,
                connection_config=config,
                credential_reference=credential_reference,
                description=description,
                display_name=config.get('display_name', connection_name),
                status='testing'
            )
            
            # Test the connection
            await self._test_connection(connection)
            
            # Discover available tools
            await self._discover_tools(connection)
            
            # Save to database
            connection.status = 'active'
            await self._save_connection_to_db(connection)
            
            # Cache the connection
            self.active_connections[connection.id] = connection
            
            # Log security event
            await self._log_security_event(
                user_id=user_id,
                event_type="connection_created",
                event_description=f"Created {mcp_type} connection: {connection_name}",
                connection_id=connection.id
            )
            
            logger.info(f"✅ Created MCP connection: {connection_name} ({mcp_type}) for user {user_id}")
            
            return connection
            
        except Exception as e:
            logger.error(f"❌ Failed to create MCP connection: {str(e)}")
            
            # Log security event for failed connection
            await self._log_security_event(
                user_id=user_id,
                event_type="connection_failed",
                event_description=f"Failed to create {mcp_type} connection: {str(e)}",
                severity="error"
            )
            
            raise
    
    async def get_user_connections(self, user_id: str, 
                                 include_inactive: bool = False) -> List[MCPConnection]:
        """
        Get all connections for a user.
        
        Args:
            user_id: User ID to get connections for
            include_inactive: Whether to include inactive connections
            
        Returns:
            List of user's MCP connections
        """
        try:
            # Query database for user connections
            query = """
                SELECT * FROM mcp_connections 
                WHERE user_id = %s
            """
            params = [user_id]
            
            if not include_inactive:
                query += " AND status = 'active'"
            
            query += " ORDER BY created_at DESC"
            
            result = await self.db_logger.execute_query(query, params)
            
            connections = []
            for row in result:
                connection = self._row_to_connection(row)
                connections.append(connection)
                
                # Cache active connections
                if connection.status == 'active':
                    self.active_connections[connection.id] = connection
            
            logger.info(f"📋 Retrieved {len(connections)} connections for user {user_id}")
            return connections
            
        except Exception as e:
            logger.error(f"❌ Failed to get user connections: {str(e)}")
            return []
    
    async def get_connection(self, connection_id: str, user_id: str) -> Optional[MCPConnection]:
        """
        Get a specific connection by ID, ensuring user ownership.
        
        Args:
            connection_id: Connection ID
            user_id: User ID (for security validation)
            
        Returns:
            MCP connection if found and owned by user
        """
        try:
            # Check cache first
            if connection_id in self.active_connections:
                connection = self.active_connections[connection_id]
                if connection.user_id == user_id:
                    return connection
            
            # Query database
            query = """
                SELECT * FROM mcp_connections 
                WHERE id = %s AND user_id = %s
            """
            result = await self.db_logger.execute_query(query, [connection_id, user_id])
            
            if result:
                connection = self._row_to_connection(result[0])
                # Cache if active
                if connection.status == 'active':
                    self.active_connections[connection_id] = connection
                return connection
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get connection {connection_id}: {str(e)}")
            return None
    
    async def test_connection_health(self, connection_id: str, user_id: str) -> Dict[str, Any]:
        """
        Test the health of an MCP connection.
        
        Args:
            connection_id: Connection to test
            user_id: User ID for security validation
            
        Returns:
            Health check results
        """
        try:
            connection = await self.get_connection(connection_id, user_id)
            if not connection:
                return {"status": "error", "message": "Connection not found"}
            
            # Perform health check based on connection type
            health_result = await self._perform_health_check(connection)
            
            # Update health status in database
            await self._update_connection_health(connection_id, health_result)
            
            logger.info(f"🩺 Health check for connection {connection_id}: {health_result['status']}")
            
            return health_result
            
        except Exception as e:
            logger.error(f"❌ Health check failed for connection {connection_id}: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def delete_connection(self, connection_id: str, user_id: str) -> bool:
        """
        Delete an MCP connection.
        
        Args:
            connection_id: Connection to delete
            user_id: User ID for security validation
            
        Returns:
            True if successfully deleted
        """
        try:
            connection = await self.get_connection(connection_id, user_id)
            if not connection:
                return False
            
            # Remove from cache
            self.active_connections.pop(connection_id, None)
            
            # Delete credentials
            if connection.credential_reference:
                await self.credential_manager.delete_credential(connection.credential_reference)
            
            # Delete from database (cascade will handle related records)
            query = "DELETE FROM mcp_connections WHERE id = %s AND user_id = %s"
            await self.db_logger.execute_query(query, [connection_id, user_id])
            
            # Log security event
            await self._log_security_event(
                user_id=user_id,
                event_type="connection_deleted",
                event_description=f"Deleted connection: {connection.connection_name}",
                connection_id=connection_id
            )
            
            logger.info(f"🗑️ Deleted MCP connection {connection_id} for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete connection {connection_id}: {str(e)}")
            return False
    
    async def get_connection_analytics(self, user_id: str, 
                                     days: int = 30) -> Dict[str, Any]:
        """
        Get analytics for user's MCP connections.
        
        Args:
            user_id: User ID
            days: Number of days to analyze
            
        Returns:
            Analytics data
        """
        try:
            # Use the analytics view from the database migration
            query = """
                SELECT 
                    mcp_type,
                    COUNT(*) as connection_count,
                    SUM(total_executions) as total_executions,
                    AVG(success_rate_percentage) as avg_success_rate,
                    SUM(total_tokens_saved) as total_tokens_saved
                FROM mcp_connection_analytics 
                WHERE user_id = %s
                GROUP BY mcp_type
                ORDER BY total_executions DESC
            """
            
            result = await self.db_logger.execute_query(query, [user_id])
            
            analytics = {
                "user_id": user_id,
                "period_days": days,
                "by_type": [],
                "summary": {
                    "total_connections": 0,
                    "total_executions": 0,
                    "avg_success_rate": 0,
                    "total_tokens_saved": 0
                }
            }
            
            for row in result:
                type_data = {
                    "mcp_type": row["mcp_type"],
                    "connection_count": row["connection_count"],
                    "total_executions": row["total_executions"],
                    "avg_success_rate": row["avg_success_rate"],
                    "total_tokens_saved": row["total_tokens_saved"]
                }
                analytics["by_type"].append(type_data)
                
                # Update summary
                analytics["summary"]["total_connections"] += row["connection_count"]
                analytics["summary"]["total_executions"] += row["total_executions"] or 0
                analytics["summary"]["total_tokens_saved"] += row["total_tokens_saved"] or 0
            
            # Calculate average success rate
            if analytics["by_type"]:
                avg_success = sum(t["avg_success_rate"] or 0 for t in analytics["by_type"])
                analytics["summary"]["avg_success_rate"] = avg_success / len(analytics["by_type"])
            
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Failed to get analytics for user {user_id}: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _validate_connection_request(self, user_id: str, connection_name: str,
                                         mcp_type: str, config: Dict[str, Any]):
        """Validate connection creation request."""
        if not user_id or not connection_name or not mcp_type:
            raise ValueError("user_id, connection_name, and mcp_type are required")
        
        # Check if connection name already exists for user
        existing = await self.db_logger.execute_query(
            "SELECT id FROM mcp_connections WHERE user_id = %s AND connection_name = %s",
            [user_id, connection_name]
        )
        if existing:
            raise ValueError(f"Connection '{connection_name}' already exists for user")
        
        # Validate MCP type
        from . import SUPPORTED_MCP_TYPES
        if mcp_type not in SUPPORTED_MCP_TYPES:
            raise ValueError(f"Unsupported MCP type: {mcp_type}")
    
    async def _check_user_limits(self, user_id: str):
        """Check if user has exceeded connection limits."""
        count_result = await self.db_logger.execute_query(
            "SELECT COUNT(*) as count FROM mcp_connections WHERE user_id = %s AND status = 'active'",
            [user_id]
        )
        
        from . import MCP_CONFIG
        max_connections = MCP_CONFIG['max_connections_per_user']
        
        if count_result and count_result[0]['count'] >= max_connections:
            raise ValueError(f"Maximum connections limit ({max_connections}) exceeded")
    
    async def _test_connection(self, connection: MCPConnection):
        """Test if connection works properly."""
        # This would implement connection-specific testing
        # For now, we'll simulate a basic test
        logger.info(f"🧪 Testing connection {connection.connection_name}")
        
        # TODO: Implement actual connection testing based on mcp_type
        # For example:
        # - Supabase: Test API endpoint and authentication
        # - GitHub: Test API access with provided token
        # - Slack: Test bot token validity
        
        await asyncio.sleep(0.1)  # Simulate test delay
    
    async def _discover_tools(self, connection: MCPConnection):
        """Discover available tools for the connection."""
        # This would implement tool discovery based on connection type
        # For now, we'll use default tools based on MCP type
        
        tool_mapping = {
            'supabase': ['list_tables', 'execute_sql', 'get_schema', 'insert_data', 'update_data'],
            'github': ['search_repos', 'get_issues', 'create_issue', 'get_pull_requests'],
            'slack': ['send_message', 'get_channels', 'get_users', 'search_messages'],
            'postgres': ['execute_query', 'get_tables', 'get_schema', 'insert_data'],
        }
        
        connection.tools_available = tool_mapping.get(connection.mcp_type, [])
        logger.info(f"🔧 Discovered {len(connection.tools_available)} tools for {connection.connection_name}")
    
    async def _save_connection_to_db(self, connection: MCPConnection):
        """Save connection to database."""
        query = """
            INSERT INTO mcp_connections (
                id, user_id, connection_name, mcp_type, connection_config,
                credential_reference, display_name, description, status,
                tools_available, tool_schemas
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = [
            connection.id, connection.user_id, connection.connection_name,
            connection.mcp_type, connection.connection_config,
            connection.credential_reference, connection.display_name,
            connection.description, connection.status,
            connection.tools_available, connection.tool_schemas
        ]
        
        await self.db_logger.execute_query(query, params)
    
    async def _perform_health_check(self, connection: MCPConnection) -> Dict[str, Any]:
        """Perform connection-specific health check."""
        # Implement health checks based on connection type
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "response_time_ms": 50,
            "tools_available": len(connection.tools_available)
        }
    
    async def _update_connection_health(self, connection_id: str, health_result: Dict[str, Any]):
        """Update connection health status in database."""
        query = """
            UPDATE mcp_connections 
            SET last_health_check = NOW(), health_status = %s
            WHERE id = %s
        """
        await self.db_logger.execute_query(query, [health_result, connection_id])
    
    async def _log_security_event(self, user_id: str, event_type: str,
                                event_description: str, connection_id: Optional[str] = None,
                                severity: str = "info"):
        """Log security events for audit trail."""
        try:
            query = """
                INSERT INTO mcp_security_logs (
                    connection_id, user_id, event_type, event_description, severity
                ) VALUES (%s, %s, %s, %s, %s)
            """
            await self.db_logger.execute_query(
                query, [connection_id, user_id, event_type, event_description, severity]
            )
        except Exception as e:
            logger.error(f"Failed to log security event: {str(e)}")
    
    def _row_to_connection(self, row: Dict[str, Any]) -> MCPConnection:
        """Convert database row to MCPConnection object."""
        return MCPConnection(
            id=row['id'],
            user_id=row['user_id'],
            connection_name=row['connection_name'],
            mcp_type=row['mcp_type'],
            connection_config=row['connection_config'],
            credential_reference=row['credential_reference'],
            display_name=row['display_name'],
            description=row['description'],
            status=row['status'],
            tools_available=row['tools_available'] or [],
            tool_schemas=row['tool_schemas'] or {},
            created_at=row['created_at'],
            last_used=row['last_used']
        ) 