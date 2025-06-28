#!/usr/bin/env python3
"""
MCP Tool Registry

Universal tool registry that makes MCP tools available to all agents.
Features simple tool patterns with platform security and user management.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

try:
    from supabase.supabase_logger import SupabaseLogger
except ImportError:
    # Fallback if supabase logger doesn't exist
    class SupabaseLogger:
        async def log_event(self, *args, **kwargs):
            pass

try:
    from .mcp_client import mcp_client, MCPProtocolClient
except ImportError:
    # Fallback if mcp_client doesn't exist
    mcp_client = None
    MCPProtocolClient = None

from .security_sandbox import MCPSecuritySandbox

logger = logging.getLogger(__name__)

@dataclass
class ToolDescriptor:
    """Universal tool descriptor for MCP tools."""
    tool_id: str
    tool_name: str
    description: str
    parameters: Dict[str, Any]
    mcp_type: str
    connection_id: Optional[str] = None
    user_id: Optional[str] = None
    requires_auth: bool = True

class MCPToolRegistry:
    """
    Universal MCP Tool Registry
    
    Makes MCP tools available to all agents with:
    - User-specific permissions and credentials
    - Security sandboxing
    - Usage tracking
    - Cost monitoring
    """
    
    def __init__(self, supabase_logger: Optional[SupabaseLogger] = None,
                 security_sandbox: Optional[MCPSecuritySandbox] = None):
        """Initialize the MCP tool registry."""
        self.supabase_logger = supabase_logger or SupabaseLogger()
        self.security_sandbox = security_sandbox or MCPSecuritySandbox()
        
        # Universal tools available to all agents
        self.universal_tools: Dict[str, ToolDescriptor] = {}
        
        # User-specific tools (maintains existing security model)
        self.user_tools: Dict[str, Dict[str, ToolDescriptor]] = {}  # user_id -> tools
        
        # Active MCP sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> session_info
        
        logger.info("🔧 MCPToolRegistry initialized with universal + user-specific tools")

    # ============================================================================
    # UNIVERSAL TOOLS
    # ============================================================================
    
    async def register_universal_tool(self, tool_name: str, description: str,
                                    mcp_type: str, function: Callable,
                                    parameters: Dict[str, Any] = None) -> str:
        """
        Register a universal tool available to all agents.
        
        Args:
            tool_name: Name of the tool
            description: Tool description
            mcp_type: Type of MCP (web_search, calculate, etc.)
            function: Tool execution function
            parameters: Tool parameter schema
            
        Returns:
            tool_id for the registered tool
        """
        tool_id = f"universal_{tool_name}"
        
        tool = ToolDescriptor(
            tool_id=tool_id,
            tool_name=tool_name,
            description=description,
            parameters=parameters or {},
            mcp_type=mcp_type,
            requires_auth=False  # Universal tools don't require user auth
        )
        
        self.universal_tools[tool_id] = tool
        
        # Register the actual function with the security sandbox
        self.security_sandbox.register_tool_function(tool_id, function)
        
        logger.info(f"🌐 Registered universal tool: {tool_name}")
        return tool_id

    async def register_user_tool(self, user_id: str, tool_name: str,
                               description: str, mcp_type: str,
                               connection_id: str, function: Callable,
                               parameters: Dict[str, Any] = None) -> str:
        """
        Register a user-specific tool (maintains existing security model).
        
        Args:
            user_id: User who owns this tool
            tool_name: Name of the tool
            description: Tool description
            mcp_type: Type of MCP
            connection_id: MCP connection ID
            function: Tool execution function
            parameters: Tool parameter schema
            
        Returns:
            tool_id for the registered tool
        """
        tool_id = f"user_{user_id}_{tool_name}"
        
        tool = ToolDescriptor(
            tool_id=tool_id,
            tool_name=tool_name,
            description=description,
            parameters=parameters or {},
            mcp_type=mcp_type,
            connection_id=connection_id,
            user_id=user_id,
            requires_auth=True
        )
        
        if user_id not in self.user_tools:
            self.user_tools[user_id] = {}
        
        self.user_tools[user_id][tool_id] = tool
        
        # Register the actual function with the security sandbox
        self.security_sandbox.register_tool_function(tool_id, function)
        
        logger.info(f"👤 Registered user tool: {tool_name} for {user_id}")
        return tool_id

    async def get_available_tools(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all available tools for a user (universal + user-specific).
        
        This is what agents call to get their tool list.
        
        Args:
            user_id: User ID (optional for universal tools only)
            
        Returns:
            List of tool descriptors
        """
        tools = []
        
        # Add universal tools (available to everyone)
        for tool in self.universal_tools.values():
            tools.append({
                'tool_id': tool.tool_id,
                'tool_name': tool.tool_name,
                'description': tool.description,
                'parameters': tool.parameters,
                'mcp_type': tool.mcp_type,
                'universal': True,
                'requires_auth': False
            })
        
        # Add user-specific tools
        if user_id and user_id in self.user_tools:
            for tool in self.user_tools[user_id].values():
                tools.append({
                    'tool_id': tool.tool_id,
                    'tool_name': tool.tool_name,
                    'description': tool.description,
                    'parameters': tool.parameters,
                    'mcp_type': tool.mcp_type,
                    'connection_id': tool.connection_id,
                    'universal': False,
                    'requires_auth': True
                })
        
        logger.debug(f"📋 Available tools for {user_id or 'universal'}: {len(tools)} tools")
        return tools

    async def execute_tool(self, tool_id: str, parameters: Dict[str, Any],
                         user_id: Optional[str] = None, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a tool with security and tracking.
        
        This is the main execution method that agents call.
        
        Args:
            tool_id: Tool to execute
            parameters: Tool parameters
            user_id: User requesting execution
            agent_id: Agent making the request
            
        Returns:
            Tool execution result
        """
        start_time = datetime.utcnow()
        
        try:
            # Find the tool
            tool = await self._find_tool(tool_id, user_id)
            if not tool:
                return {
                    "success": False,
                    "error": f"Tool '{tool_id}' not found",
                    "tool_name": tool_id
                }
            
            # Security check
            if tool.requires_auth and not user_id:
                return {
                    "success": False,
                    "error": "Authentication required for this tool",
                    "tool_name": tool.tool_name
                }
            
            # Execute with security sandbox
            result = await self.security_sandbox.sandbox_execute(
                tool_id=tool_id,
                tool_name=tool.tool_name,
                parameters=parameters,
                user_id=user_id,
                timeout=30
            )
            
            # Track usage
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._track_tool_usage(tool, parameters, result, user_id, agent_id, processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_id
            }

    # ============================================================================
    # MCP CONNECTION MANAGEMENT
    # ============================================================================
    
    async def connect_mcp(self, mcp_id: str, name: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to an MCP server and register its tools."""
        try:
            # Connect to MCP server
            session = await mcp_client.connect_to_mcp_server(
                server_url=credentials.get('url'),
                credentials=credentials
            )
            
            if not session:
                return {"success": False, "error": "Failed to connect to MCP server"}
            
            # Store session info
            self.active_sessions[session.session_id] = {
                "mcp_id": mcp_id,
                "name": name,
                "session": session,
                "credentials": credentials,
                "connected_at": datetime.utcnow()
            }
            
            logger.info(f"✅ Connected to MCP: {name} with {len(session.tools)} tools")
            
            return {
                "success": True,
                "session_id": session.session_id,
                "tools_discovered": len(session.tools)
            }
            
        except Exception as e:
            logger.error(f"Failed to connect MCP {mcp_id}: {e}")
            return {"success": False, "error": str(e)}

    async def disconnect_mcp(self, session_id: str):
        """Disconnect from an MCP server."""
        if session_id in self.active_sessions:
            await mcp_client.close_session(session_id)
            del self.active_sessions[session_id]
            logger.info(f"🔌 Disconnected MCP session: {session_id}")

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    async def _find_tool(self, tool_id: str, user_id: Optional[str] = None) -> Optional[ToolDescriptor]:
        """Find a tool by ID."""
        # Check universal tools first
        if tool_id in self.universal_tools:
            return self.universal_tools[tool_id]
        
        # Check user-specific tools
        if user_id and user_id in self.user_tools and tool_id in self.user_tools[user_id]:
            return self.user_tools[user_id][tool_id]
        
        return None

    async def _track_tool_usage(self, tool: ToolDescriptor, parameters: Dict[str, Any],
                              result: Dict[str, Any], user_id: Optional[str],
                              agent_id: Optional[str], processing_time_ms: float):
        """Track tool usage for analytics."""
        try:
            await self.supabase_logger.log_event(
                event_type="mcp_tool_executed",
                event_data={
                    "tool_id": tool.tool_id,
                    "tool_name": tool.tool_name,
                    "mcp_type": tool.mcp_type,
                    "success": result.get("success", False),
                    "parameters": parameters,
                    "processing_time_ms": processing_time_ms,
                    "universal_tool": not tool.requires_auth,
                    "agent_id": agent_id
                },
                user_id=user_id
            )
        except Exception as e:
            logger.warning(f"Failed to track tool usage: {e}")

    async def get_tool_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            "universal_tools": len(self.universal_tools),
            "user_tools": len(self.user_tools.get(user_id, {})) if user_id else 0,
            "active_sessions": len(self.active_sessions),
            "total_tools": len(self.universal_tools) + (len(self.user_tools.get(user_id, {})) if user_id else 0)
        }

    async def close(self):
        """Close all MCP sessions and cleanup."""
        for session_id in list(self.active_sessions.keys()):
            await self.disconnect_mcp(session_id)
        
        logger.info("🔌 MCPToolRegistry closed") 