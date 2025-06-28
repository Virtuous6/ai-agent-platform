#!/usr/bin/env python3
"""
Universal MCP Tools Integration
==============================

This module properly integrates MCP tools with the agent system.
Makes it easy to register tools and use them from agents.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json

from tools.standard_library import get_standard_tools, execute_tool as execute_standard_tool
from storage.supabase import SupabaseLogger

logger = logging.getLogger(__name__)

class UniversalMCPToolRegistry:
    """
    Central registry for all tools - both standard library and MCP tools.
    Stores tool metadata in Supabase for persistence.
    """
    
    def __init__(self, supabase_logger: Optional[SupabaseLogger] = None):
        """Initialize the universal tool registry."""
        self.supabase = supabase_logger or SupabaseLogger()
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_functions: Dict[str, Callable] = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the registry and load tools."""
        if self._initialized:
            return
            
        try:
            # Load standard library tools first
            await self._load_standard_tools()
            
            # Load tools from Supabase
            await self._load_supabase_tools()
            
            # Try to load MCP tools if available
            await self._load_mcp_tools()
            
            self._initialized = True
            logger.info(f"✅ Tool registry initialized with {len(self.tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize tool registry: {e}")
    
    async def _load_standard_tools(self):
        """Load tools from the standard library."""
        try:
            standard_tools = get_standard_tools()
            
            for tool in standard_tools:
                tool_key = f"standard:{tool['name']}"
                self.tools[tool_key] = {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                    "source": "standard_library",
                    "enabled": True
                }
                self.tool_functions[tool_key] = tool["function"]
                
            logger.info(f"Loaded {len(standard_tools)} standard tools")
            
        except Exception as e:
            logger.error(f"Failed to load standard tools: {e}")
    
    async def _load_supabase_tools(self):
        """Load tools from Supabase 'tools' table."""
        try:
            if not self.supabase or not hasattr(self.supabase, 'client'):
                return
                
            # Query tools from Supabase
            result = self.supabase.client.table("tools")\
                .select("*")\
                .eq("is_active", True)\
                .execute()
            
            if result.data:
                for tool_data in result.data:
                    tool_type = tool_data.get('tool_type', 'custom')
                    tool_key = f"{tool_type}:{tool_data['tool_name']}"
                    
                    self.tools[tool_key] = {
                        "name": tool_data["tool_name"],
                        "description": tool_data["description"],
                        "parameters": tool_data.get("tool_schema", {}),
                        "source": tool_type,
                        "enabled": True,
                        "category": tool_data.get("category", "general"),
                        "connection_template": tool_data.get("connection_template", {})
                    }
                    
                logger.info(f"Loaded {len(result.data)} tools from Supabase")
                
        except Exception as e:
            logger.warning(f"Failed to load Supabase tools: {e}")
    
    async def _load_mcp_tools(self):
        """Load MCP tools if MCP is properly set up."""
        try:
            # Try to import MCP registry
            from agent_mcp.registry import MCPRegistry
            mcp_registry = MCPRegistry()
            
            # Get user's MCP connections
            if self.supabase and hasattr(self.supabase, 'client'):
                connections = self.supabase.client.table("mcp_connections")\
                    .select("*")\
                    .eq("status", "active")\
                    .execute()
                
                if connections.data:
                    for conn in connections.data:
                        # Load tools from each connection
                        # This would require actual MCP connection logic
                        logger.info(f"Found MCP connection: {conn['service_name']}")
                
        except ImportError:
            logger.debug("MCP not available, skipping MCP tool loading")
        except Exception as e:
            logger.warning(f"Failed to load MCP tools: {e}")
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], 
                          user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a tool by name, regardless of its source.
        
        Args:
            tool_name: Name of the tool (can be with or without prefix)
            parameters: Tool parameters
            user_id: Optional user ID for tracking
            
        Returns:
            Tool execution result
        """
        try:
            # Find the tool (check with and without prefixes)
            tool_key = None
            tool_info = None
            
            # Direct match
            if tool_name in self.tools:
                tool_key = tool_name
                tool_info = self.tools[tool_name]
            else:
                # Search without prefix
                for key, info in self.tools.items():
                    if info["name"] == tool_name or key.endswith(f":{tool_name}"):
                        tool_key = key
                        tool_info = info
                        break
            
            if not tool_info:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not found",
                    "available_tools": list(self.get_tool_names())
                }
            
            # Execute based on source
            source = tool_info["source"]
            
            if source == "standard_library":
                # Use standard library execution
                if tool_key in self.tool_functions:
                    # Call function directly with parameters
                    func = self.tool_functions[tool_key]
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: func(**parameters)
                    )
                else:
                    # Use standard library execute_tool
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: execute_standard_tool(tool_info["name"], **parameters)
                    )
                    
            elif source == "mcp_tool":
                # Use MCP execution
                result = await self._execute_mcp_tool(tool_key, tool_info, user_id, **parameters)
                
            elif source == "web_search":
                # Execute web search tool
                result = await self._execute_web_search(tool_info, **parameters)
                
            else:
                # Execute as generic tool
                result = await self._execute_generic_tool(tool_info, **parameters)
            
            # Log tool usage
            await self._log_tool_usage(tool_name, parameters, result, user_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed for '{tool_name}': {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_mcp_tool(self, tool_key: str, tool_info: Dict[str, Any], 
                               user_id: Optional[str], **kwargs) -> Dict[str, Any]:
        """Execute an MCP tool."""
        try:
            # Get user's connection for this tool
            connection_template = tool_info.get("connection_template", {})
            
            # TODO: Implement actual MCP execution
            # This would require:
            # 1. Getting user's connection credentials
            # 2. Establishing MCP connection
            # 3. Calling the tool through MCP protocol
            
            return {
                "success": False,
                "error": "MCP tool execution not yet implemented",
                "tool_info": tool_info
            }
            
        except Exception as e:
            return {"success": False, "error": f"MCP execution failed: {e}"}
    
    async def _execute_web_search(self, tool_info: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute web search tool."""
        try:
            # Import web search if available
            from agent_mcp.tools.web_search import search_web
            
            query = kwargs.get("query", "")
            if not query:
                return {"success": False, "error": "No search query provided"}
            
            results = await search_web(query)
            return {
                "success": True,
                "results": results,
                "source": "web_search"
            }
            
        except ImportError:
            return {"success": False, "error": "Web search not available"}
        except Exception as e:
            return {"success": False, "error": f"Web search failed: {e}"}
    
    async def _execute_generic_tool(self, tool_info: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute a generic tool."""
        return {
            "success": False,
            "error": f"Generic tool execution not implemented for {tool_info['name']}"
        }
    
    async def _log_tool_usage(self, tool_name: str, parameters: Dict[str, Any], 
                             result: Dict[str, Any], user_id: Optional[str]):
        """Log tool usage to Supabase."""
        try:
            if not self.supabase or not hasattr(self.supabase, 'client'):
                return
                
            # Log to tool_usage_analytics table
            self.supabase.client.table("tool_usage_analytics").insert({
                "tool_name": tool_name,
                "user_id": user_id or "system",
                "execution_time_ms": result.get("execution_time_ms", 0),
                "success": result.get("success", False),
                "error_message": result.get("error"),
                "input_parameters": parameters,
                "output_data": result,
                "timestamp": datetime.utcnow().isoformat()
            }).execute()
            
        except Exception as e:
            logger.debug(f"Failed to log tool usage: {e}")
    
    def get_tools_for_agent(self, agent_specialty: str = None) -> List[Dict[str, Any]]:
        """Get tools suitable for a specific agent specialty."""
        # For now, return all enabled tools
        # Later we can filter based on agent specialty
        return [
            {
                "name": info["name"],
                "description": info["description"],
                "key": key,
                "source": info["source"],
                "category": info.get("category", "general")
            }
            for key, info in self.tools.items()
            if info.get("enabled", True)
        ]
    
    def get_tool_names(self) -> List[str]:
        """Get list of all tool names."""
        return [info["name"] for info in self.tools.values()]
    
    def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """Search for tools by name or description."""
        query_lower = query.lower()
        matching_tools = []
        
        for key, info in self.tools.items():
            if (query_lower in info["name"].lower() or 
                query_lower in info["description"].lower()):
                matching_tools.append({
                    "key": key,
                    "name": info["name"],
                    "description": info["description"],
                    "source": info["source"],
                    "category": info.get("category", "general")
                })
        
        return matching_tools

# Global registry instance
_global_registry = None

def get_tool_registry() -> UniversalMCPToolRegistry:
    """Get the global tool registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = UniversalMCPToolRegistry()
    return _global_registry

async def initialize_tools():
    """Initialize the global tool registry."""
    registry = get_tool_registry()
    await registry.initialize()
    return registry 