#!/usr/bin/env python3
"""
Universal MCP Tools

Universal tool availability patterns while maintaining
platform security and user management features.

Key Features:
- Universal tools available to all agents
- Simple registration and usage patterns
- Maintains user-specific security and permissions
- Integrates with existing platform features
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from mcp.tool_registry import MCPToolRegistry, ToolDescriptor
from mcp.security_sandbox import MCPSecuritySandbox
try:
    from supabase.supabase_logger import SupabaseLogger
except ImportError:
    # Fallback if supabase logger doesn't exist
    class SupabaseLogger:
        async def log_event(self, *args, **kwargs):
            pass
from tools.standard_library import get_standard_tools

logger = logging.getLogger(__name__)

class UniversalMCPTools:
    """
    Universal MCP tool manager with simple interface.
    
    Provides:
    - Universal tools available to all agents
    - Simple tool registration
    - Platform integration (security, tracking, user management)
    - Standard tool library integration
    """
    
    def __init__(self, supabase_logger: Optional[SupabaseLogger] = None):
        """Initialize universal MCP tools."""
        self.registry = MCPToolRegistry(supabase_logger)
        self.supabase_logger = supabase_logger
        self._initialized = False
        
        logger.info("🌐 Universal MCP Tools initializing...")
    
    async def initialize(self):
        """Initialize universal tools library."""
        if self._initialized:
            return
        
        # Load standard tools as universal MCP tools
        await self._load_standard_tools()
        
        # Load universal MCP tools
        await self._load_universal_mcp_tools()
        
        self._initialized = True
        logger.info("✅ Universal MCP Tools initialized")
    
    async def _load_standard_tools(self):
        """Load standard tools as universal tools."""
        try:
            standard_tools = get_standard_tools()
            
            for tool_config in standard_tools:
                await self.registry.register_universal_tool(
                    tool_name=tool_config["name"],
                    description=tool_config["description"],
                    mcp_type="standard",
                    function=tool_config["function"],
                    parameters=tool_config.get("parameters", {})
                )
                
            logger.info(f"📦 Loaded {len(standard_tools)} standard tools as universal tools")
            
        except Exception as e:
            logger.error(f"Error loading standard tools: {e}")
    
    async def _load_universal_mcp_tools(self):
        """Load universal MCP tools that all agents can access."""
        universal_tools = [
            {
                "name": "web_search",
                "description": "Search the web for information",
                "mcp_type": "search",
                "function": self._universal_web_search,
                "parameters": {
                    "query": {"type": "string", "required": True},
                    "num_results": {"type": "integer", "default": 5}
                }
            },
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "mcp_type": "math", 
                "function": self._universal_calculate,
                "parameters": {
                    "expression": {"type": "string", "required": True}
                }
            },
            {
                "name": "get_time",
                "description": "Get current date and time",
                "mcp_type": "utility",
                "function": self._universal_get_time,
                "parameters": {}
            },
            {
                "name": "format_text",
                "description": "Format and transform text",
                "mcp_type": "text",
                "function": self._universal_format_text,
                "parameters": {
                    "text": {"type": "string", "required": True},
                    "format": {"type": "string", "default": "clean"}
                }
            }
        ]
        
        for tool_config in universal_tools:
            await self.registry.register_universal_tool(
                tool_name=tool_config["name"],
                description=tool_config["description"],
                mcp_type=tool_config["mcp_type"],
                function=tool_config["function"],
                parameters=tool_config["parameters"]
            )
        
        logger.info(f"🌐 Loaded {len(universal_tools)} universal MCP tools")
    
    # ============================================================================
    # UNIVERSAL TOOL IMPLEMENTATIONS
    # ============================================================================
    
    async def _universal_web_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Universal web search tool available to all agents."""
        try:
            # In a real implementation, this would use a real search API
            # For now, simulate search results
            results = []
            for i in range(min(num_results, 10)):
                results.append({
                    "title": f"Search result {i+1} for '{query}'",
                    "url": f"https://example.com/result{i+1}",
                    "snippet": f"This is a snippet for result {i+1} about {query}",
                    "rank": i+1
                })
            
            return {
                "success": True,
                "query": query,
                "organic": results,
                "num_results": len(results)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _universal_calculate(self, expression: str) -> Dict[str, Any]:
        """Universal calculation tool available to all agents."""
        try:
            # Clean expression
            expression = expression.strip()
            
            # Safety check - only allow safe mathematical operations
            allowed_chars = set('0123456789+-*/().,e ')
            if not all(c in allowed_chars for c in expression.replace(' ', '')):
                return {
                    "success": False, 
                    "error": "Invalid characters in expression. Only numbers, +, -, *, /, (), and spaces allowed."
                }
            
            # Check for dangerous patterns
            dangerous_patterns = ['import', 'exec', 'eval', '__', 'lambda', 'def', 'class']
            expression_lower = expression.lower()
            if any(pattern in expression_lower for pattern in dangerous_patterns):
                return {
                    "success": False, 
                    "error": "Expression contains potentially dangerous patterns."
                }
            
            # Safely evaluate the mathematical expression
            result = eval(expression, {"__builtins__": {}}, {})
            
            return {
                "success": True,
                "expression": expression,
                "result": result
            }
            
        except Exception as e:
            return {"success": False, "error": f"Calculation error: {str(e)}"}
    
    async def _universal_get_time(self) -> Dict[str, Any]:
        """Universal time tool available to all agents."""
        try:
            now = datetime.utcnow()
            
            return {
                "success": True,
                "utc_time": now.isoformat(),
                "timestamp": now.timestamp(),
                "formatted": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S")
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _universal_format_text(self, text: str, format: str = "clean") -> Dict[str, Any]:
        """Universal text formatting tool available to all agents."""
        try:
            if format == "clean":
                # Clean up text
                result = " ".join(text.split())
                result = result.strip()
            elif format == "upper":
                result = text.upper()
            elif format == "lower":
                result = text.lower()
            elif format == "title":
                result = text.title()
            elif format == "reverse":
                result = text[::-1]
            else:
                result = text
            
            return {
                "success": True,
                "original": text,
                "formatted": result,
                "format_applied": format
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ============================================================================
    # AGENT INTEGRATION
    # ============================================================================
    
    async def get_tools_for_agent(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all available tools for an agent (universal + user-specific).
        
        This is the main method agents call for tool discovery.
        
        Args:
            user_id: User ID for user-specific tools
            
        Returns:
            List of available tools
        """
        await self.initialize()
        return await self.registry.get_available_tools(user_id)
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any],
                         user_id: Optional[str] = None, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a tool (universal or user-specific).
        
        Simple interface for tool execution.
        
        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            user_id: User requesting execution
            agent_id: Agent making the request
            
        Returns:
            Tool execution result
        """
        await self.initialize()
        
        # Find tool by name
        available_tools = await self.get_tools_for_agent(user_id)
        tool_id = None
        
        for tool in available_tools:
            if tool["tool_name"] == tool_name:
                tool_id = tool["tool_id"]
                break
        
        if not tool_id:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "available_tools": [t["tool_name"] for t in available_tools]
            }
        
        return await self.registry.execute_tool(tool_id, parameters, user_id, agent_id)
    
    async def list_tools(self, user_id: Optional[str] = None) -> List[str]:
        """
        List available tool names.
        
        Args:
            user_id: User ID for user-specific tools
            
        Returns:
            List of tool names
        """
        tools = await self.get_tools_for_agent(user_id)
        return [tool["tool_name"] for tool in tools]
    
    async def get_tool_info(self, tool_name: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific tool.
        
        Args:
            tool_name: Tool name
            user_id: User ID
            
        Returns:
            Tool information or None if not found
        """
        tools = await self.get_tools_for_agent(user_id)
        
        for tool in tools:
            if tool["tool_name"] == tool_name:
                return tool
        
        return None
    
    # ============================================================================
    # USER TOOL MANAGEMENT (maintains existing security model)
    # ============================================================================
    
    async def register_user_tool(self, user_id: str, tool_name: str, description: str,
                                mcp_type: str, connection_id: str, function: Callable,
                                parameters: Dict[str, Any] = None) -> str:
        """
        Register a user-specific tool.
        
        Maintains the existing security model while providing simple interface.
        """
        await self.initialize()
        
        return await self.registry.register_user_tool(
            user_id=user_id,
            tool_name=tool_name,
            description=description,
            mcp_type=mcp_type,
            connection_id=connection_id,
            function=function,
            parameters=parameters
        )
    
    async def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get tool usage statistics."""
        await self.initialize()
        return await self.registry.get_tool_stats(user_id)
    
    async def close(self):
        """Close and cleanup."""
        if hasattr(self, 'registry'):
            await self.registry.close()

# Global instance
universal_mcp_tools = UniversalMCPTools()

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def get_universal_tools(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get universal tools (convenience function)."""
    return await universal_mcp_tools.get_tools_for_agent(user_id)

async def execute_universal_tool(tool_name: str, parameters: Dict[str, Any],
                               user_id: Optional[str] = None) -> Dict[str, Any]:
    """Execute universal tool (convenience function)."""
    return await universal_mcp_tools.execute_tool(tool_name, parameters, user_id)

async def list_universal_tools(user_id: Optional[str] = None) -> List[str]:
    """List universal tool names (convenience function)."""
    return await universal_mcp_tools.list_tools(user_id) 