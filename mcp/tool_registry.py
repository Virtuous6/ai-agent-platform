"""
MCP Tool Registry

Manages dynamic discovery, registration, and execution of MCP tools.
Provides a unified interface for tool management across different MCP connections.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MCPTool:
    """Represents an MCP tool."""
    tool_id: str
    connection_id: str
    tool_name: str
    description: str
    parameters: Dict[str, Any]
    execution_function: Callable
    cost_per_call: float = 0.0
    
class MCPToolRegistry:
    """
    Registry for MCP tools across all connections.
    
    Manages tool discovery, registration, and execution with
    cost tracking and usage analytics.
    """
    
    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, MCPTool] = {}
        self.tool_usage: Dict[str, int] = {}
        
        logger.info("🔧 MCP Tool Registry initialized")
    
    async def register_tool(self, connection_id: str, tool_name: str, 
                          description: str, parameters: Dict[str, Any],
                          execution_function: Callable) -> str:
        """
        Register a new tool from an MCP connection.
        
        Args:
            connection_id: MCP connection providing the tool
            tool_name: Name of the tool
            description: Tool description
            parameters: Tool parameter schema
            execution_function: Function to execute the tool
            
        Returns:
            Tool ID for future reference
        """
        tool_id = f"{connection_id}_{tool_name}"
        
        tool = MCPTool(
            tool_id=tool_id,
            connection_id=connection_id,
            tool_name=tool_name,
            description=description,
            parameters=parameters,
            execution_function=execution_function
        )
        
        self.tools[tool_id] = tool
        self.tool_usage[tool_id] = 0
        
        logger.info(f"🔧 Registered tool: {tool_name} from connection {connection_id}")
        return tool_id
    
    async def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a registered tool.
        
        Args:
            tool_id: Tool to execute
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        try:
            tool = self.tools.get(tool_id)
            if not tool:
                return {"error": f"Tool {tool_id} not found"}
            
            # Track usage
            self.tool_usage[tool_id] += 1
            
            # Execute tool
            result = await tool.execution_function(parameters)
            
            logger.info(f"🛠️ Executed tool: {tool.tool_name}")
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"error": str(e)}
    
    def get_available_tools(self, connection_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of available tools.
        
        Args:
            connection_id: Optional connection filter
            
        Returns:
            List of tool information
        """
        tools = []
        for tool in self.tools.values():
            if connection_id is None or tool.connection_id == connection_id:
                tools.append({
                    "tool_id": tool.tool_id,
                    "tool_name": tool.tool_name,
                    "description": tool.description,
                    "connection_id": tool.connection_id,
                    "usage_count": self.tool_usage.get(tool.tool_id, 0)
                })
        
        return tools
    
    def remove_tools_for_connection(self, connection_id: str) -> int:
        """
        Remove all tools for a connection.
        
        Args:
            connection_id: Connection to remove tools for
            
        Returns:
            Number of tools removed
        """
        tools_to_remove = [
            tool_id for tool_id, tool in self.tools.items()
            if tool.connection_id == connection_id
        ]
        
        for tool_id in tools_to_remove:
            del self.tools[tool_id]
            self.tool_usage.pop(tool_id, None)
        
        logger.info(f"🗑️ Removed {len(tools_to_remove)} tools for connection {connection_id}")
        return len(tools_to_remove) 