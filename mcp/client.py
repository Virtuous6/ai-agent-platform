"""
Official MCP Client Implementation
================================

This module provides a proper Model Context Protocol (MCP) client
for connecting to and using other MCP servers.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
import json
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)

class MCPClient:
    """
    MCP Client for connecting to other MCP servers.
    
    This client can discover and use tools from external MCP servers,
    following the official Model Context Protocol specification.
    """
    
    def __init__(self):
        """Initialize the MCP client."""
        self.sessions: Dict[str, ClientSession] = {}
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        
    async def connect_to_server(self, server_name: str, server_config: Dict[str, Any]) -> bool:
        """
        Connect to an MCP server.
        
        Args:
            server_name: Unique name for the server
            server_config: Configuration including command, args, env
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=server_config.get("command"),
                args=server_config.get("args", []),
                env=server_config.get("env", {})
            )
            
            # Connect using stdio client
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()
                    
                    # Store the session (note: this is simplified - in practice
                    # you'd want to manage the connection lifecycle properly)
                    self.sessions[server_name] = session
                    
                    # List available tools
                    tools_result = await session.list_tools()
                    
                    # Store tool information
                    for tool in tools_result.tools:
                        tool_key = f"{server_name}:{tool.name}"
                        self.available_tools[tool_key] = {
                            "name": tool.name,
                            "description": tool.description,
                            "server": server_name,
                            "inputSchema": tool.inputSchema
                        }
                    
                    logger.info(f"Connected to MCP server '{server_name}' with {len(tools_result.tools)} tools")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{server_name}': {e}")
            return False
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on a connected MCP server.
        
        Args:
            server_name: Name of the server
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        try:
            session = self.sessions.get(server_name)
            if not session:
                raise ValueError(f"Not connected to server '{server_name}'")
            
            # Call the tool
            result = await session.call_tool(tool_name, arguments)
            
            logger.info(f"Called tool '{tool_name}' on server '{server_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}' on server '{server_name}': {e}")
            raise
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all available tools from connected servers."""
        return self.available_tools.copy()
    
    def get_tools_by_server(self, server_name: str) -> List[Dict[str, Any]]:
        """Get tools available from a specific server."""
        return [
            tool for tool_key, tool in self.available_tools.items()
            if tool["server"] == server_name
        ]
    
    async def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for tools by description or name.
        
        Args:
            query: Search query
            
        Returns:
            List of matching tools
        """
        query_lower = query.lower()
        matching_tools = []
        
        for tool_key, tool in self.available_tools.items():
            if (query_lower in tool["name"].lower() or 
                query_lower in tool["description"].lower()):
                matching_tools.append(tool)
        
        return matching_tools
    
    async def disconnect_from_server(self, server_name: str):
        """Disconnect from an MCP server."""
        try:
            if server_name in self.sessions:
                # Remove session (connection will be closed when context exits)
                del self.sessions[server_name]
                
                # Remove tools from this server
                tools_to_remove = [
                    tool_key for tool_key in self.available_tools.keys()
                    if self.available_tools[tool_key]["server"] == server_name
                ]
                
                for tool_key in tools_to_remove:
                    del self.available_tools[tool_key]
                
                logger.info(f"Disconnected from MCP server '{server_name}'")
            
        except Exception as e:
            logger.error(f"Error disconnecting from server '{server_name}': {e}")
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        for server_name in list(self.sessions.keys()):
            await self.disconnect_from_server(server_name)

class MCPRegistry:
    """Registry for managing multiple MCP connections."""
    
    def __init__(self):
        """Initialize the MCP registry."""
        self.client = MCPClient()
        self.server_configs: Dict[str, Dict[str, Any]] = {}
        
    def register_server(self, server_name: str, config: Dict[str, Any]):
        """Register an MCP server configuration."""
        self.server_configs[server_name] = config
        logger.info(f"Registered MCP server config: {server_name}")
        
    async def connect_to_registered_servers(self) -> Dict[str, bool]:
        """Connect to all registered servers."""
        results = {}
        
        for server_name, config in self.server_configs.items():
            try:
                success = await self.client.connect_to_server(server_name, config)
                results[server_name] = success
            except Exception as e:
                logger.error(f"Failed to connect to {server_name}: {e}")
                results[server_name] = False
                
        return results
    
    async def auto_discover_tools(self, capability_needed: str) -> List[Dict[str, Any]]:
        """
        Auto-discover tools that might satisfy a needed capability.
        
        Args:
            capability_needed: Description of needed capability
            
        Returns:
            List of potentially matching tools
        """
        return await self.client.search_tools(capability_needed)
    
    def get_client(self) -> MCPClient:
        """Get the underlying MCP client."""
        return self.client

# Global registry instance
mcp_registry = MCPRegistry()

# Example server configurations
EXAMPLE_SERVERS = {
    "file_operations": {
        "command": "mcp-server-filesystem",
        "args": ["--root", "/tmp"],
        "env": {}
    },
    "web_search": {
        "command": "mcp-server-search", 
        "args": [],
        "env": {"SEARCH_API_KEY": "your-api-key"}
    }
}

async def setup_example_servers():
    """Setup example MCP server connections."""
    for server_name, config in EXAMPLE_SERVERS.items():
        mcp_registry.register_server(server_name, config)
    
    # Connect to registered servers
    results = await mcp_registry.connect_to_registered_servers()
    
    for server_name, success in results.items():
        if success:
            logger.info(f"✅ Connected to {server_name}")
        else:
            logger.error(f"❌ Failed to connect to {server_name}")
    
    return results

if __name__ == "__main__":
    # Test the MCP client
    async def test_client():
        await setup_example_servers()
        
        # List available tools
        tools = mcp_registry.get_client().get_available_tools()
        print(f"Available tools: {list(tools.keys())}")
        
        # Search for tools
        search_results = await mcp_registry.auto_discover_tools("file")
        print(f"File-related tools: {len(search_results)}")
    
    asyncio.run(test_client()) 