"""
MCP Registry
===========

Simple registry for managing MCP server connections and available tools.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import json

from .client import MCPClient, MCPRegistry
from .server import AIAgentMCPServer

logger = logging.getLogger(__name__)

@dataclass
class MCPServerInfo:
    """Information about an MCP server."""
    name: str
    description: str
    command: str
    args: List[str]
    env: Dict[str, str]
    status: str = "disconnected"  # disconnected, connected, error
    last_connected: Optional[datetime] = None
    tools_count: int = 0

@dataclass
class MCPToolInfo:
    """Information about an MCP tool."""
    name: str
    description: str
    server_name: str
    parameters: Dict[str, Any]
    last_used: Optional[datetime] = None
    usage_count: int = 0

class SimpleMCPRegistry:
    """
    Simple registry for managing MCP servers and tools.
    
    This provides a unified interface for:
    - Discovering available MCP servers
    - Connecting to servers
    - Finding and using tools
    - Managing server lifecycle
    """
    
    def __init__(self):
        """Initialize the MCP registry."""
        self.client = MCPClient()
        self.servers: Dict[str, MCPServerInfo] = {}
        self.tools: Dict[str, MCPToolInfo] = {}
        self.local_server: Optional[AIAgentMCPServer] = None
        
        # Load default server configurations
        self._load_default_servers()
        
    def _load_default_servers(self):
        """Load default MCP server configurations."""
        default_servers = {
            "filesystem": {
                "description": "File system operations",
                "command": "npx",
                "args": ["@modelcontextprotocol/server-filesystem", "/tmp"],
                "env": {}
            },
            "brave_search": {
                "description": "Brave web search",
                "command": "npx", 
                "args": ["@modelcontextprotocol/server-brave-search"],
                "env": {"BRAVE_API_KEY": ""}
            },
            "github": {
                "description": "GitHub operations",
                "command": "npx",
                "args": ["@modelcontextprotocol/server-github"],
                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": ""}
            },
            "postgres": {
                "description": "PostgreSQL database operations",
                "command": "npx",
                "args": ["@modelcontextprotocol/server-postgres"],
                "env": {"POSTGRES_CONNECTION_STRING": ""}
            }
        }
        
        for name, config in default_servers.items():
            self.register_server(
                name=name,
                description=config["description"],
                command=config["command"],
                args=config["args"],
                env=config["env"]
            )
    
    def register_server(self, name: str, description: str, command: str, 
                       args: List[str], env: Dict[str, str]):
        """Register an MCP server configuration."""
        server_info = MCPServerInfo(
            name=name,
            description=description,
            command=command,
            args=args,
            env=env
        )
        self.servers[name] = server_info
        logger.info(f"Registered MCP server: {name}")
    
    async def connect_to_server(self, server_name: str) -> bool:
        """Connect to a specific MCP server."""
        if server_name not in self.servers:
            logger.error(f"Server '{server_name}' not registered")
            return False
        
        server = self.servers[server_name]
        
        try:
            config = {
                "command": server.command,
                "args": server.args,
                "env": server.env
            }
            
            success = await self.client.connect_to_server(server_name, config)
            
            if success:
                server.status = "connected"
                server.last_connected = datetime.now()
                
                # Update tool registry
                await self._update_tools_from_server(server_name)
                
                logger.info(f"✅ Connected to MCP server: {server_name}")
            else:
                server.status = "error"
                logger.error(f"❌ Failed to connect to MCP server: {server_name}")
            
            return success
            
        except Exception as e:
            server.status = "error"
            logger.error(f"Error connecting to server '{server_name}': {e}")
            return False
    
    async def _update_tools_from_server(self, server_name: str):
        """Update tool registry with tools from a connected server."""
        try:
            server_tools = self.client.get_tools_by_server(server_name)
            
            for tool in server_tools:
                tool_key = f"{server_name}:{tool['name']}"
                
                tool_info = MCPToolInfo(
                    name=tool['name'],
                    description=tool['description'],
                    server_name=server_name,
                    parameters=tool.get('inputSchema', {})
                )
                
                self.tools[tool_key] = tool_info
            
            # Update server tool count
            if server_name in self.servers:
                self.servers[server_name].tools_count = len(server_tools)
            
            logger.info(f"Updated tools from server '{server_name}': {len(server_tools)} tools")
            
        except Exception as e:
            logger.error(f"Error updating tools from server '{server_name}': {e}")
    
    async def connect_to_all_servers(self) -> Dict[str, bool]:
        """Connect to all registered servers."""
        results = {}
        
        for server_name in self.servers.keys():
            results[server_name] = await self.connect_to_server(server_name)
        
        return results
    
    async def find_tools_for_capability(self, capability_description: str) -> List[MCPToolInfo]:
        """Find tools that might provide a needed capability."""
        matching_tools = []
        capability_lower = capability_description.lower()
        
        for tool_info in self.tools.values():
            if (capability_lower in tool_info.name.lower() or
                capability_lower in tool_info.description.lower()):
                matching_tools.append(tool_info)
        
        # Sort by usage count (most used first)
        matching_tools.sort(key=lambda t: t.usage_count, reverse=True)
        
        return matching_tools
    
    async def call_tool(self, tool_key: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool."""
        if tool_key not in self.tools:
            raise ValueError(f"Tool '{tool_key}' not found")
        
        tool_info = self.tools[tool_key]
        
        try:
            # Call the tool
            result = await self.client.call_tool(
                tool_info.server_name,
                tool_info.name,
                arguments
            )
            
            # Update usage statistics
            tool_info.last_used = datetime.now()
            tool_info.usage_count += 1
            
            logger.info(f"Called tool '{tool_key}' successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to call tool '{tool_key}': {e}")
            raise
    
    def get_available_tools(self) -> Dict[str, MCPToolInfo]:
        """Get all available tools."""
        return self.tools.copy()
    
    def get_connected_servers(self) -> List[MCPServerInfo]:
        """Get list of connected servers."""
        return [
            server for server in self.servers.values()
            if server.status == "connected"
        ]
    
    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all servers."""
        return {
            name: {
                "status": server.status,
                "description": server.description,
                "tools_count": server.tools_count,
                "last_connected": server.last_connected.isoformat() if server.last_connected else None
            }
            for name, server in self.servers.items()
        }
    
    def start_local_server(self, port: int = 8000):
        """Start the local MCP server for exposing agent capabilities."""
        if not self.local_server:
            self.local_server = AIAgentMCPServer()
        
        # Note: This would need to be run in a separate process/thread
        # for a real implementation
        logger.info(f"Local MCP server would start on port {port}")
        return self.local_server
    
    async def shutdown(self):
        """Shutdown all connections and cleanup."""
        try:
            await self.client.disconnect_all()
            
            # Reset server statuses
            for server in self.servers.values():
                server.status = "disconnected"
            
            # Clear tools
            self.tools.clear()
            
            logger.info("MCP Registry shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during MCP Registry shutdown: {e}")

# Global registry instance
mcp_registry = SimpleMCPRegistry()

async def get_mcp_registry() -> SimpleMCPRegistry:
    """Get the global MCP registry instance."""
    return mcp_registry

async def quick_setup_mcp():
    """Quick setup for common MCP servers."""
    logger.info("🚀 Starting MCP quick setup...")
    
    # Try to connect to commonly available servers
    results = await mcp_registry.connect_to_all_servers()
    
    connected_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    logger.info(f"📊 MCP Setup complete: {connected_count}/{total_count} servers connected")
    
    if connected_count > 0:
        tools = mcp_registry.get_available_tools()
        logger.info(f"🔧 Available MCP tools: {len(tools)}")
        
        # Log some example tools
        for i, (tool_key, tool_info) in enumerate(tools.items()):
            if i < 3:  # Show first 3 tools
                logger.info(f"  - {tool_key}: {tool_info.description}")
        
        if len(tools) > 3:
            logger.info(f"  ... and {len(tools) - 3} more tools")
    
    return mcp_registry

if __name__ == "__main__":
    # Test the registry
    async def test_registry():
        registry = await quick_setup_mcp()
        
        # Show server status
        status = registry.get_server_status()
        print(f"Server Status: {json.dumps(status, indent=2)}")
        
        # Try to find file-related tools
        file_tools = await registry.find_tools_for_capability("file")
        print(f"File tools found: {len(file_tools)}")
    
    asyncio.run(test_registry()) 