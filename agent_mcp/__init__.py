"""
Model Context Protocol (MCP) Integration
=======================================

Official MCP implementation for the AI Agent Platform using the
official Python SDK and FastMCP framework.

This module provides:
- MCP Server: Expose agent capabilities as MCP tools
- MCP Client: Connect to and use external MCP servers  
- MCP Registry: Manage server connections and tool discovery
- Tool Examples: Reference implementations using official patterns

Key Features:
- ✅ Official MCP Python SDK compliance
- ✅ FastMCP server implementation
- ✅ Standard JSON-RPC 2.0 protocol
- ✅ Proper tool discovery and execution
- ✅ Server lifecycle management
- ✅ Industry-standard patterns

Usage:
    # Start MCP server
    from mcp import start_mcp_server
    await start_mcp_server()
    
    # Use MCP client
    from mcp import mcp_registry
    await mcp_registry.connect_to_server("filesystem", config)
    tools = mcp_registry.find_tools_for_capability("file")
    
    # Quick setup
    from mcp import quick_setup_mcp
    registry = await quick_setup_mcp()
"""

import logging
from typing import Dict, Any, List, Optional

# Core MCP components
from .server import AIAgentMCPServer, mcp_server, start_mcp_server
from .client import MCPClient, MCPRegistry, mcp_registry as client_registry
from .registry import SimpleMCPRegistry, mcp_registry, quick_setup_mcp, get_mcp_registry

# Tool examples
from .tools.web_search import web_search_mcp

logger = logging.getLogger(__name__)

# Public API exports
__all__ = [
    # Server components
    "AIAgentMCPServer",
    "mcp_server", 
    "start_mcp_server",
    
    # Client components
    "MCPClient",
    "MCPRegistry",
    "client_registry",
    
    # Registry components
    "SimpleMCPRegistry",
    "mcp_registry",
    "quick_setup_mcp",
    "get_mcp_registry",
    
    # Tool examples
    "web_search_mcp",
    
    # Convenience functions
    "setup_mcp_integration",
    "get_mcp_status",
    "find_mcp_tools"
]

async def setup_mcp_integration() -> Dict[str, Any]:
    """
    Setup complete MCP integration for the AI Agent Platform.
    
    Returns:
        Status information about the MCP setup
    """
    logger.info("🚀 Setting up MCP integration...")
    
    try:
        # Setup MCP registry with default servers
        registry = await quick_setup_mcp()
        
        # Get status
        status = registry.get_server_status()
        tools = registry.get_available_tools()
        
        result = {
            "status": "success",
            "servers": status,
            "tools_count": len(tools),
            "connected_servers": len([s for s in status.values() if s["status"] == "connected"]),
            "total_servers": len(status),
            "mcp_version": "official_sdk",
            "implementation": "FastMCP"
        }
        
        logger.info(f"✅ MCP Integration complete: {result['connected_servers']}/{result['total_servers']} servers")
        return result
        
    except Exception as e:
        logger.error(f"❌ MCP Integration failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "mcp_version": "official_sdk",
            "implementation": "FastMCP"
        }

async def get_mcp_status() -> Dict[str, Any]:
    """Get current MCP integration status."""
    try:
        registry = await get_mcp_registry()
        
        return {
            "mcp_enabled": True,
            "implementation": "official_mcp_sdk",
            "servers": registry.get_server_status(),
            "tools": len(registry.get_available_tools()),
            "connected_servers": len(registry.get_connected_servers())
        }
        
    except Exception as e:
        logger.error(f"Error getting MCP status: {e}")
        return {
            "mcp_enabled": False,
            "error": str(e),
            "implementation": "official_mcp_sdk"
        }

async def find_mcp_tools(capability: str) -> List[Dict[str, Any]]:
    """
    Find MCP tools that provide a specific capability.
    
    Args:
        capability: Description of needed capability
        
    Returns:
        List of matching tools with their details
    """
    try:
        registry = await get_mcp_registry()
        tools = await registry.find_tools_for_capability(capability)
        
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "server": tool.server_name,
                "usage_count": tool.usage_count,
                "last_used": tool.last_used.isoformat() if tool.last_used else None
            }
            for tool in tools
        ]
        
    except Exception as e:
        logger.error(f"Error finding MCP tools for '{capability}': {e}")
        return []

# Compatibility with existing code that might import old classes
# These are deprecated and will be removed in future versions
class DeprecatedMCPClient:
    """Deprecated: Use MCPClient instead."""
    def __init__(self, *args, **kwargs):
        logger.warning("DeprecatedMCPClient is deprecated. Use mcp.MCPClient instead.")
        raise ImportError("Old MCP implementation has been replaced. Use the new official SDK implementation.")

class DeprecatedMCPProtocolClient:
    """Deprecated: Use MCPClient instead."""
    def __init__(self, *args, **kwargs):
        logger.warning("MCPProtocolClient is deprecated. Use mcp.MCPClient instead.")
        raise ImportError("Old MCP implementation has been replaced. Use the new official SDK implementation.")

# For backwards compatibility, map old names to new ones with deprecation warnings
MCPProtocolClient = DeprecatedMCPProtocolClient

logger.info("✅ MCP module loaded with official SDK implementation")
logger.info("📦 Available: Server, Client, Registry, Tools")
logger.info("🔧 Use quick_setup_mcp() to get started") 