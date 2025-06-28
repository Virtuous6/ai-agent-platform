"""
MCP Integration Example
======================

Example showing how to use the official MCP implementation
in the AI Agent Platform.
"""

import asyncio
import logging
from typing import Dict, Any

from . import (
    setup_mcp_integration,
    get_mcp_status, 
    find_mcp_tools,
    mcp_registry,
    start_mcp_server
)

logger = logging.getLogger(__name__)

async def example_mcp_integration():
    """
    Complete example of MCP integration.
    """
    print("🚀 MCP Integration Example")
    print("=" * 50)
    
    # 1. Setup MCP Integration
    print("\n1. Setting up MCP integration...")
    setup_result = await setup_mcp_integration()
    print(f"   Status: {setup_result['status']}")
    print(f"   Connected servers: {setup_result.get('connected_servers', 0)}")
    print(f"   Total tools: {setup_result.get('tools_count', 0)}")
    
    # 2. Check MCP Status
    print("\n2. Checking MCP status...")
    status = await get_mcp_status()
    print(f"   MCP enabled: {status['mcp_enabled']}")
    print(f"   Implementation: {status['implementation']}")
    
    if status['mcp_enabled']:
        print(f"   Connected servers: {status['connected_servers']}")
        print(f"   Available tools: {status['tools']}")
        
        # Show server details
        for server_name, server_info in status['servers'].items():
            print(f"   - {server_name}: {server_info['status']} ({server_info['tools_count']} tools)")
    
    # 3. Find tools for specific capabilities
    print("\n3. Finding tools for specific capabilities...")
    
    capabilities = ["file", "search", "data"]
    for capability in capabilities:
        tools = await find_mcp_tools(capability)
        print(f"   {capability} tools found: {len(tools)}")
        
        for tool in tools[:2]:  # Show first 2 tools
            print(f"     - {tool['name']} (server: {tool['server']})")
            print(f"       {tool['description']}")
    
    # 4. Example tool usage (if tools are available)
    print("\n4. Example tool usage...")
    
    try:
        registry = mcp_registry
        available_tools = registry.get_available_tools()
        
        if available_tools:
            # Try to use the first available tool
            tool_key = list(available_tools.keys())[0]
            tool_info = available_tools[tool_key]
            
            print(f"   Trying to use tool: {tool_key}")
            print(f"   Description: {tool_info.description}")
            
            # Example tool call (this might fail if server isn't actually connected)
            try:
                result = await registry.call_tool(tool_key, {"test": "example"})
                print(f"   Tool result: {result}")
            except Exception as e:
                print(f"   Tool call failed (expected): {e}")
        else:
            print("   No tools available (servers not connected)")
            
    except Exception as e:
        print(f"   Tool usage example failed: {e}")
    
    print("\n✅ MCP Integration Example complete!")

async def example_mcp_server():
    """
    Example of running our own MCP server.
    """
    print("\n🖥️  Starting MCP Server Example")
    print("=" * 50)
    
    try:
        print("Starting MCP server on localhost:8000...")
        print("Server will expose AI Agent Platform capabilities as MCP tools")
        print("Press Ctrl+C to stop")
        
        # This would start the server (commented out to avoid blocking)
        # await start_mcp_server()
        
        print("(Server start commented out to avoid blocking the example)")
        
    except KeyboardInterrupt:
        print("\nMCP Server stopped")
    except Exception as e:
        print(f"MCP Server error: {e}")

async def example_agent_mcp_integration():
    """
    Example showing how agents can use MCP tools.
    """
    print("\n🤖 Agent MCP Integration Example")
    print("=" * 50)
    
    # This simulates how an agent would use MCP tools
    
    # 1. Agent needs a capability
    needed_capability = "file operations"
    print(f"Agent needs: {needed_capability}")
    
    # 2. Find MCP tools that provide this capability
    tools = await find_mcp_tools("file")
    print(f"Found {len(tools)} relevant tools")
    
    # 3. Agent selects the best tool
    if tools:
        best_tool = tools[0]  # First tool (sorted by usage)
        print(f"Agent selected: {best_tool['name']}")
        print(f"From server: {best_tool['server']}")
        
        # 4. Agent would use the tool
        print("Agent would call the tool with appropriate parameters")
        
        # In practice, this would be:
        # result = await mcp_registry.call_tool(f"{best_tool['server']}:{best_tool['name']}", parameters)
        
    else:
        print("No suitable tools found - agent would use fallback approach")

if __name__ == "__main__":
    # Run the examples
    async def run_examples():
        await example_mcp_integration()
        await example_mcp_server()
        await example_agent_mcp_integration()
    
    asyncio.run(run_examples()) 