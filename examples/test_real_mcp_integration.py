#!/usr/bin/env python3
"""
Test Real MCP Integration

This script tests the complete MCP integration flow:
1. Real protocol client connection
2. Tool discovery
3. Tool execution
4. Session management

Run this to validate that your MCP servers are working with our platform.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp.mcp_client import mcp_client, MCPProtocolClient
from database.supabase_logger import SupabaseLogger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_mcp_connection(server_url: str, credentials: dict = None):
    """Test MCP server connection and tool discovery."""
    logger.info(f"🧪 Testing MCP connection to: {server_url}")
    
    try:
        # Test connection
        session = await mcp_client.connect_to_mcp_server(server_url, credentials)
        
        logger.info(f"✅ Connected! Session ID: {session.session_id}")
        logger.info(f"🔧 Available tools: {len(session.tools)}")
        
        for tool in session.tools:
            logger.info(f"   - {tool.name}: {tool.description}")
        
        return session
        
    except Exception as e:
        logger.error(f"❌ Connection failed: {str(e)}")
        return None

async def test_tool_execution(session_id: str, tool_name: str, parameters: dict = None):
    """Test tool execution on MCP server."""
    logger.info(f"🔧 Testing tool execution: {tool_name}")
    
    try:
        result = await mcp_client.execute_tool(session_id, tool_name, parameters)
        
        if result.get("success"):
            logger.info(f"✅ Tool executed successfully!")
            logger.info(f"📊 Result: {result.get('data', 'No data')}")
        else:
            logger.error(f"❌ Tool execution failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Tool execution error: {str(e)}")
        return {"success": False, "error": str(e)}

async def test_database_integration():
    """Test that MCP connections work with our database."""
    logger.info("🗄️ Testing database integration...")
    
    try:
        db = SupabaseLogger()
        
        # Get existing MCP connections
        result = db.client.table("mcp_connections").select("*").limit(5).execute()
        
        if result.data:
            logger.info(f"✅ Found {len(result.data)} MCP connections in database")
            for conn in result.data:
                logger.info(f"   - {conn.get('connection_name')}: {conn.get('mcp_server_url')}")
        else:
            logger.warning("⚠️ No MCP connections found in database")
            
        return result.data
        
    except Exception as e:
        logger.error(f"❌ Database integration test failed: {str(e)}")
        return []

async def main():
    """Run comprehensive MCP integration tests."""
    print("🚀 AI Agent Platform - Real MCP Integration Test")
    print("=" * 60)
    
    # Test database integration first
    connections = await test_database_integration()
    
    if not connections:
        print("\n⚠️ No MCP connections found in database.")
        print("Use `/mcp connect custom [name] [url]` in Slack to add connections first.")
        return
    
    # Test each connection
    test_results = []
    
    for conn in connections[:2]:  # Test first 2 connections
        connection_name = conn.get('connection_name')
        server_url = conn.get('mcp_server_url')
        
        print(f"\n🔗 Testing connection: {connection_name}")
        print(f"📡 Server URL: {server_url}")
        
        # Test connection
        session = await test_mcp_connection(server_url)
        
        if session:
            # Test tool discovery
            tools = await mcp_client.get_session_tools(session.session_id)
            
            if tools:
                # Test first tool execution
                first_tool = tools[0]
                print(f"\n🔧 Testing tool: {first_tool.name}")
                
                # Use simple parameters for testing
                test_params = {}
                if "get" in first_tool.name.lower():
                    test_params = {"limit": 3}
                elif "list" in first_tool.name.lower():
                    test_params = {"maxResults": 3}
                
                result = await test_tool_execution(
                    session.session_id, 
                    first_tool.name, 
                    test_params
                )
                
                test_results.append({
                    "connection": connection_name,
                    "session_created": True,
                    "tools_discovered": len(tools),
                    "tool_execution": result.get("success", False)
                })
            else:
                test_results.append({
                    "connection": connection_name,
                    "session_created": True,
                    "tools_discovered": 0,
                    "tool_execution": False
                })
                
            # Clean up session
            await mcp_client.close_session(session.session_id)
        else:
            test_results.append({
                "connection": connection_name,
                "session_created": False,
                "tools_discovered": 0,
                "tool_execution": False
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for result in test_results:
        status = "✅" if all([
            result["session_created"],
            result["tools_discovered"] > 0,
            result["tool_execution"]
        ]) else "❌"
        
        print(f"{status} {result['connection']}:")
        print(f"   Session: {'✅' if result['session_created'] else '❌'}")
        print(f"   Tools: {result['tools_discovered']}")
        print(f"   Execution: {'✅' if result['tool_execution'] else '❌'}")
    
    # Overall status
    successful_tests = sum(1 for r in test_results if r["tool_execution"])
    total_tests = len(test_results)
    
    if successful_tests == total_tests and total_tests > 0:
        print(f"\n🎉 ALL TESTS PASSED! ({successful_tests}/{total_tests})")
        print("Your MCP integration is working correctly!")
    elif successful_tests > 0:
        print(f"\n⚠️ PARTIAL SUCCESS ({successful_tests}/{total_tests})")
        print("Some MCP connections are working. Check failed connections.")
    else:
        print(f"\n❌ ALL TESTS FAILED ({successful_tests}/{total_tests})")
        print("Check your MCP server configurations and network connectivity.")

if __name__ == "__main__":
    asyncio.run(main()) 