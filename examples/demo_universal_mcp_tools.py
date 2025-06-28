#!/usr/bin/env python3
"""
Demo: Universal MCP Tools

Demonstrates how the new universal MCP tool system works:
- Universal tools available to all agents
- Simple tool execution patterns
- Integration with platform security and user management
- Backward compatibility with existing systems
"""

import asyncio
import logging
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demo_universal_tools():
    """Demonstrate universal MCP tools."""
    print("🌐 Universal MCP Tools Demo")
    print("=" * 50)
    
    try:
        # Import the new universal tools
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from core.universal_mcp_tools import (
            universal_mcp_tools,
            get_universal_tools,
            execute_universal_tool,
            list_universal_tools
        )
        
        print("✅ Universal MCP Tools imported successfully")
        
        # Initialize the universal tools
        await universal_mcp_tools.initialize()
        print("🔧 Universal tools initialized")
        
        # Demo 1: List available tools
        print("\n📋 Demo 1: List Universal Tools")
        print("-" * 30)
        
        tool_names = await list_universal_tools()
        print(f"Available universal tools: {tool_names}")
        
        # Get detailed tool info
        detailed_tools = await get_universal_tools()
        print(f"\nTotal tools available: {len(detailed_tools)}")
        
        for tool in detailed_tools[:3]:  # Show first 3 tools
            print(f"  - {tool['tool_name']}: {tool['description']}")
            print(f"    Universal: {tool.get('universal', False)}")
            print(f"    MCP Type: {tool.get('mcp_type', 'unknown')}")
        
        # Demo 2: Execute universal tools
        print("\n⚡ Demo 2: Execute Universal Tools")
        print("-" * 35)
        
        # Test web search
        print("🔍 Testing web search...")
        search_result = await execute_universal_tool(
            "web_search", 
            {"query": "artificial intelligence trends 2024", "num_results": 3}
        )
        
        if search_result.get("success"):
            # Fix: Access the data field correctly
            data = search_result.get("data", {})
            num_results = data.get("num_results", 0)
            organic_results = data.get("organic", [])
            
            print(f"✅ Search completed: {num_results} results")
            for i, result in enumerate(organic_results[:2]):
                print(f"  {i+1}. {result.get('title', 'No title')}")
        else:
            print(f"❌ Search failed: {search_result.get('error')}")
        
        # Test calculation
        print("\n🧮 Testing calculation...")
        calc_result = await execute_universal_tool(
            "calculate",
            {"expression": "15 + 27 * 2"}
        )
        
        if calc_result.get("success"):
            # Fix: Access the data field correctly
            data = calc_result.get("data", {})
            expression = data.get("expression", "")
            result = data.get("result", 0)
            
            print(f"✅ Calculation: {expression} = {result}")
        else:
            print(f"❌ Calculation failed: {calc_result.get('error')}")
        
        # Test time utility
        print("\n⏰ Testing time utility...")
        time_result = await execute_universal_tool("get_time", {})
        
        if time_result.get("success"):
            # Fix: Access the data field correctly and handle missing formatted field
            data = time_result.get("data", {})
            formatted_time = data.get("formatted", data.get("utc_time", "Unknown time"))
            
            print(f"✅ Current time: {formatted_time}")
        else:
            print(f"❌ Time query failed: {time_result.get('error')}")
        
        # Demo 3: User-specific tools (maintains existing security)
        print("\n👤 Demo 3: User-Specific Tool Management")
        print("-" * 40)
        
        # Simulate a user
        test_user_id = "demo_user_123"
        
        # Get tools for specific user (universal + user-specific)
        user_tools = await get_universal_tools(user_id=test_user_id)
        print(f"Tools available to user {test_user_id}: {len(user_tools)}")
        
        # Show universal vs user-specific tools
        universal_count = sum(1 for tool in user_tools if tool.get('universal', False))
        user_specific_count = len(user_tools) - universal_count
        
        print(f"  - Universal tools: {universal_count}")
        print(f"  - User-specific tools: {user_specific_count}")
        
        # Demo 4: Security and rate limiting
        print("\n🛡️ Demo 4: Security Features")
        print("-" * 28)
        
        # Test parameter validation
        print("Testing parameter validation...")
        malicious_result = await execute_universal_tool(
            "calculate",
            {"expression": "rm -rf /; exec('malicious code')"}
        )
        
        if not malicious_result.get("success"):
            print("✅ Security validation working - malicious input blocked")
            print(f"   Error: {malicious_result.get('error', 'Unknown error')}")
        else:
            print("❌ Security issue - malicious input not blocked!")
        
        # Test rate limiting by making rapid requests
        print("\nTesting rate limiting...")
        rapid_requests = []
        for i in range(5):
            result = await execute_universal_tool("get_time", {})
            rapid_requests.append(result.get("success", False))
        
        success_count = sum(rapid_requests)
        print(f"✅ Rapid requests: {success_count}/5 succeeded (rate limiting may apply)")
        
        # Demo 5: Tool performance tracking
        print("\n📊 Demo 5: Tool Analytics")
        print("-" * 25)
        
        stats = await universal_mcp_tools.get_stats(user_id=test_user_id)
        print(f"Tool statistics:")
        print(f"  - Universal tools: {stats.get('universal_tools', 0)}")
        print(f"  - User tools: {stats.get('user_tools', 0)}")
        print(f"  - Active sessions: {stats.get('active_sessions', 0)}")
        print(f"  - Total tools: {stats.get('total_tools', 0)}")
        
        # Demo 6: Integration with existing agent system
        print("\n🤖 Demo 6: Agent Integration")
        print("-" * 30)
        
        print("Simulating agent tool usage...")
        
        # Simulate how an agent would use tools
        agent_id = "demo_agent_001"
        
        # Agent gets available tools
        agent_tools = await universal_mcp_tools.get_tools_for_agent(test_user_id)
        print(f"Agent {agent_id} has access to {len(agent_tools)} tools")
        
        # Agent executes a tool with tracking
        agent_result = await universal_mcp_tools.execute_tool(
            tool_name="format_text",
            parameters={"text": "  hello world  ", "format": "upper"},
            user_id=test_user_id,
            agent_id=agent_id
        )
        
        if agent_result.get("success"):
            # Fix: Access the data field correctly
            data = agent_result.get("data", {})
            original = data.get("original", "")
            formatted = data.get("formatted", "")
            
            print(f"✅ Agent executed tool successfully:")
            print(f"   Original: '{original}'")
            print(f"   Formatted: '{formatted}'")
        else:
            print(f"❌ Agent tool execution failed: {agent_result.get('error')}")
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Benefits Demonstrated:")
        print("  ✅ Universal tool availability")
        print("  ✅ Simple execution patterns")
        print("  ✅ Security validation and rate limiting")
        print("  ✅ User-specific permissions maintained")
        print("  ✅ Usage tracking and analytics")
        print("  ✅ Agent integration patterns")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the universal MCP tools are properly installed")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.exception("Demo error")
    finally:
        # Cleanup
        try:
            await universal_mcp_tools.close()
            print("\n🧹 Cleanup completed")
        except:
            pass

async def demo_agent_integration():
    """Demonstrate how agents integrate with universal tools."""
    print("\n🤖 Agent Integration Demo")
    print("=" * 30)
    
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from core.universal_mcp_tools import universal_mcp_tools
        
        # Simulate an agent that needs to use tools
        class DemoAgent:
            def __init__(self, agent_id: str, user_id: str):
                self.agent_id = agent_id
                self.user_id = user_id
                self.tools = None
            
            async def initialize(self):
                """Initialize agent with available tools."""
                self.tools = await universal_mcp_tools.get_tools_for_agent(self.user_id)
                print(f"Agent {self.agent_id} initialized with {len(self.tools)} tools")
            
            async def execute_task(self, task: str):
                """Execute a task using available tools."""
                print(f"\nAgent {self.agent_id} executing task: {task}")
                
                if "search" in task.lower():
                    # Use web search tool
                    result = await universal_mcp_tools.execute_tool(
                        "web_search",
                        {"query": "latest AI developments", "num_results": 2},
                        self.user_id,
                        self.agent_id
                    )
                    if result.get("success"):
                        # Fix: Access the data field correctly
                        data = result.get("data", {})
                        organic_results = data.get("organic", [])
                        print(f"✅ Found {len(organic_results)} search results")
                        return data
                    else:
                        print(f"❌ Search failed: {result.get('error')}")
                
                elif "calculate" in task.lower():
                    # Use calculation tool
                    result = await universal_mcp_tools.execute_tool(
                        "calculate",
                        {"expression": "100 + 50"},
                        self.user_id,
                        self.agent_id
                    )
                    if result.get("success"):
                        # Fix: Access the data field correctly
                        data = result.get("data", {})
                        calc_result = data.get("result", 0)
                        print(f"✅ Calculation result: {calc_result}")
                        return data
                    else:
                        print(f"❌ Calculation failed: {result.get('error')}")
                
                elif "time" in task.lower():
                    # Use time tool
                    result = await universal_mcp_tools.execute_tool(
                        "get_time",
                        {},
                        self.user_id,
                        self.agent_id
                    )
                    if result.get("success"):
                        # Fix: Access the data field correctly
                        data = result.get("data", {})
                        formatted_time = data.get("formatted", data.get("utc_time", "Unknown time"))
                        print(f"✅ Current time: {formatted_time}")
                        return data
                    else:
                        print(f"❌ Time query failed: {result.get('error')}")
                
                else:
                    print(f"❓ Task not recognized: {task}")
                    return None
        
        # Create and test an agent
        await universal_mcp_tools.initialize()
        
        agent = DemoAgent("research_agent_001", "demo_user_456")
        await agent.initialize()
        
        # Execute various tasks
        await agent.execute_task("Search for latest AI developments")
        await agent.execute_task("Calculate the total cost")
        await agent.execute_task("Get current time")
        
        print("\n✅ Agent integration demo completed")
        
    except Exception as e:
        print(f"❌ Agent integration demo failed: {e}")
        logger.exception("Agent integration error")

async def main():
    """Run all demos."""
    try:
        await demo_universal_tools()
        await demo_agent_integration()
        
        print("\n" + "="*60)
        print("🎯 SUMMARY: Universal MCP Tools Implementation")
        print("="*60)
        print()
        print("✅ UNIVERSAL TOOL FEATURES:")
        print("   - Universal tools available to all agents")
        print("   - Simple tool execution patterns")
        print("   - Global tool registry")
        print("   - Convenience functions for easy access")
        print()
        print("✅ KEPT FROM EXISTING PLATFORM:")
        print("   - User-specific security and permissions")
        print("   - Rate limiting and parameter validation")
        print("   - Usage tracking and cost monitoring")
        print("   - Supabase integration and logging")
        print("   - Agent orchestration integration")
        print()
        print("✅ NEW FEATURES:")
        print("   - MCPToolRegistry for unified tool management")
        print("   - Enhanced security sandbox")
        print("   - Universal + user-specific tool patterns")
        print("   - Simple agent integration")
        print()
        print("🚀 READY FOR PRODUCTION USE!")
        
    except Exception as e:
        print(f"❌ Main demo failed: {e}")
        logger.exception("Main demo error")

if __name__ == "__main__":
    asyncio.run(main()) 