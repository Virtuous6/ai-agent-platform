"""
Setup Serper Web Search MCP

This script demonstrates how to connect the Serper MCP to the AI Agent Platform.
Run this to set up web search capabilities using your Serper.dev API key.
"""

import asyncio
import logging
import os
from typing import Dict, Any

# Platform imports
from database.supabase_logger import SupabaseLogger
from events.event_bus import EventBus
from mcp.tool_registry import MCPToolRegistry
from mcp.credential_store import CredentialManager
from mcp.mcp_discovery_engine import MCPDiscoveryEngine
from mcp.run_cards.serper_card import SerperMCP, SERPER_TOOLS
from agents.universal_agent import UniversalAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_serper_mcp(api_key: str) -> Dict[str, Any]:
    """
    Set up Serper MCP with the provided API key.
    
    Args:
        api_key: Your Serper.dev API key
        
    Returns:
        Setup result with connection status
    """
    logger.info("🔍 Setting up Serper Web Search MCP...")
    
    try:
        # Initialize platform components
        credential_manager = CredentialManager()
        tool_registry = MCPToolRegistry()
        event_bus = EventBus()
        
        # Step 1: Test the API key
        logger.info("🧪 Testing Serper API connection...")
        serper = SerperMCP(api_key)
        test_result = await serper.test_connection()
        
        if not test_result["success"]:
            logger.error(f"❌ Serper API test failed: {test_result['error']}")
            return {
                "success": False,
                "error": "API key validation failed",
                "details": test_result
            }
        
        logger.info("✅ Serper API connection successful!")
        
        # Step 2: Store the API key securely
        logger.info("🔐 Storing API credentials...")
        await credential_manager.store_credential(
            service_id="serper",
            credential_type="serper_api_key", 
            credential_value=api_key,
            metadata={
                "service_name": "Serper Web Search",
                "api_url": "https://google.serper.dev",
                "setup_date": "2024-01-01"  # Would be actual date
            }
        )
        
        # Step 3: Register Serper tools in the MCP registry
        logger.info("📦 Registering Serper MCP tools...")
        
        for tool_config in SERPER_TOOLS:
            # Add API key to tool configuration
            tool_config["credentials"] = {"api_key": api_key}
            
            await tool_registry.register_tool(
                tool_id=tool_config["tool_id"],
                tool_name=tool_config["tool_name"],
                description=tool_config["description"],
                function=tool_config["function"],
                parameters=tool_config["parameters"],
                mcp_type="serper",
                credentials={"api_key": api_key}
            )
            
            logger.info(f"   ✅ Registered: {tool_config['tool_name']}")
        
        # Step 4: Test the registered tools
        logger.info("🧪 Testing registered tools...")
        
        test_query = "latest AI developments"
        search_result = await tool_registry.execute_tool(
            "serper_web_search",
            {"query": test_query, "num_results": 3, "api_key": api_key}
        )
        
        if search_result.get("success"):
            num_results = len(search_result.get("organic", []))
            logger.info(f"✅ Web search test successful! Found {num_results} results for '{test_query}'")
        else:
            logger.warning(f"⚠️ Web search test failed: {search_result.get('error')}")
        
        # Step 5: Publish MCP ready event
        await event_bus.publish(
            "mcp_connected",
            {
                "mcp_id": "serper",
                "mcp_name": "Serper Web Search MCP",
                "tools_registered": len(SERPER_TOOLS),
                "capabilities": ["web_search", "news_search", "images_search", "places_search"]
            },
            source="serper_setup"
        )
        
        logger.info("🎉 Serper MCP setup completed successfully!")
        
        return {
            "success": True,
            "mcp_id": "serper",
            "tools_registered": len(SERPER_TOOLS),
            "test_results": {
                "api_connection": test_result,
                "web_search": search_result.get("success", False)
            },
            "message": "Serper Web Search MCP is ready to use!"
        }
        
    except Exception as e:
        logger.error(f"❌ Serper MCP setup failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to set up Serper MCP"
        }

async def test_agent_with_serper(api_key: str):
    """Test an agent using the newly connected Serper MCP."""
    logger.info("\n🤖 Testing agent with Serper MCP integration...")
    
    # Initialize platform components
    supabase_logger = SupabaseLogger()
    event_bus = EventBus()
    tool_registry = MCPToolRegistry()
    mcp_discovery = MCPDiscoveryEngine()
    
    # Set up the registry with Serper tools
    for tool_config in SERPER_TOOLS:
        tool_config["credentials"] = {"api_key": api_key}
        await tool_registry.register_tool(
            tool_id=tool_config["tool_id"],
            tool_name=tool_config["tool_name"],
            description=tool_config["description"],
            function=tool_config["function"],
            parameters=tool_config["parameters"],
            mcp_type="serper",
            credentials={"api_key": api_key}
        )
    
    # Create an agent that can use web search
    agent = UniversalAgent(
        specialty="Research Assistant",
        system_prompt="You are a research assistant who can search the web for current information and provide comprehensive answers.",
        temperature=0.3,
        mcp_discovery_engine=mcp_discovery,
        mcp_tool_registry=tool_registry,
        event_bus=event_bus,
        supabase_logger=supabase_logger
    )
    
    # Test with a search-requiring query
    test_queries = [
        "What are the latest developments in AI in 2024?",
        "Find recent news about climate change",
        "Search for information about the current stock market trends"
    ]
    
    for query in test_queries:
        logger.info(f"\n📝 Testing query: '{query}'")
        
        result = await agent.process_message(
            query,
            {
                "user_id": "test_user",
                "conversation_id": "test_conv",
                "channel_id": "test_channel"
            }
        )
        
        logger.info(f"🤖 Agent response confidence: {result['confidence']}")
        if result.get('metadata', {}).get('mcp_integration', {}).get('gap_detected'):
            logger.info("🔍 MCP tool gap detected - system would suggest Serper MCP")
        else:
            logger.info("✅ Agent processed successfully")
        
        logger.info(f"📄 Response preview: {result['response'][:200]}...")

async def demo_mcp_discovery():
    """Demonstrate how the MCP discovery engine finds Serper for search queries."""
    logger.info("\n🔍 Demonstrating MCP Discovery for Search Queries...")
    
    discovery = MCPDiscoveryEngine()
    
    search_queries = [
        "I need to search the web for recent news",
        "Can you find information about Python programming?", 
        "Look up the latest stock prices",
        "Search for restaurants near me",
        "Find recent research papers on machine learning"
    ]
    
    for query in search_queries:
        logger.info(f"\n📝 Query: '{query}'")
        
        solutions = await discovery.find_mcp_solutions(
            capability_needed="web search",
            description=query,
            context={"message": query}
        )
        
        if solutions:
            for i, solution in enumerate(solutions):
                logger.info(f"   {i+1}. 📦 {solution.capability.name}")
                logger.info(f"      Match: {solution.match_type} (score: {solution.match_score:.2f})")
                logger.info(f"      Setup: {solution.setup_needed}")
        else:
            logger.info("   ❌ No MCP solutions found")

async def main():
    """Main setup and demo flow."""
    logger.info("🚀 Serper MCP Setup and Demo")
    logger.info("="*60)
    
    # Get API key from environment or prompt user
    api_key = os.getenv("SERPER_API_KEY")
    
    if not api_key:
        logger.warning("⚠️ SERPER_API_KEY environment variable not set")
        logger.info("Please set your Serper API key:")
        logger.info("export SERPER_API_KEY='your_api_key_here'")
        logger.info("\nOr get one from: https://serper.dev/")
        
        # For demo purposes, use a placeholder
        api_key = "demo_api_key_replace_with_real_key"
        logger.info("🔧 Using placeholder API key for demonstration...")
    else:
        logger.info(f"✅ Using API key: {'*' * 10}{api_key[-5:]}")
    
    try:
        # Demo 1: MCP Discovery
        await demo_mcp_discovery()
        
        # Demo 2: Setup Serper MCP (if real API key)
        if api_key != "demo_api_key_replace_with_real_key":
            logger.info("\n" + "="*60)
            logger.info("🔧 Setting up Serper MCP...")
            
            setup_result = await setup_serper_mcp(api_key)
            
            if setup_result["success"]:
                logger.info("✅ Serper MCP setup successful!")
                
                # Demo 3: Test agent with Serper
                await test_agent_with_serper(api_key)
            else:
                logger.error(f"❌ Setup failed: {setup_result['error']}")
        else:
            logger.info("\n💡 To actually connect Serper MCP:")
            logger.info("   1. Get API key from https://serper.dev/")
            logger.info("   2. Set SERPER_API_KEY environment variable")
            logger.info("   3. Run this script again")
        
        logger.info("\n🎉 Demo completed!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 