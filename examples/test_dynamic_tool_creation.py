"""
Test Dynamic Tool Creation System

This example demonstrates how the dynamic tool creation system works
when a user requests something that requires a tool the agent doesn't have.
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

# Mock imports (replace with actual imports in production)
from database.supabase_logger import SupabaseLogger
from events.event_bus import EventBus
from mcp.tool_registry import MCPToolRegistry
from mcp.security_sandbox import MCPSecuritySandbox
from mcp.dynamic_tool_builder import DynamicToolBuilder
from agents.universal_agent import UniversalAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockSlackApp:
    """Mock Slack app for testing."""
    def __init__(self):
        self.messages = []
    
    async def send_message(self, user_id: str, message: str, blocks=None):
        self.messages.append({
            'user_id': user_id,
            'message': message,
            'blocks': blocks,
            'timestamp': datetime.utcnow()
        })
        logger.info(f"📱 Slack message to {user_id}: {message}")

async def test_dynamic_tool_creation():
    """
    Test the complete dynamic tool creation workflow:
    
    1. User requests weather data (agent doesn't have weather tool)
    2. Agent detects tool gap
    3. System requests tool creation
    4. User provides API credentials
    5. Tool is built and deployed
    6. Original task is completed
    """
    
    logger.info("🚀 Starting Dynamic Tool Creation Test")
    
    # Initialize platform components
    logger.info("🔧 Initializing platform components...")
    
    supabase_logger = SupabaseLogger()
    event_bus = EventBus()
    tool_registry = MCPToolRegistry()
    security_sandbox = MCPSecuritySandbox()
    
    # Initialize dynamic tool builder
    tool_builder = DynamicToolBuilder(
        supabase_logger=supabase_logger,
        tool_registry=tool_registry,
        event_bus=event_bus,
        security_sandbox=security_sandbox
    )
    
    # Create enhanced agent (weather specialist that doesn't have weather tools yet)
    agent = UniversalAgent(
        specialty="Weather Information Specialist",
        system_prompt="You are a weather expert who provides current weather conditions and forecasts for any location.",
        temperature=0.3,
        dynamic_tool_builder=tool_builder,
        mcp_discovery_engine=None,  # Will be set by tool_builder
        mcp_tool_registry=tool_registry,
        event_bus=event_bus
    )
    
    logger.info("✅ Platform components initialized")
    
    # Simulate user request that needs a tool the agent doesn't have
    logger.info("\n" + "="*60)
    logger.info("📝 STEP 1: User makes request agent can't handle")
    logger.info("="*60)
    
    user_message = "What's the current weather in Tokyo, Japan?"
    user_context = {
        "user_id": "test_user_123",
        "channel_id": "general",
        "conversation_id": "conv_456"
    }
    
    logger.info(f"User: {user_message}")
    
    # Process message - this should detect tool gap
    logger.info("\n" + "="*60)
    logger.info("🔍 STEP 2: Agent detects tool gap")
    logger.info("="*60)
    
    result = await agent.process_message(user_message, user_context)
    
    logger.info(f"Agent response: {result['response'][:200]}...")
    
    if result.get('tool_gap_detected'):
        logger.info("✅ Tool gap detected successfully!")
        logger.info(f"   Gap ID: {result['gap_id']}")
        logger.info(f"   Capability needed: {result['capability_needed']}")
        logger.info(f"   Request ID: {result['request_id']}")
    else:
        logger.error("❌ Tool gap was not detected")
        return
    
    # Simulate checking for collaboration requests
    logger.info("\n" + "="*60)
    logger.info("🤝 STEP 3: Check user collaboration requests")
    logger.info("="*60)
    
    collaboration_request = await agent.check_mcp_tool_requests_status(user_context['user_id'])
    
    if collaboration_request:
        logger.info("✅ Collaboration request found!")
        logger.info(f"   Tool: {collaboration_request['tool_name']}")
        logger.info(f"   Help needed: {collaboration_request['help_needed']}")
    else:
        logger.warning("⚠️ No collaboration request found, simulating user input anyway...")
    
    # Simulate user providing tool information
    logger.info("\n" + "="*60)
    logger.info("👤 STEP 4: User provides tool building information")
    logger.info("="*60)
    
    # Simulate user input (API credentials, etc.)
    user_input = {
        "api_credentials": {
            "url": "https://api.openweathermap.org/data/2.5/weather",
            "key": "test_api_key_12345"
        },
        "testing_data": "q=Tokyo,JP&units=metric&appid=test_api_key_12345"
    }
    
    logger.info("User provides:")
    logger.info(f"   API URL: {user_input['api_credentials']['url']}")
    logger.info(f"   API Key: {'*' * 10}{user_input['api_credentials']['key'][-5:]}")
    logger.info(f"   Test data: {user_input['testing_data']}")
    
    # Submit user input to tool builder
    request_id = result['request_id']
    input_result = await tool_builder.handle_user_input(request_id, user_input)
    
    if input_result['success']:
        logger.info("✅ User input processed successfully!")
    else:
        logger.error(f"❌ Error processing user input: {input_result['error']}")
        return
    
    # Simulate tool building process (normally this runs automatically)
    logger.info("\n" + "="*60)
    logger.info("🔨 STEP 5: Tool building and deployment")
    logger.info("="*60)
    
    # Check if tool building completed
    await asyncio.sleep(2)  # Simulate building time
    
    request = tool_builder.active_requests.get(request_id)
    if request:
        logger.info(f"Tool building status: {request.status.value}")
        
        # If completed, simulate tool ready notification
        if request.status.value == "completed":
            logger.info("✅ Tool building completed!")
            
            # Simulate tool ready notification
            logger.info("\n" + "="*60)
            logger.info("🎉 STEP 6: Complete original task with new tool")
            logger.info("="*60)
            
            completion_result = await agent.handle_tool_ready(request_id)
            
            if completion_result:
                logger.info("✅ Original task completed with new tool!")
                logger.info(f"Final response: {completion_result['response'][:300]}...")
                
                if completion_result.get('tool_creation_success'):
                    logger.info("🎯 Tool was successfully created and used!")
            else:
                logger.warning("⚠️ Task completion not found")
    
    # Show final statistics
    logger.info("\n" + "="*60)
    logger.info("📊 FINAL STATISTICS")
    logger.info("="*60)
    
    agent_summary = await agent.get_tool_requests_summary()
    logger.info(f"Tool requests: {agent_summary['active_tool_requests']}")
    logger.info(f"Pending tasks: {agent_summary['pending_user_tasks']}")
    logger.info(f"Capabilities requested: {agent_summary['capabilities_requested']}")
    
    # Show tool registry state
    available_tools = await tool_registry.get_available_tools()
    logger.info(f"Available tools after test: {len(available_tools)}")
    
    for tool in available_tools:
        logger.info(f"   - {tool['tool_name']}: {tool['description']}")
    
    logger.info("\n🎉 Dynamic Tool Creation Test Completed!")
    
    return {
        "test_passed": True,
        "tool_gap_detected": result.get('tool_gap_detected', False),
        "tool_built": request.status.value == "completed" if request else False,
        "task_completed": completion_result is not None,
        "tools_created": len(available_tools)
    }

async def test_tool_gap_detection():
    """Test just the tool gap detection functionality."""
    logger.info("\n🔍 Testing Tool Gap Detection Only")
    
    # Initialize minimal components
    tool_builder = DynamicToolBuilder(
        supabase_logger=SupabaseLogger(),
        tool_registry=MCPToolRegistry(), 
        event_bus=EventBus(),
        security_sandbox=MCPSecuritySandbox()
    )
    
    # Test various messages to see what tool gaps are detected
    test_messages = [
        "What's the weather in Tokyo?",
        "Show me the latest stock price for AAPL",
        "Send an email to john@example.com", 
        "What's 2 + 2?",  # Should not need external tool
        "Translate 'hello' to Spanish",  # Should not need external tool
        "Get me data from our company database",
        "Search the web for recent AI news"
    ]
    
    for message in test_messages:
        logger.info(f"\nTesting: '{message}'")
        
        gap = await tool_builder.detect_tool_gap(
            agent_id="test_agent",
            message=message,
            context={"user_id": "test_user"}
        )
        
        if gap:
            logger.info(f"✅ Gap detected: {gap.capability_needed}")
            logger.info(f"   Description: {gap.description}")
            logger.info(f"   Priority: {gap.priority}")
            logger.info(f"   Solutions: {gap.suggested_solutions}")
        else:
            logger.info("❌ No tool gap detected")

async def main():
    """Run all tests."""
    logger.info("🧪 Starting Dynamic Tool Creation Tests")
    
    try:
        # Test 1: Tool gap detection
        await test_tool_gap_detection()
        
        # Test 2: Full workflow
        result = await test_dynamic_tool_creation()
        
        logger.info("\n" + "="*60)
        logger.info("🎯 TEST RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Test passed: {result['test_passed']}")
        logger.info(f"Tool gap detected: {result['tool_gap_detected']}")
        logger.info(f"Tool built: {result['tool_built']}")
        logger.info(f"Task completed: {result['task_completed']}")
        logger.info(f"Tools created: {result['tools_created']}")
        
        if all([
            result['test_passed'],
            result['tool_gap_detected'],
            result['tool_built'],
            result['task_completed']
        ]):
            logger.info("🎉 ALL TESTS PASSED! Dynamic tool creation is working!")
        else:
            logger.warning("⚠️ Some tests failed. Check the logs above.")
    
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 