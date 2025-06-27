#!/usr/bin/env python3
"""
Comprehensive Slash Commands Test Suite

Tests all available slash commands in the AI Agent Platform:
- Feedback commands: /improve, /save-workflow, /list-workflows, /suggest
- System commands: /metrics
- MCP commands: /mcp with various subcommands

This test uses mock Slack objects to simulate command execution without
requiring actual Slack infrastructure.
"""

import asyncio
import sys
import os
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slack_interface.slack_bot import AIAgentSlackBot
from mcp.slack_interface.mcp_commands import MCPSlackCommands
from database.supabase_logger import SupabaseLogger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockSlackApp:
    """Mock Slack App for testing slash commands."""
    
    def __init__(self):
        self.commands = {}
        self.client = AsyncMock()
        
    def command(self, command_name):
        """Decorator to register command handlers."""
        def decorator(func):
            self.commands[command_name] = func
            return func
        return decorator
    
    def event(self, event_type):
        """Decorator to register event handlers."""
        def decorator(func):
            return func
        return decorator
    
    def action(self, action_id):
        """Decorator to register action handlers."""
        def decorator(func):
            return func
        return decorator
    
    def view(self, callback_id):
        """Decorator to register view handlers."""
        def decorator(func):
            return func
        return decorator

class SlashCommandTester:
    """Test suite for slash commands."""
    
    def __init__(self):
        self.results = []
        self.mock_app = MockSlackApp()
        
    async def setup_test_environment(self):
        """Set up the test environment with mocked dependencies."""
        logger.info("🔧 Setting up test environment...")
        
        # Mock environment variables
        os.environ.setdefault('SUPABASE_URL', 'https://test.supabase.co')
        os.environ.setdefault('SUPABASE_KEY', 'test-key')
        os.environ.setdefault('OPENAI_API_KEY', 'test-key')
        os.environ.setdefault('SLACK_BOT_TOKEN', 'xoxb-test')
        os.environ.setdefault('SLACK_APP_TOKEN', 'xapp-test')
        
        # Initialize components with mocks
        with patch('slack_interface.slack_bot.AsyncApp') as mock_slack_app:
            mock_slack_app.return_value = self.mock_app
            
            # Mock Supabase logger
            with patch('database.supabase_logger.SupabaseLogger') as mock_supabase:
                mock_supabase_instance = AsyncMock()
                mock_supabase.return_value = mock_supabase_instance
                
                # Mock orchestrator
                with patch('orchestrator.agent_orchestrator.AgentOrchestrator') as mock_orchestrator:
                    mock_orchestrator_instance = AsyncMock()
                    mock_orchestrator.return_value = mock_orchestrator_instance
                    
                    # Initialize the bot
                    self.bot = AIAgentSlackBot()
                    self.bot.app = self.mock_app
                    
                    # Initialize MCP commands
                    self.mcp_commands = MCPSlackCommands(mock_supabase_instance)
                    self.mcp_commands.register_handlers(self.mock_app)
        
        logger.info("✅ Test environment set up successfully")
    
    def create_mock_command(self, command_text: str = "", user_id: str = "U123456", 
                          channel_id: str = "C123456") -> Dict[str, Any]:
        """Create a mock Slack command object."""
        return {
            "text": command_text,
            "user_id": user_id,
            "channel_id": channel_id,
            "command": "/test",
            "response_url": "https://hooks.slack.com/commands/test"
        }
    
    async def create_mock_responses(self):
        """Create mock response objects."""
        ack = AsyncMock()
        say = AsyncMock()
        client = AsyncMock()
        return ack, say, client
    
    async def test_feedback_commands(self):
        """Test all feedback-related slash commands."""
        logger.info("🔄 Testing feedback commands...")
        
        feedback_commands = [
            ("/improve", "improve", "Make my workflow faster"),
            ("/save-workflow", "save-workflow", "customer_support_flow"),
            ("/list-workflows", "list-workflows", ""),
            ("/suggest", "feedback", "The system works great!")
        ]
        
        for cmd_name, cmd_type, cmd_text in feedback_commands:
            try:
                logger.info(f"   Testing {cmd_name}...")
                
                # Create mock objects
                command = self.create_mock_command(cmd_text)
                ack, say, client = await self.create_mock_responses()
                
                # Test the command
                if hasattr(self.bot, '_handle_feedback_command') and self.bot.feedback_handler:
                    await self.bot._handle_feedback_command(cmd_type, command, say, client)
                    
                    # Check if say was called
                    assert say.called, f"{cmd_name} should call say()"
                    
                    # Get the response
                    response = say.call_args[0][0] if say.call_args[0] else say.call_args[1]
                    logger.info(f"     ✅ {cmd_name} responded: {str(response)[:100]}...")
                    
                    self.results.append({
                        "command": cmd_name,
                        "status": "✅ PASS",
                        "response": str(response)[:200]
                    })
                else:
                    logger.warning(f"     ⚠️ {cmd_name} handler not found or feedback handler not available")
                    self.results.append({
                        "command": cmd_name,
                        "status": "⚠️ NOT IMPLEMENTED",
                        "response": "Handler method not found or feedback handler not available"
                    })
                    
            except Exception as e:
                logger.error(f"     ❌ {cmd_name} failed: {str(e)}")
                self.results.append({
                    "command": cmd_name,
                    "status": "❌ FAIL",
                    "response": str(e)
                })
    
    async def test_metrics_command(self):
        """Test the metrics slash command."""
        logger.info("📊 Testing metrics command...")
        
        try:
            # Create mock objects
            command = self.create_mock_command("")
            ack, say, client = await self.create_mock_responses()
            
            # Test the command
            if hasattr(self.bot, '_handle_metrics_command'):
                await self.bot._handle_metrics_command(command, say, client)
                
                assert say.called, "/metrics should call say()"
                response = say.call_args[0][0] if say.call_args[0] else say.call_args[1]
                logger.info(f"   ✅ /metrics responded: {str(response)[:100]}...")
                
                self.results.append({
                    "command": "/metrics",
                    "status": "✅ PASS",
                    "response": str(response)[:200]
                })
            else:
                logger.warning("   ⚠️ /metrics handler not found")
                self.results.append({
                    "command": "/metrics",
                    "status": "⚠️ NOT IMPLEMENTED",
                    "response": "Handler method not found"
                })
                
        except Exception as e:
            logger.error(f"   ❌ /metrics failed: {str(e)}")
            self.results.append({
                "command": "/metrics",
                "status": "❌ FAIL",
                "response": str(e)
            })
    
    async def test_mcp_commands(self):
        """Test all MCP slash commands."""
        logger.info("🔌 Testing MCP commands...")
        
        mcp_commands = [
            ("/mcp", "help", ""),
            ("/mcp", "connect", ""),
            ("/mcp", "connect", "supabase"),
            ("/mcp", "list", ""),
            ("/mcp", "tools", ""),
            ("/mcp", "disconnect", "test_connection"),
            ("/mcp", "test", "test_connection"),
            ("/mcp", "analytics", "")
        ]
        
        for cmd_name, subcommand, args in mcp_commands:
            try:
                cmd_text = f"{subcommand} {args}".strip() if subcommand != "help" else ""
                test_name = f"{cmd_name} {cmd_text}".strip()
                
                logger.info(f"   Testing {test_name}...")
                
                # Create mock objects
                command = self.create_mock_command(cmd_text)
                ack, say, client = await self.create_mock_responses()
                
                # Test the command
                if hasattr(self.mcp_commands, '_handle_mcp_command'):
                    await self.mcp_commands._handle_mcp_command(command, client, say)
                    
                    # Check if say was called
                    assert say.called, f"{test_name} should call say()"
                    
                    # Get the response
                    response = say.call_args[0][0] if say.call_args[0] else say.call_args[1]
                    logger.info(f"     ✅ {test_name} responded: {str(response)[:100]}...")
                    
                    self.results.append({
                        "command": test_name,
                        "status": "✅ PASS",
                        "response": str(response)[:200]
                    })
                else:
                    logger.warning(f"     ⚠️ {test_name} handler not found")
                    self.results.append({
                        "command": test_name,
                        "status": "⚠️ NOT IMPLEMENTED",
                        "response": "Handler method not found"
                    })
                    
            except Exception as e:
                logger.error(f"     ❌ {test_name} failed: {str(e)}")
                self.results.append({
                    "command": test_name,
                    "status": "❌ FAIL",
                    "response": str(e)
                })
    
    async def test_command_registration(self):
        """Test that all commands are properly registered."""
        logger.info("📋 Testing command registration...")
        
        expected_commands = [
            "/improve", "/save-workflow", "/list-workflows", 
            "/suggest", "/metrics", "/mcp"
        ]
        
        # Check if commands are registered in mock app
        registered_commands = list(self.mock_app.commands.keys())
        
        for cmd in expected_commands:
            if cmd in registered_commands:
                logger.info(f"   ✅ {cmd} is registered")
                self.results.append({
                    "command": f"{cmd} (registration)",
                    "status": "✅ REGISTERED",
                    "response": "Command handler found and registered"
                })
            else:
                logger.warning(f"   ⚠️ {cmd} is not registered")
                self.results.append({
                    "command": f"{cmd} (registration)",
                    "status": "⚠️ NOT REGISTERED",
                    "response": "Command handler not found in app"
                })
    
    async def run_all_tests(self):
        """Run all slash command tests."""
        logger.info("🚀 Starting comprehensive slash commands test...")
        
        # Setup
        await self.setup_test_environment()
        
        # Test registration
        await self.test_command_registration()
        
        # Test individual command types
        await self.test_feedback_commands()
        await self.test_metrics_command()
        await self.test_mcp_commands()
        
        # Generate report
        await self.generate_report()
    
    async def generate_report(self):
        """Generate a comprehensive test report."""
        logger.info("📊 Generating test report...")
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r["status"].startswith("✅")])
        failed_tests = len([r for r in self.results if r["status"].startswith("❌")])
        not_implemented = len([r for r in self.results if r["status"].startswith("⚠️")])
        
        print("\n" + "="*80)
        print("🧪 SLASH COMMANDS TEST REPORT")
        print("="*80)
        print(f"📊 Summary:")
        print(f"   Total Tests:      {total_tests}")
        print(f"   ✅ Passed:        {passed_tests}")
        print(f"   ❌ Failed:        {failed_tests}")
        print(f"   ⚠️ Not Implemented: {not_implemented}")
        print(f"   📈 Success Rate:  {(passed_tests/total_tests)*100:.1f}%")
        print("\n" + "-"*80)
        print("📋 Detailed Results:")
        print("-"*80)
        
        # Group results by status
        for status_type in ["✅ PASS", "✅ REGISTERED", "⚠️ NOT IMPLEMENTED", "⚠️ NOT REGISTERED", "❌ FAIL"]:
            matching_results = [r for r in self.results if r["status"] == status_type]
            if matching_results:
                print(f"\n{status_type}:")
                for result in matching_results:
                    print(f"   • {result['command']}")
                    if result['response'] and len(result['response']) > 50:
                        print(f"     Response: {result['response'][:100]}...")
                    elif result['response']:
                        print(f"     Response: {result['response']}")
        
        print("\n" + "="*80)
        print("💡 Recommendations:")
        print("="*80)
        
        if failed_tests > 0:
            print("❌ Failed Commands:")
            failed = [r for r in self.results if r["status"].startswith("❌")]
            for result in failed:
                print(f"   • Fix {result['command']}: {result['response']}")
        
        if not_implemented > 0:
            print("⚠️ Missing Implementations:")
            missing = [r for r in self.results if r["status"].startswith("⚠️")]
            for result in missing:
                print(f"   • Implement {result['command']}: {result['response']}")
        
        if passed_tests == total_tests:
            print("🎉 All slash commands are working perfectly!")
        
        print("\n" + "="*80)
        
        # Return summary for programmatic use
        return {
            "total": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "not_implemented": not_implemented,
            "success_rate": (passed_tests/total_tests)*100,
            "details": self.results
        }

async def main():
    """Main test execution."""
    tester = SlashCommandTester()
    
    try:
        report = await tester.run_all_tests()
        
        # Exit with appropriate code
        if report["failed"] > 0:
            logger.error("❌ Some tests failed")
            sys.exit(1)
        elif report["not_implemented"] > 0:
            logger.warning("⚠️ Some commands not implemented")
            sys.exit(2)
        else:
            logger.info("✅ All tests passed!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {str(e)}")
        sys.exit(3)

if __name__ == "__main__":
    asyncio.run(main()) 