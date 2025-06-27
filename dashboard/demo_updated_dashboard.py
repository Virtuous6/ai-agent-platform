#!/usr/bin/env python3
"""
Demo script for the updated dashboard with real Supabase data.

This script demonstrates the new dashboard features including:
- Messages view from Supabase
- Workflow runs (runbook executions)
- MCP connections with ability to add new ones
- Conversations view
- Real cost analytics
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.terminal_dashboard import TerminalDashboard
from database.supabase_logger import SupabaseLogger
from orchestrator.agent_orchestrator import AgentOrchestrator
from events.event_bus import EventBus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def create_sample_data():
    """Create some sample data in Supabase for dashboard testing."""
    logger.info("Creating sample data for dashboard testing...")
    
    try:
        # Initialize Supabase logger
        db_logger = SupabaseLogger()
        
        # Create sample conversation
        conversation_id = await db_logger.start_conversation(
            user_id="demo_user",
            channel_id="dashboard_demo"
        )
        
        # Add sample messages
        await db_logger.log_message(
            conversation_id=conversation_id,
            role="user",
            content="Hello! Can you help me with setting up a new MCP connection?",
            user_id="demo_user",
            metadata={"source": "dashboard_demo"}
        )
        
        await db_logger.log_message(
            conversation_id=conversation_id,
            role="assistant",
            content="Of course! I can help you set up MCP connections. What type of service would you like to connect to?",
            user_id="demo_user",
            metadata={"source": "dashboard_demo"}
        )
        
        # Add sample runbook execution
        try:
            await db_logger.log_runbook_execution(
                runbook_name="setup-github-mcp",
                user_id="demo_user",
                conversation_id=conversation_id,
                execution_state={"step": "connecting", "progress": 0.5},
                agents_used=["general_agent", "github_specialist"],
                tools_used=["github_api", "credential_manager"],
                total_tokens=1247,
                estimated_cost=0.0156
            )
        except Exception as e:
            logger.warning(f"Could not create runbook execution sample: {e}")
        
        # Add sample MCP connection
        try:
            connection_data = {
                "user_id": "demo_user",
                "connection_name": "demo_github",
                "mcp_type": "github",
                "display_name": "Demo GitHub Connection",
                "description": "Sample GitHub connection for dashboard testing",
                "connection_config": {
                    "default_org": "my-org",
                    "include_private": False
                },
                "tools_available": ["search_repos", "get_issues", "create_issue"],
                "status": "active",
                "total_executions": 42,
                "successful_executions": 39,
                "estimated_monthly_cost": 5.50
            }
            
            result = db_logger.client.table("mcp_connections") \
                .insert(connection_data) \
                .execute()
            
            if result.data:
                logger.info(f"Created sample MCP connection: {result.data[0]['id']}")
        except Exception as e:
            logger.warning(f"Could not create MCP connection sample: {e}")
        
        # End conversation
        await db_logger.end_conversation(
            conversation_id=conversation_id,
            total_messages=2,
            total_tokens=245,
            total_cost=0.0012
        )
        
        logger.info("Sample data created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        return False

async def demo_dashboard():
    """Run the updated dashboard demo."""
    print("🚀 AI Agent Platform - Updated Dashboard Demo")
    print("=" * 60)
    print()
    print("📊 Features:")
    print("  • Real Supabase data integration")
    print("  • Messages view (table: messages)")
    print("  • Workflow runs (table: runbook_executions)")
    print("  • MCP connections (table: mcp_connections)")
    print("  • Conversations view (table: conversations)")
    print("  • Real-time cost analytics")
    print()
    print("🎮 Navigation:")
    print("  [1] Overview    [2] Agents     [3] Costs")
    print("  [4] Events      [5] Logs       [6] Messages")
    print("  [7] Workflows   [8] MCPs       [9] Conversations")
    print("  [A] Add MCP     [R] Refresh    [Q] Quit")
    print()
    
    # Create sample data first
    print("📝 Creating sample data...")
    sample_created = await create_sample_data()
    if sample_created:
        print("✅ Sample data created successfully!")
    else:
        print("⚠️  Could not create sample data, but dashboard will still work")
    print()
    
    try:
        # Initialize components
        print("🔧 Initializing dashboard components...")
        
        # Supabase logger
        db_logger = SupabaseLogger()
        
        # Agent orchestrator (optional)
        try:
            orchestrator = AgentOrchestrator()
        except Exception as e:
            logger.warning(f"Could not initialize orchestrator: {e}")
            orchestrator = None
        
        # Event bus (optional)
        try:
            event_bus = EventBus()
        except Exception as e:
            logger.warning(f"Could not initialize event bus: {e}")
            event_bus = None
        
        # Initialize dashboard
        dashboard = TerminalDashboard(
            db_logger=db_logger,
            orchestrator=orchestrator,
            event_bus=event_bus
        )
        
        print("✅ Dashboard initialized successfully!")
        print()
        print("🖥️  Starting terminal dashboard...")
        print("   (If you don't see the dashboard, your terminal may not support curses)")
        print()
        
        # Start dashboard
        dashboard.start()
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard demo stopped by user")
    except Exception as e:
        logger.error(f"Dashboard demo error: {e}")
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   • Check your Supabase connection")
        print("   • Ensure your terminal supports curses")
        print("   • Try running in a proper terminal (not IDE)")

def main():
    """Main entry point."""
    try:
        asyncio.run(demo_dashboard())
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 