#!/usr/bin/env python3
"""
Demo launcher for the AI Agent Platform Web Dashboard

This script launches the web dashboard and optionally creates sample data
for demonstration purposes.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def create_sample_data():
    """Create sample data for dashboard demo."""
    try:
        from database.supabase_logger import SupabaseLogger
        
        print("📝 Creating sample data for dashboard demo...")
        
        db_logger = SupabaseLogger()
        
        # Create sample conversation
        conversation_id = await db_logger.start_conversation(
            user_id="web_demo_user",
            channel_id="web_dashboard_demo"
        )
        
        # Add sample messages
        await db_logger.log_message(
            conversation_id=conversation_id,
            role="user",
            content="Can you help me set up a new MCP connection for GitHub?",
            user_id="web_demo_user",
            metadata={"source": "web_dashboard_demo"}
        )
        
        await db_logger.log_message(
            conversation_id=conversation_id,
            role="assistant", 
            content="I'd be happy to help you set up a GitHub MCP connection! Let me guide you through the process.",
            user_id="web_demo_user",
            metadata={"source": "web_dashboard_demo"}
        )
        
        await db_logger.log_message(
            conversation_id=conversation_id,
            role="user",
            content="Great! What information do I need to provide?",
            user_id="web_demo_user",
            metadata={"source": "web_dashboard_demo"}
        )
        
        # Add sample MCP connection
        connection_data = {
            "user_id": "web_demo_user",
            "connection_name": "my_github_repo",
            "mcp_type": "github",
            "display_name": "My GitHub Repository",
            "description": "Connection to my GitHub repositories for code management",
            "connection_config": {
                "default_org": "my-organization",
                "include_private": True
            },
            "tools_available": ["search_repos", "get_issues", "create_issue", "get_pull_requests"],
            "status": "active",
            "total_executions": 28,
            "successful_executions": 26,
            "estimated_monthly_cost": 7.25
        }
        
        result = db_logger.client.table("mcp_connections") \
            .insert(connection_data) \
            .execute()
        
        if result.data:
            print(f"✅ Created sample MCP connection: {result.data[0]['id']}")
        
        # End conversation
        await db_logger.end_conversation(
            conversation_id=conversation_id,
            total_messages=3,
            total_tokens=487,
            total_cost=0.0031
        )
        
        print("✅ Sample data created successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not create sample data: {e}")
        print("   Dashboard will still work with existing data")
        return False

def main():
    """Main demo launcher."""
    print("🚀 AI Agent Platform - Web Dashboard Demo")
    print("=" * 60)
    print()
    print("🌐 Features:")
    print("  • Modern web interface with real Supabase data")
    print("  • Interactive MCP connection management")
    print("  • Real-time monitoring and analytics")
    print("  • Multi-tab navigation with live updates")
    print()
    
    # Check if we should create sample data
    create_samples = input("📝 Create sample data for demo? [y/N]: ").lower().strip()
    
    if create_samples == 'y':
        try:
            asyncio.run(create_sample_data())
        except Exception as e:
            print(f"⚠️  Sample data creation failed: {e}")
    
    print()
    print("🔧 Starting web dashboard...")
    print()
    
    # Import and run the dashboard
    try:
        from apps.web_dashboard import app, initialize_components
        
        # Initialize components
        print("⚙️  Initializing dashboard components...")
        if initialize_components():
            print("✅ Components initialized successfully!")
        else:
            print("⚠️  Some components failed to initialize")
        
        print()
        print("🌐 Web Dashboard Starting...")
        print("📊 URL: http://localhost:5000")
        print("🏥 Health: http://localhost:5000/health")
        print()
        print("🎮 Dashboard Features:")
        print("  • Overview - System metrics and activity")
        print("  • Messages - Real conversation data")
        print("  • Workflows - Runbook execution tracking")
        print("  • MCPs - Connection management with '+ Add' button")
        print("  • Conversations - User session monitoring")
        print("  • Costs - Real-time cost analytics")
        print()
        print("💡 Tip: Click the MCP tab and try adding a new connection!")
        print()
        print("Press Ctrl+C to stop the dashboard")
        print("-" * 60)
        
        # Run the Flask app
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=False,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Dashboard error: {e}")
        print("\n💡 Troubleshooting:")
        print("   • Check Supabase environment variables")
        print("   • Ensure Flask is installed: pip install flask")
        print("   • Verify database migrations are applied")

if __name__ == "__main__":
    main() 