#!/usr/bin/env python3
"""
AI Agent Platform Dashboard Demo

This script demonstrates how to launch and use the retro terminal dashboard
for monitoring your self-improving AI agent platform.

Features demonstrated:
- Real-time Supabase log streaming
- Agent performance monitoring
- Cost analytics and optimization
- Event bus integration
- System health tracking

Usage:
    python demo_dashboard.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def demo_dashboard():
    """Demonstrate the dashboard with your AI agent platform."""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                           SLACKER DEMO                                   ║
    ║                                                                           ║
    ║  🚀 Real-time monitoring for your self-improving AI system               ║
    ║  🖥️  Retro terminal interface with live data streaming                   ║
    ║  📊 System health, agent performance, and cost analytics                 ║
    ║  🔄 Event streams and Supabase log integration                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("📋 Pre-flight checks...")
    
    # Check environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if os.getenv(var):
            print(f"  ✓ {var} configured")
        else:
            print(f"  ⚠️  {var} missing (dashboard will run with mock data)")
            missing_vars.append(var)
    
    # Check optional Supabase variables
    supabase_vars = ['SUPABASE_URL', 'SUPABASE_ANON_KEY']
    supabase_configured = all(os.getenv(var) for var in supabase_vars)
    
    if supabase_configured:
        print("  ✓ Supabase configuration detected")
        print("  📡 Real-time log streaming will be enabled")
    else:
        print("  ⚠️  Supabase not configured (using mock data)")
    
    print("\n🔧 Testing dashboard components...")
    
    try:
        # Test dashboard imports
        from dashboard.terminal_dashboard import TerminalDashboard
        from dashboard.utils.real_time_updater import RealTimeUpdater
        from dashboard.styles.colors import TerminalColors
        from dashboard.styles.ascii_art import ASCII_ART
        print("  ✓ Dashboard modules imported successfully")
        
        # Test color schemes
        colors = TerminalColors(scheme="green_terminal")
        print(f"  ✓ Color schemes available: green_terminal, amber_terminal, blue_terminal, matrix")
        
        # Test ASCII art
        header_lines = len(ASCII_ART["header"].split('\n'))
        print(f"  ✓ ASCII art loaded ({header_lines} header lines)")
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return
    
    print("\n🤖 Testing AI agent platform integration...")
    
    try:
        # Test Supabase logger
        from database.supabase_logger import SupabaseLogger
        if supabase_configured:
            db_logger = SupabaseLogger()
            print("  ✓ Supabase logger initialized")
        else:
            db_logger = None
            print("  ⚠️  Supabase logger disabled (no config)")
    except Exception as e:
        print(f"  ⚠️  Supabase logger unavailable: {e}")
        db_logger = None
    
    try:
        # Test orchestrator
        from orchestrator.agent_orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator(db_logger=db_logger)
        print("  ✓ Agent orchestrator initialized")
    except Exception as e:
        print(f"  ⚠️  Agent orchestrator unavailable: {e}")
        orchestrator = None
    
    try:
        # Test event bus
        from events.event_bus import EventBus
        event_bus = EventBus()
        await event_bus.start()
        print("  ✓ Event bus initialized")
    except Exception as e:
        print(f"  ⚠️  Event bus unavailable: {e}")
        event_bus = None
    
    print("\n📊 Initializing dashboard...")
    
    # Initialize dashboard
    dashboard = TerminalDashboard(
        db_logger=db_logger,
        orchestrator=orchestrator,
        event_bus=event_bus
    )
    
    print("  ✓ Terminal dashboard initialized")
    
    # Test data gathering
    print("\n🔄 Testing real-time data gathering...")
    
    try:
        data = await dashboard.get_api_data()
        
        # Show sample data structure
        print("  ✓ Data gathering successful")
        print(f"  📈 System health: {data.get('overview', {}).get('system_health', {}).get('overall_score', 'N/A')}")
        print(f"  🤖 Active agents: {data.get('overview', {}).get('agent_ecosystem', {}).get('active_count', 'N/A')}")
        print(f"  💰 Daily cost: ${data.get('costs', {}).get('daily_cost', 'N/A')}")
        print(f"  📝 Recent logs: {len(data.get('logs', {}).get('recent_logs', []))} entries")
        
    except Exception as e:
        print(f"  ⚠️  Data gathering error: {e}")
    
    print("\n🎮 Dashboard Controls:")
    print("  [1] System Overview    - Health metrics and activity")
    print("  [2] Agent Monitoring   - Performance and efficiency")
    print("  [3] Cost Analytics     - Financial optimization")
    print("  [4] Event Stream       - Real-time system events")
    print("  [5] Real-time Logs     - Live Supabase streaming")
    print("  [R] Refresh Data       - Force data update")
    print("  [Q] Quit Dashboard     - Exit gracefully")
    
    print("\n🚀 Launching dashboard...")
    print("⌨️  Use keyboard navigation (numbers 1-5)")
    print("🔄 Data refreshes every 2 seconds")
    print("💡 Press Q to quit when done")
    print("─" * 75)
    
    # Give user a moment to read
    await asyncio.sleep(2)
    
    try:
        # Launch the dashboard
        dashboard.start()
        
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard demo completed!")
    except Exception as e:
        print(f"\n\n❌ Dashboard error: {e}")
        print("💡 This might happen if running in an IDE console.")
        print("🔧 Try running in a proper terminal for full experience.")
    finally:
        # Cleanup
        if event_bus:
            await event_bus.stop()
        print("\n✨ Demo cleanup completed")

def main():
    """Main demo function."""
    
    print("🎯 Starting SLACKER Dashboard Demo...")
    
    try:
        asyncio.run(demo_dashboard())
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Demo error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("  - Run in a proper terminal (not IDE console)")
        print("  - Ensure Python 3.8+ is installed")
        print("  - Check that all dependencies are installed: pip install -r requirements.txt")
        print("  - Verify environment variables are set")
    
    print("\n📚 For more information, see: dashboard/README.md")
    print("🚀 Ready to launch SLACKER? Run: python dashboard/launch_dashboard.py")

if __name__ == "__main__":
    main() 