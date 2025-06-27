#!/usr/bin/env python3
"""
Dashboard Launcher for AI Agent Platform

Launches the retro terminal dashboard with connections to:
- Supabase database for real-time logs
- Agent orchestrator for system metrics
- Event bus for live event streaming
- Cost analytics for financial monitoring

Usage:
    python dashboard/launch_dashboard.py
    
    or from the main directory:
    python -m dashboard.launch_dashboard
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables and paths."""
    # Ensure we have the project root in the path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Check for required environment variables
    required_env_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.info("Dashboard will run with limited functionality")

async def initialize_platform_components():
    """Initialize the AI agent platform components."""
    try:
        # Initialize Supabase logger
        from database.supabase_logger import SupabaseLogger
        db_logger = SupabaseLogger()
        logger.info("✓ Supabase logger initialized")
    except Exception as e:
        logger.warning(f"Could not initialize Supabase logger: {e}")
        db_logger = None
    
    try:
        # Initialize orchestrator (simplified version)
        from orchestrator.agent_orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator(db_logger=db_logger)
        logger.info("✓ Agent orchestrator initialized")
    except Exception as e:
        logger.warning(f"Could not initialize orchestrator: {e}")
        orchestrator = None
    
    try:
        # Initialize event bus
        from events.event_bus import get_event_bus
        event_bus = get_event_bus()
        await event_bus.start()
        logger.info("✓ Event bus initialized")
    except Exception as e:
        logger.warning(f"Could not initialize event bus: {e}")
        event_bus = None
    
    return db_logger, orchestrator, event_bus

async def main():
    """Main dashboard launcher."""
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                               SLACKER                                     ║
    ║                         >>> INITIALIZING <<<                             ║
    ║                                                                           ║
    ║  ░▒▓█ SELF-IMPROVING AI SYSTEM MONITORING █▓▒░                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Setup environment
    setup_environment()
    
    # Initialize platform components
    logger.info("Initializing AI Agent Platform components...")
    db_logger, orchestrator, event_bus = await initialize_platform_components()
    
    # Initialize dashboard
    try:
        from dashboard.terminal_dashboard import TerminalDashboard
        
        dashboard = TerminalDashboard(
            db_logger=db_logger,
            orchestrator=orchestrator,
            event_bus=event_bus
        )
        
        logger.info("✓ Terminal dashboard initialized")
        
        # Start the dashboard
        logger.info("Starting dashboard...")
        print("\n🚀 Launching retro terminal dashboard...")
        print("📊 Real-time monitoring active")
        print("🔄 Press Ctrl+C to exit\n")
        
        # Run dashboard
        dashboard.start()
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        print(f"\n❌ Dashboard error: {e}")
        print("🔧 Check logs for more details")
    finally:
        # Cleanup
        if event_bus:
            await event_bus.stop()
        logger.info("Dashboard shutdown complete")

def sync_main():
    """Synchronous wrapper for async main."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Have a great day!")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sync_main() 