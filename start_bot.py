#!/usr/bin/env python3
"""
Startup script for AI Agent Platform
Ensures proper environment and starts the Slack bot with the new simplified architecture
"""

import os
import sys
import asyncio
import logging
import signal
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_venv():
    """Check if we're running in a virtual environment."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def check_env_file():
    """Check if .env file exists and has required variables."""
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found!")
        print("🔧 Run: python setup.py")
        return False
    
    # Read .env file and check for placeholder values
    with open(env_path) as f:
        env_content = f.read()
    
    if "your-bot-token-here" in env_content:
        print("⚠️  .env file contains placeholder values!")
        print("📝 Please edit .env with your actual Slack credentials")
        print("💡 Follow the Slack app setup instructions from: python setup.py")
        return False
    
    return True

async def initialize_platform():
    """Initialize the platform components."""
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        # Initialize core components
        from storage.supabase import SupabaseLogger
        from core.events import EventBus
        from evolution.tracker import WorkflowTracker
        from evolution.learner import Learner
        from core.orchestrator import Orchestrator
        
        # Initialize components
        logger.info("🔧 Initializing platform components...")
        
        storage = SupabaseLogger()
        event_bus = EventBus()
        tracker = WorkflowTracker(db_logger=storage)
        learner = Learner(storage=storage)
        
        # Initialize orchestrator with components
        orchestrator = Orchestrator(storage=storage, event_bus=event_bus)
        orchestrator.set_components(tracker, learner)
        
        logger.info("✅ Platform components initialized")
        return orchestrator, event_bus, storage
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize platform: {e}")
        raise

async def start_slack_bot(orchestrator, event_bus, storage):
    """Start the Slack bot with the initialized platform."""
    bot = None
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        """Handle shutdown signals gracefully."""
        logger.info("\n🛑 Shutdown signal received...")
        shutdown_event.set()
    
    # Set up signal handlers for graceful shutdown
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
    
    try:
        # Import the new Slack adapter
        from adapters.slack import SlackBot
        
        # Initialize and start the bot
        bot = SlackBot(orchestrator=orchestrator, event_bus=event_bus, storage=storage)
        
        # Start bot in background task
        bot_task = asyncio.create_task(bot.start())
        
        # Wait for either bot completion or shutdown signal
        done, pending = await asyncio.wait(
            [bot_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Check if bot task completed with error
        for task in done:
            if task == bot_task and not task.cancelled():
                try:
                    await task  # This will raise any exception that occurred
                except Exception as e:
                    logger.error(f"❌ Bot error: {e}")
                    raise
        
    except Exception as e:
        if "KeyboardInterrupt" not in str(type(e)):
            logger.error(f"❌ Error running bot: {e}")
            import traceback
            traceback.print_exc()
    finally:
        # Cleanup
        if bot:
            await bot.stop()

async def main():
    """Main startup process."""
    print("🤖 Starting AI Agent Platform Bot")
    print("=" * 40)
    
    # Check virtual environment
    if not check_venv():
        print("⚠️  Not running in virtual environment!")
        print("🔧 Activate it with: source venv/bin/activate")
        return
    
    # Check .env file
    if not check_env_file():
        return
    
    print("✅ Environment looks good!")
    print("🚀 Initializing platform...")
    
    try:
        # Add current directory to Python path for proper imports
        current_dir = str(Path(__file__).parent.absolute())
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Initialize platform and start bot
        orchestrator, event_bus, storage = await initialize_platform()
        
        print("🤖 Starting Slack bot...")
        print("📝 Use Ctrl+C to stop the bot")
        print("-" * 40)
        
        await start_slack_bot(orchestrator, event_bus, storage)
        
    except Exception as e:
        if "KeyboardInterrupt" not in str(type(e)):
            print(f"❌ Error starting bot: {e}")
            print("🔧 Check your .env configuration and try again")
            import traceback
            traceback.print_exc()
    
    print("\n👋 Bot stopped gracefully")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Silently handle KeyboardInterrupt at the top level
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc() 