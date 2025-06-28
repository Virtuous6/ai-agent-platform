"""
Filename: main.py
Purpose: Main entry point for the AI Agent Platform
Dependencies: asyncio, logging, os

Clean, simple entry point that initializes all components.
"""

import asyncio
import logging
import os
import sys
from typing import Optional

# Fix tokenizers parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Suppress FAISS/NumPy deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="faiss")
warnings.filterwarnings("ignore", message=".*numpy.core._multiarray_umath.*")
warnings.filterwarnings("ignore", message=".*builtin type.*has no __module__ attribute.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIAgentPlatform:
    """Main application class that orchestrates all components."""
    
    def __init__(self):
        """Initialize the AI Agent Platform."""
        self.orchestrator = None
        self.event_bus = None
        self.storage = None
        self.tracker = None
        self.learner = None
        self.workflow_engine = None
        self.adapter = None
        
    async def initialize(self):
        """Initialize all components in the correct order."""
        try:
            logger.info("Initializing AI Agent Platform...")
            
            # 1. Initialize storage (Supabase)
            from storage.supabase import SupabaseLogger
            self.storage = SupabaseLogger()
            logger.info("✓ Storage initialized")
            
            # 2. Initialize event bus
            from core.events import EventBus
            self.event_bus = EventBus()
            logger.info("✓ Event bus initialized")
            
            # 3. Initialize orchestrator
            from core.orchestrator import Orchestrator
            self.orchestrator = Orchestrator(
                storage=self.storage,
                event_bus=self.event_bus
            )
            logger.info("✓ Orchestrator initialized")
            
            # 4. Initialize tracker
            from evolution.tracker import WorkflowTracker
            self.tracker = WorkflowTracker(db_logger=self.storage)
            logger.info("✓ Workflow tracker initialized")
            
            # 5. Initialize learner
            from evolution.learner import Learner
            self.learner = Learner(
                storage=self.storage,
                event_bus=self.event_bus
            )
            logger.info("✓ Learner initialized")
            
            # 6. Connect components
            self.orchestrator.set_components(self.tracker, self.learner)
            
            # 7. Initialize workflow engine
            from core.workflow import WorkflowEngine
            self.workflow_engine = WorkflowEngine(orchestrator=self.orchestrator)
            logger.info("✓ Workflow engine initialized")
            
            # 8. Start background tasks
            await self.orchestrator.start_periodic_cleanup()
            logger.info("✓ Background tasks started")
            
            logger.info("AI Agent Platform initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize platform: {str(e)}")
            return False
    
    async def start_slack_adapter(self):
        """Start the Slack adapter if configured."""
        slack_token = os.getenv("SLACK_BOT_TOKEN")
        if not slack_token:
            logger.warning("SLACK_BOT_TOKEN not found, skipping Slack adapter")
            return False
        
        try:
            from adapters.slack import SlackAdapter
            self.adapter = SlackAdapter(
                token=slack_token,
                orchestrator=self.orchestrator
            )
            await self.adapter.start()
            logger.info("✓ Slack adapter started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Slack adapter: {str(e)}")
            return False
    
    async def run_cli(self):
        """Run in CLI mode for testing."""
        logger.info("Starting CLI mode...")
        print("\n🤖 AI Agent Platform - CLI Mode")
        print("Type 'exit' to quit, '/help' for commands\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'exit':
                    break
                    
                if user_input == '/help':
                    print("""
Available commands:
/improve - Improve your last workflow
/save-workflow [name] - Save current workflow
/list-workflows - List your saved workflows
/status - Show system status
/agents - List active agents
/help - Show this help
                    """)
                    continue
                
                # Process through orchestrator
                context = {
                    "user_id": "cli_user",
                    "channel": "cli",
                    "platform": "cli"
                }
                
                response = await self.orchestrator.process(user_input, context)
                print(f"\nAgent: {response}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"CLI error: {str(e)}")
                print(f"Error: {str(e)}")
    
    async def run(self, mode: str = "slack"):
        """Run the platform in the specified mode."""
        # Initialize platform
        if not await self.initialize():
            logger.error("Failed to initialize platform")
            return
        
        try:
            if mode == "slack":
                # Start Slack adapter
                if await self.start_slack_adapter():
                    # Keep running
                    logger.info("Platform running in Slack mode...")
                    while True:
                        await asyncio.sleep(60)  # Keep alive
                else:
                    logger.error("Failed to start Slack adapter, falling back to CLI")
                    await self.run_cli()
                    
            elif mode == "cli":
                # Run in CLI mode
                await self.run_cli()
                
            else:
                logger.error(f"Unknown mode: {mode}")
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown all components."""
        logger.info("Shutting down AI Agent Platform...")
        
        # Stop adapter
        if self.adapter:
            await self.adapter.stop()
        
        # Stop orchestrator
        if self.orchestrator:
            await self.orchestrator.shutdown()
        
        # Close other components
        if self.learner:
            await self.learner.close()
        
        logger.info("Shutdown complete")

async def main():
    """Main entry point."""
    # Determine mode from command line or environment
    mode = "slack"  # Default to Slack
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    elif not os.getenv("SLACK_BOT_TOKEN"):
        mode = "cli"
    
    # Create and run platform
    platform = AIAgentPlatform()
    await platform.run(mode)

if __name__ == "__main__":
    # Run the platform
    asyncio.run(main()) 