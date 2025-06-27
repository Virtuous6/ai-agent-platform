#!/usr/bin/env python3
"""
Startup script for AI Agent Platform
Ensures proper environment and starts the Slack bot
"""

import os
import sys
import subprocess
from pathlib import Path

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

def main():
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
    print("🚀 Starting Slack bot...")
    print("📝 Use Ctrl+C to stop the bot")
    print("-" * 40)
    
    try:
        # Add current directory to Python path for proper imports
        current_dir = str(Path(__file__).parent.absolute())
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import and run the slack bot directly instead of subprocess
        from slack_interface.slack_bot import main as slack_main
        import asyncio
        
        # Run the async slack bot
        asyncio.run(slack_main())
        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        print("🔧 Check your .env configuration and try again")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 