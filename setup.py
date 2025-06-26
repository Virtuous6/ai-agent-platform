#!/usr/bin/env python3
"""
Setup script for AI Agent Platform
Helps with initial configuration and environment setup
"""

import os
import secrets
import string
from pathlib import Path

def generate_secret_key(length=32):
    """Generate a secure random key."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def create_env_file():
    """Create .env file from template with generated secrets."""
    env_content = f"""# ==============================================
# AI Agent Platform Environment Configuration
# ==============================================
# Created by setup.py - Replace placeholder values with your actual credentials

# Slack Configuration
# Get these from your Slack app at https://api.slack.com/apps
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_SIGNING_SECRET=your-signing-secret-here
SLACK_APP_TOKEN=xapp-your-app-token-here

# Supabase Configuration
# Get these from your Supabase project dashboard
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key-here
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here

# AI/LLM Configuration
# Get API keys from respective providers
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Redis Configuration (Optional - can skip for now)
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=

# Environment Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=true

# Analytics and Monitoring
ENABLE_ANALYTICS=true
ENABLE_PERFORMANCE_TRACKING=true

# Security (auto-generated)
ENCRYPTION_KEY={generate_secret_key(32)}
JWT_SECRET={generate_secret_key(64)}
"""
    
    env_path = Path(".env")
    if env_path.exists():
        print("⚠️  .env file already exists. Skipping creation.")
        return False
    
    with open(env_path, "w") as f:
        f.write(env_content)
    
    print("✅ Created .env file with auto-generated secrets")
    print("📝 Please edit .env and add your Slack and Supabase credentials")
    return True

def check_dependencies():
    """Check if required dependencies can be imported."""
    required_packages = [
        ('slack_bolt', 'slack-bolt'),
        ('supabase', 'supabase'), 
        ('openai', 'openai'),
        ('yaml', 'pyyaml')
    ]
    
    missing = []
    for module, package in required_packages:
        try:
            __import__(module)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing.append(package)
    
    if missing:
        print(f"\n📦 Run: pip install {' '.join(missing)}")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def print_slack_setup_instructions():
    """Print detailed Slack app setup instructions."""
    print("""
🔧 SLACK APP SETUP INSTRUCTIONS:

1. Go to https://api.slack.com/apps and click "Create New App"
2. Choose "From scratch" and name it "AI Agent Platform"
3. Select your workspace

4. BASIC INFORMATION:
   - Copy the "Signing Secret" to your .env file (SLACK_SIGNING_SECRET)

5. OAUTH & PERMISSIONS:
   - Add these Bot Token Scopes:
     • app_mentions:read
     • channels:history
     • chat:write
     • im:history
     • im:read
     • im:write
   - Install the app to your workspace
   - Copy the "Bot User OAuth Token" to your .env file (SLACK_BOT_TOKEN)

6. SOCKET MODE:
   - Enable Socket Mode
   - Generate an App-Level Token with connections:write scope
   - Copy this token to your .env file (SLACK_APP_TOKEN)

7. EVENT SUBSCRIPTIONS:
   - Enable Events (this will be handled via Socket Mode)
   - Subscribe to these bot events:
     • app_mention
     • message.im

8. APP HOME:
   - Enable App Home tab

9. INSTALL & TEST:
   - Install the app to your workspace
   - Invite the bot to a channel: /invite @your-bot-name
   - Test with a mention: @your-bot-name hello
""")

def main():
    """Main setup process."""
    print("🤖 AI Agent Platform Setup")
    print("=" * 40)
    
    # Create .env file
    env_created = create_env_file()
    
    # Check dependencies
    print("\n📦 Checking Dependencies...")
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Please install missing dependencies first:")
        print("   pip install -r requirements.txt")
        return
    
    # Show next steps
    print("\n🎯 NEXT STEPS:")
    if env_created:
        print("1. Edit the .env file with your actual credentials")
    print("2. Set up your Slack app (see instructions below)")
    print("3. Add your credentials to .env")
    print("4. Run: python slack_interface/slack_bot.py")
    
    # Print Slack setup instructions
    print_slack_setup_instructions()
    
    print("🚀 Once configured, test the bot by mentioning it in Slack!")

if __name__ == "__main__":
    main() 