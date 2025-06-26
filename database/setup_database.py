#!/usr/bin/env python3
"""
Database setup script for AI Agent Platform
Creates all necessary tables and indexes in Supabase
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from supabase_logger import SupabaseLogger
except ImportError:
    # If running from project root
    sys.path.insert(0, str(Path(__file__).parent))
    from supabase_logger import SupabaseLogger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_database():
    """Set up the database schema for the AI Agent Platform."""
    print("🗄️  AI Agent Platform Database Setup")
    print("=" * 50)
    
    try:
        # Initialize logger (this will create the client)
        supabase_logger = SupabaseLogger()
        
        print("✅ Connected to Supabase")
        print(f"   Project ID: {supabase_logger.project_id}")
        print(f"   URL: {supabase_logger.supabase_url}")
        
        # Perform health check
        health = supabase_logger.health_check()
        if health["status"] != "healthy":
            print("❌ Health check failed!")
            print(f"   Error: {health.get('error', 'Unknown error')}")
            return False
        
        print("✅ Health check passed")
        
        # Note: The current Supabase Python library doesn't support direct SQL execution
        # for schema creation. In production, you would typically use Supabase migrations
        # or the SQL editor in the Supabase dashboard.
        
        print("\n📋 Required Database Schema:")
        print("=" * 30)
        
        schema_sql = """
-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    thread_ts TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'archived', 'closed')),
    assigned_agent TEXT,
    message_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    content TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    message_type TEXT NOT NULL CHECK (message_type IN ('user_message', 'bot_response', 'system', 'error')),
    agent_type TEXT,
    agent_response JSONB,
    routing_confidence FLOAT,
    escalation_suggestion TEXT,
    processing_time_ms FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id TEXT PRIMARY KEY,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Agent metrics table
CREATE TABLE IF NOT EXISTS agent_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name TEXT NOT NULL,
    date DATE DEFAULT CURRENT_DATE,
    request_count INTEGER DEFAULT 0,
    response_time_avg FLOAT DEFAULT 0,
    success_rate FLOAT DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    escalation_count INTEGER DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(agent_name, date)
);

-- Routing decisions table (for analytics)
CREATE TABLE IF NOT EXISTS routing_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    selected_agent TEXT NOT NULL,
    confidence_score FLOAT NOT NULL,
    all_scores JSONB,
    routing_time_ms FLOAT,
    was_explicit_mention BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_channel_id ON conversations(channel_id);
CREATE INDEX IF NOT EXISTS idx_conversations_last_activity ON conversations(last_activity);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_message_type ON messages(message_type);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_date ON agent_metrics(date);
CREATE INDEX IF NOT EXISTS idx_routing_decisions_created_at ON routing_decisions(created_at);
"""
        
        print(schema_sql)
        
        print("\n📝 Setup Instructions:")
        print("=" * 30)
        print("1. Go to your Supabase dashboard:")
        print(f"   https://supabase.com/dashboard/project/{supabase_logger.project_id}")
        print("2. Navigate to SQL Editor")
        print("3. Copy and paste the SQL schema above")
        print("4. Run the SQL to create tables and indexes")
        print("5. Come back and run this script again to verify")
        
        # Try to test if tables exist by attempting a simple query
        try:
            result = supabase_logger.client.table("conversations").select("id").limit(1).execute()
            print("\n✅ Tables appear to be set up correctly!")
            print("🚀 Database is ready for the AI Agent Platform")
            return True
        except Exception as e:
            print(f"\n⚠️  Tables may not exist yet. Error: {str(e)}")
            print("Please follow the setup instructions above.")
            return False
            
    except Exception as e:
        print(f"❌ Setup failed: {str(e)}")
        return False

async def test_database_operations():
    """Test basic database operations."""
    print("\n🧪 Testing Database Operations")
    print("=" * 30)
    
    try:
        logger_instance = SupabaseLogger()
        
        # Test conversation creation
        conv_id = await logger_instance.log_conversation_start(
            user_id="test_user",
            channel_id="test_channel"
        )
        
        if conv_id:
            print(f"✅ Created test conversation: {conv_id}")
            
            # Test message logging
            success = await logger_instance.log_message(
                conversation_id=conv_id,
                user_id="test_user",
                content="Hello, this is a test message",
                message_type="user_message"
            )
            
            if success:
                print("✅ Logged test message")
                
                # Test agent metrics
                success = await logger_instance.update_agent_metrics(
                    agent_name="test_agent",
                    response_time_ms=150.5
                )
                
                if success:
                    print("✅ Updated agent metrics")
                    
                    # Clean up test data
                    await logger_instance.close_conversation(conv_id)
                    print("✅ Cleaned up test conversation")
                    
                    print("\n🎉 All database operations working correctly!")
                    return True
        
        print("❌ Database operations failed")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting database setup...")
    
    # Check environment variables
    if not os.getenv("SUPABASE_KEY"):
        print("❌ SUPABASE_KEY environment variable not found!")
        print("Please set up your .env file with Supabase credentials")
        sys.exit(1)
    
    # Run setup
    setup_success = asyncio.run(setup_database())
    
    if setup_success:
        # Run tests
        test_success = asyncio.run(test_database_operations())
        
        if test_success:
            print("\n🚀 Database setup complete and tested!")
        else:
            print("\n⚠️  Setup complete but tests failed")
    else:
        print("\n❌ Database setup incomplete") 