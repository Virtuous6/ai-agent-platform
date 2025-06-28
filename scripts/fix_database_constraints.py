#!/usr/bin/env python3
"""
Fix database constraints to allow new content types.
This script updates the conversation_embeddings table constraint to include 'learned_pattern' and 'working_memory'.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.supabase import SupabaseLogger
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fix_database_constraints():
    """Update database constraints to fix content_type errors."""
    try:
        logger.info("🔧 Fixing database constraints...")
        
        # Initialize Supabase client
        db_logger = SupabaseLogger()
        
        # Update constraint to allow new content types
        constraint_sql = """
        ALTER TABLE conversation_embeddings 
        DROP CONSTRAINT IF EXISTS conversation_embeddings_content_type_check;
        
        ALTER TABLE conversation_embeddings 
        ADD CONSTRAINT conversation_embeddings_content_type_check 
        CHECK (content_type IN ('message', 'response', 'summary', 'insight', 'learned_pattern', 'working_memory'));
        """
        
        # Execute the constraint update using RPC
        result = db_logger.client.rpc('exec_sql', {'sql_query': constraint_sql}).execute()
        
        logger.info("✅ Database constraint updated successfully")
        logger.info("Content types now allowed: message, response, summary, insight, learned_pattern, working_memory")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update database constraint: {e}")
        logger.info("You may need to run this SQL manually in your Supabase SQL editor:")
        logger.info("""
        ALTER TABLE conversation_embeddings 
        DROP CONSTRAINT IF EXISTS conversation_embeddings_content_type_check;
        
        ALTER TABLE conversation_embeddings 
        ADD CONSTRAINT conversation_embeddings_content_type_check 
        CHECK (content_type IN ('message', 'response', 'summary', 'insight', 'learned_pattern', 'working_memory'));
        """)
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(fix_database_constraints()) 