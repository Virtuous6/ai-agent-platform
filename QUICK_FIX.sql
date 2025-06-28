-- QUICK FIX: Update conversation_embeddings constraint
-- Run this in your Supabase SQL Editor to fix content_type errors

-- Step 1: Drop the existing constraint
ALTER TABLE conversation_embeddings 
DROP CONSTRAINT IF EXISTS conversation_embeddings_content_type_check;

-- Step 2: Add updated constraint with new content types
ALTER TABLE conversation_embeddings 
ADD CONSTRAINT conversation_embeddings_content_type_check 
CHECK (content_type IN ('message', 'response', 'summary', 'insight', 'learned_pattern', 'working_memory'));

-- Verify the constraint was updated
SELECT conname, consrc 
FROM pg_constraint 
WHERE conname = 'conversation_embeddings_content_type_check'; 