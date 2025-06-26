-- Vector Memory & Knowledge Graph Foundation Migration
-- Run this in your Supabase SQL editor or via migration tools

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- VECTOR MEMORY TABLES
-- ============================================================================

-- Conversation embeddings for semantic search and memory retrieval
CREATE TABLE IF NOT EXISTS conversation_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    
    -- Vector embedding (OpenAI ada-002 dimensions)
    embedding vector(1536),
    
    -- Searchable content
    content_summary TEXT NOT NULL,
    content_type TEXT DEFAULT 'message' CHECK (content_type IN ('message', 'response', 'summary', 'insight')),
    
    -- Semantic metadata
    topics JSONB DEFAULT '[]',  -- Extracted topics/themes
    entities JSONB DEFAULT '[]',  -- Named entities, people, concepts
    sentiment FLOAT DEFAULT 0.0,  -- Sentiment score (-1 to 1)
    
    -- Indexing and retrieval
    user_id TEXT NOT NULL,
    channel_id TEXT,
    importance_score FLOAT DEFAULT 0.5,  -- Relevance importance (0-1)
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- High-performance vector similarity search index
CREATE INDEX IF NOT EXISTS conversation_embeddings_vector_idx 
ON conversation_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Standard indexes for filtering
CREATE INDEX IF NOT EXISTS conversation_embeddings_user_idx ON conversation_embeddings(user_id);
CREATE INDEX IF NOT EXISTS conversation_embeddings_conversation_idx ON conversation_embeddings(conversation_id);
CREATE INDEX IF NOT EXISTS conversation_embeddings_created_idx ON conversation_embeddings(created_at DESC);
CREATE INDEX IF NOT EXISTS conversation_embeddings_importance_idx ON conversation_embeddings(importance_score DESC);

-- ============================================================================
-- KNOWLEDGE GRAPH TABLES
-- ============================================================================

-- User relationships and organizational structure
CREATE TABLE IF NOT EXISTS user_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_user_id TEXT NOT NULL,
    to_user_id TEXT NOT NULL,
    
    -- Relationship types
    relationship_type TEXT NOT NULL CHECK (relationship_type IN (
        'collaborates', 'reports_to', 'manages', 'same_team', 'same_project',
        'mentor_mentee', 'expert_in', 'frequent_contact', 'escalation_path'
    )),
    
    -- Relationship strength and metadata
    strength FLOAT DEFAULT 0.5 CHECK (strength >= 0.0 AND strength <= 1.0),
    interaction_count INTEGER DEFAULT 0,
    last_interaction TIMESTAMPTZ,
    
    -- Additional context
    metadata JSONB DEFAULT '{}',  -- Custom relationship data
    created_by TEXT,  -- Who established this relationship
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Prevent duplicate relationships
    UNIQUE(from_user_id, to_user_id, relationship_type)
);

-- Indexes for relationship queries
CREATE INDEX IF NOT EXISTS user_relationships_from_user_idx ON user_relationships(from_user_id);
CREATE INDEX IF NOT EXISTS user_relationships_to_user_idx ON user_relationships(to_user_id);
CREATE INDEX IF NOT EXISTS user_relationships_type_idx ON user_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS user_relationships_strength_idx ON user_relationships(strength DESC);

-- ============================================================================
-- RUNBOOK EXECUTION TRACKING
-- ============================================================================

-- Track runbook executions for analytics and debugging
CREATE TABLE IF NOT EXISTS runbook_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Execution context
    runbook_name TEXT NOT NULL,
    runbook_version TEXT DEFAULT '1.0.0',
    user_id TEXT NOT NULL,
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    
    -- Execution state and progress
    execution_state JSONB DEFAULT '{}',  -- Full LangGraph state
    current_step TEXT,
    completed_steps TEXT[] DEFAULT '{}',
    
    -- Status tracking
    status TEXT DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'paused', 'cancelled')),
    progress_percentage FLOAT DEFAULT 0.0 CHECK (progress_percentage >= 0.0 AND progress_percentage <= 100.0),
    
    -- Timing and performance
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    execution_time_seconds FLOAT,
    
    -- Error handling
    error_message TEXT,
    error_step TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Agent and tool usage
    agents_used TEXT[] DEFAULT '{}',
    tools_used TEXT[] DEFAULT '{}',
    
    -- Cost tracking
    total_tokens INTEGER DEFAULT 0,
    estimated_cost FLOAT DEFAULT 0.0,
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for runbook analytics
CREATE INDEX IF NOT EXISTS runbook_executions_name_idx ON runbook_executions(runbook_name);
CREATE INDEX IF NOT EXISTS runbook_executions_user_idx ON runbook_executions(user_id);
CREATE INDEX IF NOT EXISTS runbook_executions_status_idx ON runbook_executions(status);
CREATE INDEX IF NOT EXISTS runbook_executions_started_idx ON runbook_executions(started_at DESC);
CREATE INDEX IF NOT EXISTS runbook_executions_conversation_idx ON runbook_executions(conversation_id);

-- ============================================================================
-- MEMORY PERFORMANCE TABLES
-- ============================================================================

-- Track vector search performance and quality
CREATE TABLE IF NOT EXISTS vector_search_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Query context
    user_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    query_embedding vector(1536),
    
    -- Search parameters
    similarity_threshold FLOAT DEFAULT 0.7,
    max_results INTEGER DEFAULT 10,
    
    -- Results and performance
    results_count INTEGER DEFAULT 0,
    top_similarity_score FLOAT,
    search_time_ms FLOAT,
    
    -- Quality metrics
    user_clicked BOOLEAN DEFAULT FALSE,  -- Did user interact with results?
    user_rating FLOAT,  -- Optional user feedback (1-5)
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Search performance indexes
CREATE INDEX IF NOT EXISTS vector_search_logs_user_idx ON vector_search_logs(user_id);
CREATE INDEX IF NOT EXISTS vector_search_logs_created_idx ON vector_search_logs(created_at DESC);

-- ============================================================================
-- KNOWLEDGE EXTRACTION TABLES
-- ============================================================================

-- Store extracted knowledge and insights from conversations
CREATE TABLE IF NOT EXISTS conversation_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Source context
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    
    -- Insight data
    insight_type TEXT NOT NULL CHECK (insight_type IN (
        'user_preference', 'expertise_area', 'frequent_question', 
        'problem_pattern', 'solution_pattern', 'knowledge_gap'
    )),
    
    title TEXT NOT NULL,
    description TEXT,
    confidence_score FLOAT DEFAULT 0.5 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    
    -- Extracted data
    entities JSONB DEFAULT '[]',  -- People, tools, technologies mentioned
    keywords TEXT[] DEFAULT '{}',
    categories TEXT[] DEFAULT '{}',
    
    -- Usage tracking
    times_referenced INTEGER DEFAULT 0,
    last_referenced TIMESTAMPTZ,
    
    -- Lifecycle
    is_active BOOLEAN DEFAULT TRUE,
    verified_by TEXT,  -- Optional human verification
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insights indexes
CREATE INDEX IF NOT EXISTS conversation_insights_user_idx ON conversation_insights(user_id);
CREATE INDEX IF NOT EXISTS conversation_insights_type_idx ON conversation_insights(insight_type);
CREATE INDEX IF NOT EXISTS conversation_insights_confidence_idx ON conversation_insights(confidence_score DESC);
CREATE INDEX IF NOT EXISTS conversation_insights_active_idx ON conversation_insights(is_active);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to update vector embedding timestamps
CREATE OR REPLACE FUNCTION update_embedding_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update timestamps
CREATE TRIGGER update_conversation_embeddings_timestamp
    BEFORE UPDATE ON conversation_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_embedding_timestamp();

CREATE TRIGGER update_user_relationships_timestamp
    BEFORE UPDATE ON user_relationships
    FOR EACH ROW
    EXECUTE FUNCTION update_embedding_timestamp();

CREATE TRIGGER update_runbook_executions_timestamp
    BEFORE UPDATE ON runbook_executions
    FOR EACH ROW
    EXECUTE FUNCTION update_embedding_timestamp();

-- ============================================================================
-- VIEWS FOR ANALYTICS
-- ============================================================================

-- View for runbook performance analytics
CREATE OR REPLACE VIEW runbook_performance AS
SELECT 
    runbook_name,
    COUNT(*) as total_executions,
    COUNT(*) FILTER (WHERE status = 'completed') as successful_executions,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_executions,
    ROUND(AVG(execution_time_seconds), 2) as avg_execution_time,
    ROUND(AVG(total_tokens), 0) as avg_tokens_used,
    ROUND(SUM(estimated_cost), 4) as total_cost,
    MAX(started_at) as last_execution
FROM runbook_executions 
GROUP BY runbook_name
ORDER BY total_executions DESC;

-- View for user memory insights
CREATE OR REPLACE VIEW user_memory_summary AS
SELECT 
    user_id,
    COUNT(DISTINCT conversation_id) as conversations_with_memory,
    COUNT(*) as total_embeddings,
    ROUND(AVG(importance_score), 2) as avg_importance,
    array_agg(DISTINCT content_type) as memory_types,
    MAX(created_at) as last_memory_created
FROM conversation_embeddings 
GROUP BY user_id
ORDER BY total_embeddings DESC;

-- ============================================================================
-- SEED DATA (Optional)
-- ============================================================================

-- Insert some default relationship types data for reference
INSERT INTO user_relationships (from_user_id, to_user_id, relationship_type, strength, metadata, created_by)
VALUES 
    ('system', 'admin', 'escalation_path', 1.0, '{"priority": "high", "purpose": "system_issues"}', 'system'),
    ('system', 'support', 'escalation_path', 0.8, '{"priority": "medium", "purpose": "user_support"}', 'system')
ON CONFLICT (from_user_id, to_user_id, relationship_type) DO NOTHING;

-- ============================================================================
-- COMMENTS AND DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE conversation_embeddings IS 'Vector embeddings of conversation content for semantic search and memory retrieval';
COMMENT ON TABLE user_relationships IS 'Graph structure representing user relationships and organizational context';
COMMENT ON TABLE runbook_executions IS 'Execution tracking and analytics for LangGraph runbook workflows';
COMMENT ON TABLE vector_search_logs IS 'Performance monitoring for vector similarity searches';
COMMENT ON TABLE conversation_insights IS 'Extracted knowledge and patterns from user conversations';

COMMENT ON COLUMN conversation_embeddings.embedding IS 'OpenAI ada-002 1536-dimensional vector embedding';
COMMENT ON COLUMN conversation_embeddings.importance_score IS 'Calculated relevance score for memory retrieval prioritization';
COMMENT ON COLUMN user_relationships.strength IS 'Relationship strength from 0.0 (weak) to 1.0 (strong)';
COMMENT ON COLUMN runbook_executions.execution_state IS 'Complete LangGraph state object for workflow debugging';

-- Migration complete!
-- 
-- Next steps:
-- 1. Verify all tables created successfully
-- 2. Test vector similarity search functionality  
-- 3. Set up appropriate RLS policies for security
-- 4. Configure monitoring for performance optimization 