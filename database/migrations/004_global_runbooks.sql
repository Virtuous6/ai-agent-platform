-- Global Runbooks Storage Migration
-- Moves runbooks from YAML files to Supabase database storage

-- ============================================================================
-- GLOBAL RUNBOOKS TABLE
-- ============================================================================

-- Store runbook definitions and metadata
CREATE TABLE IF NOT EXISTS global_runbooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Basic identification
    name TEXT NOT NULL UNIQUE,
    version TEXT NOT NULL DEFAULT '1.0.0',
    
    -- Content and metadata
    description TEXT,
    author TEXT DEFAULT 'AI Agent Platform',
    
    -- Runbook definition (YAML content as JSONB for queryability)
    definition JSONB NOT NULL,
    
    -- Original YAML content for backward compatibility
    yaml_content TEXT,
    
    -- Status and lifecycle
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'archived', 'draft')),
    is_system_runbook BOOLEAN DEFAULT TRUE,
    
    -- Usage and performance tracking
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMPTZ,
    success_rate FLOAT DEFAULT 0.0,
    avg_execution_time FLOAT DEFAULT 0.0,
    
    -- Categorization and discovery
    category TEXT DEFAULT 'general',
    tags TEXT[] DEFAULT '{}',
    priority INTEGER DEFAULT 1 CHECK (priority >= 1 AND priority <= 10),
    
    -- LLM integration context
    llm_context TEXT,
    agent_compatibility TEXT[] DEFAULT '{}', -- ['general', 'technical', 'research']
    
    -- Version control
    previous_version_id UUID REFERENCES global_runbooks(id),
    created_by TEXT DEFAULT 'system',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS global_runbooks_name_idx ON global_runbooks(name);
CREATE INDEX IF NOT EXISTS global_runbooks_status_idx ON global_runbooks(status);
CREATE INDEX IF NOT EXISTS global_runbooks_category_idx ON global_runbooks(category);
CREATE INDEX IF NOT EXISTS global_runbooks_priority_idx ON global_runbooks(priority DESC);
CREATE INDEX IF NOT EXISTS global_runbooks_usage_idx ON global_runbooks(usage_count DESC);
CREATE INDEX IF NOT EXISTS global_runbooks_success_idx ON global_runbooks(success_rate DESC);

-- GIN index for JSONB definition queries
CREATE INDEX IF NOT EXISTS global_runbooks_definition_idx ON global_runbooks USING GIN(definition);

-- GIN index for tags array
CREATE INDEX IF NOT EXISTS global_runbooks_tags_idx ON global_runbooks USING GIN(tags);

-- ============================================================================
-- RUNBOOK TRIGGERS TABLE
-- ============================================================================

-- Store trigger conditions separately for better querying
CREATE TABLE IF NOT EXISTS runbook_triggers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    runbook_id UUID NOT NULL REFERENCES global_runbooks(id) ON DELETE CASCADE,
    
    -- Trigger definition
    condition_type TEXT NOT NULL CHECK (condition_type IN (
        'message_contains', 'agent_mention', 'question_detected', 
        'user_intent', 'context_match', 'pattern_match'
    )),
    
    parameters JSONB DEFAULT '{}',
    priority INTEGER DEFAULT 1,
    
    -- Activation rules
    is_active BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for trigger matching
CREATE INDEX IF NOT EXISTS runbook_triggers_runbook_idx ON runbook_triggers(runbook_id);
CREATE INDEX IF NOT EXISTS runbook_triggers_condition_idx ON runbook_triggers(condition_type);
CREATE INDEX IF NOT EXISTS runbook_triggers_priority_idx ON runbook_triggers(priority DESC);
CREATE INDEX IF NOT EXISTS runbook_triggers_active_idx ON runbook_triggers(is_active);

-- ============================================================================
-- RUNBOOK VERSIONS TABLE
-- ============================================================================

-- Track runbook version history
CREATE TABLE IF NOT EXISTS runbook_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    runbook_id UUID NOT NULL REFERENCES global_runbooks(id) ON DELETE CASCADE,
    
    version TEXT NOT NULL,
    definition JSONB NOT NULL,
    yaml_content TEXT,
    
    -- Change tracking
    changes_summary TEXT,
    changed_by TEXT,
    migration_notes TEXT,
    
    -- Deployment tracking
    deployed_at TIMESTAMPTZ,
    is_current BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for version management
CREATE INDEX IF NOT EXISTS runbook_versions_runbook_idx ON runbook_versions(runbook_id);
CREATE INDEX IF NOT EXISTS runbook_versions_version_idx ON runbook_versions(version);
CREATE INDEX IF NOT EXISTS runbook_versions_current_idx ON runbook_versions(is_current);
CREATE INDEX IF NOT EXISTS runbook_versions_deployed_idx ON runbook_versions(deployed_at DESC);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to update runbook usage statistics
CREATE OR REPLACE FUNCTION update_runbook_usage(runbook_name TEXT, execution_success BOOLEAN, execution_time FLOAT)
RETURNS VOID AS $$
BEGIN
    UPDATE global_runbooks 
    SET 
        usage_count = usage_count + 1,
        last_used = NOW(),
        success_rate = CASE 
            WHEN usage_count = 0 THEN CASE WHEN execution_success THEN 1.0 ELSE 0.0 END
            ELSE (success_rate * usage_count + CASE WHEN execution_success THEN 1.0 ELSE 0.0 END) / (usage_count + 1)
        END,
        avg_execution_time = CASE 
            WHEN usage_count = 0 THEN execution_time
            ELSE (avg_execution_time * usage_count + execution_time) / (usage_count + 1)
        END,
        updated_at = NOW()
    WHERE name = runbook_name;
END;
$$ LANGUAGE plpgsql;

-- Function to get active runbooks by category
CREATE OR REPLACE FUNCTION get_active_runbooks(category_filter TEXT DEFAULT NULL)
RETURNS TABLE(
    id UUID,
    name TEXT,
    version TEXT,
    description TEXT,
    definition JSONB,
    category TEXT,
    priority INTEGER,
    usage_count INTEGER,
    success_rate FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        r.id,
        r.name,
        r.version,
        r.description,
        r.definition,
        r.category,
        r.priority,
        r.usage_count,
        r.success_rate
    FROM global_runbooks r
    WHERE 
        r.status = 'active'
        AND (category_filter IS NULL OR r.category = category_filter)
    ORDER BY r.priority DESC, r.success_rate DESC, r.usage_count DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to find runbooks by trigger conditions
CREATE OR REPLACE FUNCTION find_matching_runbooks(
    message_text TEXT,
    agent_type TEXT DEFAULT NULL,
    user_context JSONB DEFAULT '{}'::JSONB
)
RETURNS TABLE(
    runbook_id UUID,
    runbook_name TEXT,
    trigger_priority INTEGER,
    match_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        r.id as runbook_id,
        r.name as runbook_name,
        t.priority as trigger_priority,
        CASE 
            WHEN t.condition_type = 'message_contains' THEN
                -- Simple keyword matching score
                (SELECT COUNT(*) FROM unnest(string_to_array(lower(t.parameters->>'keywords'::text), ',')) AS keyword
                 WHERE lower(message_text) LIKE '%' || trim(keyword) || '%') * 1.0 / 
                (SELECT COUNT(*) FROM unnest(string_to_array(t.parameters->>'keywords'::text, ',')))
            WHEN t.condition_type = 'agent_mention' THEN
                CASE WHEN agent_type = t.parameters->>'agent' THEN 1.0 ELSE 0.0 END
            ELSE 0.5  -- Default match score for other conditions
        END as match_score
    FROM global_runbooks r
    JOIN runbook_triggers t ON r.id = t.runbook_id
    WHERE 
        r.status = 'active' 
        AND t.is_active = TRUE
        AND (agent_type IS NULL OR agent_type = ANY(r.agent_compatibility))
    ORDER BY t.priority DESC, match_score DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- UPDATE TIMESTAMP TRIGGERS
-- ============================================================================

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_runbook_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_global_runbooks_timestamp
    BEFORE UPDATE ON global_runbooks
    FOR EACH ROW
    EXECUTE FUNCTION update_runbook_timestamp();

-- ============================================================================
-- VIEWS FOR ANALYTICS
-- ============================================================================

-- View for runbook performance dashboard
CREATE OR REPLACE VIEW runbook_performance_dashboard AS
SELECT 
    r.name,
    r.category,
    r.version,
    r.usage_count,
    r.success_rate,
    r.avg_execution_time,
    r.last_used,
    COUNT(t.id) as trigger_count,
    r.status,
    r.priority
FROM global_runbooks r
LEFT JOIN runbook_triggers t ON r.id = t.runbook_id AND t.is_active = TRUE
GROUP BY r.id, r.name, r.category, r.version, r.usage_count, r.success_rate, 
         r.avg_execution_time, r.last_used, r.status, r.priority
ORDER BY r.usage_count DESC, r.success_rate DESC;

-- View for runbook execution correlation
CREATE OR REPLACE VIEW runbook_execution_analytics AS
SELECT 
    r.name as runbook_name,
    r.category,
    r.success_rate as stored_success_rate,
    COUNT(e.id) as total_executions,
    COUNT(e.id) FILTER (WHERE e.status = 'completed') as successful_executions,
    ROUND(AVG(e.execution_time_seconds), 2) as avg_actual_execution_time,
    ROUND(AVG(e.total_tokens), 0) as avg_tokens_used,
    ROUND(SUM(e.estimated_cost), 4) as total_cost,
    MAX(e.started_at) as last_execution_time
FROM global_runbooks r
LEFT JOIN runbook_executions e ON r.name = e.runbook_name
GROUP BY r.id, r.name, r.category, r.success_rate
ORDER BY total_executions DESC;

-- ============================================================================
-- SEED DATA - MIGRATE EXISTING RUNBOOKS
-- ============================================================================

-- Insert the existing answer-question runbook as an example
INSERT INTO global_runbooks (name, version, description, definition, category, tags, llm_context, agent_compatibility, priority)
VALUES (
    'answer-question',
    '1.0.0',
    'Intelligent question answering workflow using general agent and web search',
    '{
        "metadata": {
            "name": "answer-question",
            "version": "1.0.0",
            "description": "Intelligent question answering workflow using general agent and web search",
            "author": "AI Agent Platform"
        },
        "steps": [
            {"id": "classify_question", "action": "analyze_message"},
            {"id": "attempt_direct_answer", "action": "invoke_agent"},
            {"id": "evaluate_response", "action": "check_response_quality"},
            {"id": "perform_web_search", "action": "invoke_tool"},
            {"id": "synthesize_answer", "action": "invoke_agent"},
            {"id": "format_final_response", "action": "format_output"}
        ],
        "configuration": {
            "timeout_seconds": 60,
            "max_retries": 2,
            "log_execution": true,
            "cache_results": true
        }
    }'::JSONB,
    'user_interaction',
    ARRAY['question', 'answer', 'web_search', 'general_agent'],
    'This runbook handles general question answering by first attempting to answer with the general agent''s knowledge base. If the agent indicates uncertainty or the question requires current information, it supplements the response with web search results for comprehensive, accurate answers.',
    ARRAY['general'],
    1
)
ON CONFLICT (name) DO NOTHING;

-- Insert triggers for the answer-question runbook
INSERT INTO runbook_triggers (runbook_id, condition_type, parameters, priority)
SELECT 
    r.id,
    'message_contains',
    '{"keywords": "what,how,why,when,where,who,?"}'::JSONB,
    3
FROM global_runbooks r WHERE r.name = 'answer-question'
ON CONFLICT DO NOTHING;

INSERT INTO runbook_triggers (runbook_id, condition_type, parameters, priority)
SELECT 
    r.id,
    'agent_mention',
    '{"agent": "general"}'::JSONB,
    2
FROM global_runbooks r WHERE r.name = 'answer-question'
ON CONFLICT DO NOTHING;

INSERT INTO runbook_triggers (runbook_id, condition_type, parameters, priority)
SELECT 
    r.id,
    'question_detected',
    '{"confidence_threshold": 0.7}'::JSONB,
    1
FROM global_runbooks r WHERE r.name = 'answer-question'
ON CONFLICT DO NOTHING;

-- ============================================================================
-- COMMENTS AND DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE global_runbooks IS 'Global runbook definitions stored in Supabase instead of YAML files';
COMMENT ON TABLE runbook_triggers IS 'Trigger conditions for runbook activation and routing';
COMMENT ON TABLE runbook_versions IS 'Version history and change tracking for runbooks';

COMMENT ON COLUMN global_runbooks.definition IS 'JSONB representation of runbook YAML structure for querying';
COMMENT ON COLUMN global_runbooks.yaml_content IS 'Original YAML content for backward compatibility';
COMMENT ON COLUMN global_runbooks.agent_compatibility IS 'Array of agent types that can execute this runbook';
COMMENT ON COLUMN global_runbooks.success_rate IS 'Calculated success rate from execution history';

-- Migration complete!
-- 
-- Next steps:
-- 1. Create runbook manager service to interact with these tables
-- 2. Update orchestrator to pull runbooks from Supabase instead of files
-- 3. Create admin interface for runbook management
-- 4. Migrate remaining YAML runbooks to database 