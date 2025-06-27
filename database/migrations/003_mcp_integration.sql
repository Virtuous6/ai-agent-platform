-- MCP (Model Context Protocol) Integration Migration
-- Adds tables for managing external tool connections and usage tracking

-- ============================================================================
-- MCP CONNECTION MANAGEMENT
-- ============================================================================

-- MCP Connections - stores user's external service connections
CREATE TABLE IF NOT EXISTS mcp_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- User and connection identity
    user_id TEXT NOT NULL,
    connection_name TEXT NOT NULL,
    mcp_type TEXT NOT NULL CHECK (mcp_type IN (
        'supabase', 'github', 'slack', 'postgres', 'mongodb', 'redis',
        'notion', 'airtable', 'google_sheets', 'jira', 'linear', 
        'custom_api', 'graphql', 'rest_api'
    )),
    
    -- Connection configuration (non-sensitive data)
    connection_config JSONB NOT NULL DEFAULT '{}',
    credential_reference TEXT, -- Reference to secure credential storage
    
    -- Connection metadata
    display_name TEXT,
    description TEXT,
    connection_url TEXT,
    
    -- Status and health
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'error', 'testing')),
    last_health_check TIMESTAMPTZ,
    health_status JSONB DEFAULT '{"status": "unknown"}',
    
    -- Available tools from this connection
    tools_available JSONB DEFAULT '[]',
    tool_schemas JSONB DEFAULT '{}', -- JSON schemas for each tool
    
    -- Usage tracking
    total_executions INTEGER DEFAULT 0,
    successful_executions INTEGER DEFAULT 0,
    last_used TIMESTAMPTZ,
    
    -- Cost tracking
    estimated_monthly_cost FLOAT DEFAULT 0.0,
    actual_monthly_cost FLOAT DEFAULT 0.0,
    
    -- Security and access
    access_level TEXT DEFAULT 'read_only' CHECK (access_level IN ('read_only', 'read_write', 'admin')),
    allowed_operations TEXT[] DEFAULT '{}',
    rate_limit_per_minute INTEGER DEFAULT 60,
    
    -- Lifecycle
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ, -- Optional expiration for temporary connections
    
    -- Constraints
    UNIQUE(user_id, connection_name)
);

-- Indexes for MCP connections
CREATE INDEX IF NOT EXISTS mcp_connections_user_idx ON mcp_connections(user_id);
CREATE INDEX IF NOT EXISTS mcp_connections_type_idx ON mcp_connections(mcp_type);
CREATE INDEX IF NOT EXISTS mcp_connections_status_idx ON mcp_connections(status);
CREATE INDEX IF NOT EXISTS mcp_connections_last_used_idx ON mcp_connections(last_used DESC);
CREATE INDEX IF NOT EXISTS mcp_connections_created_idx ON mcp_connections(created_at DESC);

-- ============================================================================
-- MCP TOOL USAGE TRACKING
-- ============================================================================

-- Track every MCP tool execution for analytics and cost management
CREATE TABLE IF NOT EXISTS mcp_tool_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Connection and execution context
    connection_id UUID REFERENCES mcp_connections(id) ON DELETE CASCADE,
    agent_id TEXT, -- Which agent executed this tool
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    
    -- Tool execution details
    tool_name TEXT NOT NULL,
    tool_parameters JSONB DEFAULT '{}',
    execution_result JSONB DEFAULT '{}',
    
    -- Performance metrics
    execution_time_ms FLOAT NOT NULL,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_details JSONB,
    retry_count INTEGER DEFAULT 0,
    
    -- Cost and token savings
    tokens_saved INTEGER DEFAULT 0, -- Tokens saved by using tool vs LLM
    estimated_cost FLOAT DEFAULT 0.0,
    
    -- Security and validation
    security_validated BOOLEAN DEFAULT TRUE,
    parameter_validation_result JSONB,
    
    -- Context and user
    user_id TEXT NOT NULL,
    user_query TEXT, -- Original user request that led to tool use
    
    executed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for tool usage analytics
CREATE INDEX IF NOT EXISTS mcp_tool_usage_connection_idx ON mcp_tool_usage(connection_id);
CREATE INDEX IF NOT EXISTS mcp_tool_usage_tool_name_idx ON mcp_tool_usage(tool_name);
CREATE INDEX IF NOT EXISTS mcp_tool_usage_user_idx ON mcp_tool_usage(user_id);
CREATE INDEX IF NOT EXISTS mcp_tool_usage_executed_idx ON mcp_tool_usage(executed_at DESC);
CREATE INDEX IF NOT EXISTS mcp_tool_usage_success_idx ON mcp_tool_usage(success);
CREATE INDEX IF NOT EXISTS mcp_tool_usage_agent_idx ON mcp_tool_usage(agent_id);

-- ============================================================================
-- MCP RUN CARDS (PRE-BUILT TEMPLATES)
-- ============================================================================

-- Pre-built connection templates for quick setup
CREATE TABLE IF NOT EXISTS mcp_run_cards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Card identity
    card_name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    mcp_type TEXT NOT NULL,
    
    -- Card content
    description TEXT NOT NULL,
    long_description TEXT,
    setup_instructions TEXT,
    
    -- Configuration template
    config_template JSONB NOT NULL DEFAULT '{}',
    required_credentials JSONB NOT NULL DEFAULT '[]',
    optional_credentials JSONB DEFAULT '[]',
    
    -- Available tools
    available_tools JSONB NOT NULL DEFAULT '[]',
    tool_descriptions JSONB DEFAULT '{}',
    example_use_cases TEXT[] DEFAULT '{}',
    
    -- Metadata and popularity
    popularity_score INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 1.0,
    average_setup_time_minutes INTEGER DEFAULT 5,
    
    -- Versioning and maintenance
    version TEXT DEFAULT '1.0.0',
    created_by TEXT DEFAULT 'system',
    is_public BOOLEAN DEFAULT TRUE,
    is_featured BOOLEAN DEFAULT FALSE,
    
    -- Support and documentation
    documentation_url TEXT,
    support_contact TEXT,
    tags TEXT[] DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for run cards
CREATE INDEX IF NOT EXISTS mcp_run_cards_type_idx ON mcp_run_cards(mcp_type);
CREATE INDEX IF NOT EXISTS mcp_run_cards_popularity_idx ON mcp_run_cards(popularity_score DESC);
CREATE INDEX IF NOT EXISTS mcp_run_cards_featured_idx ON mcp_run_cards(is_featured);
CREATE INDEX IF NOT EXISTS mcp_run_cards_public_idx ON mcp_run_cards(is_public);

-- ============================================================================
-- MCP SECURITY AND ACCESS CONTROL
-- ============================================================================

-- Track security events and access attempts
CREATE TABLE IF NOT EXISTS mcp_security_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Event context
    connection_id UUID REFERENCES mcp_connections(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    agent_id TEXT,
    
    -- Security event details
    event_type TEXT NOT NULL CHECK (event_type IN (
        'connection_created', 'connection_deleted', 'credential_updated',
        'tool_executed', 'access_denied', 'rate_limit_exceeded',
        'suspicious_activity', 'parameter_injection_attempt',
        'unauthorized_operation', 'connection_compromised'
    )),
    
    -- Event data
    event_description TEXT,
    event_data JSONB DEFAULT '{}',
    severity TEXT DEFAULT 'info' CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    
    -- Request details
    source_ip TEXT,
    user_agent TEXT,
    request_id TEXT,
    
    -- Response and action taken
    action_taken TEXT,
    blocked BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for security monitoring
CREATE INDEX IF NOT EXISTS mcp_security_logs_user_idx ON mcp_security_logs(user_id);
CREATE INDEX IF NOT EXISTS mcp_security_logs_event_type_idx ON mcp_security_logs(event_type);
CREATE INDEX IF NOT EXISTS mcp_security_logs_severity_idx ON mcp_security_logs(severity);
CREATE INDEX IF NOT EXISTS mcp_security_logs_created_idx ON mcp_security_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS mcp_security_logs_blocked_idx ON mcp_security_logs(blocked);

-- ============================================================================
-- MCP ANALYTICS AND INSIGHTS
-- ============================================================================

-- Store insights about tool usage patterns
CREATE TABLE IF NOT EXISTS mcp_usage_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Insight context
    user_id TEXT NOT NULL,
    insight_type TEXT NOT NULL CHECK (insight_type IN (
        'frequently_used_tool', 'cost_optimization', 'new_tool_suggestion',
        'security_recommendation', 'performance_issue', 'usage_pattern'
    )),
    
    -- Insight data
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    recommendation TEXT,
    confidence_score FLOAT DEFAULT 0.5,
    
    -- Supporting data
    supporting_data JSONB DEFAULT '{}',
    affected_connections UUID[] DEFAULT '{}',
    potential_savings FLOAT DEFAULT 0.0,
    
    -- Lifecycle
    is_active BOOLEAN DEFAULT TRUE,
    user_acknowledged BOOLEAN DEFAULT FALSE,
    user_feedback TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ
);

-- Indexes for insights
CREATE INDEX IF NOT EXISTS mcp_usage_insights_user_idx ON mcp_usage_insights(user_id);
CREATE INDEX IF NOT EXISTS mcp_usage_insights_type_idx ON mcp_usage_insights(insight_type);
CREATE INDEX IF NOT EXISTS mcp_usage_insights_active_idx ON mcp_usage_insights(is_active);

-- ============================================================================
-- HELPER FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update MCP connection timestamps
CREATE OR REPLACE FUNCTION update_mcp_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_mcp_connections_timestamp
    BEFORE UPDATE ON mcp_connections
    FOR EACH ROW
    EXECUTE FUNCTION update_mcp_timestamp();

CREATE TRIGGER update_mcp_run_cards_timestamp
    BEFORE UPDATE ON mcp_run_cards
    FOR EACH ROW
    EXECUTE FUNCTION update_mcp_timestamp();

-- Function to increment tool usage counters
CREATE OR REPLACE FUNCTION increment_tool_usage()
RETURNS TRIGGER AS $$
BEGIN
    -- Update connection usage counters
    UPDATE mcp_connections 
    SET 
        total_executions = total_executions + 1,
        successful_executions = CASE WHEN NEW.success THEN successful_executions + 1 ELSE successful_executions END,
        last_used = NOW()
    WHERE id = NEW.connection_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update usage stats
CREATE TRIGGER increment_mcp_tool_usage
    AFTER INSERT ON mcp_tool_usage
    FOR EACH ROW
    EXECUTE FUNCTION increment_tool_usage();

-- ============================================================================
-- ANALYTICS VIEWS
-- ============================================================================

-- View for MCP connection performance
CREATE OR REPLACE VIEW mcp_connection_analytics AS
SELECT 
    c.id,
    c.user_id,
    c.connection_name,
    c.mcp_type,
    c.status,
    c.total_executions,
    c.successful_executions,
    CASE 
        WHEN c.total_executions > 0 
        THEN ROUND((c.successful_executions::FLOAT / c.total_executions * 100), 2)
        ELSE 0 
    END as success_rate_percentage,
    c.last_used,
    COUNT(tu.id) as executions_last_24h,
    ROUND(AVG(tu.execution_time_ms), 2) as avg_execution_time_ms,
    SUM(tu.tokens_saved) as total_tokens_saved
FROM mcp_connections c
LEFT JOIN mcp_tool_usage tu ON c.id = tu.connection_id 
    AND tu.executed_at > NOW() - INTERVAL '24 hours'
GROUP BY c.id, c.user_id, c.connection_name, c.mcp_type, c.status, 
         c.total_executions, c.successful_executions, c.last_used
ORDER BY c.total_executions DESC;

-- View for tool popularity analytics
CREATE OR REPLACE VIEW mcp_tool_popularity AS
SELECT 
    tool_name,
    mcp_type,
    COUNT(*) as total_uses,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT connection_id) as connections_used,
    ROUND(AVG(execution_time_ms), 2) as avg_execution_time,
    COUNT(*) FILTER (WHERE success = true) as successful_uses,
    ROUND(AVG(tokens_saved), 0) as avg_tokens_saved,
    MAX(executed_at) as last_used
FROM mcp_tool_usage tu
JOIN mcp_connections c ON tu.connection_id = c.id
GROUP BY tool_name, mcp_type
ORDER BY total_uses DESC;

-- View for user MCP summary
CREATE OR REPLACE VIEW user_mcp_summary AS
SELECT 
    c.user_id,
    COUNT(DISTINCT c.id) as total_connections,
    COUNT(DISTINCT c.mcp_type) as unique_connection_types,
    COUNT(DISTINCT tu.tool_name) as unique_tools_used,
    SUM(c.total_executions) as total_tool_executions,
    SUM(tu.tokens_saved) as total_tokens_saved,
    ROUND(AVG(tu.execution_time_ms), 2) as avg_tool_execution_time,
    MAX(c.last_used) as last_mcp_activity
FROM mcp_connections c
LEFT JOIN mcp_tool_usage tu ON c.id = tu.connection_id
GROUP BY c.user_id
ORDER BY total_tool_executions DESC;

-- ============================================================================
-- SEED DATA - POPULAR RUN CARDS
-- ============================================================================

-- Insert default run cards for popular services
INSERT INTO mcp_run_cards (card_name, display_name, mcp_type, description, config_template, required_credentials, available_tools, tool_descriptions, example_use_cases, tags)
VALUES 
    (
        'supabase_database',
        'Supabase Database',
        'supabase',
        'Connect to your Supabase database for data operations',
        '{"features": ["database", "auth", "storage"], "default_schema": "public"}',
        '["url", "service_role_key"]',
        '["list_tables", "execute_sql", "get_schema", "insert_data", "update_data", "delete_data", "get_user_data", "manage_auth"]',
        '{"list_tables": "Get all tables in your database", "execute_sql": "Run custom SQL queries", "get_schema": "Get table schemas and relationships"}',
        '["Database queries", "User management", "Data analysis", "Reports generation"]',
        '["database", "sql", "backend", "popular"]'
    ),
    (
        'github_integration',
        'GitHub Integration',
        'github',
        'Connect to GitHub for repository and issue management',
        '{"default_org": "", "include_private": false}',
        '["access_token"]',
        '["search_repos", "get_issues", "create_issue", "get_pull_requests", "create_pr", "get_file_content", "commit_files", "get_repo_stats"]',
        '{"search_repos": "Search repositories", "get_issues": "Get repository issues", "create_issue": "Create new issues"}',
        '["Code reviews", "Issue tracking", "Repository management", "CI/CD monitoring"]',
        '["git", "development", "collaboration", "popular"]'
    ),
    (
        'slack_workspace',
        'Slack Workspace',
        'slack',
        'Connect to your Slack workspace for team communication',
        '{"default_channel": "general", "include_dm": true}',
        '["bot_token", "app_token"]',
        '["send_message", "get_channels", "get_users", "search_messages", "create_channel", "get_channel_history"]',
        '{"send_message": "Send messages to channels", "search_messages": "Search conversation history", "get_users": "Get workspace members"}',
        '["Team communication", "Message broadcasting", "User lookup", "Channel management"]',
        '["communication", "team", "messaging", "popular"]'
    ),
    (
        'postgres_database',
        'PostgreSQL Database',
        'postgres',
        'Connect to any PostgreSQL database',
        '{"ssl_mode": "require", "connection_pool_size": 5}',
        '["host", "port", "database", "username", "password"]',
        '["execute_query", "get_tables", "get_schema", "insert_data", "update_data", "delete_data", "create_backup", "analyze_performance"]',
        '{"execute_query": "Run SQL queries", "get_tables": "List database tables", "analyze_performance": "Get query performance insights"}',
        '["Data analysis", "Database administration", "Performance monitoring", "Data migration"]',
        '["database", "sql", "postgres", "analytics"]'
    )
ON CONFLICT (card_name) DO NOTHING;

-- ============================================================================
-- COMMENTS AND DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE mcp_connections IS 'User connections to external services via Model Context Protocol';
COMMENT ON TABLE mcp_tool_usage IS 'Detailed logging of every MCP tool execution for analytics and cost tracking';
COMMENT ON TABLE mcp_run_cards IS 'Pre-built templates for quick setup of popular service connections';
COMMENT ON TABLE mcp_security_logs IS 'Security event logging for MCP connections and tool usage';
COMMENT ON TABLE mcp_usage_insights IS 'AI-generated insights about user tool usage patterns and optimizations';

COMMENT ON COLUMN mcp_connections.credential_reference IS 'Reference to secure credential storage (env var name, secret manager path, etc.)';
COMMENT ON COLUMN mcp_connections.tools_available IS 'JSON array of tool names available from this connection';
COMMENT ON COLUMN mcp_tool_usage.tokens_saved IS 'Estimated tokens saved by using this tool instead of LLM generation';
COMMENT ON COLUMN mcp_run_cards.config_template IS 'JSON template for connection configuration with default values';

-- Migration complete!
-- 
-- Next steps:
-- 1. Verify all MCP tables created successfully
-- 2. Test connection creation and tool execution flows
-- 3. Set up appropriate RLS policies for multi-tenant security
-- 4. Configure monitoring and alerting for security events 