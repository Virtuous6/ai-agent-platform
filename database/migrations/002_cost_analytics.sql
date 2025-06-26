-- Cost Analytics & Optimization Migration
-- Enhanced cost tracking and optimization tables for AI Agent Platform

-- ============================================================================
-- COST ANALYTICS TABLES
-- ============================================================================

-- Cost optimization opportunities tracking
CREATE TABLE IF NOT EXISTS cost_optimizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Optimization details
    optimization_type TEXT NOT NULL CHECK (optimization_type IN (
        'prompt_compression', 'model_downgrade', 'intelligent_caching',
        'batch_processing', 'temperature_optimization', 'token_limit_optimization',
        'context_pruning', 'response_caching'
    )),
    
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    
    -- Affected components
    affected_agents TEXT[] DEFAULT '{}',
    agent_types TEXT[] DEFAULT '{}',
    
    -- Cost impact
    current_cost DECIMAL(10,6) DEFAULT 0.0,
    projected_cost DECIMAL(10,6) DEFAULT 0.0,
    potential_savings DECIMAL(10,6) DEFAULT 0.0,
    savings_percentage DECIMAL(5,2) DEFAULT 0.0,
    
    -- Implementation details
    implementation_complexity TEXT DEFAULT 'medium' CHECK (implementation_complexity IN ('low', 'medium', 'high')),
    risk_level TEXT DEFAULT 'low' CHECK (risk_level IN ('low', 'medium', 'high')),
    implementation_steps JSONB DEFAULT '[]',
    
    -- Impact assessment
    expected_impact JSONB DEFAULT '{}',
    confidence DECIMAL(3,2) DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    priority INTEGER DEFAULT 3 CHECK (priority >= 1 AND priority <= 5),
    
    -- Status tracking
    status TEXT DEFAULT 'identified' CHECK (status IN ('identified', 'planned', 'testing', 'implemented', 'failed', 'cancelled')),
    
    -- Applied optimization tracking
    applied_at TIMESTAMPTZ,
    applied_by TEXT,
    results JSONB DEFAULT '{}',
    actual_savings DECIMAL(10,6),
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for cost optimization queries
CREATE INDEX IF NOT EXISTS cost_optimizations_type_idx ON cost_optimizations(optimization_type);
CREATE INDEX IF NOT EXISTS cost_optimizations_status_idx ON cost_optimizations(status);
CREATE INDEX IF NOT EXISTS cost_optimizations_priority_idx ON cost_optimizations(priority DESC);
CREATE INDEX IF NOT EXISTS cost_optimizations_savings_idx ON cost_optimizations(potential_savings DESC);
CREATE INDEX IF NOT EXISTS cost_optimizations_created_idx ON cost_optimizations(created_at DESC);

-- ============================================================================
-- INTELLIGENT CACHING TABLES
-- ============================================================================

-- Query response cache with similarity matching
CREATE TABLE IF NOT EXISTS query_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Query identification
    query_hash TEXT NOT NULL UNIQUE,
    query_text TEXT NOT NULL,
    query_normalized TEXT NOT NULL,  -- Normalized for similarity matching
    
    -- Response data
    response_text TEXT NOT NULL,
    response_quality_score DECIMAL(3,2) DEFAULT 0.8,
    
    -- Cost metrics
    tokens_saved INTEGER DEFAULT 0,
    cost_saved DECIMAL(10,6) DEFAULT 0.0,
    original_tokens INTEGER DEFAULT 0,
    original_cost DECIMAL(10,6) DEFAULT 0.0,
    
    -- Usage tracking
    hit_count INTEGER DEFAULT 0,
    miss_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    
    -- Cache management
    expiry TIMESTAMPTZ,
    ttl_hours INTEGER DEFAULT 24,
    cache_size_bytes INTEGER,
    
    -- Context and metadata
    agent_context JSONB DEFAULT '{}',
    user_context JSONB DEFAULT '{}',
    cache_tags TEXT[] DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- High-performance indexes for cache lookup
CREATE INDEX IF NOT EXISTS query_cache_hash_idx ON query_cache(query_hash);
CREATE INDEX IF NOT EXISTS query_cache_normalized_idx ON query_cache(query_normalized);
CREATE INDEX IF NOT EXISTS query_cache_expiry_idx ON query_cache(expiry) WHERE expiry IS NOT NULL;
CREATE INDEX IF NOT EXISTS query_cache_accessed_idx ON query_cache(last_accessed DESC);
CREATE INDEX IF NOT EXISTS query_cache_tags_idx ON query_cache USING GIN(cache_tags);

-- ============================================================================
-- COST TRACKING ENHANCEMENTS
-- ============================================================================

-- Enhanced agent cost metrics (extends existing token_usage)
CREATE TABLE IF NOT EXISTS agent_cost_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Agent identification
    agent_id TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    specialty TEXT,
    
    -- Time period
    date DATE NOT NULL,
    hour INTEGER CHECK (hour >= 0 AND hour <= 23),
    
    -- Cost metrics
    total_cost DECIMAL(10,6) DEFAULT 0.0,
    total_tokens INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_requests INTEGER DEFAULT 0,
    successful_requests INTEGER DEFAULT 0,
    
    -- Efficiency metrics
    cost_per_token DECIMAL(10,8) DEFAULT 0.0,
    cost_per_request DECIMAL(10,6) DEFAULT 0.0,
    cost_per_success DECIMAL(10,6) DEFAULT 0.0,
    efficiency_score DECIMAL(3,2) DEFAULT 0.0,
    
    -- Model usage distribution
    model_distribution JSONB DEFAULT '{}',
    
    -- Performance metrics
    avg_response_time DECIMAL(8,3) DEFAULT 0.0,
    success_rate DECIMAL(3,2) DEFAULT 0.0,
    user_satisfaction DECIMAL(3,2) DEFAULT 0.0,
    
    -- Optimization tracking
    optimizations_applied INTEGER DEFAULT 0,
    cache_hit_rate DECIMAL(3,2) DEFAULT 0.0,
    prompt_efficiency_score DECIMAL(3,2) DEFAULT 0.0,
    
    -- Trend indicators
    cost_trend TEXT DEFAULT 'stable' CHECK (cost_trend IN ('increasing', 'decreasing', 'stable')),
    usage_trend TEXT DEFAULT 'stable' CHECK (usage_trend IN ('increasing', 'decreasing', 'stable')),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Unique constraint for agent/date/hour
    UNIQUE(agent_id, date, hour)
);

-- Indexes for cost metrics analysis
CREATE INDEX IF NOT EXISTS agent_cost_metrics_agent_idx ON agent_cost_metrics(agent_id);
CREATE INDEX IF NOT EXISTS agent_cost_metrics_date_idx ON agent_cost_metrics(date DESC);
CREATE INDEX IF NOT EXISTS agent_cost_metrics_cost_idx ON agent_cost_metrics(total_cost DESC);
CREATE INDEX IF NOT EXISTS agent_cost_metrics_efficiency_idx ON agent_cost_metrics(efficiency_score DESC);
CREATE INDEX IF NOT EXISTS agent_cost_metrics_type_idx ON agent_cost_metrics(agent_type);

-- ============================================================================
-- COST ISSUE TRACKING
-- ============================================================================

-- Cost issues and alerts
CREATE TABLE IF NOT EXISTS cost_issues (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Issue identification
    issue_type TEXT NOT NULL CHECK (issue_type IN (
        'high_token_usage', 'expensive_model_overuse', 'redundant_queries',
        'inefficient_prompts', 'unnecessary_context', 'suboptimal_temperature',
        'token_waste', 'cache_miss_rate', 'cost_spike', 'budget_exceeded'
    )),
    
    severity TEXT DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    
    -- Affected components
    affected_agents TEXT[] DEFAULT '{}',
    affected_operations TEXT[] DEFAULT '{}',
    
    -- Cost impact
    cost_impact DECIMAL(10,6) DEFAULT 0.0,
    frequency INTEGER DEFAULT 1,
    first_detected TIMESTAMPTZ DEFAULT NOW(),
    last_detected TIMESTAMPTZ DEFAULT NOW(),
    
    -- Issue details
    issue_data JSONB DEFAULT '{}',
    root_cause TEXT,
    suggested_actions TEXT[] DEFAULT '{}',
    
    -- Resolution tracking
    status TEXT DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'resolved', 'ignored')),
    resolved_at TIMESTAMPTZ,
    resolved_by TEXT,
    resolution_details JSONB DEFAULT '{}',
    
    -- Prevention
    prevention_applied BOOLEAN DEFAULT FALSE,
    prevention_details JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for cost issue management
CREATE INDEX IF NOT EXISTS cost_issues_type_idx ON cost_issues(issue_type);
CREATE INDEX IF NOT EXISTS cost_issues_severity_idx ON cost_issues(severity);
CREATE INDEX IF NOT EXISTS cost_issues_status_idx ON cost_issues(status);
CREATE INDEX IF NOT EXISTS cost_issues_detected_idx ON cost_issues(last_detected DESC);
CREATE INDEX IF NOT EXISTS cost_issues_impact_idx ON cost_issues(cost_impact DESC);

-- ============================================================================
-- COST ANALYSIS VIEWS
-- ============================================================================

-- Daily cost summary view with optimization insights
CREATE OR REPLACE VIEW daily_cost_analysis AS
SELECT 
    date,
    SUM(total_cost) as daily_cost,
    SUM(total_tokens) as daily_tokens,
    SUM(total_requests) as daily_requests,
    AVG(efficiency_score) as avg_efficiency,
    AVG(cost_per_request) as avg_cost_per_request,
    AVG(cache_hit_rate) as avg_cache_hit_rate,
    COUNT(DISTINCT agent_id) as active_agents,
    
    -- Cost breakdown by model
    JSONB_OBJECT_AGG(
        'model_costs',
        (
            SELECT JSONB_OBJECT_AGG(key, value::DECIMAL)
            FROM JSONB_EACH_TEXT(
                JSONB_CONCAT_AGG(model_distribution)
            )
        )
    ) as model_cost_breakdown,
    
    -- Optimization opportunities
    (
        SELECT COUNT(*)
        FROM cost_optimizations co
        WHERE co.status = 'identified'
        AND co.created_at::DATE = agent_cost_metrics.date
    ) as optimization_opportunities,
    
    -- Issues count
    (
        SELECT COUNT(*)
        FROM cost_issues ci
        WHERE ci.last_detected::DATE = agent_cost_metrics.date
        AND ci.status = 'open'
    ) as active_issues

FROM agent_cost_metrics
GROUP BY date
ORDER BY date DESC;

-- Agent cost efficiency ranking
CREATE OR REPLACE VIEW agent_efficiency_ranking AS
SELECT 
    agent_id,
    agent_type,
    specialty,
    
    -- Cost metrics (last 7 days)
    SUM(total_cost) as weekly_cost,
    SUM(total_tokens) as weekly_tokens,
    SUM(total_requests) as weekly_requests,
    AVG(efficiency_score) as avg_efficiency,
    AVG(cost_per_request) as avg_cost_per_request,
    AVG(success_rate) as avg_success_rate,
    
    -- Ranking scores
    RANK() OVER (ORDER BY AVG(efficiency_score) DESC) as efficiency_rank,
    RANK() OVER (ORDER BY AVG(cost_per_request) ASC) as cost_efficiency_rank,
    RANK() OVER (ORDER BY SUM(total_cost) DESC) as total_cost_rank,
    
    -- Optimization potential
    (
        SELECT COUNT(*)
        FROM cost_optimizations co
        WHERE co.affected_agents @> ARRAY[agent_cost_metrics.agent_id]
        AND co.status = 'identified'
    ) as optimization_count,
    
    (
        SELECT COALESCE(SUM(potential_savings), 0)
        FROM cost_optimizations co
        WHERE co.affected_agents @> ARRAY[agent_cost_metrics.agent_id]
        AND co.status = 'identified'
    ) as potential_savings

FROM agent_cost_metrics
WHERE date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY agent_id, agent_type, specialty
ORDER BY avg_efficiency DESC, avg_cost_per_request ASC;

-- ============================================================================
-- COST OPTIMIZATION FUNCTIONS
-- ============================================================================

-- Function to calculate cost savings from optimization
CREATE OR REPLACE FUNCTION calculate_optimization_impact(
    optimization_id UUID,
    baseline_days INTEGER DEFAULT 7
) RETURNS TABLE (
    estimated_daily_savings DECIMAL,
    estimated_monthly_savings DECIMAL,
    affected_operations INTEGER,
    confidence_score DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(co.potential_savings / 30, 0) as estimated_daily_savings,
        COALESCE(co.potential_savings, 0) as estimated_monthly_savings,
        ARRAY_LENGTH(co.affected_agents, 1) as affected_operations,
        co.confidence as confidence_score
    FROM cost_optimizations co
    WHERE co.id = optimization_id;
END;
$$ LANGUAGE plpgsql;

-- Function to identify cost anomalies
CREATE OR REPLACE FUNCTION detect_cost_anomalies(
    lookback_days INTEGER DEFAULT 7,
    threshold_multiplier DECIMAL DEFAULT 2.0
) RETURNS TABLE (
    agent_id TEXT,
    anomaly_type TEXT,
    current_cost DECIMAL,
    baseline_cost DECIMAL,
    cost_increase DECIMAL,
    severity TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH baseline_costs AS (
        SELECT 
            acm.agent_id,
            AVG(acm.total_cost) as avg_baseline_cost,
            STDDEV(acm.total_cost) as stddev_cost
        FROM agent_cost_metrics acm
        WHERE acm.date >= CURRENT_DATE - INTERVAL '30 days'
        AND acm.date < CURRENT_DATE - INTERVAL '7 days'
        GROUP BY acm.agent_id
    ),
    recent_costs AS (
        SELECT 
            acm.agent_id,
            AVG(acm.total_cost) as avg_recent_cost
        FROM agent_cost_metrics acm
        WHERE acm.date >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY acm.agent_id
    )
    SELECT 
        rc.agent_id,
        'cost_spike' as anomaly_type,
        rc.avg_recent_cost as current_cost,
        bc.avg_baseline_cost as baseline_cost,
        (rc.avg_recent_cost - bc.avg_baseline_cost) as cost_increase,
        CASE 
            WHEN rc.avg_recent_cost > bc.avg_baseline_cost * 3 THEN 'critical'
            WHEN rc.avg_recent_cost > bc.avg_baseline_cost * 2 THEN 'high'
            WHEN rc.avg_recent_cost > bc.avg_baseline_cost * 1.5 THEN 'medium'
            ELSE 'low'
        END as severity
    FROM recent_costs rc
    JOIN baseline_costs bc ON rc.agent_id = bc.agent_id
    WHERE rc.avg_recent_cost > bc.avg_baseline_cost * threshold_multiplier
    ORDER BY cost_increase DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ============================================================================

-- Function to update cost metrics timestamp
CREATE OR REPLACE FUNCTION update_cost_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for timestamp updates
CREATE TRIGGER update_cost_optimizations_timestamp
    BEFORE UPDATE ON cost_optimizations
    FOR EACH ROW
    EXECUTE FUNCTION update_cost_timestamp();

CREATE TRIGGER update_query_cache_timestamp
    BEFORE UPDATE ON query_cache
    FOR EACH ROW
    EXECUTE FUNCTION update_cost_timestamp();

CREATE TRIGGER update_agent_cost_metrics_timestamp
    BEFORE UPDATE ON agent_cost_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_cost_timestamp();

CREATE TRIGGER update_cost_issues_timestamp
    BEFORE UPDATE ON cost_issues
    FOR EACH ROW
    EXECUTE FUNCTION update_cost_timestamp();

-- ============================================================================
-- INITIAL DATA AND CONFIGURATION
-- ============================================================================

-- Insert default cost optimization templates
INSERT INTO cost_optimizations (
    optimization_type, title, description, implementation_complexity, 
    risk_level, priority, confidence, status
) VALUES 
(
    'prompt_compression',
    'Optimize verbose system prompts',
    'Compress system prompts to reduce token usage while maintaining effectiveness',
    'low', 'low', 4, 0.85, 'identified'
),
(
    'model_downgrade',
    'Use GPT-3.5 for simple tasks',
    'Automatically route simple queries to cheaper models',
    'medium', 'medium', 3, 0.75, 'identified'
),
(
    'intelligent_caching',
    'Implement response caching',
    'Cache similar queries to reduce redundant LLM calls',
    'medium', 'low', 5, 0.90, 'identified'
)
ON CONFLICT DO NOTHING;

-- Create cost monitoring configuration
CREATE TABLE IF NOT EXISTS cost_monitoring_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_name TEXT UNIQUE NOT NULL,
    
    -- Thresholds
    daily_cost_limit DECIMAL(10,2) DEFAULT 10.00,
    hourly_cost_limit DECIMAL(10,2) DEFAULT 2.00,
    cost_per_interaction_max DECIMAL(10,6) DEFAULT 0.02,
    token_efficiency_min DECIMAL(3,2) DEFAULT 0.75,
    cache_hit_rate_min DECIMAL(3,2) DEFAULT 0.30,
    
    -- Alert settings
    cost_spike_threshold DECIMAL(3,2) DEFAULT 1.50,  -- 50% increase
    enable_alerts BOOLEAN DEFAULT TRUE,
    alert_emails TEXT[] DEFAULT '{}',
    
    -- Optimization settings
    auto_optimize BOOLEAN DEFAULT FALSE,
    optimization_confidence_min DECIMAL(3,2) DEFAULT 0.80,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default monitoring configuration
INSERT INTO cost_monitoring_config (config_name) 
VALUES ('default')
ON CONFLICT (config_name) DO NOTHING; 