-- Migration 005: Dynamic Agent Prompts
-- Enables dynamic, learning-driven agent prompts stored in Supabase

-- Agent Prompt Templates Table
CREATE TABLE IF NOT EXISTS agent_prompt_templates (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    agent_type VARCHAR(100) NOT NULL,
    specialty VARCHAR(200),
    complexity_level VARCHAR(50), -- simple, medium, complex, adaptive
    prompt_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    
    -- Core prompt components
    system_prompt TEXT NOT NULL,
    tool_decision_guidance TEXT,
    communication_style TEXT,
    quality_standards TEXT,
    
    -- Tool-specific sections
    mcp_tool_guidance TEXT,
    tool_gap_detection_prompt TEXT,
    tool_selection_criteria TEXT,
    
    -- Learning and adaptation
    performance_score DECIMAL(3,2) DEFAULT 0.50,
    usage_count INTEGER DEFAULT 0,
    success_rate DECIMAL(3,2) DEFAULT 0.00,
    avg_user_satisfaction DECIMAL(3,2) DEFAULT 0.00,
    
    -- A/B testing
    is_active BOOLEAN DEFAULT true,
    test_group VARCHAR(50), -- 'control', 'variant_a', 'variant_b', etc.
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    tags TEXT[],
    
    -- Ensure we don't have duplicate active prompts for same conditions
    UNIQUE(agent_type, specialty, complexity_level, test_group) 
    WHERE is_active = true
);

-- Prompt Performance Tracking Table
CREATE TABLE IF NOT EXISTS prompt_performance_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    prompt_template_id UUID REFERENCES agent_prompt_templates(id),
    
    -- Context of usage
    user_message_preview TEXT,
    complexity_score DECIMAL(3,2),
    context_type VARCHAR(100),
    
    -- Performance metrics
    tool_decision_accuracy DECIMAL(3,2), -- Did it choose the right tool approach?
    response_quality_score DECIMAL(3,2),
    user_satisfaction_score DECIMAL(3,2),
    tool_gap_detected BOOLEAN DEFAULT false,
    tool_successfully_used BOOLEAN DEFAULT false,
    
    -- Technical metrics
    tokens_used INTEGER,
    processing_time_ms INTEGER,
    cost_estimate DECIMAL(8,4),
    
    -- Results
    tools_invoked TEXT[], -- Array of tool names used
    mcp_solutions_found INTEGER DEFAULT 0,
    custom_tools_created INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Prompt Optimization Insights Table
CREATE TABLE IF NOT EXISTS prompt_optimization_insights (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    agent_type VARCHAR(100) NOT NULL,
    specialty VARCHAR(200),
    
    -- Learning insights
    insight_type VARCHAR(100), -- 'tool_selection_improvement', 'communication_optimization', etc.
    insight_description TEXT,
    suggested_prompt_changes JSONB,
    confidence_score DECIMAL(3,2),
    
    -- Supporting data
    sample_interactions JSONB, -- Sample interactions that led to this insight
    performance_improvement_estimate DECIMAL(3,2),
    
    -- Implementation tracking
    status VARCHAR(50) DEFAULT 'pending', -- pending, implemented, tested, rejected
    implemented_at TIMESTAMP WITH TIME ZONE,
    test_results JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system_learning'
);

-- Indexes for performance
CREATE INDEX idx_agent_prompts_lookup ON agent_prompt_templates(agent_type, specialty, complexity_level, is_active);
CREATE INDEX idx_prompt_performance_template ON prompt_performance_logs(prompt_template_id);
CREATE INDEX idx_prompt_performance_time ON prompt_performance_logs(created_at);
CREATE INDEX idx_optimization_insights_agent ON prompt_optimization_insights(agent_type, specialty, status);

-- Functions for prompt selection
CREATE OR REPLACE FUNCTION get_optimal_prompt(
    p_agent_type VARCHAR(100),
    p_specialty VARCHAR(200) DEFAULT NULL,
    p_complexity_level VARCHAR(50) DEFAULT 'medium',
    p_test_group VARCHAR(50) DEFAULT 'control'
) RETURNS TABLE (
    template_id UUID,
    system_prompt TEXT,
    tool_decision_guidance TEXT,
    communication_style TEXT,
    tool_selection_criteria TEXT,
    performance_score DECIMAL(3,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        apt.id as template_id,
        apt.system_prompt,
        apt.tool_decision_guidance,
        apt.communication_style,
        apt.tool_selection_criteria,
        apt.performance_score
    FROM agent_prompt_templates apt
    WHERE apt.agent_type = p_agent_type
        AND apt.is_active = true
        AND (p_specialty IS NULL OR apt.specialty = p_specialty OR apt.specialty IS NULL)
        AND (apt.complexity_level = p_complexity_level OR apt.complexity_level = 'adaptive')
        AND (apt.test_group = p_test_group OR apt.test_group IS NULL)
    ORDER BY 
        apt.performance_score DESC,
        apt.usage_count DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to log prompt performance
CREATE OR REPLACE FUNCTION log_prompt_performance(
    p_template_id UUID,
    p_user_message_preview TEXT,
    p_complexity_score DECIMAL(3,2),
    p_tool_decision_accuracy DECIMAL(3,2),
    p_response_quality_score DECIMAL(3,2),
    p_tools_invoked TEXT[] DEFAULT ARRAY[]::TEXT[],
    p_mcp_solutions_found INTEGER DEFAULT 0
) RETURNS UUID AS $$
DECLARE
    log_id UUID;
BEGIN
    INSERT INTO prompt_performance_logs (
        prompt_template_id,
        user_message_preview,
        complexity_score,
        tool_decision_accuracy,
        response_quality_score,
        tools_invoked,
        mcp_solutions_found
    ) VALUES (
        p_template_id,
        p_user_message_preview,
        p_complexity_score,
        p_tool_decision_accuracy,
        p_response_quality_score,
        p_tools_invoked,
        p_mcp_solutions_found
    ) RETURNING id INTO log_id;
    
    -- Update template performance metrics
    UPDATE agent_prompt_templates 
    SET 
        usage_count = usage_count + 1,
        performance_score = (
            SELECT AVG(tool_decision_accuracy * 0.4 + response_quality_score * 0.6)
            FROM prompt_performance_logs 
            WHERE prompt_template_id = p_template_id
            AND created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_template_id;
    
    RETURN log_id;
END;
$$ LANGUAGE plpgsql;

-- Insert some initial prompt templates
INSERT INTO agent_prompt_templates (
    agent_type, 
    specialty,
    complexity_level,
    system_prompt,
    tool_decision_guidance,
    communication_style,
    tool_selection_criteria,
    test_group
) VALUES 
-- Universal Agent - Tool-Optimized Prompts
(
    'universal',
    'general',
    'medium',
    'You are a versatile AI specialist integrated into a self-improving platform with dynamic tool access.',
    'TOOL DECISION FRAMEWORK:
1. **Assess Need**: Does this require real-time data, external services, or specialized processing?
2. **MCP-First**: Always check for existing MCP tools before creating custom solutions
3. **Gap Detection**: If confidence < 0.7, identify what tools are missing
4. **User Guidance**: Clearly explain what tools you''re using and why

For "weather in Spain" → USE web search MCP (Serper)
For "Airtable records" → DETECT gap, suggest Airtable MCP setup
For "analyze data" → CHECK for data analysis MCPs first',
    'Professional yet approachable. Explain your tool usage transparently.',
    'Priority: Existing MCPs > MCP Library > Custom Tool Creation > Inform user of limitations',
    'control'
),
-- Research Agent - Enhanced Tool Logic  
(
    'universal',
    'research',
    'medium',
    'You are a research specialist with access to real-time information gathering tools.',
    'RESEARCH TOOL STRATEGY:
1. **Information Gathering**: Use Serper web search for current information
2. **Data Analysis**: Look for analytics MCPs for complex data work
3. **Source Verification**: Always indicate tool sources and freshness
4. **Deep Research**: For comprehensive research, combine multiple tool approaches',
    'Analytical and thorough. Always cite your information sources and methods.',
    'Real-time data: Serper MCP, Historical data: Database MCPs, Analysis: Analytics tools',
    'control'
),
-- Technical Agent - Development-Focused
(
    'universal', 
    'technical',
    'medium',
    'You are a technical expert with access to development and infrastructure tools.',
    'TECHNICAL TOOL APPROACH:
1. **Code Solutions**: Check for GitHub, IDE, or development MCPs
2. **Infrastructure**: Look for cloud provider and deployment MCPs  
3. **Debugging**: Use error analysis and logging tools
4. **API Integration**: Prioritize API connection MCPs for external services',
    'Precise and technically accurate. Show exact tool usage and configuration.',
    'Development: GitHub MCP, Infrastructure: Cloud MCPs, APIs: Service-specific MCPs',
    'control'
);

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE ON agent_prompt_templates TO authenticated;
GRANT SELECT, INSERT ON prompt_performance_logs TO authenticated;
GRANT SELECT, INSERT, UPDATE ON prompt_optimization_insights TO authenticated;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO authenticated; 