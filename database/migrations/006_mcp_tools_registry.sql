-- Migration: MCP Tools Registry
-- Description: Store custom tools and track tool usage

-- Create table for custom MCP tools
CREATE TABLE IF NOT EXISTS mcp_tools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    parameters JSONB DEFAULT '{}',
    implementation TEXT, -- For custom tools, the actual code
    source TEXT NOT NULL DEFAULT 'custom', -- 'custom', 'mcp', 'standard'
    server TEXT, -- For MCP tools, which server provides it
    is_active BOOLEAN DEFAULT true,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Create table for tool usage tracking
CREATE TABLE IF NOT EXISTS tool_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tool_name TEXT NOT NULL,
    agent_id TEXT,
    user_id TEXT,
    parameters JSONB DEFAULT '{}',
    result JSONB DEFAULT '{}',
    success BOOLEAN DEFAULT false,
    error_message TEXT,
    execution_time_ms INTEGER,
    timestamp TIMESTAMPTZ DEFAULT now()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_mcp_tools_name ON mcp_tools(name);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_active ON mcp_tools(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_tool_usage_tool_name ON tool_usage(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_usage_timestamp ON tool_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_tool_usage_user ON tool_usage(user_id);

-- Add some example custom tools
INSERT INTO mcp_tools (name, description, parameters, implementation, source)
VALUES 
    ('format_json', 'Format JSON string for readability', 
     '{"json_string": {"type": "string", "required": true}, "indent": {"type": "integer", "default": 2}}',
     'import json
try:
    data = json.loads(kwargs["json_string"])
    formatted = json.dumps(data, indent=kwargs.get("indent", 2))
    result = {"success": True, "formatted": formatted}
except Exception as e:
    result = {"success": False, "error": str(e)}',
     'custom'),
     
    ('word_count', 'Count words in text',
     '{"text": {"type": "string", "required": true}}',
     'text = kwargs.get("text", "")
words = len(text.split())
chars = len(text)
lines = len(text.splitlines())
result = {"success": True, "words": words, "characters": chars, "lines": lines}',
     'custom')
ON CONFLICT (name) DO NOTHING;

-- Add RLS policies
ALTER TABLE mcp_tools ENABLE ROW LEVEL SECURITY;
ALTER TABLE tool_usage ENABLE ROW LEVEL SECURITY;

-- Tools are readable by everyone but only admins can modify
CREATE POLICY "Tools are viewable by all" ON mcp_tools
    FOR SELECT USING (true);

-- Tool usage is viewable by the user who used it
CREATE POLICY "Users can view their own tool usage" ON tool_usage
    FOR SELECT USING (auth.uid()::text = user_id OR user_id IS NULL); 