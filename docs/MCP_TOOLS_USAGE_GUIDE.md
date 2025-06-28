# MCP Tools Usage Guide

## Overview

The AI Agent Platform now has proper MCP (Model Context Protocol) tool integration that allows agents to call tools from multiple sources:

1. **Standard Library Tools** - Built-in tools like calculate, web_search, file operations
2. **Supabase Tools** - Tools stored in the `tools` table in Supabase
3. **MCP Tools** - Tools from MCP connections (coming soon)

## How It Works

### Tool Registry

All tools are managed through the `UniversalMCPToolRegistry` which:
- Loads tools from multiple sources automatically
- Provides a unified interface for tool execution
- Tracks tool usage in Supabase
- Handles errors gracefully

### Agent Integration

Agents automatically have access to tools through the registry:

```python
# Agent automatically initializes tool registry
agent = UniversalAgent(
    specialty="General Assistant",
    system_prompt="You are a helpful assistant that can use tools."
)

# Agent can now use tools when processing messages
result = await agent.process_message("Calculate 15% tip on $50")
```

### Available Tools

Check available tools in Supabase:
```sql
SELECT tool_name, tool_type, description 
FROM tools 
WHERE is_active = true;
```

Current tools include:
- **calculate** - Mathematical calculations
- **web_search** - Search the web
- **read_file** - Read files
- **write_file** - Write files
- **visit_webpage** - Extract webpage content

## Adding New Tools

### 1. Add to Supabase

Insert a new tool into the `tools` table:

```sql
INSERT INTO tools (tool_name, tool_type, description, tool_schema, is_active)
VALUES (
    'my_tool',
    'custom',
    'Description of what the tool does',
    '{"type": "object", "properties": {"param1": {"type": "string"}}}',
    true
);
```

### 2. Implement Tool Logic

For custom tools, you'll need to implement the execution logic. Currently, this requires adding to the standard library or creating an MCP server.

## Tool Usage Tracking

All tool usage is automatically tracked in `tool_usage_analytics`:

```sql
SELECT tool_name, COUNT(*) as usage_count, 
       AVG(execution_time_ms) as avg_time_ms,
       SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count
FROM tool_usage_analytics
GROUP BY tool_name
ORDER BY usage_count DESC;
```

## Testing Tools

Use the test script to verify tools are working:

```bash
python test_mcp_tools_integration.py
```

This will:
1. Test the tool registry
2. Test tool execution
3. Test agent integration
4. Check Supabase connections

## Troubleshooting

### Tools Not Found
- Check if tools are loaded: Look for "Tool registry initialized with X tools" in logs
- Verify Supabase connection: Check SUPABASE_URL and SUPABASE_KEY env vars
- Check tool status: Ensure `is_active = true` in tools table

### Tool Execution Fails
- Check tool parameters match the schema
- Look for error details in logs
- Check `tool_usage_analytics` for error messages

### Agent Not Using Tools
- Ensure agent has tool registry initialized
- Check if message triggers tool detection
- Verify tool names match expected patterns

## Next Steps

1. **MCP Server Integration** - Connect actual MCP servers for more tools
2. **Dynamic Tool Creation** - Allow creating tools through the UI
3. **Tool Chaining** - Enable tools to call other tools
4. **Custom Tool Runtime** - Safe execution environment for custom tools 