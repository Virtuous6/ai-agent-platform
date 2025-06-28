"""
MCP (Model Context Protocol) integration package.

Provides tools, security, and universal access patterns for MCP tools
in the self-improving AI agent platform.

Core Components:
- Connection Manager: Handles MCP service connections
- Credential Store: Secure credential management
- Tool Registry: Dynamic tool discovery and execution
- Security Sandbox: Isolated tool execution environment
- Run Cards: Pre-built connection templates
- Slack Interface: User-friendly MCP management via Slack

Integration Points:
- Agent System: Tools become available to all agents
- Goal Orchestrator: MCP tools can be used in goal execution
- Cost Tracking: Tool usage costs tracked alongside LLM costs
- Approval System: Expensive tools require human approval
- Memory System: Tool results stored in vector memory
"""

# Import core modules that we've created
from .tool_registry import MCPToolRegistry, ToolDescriptor
from .security_sandbox import MCPSecuritySandbox

# Import universal tools
try:
    from .universal_mcp_tools import (
        UniversalMCPTools, 
        universal_mcp_tools,
        get_universal_tools,
        execute_universal_tool,
        list_universal_tools
    )
    HAS_UNIVERSAL_TOOLS = True
except ImportError:
    HAS_UNIVERSAL_TOOLS = False

__all__ = [
    # Core MCP components
    'MCPToolRegistry',
    'ToolDescriptor', 
    'MCPSecuritySandbox',
    'HAS_UNIVERSAL_TOOLS'
]

# Add universal tools to __all__ if available
if HAS_UNIVERSAL_TOOLS:
    __all__.extend([
        'UniversalMCPTools',
        'universal_mcp_tools', 
        'get_universal_tools',
        'execute_universal_tool',
        'list_universal_tools'
    ])

# Configuration constants
MCP_CONFIG = {
    'max_connections_per_user': 10,
    'tool_timeout_seconds': 30,
    'sandbox_enabled': True,
    'rate_limit_per_minute': 100,
    'max_retry_attempts': 3,
    'credential_encryption_enabled': True,
    'audit_logging_enabled': True
}

# Supported MCP types
SUPPORTED_MCP_TYPES = [
    'supabase', 'github', 'slack', 'postgres', 'mongodb', 'redis',
    'notion', 'airtable', 'google_sheets', 'jira', 'linear',
    'custom_api', 'graphql', 'rest_api'
]

# Security constants
SECURITY_CONFIG = {
    'max_execution_time_seconds': 30,
    'memory_limit_mb': 128,
    'network_restrictions': ['localhost', 'private_ranges'],
    'allowed_protocols': ['https', 'wss'],
    'blocked_domains': ['malicious-site.com'],
    'parameter_validation_enabled': True
} 