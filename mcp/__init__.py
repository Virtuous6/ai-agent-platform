"""
MCP (Model Context Protocol) Integration Package

Enables dynamic connection to external tools, databases, and services
through a secure, user-managed interface integrated with the AI Agent Platform.

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

from .connection_manager import MCPConnectionManager
from .credential_store import CredentialManager
from .tool_registry import MCPToolRegistry
from .security_sandbox import MCPSecuritySandbox

# Run cards for quick setup
from .run_cards.supabase_card import SupabaseRunCard
from .run_cards.github_card import GitHubRunCard
from .run_cards.slack_card import SlackRunCard
from .run_cards.custom_card import CustomRunCard

# Slack interface for user management
from .slack_interface.mcp_commands import MCPSlackCommands
from .slack_interface.connection_modal import MCPConnectionModal
from .slack_interface.tool_browser import MCPToolBrowser

__version__ = "1.0.0"

__all__ = [
    # Core components
    'MCPConnectionManager',
    'CredentialManager', 
    'MCPToolRegistry',
    'MCPSecuritySandbox',
    
    # Run cards
    'SupabaseRunCard',
    'GitHubRunCard', 
    'SlackRunCard',
    'CustomRunCard',
    
    # Slack interface
    'MCPSlackCommands',
    'MCPConnectionModal',
    'MCPToolBrowser'
]

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