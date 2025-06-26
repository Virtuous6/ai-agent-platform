# AI Agent Platform - Project Context

## Overview
This is an AI Agent Platform designed to be self-documenting, runbook-driven, and LLM-friendly. The platform orchestrates specialized AI agents through a Slack interface, using structured workflows defined in YAML runbooks.

## Core Architecture

### Components
- **Slack Bot**: Primary user interface for all interactions
- **Agent Orchestrator**: Routes requests to specialized agents (General, Technical, Research)
- **MCP Tools**: Model Context Protocol implementations for agent actions
- **Supabase Integration**: Logging, analytics, and persistence
- **Runbook System**: YAML-based workflows for business logic
- **State Management**: Conversation context and user preferences

### Design Principles
1. **Documentation First**: Every directory has README.llm.md for LLM understanding
2. **Runbook-Driven**: Business logic lives in YAML, code handles execution
3. **Self-Documenting**: Code and structure readable by both humans and LLMs
4. **Async Everything**: Built for concurrent operations
5. **State-Aware**: Maintains context across interactions

## Directory Structure
```
ai-agent-platform/
├── .llm/                          # Project documentation for LLMs
├── agents/                        # Specialized AI agents
│   ├── general/                   # General purpose agent
│   ├── technical/                 # Technical support agent
│   └── research/                  # Research and analysis agent
├── orchestrator/                  # Central routing and coordination
├── slack_interface/               # Slack bot implementation
├── tools/                         # MCP tools for agent actions
├── runbooks/                      # YAML workflow definitions
│   ├── active/                    # Production runbooks
│   ├── templates/                 # Runbook templates
│   └── validation/                # Runbook validation tools
├── state/                         # State management components
├── database/                      # Database integration (Supabase)
├── tests/                         # Test files and utilities
└── config/                        # Configuration files

```

## Getting Started
1. Each component is self-contained with its own README.llm.md
2. Start with slack_interface/ for user interactions
3. Orchestrator routes to agents based on request type
4. Agents use tools to take actions
5. All interactions logged to Supabase

## Development Workflow
1. Read relevant README.llm.md files first
2. Check for applicable runbooks
3. Implement code following async patterns
4. Update documentation
5. Test with integration approach

## Key Files to Understand
- `slack_interface/slack_bot.py` - Main entry point
- `orchestrator/agent_router.py` - Request routing logic
- `runbooks/active/` - Current business workflows
- `state/context_manager.py` - Conversation state handling 