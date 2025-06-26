"""
AI Agent Orchestrator Package

Enhanced with LangGraph workflow execution capabilities for dynamic runbook processing
and intelligent agent coordination.
"""

from .agent_orchestrator import AgentOrchestrator, AgentType

# LangGraph integration (optional import for graceful fallback)
try:
    from .langgraph import LangGraphWorkflowEngine, RunbookToGraphConverter
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

__all__ = ['AgentOrchestrator', 'AgentType']

if LANGGRAPH_AVAILABLE:
    __all__.extend(['LangGraphWorkflowEngine', 'RunbookToGraphConverter']) 