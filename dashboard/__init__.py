"""
AI Agent Platform Dashboard

A retro terminal-style dashboard for monitoring the self-improving AI agent platform.
Provides real-time insights for both human operators and AI agents.

Features:
- Real-time Supabase log streaming
- Terminal-style retro interface
- Modular component architecture
- AI-readable data endpoints
- Live system health monitoring
"""

from .terminal_dashboard import TerminalDashboard
from .components.system_health import SystemHealthMonitor
from .components.agent_monitor import AgentMonitor
from .components.cost_analytics import CostAnalytics
from .components.event_stream import EventStreamViewer
from .components.logs_viewer import LogsViewer

__version__ = "1.0.0"
__all__ = [
    "TerminalDashboard",
    "SystemHealthMonitor", 
    "AgentMonitor",
    "CostAnalytics",
    "EventStreamViewer",
    "LogsViewer"
] 