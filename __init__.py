"""
AI Agent Platform

An LLM-powered AI agent platform that uses ChatGPT agents with specialized 
domain expertise, enhanced by runbook frameworks for consistent intelligence.

The platform leverages OpenAI's ChatGPT for natural language understanding 
while maintaining cost efficiency and performance optimization.
"""

__version__ = "1.0.0"
__author__ = "AI Agent Platform Team"
__description__ = "LLM-powered AI agent platform with Slack integration"

# Main components available for import
from orchestrator import AgentOrchestrator, AgentType
from agents import GeneralAgent, ResearchAgent, TechnicalAgent
from database import SupabaseLogger

__all__ = [
    'AgentOrchestrator', 
    'AgentType',
    'GeneralAgent', 
    'ResearchAgent', 
    'TechnicalAgent',
    'SupabaseLogger'
] 