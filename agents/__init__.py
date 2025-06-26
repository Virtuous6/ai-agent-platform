"""
AI Agents Package

This package contains specialized LLM-powered agents for different domains:
- General Agent: Conversational interactions with higher temperature
- Technical Agent: Precise technical responses with lower temperature  
- Research Agent: Analytical research responses with balanced temperature
"""

from .general.general_agent import GeneralAgent
from .research.research_agent import ResearchAgent
from .technical.technical_agent import TechnicalAgent

__all__ = ['GeneralAgent', 'ResearchAgent', 'TechnicalAgent'] 