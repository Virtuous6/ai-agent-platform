"""
Improvement agents package for the AI Agent Platform.

This package provides intelligent workflow analysis and continuous improvement capabilities.
"""

from .workflow_analyst import (
    WorkflowAnalyst,
    WorkflowPattern,
    OptimizationOpportunity,
    AgentGap,
    AnalysisType,
    PatternStrength,
    OptimizationType
)

__all__ = [
    "WorkflowAnalyst",
    "WorkflowPattern", 
    "OptimizationOpportunity",
    "AgentGap",
    "AnalysisType",
    "PatternStrength",
    "OptimizationType"
] 