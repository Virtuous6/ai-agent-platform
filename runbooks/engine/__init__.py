"""
Runbook Engine Package

Provides dynamic execution capabilities for YAML-defined runbooks.
Converts static runbook definitions into executable workflows.
"""

from .runbook_executor import RunbookExecutor

__all__ = [
    'RunbookExecutor'
] 