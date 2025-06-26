"""
Resources package for the Virtuous6 AI Platform.

This package manages shared resources across all agents:
- LLM connection pools
- Tool instance sharing  
- Database connection pooling
- Vector memory allocation
- Resource usage tracking and fair scheduling
"""

from .pool_manager import ResourcePoolManager, ResourceType, ResourceRequest

__all__ = [
    'ResourcePoolManager',
    'ResourceType', 
    'ResourceRequest'
] 