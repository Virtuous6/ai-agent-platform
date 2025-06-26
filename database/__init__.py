"""
Database Integration Package

This package handles database operations and logging for the AI Agent Platform.
Contains Supabase integration for conversation tracking, analytics, and metrics.
"""

# Supabase module
from .supabase_logger import SupabaseLogger

__all__ = ['SupabaseLogger'] 