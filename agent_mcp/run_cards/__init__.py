"""
MCP Run Cards

Pre-built connection templates for quick setup of popular services.
Run cards provide one-click setup experiences for common integrations.
"""

from .supabase_card import SupabaseRunCard
from .github_card import GitHubRunCard  
from .slack_card import SlackRunCard
from .custom_card import CustomRunCard

__all__ = [
    'SupabaseRunCard',
    'GitHubRunCard',
    'SlackRunCard', 
    'CustomRunCard'
]

# MCP Run Cards package 