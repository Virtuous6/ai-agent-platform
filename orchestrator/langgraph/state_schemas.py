"""
LangGraph State Schemas

Defines state structures for runbook execution, agent coordination,
and tool integration within LangGraph workflows.
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
import operator

class RunbookState(TypedDict):
    """State schema for runbook execution workflows."""
    
    # User context
    user_id: str
    user_message: str
    conversation_id: Optional[str]
    channel_id: Optional[str]
    
    # Execution context
    current_step: str
    execution_history: Annotated[List[Dict[str, Any]], operator.add]
    error_count: int
    
    # Agent responses
    agent_responses: Dict[str, Any]
    selected_agent: Optional[str]
    routing_confidence: float
    
    # Tool results
    tool_results: Dict[str, Any]
    web_search_results: Optional[List[Dict[str, Any]]]
    
    # Memory and context
    conversation_history: List[Dict[str, Any]]
    retrieved_memories: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    
    # Output and metadata
    final_response: Optional[str]
    confidence_score: float
    processing_time_ms: float
    tokens_used: int
    estimated_cost: float
    
    # Workflow control
    needs_escalation: bool
    escalation_reason: Optional[str]
    retry_count: int
    max_retries: int

class AgentState(TypedDict):
    """State schema for individual agent processing."""
    
    agent_type: str
    agent_name: str
    prompt_template: str
    temperature: float
    max_tokens: int
    
    input_message: str
    context: Dict[str, Any]
    
    response: Optional[str]
    confidence: float
    reasoning: Optional[str]
    
    token_usage: Dict[str, int]
    processing_time: float
    cost: float
    
    escalation_suggestion: Optional[Dict[str, Any]]
    tool_suggestions: List[str]

class ToolState(TypedDict):
    """State schema for tool execution."""
    
    tool_name: str
    tool_parameters: Dict[str, Any]
    execution_context: Dict[str, Any]
    
    result: Optional[Any]
    success: bool
    error_message: Optional[str]
    
    execution_time: float
    retry_count: int
    
    metadata: Dict[str, Any] 