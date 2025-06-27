"""
Runbook to LangGraph Converter

Converts YAML-defined runbooks into executable LangGraph state machines.
Preserves the intelligence and logic of runbooks while making them dynamic.
"""

import asyncio
import logging
import yaml
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

from .state_schemas import RunbookState

logger = logging.getLogger(__name__)

class RunbookToGraphConverter:
    """
    Converts YAML runbooks into executable LangGraph workflows.
    
    Maintains the original runbook logic while adding dynamic execution
    capabilities and real-time decision making.
    """
    
    def __init__(self):
        """Initialize the converter."""
        self.node_builders = {
            'analyze_message': self._build_analyze_node,
            'invoke_agent': self._build_agent_node,
            'invoke_tool': self._build_tool_node,
            'check_response_quality': self._build_quality_check_node,
            'format_output': self._build_format_node,
            'validate_message': self._build_validation_node,
            'custom_logic': self._build_custom_node
        }
        logger.info("Runbook converter initialized")
    
    async def convert_runbook_to_graph(self, runbook_path: str, 
                                     agents: Dict[str, Any], 
                                     tools: Dict[str, Any]) -> Optional[Any]:
        """
        Convert a YAML runbook to an executable LangGraph workflow.
        
        Args:
            runbook_path: Path to YAML runbook file
            agents: Available agent instances
            tools: Available tool instances
            
        Returns:
            Compiled LangGraph workflow
        """
        if not LANGGRAPH_AVAILABLE:
            logger.error("LangGraph not available - cannot convert runbooks")
            return None
        
        try:
            # Load and parse runbook
            runbook_data = await self._load_runbook(runbook_path)
            
            # Create state graph
            workflow = StateGraph(RunbookState)
            
            # Build nodes from runbook steps
            await self._build_nodes(workflow, runbook_data, agents, tools)
            
            # Build edges and conditions
            await self._build_edges(workflow, runbook_data)
            
            # Set entry point
            workflow.set_entry_point("start")
            
            # Compile and return
            compiled_workflow = workflow.compile()
            logger.info(f"Successfully converted runbook: {runbook_path}")
            return compiled_workflow
            
        except Exception as e:
            logger.error(f"Failed to convert runbook {runbook_path}: {str(e)}")
            raise ValueError(f"Runbook conversion failed: {str(e)}")
    
    async def _load_runbook(self, runbook_path: str) -> Dict[str, Any]:
        """Load and validate runbook YAML file."""
        try:
            with open(runbook_path, 'r') as file:
                runbook_data = yaml.safe_load(file)
            
            # Basic validation
            if 'steps' not in runbook_data:
                raise ValueError("Runbook missing 'steps' section")
            
            return runbook_data
            
        except FileNotFoundError:
            raise ValueError(f"Runbook file not found: {runbook_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in runbook: {str(e)}")
    
    async def _build_nodes(self, workflow: Any, runbook_data: Dict[str, Any], 
                         agents: Dict[str, Any], tools: Dict[str, Any]):
        """Build LangGraph nodes from runbook steps."""
        
        # Add start node
        workflow.add_node("start", self._create_start_node())
        
        # Add step nodes
        for step in runbook_data['steps']:
            step_id = step['id']
            action = step['action']
            
            if action in self.node_builders:
                node_function = self.node_builders[action](step, agents, tools)
                workflow.add_node(step_id, node_function)
            else:
                # Generic node for unrecognized actions
                workflow.add_node(step_id, self._create_generic_node(step))
        
        # Add end node
        workflow.add_node("end", self._create_end_node())
    
    async def _build_edges(self, workflow: Any, runbook_data: Dict[str, Any]):
        """Build edges and conditional routing between nodes."""
        
        # Connect start to first step
        if runbook_data['steps']:
            first_step = runbook_data['steps'][0]['id']
            workflow.add_edge("start", first_step)
        
        # Connect steps sequentially with conditional logic
        steps = runbook_data['steps']
        for i, step in enumerate(steps):
            step_id = step['id']
            
            # Check for conditions in the step
            if 'conditions' in step:
                # Add conditional edges
                condition_func = self._create_condition_function(step['conditions'])
                next_steps = self._determine_next_steps(step, steps, i)
                workflow.add_conditional_edges(step_id, condition_func, next_steps)
            else:
                # Simple sequential edge
                if i < len(steps) - 1:
                    next_step = steps[i + 1]['id']
                    workflow.add_edge(step_id, next_step)
                else:
                    # Last step connects to end
                    workflow.add_edge(step_id, "end")
        
        # Connect end to END
        workflow.add_edge("end", END)
    
    def _create_start_node(self) -> Callable:
        """Create the workflow start node."""
        def start_node(state: RunbookState) -> RunbookState:
            return {
                **state,
                'current_step': 'start',
                'execution_history': [{'step': 'start', 'timestamp': str(asyncio.get_event_loop().time())}],
                'error_count': 0,
                'retry_count': 0
            }
        return start_node
    
    def _create_end_node(self) -> Callable:
        """Create the workflow end node."""
        def end_node(state: RunbookState) -> RunbookState:
            return {
                **state,
                'current_step': 'completed',
                'execution_history': state['execution_history'] + [
                    {'step': 'end', 'timestamp': str(asyncio.get_event_loop().time())}
                ]
            }
        return end_node
    
    def _create_generic_node(self, step: Dict[str, Any]) -> Callable:
        """Create a generic node for unrecognized step types."""
        def generic_node(state: RunbookState) -> RunbookState:
            logger.warning(f"Generic node executed for step: {step['id']}")
            return {
                **state,
                'current_step': step['id'],
                'execution_history': state['execution_history'] + [
                    {'step': step['id'], 'action': 'generic', 'timestamp': str(asyncio.get_event_loop().time())}
                ]
            }
        return generic_node
    
    def _build_analyze_node(self, step: Dict[str, Any], agents: Dict[str, Any], 
                          tools: Dict[str, Any]) -> Callable:
        """Build analysis node that classifies messages."""
        async def analyze_node(state: RunbookState) -> RunbookState:
            try:
                # Extract classification patterns from step
                patterns = step.get('parameters', {}).get('classification_patterns', {})
                message = state['user_message'].lower()
                
                # Simple pattern matching (can be enhanced with LLM)
                classification_result = {}
                for category, keywords in patterns.items():
                    if any(keyword in message for keyword in keywords):
                        classification_result['question_type'] = category
                        break
                else:
                    classification_result['question_type'] = 'general'
                
                return {
                    **state,
                    'current_step': step['id'],
                    'tool_results': {**state.get('tool_results', {}), 'classification': classification_result},
                    'execution_history': state['execution_history'] + [
                        {'step': step['id'], 'result': classification_result, 'timestamp': str(asyncio.get_event_loop().time())}
                    ]
                }
            except Exception as e:
                logger.error(f"Analysis node error: {e}")
                return {**state, 'error_count': state['error_count'] + 1}
        
        return analyze_node
    
    def _build_agent_node(self, step: Dict[str, Any], agents: Dict[str, Any], 
                        tools: Dict[str, Any]) -> Callable:
        """Build agent invocation node."""
        async def agent_node(state: RunbookState) -> RunbookState:
            try:
                agent_name = step.get('parameters', {}).get('agent', 'general')
                agent = agents.get(agent_name)
                
                if not agent:
                    raise ValueError(f"Agent '{agent_name}' not available")
                
                # Prepare context for agent
                context = {
                    'user_id': state['user_id'],
                    'conversation_history': state['conversation_history'],
                    'user_preferences': state['user_preferences']
                }
                
                # Call agent
                if hasattr(agent, 'process_message'):
                    result = await agent.process_message(state['user_message'], context)
                else:
                    result = {'response': 'Agent method not available', 'confidence': 0.5}
                
                return {
                    **state,
                    'current_step': step['id'],
                    'selected_agent': agent_name,
                    'agent_responses': {**state.get('agent_responses', {}), agent_name: result},
                    'execution_history': state['execution_history'] + [
                        {'step': step['id'], 'agent': agent_name, 'timestamp': str(asyncio.get_event_loop().time())}
                    ]
                }
            except Exception as e:
                logger.error(f"Agent node error: {e}")
                return {**state, 'error_count': state['error_count'] + 1}
        
        return agent_node
    
    def _build_tool_node(self, step: Dict[str, Any], agents: Dict[str, Any], 
                       tools: Dict[str, Any]) -> Callable:
        """Build tool invocation node."""
        async def tool_node(state: RunbookState) -> RunbookState:
            try:
                tool_name = step.get('parameters', {}).get('tool', 'web_search')
                tool = tools.get(tool_name)
                
                if not tool:
                    logger.warning(f"Tool '{tool_name}' not available")
                    return {**state, 'error_count': state['error_count'] + 1}
                
                # Build search query (simplified)
                query = state['user_message']
                
                # Execute tool (placeholder - implement actual tool calls)
                result = {'results': [], 'query': query, 'tool': tool_name}
                
                return {
                    **state,
                    'current_step': step['id'],
                    'tool_results': {**state.get('tool_results', {}), tool_name: result},
                    'execution_history': state['execution_history'] + [
                        {'step': step['id'], 'tool': tool_name, 'timestamp': str(asyncio.get_event_loop().time())}
                    ]
                }
            except Exception as e:
                logger.error(f"Tool node error: {e}")
                return {**state, 'error_count': state['error_count'] + 1}
        
        return tool_node
    
    def _build_quality_check_node(self, step: Dict[str, Any], agents: Dict[str, Any], 
                                tools: Dict[str, Any]) -> Callable:
        """Build response quality assessment node."""
        def quality_check_node(state: RunbookState) -> RunbookState:
            try:
                # Get last agent response
                agent_responses = state.get('agent_responses', {})
                if not agent_responses:
                    return {**state, 'needs_escalation': True, 'escalation_reason': 'No agent response available'}
                
                # Simple quality check (can be enhanced)
                last_response = list(agent_responses.values())[-1]
                response_text = last_response.get('response', '') if isinstance(last_response, dict) else str(last_response)
                
                # Check for patterns indicating need for web search
                needs_search_patterns = step.get('parameters', {}).get('check_for_patterns', [])
                needs_web_search = any(pattern.lower() in response_text.lower() for pattern in needs_search_patterns)
                
                return {
                    **state,
                    'current_step': step['id'],
                    'needs_escalation': needs_web_search,
                    'tool_results': {**state.get('tool_results', {}), 'quality_check': {
                        'needs_web_search': needs_web_search,
                        'response_length': len(response_text)
                    }},
                    'execution_history': state['execution_history'] + [
                        {'step': step['id'], 'needs_web_search': needs_web_search, 'timestamp': str(asyncio.get_event_loop().time())}
                    ]
                }
            except Exception as e:
                logger.error(f"Quality check node error: {e}")
                return {**state, 'error_count': state['error_count'] + 1}
        
        return quality_check_node
    
    def _build_format_node(self, step: Dict[str, Any], agents: Dict[str, Any], 
                         tools: Dict[str, Any]) -> Callable:
        """Build response formatting node."""
        def format_node(state: RunbookState) -> RunbookState:
            try:
                # Gather all responses
                agent_responses = state.get('agent_responses', {})
                tool_results = state.get('tool_results', {})
                
                # Build final response (simplified)
                final_response = ""
                
                # Add agent response
                if agent_responses:
                    last_response = list(agent_responses.values())[-1]
                    if isinstance(last_response, dict):
                        final_response = last_response.get('response', 'No response available')
                    else:
                        final_response = str(last_response)
                
                # Add web search results if available
                if 'web_search' in tool_results:
                    final_response += "\n\n*Enhanced with current information*"
                
                return {
                    **state,
                    'current_step': step['id'],
                    'final_response': final_response,
                    'confidence_score': 0.8,  # Simplified confidence
                    'execution_history': state['execution_history'] + [
                        {'step': step['id'], 'formatted': True, 'timestamp': str(asyncio.get_event_loop().time())}
                    ]
                }
            except Exception as e:
                logger.error(f"Format node error: {e}")
                return {**state, 'error_count': state['error_count'] + 1}
        
        return format_node
    
    def _build_validation_node(self, step: Dict[str, Any], agents: Dict[str, Any], 
                             tools: Dict[str, Any]) -> Callable:
        """Build input validation node."""
        def validation_node(state: RunbookState) -> RunbookState:
            # Simple validation (can be enhanced)
            is_valid = len(state['user_message'].strip()) > 0
            
            return {
                **state,
                'current_step': step['id'],
                'tool_results': {**state.get('tool_results', {}), 'validation': {'is_valid': is_valid}},
                'execution_history': state['execution_history'] + [
                    {'step': step['id'], 'valid': is_valid, 'timestamp': str(asyncio.get_event_loop().time())}
                ]
            }
        
        return validation_node
    
    def _build_custom_node(self, step: Dict[str, Any], agents: Dict[str, Any], 
                         tools: Dict[str, Any]) -> Callable:
        """Build custom logic node."""
        def custom_node(state: RunbookState) -> RunbookState:
            # Placeholder for custom logic
            return {
                **state,
                'current_step': step['id'],
                'execution_history': state['execution_history'] + [
                    {'step': step['id'], 'action': 'custom', 'timestamp': str(asyncio.get_event_loop().time())}
                ]
            }
        
        return custom_node
    
    def _create_condition_function(self, conditions: Dict[str, Any]) -> Callable:
        """Create conditional routing function."""
        def condition_router(state: RunbookState) -> str:
            # Check for web search need
            if 'needs_web_search' in conditions:
                quality_check = state.get('tool_results', {}).get('quality_check', {})
                if quality_check.get('needs_web_search', False):
                    return 'needs_web_search'
            
            # Default routing
            return 'continue'
        
        return condition_router
    
    def _determine_next_steps(self, step: Dict[str, Any], all_steps: List[Dict[str, Any]], 
                            current_index: int) -> Dict[str, str]:
        """Determine next steps based on conditions."""
        conditions = step.get('conditions', {})
        
        # Find web search step
        web_search_step = None
        for s in all_steps:
            if s.get('action') == 'invoke_tool' and s.get('parameters', {}).get('tool') == 'web_search':
                web_search_step = s['id']
                break
        
        next_steps = {
            'continue': all_steps[current_index + 1]['id'] if current_index + 1 < len(all_steps) else 'end',
            'needs_web_search': web_search_step or 'end'
        }
        
        return next_steps 