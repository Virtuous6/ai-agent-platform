"""
LangGraph Workflow Engine

Core engine that executes runbooks as LangGraph state machines with full
LLM agent integration and error handling.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, TypedDict
from datetime import datetime
import json

try:
    from langgraph.graph import StateGraph, END
    from langchain_core.runnables import RunnableConfig
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Graceful fallback when LangGraph not available
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

from .state_schemas import RunbookState
from .runbook_converter import RunbookToGraphConverter

logger = logging.getLogger(__name__)

class LangGraphWorkflowEngine:
    """
    Executes runbooks as dynamic LangGraph workflows with LLM integration.
    
    Transforms static YAML runbooks into intelligent, adaptive workflows
    that can make real-time decisions using LLM agents.
    """
    
    def __init__(self, agents: Dict[str, Any], tools: Dict[str, Any], 
                 supabase_logger=None):
        """Initialize workflow engine with agents and tools."""
        self.agents = agents
        self.tools = tools
        self.supabase_logger = supabase_logger
        self.active_workflows: Dict[str, Any] = {}
        self._runbook_manager_initialized = False
        
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available - workflow engine in fallback mode")
            self.converter = None
        else:
            self.converter = RunbookToGraphConverter()
        
        logger.info("LangGraph Workflow Engine initialized")
    
    async def initialize_runbook_manager(self, supabase_url: str, supabase_key: str) -> bool:
        """Initialize the runbook manager for Supabase access."""
        try:
            from runbooks.manager import initialize_runbook_manager
            success = await initialize_runbook_manager(supabase_url, supabase_key)
            self._runbook_manager_initialized = success
            if success:
                logger.info("Runbook manager initialized for workflow engine")
            else:
                logger.error("Failed to initialize runbook manager")
            return success
        except Exception as e:
            logger.error(f"Error initializing runbook manager: {e}")
            return False
    
    async def load_runbook_workflow(self, runbook_path: str) -> Optional[Any]:
        """
        Load and compile a runbook into an executable LangGraph workflow.
        
        Args:
            runbook_path: Path to YAML runbook file
            
        Returns:
            Compiled LangGraph workflow or None if LangGraph unavailable
        """
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available - cannot load workflow")
            return None
        
        try:
            workflow = await self.converter.convert_runbook_to_graph(
                runbook_path, self.agents, self.tools
            )
            
            workflow_name = runbook_path.split('/')[-1].replace('.yaml', '')
            self.active_workflows[workflow_name] = workflow
            
            logger.info(f"Loaded runbook workflow: {workflow_name}")
            return workflow
            
        except Exception as e:
            logger.error(f"Failed to load runbook {runbook_path}: {str(e)}")
            raise ValueError(
                f"Runbook loading failed for '{runbook_path}'. "
                f"Error: {str(e)}. Check YAML structure and agent availability."
            )
    
    async def load_runbook_from_supabase(self, runbook_name: str) -> Optional[Any]:
        """
        Load and compile a runbook from Supabase into an executable LangGraph workflow.
        
        Args:
            runbook_name: Name of the runbook in Supabase
            
        Returns:
            Compiled LangGraph workflow or None if unavailable
        """
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available - cannot load workflow")
            return None
        
        try:
            # Import and get runbook manager
            from runbooks.manager import get_runbook_manager
            runbook_manager = await get_runbook_manager()
            
            # Get runbook from Supabase
            runbook = await runbook_manager.get_runbook(runbook_name)
            if not runbook:
                logger.error(f"Runbook '{runbook_name}' not found in Supabase")
                return None
            
            # Convert Supabase runbook to workflow
            workflow = await self.converter.convert_supabase_runbook_to_graph(
                runbook, self.agents, self.tools
            )
            
            self.active_workflows[runbook_name] = workflow
            
            logger.info(f"Loaded Supabase runbook workflow: {runbook_name}")
            return workflow
            
        except Exception as e:
            logger.error(f"Failed to load Supabase runbook {runbook_name}: {str(e)}")
            raise ValueError(
                f"Supabase runbook loading failed for '{runbook_name}'. "
                f"Error: {str(e)}. Check runbook definition and agent availability."
            )
    
    async def execute_workflow(self, workflow_name: str, initial_state: Dict[str, Any],
                             config: Optional[Any] = None) -> Dict[str, Any]:
        """
        Execute a loaded workflow with initial state.
        
        Args:
            workflow_name: Name of loaded workflow
            initial_state: Initial workflow state
            config: Optional execution configuration
            
        Returns:
            Final workflow state and results
        """
        if not LANGGRAPH_AVAILABLE:
            raise ValueError("LangGraph not available - cannot execute workflows")
        
        if workflow_name not in self.active_workflows:
            raise ValueError(
                f"Workflow '{workflow_name}' not loaded. "
                f"Available workflows: {list(self.active_workflows.keys())}"
            )
        
        workflow = self.active_workflows[workflow_name]
        
        try:
            # Track execution start
            execution_id = await self._log_execution_start(workflow_name, initial_state)
            
            # Execute workflow
            start_time = datetime.utcnow()
            result = await workflow.ainvoke(initial_state, config=config)
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Log successful completion
            await self._log_execution_completion(
                execution_id, result, execution_time, success=True
            )
            
            logger.info(f"Workflow '{workflow_name}' completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            # Log execution failure
            await self._log_execution_completion(
                execution_id, {"error": str(e)}, 0, success=False
            )
            
            logger.error(f"Workflow '{workflow_name}' failed: {str(e)}")
            raise RuntimeError(
                f"Workflow execution failed for '{workflow_name}'. "
                f"Error: {str(e)}. Check agent availability and tool configuration."
            )
    
    async def _log_execution_start(self, workflow_name: str, initial_state: Dict[str, Any]) -> str:
        """Log workflow execution start to database."""
        if not self.supabase_logger:
            return "no_tracking"
        
        try:
            execution_data = {
                "runbook_name": workflow_name,
                "user_id": initial_state.get("user_id", "unknown"),
                "conversation_id": initial_state.get("conversation_id"),
                "execution_state": initial_state,
                "status": "running",
                "metadata": {
                    "started_by": "langgraph_engine",
                    "initial_state_size": len(str(initial_state))
                }
            }
            
            result = self.supabase_logger.client.table("runbook_executions").insert(execution_data).execute()
            return result.data[0]["id"] if result.data else "failed_to_log"
            
        except Exception as e:
            logger.warning(f"Failed to log execution start: {e}")
            return "log_failed"
    
    async def _log_execution_completion(self, execution_id: str, result: Dict[str, Any], 
                                      execution_time: float, success: bool):
        """Log workflow execution completion."""
        if not self.supabase_logger or execution_id in ["no_tracking", "failed_to_log", "log_failed"]:
            return
        
        try:
            update_data = {
                "status": "completed" if success else "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "execution_state": result,
                "metadata": {
                    "execution_time_seconds": execution_time,
                    "success": success,
                    "result_size": len(str(result))
                }
            }
            
            if not success:
                update_data["error_message"] = result.get("error", "Unknown error")
            
            self.supabase_logger.client.table("runbook_executions").update(update_data).eq("id", execution_id).execute()
            
        except Exception as e:
            logger.warning(f"Failed to log execution completion: {e}")
    
    def get_loaded_workflows(self) -> List[str]:
        """Get list of currently loaded workflow names."""
        return list(self.active_workflows.keys())
    
    def is_available(self) -> bool:
        """Check if LangGraph is available and engine is functional."""
        return LANGGRAPH_AVAILABLE
    
    async def close(self):
        """Clean up workflow engine resources."""
        self.active_workflows.clear()
        logger.info("LangGraph Workflow Engine closed") 