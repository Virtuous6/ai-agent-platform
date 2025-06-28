"""
Workflow Tracking Module

Provides centralized tracking for all workflow executions, agent spawning,
and improvement tasks to support the self-improving system.
"""

import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class WorkflowRun:
    """Represents a workflow execution."""
    run_id: str
    workflow_type: str
    user_id: str
    conversation_id: Optional[str] = None
    trigger_message: Optional[str] = None
    goal_id: Optional[str] = None
    parent_goal_id: Optional[str] = None
    agents_used: List[str] = None
    tools_used: List[str] = None
    steps_completed: List[str] = None
    status: str = "running"
    started_at: datetime = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    success: bool = False
    response: Optional[str] = None
    tokens_used: int = 0
    estimated_cost: float = 0.0
    confidence_score: Optional[float] = None
    pattern_signature: Optional[str] = None
    automation_potential: Optional[float] = None
    metadata: Dict[str, Any] = None
    error_details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.utcnow()
        if self.agents_used is None:
            self.agents_used = []
        if self.tools_used is None:
            self.tools_used = []
        if self.steps_completed is None:
            self.steps_completed = []
        if self.metadata is None:
            self.metadata = {}

class WorkflowTracker:
    """Tracks workflow executions for pattern recognition and improvement."""
    
    def __init__(self, db_logger=None):
        self.db_logger = db_logger
        self.active_runs: Dict[str, WorkflowRun] = {}
        
    async def start_workflow(self, run_id: str, message: str, context: Dict[str, Any]) -> str:
        """Start tracking a new workflow execution."""
        user_id = context.get("user_id", "anonymous")
        
        workflow_run = WorkflowRun(
            run_id=run_id,
            workflow_type="user_request",
            user_id=user_id,
            conversation_id=context.get("conversation_id"),
            trigger_message=message,
            goal_id=context.get("goal_id"),
            parent_goal_id=context.get("parent_goal_id")
        )
        
        self.active_runs[run_id] = workflow_run
        
        # Save to database
        if self.db_logger:
            try:
                run_data = asdict(workflow_run)
                run_data['started_at'] = run_data['started_at'].isoformat()
                self.db_logger.client.table("workflow_runs").insert(run_data).execute()
            except Exception as e:
                logger.error(f"Failed to save workflow run start: {e}")
        
        return run_id
    
    async def track_agent_used(self, run_id: str, agent_id: str):
        """Track that an agent was used in the workflow."""
        if run_id in self.active_runs:
            if agent_id not in self.active_runs[run_id].agents_used:
                self.active_runs[run_id].agents_used.append(agent_id)
    
    async def track_tool_used(self, run_id: str, tool_name: str):
        """Track that a tool was used in the workflow."""
        if run_id in self.active_runs:
            if tool_name not in self.active_runs[run_id].tools_used:
                self.active_runs[run_id].tools_used.append(tool_name)
    
    async def track_step_completed(self, run_id: str, step_name: str):
        """Track completion of a workflow step."""
        if run_id in self.active_runs:
            self.active_runs[run_id].steps_completed.append(step_name)
    
    async def complete_workflow(self, run_id: str, result: Dict[str, Any]):
        """Complete a workflow execution."""
        if run_id not in self.active_runs:
            logger.warning(f"Workflow run {run_id} not found")
            return
        
        workflow_run = self.active_runs[run_id]
        workflow_run.completed_at = datetime.utcnow()
        workflow_run.duration_ms = int((workflow_run.completed_at - workflow_run.started_at).total_seconds() * 1000)
        
        # Extract values from result
        success = result.get("success", True)
        response = result.get("response", "")
        tokens_used = result.get("tokens_used", 0)
        estimated_cost = result.get("estimated_cost", 0.0)
        confidence_score = result.get("confidence_score")
        pattern_signature = result.get("pattern_signature")
        automation_potential = result.get("automation_potential")
        metadata = result.get("metadata", {})
        
        workflow_run.status = "completed" if success else "failed"
        workflow_run.success = success
        workflow_run.response = response
        workflow_run.tokens_used = tokens_used
        workflow_run.estimated_cost = estimated_cost
        workflow_run.confidence_score = confidence_score
        workflow_run.pattern_signature = pattern_signature
        workflow_run.automation_potential = automation_potential
        
        if metadata:
            workflow_run.metadata.update(metadata)
        
        # Notify goal completion if linked
        if workflow_run.goal_id and success:
            await self._notify_goal_completion(workflow_run.goal_id, workflow_run)
        
        # Save to database
        if self.db_logger:
            try:
                update_data = {
                    "completed_at": workflow_run.completed_at.isoformat(),
                    "duration_ms": workflow_run.duration_ms,
                    "status": workflow_run.status,
                    "success": workflow_run.success,
                    "response": workflow_run.response,
                    "tokens_used": workflow_run.tokens_used,
                    "estimated_cost": workflow_run.estimated_cost,
                    "confidence_score": workflow_run.confidence_score,
                    "pattern_signature": workflow_run.pattern_signature,
                    "automation_potential": workflow_run.automation_potential,
                    "agents_used": workflow_run.agents_used,
                    "tools_used": workflow_run.tools_used,
                    "steps_completed": workflow_run.steps_completed,
                    "metadata": workflow_run.metadata,
                    "goal_id": workflow_run.goal_id,
                    "parent_goal_id": workflow_run.parent_goal_id
                }
                
                self.db_logger.client.table("workflow_runs").update(update_data).eq("run_id", run_id).execute()
            except Exception as e:
                logger.error(f"Failed to save workflow run completion: {e}")
        
        # Remove from active runs
        del self.active_runs[run_id]
    
    async def _notify_goal_completion(self, goal_id: str, workflow_run: WorkflowRun):
        """Notify goal manager when a workflow completes successfully."""
        try:
            # This will be implemented when we integrate with goal manager
            logger.info(f"Workflow {workflow_run.run_id} completed for goal {goal_id}")
            
            # Future: await self.goal_manager.workflow_completed(goal_id, workflow_run)
            
        except Exception as e:
            logger.error(f"Error notifying goal completion: {e}")
    
    async def get_workflows_for_goal(self, goal_id: str) -> List[Dict[str, Any]]:
        """Get all workflows associated with a goal."""
        if not self.db_logger:
            return []
        
        try:
            result = self.db_logger.client.table("workflow_runs").select("*").eq("goal_id", goal_id).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get workflows for goal {goal_id}: {e}")
            return []

    async def fail_workflow(self, run_id: str, error_message: str, 
                          error_details: Dict[str, Any] = None):
        """Mark a workflow as failed."""
        if run_id not in self.active_runs:
            logger.warning(f"Workflow run {run_id} not found")
            return
        
        workflow_run = self.active_runs[run_id]
        workflow_run.status = "failed"
        workflow_run.success = False
        workflow_run.error_details = error_details or {"error": error_message}
        
        await self.complete_workflow(
            run_id=run_id,
            result={
                "success": False,
                "metadata": {"error_message": error_message}
            }
        )
    
    async def track_error(self, run_id: str, error: str):
        """Track an error that occurred during workflow execution."""
        if run_id in self.active_runs:
            await self.fail_workflow(run_id, error)
        else:
            logger.warning(f"Cannot track error for unknown workflow {run_id}")
    
    async def get_recent_workflows(self, user_id: str = None, 
                                 workflow_type: str = None,
                                 days: int = 7) -> List[Dict[str, Any]]:
        """Get recent workflow executions for analysis."""
        if not self.db_logger:
            return []
        
        try:
            from_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            query = self.db_logger.client.table("workflow_runs").select("*").gte("created_at", from_date)
            
            if user_id:
                query = query.eq("user_id", user_id)
            if workflow_type:
                query = query.eq("workflow_type", workflow_type)
            
            result = query.execute()
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to get recent workflows: {e}")
            return []
    
    async def get_workflow_patterns(self, min_frequency: int = 3) -> List[Dict[str, Any]]:
        """Analyze workflows to find patterns."""
        if not self.db_logger:
            return []
        
        try:
            # Get pattern signatures with frequency
            result = self.db_logger.client.rpc(
                "get_workflow_patterns",
                {"min_frequency": min_frequency}
            ).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            # Fallback to manual pattern extraction
            logger.warning(f"Pattern RPC failed, using fallback: {e}")
            workflows = await self.get_recent_workflows(days=30)
            
            # Group by pattern signature
            patterns = {}
            for workflow in workflows:
                sig = workflow.get("pattern_signature")
                if sig:
                    if sig not in patterns:
                        patterns[sig] = {
                            "pattern_signature": sig,
                            "frequency": 0,
                            "avg_duration_ms": 0,
                            "success_rate": 0,
                            "total_success": 0,
                            "automation_potential": 0
                        }
                    
                    patterns[sig]["frequency"] += 1
                    patterns[sig]["avg_duration_ms"] += workflow.get("duration_ms", 0)
                    if workflow.get("success"):
                        patterns[sig]["total_success"] += 1
                    patterns[sig]["automation_potential"] = max(
                        patterns[sig]["automation_potential"],
                        workflow.get("automation_potential", 0)
                    )
            
            # Calculate averages
            pattern_list = []
            for sig, data in patterns.items():
                if data["frequency"] >= min_frequency:
                    data["avg_duration_ms"] = data["avg_duration_ms"] / data["frequency"]
                    data["success_rate"] = data["total_success"] / data["frequency"]
                    pattern_list.append(data)
            
            return pattern_list 