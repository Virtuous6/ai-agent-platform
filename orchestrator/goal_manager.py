"""
Goal Management System for AI Agent Platform

Provides goal state management, progress tracking, and completion assessment
for orchestrator-driven workflows.
"""

import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class GoalStatus(Enum):
    """Goal execution status."""
    CREATED = "created"
    IN_PROGRESS = "in_progress" 
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class GoalPriority(Enum):
    """Goal priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5

@dataclass
class GoalCriteria:
    """Success criteria for a goal."""
    criteria_id: str
    description: str
    required: bool = True
    completed: bool = False
    completion_evidence: Optional[str] = None
    weight: float = 1.0  # For weighted completion calculation

@dataclass 
class GoalProgress:
    """Goal progress tracking."""
    goal_id: str
    completion_percentage: float
    completed_criteria: List[str]
    pending_criteria: List[str] 
    blocking_issues: List[str]
    agent_contributions: Dict[str, float]
    estimated_completion: Optional[datetime]
    needs_more_agents: bool = False
    needs_human_input: bool = False
    
@dataclass
class Goal:
    """Represents a goal being executed by the orchestrator."""
    goal_id: str
    description: str
    success_criteria: List[GoalCriteria]
    status: GoalStatus
    priority: GoalPriority
    created_by: str
    assigned_agents: List[str]
    progress_percentage: float = 0.0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    parent_goal_id: Optional[str] = None  # For sub-goals
    child_goal_ids: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
        if self.child_goal_ids is None:
            self.child_goal_ids = []

class GoalManager:
    """Manages goal lifecycle, progress tracking, and completion assessment."""
    
    def __init__(self, db_logger=None):
        self.db_logger = db_logger
        self.active_goals: Dict[str, Goal] = {}
        self.goal_history: List[Goal] = []
        
    async def create_goal(self, description: str, success_criteria: List[str],
                         priority: GoalPriority = GoalPriority.MEDIUM,
                         created_by: str = "system",
                         deadline: Optional[datetime] = None,
                         parent_goal_id: Optional[str] = None) -> str:
        """Create a new goal with success criteria."""
        
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        
        # Convert criteria strings to GoalCriteria objects
        criteria_objects = []
        for i, criteria_desc in enumerate(success_criteria):
            criteria = GoalCriteria(
                criteria_id=f"{goal_id}_criteria_{i}",
                description=criteria_desc,
                required=True,
                weight=1.0 / len(success_criteria)  # Equal weighting by default
            )
            criteria_objects.append(criteria)
        
        goal = Goal(
            goal_id=goal_id,
            description=description,
            success_criteria=criteria_objects,
            status=GoalStatus.CREATED,
            priority=priority,
            created_by=created_by,
            assigned_agents=[],
            deadline=deadline,
            parent_goal_id=parent_goal_id
        )
        
        self.active_goals[goal_id] = goal
        
        # Add to parent's children if applicable
        if parent_goal_id and parent_goal_id in self.active_goals:
            self.active_goals[parent_goal_id].child_goal_ids.append(goal_id)
        
        # Log goal creation
        if self.db_logger:
            try:
                goal_data = {
                    "goal_id": goal_id,
                    "description": description,
                    "success_criteria": [asdict(c) for c in criteria_objects],
                    "priority": priority.value,
                    "created_by": created_by,
                    "created_at": goal.created_at.isoformat(),
                    "deadline": deadline.isoformat() if deadline else None,
                    "parent_goal_id": parent_goal_id,
                    "status": GoalStatus.CREATED.value
                }
                self.db_logger.client.table("goals").insert(goal_data).execute()
            except Exception as e:
                logger.warning(f"Failed to log goal creation: {e}")
        
        logger.info(f"Created goal {goal_id}: {description}")
        return goal_id
    
    async def start_goal(self, goal_id: str) -> bool:
        """Mark a goal as started."""
        if goal_id not in self.active_goals:
            return False
        
        goal = self.active_goals[goal_id]
        goal.status = GoalStatus.IN_PROGRESS
        goal.started_at = datetime.utcnow()
        
        await self._update_goal_in_db(goal)
        logger.info(f"Started goal {goal_id}")
        return True
    
    async def assign_agent_to_goal(self, goal_id: str, agent_id: str) -> bool:
        """Assign an agent to work on a goal."""
        if goal_id not in self.active_goals:
            return False
        
        goal = self.active_goals[goal_id]
        if agent_id not in goal.assigned_agents:
            goal.assigned_agents.append(agent_id)
            await self._update_goal_in_db(goal)
            logger.info(f"Assigned agent {agent_id} to goal {goal_id}")
        
        return True
    
    async def update_criteria_completion(self, goal_id: str, criteria_id: str, 
                                       completed: bool, evidence: str = None) -> bool:
        """Update completion status of a specific criteria."""
        if goal_id not in self.active_goals:
            return False
        
        goal = self.active_goals[goal_id]
        
        # Find and update criteria
        for criteria in goal.success_criteria:
            if criteria.criteria_id == criteria_id:
                criteria.completed = completed
                criteria.completion_evidence = evidence
                break
        else:
            logger.warning(f"Criteria {criteria_id} not found in goal {goal_id}")
            return False
        
        # Recalculate progress
        await self.calculate_goal_progress(goal_id)
        await self._update_goal_in_db(goal)
        
        logger.info(f"Updated criteria {criteria_id} completion: {completed}")
        return True
    
    async def calculate_goal_progress(self, goal_id: str) -> GoalProgress:
        """Calculate current progress for a goal."""
        if goal_id not in self.active_goals:
            raise ValueError(f"Goal {goal_id} not found")
        
        goal = self.active_goals[goal_id]
        
        # Calculate weighted completion percentage
        total_weight = sum(c.weight for c in goal.success_criteria)
        completed_weight = sum(c.weight for c in goal.success_criteria if c.completed)
        completion_percentage = (completed_weight / total_weight) * 100 if total_weight > 0 else 0
        
        # Update goal progress
        goal.progress_percentage = completion_percentage
        
        # Categorize criteria
        completed_criteria = [c.criteria_id for c in goal.success_criteria if c.completed]
        pending_criteria = [c.criteria_id for c in goal.success_criteria if not c.completed]
        
        # Check for blocking issues (simplified)
        blocking_issues = []
        if len(goal.assigned_agents) == 0 and goal.status == GoalStatus.IN_PROGRESS:
            blocking_issues.append("No agents assigned to goal")
        
        # Assess need for more agents or human input
        needs_more_agents = (
            completion_percentage < 50 and 
            len(goal.assigned_agents) < 3 and
            goal.priority.value >= GoalPriority.HIGH.value
        )
        
        needs_human_input = (
            completion_percentage > 80 or 
            len(blocking_issues) > 0 or
            goal.priority.value >= GoalPriority.CRITICAL.value
        )
        
        # Estimate completion time (simplified)
        estimated_completion = None
        if completion_percentage > 0 and goal.started_at:
            elapsed = datetime.utcnow() - goal.started_at
            total_estimated = elapsed * (100 / completion_percentage)
            estimated_completion = goal.started_at + total_estimated
        
        progress = GoalProgress(
            goal_id=goal_id,
            completion_percentage=completion_percentage,
            completed_criteria=completed_criteria,
            pending_criteria=pending_criteria,
            blocking_issues=blocking_issues,
            agent_contributions={agent_id: 1.0/len(goal.assigned_agents) 
                               for agent_id in goal.assigned_agents},
            estimated_completion=estimated_completion,
            needs_more_agents=needs_more_agents,
            needs_human_input=needs_human_input
        )
        
        return progress
    
    async def is_goal_complete(self, goal_id: str) -> bool:
        """Check if a goal is complete based on its criteria."""
        if goal_id not in self.active_goals:
            return False
        
        goal = self.active_goals[goal_id]
        
        # Check if all required criteria are completed
        required_criteria = [c for c in goal.success_criteria if c.required]
        completed_required = [c for c in required_criteria if c.completed]
        
        return len(completed_required) == len(required_criteria)
    
    async def complete_goal(self, goal_id: str, completion_notes: str = None) -> bool:
        """Mark a goal as completed."""
        if goal_id not in self.active_goals:
            return False
        
        goal = self.active_goals[goal_id]
        goal.status = GoalStatus.COMPLETED
        goal.completed_at = datetime.utcnow()
        goal.progress_percentage = 100.0
        
        if completion_notes:
            goal.metadata["completion_notes"] = completion_notes
        
        # Move to history
        self.goal_history.append(goal)
        del self.active_goals[goal_id]
        
        await self._update_goal_in_db(goal)
        logger.info(f"Completed goal {goal_id}: {goal.description}")
        
        return True
    
    async def fail_goal(self, goal_id: str, failure_reason: str) -> bool:
        """Mark a goal as failed."""
        if goal_id not in self.active_goals:
            return False
        
        goal = self.active_goals[goal_id]
        goal.status = GoalStatus.FAILED
        goal.completed_at = datetime.utcnow()
        goal.metadata["failure_reason"] = failure_reason
        
        # Move to history
        self.goal_history.append(goal)
        del self.active_goals[goal_id]
        
        await self._update_goal_in_db(goal)
        logger.warning(f"Failed goal {goal_id}: {failure_reason}")
        
        return True
    
    async def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return self.active_goals.get(goal_id)
    
    async def get_active_goals(self, created_by: str = None, 
                             priority: GoalPriority = None) -> List[Goal]:
        """Get all active goals, optionally filtered."""
        goals = list(self.active_goals.values())
        
        if created_by:
            goals = [g for g in goals if g.created_by == created_by]
        
        if priority:
            goals = [g for g in goals if g.priority == priority]
        
        return goals
    
    async def get_goals_needing_attention(self) -> List[Goal]:
        """Get goals that need human attention or more agents."""
        attention_goals = []
        
        for goal in self.active_goals.values():
            progress = await self.calculate_goal_progress(goal.goal_id)
            
            if progress.needs_human_input or progress.needs_more_agents or progress.blocking_issues:
                attention_goals.append(goal)
        
        return attention_goals
    
    async def create_sub_goal(self, parent_goal_id: str, description: str, 
                            success_criteria: List[str]) -> str:
        """Create a sub-goal under a parent goal."""
        if parent_goal_id not in self.active_goals:
            raise ValueError(f"Parent goal {parent_goal_id} not found")
        
        parent_goal = self.active_goals[parent_goal_id]
        
        sub_goal_id = await self.create_goal(
            description=description,
            success_criteria=success_criteria,
            priority=parent_goal.priority,
            created_by=parent_goal.created_by,
            parent_goal_id=parent_goal_id
        )
        
        return sub_goal_id
    
    async def get_goal_hierarchy(self, goal_id: str) -> Dict[str, Any]:
        """Get the full hierarchy for a goal (parent and children)."""
        if goal_id not in self.active_goals:
            return {}
        
        goal = self.active_goals[goal_id]
        
        hierarchy = {
            "goal": asdict(goal),
            "children": [],
            "parent": None
        }
        
        # Get parent
        if goal.parent_goal_id and goal.parent_goal_id in self.active_goals:
            hierarchy["parent"] = asdict(self.active_goals[goal.parent_goal_id])
        
        # Get children
        for child_id in goal.child_goal_ids:
            if child_id in self.active_goals:
                hierarchy["children"].append(asdict(self.active_goals[child_id]))
        
        return hierarchy
    
    async def _update_goal_in_db(self, goal: Goal):
        """Update goal in database."""
        if not self.db_logger:
            return
        
        try:
            goal_data = {
                "goal_id": goal.goal_id,
                "description": goal.description,
                "success_criteria": [asdict(c) for c in goal.success_criteria],
                "status": goal.status.value,
                "priority": goal.priority.value,
                "assigned_agents": goal.assigned_agents,
                "progress_percentage": goal.progress_percentage,
                "started_at": goal.started_at.isoformat() if goal.started_at else None,
                "completed_at": goal.completed_at.isoformat() if goal.completed_at else None,
                "deadline": goal.deadline.isoformat() if goal.deadline else None,
                "metadata": goal.metadata,
                "child_goal_ids": goal.child_goal_ids,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Upsert the goal
            existing = self.db_logger.client.table("goals").select("goal_id").eq("goal_id", goal.goal_id).execute()
            
            if existing.data:
                self.db_logger.client.table("goals").update(goal_data).eq("goal_id", goal.goal_id).execute()
            else:
                goal_data["created_at"] = goal.created_at.isoformat()
                self.db_logger.client.table("goals").insert(goal_data).execute()
                
        except Exception as e:
            logger.warning(f"Failed to update goal in database: {e}")
    
    async def close(self):
        """Clean up goal manager resources."""
        logger.info("Goal Manager closed") 