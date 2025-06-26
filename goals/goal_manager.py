"""
Filename: goal_manager.py
Purpose: Intelligent goal decomposition and task orchestration system
Dependencies: langchain, openai, asyncio, logging, typing, supabase

This module transforms high-level user goals into actionable task sequences,
manages resource budgets, and coordinates multi-agent workflows.
"""

import asyncio
import logging
import os
import json
import uuid
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class GoalType(Enum):
    """Hierarchical goal types for strategic planning."""
    STRATEGIC = "strategic"     # Long-term, high-level objectives
    TACTICAL = "tactical"       # Medium-term, specific initiatives
    OPERATIONAL = "operational" # Short-term, immediate tasks

class GoalStatus(Enum):
    """Goal execution status tracking."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class TaskStatus(Enum):
    """Individual task status tracking."""
    WAITING = "waiting"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

class Priority(Enum):
    """Task and goal priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    """Individual task within a goal."""
    id: str
    goal_id: str
    title: str
    description: str
    assigned_agent: Optional[str] = None
    required_agent_type: Optional[str] = None
    dependencies: List[str] = None  # Task IDs this task depends on
    estimated_duration: Optional[int] = None  # Minutes
    estimated_cost: Optional[float] = None  # USD
    priority: Priority = Priority.MEDIUM
    status: TaskStatus = TaskStatus.WAITING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class Goal:
    """Hierarchical goal with task decomposition."""
    id: str
    user_id: str
    title: str
    description: str
    goal_type: GoalType
    parent_goal_id: Optional[str] = None
    child_goals: List[str] = None
    tasks: List[Task] = None
    priority: Priority = Priority.MEDIUM
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    max_agents: int = 5
    max_cost: float = 2.0  # USD
    max_duration: Optional[int] = None  # Minutes
    actual_cost: float = 0.0
    actual_duration: Optional[int] = None
    success_criteria: List[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.child_goals is None:
            self.child_goals = []
        if self.tasks is None:
            self.tasks = []
        if self.success_criteria is None:
            self.success_criteria = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class GoalDecomposition(BaseModel):
    """LLM response schema for goal decomposition."""
    goal_type: str = Field(description="strategic, tactical, or operational")
    tasks: List[Dict[str, Any]] = Field(description="List of decomposed tasks")
    dependencies: List[List[str]] = Field(description="Task dependencies as [task_id, depends_on_task_id] pairs")
    estimated_duration: int = Field(description="Total estimated duration in minutes")
    estimated_cost: float = Field(description="Total estimated cost in USD")
    success_criteria: List[str] = Field(description="Measurable success criteria")
    required_agents: List[str] = Field(description="Types of agents needed")

class AgentAssignment(BaseModel):
    """LLM response schema for agent assignment."""
    task_id: str = Field(description="Task ID to assign")
    agent_type: str = Field(description="Best agent type for this task")
    agent_specialty: Optional[str] = Field(description="Specific specialty if new agent needed")
    confidence: float = Field(description="Confidence in assignment (0.0-1.0)")
    reasoning: str = Field(description="Why this agent is best for the task")

class GoalManager:
    """
    Intelligent goal management system that decomposes high-level goals into actionable tasks
    and coordinates multi-agent execution with resource management.
    """
    
    def __init__(self, orchestrator=None, db_logger=None):
        """
        Initialize the Goal Manager with LLM capabilities and resource tracking.
        
        Args:
            orchestrator: Agent orchestrator for spawning specialists
            db_logger: Supabase logger for persistent storage
        """
        self.orchestrator = orchestrator
        self.db_logger = db_logger
        
        # In-memory storage (will be persisted to Supabase)
        self.goals: Dict[str, Goal] = {}
        self.tasks: Dict[str, Task] = {}
        self.active_executions: Dict[str, asyncio.Task] = {}
        
        # Resource tracking
        self.resource_usage = defaultdict(float)  # Track costs per user
        self.agent_assignments = defaultdict(list)  # Track agent usage
        
        # Initialize LLMs for different purposes
        self.decomposition_llm = ChatOpenAI(
            model="gpt-4-0125-preview",  # More capable model for complex decomposition
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2000,
        )
        
        self.assignment_llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",  # Faster model for agent assignment
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=500,
        )
        
        # Create prompt templates
        self.decomposition_prompt = self._create_decomposition_prompt()
        self.assignment_prompt = self._create_assignment_prompt()
        self.progress_prompt = self._create_progress_prompt()
        
        # Create parsing chains
        self.decomposition_parser = JsonOutputParser(pydantic_object=GoalDecomposition)
        self.assignment_parser = JsonOutputParser(pydantic_object=AgentAssignment)
        
        self.decomposition_chain = self.decomposition_prompt | self.decomposition_llm | self.decomposition_parser
        self.assignment_chain = self.assignment_prompt | self.assignment_llm | self.assignment_parser
        
        logger.info("Goal Manager initialized with intelligent decomposition capabilities")
    
    def _create_decomposition_prompt(self) -> ChatPromptTemplate:
        """Create prompt for intelligent goal decomposition."""
        
        system_template = """You are an expert goal decomposition specialist for an AI Agent Platform. 
Your role is to break down high-level user goals into specific, actionable tasks.

**Available Agent Types:**
1. **General Agent** - Conversations, questions, general assistance
2. **Technical Agent** - Programming, debugging, system administration, DevOps
3. **Research Agent** - Research, analysis, data gathering, market intelligence
4. **Universal Specialists** - Can be spawned for any specific domain (e.g., "API Documentation Expert", "Python Performance Optimizer")

**Goal Types:**
- **Strategic** (weeks/months): High-level business or project objectives
- **Tactical** (days/weeks): Specific initiatives to support strategic goals
- **Operational** (hours/days): Immediate actionable tasks

**Decomposition Principles:**
1. Break goals into specific, measurable tasks
2. Each task should be completable by a single agent
3. Identify dependencies between tasks
4. Estimate realistic time and cost for each task
5. Consider resource constraints (max 5 agents, $2 budget typical)
6. Define clear success criteria

**Cost Estimation Guidelines:**
- Simple questions/conversations: $0.05-0.15
- Technical analysis/debugging: $0.20-0.50
- Research tasks: $0.30-0.80
- Complex multi-step tasks: $0.50-1.50

**Output Format:**
Always respond with valid JSON matching the required schema.

User Context: {context}
Goal Type Hint: {goal_type_hint}"""

        human_template = """User Goal: "{goal_description}"

Please decompose this goal into specific tasks with the following considerations:
1. What type of goal is this? (strategic/tactical/operational)
2. What specific tasks are needed to complete this goal?
3. What are the dependencies between tasks?
4. What types of agents are needed?
5. What are realistic time and cost estimates?
6. How will we measure success?

Decompose this goal:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_assignment_prompt(self) -> ChatPromptTemplate:
        """Create prompt for intelligent agent assignment."""
        
        system_template = """You are an agent assignment specialist for an AI Agent Platform.
Your role is to assign tasks to the most appropriate agents based on task requirements.

**Available Agent Types:**
1. **General Agent** - General conversations, basic questions, coordination
2. **Technical Agent** - Programming, debugging, technical documentation, system design
3. **Research Agent** - Research, data analysis, competitive intelligence, market research
4. **Universal Specialists** - Domain experts spawned on-demand

**Assignment Guidelines:**
1. Match task complexity to agent capabilities
2. Consider current agent workload
3. Minimize agent switching for efficiency
4. Spawn specialists only when general agents can't handle the task
5. Group related tasks to the same agent when possible

**Specialist Spawning Criteria:**
- Task requires deep domain expertise not covered by general agents
- Task involves specialized tools or knowledge
- Multiple similar tasks could benefit from a domain expert

Current Available Agents: {available_agents}
Task Queue: {task_queue}"""

        human_template = """Task to Assign:
ID: {task_id}
Title: {task_title}  
Description: {task_description}
Required Agent Type: {required_agent_type}

Context: {context}

Assign the best agent for this task:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_progress_prompt(self) -> ChatPromptTemplate:
        """Create prompt for progress analysis and optimization."""
        
        system_template = """You are a goal progress analyzer for an AI Agent Platform.
Analyze goal execution progress and suggest optimizations.

**Analysis Framework:**
1. **Progress Assessment**: How much of the goal is complete?
2. **Bottleneck Identification**: What's blocking progress?
3. **Resource Optimization**: Are resources being used efficiently?
4. **Timeline Adjustment**: Should estimates be revised?
5. **Agent Performance**: Are agents performing well?
6. **Success Prediction**: Likelihood of goal completion?

**Optimization Strategies:**
- Parallel task execution
- Agent reallocation
- Task priority adjustment
- Resource budget reallocation
- Dependency restructuring

Goal Data: {goal_data}
Task Status: {task_status}
Resource Usage: {resource_usage}"""

        human_template = """Analyze progress for goal: "{goal_title}"

Current status: {current_status}
Completed tasks: {completed_tasks}
Blocked tasks: {blocked_tasks}
Time elapsed: {time_elapsed}
Budget used: {budget_used}

Provide analysis and optimization recommendations:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    async def create_goal(self, user_id: str, goal_description: str, 
                         context: Optional[Dict[str, Any]] = None,
                         goal_type_hint: Optional[str] = None,
                         max_agents: int = 5,
                         max_cost: float = 2.0,
                         max_duration: Optional[int] = None) -> str:
        """
        Create a new goal with intelligent task decomposition.
        
        Args:
            user_id: User creating the goal
            goal_description: High-level goal description
            context: Additional context for decomposition
            goal_type_hint: Hint for goal type (strategic/tactical/operational)
            max_agents: Maximum agents for this goal
            max_cost: Maximum cost budget in USD
            max_duration: Maximum duration in minutes
            
        Returns:
            Goal ID if successful
        """
        try:
            logger.info(f"Creating goal for user {user_id}: {goal_description[:100]}...")
            
            # Generate unique goal ID
            goal_id = f"goal_{uuid.uuid4().hex[:12]}"
            
            # Decompose goal using LLM
            decomposition = await self._decompose_goal(
                goal_description, context or {}, goal_type_hint
            )
            
            # Create goal object
            goal = Goal(
                id=goal_id,
                user_id=user_id,
                title=self._extract_title_from_description(goal_description),
                description=goal_description,
                goal_type=GoalType(decomposition.get("goal_type", "operational")),
                max_agents=max_agents,
                max_cost=max_cost,
                max_duration=max_duration,
                success_criteria=decomposition.get("success_criteria", ["Goal completed successfully"])
            )
            
            # Create tasks from decomposition
            task_objects = []
            for i, task_data in enumerate(decomposition.get("tasks", [])):
                task_id = f"{goal_id}_task_{i+1:02d}"
                
                task = Task(
                    id=task_id,
                    goal_id=goal_id,
                    title=task_data.get("title", f"Task {i+1}"),
                    description=task_data.get("description", goal_description),
                    required_agent_type=task_data.get("agent_type"),
                    estimated_duration=task_data.get("duration_minutes"),
                    estimated_cost=task_data.get("cost"),
                    priority=Priority(task_data.get("priority", 2))
                )
                
                task_objects.append(task)
                self.tasks[task_id] = task
            
            # Set up task dependencies
            for dep_pair in decomposition.get("dependencies", []):
                if len(dep_pair) == 2:
                    task_id, depends_on = dep_pair
                    if task_id in self.tasks:
                        self.tasks[task_id].dependencies.append(depends_on)
            
            goal.tasks = task_objects
            self.goals[goal_id] = goal
            
            # Persist to database
            await self._save_goal_to_db(goal)
            
            # Log goal creation event
            if self.db_logger:
                await self.db_logger.log_event(
                    "goal_created",
                    {
                        "goal_id": goal_id,
                        "user_id": user_id,
                        "goal_type": goal.goal_type.value,
                        "task_count": len(task_objects),
                        "estimated_cost": decomposition.get("estimated_cost", 0.0),
                        "estimated_duration": decomposition.get("estimated_duration", 30)
                    },
                    user_id
                )
            
            logger.info(f"Created goal {goal_id} with {len(task_objects)} tasks")
            return goal_id
            
        except Exception as e:
            logger.error(f"Error creating goal: {str(e)}")
            raise
    
    async def execute_goal(self, goal_id: str, auto_start: bool = True) -> bool:
        """
        Execute a goal by orchestrating task execution with appropriate agents.
        
        Args:
            goal_id: Goal to execute
            auto_start: Automatically start task execution
            
        Returns:
            True if execution started successfully
        """
        try:
            if goal_id not in self.goals:
                raise ValueError(f"Goal {goal_id} not found")
            
            goal = self.goals[goal_id]
            
            if goal.status not in [GoalStatus.PENDING, GoalStatus.BLOCKED]:
                logger.warning(f"Goal {goal_id} is not in a state to be executed: {goal.status}")
                return False
            
            # Check resource budget
            if goal.actual_cost >= goal.max_cost:
                logger.warning(f"Goal {goal_id} has exceeded cost budget")
                goal.status = GoalStatus.BLOCKED
                return False
            
            # Update goal status
            goal.status = GoalStatus.IN_PROGRESS
            goal.started_at = datetime.utcnow()
            
            # Start execution task
            execution_task = asyncio.create_task(self._execute_goal_tasks(goal_id))
            self.active_executions[goal_id] = execution_task
            
            # Log execution start
            if self.db_logger:
                await self.db_logger.log_event(
                    "goal_execution_started",
                    {"goal_id": goal_id, "task_count": len(goal.tasks)},
                    goal.user_id
                )
            
            logger.info(f"Started execution of goal {goal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing goal {goal_id}: {str(e)}")
            return False
    
    async def _execute_goal_tasks(self, goal_id: str):
        """Execute all tasks for a goal in dependency order."""
        try:
            goal = self.goals[goal_id]
            completed_tasks = set()
            
            while len(completed_tasks) < len(goal.tasks):
                # Find ready tasks (dependencies satisfied)
                ready_tasks = []
                
                for task in goal.tasks:
                    if (task.id not in completed_tasks and 
                        task.status == TaskStatus.WAITING and
                        all(dep in completed_tasks for dep in task.dependencies)):
                        ready_tasks.append(task)
                
                if not ready_tasks:
                    # Check if we're blocked
                    remaining_tasks = [t for t in goal.tasks if t.id not in completed_tasks]
                    if any(t.status == TaskStatus.FAILED for t in remaining_tasks):
                        goal.status = GoalStatus.FAILED
                        break
                    elif any(t.status == TaskStatus.BLOCKED for t in remaining_tasks):
                        goal.status = GoalStatus.BLOCKED
                        break
                    else:
                        # Wait a bit and retry
                        await asyncio.sleep(1)
                        continue
                
                # Execute ready tasks (potentially in parallel)
                task_executions = []
                for task in ready_tasks[:goal.max_agents]:  # Respect agent limit
                    task_executions.append(self._execute_task(task))
                
                # Wait for at least one task to complete
                if task_executions:
                    done, pending = await asyncio.wait(
                        task_executions, 
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Process completed tasks
                    for task_result in done:
                        task_id, success = await task_result
                        if success:
                            completed_tasks.add(task_id)
                            # Update goal progress
                            goal.progress = len(completed_tasks) / len(goal.tasks)
                
                # Small delay to prevent tight loops
                await asyncio.sleep(0.1)
            
            # Finalize goal
            if len(completed_tasks) == len(goal.tasks):
                goal.status = GoalStatus.COMPLETED
                goal.completed_at = datetime.utcnow()
                
                # Calculate actual duration
                if goal.started_at:
                    duration = (goal.completed_at - goal.started_at).total_seconds() / 60
                    goal.actual_duration = int(duration)
                
                logger.info(f"Goal {goal_id} completed successfully")
                
                # Log completion
                if self.db_logger:
                    await self.db_logger.log_event(
                        "goal_completed",
                        {
                            "goal_id": goal_id,
                            "completion_time": goal.actual_duration,
                            "final_cost": goal.actual_cost,
                            "success_rate": len(completed_tasks) / len(goal.tasks)
                        },
                        goal.user_id
                    )
            
            # Update database
            await self._save_goal_to_db(goal)
            
        except Exception as e:
            logger.error(f"Error executing goal tasks for {goal_id}: {str(e)}")
            goal.status = GoalStatus.FAILED
    
    async def _execute_task(self, task: Task) -> Tuple[str, bool]:
        """Execute a single task with appropriate agent."""
        try:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.utcnow()
            
            # Assign agent to task
            agent = await self._assign_agent_to_task(task)
            
            if not agent:
                task.status = TaskStatus.FAILED
                task.error_message = "Could not assign agent to task"
                return task.id, False
            
            # Execute task with agent
            context = {
                "goal_id": task.goal_id,
                "task_id": task.id,
                "user_id": self.goals[task.goal_id].user_id
            }
            
            result = await agent.process_message(task.description, context)
            
            # Update task with result
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
            # Update costs
            if "processing_cost" in result:
                cost = result["processing_cost"]
                task.estimated_cost = cost
                self.goals[task.goal_id].actual_cost += cost
            
            logger.info(f"Task {task.id} completed successfully")
            return task.id, True
            
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {str(e)}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            return task.id, False
    
    async def _assign_agent_to_task(self, task: Task) -> Optional[Any]:
        """Assign the best agent to a task."""
        try:
            # Check if task already has an assigned agent
            if task.assigned_agent:
                if self.orchestrator:
                    return await self.orchestrator.get_or_load_agent(task.assigned_agent)
            
            # Use LLM to determine best agent assignment
            available_agents = []
            if self.orchestrator:
                stats = self.orchestrator.get_agent_stats()
                available_agents = stats.get("available_agents", [])
            
            assignment = await self.assignment_chain.ainvoke({
                "task_id": task.id,
                "task_title": task.title,
                "task_description": task.description,
                "required_agent_type": task.required_agent_type or "auto",
                "available_agents": json.dumps(available_agents),
                "task_queue": "[]",  # Could add current task queue
                "context": f"Goal: {self.goals[task.goal_id].title}"
            })
            
            # Get or spawn the recommended agent
            if assignment["agent_type"] in ["general", "technical", "research"]:
                # Use existing agent
                if self.orchestrator:
                    # Route through orchestrator
                    context = {"task_id": task.id, "goal_id": task.goal_id}
                    routing_result = await self.orchestrator.route_request(task.description, context)
                    return routing_result.get("agent")
            else:
                # Spawn specialist agent
                if self.orchestrator and assignment.get("agent_specialty"):
                    agent_id = await self.orchestrator.spawn_specialist_agent(
                        assignment["agent_specialty"],
                        {"task": task.description, "goal": self.goals[task.goal_id].title}
                    )
                    if agent_id:
                        task.assigned_agent = agent_id
                        return await self.orchestrator.get_or_load_agent(agent_id)
            
            # Fallback to general agent
            if self.orchestrator:
                return self.orchestrator.general_agent
            
            return None
            
        except Exception as e:
            logger.error(f"Error assigning agent to task {task.id}: {str(e)}")
            return None
    
    async def _decompose_goal(self, goal_description: str, context: Dict[str, Any], 
                            goal_type_hint: Optional[str]) -> Dict[str, Any]:
        """Use LLM to decompose goal into tasks."""
        try:
            decomposition = await self.decomposition_chain.ainvoke({
                "goal_description": goal_description,
                "context": json.dumps(context, indent=2),
                "goal_type_hint": goal_type_hint or "auto-detect"
            })
            
            return decomposition
            
        except Exception as e:
            logger.error(f"Error decomposing goal: {str(e)}")
            # Provide fallback decomposition
            return {
                "goal_type": "operational",
                "tasks": [
                    {
                        "title": "Complete Goal",
                        "description": goal_description,
                        "agent_type": "general",
                        "duration_minutes": 30,
                        "cost": 0.50,
                        "priority": 2
                    }
                ],
                "dependencies": [],
                "estimated_duration": 30,
                "estimated_cost": 0.50,
                "success_criteria": ["Goal request completed successfully"],
                "required_agents": ["general"]
            }
    
    def _extract_title_from_description(self, description: str) -> str:
        """Extract a concise title from goal description."""
        # Simple title extraction - could be enhanced with LLM
        words = description.split()
        if len(words) <= 8:
            return description
        return " ".join(words[:8]) + "..."
    
    async def _save_goal_to_db(self, goal: Goal):
        """Save goal and tasks to Supabase database."""
        try:
            if not self.db_logger:
                return
            
            # Convert goal to dict for storage
            goal_data = {
                "id": goal.id,
                "user_id": goal.user_id,
                "title": goal.title,
                "description": goal.description,
                "goal_type": goal.goal_type.value,
                "parent_goal_id": goal.parent_goal_id,
                "child_goals": goal.child_goals,
                "priority": goal.priority.value,
                "status": goal.status.value,
                "progress": goal.progress,
                "max_agents": goal.max_agents,
                "max_cost": goal.max_cost,
                "max_duration": goal.max_duration,
                "actual_cost": goal.actual_cost,
                "actual_duration": goal.actual_duration,
                "success_criteria": goal.success_criteria,
                "created_at": goal.created_at.isoformat(),
                "started_at": goal.started_at.isoformat() if goal.started_at else None,
                "completed_at": goal.completed_at.isoformat() if goal.completed_at else None
            }
            
            # Save goal data as an event for now (could create dedicated table)
            await self.db_logger.log_event(
                "goal_data_update",
                goal_data,
                goal.user_id
            )
            
            # Save task data
            for task in goal.tasks:
                task_data = {
                    "id": task.id,
                    "goal_id": task.goal_id,
                    "title": task.title,
                    "description": task.description,
                    "assigned_agent": task.assigned_agent,
                    "required_agent_type": task.required_agent_type,
                    "dependencies": task.dependencies,
                    "estimated_duration": task.estimated_duration,
                    "estimated_cost": task.estimated_cost,
                    "priority": task.priority.value,
                    "status": task.status.value,
                    "result": task.result,
                    "error_message": task.error_message,
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None
                }
                
                await self.db_logger.log_event(
                    "task_data_update",
                    task_data,
                    goal.user_id
                )
            
        except Exception as e:
            logger.error(f"Error saving goal to database: {str(e)}")
    
    def get_goal_status(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive status of a goal."""
        if goal_id not in self.goals:
            return None
        
        goal = self.goals[goal_id]
        
        # Calculate task statistics
        completed_tasks = [t for t in goal.tasks if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in goal.tasks if t.status == TaskStatus.FAILED]
        in_progress_tasks = [t for t in goal.tasks if t.status == TaskStatus.IN_PROGRESS]
        blocked_tasks = [t for t in goal.tasks if t.status == TaskStatus.BLOCKED]
        
        return {
            "goal_id": goal.id,
            "title": goal.title,
            "status": goal.status.value,
            "progress": goal.progress,
            "goal_type": goal.goal_type.value,
            "priority": goal.priority.value,
            "task_summary": {
                "total": len(goal.tasks),
                "completed": len(completed_tasks),
                "failed": len(failed_tasks),
                "in_progress": len(in_progress_tasks),
                "blocked": len(blocked_tasks)
            },
            "resource_usage": {
                "actual_cost": goal.actual_cost,
                "max_cost": goal.max_cost,
                "cost_remaining": goal.max_cost - goal.actual_cost,
                "actual_duration": goal.actual_duration,
                "max_duration": goal.max_duration
            },
            "timeline": {
                "created_at": goal.created_at.isoformat(),
                "started_at": goal.started_at.isoformat() if goal.started_at else None,
                "completed_at": goal.completed_at.isoformat() if goal.completed_at else None
            },
            "success_criteria": goal.success_criteria,
            "tasks": [
                {
                    "id": task.id,
                    "title": task.title,
                    "status": task.status.value,
                    "assigned_agent": task.assigned_agent,
                    "estimated_cost": task.estimated_cost,
                    "result_summary": str(task.result)[:100] + "..." if task.result else None
                }
                for task in goal.tasks
            ]
        }
    
    def get_user_goals(self, user_id: str, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all goals for a user with optional status filtering."""
        user_goals = [goal for goal in self.goals.values() if goal.user_id == user_id]
        
        if status_filter:
            status_enum = GoalStatus(status_filter)
            user_goals = [goal for goal in user_goals if goal.status == status_enum]
        
        return [self.get_goal_status(goal.id) for goal in user_goals]
    
    async def cancel_goal(self, goal_id: str, reason: str = "User requested") -> bool:
        """Cancel a goal and all its active tasks."""
        try:
            if goal_id not in self.goals:
                return False
            
            goal = self.goals[goal_id]
            goal.status = GoalStatus.CANCELLED
            goal.completed_at = datetime.utcnow()
            
            # Cancel active tasks
            for task in goal.tasks:
                if task.status == TaskStatus.IN_PROGRESS:
                    task.status = TaskStatus.FAILED
                    task.error_message = f"Cancelled: {reason}"
                    task.completed_at = datetime.utcnow()
            
            # Cancel execution task
            if goal_id in self.active_executions:
                self.active_executions[goal_id].cancel()
                del self.active_executions[goal_id]
            
            # Log cancellation
            if self.db_logger:
                await self.db_logger.log_event(
                    "goal_cancelled",
                    {"goal_id": goal_id, "reason": reason},
                    goal.user_id
                )
            
            # Update database
            await self._save_goal_to_db(goal)
            
            logger.info(f"Goal {goal_id} cancelled: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling goal {goal_id}: {str(e)}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive goal management system statistics."""
        total_goals = len(self.goals)
        active_goals = len([g for g in self.goals.values() if g.status == GoalStatus.IN_PROGRESS])
        completed_goals = len([g for g in self.goals.values() if g.status == GoalStatus.COMPLETED])
        
        total_tasks = sum(len(g.tasks) for g in self.goals.values())
        completed_tasks = sum(
            len([t for t in g.tasks if t.status == TaskStatus.COMPLETED]) 
            for g in self.goals.values()
        )
        
        total_cost = sum(g.actual_cost for g in self.goals.values())
        
        return {
            "goal_statistics": {
                "total_goals": total_goals,
                "active_goals": active_goals,
                "completed_goals": completed_goals,
                "success_rate": completed_goals / total_goals if total_goals > 0 else 0.0
            },
            "task_statistics": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0.0
            },
            "resource_usage": {
                "total_cost": total_cost,
                "active_executions": len(self.active_executions),
                "avg_cost_per_goal": total_cost / total_goals if total_goals > 0 else 0.0
            },
            "performance_metrics": {
                "goals_by_type": {
                    goal_type.value: len([g for g in self.goals.values() if g.goal_type == goal_type])
                    for goal_type in GoalType
                },
                "goals_by_status": {
                    status.value: len([g for g in self.goals.values() if g.status == status])
                    for status in GoalStatus
                }
            }
        }
    
    async def close(self):
        """Clean up goal manager resources."""
        try:
            logger.info("Closing Goal Manager...")
            
            # Cancel all active executions
            for goal_id, execution_task in self.active_executions.items():
                execution_task.cancel()
                try:
                    await execution_task
                except asyncio.CancelledError:
                    pass
            
            self.active_executions.clear()
            
            # Close LLM connections
            if hasattr(self.decomposition_llm, 'client') and hasattr(self.decomposition_llm.client, 'close'):
                await self.decomposition_llm.client.close()
            
            if hasattr(self.assignment_llm, 'client') and hasattr(self.assignment_llm.client, 'close'):
                await self.assignment_llm.client.close()
            
            logger.info("Goal Manager closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing Goal Manager: {e}") 