"""
Filename: improvement_orchestrator.py
Purpose: Central orchestrator for coordinating all improvement agents in continuous cycles
Dependencies: asyncio, logging, typing, langchain, supabase

This module coordinates all improvement agents to create a truly self-improving AI system
that learns, optimizes, and evolves continuously without impacting user experience.
"""

import asyncio
import logging
import os
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ImprovementCycle(Enum):
    """Types of improvement cycles."""
    REAL_TIME = "real_time"          # Continuous monitoring
    HOURLY = "hourly"               # Every hour
    DAILY = "daily"                 # Every 24 hours  
    WEEKLY = "weekly"               # Every 7 days
    MONTHLY = "monthly"             # Every 30 days

class ImprovementPriority(Enum):
    """Priority levels for improvement tasks."""
    CRITICAL = 5    # Immediate action required
    HIGH = 4        # Should be addressed soon
    MEDIUM = 3      # Normal priority
    LOW = 2         # Can be deferred
    MINIMAL = 1     # Background optimization

class ResourceAllocation(Enum):
    """Resource allocation levels for improvement activities."""
    MINIMAL = "minimal"      # <5% system resources
    LOW = "low"             # 5-15% system resources
    MODERATE = "moderate"   # 15-30% system resources
    HIGH = "high"           # 30-50% system resources
    INTENSIVE = "intensive" # >50% system resources (off-hours only)

@dataclass
class ImprovementTask:
    """Represents an improvement task to be executed."""
    id: str
    agent_type: str
    method_name: str
    priority: ImprovementPriority
    cycle: ImprovementCycle
    resource_allocation: ResourceAllocation
    parameters: Dict[str, Any]
    expected_duration: timedelta
    expected_benefit: Dict[str, float]  # {"cost_savings": 25.0, "performance_gain": 15.0}
    dependencies: List[str]  # Task IDs this depends on
    max_retries: int = 3
    created_at: datetime = None
    scheduled_for: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class ImprovementResult:
    """Results from executing an improvement task."""
    task_id: str
    success: bool
    duration: timedelta
    actual_benefit: Dict[str, float]
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    error_message: Optional[str] = None
    side_effects: List[str] = None
    roi: float = 0.0  # Return on investment
    confidence: float = 0.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.side_effects is None:
            self.side_effects = []

@dataclass
class SystemHealth:
    """Overall system health metrics."""
    overall_score: float
    performance_score: float
    cost_efficiency_score: float
    user_satisfaction_score: float
    improvement_velocity: float
    active_issues_count: int
    recent_optimizations: int
    roi_last_30_days: float
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class ImprovementPlan(BaseModel):
    """LLM response schema for improvement planning."""
    priority_tasks: List[Dict[str, Any]] = Field(description="High priority improvement tasks")
    optimization_sequence: List[str] = Field(description="Optimal sequence for executing improvements")
    resource_allocation: Dict[str, str] = Field(description="Resource allocation recommendations")
    risk_assessment: Dict[str, Any] = Field(description="Risk analysis for proposed improvements")
    expected_roi: Dict[str, float] = Field(description="Expected return on investment")

class ImprovementOrchestrator:
    """
    Central orchestrator that coordinates all improvement agents in continuous cycles.
    
    Features:
    - Continuous improvement loops (real-time, hourly, daily, weekly)
    - Priority-based task management with dependency resolution
    - Resource allocation to prevent impact on user experience
    - ROI tracking and optimization effectiveness measurement
    - Intelligent coordination of all improvement agents
    - Non-disruptive operation with load balancing
    """
    
    def __init__(self, main_orchestrator=None, db_logger=None):
        """
        Initialize the Improvement Orchestrator.
        
        Args:
            main_orchestrator: Reference to main agent orchestrator
            db_logger: Supabase logger for metrics and tracking
        """
        self.main_orchestrator = main_orchestrator
        self.db_logger = db_logger
        
        # Core components - will be initialized lazily
        self.improvement_agents: Dict[str, Any] = {}
        self.active_tasks: Dict[str, ImprovementTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)  # Keep last 1000 tasks
        self.task_queue: Dict[ImprovementCycle, List[ImprovementTask]] = {
            cycle: [] for cycle in ImprovementCycle
        }
        
        # System monitoring
        self.system_health_history: deque = deque(maxlen=168)  # 1 week of hourly data
        self.current_system_load: float = 0.0
        self.improvement_metrics: Dict[str, Any] = {}
        
        # Coordination state
        self.cycle_last_run: Dict[ImprovementCycle, datetime] = {}
        self.cycle_schedules: Dict[ImprovementCycle, timedelta] = {
            ImprovementCycle.REAL_TIME: timedelta(seconds=30),
            ImprovementCycle.HOURLY: timedelta(hours=1),
            ImprovementCycle.DAILY: timedelta(days=1),
            ImprovementCycle.WEEKLY: timedelta(days=7),
            ImprovementCycle.MONTHLY: timedelta(days=30)
        }
        
        # Resource management
        self.max_concurrent_tasks = 3
        self.resource_usage_threshold = 0.3  # Don't exceed 30% system resources
        self.user_activity_threshold = 0.1   # Pause improvements if high user activity
        
        # Performance tracking
        self.improvement_roi_tracking: Dict[str, List[float]] = defaultdict(list)
        self.agent_effectiveness: Dict[str, float] = {}
        self.optimization_success_rates: Dict[str, float] = {}
        
        # Initialize LLM for intelligent coordination
        self.coordination_llm = ChatOpenAI(
            model="gpt-4-0125-preview",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2000,
        )
        
        # Create coordination prompt
        self.coordination_prompt = self._create_coordination_prompt()
        self.coordination_parser = JsonOutputParser(pydantic_object=ImprovementPlan)
        self.coordination_chain = self.coordination_prompt | self.coordination_llm | self.coordination_parser
        
        # Task management
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.orchestrator_task: Optional[asyncio.Task] = None
        
        # Initialize improvement cycles
        self._initialize_improvement_agents()
        self._start_orchestration_loop()
        
        logger.info("Improvement Orchestrator initialized with continuous self-improvement cycles")

    def _create_coordination_prompt(self) -> ChatPromptTemplate:
        """Create prompt for intelligent improvement coordination."""
        
        system_template = """You are the Improvement Orchestrator for a self-improving AI Agent Platform.
Your role is to intelligently coordinate all improvement agents to maximize system performance
while ensuring zero impact on user experience.

AVAILABLE IMPROVEMENT AGENTS:
- WorkflowAnalyst: Analyzes workflows for patterns and optimizations
- AgentPerformanceAnalyst: Optimizes individual agent performance
- CostOptimizer: Reduces operational costs through intelligent optimization
- ErrorRecovery: Learns from errors and implements recovery strategies
- PatternRecognition: Identifies user behavior patterns for automation
- KnowledgeGraph: Builds knowledge networks for cross-agent learning
- FeedbackHandler: Processes user feedback for system improvements

SYSTEM CONSTRAINTS:
- Maximum 30% system resource usage during peak hours
- Zero user experience impact (response times must remain under 2s)
- Prioritize high-ROI improvements (>20% benefit/cost ratio)
- Coordinate agents to avoid conflicts and maximize synergies
- Ensure dependency resolution and optimal execution sequencing

CURRENT SYSTEM STATE:
System Health: {system_health}
Active User Load: {user_load}
Recent Performance: {recent_performance}
Available Resources: {available_resources}
Pending Improvements: {pending_tasks}

Analyze the current state and create an optimal improvement plan that maximizes
value while respecting all constraints."""

        human_template = """Based on the current system state, create an improvement plan that:

1. Identifies the highest priority improvements based on ROI and impact
2. Sequences improvements to maximize effectiveness and minimize conflicts
3. Allocates resources appropriately based on current system load
4. Assesses risks and provides mitigation strategies
5. Estimates expected return on investment

Focus on improvements that will have the greatest positive impact on:
- User experience and satisfaction
- System performance and reliability
- Operational cost efficiency
- Learning and adaptation capabilities

System Metrics:
{system_metrics}

Recent Improvement History:
{improvement_history}

Current Issues and Opportunities:
{current_issues}

Create a comprehensive improvement plan with specific tasks, priorities, and expected outcomes."""

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])

    def _initialize_improvement_agents(self):
        """Initialize all improvement agents for coordination."""
        try:
            # Import improvement agents
            from agents.improvement.workflow_analyst import WorkflowAnalyst
            from agents.improvement.agent_performance_analyst import AgentPerformanceAnalyst
            from agents.improvement.cost_optimizer import CostOptimizer
            from agents.improvement.error_recovery import ErrorRecoveryAgent
            from agents.improvement.pattern_recognition import PatternRecognitionEngine
            from agents.improvement.knowledge_graph import KnowledgeGraphBuilder
            from agents.improvement.feedback_handler import FeedbackHandler
            
            # Initialize agents with coordination capability
            self.improvement_agents = {
                "workflow_analyst": WorkflowAnalyst(
                    db_logger=self.db_logger,
                    orchestrator=self.main_orchestrator
                ),
                "agent_performance_analyst": AgentPerformanceAnalyst(
                    db_logger=self.db_logger,
                    orchestrator=self.main_orchestrator
                ),
                "cost_optimizer": CostOptimizer(
                    db_logger=self.db_logger,
                    orchestrator=self.main_orchestrator
                ),
                "error_recovery": ErrorRecoveryAgent(
                    db_logger=self.db_logger,
                    orchestrator=self.main_orchestrator
                ),
                "pattern_recognition": PatternRecognitionEngine(
                    db_logger=self.db_logger,
                    orchestrator=self.main_orchestrator
                ),
                "knowledge_graph": KnowledgeGraphBuilder(
                    db_logger=self.db_logger,
                    orchestrator=self.main_orchestrator
                ),
                "feedback_handler": FeedbackHandler(
                    db_logger=self.db_logger,
                    orchestrator=self.main_orchestrator
                )
            }
            
            logger.info(f"Initialized {len(self.improvement_agents)} improvement agents")
            
        except ImportError as e:
            logger.error(f"Failed to import improvement agents: {e}")
            self.improvement_agents = {}

    def _start_orchestration_loop(self):
        """Start the main orchestration loop."""
        if self.orchestrator_task is None or self.orchestrator_task.done():
            self.orchestrator_task = asyncio.create_task(self._orchestration_loop())
            logger.info("Started improvement orchestration loop")

    async def _orchestration_loop(self):
        """Main orchestration loop that manages all improvement cycles."""
        logger.info("Improvement orchestration loop started")
        
        while True:
            try:
                # Check system health and load
                system_health = await self._assess_system_health()
                user_load = await self._measure_user_activity()
                
                # Update system state
                self.current_system_load = await self._calculate_system_load()
                
                # Only proceed if system is healthy and not overloaded
                if (system_health.overall_score > 0.7 and 
                    user_load < self.user_activity_threshold and
                    self.current_system_load < self.resource_usage_threshold):
                    
                    # Execute due improvement cycles
                    await self._execute_due_cycles()
                    
                    # Process pending tasks
                    await self._process_task_queue()
                    
                    # Update metrics and ROI tracking
                    await self._update_improvement_metrics()
                
                else:
                    logger.debug(f"Pausing improvements - Health: {system_health.overall_score:.2f}, "
                               f"Load: {user_load:.2f}, Resources: {self.current_system_load:.2f}")
                
                # Sleep for 30 seconds before next cycle
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _execute_due_cycles(self):
        """Execute improvement cycles that are due."""
        now = datetime.utcnow()
        
        for cycle, schedule in self.cycle_schedules.items():
            last_run = self.cycle_last_run.get(cycle)
            
            if last_run is None or (now - last_run) >= schedule:
                try:
                    await self._execute_cycle(cycle)
                    self.cycle_last_run[cycle] = now
                except Exception as e:
                    logger.error(f"Error executing {cycle.value} cycle: {e}")

    async def _execute_cycle(self, cycle: ImprovementCycle):
        """Execute a specific improvement cycle."""
        logger.info(f"Executing {cycle.value} improvement cycle")
        
        # Generate improvement plan for this cycle
        improvement_plan = await self._generate_improvement_plan(cycle)
        
        # Create tasks based on the plan
        tasks = await self._create_tasks_from_plan(improvement_plan, cycle)
        
        # Add tasks to appropriate queue
        self.task_queue[cycle].extend(tasks)
        
        # Execute immediate high-priority tasks if resources allow
        if cycle == ImprovementCycle.REAL_TIME or cycle == ImprovementCycle.HOURLY:
            await self._execute_immediate_tasks(tasks)

    async def _generate_improvement_plan(self, cycle: ImprovementCycle) -> Dict[str, Any]:
        """Generate an intelligent improvement plan using LLM coordination."""
        try:
            # Gather current system state
            system_health = await self._assess_system_health()
            system_metrics = await self._gather_system_metrics()
            improvement_history = self._get_recent_improvement_history()
            current_issues = await self._identify_current_issues()
            
            # Create context for LLM
            context = {
                "system_health": asdict(system_health),
                "user_load": await self._measure_user_activity(),
                "recent_performance": await self._get_recent_performance_metrics(),
                "available_resources": 1.0 - self.current_system_load,
                "pending_tasks": len(sum(self.task_queue.values(), [])),
                "system_metrics": system_metrics,
                "improvement_history": improvement_history,
                "current_issues": current_issues
            }
            
            # Generate improvement plan
            plan = await self.coordination_chain.ainvoke(context)
            
            logger.info(f"Generated improvement plan for {cycle.value} with "
                       f"{len(plan.get('priority_tasks', []))} priority tasks")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error generating improvement plan: {e}")
            return {"priority_tasks": [], "optimization_sequence": [], 
                   "resource_allocation": {}, "risk_assessment": {}, "expected_roi": {}}

    async def _create_tasks_from_plan(self, plan: Dict[str, Any], cycle: ImprovementCycle) -> List[ImprovementTask]:
        """Create improvement tasks from the generated plan."""
        tasks = []
        
        for task_data in plan.get("priority_tasks", []):
            try:
                task = ImprovementTask(
                    id=str(uuid.uuid4()),
                    agent_type=task_data.get("agent_type", ""),
                    method_name=task_data.get("method", ""),
                    priority=ImprovementPriority(task_data.get("priority", 3)),
                    cycle=cycle,
                    resource_allocation=ResourceAllocation(task_data.get("resource_allocation", "moderate")),
                    parameters=task_data.get("parameters", {}),
                    expected_duration=timedelta(minutes=task_data.get("duration_minutes", 5)),
                    expected_benefit=task_data.get("expected_benefit", {}),
                    dependencies=task_data.get("dependencies", [])
                )
                tasks.append(task)
                
            except Exception as e:
                logger.error(f"Error creating task from plan: {e}")
        
        return tasks

    async def _execute_immediate_tasks(self, tasks: List[ImprovementTask]):
        """Execute high-priority tasks immediately if resources allow."""
        # Sort by priority
        urgent_tasks = [t for t in tasks if t.priority.value >= 4]
        urgent_tasks.sort(key=lambda x: x.priority.value, reverse=True)
        
        # Execute up to max concurrent tasks
        for task in urgent_tasks[:self.max_concurrent_tasks]:
            if len(self.running_tasks) < self.max_concurrent_tasks:
                await self._execute_task(task)

    async def _process_task_queue(self):
        """Process queued improvement tasks based on priority and resources."""
        # Collect all pending tasks across cycles
        all_tasks = []
        for cycle_tasks in self.task_queue.values():
            all_tasks.extend(cycle_tasks)
        
        if not all_tasks:
            return
        
        # Sort by priority and resolve dependencies
        ready_tasks = self._resolve_task_dependencies(all_tasks)
        ready_tasks.sort(key=lambda x: x.priority.value, reverse=True)
        
        # Execute tasks within resource limits
        for task in ready_tasks:
            if (len(self.running_tasks) < self.max_concurrent_tasks and
                await self._can_execute_task(task)):
                
                await self._execute_task(task)
                
                # Remove from queue
                for cycle_tasks in self.task_queue.values():
                    if task in cycle_tasks:
                        cycle_tasks.remove(task)
                        break

    def _resolve_task_dependencies(self, tasks: List[ImprovementTask]) -> List[ImprovementTask]:
        """Resolve task dependencies and return tasks ready for execution."""
        completed_task_ids = {task.task_id for task in self.completed_tasks}
        running_task_ids = set(self.running_tasks.keys())
        
        ready_tasks = []
        for task in tasks:
            # Check if all dependencies are completed
            dependencies_met = all(
                dep_id in completed_task_ids for dep_id in task.dependencies
            )
            
            # Check if not already running
            not_running = task.id not in running_task_ids
            
            if dependencies_met and not_running:
                ready_tasks.append(task)
        
        return ready_tasks

    async def _can_execute_task(self, task: ImprovementTask) -> bool:
        """Check if a task can be executed given current system constraints."""
        # Check resource allocation
        required_resources = self._get_resource_requirement(task.resource_allocation)
        available_resources = 1.0 - self.current_system_load
        
        if required_resources > available_resources:
            return False
        
        # Check agent availability
        agent = self.improvement_agents.get(task.agent_type)
        if not agent:
            logger.warning(f"Agent {task.agent_type} not available for task {task.id}")
            return False
        
        # Check method exists
        if not hasattr(agent, task.method_name):
            logger.warning(f"Method {task.method_name} not found on agent {task.agent_type}")
            return False
        
        return True

    def _get_resource_requirement(self, allocation: ResourceAllocation) -> float:
        """Get resource requirement as fraction (0.0-1.0) for allocation level."""
        requirements = {
            ResourceAllocation.MINIMAL: 0.05,
            ResourceAllocation.LOW: 0.10,
            ResourceAllocation.MODERATE: 0.20,
            ResourceAllocation.HIGH: 0.40,
            ResourceAllocation.INTENSIVE: 0.60
        }
        return requirements.get(allocation, 0.20)

    async def _execute_task(self, task: ImprovementTask):
        """Execute an improvement task asynchronously."""
        logger.info(f"Executing improvement task: {task.agent_type}.{task.method_name}")
        
        # Create async task for execution
        execution_task = asyncio.create_task(self._run_improvement_task(task))
        self.running_tasks[task.id] = execution_task
        self.active_tasks[task.id] = task

    async def _run_improvement_task(self, task: ImprovementTask) -> ImprovementResult:
        """Run a single improvement task and track results."""
        start_time = datetime.utcnow()
        metrics_before = await self._capture_system_metrics()
        
        try:
            # Get the agent and method
            agent = self.improvement_agents[task.agent_type]
            method = getattr(agent, task.method_name)
            
            # Execute the improvement
            if task.parameters:
                result = await method(**task.parameters)
            else:
                result = await method()
            
            # Capture metrics after
            metrics_after = await self._capture_system_metrics()
            duration = datetime.utcnow() - start_time
            
            # Calculate actual benefit and ROI
            actual_benefit = await self._calculate_actual_benefit(
                metrics_before, metrics_after, task.expected_benefit
            )
            roi = await self._calculate_roi(task, actual_benefit, duration)
            
            # Create result record
            improvement_result = ImprovementResult(
                task_id=task.id,
                success=True,
                duration=duration,
                actual_benefit=actual_benefit,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                roi=roi,
                confidence=0.8
            )
            
            # Track success
            await self._track_improvement_success(task, improvement_result)
            
            logger.info(f"Completed improvement task {task.id} with ROI: {roi:.2f}")
            
            return improvement_result
            
        except Exception as e:
            duration = datetime.utcnow() - start_time
            error_result = ImprovementResult(
                task_id=task.id,
                success=False,
                duration=duration,
                actual_benefit={},
                metrics_before=metrics_before,
                metrics_after={},
                error_message=str(e),
                roi=-1.0,
                confidence=0.0
            )
            
            # Track failure
            await self._track_improvement_failure(task, error_result)
            
            logger.error(f"Failed improvement task {task.id}: {e}")
            
            return error_result
            
        finally:
            # Clean up
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            
            # Move to completed
            self.completed_tasks.append(improvement_result)

    # Helper methods for metrics and calculations

    async def _assess_system_health(self) -> SystemHealth:
        """Assess overall system health."""
        try:
            # Get metrics from various sources
            performance_metrics = await self._get_performance_metrics()
            cost_metrics = await self._get_cost_metrics()
            user_satisfaction = await self._get_user_satisfaction()
            
            # Calculate component scores
            performance_score = performance_metrics.get("avg_success_rate", 0.8)
            cost_efficiency = cost_metrics.get("efficiency_score", 0.7)
            user_score = user_satisfaction.get("satisfaction_score", 0.75)
            
            # Calculate improvement velocity
            recent_improvements = len([
                task for task in self.completed_tasks
                if task.created_at > datetime.utcnow() - timedelta(days=7) and task.success
            ])
            improvement_velocity = min(recent_improvements / 10.0, 1.0)  # Normalize to 0-1
            
            # Calculate overall score
            overall_score = (
                performance_score * 0.3 +
                cost_efficiency * 0.25 +
                user_score * 0.25 +
                improvement_velocity * 0.2
            )
            
            # Count active issues
            active_issues = await self._count_active_issues()
            
            # Calculate ROI for last 30 days
            roi_30_days = await self._calculate_recent_roi(days=30)
            
            health = SystemHealth(
                overall_score=overall_score,
                performance_score=performance_score,
                cost_efficiency_score=cost_efficiency,
                user_satisfaction_score=user_score,
                improvement_velocity=improvement_velocity,
                active_issues_count=active_issues,
                recent_optimizations=recent_improvements,
                roi_last_30_days=roi_30_days
            )
            
            # Store in history
            self.system_health_history.append(health)
            
            return health
            
        except Exception as e:
            logger.error(f"Error assessing system health: {e}")
            return SystemHealth(
                overall_score=0.5,
                performance_score=0.5,
                cost_efficiency_score=0.5,
                user_satisfaction_score=0.5,
                improvement_velocity=0.0,
                active_issues_count=0,
                recent_optimizations=0,
                roi_last_30_days=0.0
            )

    async def _measure_user_activity(self) -> float:
        """Measure current user activity level (0.0-1.0)."""
        try:
            if not self.db_logger:
                return 0.1  # Default low activity
            
            # Get recent activity from last 10 minutes
            now = datetime.utcnow()
            start_time = now - timedelta(minutes=10)
            
            query = """
            SELECT COUNT(*) as activity_count
            FROM workflow_runs 
            WHERE created_at >= %s
            """
            
            result = await self.db_logger.execute_query(query, (start_time,))
            activity_count = result[0]["activity_count"] if result else 0
            
            # Normalize to 0-1 scale (10+ requests in 10 min = high activity)
            return min(activity_count / 10.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error measuring user activity: {e}")
            return 0.1

    async def _calculate_system_load(self) -> float:
        """Calculate current system resource utilization (0.0-1.0)."""
        # Factor in running improvement tasks
        task_load = len(self.running_tasks) / self.max_concurrent_tasks
        
        # Factor in recent user activity
        user_load = await self._measure_user_activity()
        
        # Combine loads (improvement tasks + user activity)
        total_load = min(task_load * 0.6 + user_load * 0.4, 1.0)
        
        return total_load

    async def _gather_system_metrics(self) -> Dict[str, Any]:
        """Gather comprehensive system metrics for analysis."""
        try:
            metrics = {}
            
            # Performance metrics
            metrics["performance"] = await self._get_performance_metrics()
            
            # Cost metrics
            metrics["cost"] = await self._get_cost_metrics()
            
            # Agent metrics
            metrics["agents"] = await self._get_agent_metrics()
            
            # Improvement metrics
            metrics["improvements"] = {
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "success_rate": self._calculate_improvement_success_rate(),
                "avg_roi": self._calculate_average_roi()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error gathering system metrics: {e}")
            return {}

    def _get_recent_improvement_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent improvement history for analysis."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        recent_results = [
            {
                "task_id": result.task_id,
                "success": result.success,
                "roi": result.roi,
                "duration": result.duration.total_seconds(),
                "benefit": result.actual_benefit
            }
            for result in self.completed_tasks
            if result.created_at > cutoff
        ]
        
        return recent_results

    async def _identify_current_issues(self) -> List[Dict[str, Any]]:
        """Identify current system issues and opportunities."""
        issues = []
        
        try:
            # Check system health
            health = await self._assess_system_health()
            
            if health.overall_score < 0.7:
                issues.append({
                    "type": "system_health",
                    "severity": "high",
                    "description": f"Overall system health is low: {health.overall_score:.2f}"
                })
            
            if health.active_issues_count > 5:
                issues.append({
                    "type": "active_issues",
                    "severity": "medium",
                    "description": f"High number of active issues: {health.active_issues_count}"
                })
            
            # Check improvement ROI
            recent_roi = await self._calculate_recent_roi(days=7)
            if recent_roi < 0.1:  # Less than 10% ROI
                issues.append({
                    "type": "low_roi",
                    "severity": "medium",
                    "description": f"Low improvement ROI: {recent_roi:.2f}"
                })
            
            # Check for failed tasks
            recent_failures = [
                result for result in self.completed_tasks
                if not result.success and result.created_at > datetime.utcnow() - timedelta(hours=24)
            ]
            
            if len(recent_failures) > 3:
                issues.append({
                    "type": "task_failures",
                    "severity": "high",
                    "description": f"High number of failed improvement tasks: {len(recent_failures)}"
                })
            
        except Exception as e:
            logger.error(f"Error identifying current issues: {e}")
        
        return issues

    # Helper methods for metrics and calculations
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            if not self.db_logger:
                return {"avg_success_rate": 0.8, "avg_response_time": 2.0}
            
            query = """
            SELECT 
                AVG(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_rate,
                AVG(EXTRACT(EPOCH FROM (end_time - start_time))) as avg_duration
            FROM workflow_runs 
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            """
            
            result = await self.db_logger.execute_query(query)
            if result:
                return {
                    "avg_success_rate": float(result[0]["success_rate"] or 0.8),
                    "avg_response_time": float(result[0]["avg_duration"] or 2.0)
                }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
        
        return {"avg_success_rate": 0.8, "avg_response_time": 2.0}

    async def _get_cost_metrics(self) -> Dict[str, Any]:
        """Get cost efficiency metrics."""
        try:
            if not self.db_logger:
                return {"efficiency_score": 0.7, "daily_cost": 5.0}
            
            query = """
            SELECT 
                AVG(efficiency_score) as avg_efficiency,
                SUM(total_cost) as total_cost
            FROM agent_cost_metrics 
            WHERE date >= CURRENT_DATE - INTERVAL '7 days'
            """
            
            result = await self.db_logger.execute_query(query)
            if result:
                return {
                    "efficiency_score": float(result[0]["avg_efficiency"] or 0.7),
                    "weekly_cost": float(result[0]["total_cost"] or 35.0)
                }
            
        except Exception as e:
            logger.error(f"Error getting cost metrics: {e}")
        
        return {"efficiency_score": 0.7, "weekly_cost": 35.0}

    async def _get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            "total_agents": len(self.improvement_agents),
            "active_agents": len([a for a in self.improvement_agents.values() if hasattr(a, 'last_activity')]),
            "avg_effectiveness": statistics.mean(self.agent_effectiveness.values()) if self.agent_effectiveness else 0.7
        }

    async def _get_user_satisfaction(self) -> Dict[str, Any]:
        """Get user satisfaction metrics."""
        try:
            # This would integrate with user feedback systems
            return {"satisfaction_score": 0.75, "feedback_count": 10}
        except Exception as e:
            logger.error(f"Error getting user satisfaction: {e}")
            return {"satisfaction_score": 0.75, "feedback_count": 0}

    async def _count_active_issues(self) -> int:
        """Count active system issues."""
        try:
            if not self.db_logger:
                return 2
            
            query = """
            SELECT COUNT(*) as issue_count
            FROM cost_issues 
            WHERE status IN ('active', 'identified') 
            AND severity IN ('high', 'critical')
            """
            
            result = await self.db_logger.execute_query(query)
            return result[0]["issue_count"] if result else 2
            
        except Exception as e:
            logger.error(f"Error counting active issues: {e}")
            return 2

    async def _get_recent_performance_metrics(self) -> Dict[str, Any]:
        """Get recent performance trends."""
        return {
            "success_rate_trend": "stable",
            "response_time_trend": "improving",
            "cost_trend": "decreasing",
            "user_satisfaction_trend": "stable"
        }

    async def _capture_system_metrics(self) -> Dict[str, Any]:
        """Capture current system metrics for before/after comparison."""
        return await self._gather_system_metrics()

    async def _calculate_actual_benefit(self, metrics_before: Dict[str, Any], 
                                       metrics_after: Dict[str, Any],
                                       expected_benefit: Dict[str, float]) -> Dict[str, float]:
        """Calculate actual benefit achieved by an improvement."""
        actual_benefit = {}
        
        try:
            # Calculate performance improvements
            perf_before = metrics_before.get("performance", {})
            perf_after = metrics_after.get("performance", {})
            
            if "avg_success_rate" in perf_before and "avg_success_rate" in perf_after:
                improvement = perf_after["avg_success_rate"] - perf_before["avg_success_rate"]
                actual_benefit["performance_gain"] = improvement * 100
            
            # Calculate cost savings
            cost_before = metrics_before.get("cost", {})
            cost_after = metrics_after.get("cost", {})
            
            if "weekly_cost" in cost_before and "weekly_cost" in cost_after:
                savings = cost_before["weekly_cost"] - cost_after["weekly_cost"]
                actual_benefit["cost_savings"] = savings
            
        except Exception as e:
            logger.error(f"Error calculating actual benefit: {e}")
        
        return actual_benefit

    async def _calculate_roi(self, task: ImprovementTask, actual_benefit: Dict[str, float], 
                            duration: timedelta) -> float:
        """Calculate return on investment for an improvement."""
        try:
            # Estimate cost of improvement (time * resource cost)
            improvement_cost = duration.total_seconds() / 3600 * 0.1  # $0.10 per hour of improvement
            
            # Calculate monetary benefit
            monetary_benefit = 0.0
            monetary_benefit += actual_benefit.get("cost_savings", 0.0)
            monetary_benefit += actual_benefit.get("performance_gain", 0.0) * 0.5  # $0.50 per % improvement
            
            # Calculate ROI
            if improvement_cost > 0:
                roi = (monetary_benefit - improvement_cost) / improvement_cost
            else:
                roi = 0.0
            
            return roi
            
        except Exception as e:
            logger.error(f"Error calculating ROI: {e}")
            return 0.0

    async def _calculate_recent_roi(self, days: int = 30) -> float:
        """Calculate average ROI for recent improvements."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        recent_rois = [
            result.roi for result in self.completed_tasks
            if result.created_at > cutoff and result.success and result.roi > 0
        ]
        
        return statistics.mean(recent_rois) if recent_rois else 0.0

    def _calculate_improvement_success_rate(self) -> float:
        """Calculate success rate of recent improvements."""
        if not self.completed_tasks:
            return 0.0
        
        successful = sum(1 for result in self.completed_tasks if result.success)
        return successful / len(self.completed_tasks)

    def _calculate_average_roi(self) -> float:
        """Calculate average ROI of all improvements."""
        if not self.completed_tasks:
            return 0.0
        
        rois = [result.roi for result in self.completed_tasks if result.success]
        return statistics.mean(rois) if rois else 0.0

    async def _track_improvement_success(self, task: ImprovementTask, result: ImprovementResult):
        """Track successful improvement for learning."""
        agent_type = task.agent_type
        
        # Update agent effectiveness
        if agent_type not in self.agent_effectiveness:
            self.agent_effectiveness[agent_type] = 0.7
        
        # Exponential moving average
        self.agent_effectiveness[agent_type] = (
            0.8 * self.agent_effectiveness[agent_type] + 0.2 * result.roi
        )
        
        # Track ROI for this improvement type
        improvement_type = f"{agent_type}.{task.method_name}"
        self.improvement_roi_tracking[improvement_type].append(result.roi)
        
        # Keep only last 50 results
        if len(self.improvement_roi_tracking[improvement_type]) > 50:
            self.improvement_roi_tracking[improvement_type] = \
                self.improvement_roi_tracking[improvement_type][-50:]

    async def _track_improvement_failure(self, task: ImprovementTask, result: ImprovementResult):
        """Track failed improvement for learning."""
        agent_type = task.agent_type
        
        # Decrease agent effectiveness
        if agent_type in self.agent_effectiveness:
            self.agent_effectiveness[agent_type] = max(
                0.1, self.agent_effectiveness[agent_type] * 0.9
            )
        
        logger.warning(f"Improvement failure tracked for {agent_type}: {result.error_message}")

    async def _update_improvement_metrics(self):
        """Update overall improvement metrics and ROI tracking."""
        try:
            # Calculate system-wide metrics
            self.improvement_metrics = {
                "total_improvements": len(self.completed_tasks),
                "success_rate": self._calculate_improvement_success_rate(),
                "average_roi": self._calculate_average_roi(),
                "active_tasks": len(self.active_tasks),
                "agent_effectiveness": dict(self.agent_effectiveness),
                "last_updated": datetime.utcnow()
            }
            
            # Store metrics in database if logger available
            if self.db_logger:
                await self._store_improvement_metrics()
                
        except Exception as e:
            logger.error(f"Error updating improvement metrics: {e}")

    async def _store_improvement_metrics(self):
        """Store improvement metrics in database."""
        try:
            query = """
            INSERT INTO improvement_metrics 
            (metrics_data, created_at) 
            VALUES (%s, %s)
            """
            
            await self.db_logger.execute_query(
                query, 
                (json.dumps(self.improvement_metrics), datetime.utcnow())
            )
            
        except Exception as e:
            logger.debug(f"Could not store improvement metrics: {e}")  # Non-critical

    # Public API methods
    
    async def get_improvement_status(self) -> Dict[str, Any]:
        """Get current improvement system status."""
        system_health = await self._assess_system_health()
        
        return {
            "system_health": asdict(system_health),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "success_rate": self._calculate_improvement_success_rate(),
            "average_roi": self._calculate_average_roi(),
            "current_load": self.current_system_load,
            "agent_effectiveness": dict(self.agent_effectiveness),
            "recent_improvements": len([
                task for task in self.completed_tasks
                if task.created_at > datetime.utcnow() - timedelta(hours=24)
            ])
        }

    async def get_improvement_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get improvement history for the specified period."""
        return self._get_recent_improvement_history(days)

    async def force_improvement_cycle(self, cycle: ImprovementCycle) -> Dict[str, Any]:
        """Force execution of a specific improvement cycle."""
        logger.info(f"Forcing {cycle.value} improvement cycle")
        
        try:
            await self._execute_cycle(cycle)
            return {
                "success": True,
                "message": f"Successfully executed {cycle.value} cycle",
                "tasks_created": len(self.task_queue[cycle])
            }
        except Exception as e:
            logger.error(f"Error forcing {cycle.value} cycle: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def pause_improvements(self, duration_minutes: int = 60):
        """Temporarily pause improvement activities."""
        logger.info(f"Pausing improvements for {duration_minutes} minutes")
        
        # Set high resource threshold to effectively pause
        self.original_threshold = self.resource_usage_threshold
        self.resource_usage_threshold = 0.01  # Very low threshold
        
        # Schedule resume
        asyncio.create_task(self._resume_improvements_after(duration_minutes))

    async def _resume_improvements_after(self, minutes: int):
        """Resume improvements after specified duration."""
        await asyncio.sleep(minutes * 60)
        self.resource_usage_threshold = getattr(self, 'original_threshold', 0.3)
        logger.info("Resumed improvement activities")

    async def close(self):
        """Clean shutdown of improvement orchestrator."""
        logger.info("Shutting down Improvement Orchestrator")
        
        # Cancel orchestrator task
        if self.orchestrator_task and not self.orchestrator_task.done():
            self.orchestrator_task.cancel()
            try:
                await self.orchestrator_task
            except asyncio.CancelledError:
                pass
        
        # Cancel running improvement tasks
        for task_id, task in self.running_tasks.items():
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        # Close improvement agents
        for agent in self.improvement_agents.values():
            if hasattr(agent, 'close'):
                await agent.close()
        
        logger.info("Improvement Orchestrator shutdown complete") 