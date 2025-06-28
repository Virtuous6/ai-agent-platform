"""
Filename: workflow.py
Purpose: Simple workflow engine for multi-agent task execution
Dependencies: asyncio, logging, typing

Enables complex multi-agent workflows with simple configuration.
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class WorkflowStep:
    """Represents a single step in a workflow."""
    def __init__(self, step_config: Dict[str, Any]):
        self.id = step_config.get("id", str(uuid.uuid4()))
        self.action = step_config.get("action", "process")
        self.agent_type = step_config.get("agent_type", "general")
        self.prompt = step_config.get("prompt", "")
        self.conditions = step_config.get("conditions", {})
        self.output_key = step_config.get("output_key", "result")
        self.parallel = step_config.get("parallel", False)
        self.retry_count = step_config.get("retry_count", 1)

class WorkflowEngine:
    """
    Simple workflow engine for executing multi-agent workflows.
    Supports parallel execution, conditions, and error handling.
    """
    
    def __init__(self, orchestrator=None):
        """Initialize workflow engine with orchestrator."""
        self.orchestrator = orchestrator
        self.active_workflows = {}
        self.workflow_definitions = {}
        
        logger.info("Workflow Engine initialized")
    
    async def execute_workflow(self, workflow_name: str, 
                             input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a named workflow with given input data."""
        workflow_id = str(uuid.uuid4())
        context = {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "data": input_data,
            "start_time": datetime.utcnow()
        }
        
        try:
            # Load workflow definition
            workflow = self._load_workflow(workflow_name)
            if not workflow:
                return {"error": f"Workflow '{workflow_name}' not found"}
            
            # Track active workflow
            self.active_workflows[workflow_id] = {
                "name": workflow_name,
                "status": "running",
                "start_time": context["start_time"]
            }
            
            # Execute workflow steps
            result = await self._execute_steps(workflow["steps"], context)
            
            # Mark workflow complete
            self.active_workflows[workflow_id]["status"] = "completed"
            self.active_workflows[workflow_id]["end_time"] = datetime.utcnow()
            
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "data": context["data"],
                "duration": (datetime.utcnow() - context["start_time"]).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            self.active_workflows[workflow_id]["status"] = "failed"
            self.active_workflows[workflow_id]["error"] = str(e)
            
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def _execute_steps(self, steps: List[Dict[str, Any]], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps with condition checking."""
        # Group steps for potential parallel execution
        step_groups = self._group_steps(steps)
        
        for group in step_groups:
            if len(group) == 1:
                # Single step - execute normally
                await self._execute_step(group[0], context)
            else:
                # Multiple steps - execute in parallel
                tasks = [self._execute_step(step, context) for step in group]
                await asyncio.gather(*tasks)
            
            # Check if we should continue
            if not self._should_continue(context):
                break
        
        return context["data"]
    
    async def _execute_step(self, step_config: Dict[str, Any], 
                          context: Dict[str, Any]) -> None:
        """Execute a single workflow step."""
        step = WorkflowStep(step_config)
        
        # Check step conditions
        if not self._check_conditions(step.conditions, context):
            logger.info(f"Skipping step {step.id} - conditions not met")
            return
        
        # Retry logic
        for attempt in range(step.retry_count):
            try:
                # Get or spawn agent
                if self.orchestrator:
                    agent = await self.orchestrator._get_or_spawn_agent(
                        step.agent_type, 
                        context
                    )
                    
                    # Prepare prompt with context
                    prompt = self._prepare_prompt(step.prompt, context)
                    
                    # Execute step
                    result = await agent.process(prompt, context)
                    
                    # Store result
                    context["data"][step.output_key] = result
                    
                    logger.info(f"Step {step.id} completed successfully")
                    return
                
            except Exception as e:
                logger.error(f"Step {step.id} attempt {attempt + 1} failed: {str(e)}")
                if attempt == step.retry_count - 1:
                    raise
                await asyncio.sleep(1)  # Brief delay before retry
    
    def _load_workflow(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """Load workflow definition."""
        # Check cache
        if workflow_name in self.workflow_definitions:
            return self.workflow_definitions[workflow_name]
        
        # Built-in workflows
        built_in = {
            "research_and_summarize": {
                "name": "Research and Summarize",
                "description": "Research a topic and provide a summary",
                "steps": [
                    {
                        "id": "research",
                        "agent_type": "research",
                        "prompt": "Research: {query}",
                        "output_key": "research_results"
                    },
                    {
                        "id": "summarize",
                        "agent_type": "general",
                        "prompt": "Summarize these research results: {research_results}",
                        "output_key": "summary"
                    }
                ]
            },
            "code_review": {
                "name": "Code Review",
                "description": "Review code and suggest improvements",
                "steps": [
                    {
                        "id": "analyze",
                        "agent_type": "technical",
                        "prompt": "Analyze this code: {code}",
                        "output_key": "analysis"
                    },
                    {
                        "id": "suggest",
                        "agent_type": "technical",
                        "prompt": "Based on analysis: {analysis}, suggest improvements",
                        "output_key": "suggestions"
                    }
                ]
            }
        }
        
        workflow = built_in.get(workflow_name)
        if workflow:
            self.workflow_definitions[workflow_name] = workflow
        
        return workflow
    
    def _group_steps(self, steps: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group steps for parallel execution where possible."""
        groups = []
        current_group = []
        
        for step in steps:
            if step.get("parallel", False) and current_group:
                # Add to current parallel group
                current_group.append(step)
            else:
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [step]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _check_conditions(self, conditions: Dict[str, Any], 
                         context: Dict[str, Any]) -> bool:
        """Check if step conditions are met."""
        if not conditions:
            return True
        
        # Simple condition checking
        for key, expected_value in conditions.items():
            actual_value = context["data"].get(key)
            if actual_value != expected_value:
                return False
        
        return True
    
    def _should_continue(self, context: Dict[str, Any]) -> bool:
        """Check if workflow should continue."""
        # Add any global stopping conditions here
        return True
    
    def _prepare_prompt(self, prompt_template: str, 
                       context: Dict[str, Any]) -> str:
        """Prepare prompt by replacing variables with context data."""
        prompt = prompt_template
        
        # Replace variables in format {variable_name}
        for key, value in context["data"].items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        
        return prompt
    
    def register_workflow(self, name: str, definition: Dict[str, Any]):
        """Register a new workflow definition."""
        self.workflow_definitions[name] = definition
        logger.info(f"Registered workflow: {name}")
    
    def list_workflows(self) -> List[str]:
        """List available workflows."""
        # Load built-in workflows first
        self._load_workflow("research_and_summarize")
        self._load_workflow("code_review")
        
        return list(self.workflow_definitions.keys())
    
    def get_active_workflows(self) -> Dict[str, Any]:
        """Get currently active workflows."""
        return self.active_workflows
    
    async def create_workflow_from_pattern(self, pattern: Dict[str, Any]) -> str:
        """Create a workflow from a discovered pattern."""
        workflow_name = f"pattern_{pattern.get('id', uuid.uuid4())}"
        
        # Convert pattern to workflow definition
        workflow_def = {
            "name": pattern.get("name", "Pattern Workflow"),
            "description": f"Auto-generated from pattern",
            "steps": []
        }
        
        # Create steps from pattern
        for i, step_type in enumerate(pattern.get("typical_steps", [])):
            workflow_def["steps"].append({
                "id": f"step_{i}",
                "agent_type": step_type,
                "prompt": "{input}",
                "output_key": f"result_{i}"
            })
        
        # Register workflow
        self.register_workflow(workflow_name, workflow_def)
        
        return workflow_name 