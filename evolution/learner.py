"""
Filename: learner.py
Purpose: Simplified learning system combining workflow analysis and user feedback
Dependencies: langchain, openai, asyncio, logging, typing, supabase

This module learns from every interaction and user feedback to continuously improve.
"""

import asyncio
import logging
import os
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

@dataclass
class WorkflowPattern:
    """Represents a discovered workflow pattern."""
    id: str
    name: str
    trigger_keywords: List[str]
    typical_steps: List[str]
    frequency: int
    success_rate: float
    avg_duration: float
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class UserWorkflow:
    """Represents a user-saved workflow."""
    id: str
    user_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    version: str = "1.0.0"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class Learner:
    """
    Simplified learning system that analyzes workflows and processes user feedback.
    Combines the best of workflow_analyst and feedback_handler without over-engineering.
    """
    
    def __init__(self, storage=None, event_bus=None):
        """Initialize the Learner with storage and event capabilities."""
        self.storage = storage
        self.event_bus = event_bus
        
        # Pattern and workflow storage
        self.discovered_patterns: Dict[str, WorkflowPattern] = {}
        self.user_workflows: Dict[str, UserWorkflow] = {}
        
        # Initialize LLM for analysis (single LLM, not multiple)
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",  # Faster, cheaper model
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=1000,
        )
        
        # Simple pattern tracking
        self.interaction_patterns = defaultdict(int)
        self.agent_usage = defaultdict(int)
        
        logger.info("Learner initialized with simplified pattern recognition")
    
    async def analyze_interaction(self, run_id: str) -> Dict[str, Any]:
        """
        Analyze a completed interaction for patterns and improvements.
        Non-blocking, runs in background.
        """
        try:
            # Get interaction data from storage
            if not self.storage:
                return {"status": "no_storage"}
            
            interaction_data = await self._get_interaction_data(run_id)
            if not interaction_data:
                return {"status": "no_data"}
            
            # Extract patterns (simple approach)
            patterns = await self._extract_patterns(interaction_data)
            
            # Store patterns if significant
            for pattern in patterns:
                if pattern["frequency"] > 3:  # Simple threshold
                    await self._store_pattern(pattern)
            
            # Emit learning event
            if self.event_bus and patterns:
                await self.event_bus.publish(
                    "pattern_discovered",
                    {"run_id": run_id, "patterns": len(patterns)},
                    source="learner"
                )
            
            return {
                "status": "analyzed",
                "patterns_found": len(patterns),
                "run_id": run_id
            }
            
        except Exception as e:
            logger.error(f"Error analyzing interaction {run_id}: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def process_user_feedback(self, command: str, user_id: str, 
                                   message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user feedback commands (/improve, /save-workflow, etc.)."""
        try:
            if command == "improve":
                return await self._handle_improve(user_id, message, context)
            elif command == "save-workflow":
                return await self._handle_save_workflow(user_id, message, context)
            elif command == "list-workflows":
                return await self._handle_list_workflows(user_id)
            else:
                return {
                    "success": False,
                    "message": f"Unknown command: {command}"
                }
                
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    async def _handle_improve(self, user_id: str, message: str, 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /improve command - improve last workflow."""
        try:
            # Get last workflow from context
            last_workflow = context.get("last_workflow")
            if not last_workflow:
                return {
                    "success": False,
                    "message": "No recent workflow found to improve."
                }
            
            # Use LLM to generate improvements
            prompt = f"""
            Improve this workflow based on user feedback:
            
            Workflow: {json.dumps(last_workflow, indent=2)}
            User Feedback: {message}
            
            Suggest specific improvements in JSON format with:
            - improved_steps: List of improved steps
            - benefits: List of benefits
            - version: New version number
            """
            
            response = await self.llm.ainvoke(prompt)
            improvements = json.loads(response.content)
            
            # Save improved workflow
            workflow_id = str(uuid.uuid4())
            improved_workflow = UserWorkflow(
                id=workflow_id,
                user_id=user_id,
                name=f"Improved: {last_workflow.get('name', 'Workflow')}",
                description=f"Improved based on: {message[:100]}",
                steps=improvements["improved_steps"],
                version=improvements["version"]
            )
            
            self.user_workflows[workflow_id] = improved_workflow
            
            # Store in database
            if self.storage:
                await self.storage.log_workflow_improvement(
                    workflow_id=workflow_id,
                    user_id=user_id,
                    improvements=improvements
                )
            
            return {
                "success": True,
                "message": f"✅ Workflow improved!\n\n**Benefits:**\n" + 
                          "\n".join(f"• {b}" for b in improvements["benefits"][:3]),
                "workflow_id": workflow_id
            }
            
        except Exception as e:
            logger.error(f"Error improving workflow: {str(e)}")
            return {
                "success": False,
                "message": f"Error improving workflow: {str(e)}"
            }
    
    async def _handle_save_workflow(self, user_id: str, message: str, 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /save-workflow command."""
        try:
            # Extract workflow name
            name = message.strip() or f"Workflow {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
            
            # Get current workflow from context
            current_workflow = context.get("current_workflow", {})
            
            # Create workflow
            workflow_id = str(uuid.uuid4())
            workflow = UserWorkflow(
                id=workflow_id,
                user_id=user_id,
                name=name,
                description=f"Saved by user",
                steps=current_workflow.get("steps", [])
            )
            
            self.user_workflows[workflow_id] = workflow
            
            # Store in database
            if self.storage:
                await self.storage.log_workflow_saved(
                    workflow_id=workflow_id,
                    user_id=user_id,
                    workflow_data=asdict(workflow)
                )
            
            return {
                "success": True,
                "message": f"✅ Workflow saved as '{name}'",
                "workflow_id": workflow_id
            }
            
        except Exception as e:
            logger.error(f"Error saving workflow: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    async def _handle_list_workflows(self, user_id: str) -> Dict[str, Any]:
        """Handle /list-workflows command."""
        try:
            user_workflows = [w for w in self.user_workflows.values() if w.user_id == user_id]
            
            if not user_workflows:
                return {
                    "success": True,
                    "message": "No saved workflows yet. Use /save-workflow to save one!"
                }
            
            # Format workflow list
            workflows_text = "📋 **Your Workflows:**\n\n"
            for i, workflow in enumerate(user_workflows[:10], 1):
                workflows_text += f"{i}. **{workflow.name}** (v{workflow.version})\n"
            
            return {
                "success": True,
                "message": workflows_text
            }
            
        except Exception as e:
            logger.error(f"Error listing workflows: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    async def _get_interaction_data(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get interaction data from storage."""
        try:
            if not self.storage:
                return None
            
            # Query workflow run data
            result = await self.storage.get_workflow_run(run_id)
            return result
            
        except Exception as e:
            logger.error(f"Error getting interaction data: {str(e)}")
            return None
    
    async def _extract_patterns(self, interaction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract patterns from interaction data (simplified)."""
        patterns = []
        
        try:
            # Simple pattern extraction
            message = interaction_data.get("initial_message", "")
            agents_used = interaction_data.get("agents_used", [])
            duration = interaction_data.get("duration", 0)
            
            # Track message patterns
            keywords = message.lower().split()[:5]  # First 5 words
            pattern_key = " ".join(keywords)
            
            self.interaction_patterns[pattern_key] += 1
            
            # Track agent usage
            for agent in agents_used:
                self.agent_usage[agent] += 1
            
            # Create pattern if frequent
            if self.interaction_patterns[pattern_key] > 3:
                patterns.append({
                    "name": f"Pattern: {pattern_key}",
                    "trigger_keywords": keywords,
                    "typical_steps": agents_used,
                    "frequency": self.interaction_patterns[pattern_key],
                    "avg_duration": duration
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error extracting patterns: {str(e)}")
            return []
    
    async def _store_pattern(self, pattern: Dict[str, Any]) -> None:
        """Store a discovered pattern."""
        try:
            pattern_id = str(uuid.uuid4())
            workflow_pattern = WorkflowPattern(
                id=pattern_id,
                name=pattern["name"],
                trigger_keywords=pattern["trigger_keywords"],
                typical_steps=pattern["typical_steps"],
                frequency=pattern["frequency"],
                success_rate=0.8,  # Default for now
                avg_duration=pattern.get("avg_duration", 0)
            )
            
            self.discovered_patterns[pattern_id] = workflow_pattern
            
            # Store in database
            if self.storage:
                await self.storage.log_pattern_discovered(
                    pattern_id=pattern_id,
                    pattern_data=asdict(workflow_pattern)
                )
            
        except Exception as e:
            logger.error(f"Error storing pattern: {str(e)}")
    
    def get_top_patterns(self, limit: int = 10) -> List[WorkflowPattern]:
        """Get most frequent patterns."""
        patterns = list(self.discovered_patterns.values())
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns[:limit]
    
    def get_agent_usage_stats(self) -> Dict[str, int]:
        """Get agent usage statistics."""
        return dict(self.agent_usage)
    
    async def close(self):
        """Cleanup resources."""
        logger.info("Learner shutting down") 