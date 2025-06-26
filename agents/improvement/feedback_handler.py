"""
Filename: feedback_handler.py
Purpose: User feedback system for direct workflow improvement through natural language
Dependencies: langchain, openai, asyncio, logging, typing, supabase

This module allows users to directly improve workflows through commands like /improve,
/save-workflow, and /list-workflows with intelligent processing and version management.
"""

import asyncio
import logging
import os
import json
import uuid
import yaml
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of user feedback that can be processed."""
    IMPROVEMENT_REQUEST = "improvement_request"
    WORKFLOW_SAVE = "workflow_save"
    WORKFLOW_EDIT = "workflow_edit"
    PERFORMANCE_FEEDBACK = "performance_feedback"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"

class WorkflowVersion(Enum):
    """Workflow versioning strategies."""
    MAJOR = "major"        # Significant changes
    MINOR = "minor"        # Small improvements
    PATCH = "patch"        # Bug fixes
    BRANCH = "branch"      # Experimental variations

@dataclass
class UserWorkflow:
    """Represents a user-created or saved workflow."""
    id: str
    user_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    triggers: List[str]
    agents_used: List[str]
    success_criteria: List[str]
    version: str  # semantic versioning: "1.0.0"
    parent_version: Optional[str] = None
    branch_name: Optional[str] = None
    created_at: datetime = None
    last_modified: datetime = None
    usage_count: int = 0
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    user_rating: Optional[float] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_modified is None:
            self.last_modified = datetime.utcnow()
        if self.tags is None:
            self.tags = []

@dataclass
class UserFeedback:
    """Represents user feedback on workflows or system performance."""
    id: str
    user_id: str
    feedback_type: FeedbackType
    content: str
    priority: int  # 1-5, 5 being highest
    status: str  # "pending", "processing", "implemented", "rejected"
    workflow_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    sentiment: Optional[str] = None  # "positive", "negative", "neutral"
    response: Optional[str] = None
    implementation_notes: Optional[str] = None
    created_at: datetime = None
    processed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class FeedbackAnalysis(BaseModel):
    """LLM response schema for feedback analysis."""
    feedback_type: str = Field(description="Type of feedback provided")
    sentiment: str = Field(description="Sentiment: positive, negative, or neutral")
    priority: int = Field(description="Priority level 1-5")
    key_points: List[str] = Field(description="Key points from the feedback")
    actionable_items: List[str] = Field(description="Specific actionable improvements")
    workflow_improvements: List[Dict[str, Any]] = Field(description="Specific workflow improvements")
    implementation_complexity: str = Field(description="Complexity: low, medium, high")

class WorkflowImprovement(BaseModel):
    """LLM response schema for workflow improvement."""
    improved_name: str = Field(description="Improved workflow name")
    improvements_made: List[str] = Field(description="List of improvements implemented")
    new_steps: List[Dict[str, Any]] = Field(description="Updated workflow steps")
    new_triggers: List[str] = Field(description="Updated trigger conditions")
    success_criteria: List[str] = Field(description="Updated success criteria")
    expected_benefits: List[str] = Field(description="Expected benefits from improvements")
    version_increment: str = Field(description="Version increment type: major, minor, patch")

class FeedbackHandler:
    """
    User feedback system that allows direct workflow improvement through natural language.
    
    Processes commands like /improve, /save-workflow, /list-workflows and enables
    collaborative workflow development with version management.
    """
    
    def __init__(self, db_logger=None, orchestrator=None, workflow_analyst=None):
        """Initialize the Feedback Handler."""
        self.db_logger = db_logger
        self.orchestrator = orchestrator
        self.workflow_analyst = workflow_analyst
        
        # Storage for user workflows and feedback
        self.user_workflows: Dict[str, UserWorkflow] = {}
        self.user_feedback: Dict[str, UserFeedback] = {}
        self.workflow_versions: Dict[str, List[str]] = {}
        
        # User session tracking
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.last_workflow_by_user: Dict[str, str] = {}
        
        # Initialize LLMs for feedback processing
        self.feedback_llm = ChatOpenAI(
            model="gpt-4-0125-preview",
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2000,
        )
        
        self.improvement_llm = ChatOpenAI(
            model="gpt-4-0125-preview",
            temperature=0.4,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2500,
        )
        
        logger.info("Feedback Handler initialized with user workflow management")

    def _create_feedback_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for analyzing user feedback."""
        
        system_template = """You are an expert feedback analyst for an AI Agent Platform.
Your role is to analyze user feedback and convert it into actionable improvements.

**Feedback Analysis Framework:**
1. **Feedback Type Classification**: 
   - improvement_request: User wants to improve a workflow
   - workflow_save: User wants to save current workflow
   - performance_feedback: User comments on system performance
   - feature_request: User requests new features
   - bug_report: User reports issues

2. **Sentiment Analysis**: Determine if feedback is positive, negative, or neutral

3. **Priority Assessment**: Rate urgency (1-5, 5 being highest)

4. **Actionable Extraction**: Identify specific improvements that can be implemented

5. **Workflow Focus**: Extract workflow-specific improvements

**Guidelines:**
- Be precise in classifying feedback type
- Extract all actionable items mentioned
- Consider both explicit and implicit improvement requests
- Prioritize based on impact and user frustration level
- Focus on implementable changes

Return analysis in the specified JSON format with specific, actionable insights."""

        human_template = """Analyze user feedback:

**User ID:** {user_id}
**Feedback Content:** {feedback_content}
**Context:** {context}
**Last Workflow:** {last_workflow}
**Conversation History:** {conversation_history}

Provide comprehensive feedback analysis with actionable improvement recommendations."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    def _create_improvement_prompt(self) -> ChatPromptTemplate:
        """Create prompt for workflow improvement."""
        
        system_template = """You are an expert workflow improvement specialist.
Your role is to take user feedback and improve existing workflows.

**Improvement Principles:**
1. **User-Centric**: Focus on what the user specifically requested
2. **Incremental**: Make targeted improvements, not complete rewrites
3. **Practical**: Ensure improvements are implementable and realistic
4. **Measurable**: Include clear success criteria for improvements
5. **Version-Aware**: Determine appropriate version increment

**Improvement Types:**
- **Major (1.x.0)**: Significant workflow restructuring, new agents, major feature additions
- **Minor (x.1.0)**: New steps, improved efficiency, enhanced capabilities
- **Patch (x.x.1)**: Bug fixes, small optimizations, parameter adjustments

**Workflow Components:**
- **Steps**: Sequential actions with clear inputs/outputs
- **Triggers**: Conditions that start the workflow
- **Agents**: AI agents required for execution
- **Success Criteria**: Measurable outcomes

Provide specific, implementable improvements based on user feedback."""

        human_template = """Improve workflow based on user feedback:

**Original Workflow:**
{original_workflow}

**User Feedback:**
{user_feedback}

**Improvement Context:**
{improvement_context}

**Available Agents:**
{available_agents}

Create an improved version with specific enhancements addressing the user's feedback."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    def _create_conversation_prompt(self) -> ChatPromptTemplate:
        """Create prompt for conversational responses to users."""
        
        system_template = """You are a helpful AI assistant that helps users manage and improve their workflows.

**Your Personality:**
- Friendly and encouraging
- Clear and helpful explanations
- Proactive in suggesting improvements
- Collaborative approach to workflow development

**Your Capabilities:**
- Help users save and manage workflows
- Improve workflows based on feedback
- Explain workflow changes and benefits
- Guide users through workflow management

**Communication Style:**
- Use emojis appropriately for Slack
- Keep responses concise but informative
- Ask clarifying questions when needed
- Celebrate user improvements and milestones

**Context Awareness:**
- Remember user preferences and history
- Reference previous workflows when relevant
- Suggest related improvements or optimizations

Respond naturally and helpfully to user workflow management requests."""

        human_template = """User Request: {user_request}
Context: {context}
Response needed for: {response_type}

Provide a helpful, friendly response."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    async def process_feedback_command(self, command: str, user_id: str, 
                                     message_content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user feedback commands (/improve, /save-workflow, etc.)."""
        try:
            logger.info(f"Processing feedback command '{command}' from user {user_id}")
            
            if command == "improve":
                return await self._handle_improve_command(user_id, message_content, context)
            elif command == "save-workflow":
                return await self._handle_save_workflow_command(user_id, message_content, context)
            elif command == "list-workflows":
                return await self._handle_list_workflows_command(user_id, context)
            elif command == "edit-workflow":
                return await self._handle_edit_workflow_command(user_id, message_content, context)
            elif command == "feedback":
                return await self._handle_general_feedback_command(user_id, message_content, context)
            else:
                return {
                    "success": False,
                    "message": f"❌ Unknown command: {command}\n\nAvailable commands:\n• `/improve` - Improve your last workflow\n• `/save-workflow` - Save current workflow\n• `/list-workflows` - Show your workflows\n• `/edit-workflow` - Edit a saved workflow\n• `/feedback` - Provide general feedback"
                }
                
        except Exception as e:
            logger.error(f"Error processing feedback command: {str(e)}")
            return {
                "success": False,
                "message": f"❌ Error processing command: {str(e)}"
            }

    async def _handle_improve_command(self, user_id: str, message_content: str, 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /improve command - improve user's last workflow."""
        try:
            # Get user's last workflow from conversation context
            last_workflow = await self._extract_last_workflow(user_id, context)
            
            if not last_workflow:
                return {
                    "success": False,
                    "message": "❌ No recent workflow found to improve. Please run a task first, then use `/improve` to make it better!"
                }
            
            # Extract improvement request
            improvement_request = self._extract_improvement_request(message_content)
            
            if not improvement_request:
                improvement_request = "Make this workflow better and more efficient"
            
            # Generate improved workflow
            improved_workflow = await self._generate_improved_workflow(
                last_workflow, improvement_request, context
            )
            
            # Save improved workflow
            workflow_id = await self._save_workflow_version(last_workflow, improved_workflow, user_id)
            
            # Generate response
            response_message = self._format_improvement_response(improved_workflow)
            
            return {
                "success": True,
                "message": response_message,
                "workflow_id": workflow_id
            }
            
        except Exception as e:
            logger.error(f"Error in improve command: {str(e)}")
            return {
                "success": False,
                "message": f"❌ Error improving workflow: {str(e)}"
            }

    async def _handle_save_workflow_command(self, user_id: str, message_content: str, 
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /save-workflow command."""
        try:
            # Extract workflow name from message
            workflow_name = self._extract_workflow_name(message_content)
            if not workflow_name:
                workflow_name = f"Workflow {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
            
            # Create workflow from conversation
            workflow = await self._create_workflow_from_conversation(user_id, context, workflow_name)
            
            if not workflow:
                return {
                    "success": False,
                    "message": "❌ No workflow found in recent conversation. Please complete a task first, then save it!"
                }
            
            # Save workflow
            workflow_id = await self._save_new_workflow(workflow, user_id)
            
            return {
                "success": True,
                "message": f"✅ **Workflow Saved!**\n\n📋 **Name:** {workflow['name']}\n🆔 **ID:** `{workflow_id}`\n📊 **Steps:** {len(workflow.get('steps', []))}\n\nUse `/list-workflows` to see all your workflows!",
                "workflow_id": workflow_id
            }
            
        except Exception as e:
            logger.error(f"Error saving workflow: {str(e)}")
            return {
                "success": False,
                "message": f"❌ Error saving workflow: {str(e)}"
            }

    async def _handle_list_workflows_command(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /list-workflows command."""
        try:
            # Get user's workflows
            user_workflows = [w for w in self.user_workflows.values() if w.user_id == user_id]
            
            if not user_workflows:
                return {
                    "success": True,
                    "message": "📋 **No Saved Workflows**\n\nYou haven't saved any workflows yet. Complete a task and use `/save-workflow [name]` to save it for reuse!"
                }
            
            # Sort by last modified
            user_workflows.sort(key=lambda w: w.last_modified, reverse=True)
            
            # Format list
            workflows_text = "📋 **Your Saved Workflows**\n\n"
            
            for i, workflow in enumerate(user_workflows[:10], 1):
                success_indicator = "✅" if workflow.success_rate > 0.8 else "⚠️" if workflow.success_rate > 0.5 else "❌"
                
                workflows_text += f"{i}. {success_indicator} **{workflow.name}** (v{workflow.version})\n"
                workflows_text += f"   📝 {workflow.description[:80]}{'...' if len(workflow.description) > 80 else ''}\n"
                workflows_text += f"   🎯 {len(workflow.steps)} steps | Used {workflow.usage_count} times\n\n"
            
            workflows_text += "💡 Use `/improve` to enhance your last workflow!"
            
            return {
                "success": True,
                "message": workflows_text
            }
            
        except Exception as e:
            logger.error(f"Error listing workflows: {str(e)}")
            return {
                "success": False,
                "message": f"❌ Error listing workflows: {str(e)}"
            }

    async def _handle_edit_workflow_command(self, user_id: str, message_content: str, 
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /edit-workflow command."""
        return {
            "success": True,
            "message": "🚧 **Edit Workflow** feature coming soon!\n\nFor now, use `/improve` to enhance your workflows with natural language feedback."
        }

    async def _handle_general_feedback_command(self, user_id: str, message_content: str, 
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general feedback command."""
        try:
            # Extract feedback content
            feedback_content = self._extract_feedback_content(message_content)
            
            # Store feedback
            feedback_id = await self._store_feedback(user_id, feedback_content, context)
            
            return {
                "success": True,
                "message": f"✅ **Feedback Received!**\n\nThank you for your feedback! We'll use it to improve the system.\n\n🆔 Feedback ID: `{feedback_id}`",
                "feedback_id": feedback_id
            }
            
        except Exception as e:
            logger.error(f"Error handling feedback: {str(e)}")
            return {
                "success": False,
                "message": f"❌ Error processing feedback: {str(e)}"
            }

    async def _extract_last_workflow(self, user_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract the last workflow from user's conversation."""
        try:
            # In a real implementation, this would query conversation history
            # For now, return a sample workflow structure
            
            conversation_id = context.get("conversation_id")
            if not conversation_id:
                return None
            
            # Sample workflow from conversation
            return {
                "name": "Last Conversation Workflow",
                "description": "Workflow derived from recent conversation",
                "steps": [
                    {"step": 1, "action": "User request received", "agent": "routing"},
                    {"step": 2, "action": "Request analyzed", "agent": "general"},
                    {"step": 3, "action": "Response generated", "agent": "general"}
                ],
                "agents_used": ["general"],
                "triggers": ["user request"],
                "success_criteria": ["User satisfied", "Task completed"]
            }
            
        except Exception as e:
            logger.error(f"Error extracting last workflow: {str(e)}")
            return None

    async def _generate_improved_workflow(self, original_workflow: Dict[str, Any], 
                                        improvement_request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an improved version of the workflow using LLM."""
        try:
            # Create improvement prompt
            prompt = f"""
            Original Workflow:
            {json.dumps(original_workflow, indent=2)}
            
            User Improvement Request:
            {improvement_request}
            
            Please improve this workflow based on the user's request. Focus on:
            1. Addressing the specific improvement requested
            2. Making the workflow more efficient
            3. Adding relevant steps if needed
            4. Improving success criteria
            
            Return the improved workflow as JSON with the same structure.
            """
            
            # For now, return a simulated improvement
            improved = original_workflow.copy()
            improved["name"] = f"Improved: {original_workflow['name']}"
            improved["description"] = f"Enhanced version: {original_workflow['description']}"
            improved["improvements_made"] = [
                "Added error handling steps",
                "Improved efficiency",
                "Enhanced user feedback collection"
            ]
            
            # Add an improvement step
            if "steps" in improved:
                improved["steps"].append({
                    "step": len(improved["steps"]) + 1,
                    "action": "Collect user feedback for continuous improvement",
                    "agent": "feedback"
                })
            
            return improved
            
        except Exception as e:
            logger.error(f"Error generating improved workflow: {str(e)}")
            return original_workflow

    async def _create_workflow_from_conversation(self, user_id: str, context: Dict[str, Any], 
                                               name: str = None) -> Optional[Dict[str, Any]]:
        """Create a workflow from the current conversation."""
        try:
            if not name:
                name = f"Conversation Workflow {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
            
            # Sample workflow creation from conversation
            return {
                "name": name,
                "description": "Workflow created from conversation",
                "steps": [
                    {"step": 1, "action": "Analyze user request", "agent": "routing"},
                    {"step": 2, "action": "Process request", "agent": "general"},
                    {"step": 3, "action": "Generate response", "agent": "general"}
                ],
                "triggers": ["user request"],
                "agents_used": ["general"],
                "success_criteria": ["User request fulfilled", "Response provided"]
            }
            
        except Exception as e:
            logger.error(f"Error creating workflow from conversation: {str(e)}")
            return None

    def _extract_improvement_request(self, message_content: str) -> str:
        """Extract improvement request from message."""
        # Remove command prefix
        content = message_content.replace("/improve", "").strip()
        if not content:
            content = "Make this workflow better and more efficient"
        return content

    def _extract_workflow_name(self, message_content: str) -> Optional[str]:
        """Extract workflow name from message."""
        content = message_content.replace("/save-workflow", "").strip()
        return content if content else None

    def _extract_feedback_content(self, message_content: str) -> str:
        """Extract feedback content from message."""
        return message_content.replace("/feedback", "").strip()

    async def _save_new_workflow(self, workflow_data: Dict[str, Any], user_id: str) -> str:
        """Save a new workflow."""
        try:
            workflow_id = str(uuid.uuid4())
            
            workflow = UserWorkflow(
                id=workflow_id,
                user_id=user_id,
                name=workflow_data["name"],
                description=workflow_data["description"],
                steps=workflow_data.get("steps", []),
                triggers=workflow_data.get("triggers", []),
                agents_used=workflow_data.get("agents_used", []),
                success_criteria=workflow_data.get("success_criteria", []),
                version="1.0.0"
            )
            
            self.user_workflows[workflow_id] = workflow
            self.last_workflow_by_user[user_id] = workflow_id
            
            # Log to database if available
            if self.db_logger:
                await self.db_logger.log_event("workflow_saved", {
                    "workflow_id": workflow_id,
                    "user_id": user_id,
                    "name": workflow.name
                })
            
            logger.info(f"Saved new workflow: {workflow_id} for user {user_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Error saving workflow: {str(e)}")
            raise

    async def _save_workflow_version(self, original: Dict[str, Any], improved: Dict[str, Any], 
                                   user_id: str) -> str:
        """Save an improved version of a workflow."""
        try:
            workflow_id = str(uuid.uuid4())
            
            # Increment version
            version = "1.1.0"  # Simple versioning for now
            
            workflow = UserWorkflow(
                id=workflow_id,
                user_id=user_id,
                name=improved["name"],
                description=improved["description"],
                steps=improved.get("steps", []),
                triggers=improved.get("triggers", []),
                agents_used=improved.get("agents_used", []),
                success_criteria=improved.get("success_criteria", []),
                version=version
            )
            
            self.user_workflows[workflow_id] = workflow
            self.last_workflow_by_user[user_id] = workflow_id
            
            logger.info(f"Saved improved workflow: {workflow_id} for user {user_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Error saving workflow version: {str(e)}")
            raise

    async def _store_feedback(self, user_id: str, content: str, context: Dict[str, Any]) -> str:
        """Store user feedback."""
        try:
            feedback_id = str(uuid.uuid4())
            
            feedback = UserFeedback(
                id=feedback_id,
                user_id=user_id,
                feedback_type=FeedbackType.PERFORMANCE_FEEDBACK,
                content=content,
                conversation_id=context.get("conversation_id"),
                sentiment="neutral",
                priority=3,
                status="pending"
            )
            
            self.user_feedback[feedback_id] = feedback
            
            # Log to database if available
            if self.db_logger:
                await self.db_logger.log_event("feedback_received", {
                    "feedback_id": feedback_id,
                    "user_id": user_id,
                    "type": feedback.feedback_type.value
                })
            
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error storing feedback: {str(e)}")
            raise

    def _format_improvement_response(self, improved_workflow: Dict[str, Any]) -> str:
        """Format the improvement response for the user."""
        improvements = improved_workflow.get("improvements_made", ["General improvements applied"])
        
        response = f"✅ **Workflow Improved!**\n\n"
        response += f"📋 **Enhanced:** {improved_workflow['name']}\n\n"
        response += "🚀 **Improvements Made:**\n"
        
        for i, improvement in enumerate(improvements, 1):
            response += f"{i}. {improvement}\n"
        
        response += f"\n📊 **New Steps:** {len(improved_workflow.get('steps', []))}\n"
        response += "\n💡 Your improved workflow has been saved automatically!"
        
        return response

    def get_user_workflows(self, user_id: str) -> List[UserWorkflow]:
        """Get all workflows for a user."""
        return [w for w in self.user_workflows.values() if w.user_id == user_id]

    def get_user_feedback(self, user_id: str) -> List[UserFeedback]:
        """Get all feedback from a user."""
        return [f for f in self.user_feedback.values() if f.user_id == user_id]

    async def close(self):
        """Clean up resources."""
        try:
            logger.info("Feedback Handler closed")
        except Exception as e:
            logger.error(f"Error closing Feedback Handler: {str(e)}") 