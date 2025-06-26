"""
Filename: universal_agent.py
Purpose: Configuration-driven universal agent for dynamic specialist creation
Dependencies: langchain, openai, asyncio, logging, typing

This module is part of the AI Agent Platform self-improvement system.
Creates specialist agents dynamically based on configuration without hard-coded classes.
"""

import asyncio
import logging
import os
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class SpecialtyType(Enum):
    """Types of specialist configurations supported."""
    TECHNICAL = "technical"
    RESEARCH = "research"
    BUSINESS = "business"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    CUSTOM = "custom"

class ToolCapability(BaseModel):
    """Represents a tool capability for the universal agent."""
    name: str
    description: str
    function: Optional[Any] = None  # Callable function
    enabled: bool = True

class UniversalAgent:
    """
    Configuration-driven universal agent that can be specialized for any domain.
    
    Uses the same LLM patterns as existing agents but allows dynamic configuration
    of specialty, prompts, temperature, and tools without creating new classes.
    """
    
    def __init__(self, 
                 specialty: str,
                 system_prompt: str,
                 temperature: float = 0.4,
                 tools: Optional[List[ToolCapability]] = None,
                 model_name: str = "gpt-3.5-turbo-0125",
                 max_tokens: int = 500,
                 agent_id: Optional[str] = None):
        """
        Initialize the Universal Agent with specific configuration.
        
        Args:
            specialty: The specialty area (e.g., "Python Optimization", "Data Analysis")
            system_prompt: Custom system prompt for this specialist
            temperature: LLM temperature for response creativity
            tools: List of tool capabilities available to this agent
            model_name: OpenAI model to use
            max_tokens: Maximum tokens for responses
            agent_id: Unique identifier for this agent instance
        """
        
        # Configuration
        self.specialty = specialty
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.agent_id = agent_id or f"universal_{specialty.lower().replace(' ', '_')}"
        self.tools = tools or []
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=max_tokens,
        )
        
        # Initialize analysis LLM (more focused, lower temperature for meta-analysis)
        self.analysis_llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=200,
        )
        
        # Initialize prompt templates
        self.main_prompt = self._create_main_prompt()
        self.improvement_prompt = self._create_improvement_prompt()
        
        # Conversation and performance tracking
        self.conversation_history = []
        self.performance_metrics = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
            "user_feedback_score": 0.0,
            "improvement_suggestions": []
        }
        
        logger.info(f"Universal Agent '{self.specialty}' initialized with model: {model_name}")
    
    def _create_main_prompt(self) -> ChatPromptTemplate:
        """Create the main conversation prompt template based on specialty."""
        
        # Create dynamic system template based on configuration
        tools_description = ""
        if self.tools:
            tool_list = [f"- {tool.name}: {tool.description}" for tool in self.tools if tool.enabled]
            tools_description = f"\n\n**Available Tools:**\n" + "\n".join(tool_list)
        
        system_template = f"""{self.system_prompt}

**Your Specialty:** {self.specialty}

**Your Role in the AI Agent Platform:**
- You are a specialized expert integrated into Slack
- Provide expert-level assistance in your specialty area
- Maintain professional yet approachable communication
- Share best practices and industry standards
- Suggest optimizations and improvements
- Escalate to other specialists when appropriate

**Communication Guidelines:**
1. Be precise and technically accurate in your specialty
2. Explain complex concepts clearly for the user's level
3. Provide actionable recommendations
4. Use examples and practical applications when helpful
5. Be proactive in suggesting improvements or alternatives
6. Keep responses concise but comprehensive for Slack format
7. Use appropriate emojis to enhance readability (not excessive){tools_description}

**Quality Standards:**
- Always prioritize accuracy and helpfulness
- If uncertain about something outside your expertise, be honest
- Suggest collaboration with other agents when beneficial
- Focus on providing value that leverages your specialty

Current conversation context: {{context}}
Recent conversation history: {{history}}"""

        human_template = f"""User message: {{message}}

As the {self.specialty} specialist, provide expert assistance. Focus on delivering value through your specialized knowledge while maintaining clear communication."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_improvement_prompt(self) -> ChatPromptTemplate:
        """Create prompt for self-improvement analysis."""
        
        system_template = f"""You are a self-improvement analyzer for the {self.specialty} specialist agent. 

Your task is to analyze the agent's performance and suggest improvements.

**Analysis Framework:**
1. **Response Quality**: Was the specialist knowledge effectively applied?
2. **User Satisfaction**: Did the response meet the user's needs?
3. **Efficiency**: Could the response be more concise or clear?
4. **Completeness**: Was anything important missed?
5. **Proactivity**: Were valuable suggestions or improvements offered?

**Improvement Areas:**
- Prompt optimization for better responses
- Knowledge gaps to address
- Communication style adjustments
- Tool utilization improvements
- Collaboration opportunities

Return analysis in JSON format:
{{
    "quality_score": float (0.0-1.0),
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "knowledge_gaps": ["gap1", "gap2"],
    "strengths": ["strength1", "strength2"],
    "next_optimization": "specific improvement to implement"
}}"""

        human_template = f"""Analyze this interaction:

**User Request:** {{user_message}}
**Agent Response:** {{agent_response}}
**User Feedback:** {{feedback}}
**Performance Metrics:** {{metrics}}

Provide improvement analysis for the {self.specialty} specialist:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user message using specialized LLM intelligence.
        
        Args:
            message: User message content
            context: Conversation context
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Universal Agent '{self.specialty}' processing: '{message[:50]}...'")
            
            # Prepare conversation history context
            history_context = self._format_conversation_history(context.get("conversation_history", []))
            
            # Execute tools if available
            tool_results = await self._execute_tools(message, context)
            
            # Generate specialist response
            try:
                with get_openai_callback() as cb:
                    response = await self._generate_specialist_response(
                        message, context, history_context, tool_results
                    )
                tokens_used = cb.total_tokens
                input_tokens = cb.prompt_tokens
                output_tokens = cb.completion_tokens
                cost = cb.total_cost
            except Exception as e:
                logger.warning(f"OpenAI callback failed, proceeding without tracking: {e}")
                response = await self._generate_specialist_response(
                    message, context, history_context, tool_results
                )
                tokens_used = 0
                input_tokens = 0
                output_tokens = 0
                cost = 0.0
            
            # Calculate processing time
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update performance metrics
            self._update_performance_metrics(tokens_used, cost, processing_time_ms, True)
            
            # Log the interaction
            interaction_log = {
                "timestamp": start_time.isoformat(),
                "message_preview": message[:100],
                "specialty": self.specialty,
                "user_id": context.get("user_id"),
                "channel_id": context.get("channel_id"),
                "tokens_used": tokens_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "processing_time_ms": processing_time_ms,
                "tool_results": tool_results
            }
            self.conversation_history.append(interaction_log)
            
            return {
                "response": response,
                "agent_id": self.agent_id,
                "specialty": self.specialty,
                "conversation_type": "specialist",
                "confidence": 0.9,  # High confidence for specialist
                "tokens_used": tokens_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "processing_cost": cost,
                "processing_time_ms": processing_time_ms,
                "tool_results": tool_results,
                "metadata": {
                    "model_used": self.llm.model_name,
                    "temperature": self.llm.temperature,
                    "agent_id": self.agent_id,
                    "specialty": self.specialty,
                    "tools_used": [tool.name for tool in self.tools if tool.enabled],
                    "performance_score": self.performance_metrics.get("average_quality_score", 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing message in Universal Agent '{self.specialty}': {str(e)}")
            
            # Update performance metrics for failure
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_performance_metrics(0, 0.0, processing_time_ms, False)
            
            return {
                "response": f"I apologize, but I'm having trouble processing your {self.specialty} request right now. Please try again or contact support if this continues.",
                "agent_id": self.agent_id,
                "specialty": self.specialty,
                "conversation_type": "error",
                "confidence": 0.0,
                "error": str(e),
                "tokens_used": 0,
                "processing_time_ms": processing_time_ms
            }
    
    async def _execute_tools(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute available tools based on the message content."""
        tool_results = {}
        
        for tool in self.tools:
            if not tool.enabled or not tool.function:
                continue
                
            try:
                # Simple tool execution - can be enhanced with more sophisticated matching
                if tool.name.lower() in message.lower():
                    result = await tool.function(message, context) if asyncio.iscoroutinefunction(tool.function) else tool.function(message, context)
                    tool_results[tool.name] = result
                    logger.debug(f"Executed tool '{tool.name}' for Universal Agent '{self.specialty}'")
            except Exception as e:
                logger.warning(f"Error executing tool '{tool.name}': {str(e)}")
                tool_results[tool.name] = {"error": str(e)}
        
        return tool_results
    
    async def _generate_specialist_response(self, message: str, context: Dict[str, Any], 
                                          history_context: str, tool_results: Dict[str, Any]) -> str:
        """Generate the specialist response using the LLM."""
        
        main_chain = self.main_prompt | self.llm
        
        # Prepare context including tool results
        enhanced_context = self._format_context(context)
        if tool_results:
            enhanced_context += f"\nTool Results: {json.dumps(tool_results, indent=2)}"
        
        response = await main_chain.ainvoke({
            "message": message,
            "context": enhanced_context,
            "history": history_context
        })
        
        return response.content
    
    async def analyze_performance(self, user_feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze agent performance and suggest improvements.
        
        Args:
            user_feedback: Optional user feedback on recent interactions
            
        Returns:
            Analysis results with improvement suggestions
        """
        try:
            if not self.conversation_history:
                return {"message": "No interactions to analyze yet"}
            
            # Get recent interaction for analysis
            recent_interaction = self.conversation_history[-1]
            
            analysis_chain = self.improvement_prompt | self.analysis_llm
            
            analysis_response = await analysis_chain.ainvoke({
                "user_message": recent_interaction.get("message_preview", ""),
                "agent_response": "Recent specialist response",  # Would need to store actual response
                "feedback": user_feedback or "No feedback provided",
                "metrics": json.dumps(self.performance_metrics, indent=2),
                "specialty": self.specialty
            })
            
            # Parse JSON response
            analysis_text = analysis_response.content.strip()
            if analysis_text.startswith("```json"):
                analysis_text = analysis_text.strip("```json").strip("```").strip()
            elif analysis_text.startswith("```"):
                analysis_text = analysis_text.strip("```").strip()
            
            analysis = json.loads(analysis_text)
            
            # Store improvement suggestions
            if "improvement_suggestions" in analysis:
                self.performance_metrics["improvement_suggestions"].extend(analysis["improvement_suggestions"])
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing performance for Universal Agent '{self.specialty}': {str(e)}")
            return {"error": str(e), "message": "Performance analysis failed"}
    
    def _update_performance_metrics(self, tokens: int, cost: float, processing_time: float, success: bool):
        """Update performance tracking metrics."""
        self.performance_metrics["total_interactions"] += 1
        
        if success:
            self.performance_metrics["successful_interactions"] += 1
        
        self.performance_metrics["total_tokens"] += tokens
        self.performance_metrics["total_cost"] += cost
        
        # Update average response time
        current_avg = self.performance_metrics["average_response_time"]
        total_interactions = self.performance_metrics["total_interactions"]
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total_interactions - 1) + processing_time) / total_interactions
        )
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for context."""
        if not history:
            return f"No previous conversation history in {self.specialty}."
        
        formatted_history = []
        for item in history[-3:]:  # Last 3 messages for context
            role = "User" if item.get("message_type") == "user_message" else "Assistant"
            content = item.get("content", "")[:100]  # Truncate for context
            formatted_history.append(f"{role}: {content}")
        
        return "\n".join(formatted_history)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information for the prompt."""
        context_parts = []
        
        if context.get("channel_type"):
            context_parts.append(f"Channel type: {context['channel_type']}")
        
        if context.get("user_id"):
            context_parts.append(f"User: {context['user_id']}")
        
        if context.get("is_thread"):
            context_parts.append("In thread conversation")
        
        context_parts.append(f"Specialty context: {self.specialty}")
        
        return ", ".join(context_parts) if context_parts else f"Standard {self.specialty} conversation"
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about this agent instance."""
        success_rate = 0.0
        if self.performance_metrics["total_interactions"] > 0:
            success_rate = self.performance_metrics["successful_interactions"] / self.performance_metrics["total_interactions"]
        
        return {
            "agent_id": self.agent_id,
            "specialty": self.specialty,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "total_interactions": self.performance_metrics["total_interactions"],
            "success_rate": success_rate,
            "total_tokens_used": self.performance_metrics["total_tokens"],
            "total_cost": self.performance_metrics["total_cost"],
            "average_response_time_ms": self.performance_metrics["average_response_time"],
            "cost_per_interaction": (
                self.performance_metrics["total_cost"] / self.performance_metrics["total_interactions"]
                if self.performance_metrics["total_interactions"] > 0 else 0.0
            ),
            "available_tools": [tool.name for tool in self.tools],
            "enabled_tools": [tool.name for tool in self.tools if tool.enabled],
            "recent_improvement_suggestions": self.performance_metrics["improvement_suggestions"][-5:],
            "created_at": datetime.utcnow().isoformat()
        }
    
    def update_configuration(self, **kwargs):
        """
        Update agent configuration dynamically.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        updatable_params = {
            "temperature": "temperature",
            "max_tokens": "max_tokens", 
            "system_prompt": "system_prompt",
            "tools": "tools"
        }
        
        for param, attr in updatable_params.items():
            if param in kwargs:
                setattr(self, attr, kwargs[param])
                logger.info(f"Updated {param} for Universal Agent '{self.specialty}'")
        
        # Recreate LLM if temperature or max_tokens changed
        if "temperature" in kwargs or "max_tokens" in kwargs:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=self.max_tokens,
            )
            logger.info(f"Recreated LLM for Universal Agent '{self.specialty}' with new parameters")
        
        # Recreate prompts if system_prompt changed
        if "system_prompt" in kwargs:
            self.main_prompt = self._create_main_prompt()
            logger.info(f"Recreated prompts for Universal Agent '{self.specialty}'")
    
    async def close(self):
        """
        Close the agent and cleanup resources.
        
        This method ensures proper cleanup of HTTP connections
        used by the OpenAI clients when shutting down.
        """
        try:
            logger.info(f"Closing Universal Agent '{self.specialty}' connections...")
            
            # Close the main LLM client
            if hasattr(self.llm, 'client') and hasattr(self.llm.client, 'close'):
                await self.llm.client.close()
            
            # Close the analysis LLM client
            if hasattr(self.analysis_llm, 'client') and hasattr(self.analysis_llm.client, 'close'):
                await self.analysis_llm.client.close()
            
            # Close any tool resources
            for tool in self.tools:
                if hasattr(tool, 'cleanup') and callable(tool.cleanup):
                    try:
                        await tool.cleanup() if asyncio.iscoroutinefunction(tool.cleanup) else tool.cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up tool '{tool.name}': {e}")
                
            logger.info(f"Universal Agent '{self.specialty}' connections closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing Universal Agent '{self.specialty}': {e}") 