"""
Filename: general_agent.py
Purpose: LLM-powered general conversation agent for the AI Agent Platform
Dependencies: langchain, openai, asyncio, logging, typing

This module is part of the AI Agent Platform.
See README.llm.md in this directory for context.
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

class ConversationType(Enum):
    """Types of conversations the general agent handles."""
    GREETING = "greeting"
    QUESTION = "question"
    GRATITUDE = "gratitude"
    SOCIAL = "social"
    HELP_REQUEST = "help_request"
    ESCALATION_NEEDED = "escalation_needed"
    GENERAL = "general"

class EscalationSuggestion(BaseModel):
    """Structured escalation suggestion from the LLM."""
    should_escalate: bool
    recommended_agent: Optional[str] = None
    confidence: float
    reasoning: str

class GeneralAgent:
    """
    LLM-powered general conversation agent that provides intelligent responses.
    
    Uses ChatOpenAI with carefully crafted prompts to maintain personality
    while providing helpful, contextual responses.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125", temperature: float = 0.7):
        """Initialize the LLM-powered General Agent."""
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=500,
        )
        
        # Initialize escalation LLM (more focused, lower temperature)
        self.escalation_llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=200,
        )
        
        # Initialize prompt templates
        self.main_prompt = self._create_main_prompt()
        self.escalation_prompt = self._create_escalation_prompt()
        
        self.conversation_history = []
        
        logger.info(f"General Agent initialized with model: {model_name}")
    
    def _create_main_prompt(self) -> ChatPromptTemplate:
        """Create the main conversation prompt template."""
        
        system_template = """You are the General Agent for an AI Agent Platform - a friendly, helpful assistant integrated into Slack. Your personality and behavior:

**Your Role:**
- Primary interface for general conversations, questions, and everyday assistance
- Bridge between users and specialized agents when needed
- Maintain warm, professional tone while being genuinely helpful

**Your Personality:**
- Warm, friendly, and approachable 😊
- Professional but not stiff - like a helpful colleague
- Enthusiastic about helping users succeed
- Clear communicator who explains things simply
- Patient and understanding

**Your Capabilities:**
- Answer general questions using your knowledge
- Provide explanations, definitions, and guidance
- Help with basic problem-solving and decision-making
- Engage in natural conversation
- Recognize when specialized help is needed

**Important Guidelines:**
1. Be conversational and natural - you're talking in Slack, not writing a formal document
2. Use emojis appropriately to convey warmth (but don't overdo it)
3. If you're uncertain about something, be honest about limitations
4. For technical coding issues, research questions, or specialized tasks, suggest escalation
5. Keep responses concise but complete - Slack messages should be scannable
6. Reference previous conversation context when relevant

**When to Suggest Escalation:**
- Technical/programming questions → Technical Agent
- Research, analysis, or data gathering → Research Agent
- Complex specialized topics outside general knowledge

Current conversation context: {context}
Recent conversation history: {history}"""

        human_template = """User message: {message}

Please respond as the General Agent, being helpful, warm, and professional. If this requires specialized assistance, you can mention it, but still provide what help you can."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_escalation_prompt(self) -> ChatPromptTemplate:
        """Create the escalation assessment prompt template."""
        
        system_template = """You are an escalation classifier for an AI Agent Platform. Analyze user messages to determine if they need specialized agent assistance.

**Available Specialized Agents:**
1. **Technical Agent** - Handles: programming, debugging, technical issues, system administration, DevOps, infrastructure
2. **Research Agent** - Handles: research tasks, data analysis, market research, competitive intelligence, information gathering

**Your Task:**
Analyze the user message and determine:
1. Should this be escalated to a specialist? (yes/no)
2. If yes, which agent? (technical or research)
3. Confidence level (0.0-1.0)
4. Brief reasoning

**Guidelines:**
- Only escalate if the user specifically needs specialized expertise
- General questions about topics can often be handled by General Agent
- Consider the user's intent and specificity
- Be conservative - don't over-escalate simple questions

Return your analysis in this exact JSON format:
{{
    "should_escalate": boolean,
    "recommended_agent": "technical|research|null",
    "confidence": float,
    "reasoning": "brief explanation"
}}"""

        human_template = """User message: "{message}"

Context: {context}

Analyze this message for escalation needs:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user message using LLM intelligence.
        
        Args:
            message: User message content
            context: Conversation context
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            logger.info(f"General Agent processing: '{message[:50]}...'")
            
            # Prepare conversation history context
            history_context = self._format_conversation_history(context.get("conversation_history", []))
            
            # Check if escalation is needed using focused LLM call
            escalation_suggestion = await self._assess_escalation_needs(message, context)
            
            # Generate main response
            try:
                with get_openai_callback() as cb:
                    response = await self._generate_response(message, context, history_context, escalation_suggestion)
                tokens_used = cb.total_tokens
                cost = cb.total_cost
            except Exception as e:
                logger.warning(f"OpenAI callback failed, proceeding without tracking: {e}")
                response = await self._generate_response(message, context, history_context, escalation_suggestion)
                tokens_used = 0
                cost = 0.0
            
            # Log the interaction
            interaction_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "message_preview": message[:100],
                "escalation_suggestion": escalation_suggestion.dict() if escalation_suggestion else None,
                "user_id": context.get("user_id"),
                "channel_id": context.get("channel_id"),
                "tokens_used": tokens_used,
                "cost": cost
            }
            self.conversation_history.append(interaction_log)
            
            return {
                "response": response,
                "escalation_suggestion": escalation_suggestion.dict() if escalation_suggestion else None,
                "conversation_type": "general",  # LLM determined this
                "confidence": 0.8,  # General agent confidence
                "tokens_used": tokens_used,
                "processing_cost": cost,
                "metadata": {
                    "model_used": self.llm.model_name,
                    "temperature": self.llm.temperature,
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "response": "I apologize, but I'm having trouble processing your message right now. Please try again or contact support if this continues.",
                "escalation_suggestion": None,
                "conversation_type": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _assess_escalation_needs(self, message: str, context: Dict[str, Any]) -> Optional[EscalationSuggestion]:
        """Assess if the message needs escalation using LLM analysis."""
        
        try:
            escalation_chain = self.escalation_prompt | self.escalation_llm
            
            response = await escalation_chain.ainvoke({
                "message": message,
                "context": context.get("channel_type", "unknown")
            })
            
            # Extract JSON from response (handle potential markdown formatting)
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text.strip("```json").strip("```").strip()
            elif response_text.startswith("```"):
                response_text = response_text.strip("```").strip()
            
            escalation_data = json.loads(response_text)
            
            if escalation_data["should_escalate"]:
                return EscalationSuggestion(**escalation_data)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error assessing escalation needs: {str(e)}")
            # Fall back to simple keyword detection
            return self._fallback_escalation_check(message)
    
    def _fallback_escalation_check(self, message: str) -> Optional[EscalationSuggestion]:
        """Fallback escalation check using simple keyword matching."""
        
        message_lower = message.lower()
        
        technical_keywords = ["code", "programming", "debug", "error", "bug", "api", "technical", "server"]
        research_keywords = ["research", "analyze", "data", "study", "market", "competitor"]
        
        technical_matches = sum(1 for keyword in technical_keywords if keyword in message_lower)
        research_matches = sum(1 for keyword in research_keywords if keyword in message_lower)
        
        if technical_matches >= 2:
            return EscalationSuggestion(
                should_escalate=True,
                recommended_agent="technical",
                confidence=0.6,
                reasoning="Multiple technical keywords detected"
            )
        elif research_matches >= 2:
            return EscalationSuggestion(
                should_escalate=True,
                recommended_agent="research",
                confidence=0.6,
                reasoning="Multiple research keywords detected"
            )
        
        return None
    
    async def _generate_response(self, message: str, context: Dict[str, Any], 
                               history_context: str, escalation_suggestion: Optional[EscalationSuggestion]) -> str:
        """Generate the main response using the LLM."""
        
        main_chain = self.main_prompt | self.llm
        
        response = await main_chain.ainvoke({
            "message": message,
            "context": self._format_context(context),
            "history": history_context
        })
        
        response_text = response.content
        
        # Add escalation note if needed
        if escalation_suggestion and escalation_suggestion.should_escalate:
            agent_name = "Technical Agent" if escalation_suggestion.recommended_agent == "technical" else "Research Agent"
            response_text += f"\n\n💡 *For more specialized help with this, you might want to mention the {agent_name} specifically.*"
        
        return response_text
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for context."""
        if not history:
            return "No previous conversation history."
        
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
        
        return ", ".join(context_parts) if context_parts else "Standard conversation"
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about conversations handled."""
        if not self.conversation_history:
            return {"total_conversations": 0}
        
        total_tokens = sum(log.get("tokens_used", 0) for log in self.conversation_history)
        total_cost = sum(log.get("cost", 0) for log in self.conversation_history)
        escalations = sum(1 for log in self.conversation_history if log.get("escalation_suggestion"))
        
        return {
            "total_conversations": len(self.conversation_history),
            "total_tokens_used": total_tokens,
            "total_cost": total_cost,
            "escalation_suggestions": escalations,
            "escalation_rate": escalations / len(self.conversation_history) if self.conversation_history else 0
        }
    
    async def close(self):
        """
        Close the agent and cleanup resources.
        
        This method ensures proper cleanup of HTTP connections
        used by the OpenAI clients when shutting down.
        """
        try:
            logger.info("Closing General Agent connections...")
            
            # Close the main LLM client
            if hasattr(self.llm, 'client') and hasattr(self.llm.client, 'close'):
                await self.llm.client.close()
            
            # Close the escalation LLM client
            if hasattr(self.escalation_llm, 'client') and hasattr(self.escalation_llm.client, 'close'):
                await self.escalation_llm.client.close()
                
            logger.info("General Agent connections closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing General Agent: {e}") 