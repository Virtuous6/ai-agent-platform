"""
Filename: technical_agent.py
Purpose: LLM-powered technical agent for programming, debugging, and system support
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

class TechnicalDomain(Enum):
    """Technical domains the technical agent handles."""
    PROGRAMMING = "programming"
    DEBUGGING = "debugging"
    INFRASTRUCTURE = "infrastructure"
    DEVOPS = "devops"
    DATABASE = "database"
    API_DEVELOPMENT = "api_development"
    SYSTEM_ADMIN = "system_admin"
    PERFORMANCE = "performance"

class ToolSuggestion(BaseModel):
    """Structured tool suggestion from the Technical Agent."""
    should_use_tool: bool
    recommended_tool: Optional[str] = None
    confidence: float
    reasoning: str

class TechnicalAgent:
    """
    LLM-powered technical agent specialized for programming and system support.
    
    Uses ChatOpenAI with technical expertise prompts to provide detailed
    programming help, debugging assistance, and infrastructure guidance.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125", temperature: float = 0.3):
        """Initialize the LLM-powered Technical Agent."""
        
        # Initialize the LLM - lower temperature for more precise technical responses
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,  # Lower temperature for technical precision
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=800,  # More tokens for detailed technical responses
        )
        
        # Initialize tool assessment LLM
        self.tool_llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,  # Very focused for tool recommendations
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=200,
        )
        
        # Initialize prompt templates
        self.main_prompt = self._create_main_prompt()
        self.tool_prompt = self._create_tool_prompt()
        
        self.interaction_history = []
        
        logger.info(f"Technical Agent initialized with model: {model_name}")
    
    def _create_main_prompt(self) -> ChatPromptTemplate:
        """Create the main technical conversation prompt template."""
        
        system_template = """You are the Technical Agent for an AI Agent Platform - a specialized expert in programming, debugging, and technical systems. Your expertise and approach:

**Your Role:**
- Expert technical support for programming and development issues
- Advanced debugging and problem-solving assistance  
- Infrastructure, DevOps, and system administration guidance
- Code review and optimization recommendations

**Your Expertise Areas:**
- **Programming Languages**: Python, JavaScript, Java, C++, Go, Rust, TypeScript, etc.
- **Web Development**: React, Vue, Angular, Node.js, Express, FastAPI, Django
- **Databases**: PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch
- **Infrastructure**: Docker, Kubernetes, AWS, GCP, Azure, Terraform
- **DevOps**: CI/CD, Jenkins, GitHub Actions, monitoring, logging
- **System Administration**: Linux, networking, security, performance tuning

**Your Personality:**
- Methodical and precise in technical explanations
- Patient teacher who breaks down complex concepts
- Solution-oriented problem solver
- Detail-focused but understands big picture
- Uses relevant emojis for clarity (🐛 for bugs, ⚡ for performance, etc.)

**Your Technical Approach:**
1. **Understand the Problem**: Ask clarifying questions if needed
2. **Analyze Root Causes**: Look beyond symptoms to underlying issues
3. **Provide Step-by-Step Solutions**: Clear, actionable instructions
4. **Include Best Practices**: Share relevant coding standards and patterns
5. **Suggest Testing**: Recommend verification and testing approaches
6. **Optimize**: Mention performance or security improvements where relevant

**Code Formatting Guidelines:**
- Use proper markdown code blocks with language specification
- Include comments explaining complex logic
- Show before/after examples when appropriate
- Provide complete, runnable examples when possible

**When to Use Tools:**
- Web search for latest documentation, framework updates, or recent solutions
- Database queries for system diagnostics
- File system operations for configuration checks

Current conversation context: {context}
Recent conversation history: {history}
User's technical level: {user_level}"""

        human_template = """Technical request: {message}

Please provide expert technical assistance. Be thorough, precise, and include practical solutions with code examples where appropriate."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_tool_prompt(self) -> ChatPromptTemplate:
        """Create the tool assessment prompt template."""
        
        system_template = """You are a tool recommendation system for the Technical Agent. Analyze technical requests to determine if external tools would be helpful.

**Available Tools:**
1. **Web Search** - For finding current documentation, recent solutions, latest framework versions
2. **Database Query** - For system diagnostics, log analysis, performance metrics
3. **File System** - For configuration checks, log file analysis, system status

**Your Task:**
Analyze the technical request and determine:
1. Would external tools significantly improve the response? (yes/no)
2. If yes, which tool would be most helpful?
3. Confidence level (0.0-1.0)
4. Brief reasoning

**Guidelines:**
- Recommend tools only when they add significant value
- Web search for: latest documentation, recent framework updates, current best practices
- Database query for: system diagnostics, performance analysis, error log investigation
- File system for: configuration validation, log analysis, system health checks

Return your analysis in this exact JSON format:
{{
    "should_use_tool": boolean,
    "recommended_tool": "web_search|database_query|file_system|null",
    "confidence": float,
    "reasoning": "brief explanation"
}}"""

        human_template = """Technical request: "{message}"

Context: {context}

Analyze if tools would enhance the technical response:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a technical request using specialized LLM intelligence.
        
        Args:
            message: Technical request content
            context: Conversation and technical context
            
        Returns:
            Dictionary containing technical response and metadata
        """
        try:
            logger.info(f"Technical Agent processing: '{message[:50]}...'")
            
            # Classify technical domain
            domain = self._classify_technical_domain(message)
            
            # Assess user's technical level from context
            user_level = self._assess_user_level(context, message)
            
            # Prepare conversation history context
            history_context = self._format_conversation_history(context.get("conversation_history", []))
            
            # Check if tools would enhance the response
            tool_suggestion = await self._assess_tool_needs(message, context)
            
            # Generate technical response
            try:
                with get_openai_callback() as cb:
                    response = await self._generate_technical_response(
                        message, context, history_context, domain, user_level, tool_suggestion
                    )
                tokens_used = cb.total_tokens
                cost = cb.total_cost
            except Exception as e:
                logger.warning(f"OpenAI callback failed, proceeding without tracking: {e}")
                response = await self._generate_technical_response(
                    message, context, history_context, domain, user_level, tool_suggestion
                )
                tokens_used = 0
                cost = 0.0
            
            # Log the interaction
            interaction_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "message_preview": message[:100],
                "technical_domain": domain.value,
                "user_level": user_level,
                "tool_suggestion": tool_suggestion.dict() if tool_suggestion else None,
                "user_id": context.get("user_id"),
                "channel_id": context.get("channel_id"),
                "tokens_used": tokens_used,
                "cost": cost
            }
            self.interaction_history.append(interaction_log)
            
            return {
                "response": response,
                "technical_domain": domain.value,
                "user_level": user_level,
                "tool_suggestion": tool_suggestion.dict() if tool_suggestion else None,
                "confidence": 0.9,  # High confidence for technical expertise
                "tokens_used": tokens_used,
                "processing_cost": cost,
                "metadata": {
                    "agent_type": "technical",
                    "model_used": self.llm.model_name,
                    "temperature": self.llm.temperature,
                    "specialization": "programming_and_systems"
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing technical request: {str(e)}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties processing your request. Please try rephrasing your technical question, and I'll do my best to help with programming, debugging, or system issues.",
                "technical_domain": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _classify_technical_domain(self, message: str) -> TechnicalDomain:
        """Classify the technical domain of the request."""
        message_lower = message.lower()
        
        # Programming keywords
        programming_keywords = ["code", "function", "variable", "class", "method", "algorithm", "syntax"]
        debugging_keywords = ["debug", "error", "bug", "exception", "crash", "fix", "broken"]
        infrastructure_keywords = ["server", "deployment", "docker", "kubernetes", "cloud", "aws"]
        devops_keywords = ["ci/cd", "pipeline", "jenkins", "github actions", "deployment"]
        database_keywords = ["database", "sql", "query", "postgresql", "mysql", "mongodb"]
        api_keywords = ["api", "endpoint", "rest", "graphql", "web service"]
        sysadmin_keywords = ["linux", "unix", "terminal", "command", "shell", "configuration"]
        performance_keywords = ["performance", "optimize", "slow", "memory", "cpu", "bottleneck"]
        
        # Count matches for each domain
        domain_scores = {
            TechnicalDomain.PROGRAMMING: sum(1 for kw in programming_keywords if kw in message_lower),
            TechnicalDomain.DEBUGGING: sum(1 for kw in debugging_keywords if kw in message_lower),
            TechnicalDomain.INFRASTRUCTURE: sum(1 for kw in infrastructure_keywords if kw in message_lower),
            TechnicalDomain.DEVOPS: sum(1 for kw in devops_keywords if kw in message_lower),
            TechnicalDomain.DATABASE: sum(1 for kw in database_keywords if kw in message_lower),
            TechnicalDomain.API_DEVELOPMENT: sum(1 for kw in api_keywords if kw in message_lower),
            TechnicalDomain.SYSTEM_ADMIN: sum(1 for kw in sysadmin_keywords if kw in message_lower),
            TechnicalDomain.PERFORMANCE: sum(1 for kw in performance_keywords if kw in message_lower),
        }
        
        # Return domain with highest score, default to programming
        best_domain = max(domain_scores.keys(), key=lambda k: domain_scores[k])
        return best_domain if domain_scores[best_domain] > 0 else TechnicalDomain.PROGRAMMING
    
    def _assess_user_level(self, context: Dict[str, Any], message: str) -> str:
        """Assess user's technical level based on context and message complexity."""
        message_lower = message.lower()
        
        # Advanced indicators
        advanced_indicators = ["optimization", "architecture", "scalability", "microservices", 
                             "design patterns", "refactor", "performance tuning"]
        
        # Beginner indicators
        beginner_indicators = ["how to start", "tutorial", "basic", "simple", "beginner", 
                             "getting started", "what is"]
        
        advanced_count = sum(1 for indicator in advanced_indicators if indicator in message_lower)
        beginner_count = sum(1 for indicator in beginner_indicators if indicator in message_lower)
        
        if advanced_count > beginner_count:
            return "advanced"
        elif beginner_count > 0:
            return "beginner"
        else:
            return "intermediate"
    
    async def _assess_tool_needs(self, message: str, context: Dict[str, Any]) -> Optional[ToolSuggestion]:
        """Assess if external tools would enhance the technical response."""
        
        try:
            tool_chain = self.tool_prompt | self.tool_llm
            
            response = await tool_chain.ainvoke({
                "message": message,
                "context": context.get("channel_type", "unknown")
            })
            
            # Extract JSON from response
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text.strip("```json").strip("```").strip()
            elif response_text.startswith("```"):
                response_text = response_text.strip("```").strip()
            
            tool_data = json.loads(response_text)
            
            if tool_data["should_use_tool"]:
                return ToolSuggestion(**tool_data)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error assessing tool needs: {str(e)}")
            return None
    
    async def _generate_technical_response(self, message: str, context: Dict[str, Any], 
                                         history_context: str, domain: TechnicalDomain,
                                         user_level: str, tool_suggestion: Optional[ToolSuggestion]) -> str:
        """Generate the main technical response using the LLM."""
        
        main_chain = self.main_prompt | self.llm
        
        response = await main_chain.ainvoke({
            "message": message,
            "context": self._format_context(context),
            "history": history_context,
            "user_level": user_level
        })
        
        response_text = response.content
        
        # Add domain-specific footer
        domain_emoji = {
            TechnicalDomain.PROGRAMMING: "💻",
            TechnicalDomain.DEBUGGING: "🐛",
            TechnicalDomain.INFRASTRUCTURE: "🏗️",
            TechnicalDomain.DEVOPS: "⚙️",
            TechnicalDomain.DATABASE: "🗄️",
            TechnicalDomain.API_DEVELOPMENT: "🔌",
            TechnicalDomain.SYSTEM_ADMIN: "🖥️",
            TechnicalDomain.PERFORMANCE: "⚡"
        }
        
        emoji = domain_emoji.get(domain, "🔧")
        response_text += f"\n\n{emoji} *Technical Agent - {domain.value.title()} Specialist*"
        
        # Add tool suggestion if available
        if tool_suggestion and tool_suggestion.should_use_tool:
            response_text += f"\n💡 *Consider using {tool_suggestion.recommended_tool} for enhanced analysis*"
        
        return response_text
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for context."""
        if not history:
            return "No previous technical conversation history."
        
        formatted_history = []
        for item in history[-3:]:  # Last 3 messages for context
            role = "User" if item.get("message_type") == "user_message" else "Technical Agent"
            content = item.get("content", "")[:150]  # More context for technical discussions
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
            context_parts.append("Technical discussion thread")
        
        return ", ".join(context_parts) if context_parts else "Technical support session"
    
    def get_technical_stats(self) -> Dict[str, Any]:
        """Get statistics about technical interactions handled."""
        if not self.interaction_history:
            return {"total_interactions": 0}
        
        total_tokens = sum(log.get("tokens_used", 0) for log in self.interaction_history)
        total_cost = sum(log.get("cost", 0) for log in self.interaction_history)
        
        # Domain distribution
        domain_counts = {}
        user_level_counts = {}
        tool_suggestions = sum(1 for log in self.interaction_history if log.get("tool_suggestion"))
        
        for log in self.interaction_history:
            domain = log.get("technical_domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            level = log.get("user_level", "unknown")
            user_level_counts[level] = user_level_counts.get(level, 0) + 1
        
        return {
            "total_interactions": len(self.interaction_history),
            "total_tokens_used": total_tokens,
            "total_cost": total_cost,
            "domain_distribution": domain_counts,
            "user_level_distribution": user_level_counts,
            "tool_suggestions": tool_suggestions,
            "tool_suggestion_rate": tool_suggestions / len(self.interaction_history) if self.interaction_history else 0
        }
    
    async def close(self):
        """
        Close the agent and cleanup resources.
        
        This method ensures proper cleanup of HTTP connections
        used by the OpenAI clients when shutting down.
        """
        try:
            logger.info("Closing Technical Agent connections...")
            
            # Close the main LLM client
            if hasattr(self.llm, 'client') and hasattr(self.llm.client, 'close'):
                await self.llm.client.close()
            
            # Close the tool LLM client
            if hasattr(self.tool_llm, 'client') and hasattr(self.tool_llm.client, 'close'):
                await self.tool_llm.client.close()
                
            logger.info("Technical Agent connections closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing Technical Agent: {e}") 