"""
Filename: research_agent.py
Purpose: LLM-powered research agent for analysis, data gathering, and insights
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

class ResearchType(Enum):
    """Types of research the research agent handles."""
    MARKET_RESEARCH = "market_research"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    DATA_ANALYSIS = "data_analysis"
    ACADEMIC_RESEARCH = "academic_research"
    INDUSTRY_TRENDS = "industry_trends"
    CUSTOMER_INSIGHTS = "customer_insights"
    TECHNOLOGY_ASSESSMENT = "technology_assessment"
    STRATEGIC_PLANNING = "strategic_planning"

class ResearchSuggestion(BaseModel):
    """Structured research methodology suggestion."""
    research_approach: str
    data_sources: List[str]
    key_questions: List[str]
    deliverables: List[str]
    timeline_estimate: str
    confidence: float

class ResearchAgent:
    """
    LLM-powered research agent specialized for analysis and data gathering.
    
    Uses ChatOpenAI with research methodology prompts to provide comprehensive
    research guidance, analysis frameworks, and insights synthesis.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125", temperature: float = 0.4):
        """Initialize the LLM-powered Research Agent."""
        
        # Initialize the LLM - balanced temperature for analytical thinking
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,  # Balanced for analytical creativity
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=800,  # More tokens for comprehensive research responses
        )
        
        # Initialize methodology assessment LLM
        self.methodology_llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,  # Lower temperature for structured methodology
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=300,
        )
        
        # Initialize prompt templates
        self.main_prompt = self._create_main_prompt()
        self.methodology_prompt = self._create_methodology_prompt()
        
        self.research_history = []
        
        logger.info(f"Research Agent initialized with model: {model_name}")
    
    def _create_main_prompt(self) -> ChatPromptTemplate:
        """Create the main research conversation prompt template."""
        
        system_template = """You are the Research Agent for an AI Agent Platform - a specialized expert in research methodology, data analysis, and strategic insights. Your expertise and approach:

**Your Role:**
- Expert research guidance and methodology design
- Comprehensive analysis and insights synthesis
- Market research and competitive intelligence
- Data-driven decision support and strategic planning

**Your Research Expertise:**
- **Market Research**: Consumer behavior, market sizing, segmentation, surveys
- **Competitive Analysis**: Competitor profiling, positioning, SWOT analysis
- **Data Analysis**: Statistical analysis, trend identification, pattern recognition
- **Academic Research**: Literature reviews, methodology design, hypothesis testing
- **Industry Analysis**: Market trends, disruption analysis, technology assessment
- **Strategic Planning**: Business intelligence, opportunity assessment, risk analysis

**Your Personality:**
- Methodical and thorough in research approach
- Analytical thinker who sees patterns and connections
- Objective and evidence-based in conclusions
- Curious investigator who asks the right questions
- Clear communicator of complex insights 📊

**Your Research Methodology:**
1. **Define Objectives**: Clarify research questions and success criteria
2. **Design Approach**: Select appropriate research methods and data sources
3. **Gather Information**: Systematic data collection and source validation
4. **Analyze Findings**: Statistical analysis, pattern identification, insight synthesis
5. **Present Insights**: Clear, actionable recommendations with supporting evidence
6. **Validate Results**: Cross-reference findings and assess reliability

**Research Deliverables You Provide:**
- Executive summaries with key findings
- Detailed analysis with supporting data
- Actionable recommendations and next steps
- Research methodology and data source transparency
- Confidence assessments and limitations

**When to Recommend External Research:**
- Primary data collection (surveys, interviews)
- Real-time market data and industry reports
- Proprietary databases and specialized sources
- Statistical analysis of large datasets

Current research context: {context}
Previous research history: {history}
Research complexity level: {complexity}"""

        human_template = """Research request: {message}

Please provide expert research guidance. Include methodology recommendations, key questions to investigate, potential data sources, and expected deliverables."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_methodology_prompt(self) -> ChatPromptTemplate:
        """Create the research methodology assessment prompt template."""
        
        system_template = """You are a research methodology advisor. Analyze research requests to recommend optimal research approaches and frameworks.

**Your Task:**
For each research request, provide structured methodology recommendations including:
1. **Research Approach**: Qualitative, quantitative, or mixed methods
2. **Data Sources**: Primary and secondary sources to investigate
3. **Key Questions**: Critical research questions to address
4. **Deliverables**: Expected outputs and formats
5. **Timeline**: Realistic time estimate for completion

**Research Frameworks:**
- **Exploratory**: For new topics or hypothesis generation
- **Descriptive**: For characterizing markets, trends, or phenomena
- **Explanatory**: For understanding relationships and causation
- **Evaluative**: For assessing effectiveness or performance

Return your recommendations in this exact JSON format:
{{
    "research_approach": "exploratory|descriptive|explanatory|evaluative",
    "data_sources": ["source1", "source2", "source3"],
    "key_questions": ["question1", "question2", "question3"],
    "deliverables": ["deliverable1", "deliverable2"],
    "timeline_estimate": "timeframe description",
    "confidence": float
}}"""

        human_template = """Research request: "{message}"

Context: {context}

Provide structured research methodology recommendations:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a research request using specialized LLM intelligence.
        
        Args:
            message: Research request content
            context: Conversation and research context
            
        Returns:
            Dictionary containing research response and methodology
        """
        try:
            logger.info(f"Research Agent processing: '{message[:50]}...'")
            
            # Classify research type
            research_type = self._classify_research_type(message)
            
            # Assess research complexity
            complexity = self._assess_research_complexity(message, context)
            
            # Prepare research history context
            history_context = self._format_research_history(context.get("conversation_history", []))
            
            # Generate research methodology recommendations
            methodology = await self._generate_methodology(message, context)
            
            # Generate research response
            try:
                with get_openai_callback() as cb:
                    response = await self._generate_research_response(
                        message, context, history_context, research_type, complexity, methodology
                    )
                tokens_used = cb.total_tokens
                cost = cb.total_cost
            except Exception as e:
                logger.warning(f"OpenAI callback failed, proceeding without tracking: {e}")
                response = await self._generate_research_response(
                    message, context, history_context, research_type, complexity, methodology
                )
                tokens_used = 0
                cost = 0.0
            
            # Log the research interaction
            research_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "message_preview": message[:100],
                "research_type": research_type.value,
                "complexity": complexity,
                "methodology": methodology.dict() if methodology else None,
                "user_id": context.get("user_id"),
                "channel_id": context.get("channel_id"),
                "tokens_used": tokens_used,
                "cost": cost
            }
            self.research_history.append(research_log)
            
            return {
                "response": response,
                "research_type": research_type.value,
                "complexity": complexity,
                "methodology": methodology.dict() if methodology else None,
                "confidence": 0.85,  # High confidence for research expertise
                "tokens_used": tokens_used,
                "processing_cost": cost,
                "metadata": {
                    "agent_type": "research",
                    "model_used": self.llm.model_name,
                    "temperature": self.llm.temperature,
                    "specialization": "research_and_analysis"
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing research request: {str(e)}")
            return {
                "response": "I apologize, but I'm experiencing difficulties processing your research request. Please try rephrasing your question about market research, analysis, or data gathering, and I'll provide comprehensive research guidance.",
                "research_type": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _classify_research_type(self, message: str) -> ResearchType:
        """Classify the type of research being requested."""
        message_lower = message.lower()
        
        # Research type keywords
        market_keywords = ["market", "customer", "consumer", "buyer", "demand", "market size"]
        competitive_keywords = ["competitor", "competition", "competitive", "rival", "benchmark"]
        data_keywords = ["data", "analytics", "statistics", "metrics", "trends", "patterns"]
        academic_keywords = ["research", "study", "academic", "literature", "paper", "methodology"]
        industry_keywords = ["industry", "sector", "vertical", "market trends", "disruption"]
        customer_keywords = ["customer", "user", "feedback", "satisfaction", "behavior"]
        technology_keywords = ["technology", "tech", "innovation", "emerging", "assessment"]
        strategic_keywords = ["strategy", "planning", "opportunity", "business case", "feasibility"]
        
        # Count matches for each research type
        type_scores = {
            ResearchType.MARKET_RESEARCH: sum(1 for kw in market_keywords if kw in message_lower),
            ResearchType.COMPETITIVE_ANALYSIS: sum(1 for kw in competitive_keywords if kw in message_lower),
            ResearchType.DATA_ANALYSIS: sum(1 for kw in data_keywords if kw in message_lower),
            ResearchType.ACADEMIC_RESEARCH: sum(1 for kw in academic_keywords if kw in message_lower),
            ResearchType.INDUSTRY_TRENDS: sum(1 for kw in industry_keywords if kw in message_lower),
            ResearchType.CUSTOMER_INSIGHTS: sum(1 for kw in customer_keywords if kw in message_lower),
            ResearchType.TECHNOLOGY_ASSESSMENT: sum(1 for kw in technology_keywords if kw in message_lower),
            ResearchType.STRATEGIC_PLANNING: sum(1 for kw in strategic_keywords if kw in message_lower),
        }
        
        # Return type with highest score, default to market research
        best_type = max(type_scores.keys(), key=lambda k: type_scores[k])
        return best_type if type_scores[best_type] > 0 else ResearchType.MARKET_RESEARCH
    
    def _assess_research_complexity(self, message: str, context: Dict[str, Any]) -> str:
        """Assess the complexity level of the research request."""
        message_lower = message.lower()
        
        # Complexity indicators
        high_complexity = ["comprehensive", "detailed", "in-depth", "extensive", "complete analysis"]
        medium_complexity = ["analyze", "compare", "evaluate", "assess", "investigate"]
        low_complexity = ["overview", "summary", "quick", "basic", "simple"]
        
        high_count = sum(1 for indicator in high_complexity if indicator in message_lower)
        medium_count = sum(1 for indicator in medium_complexity if indicator in message_lower)
        low_count = sum(1 for indicator in low_complexity if indicator in message_lower)
        
        if high_count > 0:
            return "high"
        elif medium_count > low_count:
            return "medium"
        else:
            return "low"
    
    async def _generate_methodology(self, message: str, context: Dict[str, Any]) -> Optional[ResearchSuggestion]:
        """Generate research methodology recommendations."""
        
        try:
            methodology_chain = self.methodology_prompt | self.methodology_llm
            
            response = await methodology_chain.ainvoke({
                "message": message,
                "context": context.get("channel_type", "research_request")
            })
            
            # Extract JSON from response
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text.strip("```json").strip("```").strip()
            elif response_text.startswith("```"):
                response_text = response_text.strip("```").strip()
            
            methodology_data = json.loads(response_text)
            
            return ResearchSuggestion(**methodology_data)
            
        except Exception as e:
            logger.warning(f"Error generating methodology: {str(e)}")
            return None
    
    async def _generate_research_response(self, message: str, context: Dict[str, Any], 
                                        history_context: str, research_type: ResearchType,
                                        complexity: str, methodology: Optional[ResearchSuggestion]) -> str:
        """Generate the main research response using the LLM."""
        
        main_chain = self.main_prompt | self.llm
        
        response = await main_chain.ainvoke({
            "message": message,
            "context": self._format_context(context),
            "history": history_context,
            "complexity": complexity
        })
        
        response_text = response.content
        
        # Add research type-specific footer
        type_emoji = {
            ResearchType.MARKET_RESEARCH: "📊",
            ResearchType.COMPETITIVE_ANALYSIS: "🔍",
            ResearchType.DATA_ANALYSIS: "📈",
            ResearchType.ACADEMIC_RESEARCH: "📚",
            ResearchType.INDUSTRY_TRENDS: "🌐",
            ResearchType.CUSTOMER_INSIGHTS: "👥",
            ResearchType.TECHNOLOGY_ASSESSMENT: "🔬",
            ResearchType.STRATEGIC_PLANNING: "🎯"
        }
        
        emoji = type_emoji.get(research_type, "🔍")
        response_text += f"\n\n{emoji} *Research Agent - {research_type.value.replace('_', ' ').title()} Specialist*"
        
        # Add methodology summary if available
        if methodology:
            response_text += f"\n\n**Recommended Approach**: {methodology.research_approach.title()}"
            response_text += f"\n**Timeline Estimate**: {methodology.timeline_estimate}"
            if methodology.confidence > 0.7:
                response_text += f"\n**Confidence**: High ({methodology.confidence:.1%})"
        
        return response_text
    
    def _format_research_history(self, history: List[Dict[str, Any]]) -> str:
        """Format research conversation history for context."""
        if not history:
            return "No previous research conversation history."
        
        formatted_history = []
        for item in history[-3:]:  # Last 3 messages for context
            role = "User" if item.get("message_type") == "user_message" else "Research Agent"
            content = item.get("content", "")[:150]  # More context for research discussions
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
            context_parts.append("Research discussion thread")
        
        if context.get("organization"):
            context_parts.append(f"Organization: {context['organization']}")
        
        return ", ".join(context_parts) if context_parts else "Research consultation session"
    
    def get_research_stats(self) -> Dict[str, Any]:
        """Get statistics about research interactions handled."""
        if not self.research_history:
            return {"total_research_requests": 0}
        
        total_tokens = sum(log.get("tokens_used", 0) for log in self.research_history)
        total_cost = sum(log.get("cost", 0) for log in self.research_history)
        
        # Research type distribution
        type_counts = {}
        complexity_counts = {}
        methodology_requests = sum(1 for log in self.research_history if log.get("methodology"))
        
        for log in self.research_history:
            research_type = log.get("research_type", "unknown")
            type_counts[research_type] = type_counts.get(research_type, 0) + 1
            
            complexity = log.get("complexity", "unknown")
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        return {
            "total_research_requests": len(self.research_history),
            "total_tokens_used": total_tokens,
            "total_cost": total_cost,
            "research_type_distribution": type_counts,
            "complexity_distribution": complexity_counts,
            "methodology_requests": methodology_requests,
            "methodology_request_rate": methodology_requests / len(self.research_history) if self.research_history else 0
        }
    
    async def close(self):
        """
        Close the agent and cleanup resources.
        
        This method ensures proper cleanup of HTTP connections
        used by the OpenAI clients when shutting down.
        """
        try:
            logger.info("Closing Research Agent connections...")
            
            # Close the main LLM client
            if hasattr(self.llm, 'client') and hasattr(self.llm.client, 'close'):
                await self.llm.client.close()
            
            # Close the methodology LLM client
            if hasattr(self.methodology_llm, 'client') and hasattr(self.methodology_llm.client, 'close'):
                await self.methodology_llm.client.close()
                
            logger.info("Research Agent connections closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing Research Agent: {e}") 