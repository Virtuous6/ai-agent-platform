"""
Query Complexity Router

Automatically assesses query complexity to determine appropriate response strategy:
- Simple (0.0-0.3): Direct agent response
- Medium (0.3-0.7): Agent + MCP tools  
- Complex (0.7-1.0): Goal creation + multi-agent workflow
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ComplexityLevel(Enum):
    """Query complexity levels for routing decisions."""
    SIMPLE = "simple"      # 0.0-0.3: Direct agent response
    MEDIUM = "medium"      # 0.3-0.7: Agent + tools
    COMPLEX = "complex"    # 0.7-1.0: Goal workflow

class ResponseStrategy(Enum):
    """Response strategies based on complexity."""
    DIRECT_RESPONSE = "direct_response"
    AGENT_WITH_TOOLS = "agent_with_tools"  
    GOAL_WORKFLOW = "goal_workflow"

class ComplexityAssessment(BaseModel):
    """Schema for LLM complexity assessment response."""
    complexity_score: float = Field(
        description="Complexity score between 0.0 and 1.0",
        ge=0.0, le=1.0
    )
    complexity_level: str = Field(
        description="Complexity level: 'simple', 'medium', or 'complex'"
    )
    reasoning: str = Field(
        description="Brief explanation of the complexity assessment"
    )
    indicators: List[str] = Field(
        description="Key indicators that influenced the complexity score"
    )
    recommended_strategy: str = Field(
        description="Recommended response strategy"
    )
    estimated_steps: int = Field(
        description="Estimated number of steps required",
        ge=1
    )

@dataclass
class ComplexityAnalysis:
    """Complete complexity analysis result."""
    score: float
    level: ComplexityLevel
    strategy: ResponseStrategy
    reasoning: str
    indicators: List[str]
    estimated_steps: int
    requires_tools: bool
    requires_multiple_agents: bool
    estimated_cost: float

class QueryComplexityRouter:
    """
    Intelligent query complexity assessment for optimal routing decisions.
    
    Analyzes user queries to determine the most appropriate response strategy,
    from simple direct answers to complex multi-agent workflows.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize the complexity router with LLM for assessment."""
        self.llm = llm or ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0.2,  # Low temperature for consistent assessment
            max_tokens=300
        )
        
        # Create complexity assessment prompt
        self.complexity_prompt = self._create_complexity_prompt()
        self.complexity_parser = PydanticOutputParser(pydantic_object=ComplexityAssessment)
        
        # Create assessment chain
        self.complexity_chain = self.complexity_prompt | self.llm | self.complexity_parser
        
        # Complexity indicators for quick assessment
        self.simple_indicators = {
            'question_words': ['what is', 'what are', 'define', 'explain', 'how do you', 'who is'],
            'direct_queries': ['meaning of', 'definition of', 'tell me about', 'describe'],
            'factual_requests': ['when did', 'where is', 'why does', 'how many']
        }
        
        self.medium_indicators = {
            'analysis_tasks': ['analyze', 'compare', 'evaluate', 'assess', 'review'],
            'research_tasks': ['research', 'investigate', 'find information', 'gather data'],
            'tool_requiring': ['search for', 'look up', 'calculate', 'translate', 'convert']
        }
        
        self.complex_indicators = {
            'creation_tasks': ['build', 'create', 'develop', 'implement', 'design'],
            'multi_step': ['deploy', 'setup', 'configure', 'optimize', 'migrate'],
            'coordination': ['manage', 'coordinate', 'integrate', 'automate', 'orchestrate']
        }
        
        # Track assessment history for learning
        self.assessment_history: List[Dict[str, Any]] = []
        
        logger.info("QueryComplexityRouter initialized")
    
    def _create_complexity_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for complexity assessment."""
        return ChatPromptTemplate.from_template("""
You are an expert AI query complexity analyzer. Assess the complexity of user requests to determine the optimal response strategy.

**Complexity Levels:**
- **Simple (0.0-0.3)**: Direct questions, definitions, explanations that need a straightforward answer
  - Examples: "What is Python?", "Explain REST APIs", "Define machine learning"
  - Strategy: Direct agent response

- **Medium (0.3-0.7)**: Tasks requiring analysis, research, or tools but single-focus
  - Examples: "Analyze this code", "Research market trends", "Find the best framework for X"  
  - Strategy: Agent + MCP tools

- **Complex (0.7-1.0)**: Multi-step projects, creation tasks, or coordination requiring multiple agents
  - Examples: "Build a web app", "Deploy infrastructure", "Create a marketing strategy"
  - Strategy: Goal-oriented workflow with multiple agents

**User Query:** {message}

**Context:** {context}

**Assessment Instructions:**
1. Analyze the query's scope, required steps, and complexity
2. Consider if it needs tools, multiple agents, or coordination
3. Estimate the number of steps required
4. Assign a precise complexity score (0.0-1.0)
5. Recommend the optimal response strategy

{format_instructions}
""")
    
    async def assess_complexity(self, message: str, context: Dict[str, Any] = None) -> ComplexityAnalysis:
        """
        Assess the complexity of a user query and determine response strategy.
        
        Args:
            message: User query to assess
            context: Additional context for assessment
            
        Returns:
            ComplexityAnalysis with routing recommendations
        """
        try:
            start_time = datetime.utcnow()
            
            # Prepare context string
            context_str = self._format_context(context or {})
            
            # Get LLM assessment
            assessment = await self.complexity_chain.ainvoke({
                "message": message,
                "context": context_str,
                "format_instructions": self.complexity_parser.get_format_instructions()
            })
            
            # Convert to analysis object
            analysis = self._create_analysis_from_assessment(assessment, message)
            
            # Track for learning
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            await self._track_assessment(message, analysis, processing_time)
            
            logger.info(f"Complexity assessment: {analysis.level.value} ({analysis.score:.2f}) - {analysis.strategy.value}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in complexity assessment: {e}")
            # Fallback to rule-based assessment
            return await self._fallback_assessment(message, context or {})
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for LLM prompt."""
        context_parts = []
        
        if context.get('user_id'):
            context_parts.append(f"User: {context['user_id']}")
        if context.get('conversation_history'):
            recent_history = context['conversation_history'][-3:]  # Last 3 messages
            context_parts.append(f"Recent conversation: {len(recent_history)} messages")
        if context.get('goal_id'):
            context_parts.append("Part of ongoing goal")
        
        return "; ".join(context_parts) if context_parts else "No additional context"
    
    def _create_analysis_from_assessment(self, assessment: ComplexityAssessment, message: str) -> ComplexityAnalysis:
        """Convert LLM assessment to ComplexityAnalysis object."""
        
        # Determine complexity level
        if assessment.complexity_score < 0.3:
            level = ComplexityLevel.SIMPLE
            strategy = ResponseStrategy.DIRECT_RESPONSE
        elif assessment.complexity_score < 0.7:
            level = ComplexityLevel.MEDIUM
            strategy = ResponseStrategy.AGENT_WITH_TOOLS
        else:
            level = ComplexityLevel.COMPLEX
            strategy = ResponseStrategy.GOAL_WORKFLOW
        
        # Determine requirements
        requires_tools = level in [ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX]
        requires_multiple_agents = level == ComplexityLevel.COMPLEX
        
        # Estimate cost based on complexity
        base_cost = 0.01  # Base cost per query
        estimated_cost = base_cost * (1 + assessment.complexity_score * 5)  # Scale with complexity
        
        return ComplexityAnalysis(
            score=assessment.complexity_score,
            level=level,
            strategy=strategy,
            reasoning=assessment.reasoning,
            indicators=assessment.indicators,
            estimated_steps=assessment.estimated_steps,
            requires_tools=requires_tools,
            requires_multiple_agents=requires_multiple_agents,
            estimated_cost=estimated_cost
        )
    
    async def _fallback_assessment(self, message: str, context: Dict[str, Any]) -> ComplexityAnalysis:
        """Fallback rule-based assessment if LLM fails."""
        logger.warning("Using fallback rule-based complexity assessment")
        
        message_lower = message.lower()
        score = 0.5  # Default medium complexity
        indicators = ["fallback_assessment"]
        
        # Check for simple indicators
        simple_count = sum(1 for word_list in self.simple_indicators.values() 
                          for word in word_list if word in message_lower)
        
        # Check for medium indicators  
        medium_count = sum(1 for word_list in self.medium_indicators.values()
                          for word in word_list if word in message_lower)
        
        # Check for complex indicators
        complex_count = sum(1 for word_list in self.complex_indicators.values()
                           for word in word_list if word in message_lower)
        
        # Calculate score based on indicator counts
        if simple_count > medium_count and simple_count > complex_count:
            score = 0.2
            indicators.append("simple_keywords")
        elif complex_count > simple_count and complex_count > medium_count:
            score = 0.8
            indicators.append("complex_keywords")
        else:
            score = 0.5
            indicators.append("medium_keywords")
        
        # Message length factor
        if len(message.split()) > 20:
            score += 0.1
            indicators.append("long_message")
        
        # Question vs command factor
        if message.strip().endswith('?'):
            score -= 0.1
            indicators.append("question_format")
        
        # Clamp score
        score = max(0.0, min(1.0, score))
        
        # Create assessment
        assessment = ComplexityAssessment(
            complexity_score=score,
            complexity_level="simple" if score < 0.3 else "medium" if score < 0.7 else "complex",
            reasoning="Rule-based fallback assessment",
            indicators=indicators,
            recommended_strategy="direct_response" if score < 0.3 else "agent_with_tools" if score < 0.7 else "goal_workflow",
            estimated_steps=1 if score < 0.3 else 3 if score < 0.7 else 5
        )
        
        return self._create_analysis_from_assessment(assessment, message)
    
    async def _track_assessment(self, message: str, analysis: ComplexityAnalysis, processing_time: float):
        """Track assessment for learning and analytics."""
        assessment_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "message_preview": message[:100],
            "complexity_score": analysis.score,
            "complexity_level": analysis.level.value,
            "strategy": analysis.strategy.value,
            "estimated_steps": analysis.estimated_steps,
            "requires_tools": analysis.requires_tools,
            "requires_multiple_agents": analysis.requires_multiple_agents,
            "estimated_cost": analysis.estimated_cost,
            "processing_time_ms": processing_time * 1000,
            "indicators": analysis.indicators
        }
        
        self.assessment_history.append(assessment_record)
        
        # Keep only recent assessments (last 1000)
        if len(self.assessment_history) > 1000:
            self.assessment_history = self.assessment_history[-1000:]
    
    def get_assessment_stats(self) -> Dict[str, Any]:
        """Get statistics about complexity assessments."""
        if not self.assessment_history:
            return {"total_assessments": 0}
        
        recent_assessments = self.assessment_history[-100:]  # Last 100
        
        # Calculate distribution
        simple_count = sum(1 for a in recent_assessments if a["complexity_level"] == "simple")
        medium_count = sum(1 for a in recent_assessments if a["complexity_level"] == "medium") 
        complex_count = sum(1 for a in recent_assessments if a["complexity_level"] == "complex")
        
        # Calculate averages
        avg_score = sum(a["complexity_score"] for a in recent_assessments) / len(recent_assessments)
        avg_processing_time = sum(a["processing_time_ms"] for a in recent_assessments) / len(recent_assessments)
        
        return {
            "total_assessments": len(self.assessment_history),
            "recent_assessments": len(recent_assessments),
            "complexity_distribution": {
                "simple": simple_count,
                "medium": medium_count,
                "complex": complex_count
            },
            "average_complexity_score": round(avg_score, 3),
            "average_processing_time_ms": round(avg_processing_time, 2),
            "most_common_strategy": max(
                ["direct_response", "agent_with_tools", "goal_workflow"],
                key=lambda s: sum(1 for a in recent_assessments if a["strategy"] == s)
            )
        }
    
    async def quick_complexity_check(self, message: str) -> Tuple[float, ComplexityLevel]:
        """
        Quick rule-based complexity check without LLM call.
        Useful for fast routing decisions.
        """
        message_lower = message.lower()
        
        # Quick simple checks
        if any(indicator in message_lower for indicator_list in self.simple_indicators.values() 
               for indicator in indicator_list):
            return 0.2, ComplexityLevel.SIMPLE
        
        # Quick complex checks
        if any(indicator in message_lower for indicator_list in self.complex_indicators.values()
               for indicator in indicator_list):
            return 0.8, ComplexityLevel.COMPLEX
        
        # Default to medium
        return 0.5, ComplexityLevel.MEDIUM 