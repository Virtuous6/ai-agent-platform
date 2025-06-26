"""
Filename: agent_performance_analyst.py
Purpose: Monitor and optimize individual agent performance through intelligent analysis
Dependencies: langchain, openai, asyncio, logging, typing, supabase

This module analyzes agent thinking patterns, resource usage, and performance metrics
to continuously optimize agent configurations and suggest improvements.
"""

import asyncio
import logging
import os
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import statistics

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class PerformanceIssueType(Enum):
    """Types of performance issues that can be identified."""
    HIGH_COST = "high_cost"
    SLOW_RESPONSE = "slow_response"
    LOW_SUCCESS_RATE = "low_success_rate"
    POOR_USER_SATISFACTION = "poor_user_satisfaction"
    INEFFICIENT_PROMPTS = "inefficient_prompts"
    REDUNDANT_CAPABILITIES = "redundant_capabilities"
    CAPABILITY_GAPS = "capability_gaps"
    SUBOPTIMAL_TEMPERATURE = "suboptimal_temperature"

class OptimizationStrategy(Enum):
    """Types of optimization strategies."""
    PROMPT_REFINEMENT = "prompt_refinement"
    TEMPERATURE_ADJUSTMENT = "temperature_adjustment"
    MODEL_CHANGE = "model_change"
    TOOL_OPTIMIZATION = "tool_optimization"
    AGENT_MERGE = "agent_merge"
    AGENT_SPLIT = "agent_split"
    CAPABILITY_ENHANCEMENT = "capability_enhancement"
    COST_REDUCTION = "cost_reduction"

@dataclass
class AgentPerformanceMetrics:
    """Comprehensive performance metrics for an agent."""
    agent_id: str
    specialty: str
    model_used: str
    
    # Usage metrics
    total_interactions: int
    successful_interactions: int
    failed_interactions: int
    
    # Performance metrics
    avg_response_time: float  # seconds
    avg_tokens_per_response: int
    avg_cost_per_interaction: float
    
    # Quality metrics
    success_rate: float
    user_satisfaction_score: float
    escalation_rate: float
    
    # Configuration metrics
    current_temperature: float
    current_max_tokens: int
    system_prompt_length: int
    
    # Efficiency metrics
    cost_efficiency_score: float  # value/cost ratio
    time_efficiency_score: float  # success/time ratio
    
    # Trend data (last 7 days)
    performance_trend: List[float]  # daily success rates
    cost_trend: List[float]  # daily costs
    
    created_at: datetime = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()

@dataclass
class PerformanceIssue:
    """Represents an identified performance issue."""
    id: str
    agent_id: str
    issue_type: PerformanceIssueType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    impact_analysis: Dict[str, float]  # {"cost_impact": 25.0, "performance_impact": 15.0}
    evidence: List[str]  # Supporting evidence
    recommended_actions: List[str]
    confidence: float
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class OptimizationRecommendation:
    """Represents an optimization recommendation."""
    id: str
    agent_id: str
    strategy: OptimizationStrategy
    title: str
    description: str
    expected_benefits: Dict[str, float]  # {"cost_reduction": 20.0, "performance_gain": 15.0}
    implementation_steps: List[str]
    risk_assessment: str
    priority: int  # 1-5, 5 being highest
    confidence: float
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class AgentAnalysis(BaseModel):
    """LLM response schema for agent performance analysis."""
    performance_score: float = Field(description="Overall performance score 0.0-1.0")
    efficiency_analysis: Dict[str, Any] = Field(description="Analysis of agent efficiency")
    issues_identified: List[Dict[str, Any]] = Field(description="Performance issues found")
    optimization_opportunities: List[Dict[str, Any]] = Field(description="Optimization recommendations")
    thinking_pattern_analysis: Dict[str, Any] = Field(description="Analysis of agent thinking patterns")
    configuration_suggestions: Dict[str, Any] = Field(description="Configuration optimization suggestions")

class ConfigurationOptimization(BaseModel):
    """LLM response schema for configuration optimization."""
    optimal_temperature: float = Field(description="Recommended temperature setting")
    optimal_max_tokens: int = Field(description="Recommended max tokens")
    prompt_improvements: List[str] = Field(description="Specific prompt improvements")
    model_recommendation: str = Field(description="Recommended model")
    expected_improvement: Dict[str, float] = Field(description="Expected improvement metrics")

class AgentPerformanceAnalyst:
    """
    Intelligent agent performance analyzer that monitors and optimizes individual agent performance.
    
    Tracks thinking patterns, resource usage, and performance metrics to provide
    continuous optimization recommendations and automatic improvements.
    """
    
    def __init__(self, db_logger=None, orchestrator=None):
        """
        Initialize the Agent Performance Analyst.
        
        Args:
            db_logger: Supabase logger for accessing performance data
            orchestrator: Agent orchestrator for applying optimizations
        """
        self.db_logger = db_logger
        self.orchestrator = orchestrator
        
        # Performance tracking
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self.performance_issues: Dict[str, PerformanceIssue] = {}
        self.optimization_recommendations: Dict[str, OptimizationRecommendation] = {}
        
        # Analysis tracking
        self.analysis_history: List[Dict[str, Any]] = []
        self.last_analysis: Optional[datetime] = None
        self.analysis_frequency = timedelta(hours=2)  # Analyze every 2 hours
        
        # Performance baselines
        self.performance_baselines = {
            "success_rate_threshold": 0.85,
            "response_time_threshold": 3.0,  # seconds
            "cost_per_interaction_threshold": 0.02,  # USD
            "user_satisfaction_threshold": 0.8,
            "token_efficiency_threshold": 0.75
        }
        
        # Initialize LLMs for different analysis types
        self.analysis_llm = ChatOpenAI(
            model="gpt-4-0125-preview",  # Advanced model for complex analysis
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2000,
        )
        
        self.optimization_llm = ChatOpenAI(
            model="gpt-4-0125-preview",  # Advanced model for optimization
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=1500,
        )
        
        self.config_llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",  # Faster model for config analysis
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=800,
        )
        
        # Create prompt templates
        self.analysis_prompt = self._create_analysis_prompt()
        self.optimization_prompt = self._create_optimization_prompt()
        self.config_prompt = self._create_config_optimization_prompt()
        
        # Create parsing chains
        self.analysis_parser = JsonOutputParser(pydantic_object=AgentAnalysis)
        self.config_parser = JsonOutputParser(pydantic_object=ConfigurationOptimization)
        
        self.analysis_chain = self.analysis_prompt | self.analysis_llm | self.analysis_parser
        self.optimization_chain = self.optimization_prompt | self.optimization_llm
        self.config_chain = self.config_prompt | self.config_llm | self.config_parser
        
        # Start periodic analysis task
        self.analysis_task = None
        self._start_periodic_analysis()
        
        logger.info("Agent Performance Analyst initialized with intelligent optimization") 

    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for comprehensive agent performance analysis."""
        
        system_template = """You are an expert AI agent performance analyst. Your role is to analyze 
individual agent performance metrics and identify optimization opportunities.

**Analysis Framework:**
1. **Performance Metrics**: Success rates, response times, cost efficiency
2. **Usage Patterns**: Interaction frequency, user satisfaction, escalation rates  
3. **Resource Utilization**: Token usage, cost per interaction, model efficiency
4. **Thinking Patterns**: Response quality, consistency, accuracy
5. **Configuration Efficiency**: Temperature, prompt effectiveness, tool usage

**Analysis Criteria:**
- Performance Score: Overall effectiveness (0.0-1.0)
- Efficiency Analysis: Resource usage vs. value delivered
- Issue Identification: Problems requiring attention
- Optimization Opportunities: Specific improvements possible
- Configuration Recommendations: Optimal settings

**Focus Areas:**
- Agents with declining performance trends
- High-cost, low-value interactions
- Inefficient prompt or temperature settings
- Redundant capabilities across agents
- Missing capabilities causing escalations

Return analysis in the specified JSON format with specific, actionable insights."""

        human_template = """Analyze performance for agent: {agent_id}

**Agent Configuration:**
- Specialty: {specialty}
- Model: {model_used}
- Temperature: {temperature}
- Max Tokens: {max_tokens}
- System Prompt Length: {prompt_length} characters

**Performance Metrics:**
- Total Interactions: {total_interactions}
- Success Rate: {success_rate}%
- Average Response Time: {avg_response_time}s
- Average Cost per Interaction: ${avg_cost}
- User Satisfaction: {user_satisfaction}/5.0
- Escalation Rate: {escalation_rate}%

**Efficiency Metrics:**
- Average Tokens per Response: {avg_tokens}
- Cost Efficiency Score: {cost_efficiency}
- Time Efficiency Score: {time_efficiency}

**Trend Data (Last 7 Days):**
- Performance Trend: {performance_trend}
- Cost Trend: {cost_trend}

**Recent Interactions Sample:**
{recent_interactions}

**Comparison with Similar Agents:**
{peer_comparison}

Provide comprehensive performance analysis with specific optimization recommendations."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_optimization_prompt(self) -> ChatPromptTemplate:
        """Create prompt for optimization strategy generation."""
        
        system_template = """You are an expert AI system optimizer specializing in agent performance improvement.

**Optimization Strategies:**
1. **Prompt Refinement**: Improve system prompts for better responses
2. **Temperature Adjustment**: Optimize creativity vs. consistency balance
3. **Model Change**: Recommend different models for cost or performance
4. **Tool Optimization**: Add, remove, or modify agent tools
5. **Agent Merge**: Combine redundant agents with similar capabilities
6. **Agent Split**: Split overloaded agents into focused specialists
7. **Capability Enhancement**: Add missing capabilities
8. **Cost Reduction**: Reduce operational costs without sacrificing quality

**Optimization Principles:**
- Maximize value delivery per dollar spent
- Maintain or improve user satisfaction
- Reduce response times while maintaining quality
- Eliminate redundancies and inefficiencies
- Enhance specialized capabilities

**Implementation Considerations:**
- Risk assessment for each optimization
- Expected benefits and potential drawbacks
- Implementation complexity and timeline
- User impact and rollback procedures

Provide specific, actionable optimization recommendations with clear implementation steps."""

        human_template = """Generate optimization strategy for agent: {agent_id}

**Current Issues Identified:**
{performance_issues}

**Agent Context:**
{agent_context}

**Performance Baseline Comparisons:**
{baseline_analysis}

**Resource Utilization Analysis:**
{resource_analysis}

**User Feedback Summary:**
{user_feedback}

Generate specific optimization recommendations with implementation details and expected outcomes."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_config_optimization_prompt(self) -> ChatPromptTemplate:
        """Create prompt for configuration parameter optimization."""
        
        system_template = """You are a specialist in AI agent configuration optimization.

**Configuration Parameters:**
- **Temperature**: Controls response creativity/consistency (0.0-2.0)
  - 0.0-0.3: Highly deterministic (technical, factual tasks)
  - 0.4-0.7: Balanced (general purpose, problem solving)
  - 0.8-1.2: Creative (brainstorming, content generation)
  - 1.3-2.0: Highly creative (artistic, experimental)

- **Max Tokens**: Controls response length
  - 50-200: Brief answers, classifications
  - 300-800: Standard responses, explanations
  - 1000-2000: Detailed analysis, tutorials
  - 2000+: Comprehensive reports, documentation

- **Model Selection**:
  - GPT-3.5-turbo: Fast, cost-effective for simple tasks
  - GPT-4: Higher accuracy for complex reasoning
  - GPT-4-turbo: Balanced performance and cost

**Optimization Goals:**
- Maximize task-specific performance
- Minimize unnecessary costs
- Optimize response quality for use case
- Balance speed and accuracy

Provide specific configuration recommendations based on agent specialty and performance data."""

        human_template = """Optimize configuration for agent: {agent_id}

**Current Configuration:**
- Temperature: {current_temperature}
- Max Tokens: {current_max_tokens}
- Model: {current_model}

**Agent Specialty:** {specialty}

**Performance Analysis:**
- Success rate with current config: {success_rate}%
- Average response quality score: {quality_score}
- Average cost per interaction: ${avg_cost}
- User satisfaction: {user_satisfaction}

**Task Type Analysis:**
{task_analysis}

**Performance Issues:**
{config_issues}

**Benchmark Comparison:**
{benchmark_data}

Recommend optimal configuration parameters with expected improvement metrics."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    async def analyze_agent_performance(self, agent_id: str, 
                                      force_analysis: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis for a specific agent.
        
        Args:
            agent_id: The agent to analyze
            force_analysis: Force analysis even if recently performed
            
        Returns:
            Comprehensive analysis results with optimization recommendations
        """
        try:
            logger.info(f"Starting performance analysis for agent: {agent_id}")
            
            # Check if analysis needed
            if not force_analysis and self._is_recent_analysis(agent_id):
                logger.info(f"Skipping analysis for {agent_id} - recently analyzed")
                return self._get_cached_analysis(agent_id)
            
            # Gather agent performance data
            metrics = await self._gather_agent_metrics(agent_id)
            if not metrics:
                logger.warning(f"No metrics found for agent: {agent_id}")
                return {"error": "No performance data available"}
            
            # Gather interaction samples
            interactions = await self._gather_recent_interactions(agent_id)
            
            # Perform LLM-based analysis
            analysis_result = await self._perform_agent_analysis(agent_id, metrics, interactions)
            
            # Process analysis results
            issues = await self._process_performance_issues(agent_id, analysis_result.get("issues_identified", []))
            recommendations = await self._process_optimization_recommendations(agent_id, analysis_result.get("optimization_opportunities", []))
            
            # Generate configuration optimizations
            config_optimizations = await self._generate_config_optimizations(agent_id, metrics)
            
            # Store analysis results
            analysis_summary = {
                "agent_id": agent_id,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "performance_score": analysis_result.get("performance_score", 0.0),
                "issues_found": len(issues),
                "recommendations_count": len(recommendations),
                "config_optimizations": config_optimizations,
                "analysis_details": analysis_result
            }
            
            # Save to database
            if self.db_logger:
                await self.db_logger.log_event("agent_performance_analysis", analysis_summary)
            
            # Update internal tracking
            self.analysis_history.append(analysis_summary)
            
            logger.info(f"Performance analysis completed for {agent_id}: Score {analysis_result.get('performance_score', 0.0):.2f}")
            
            return analysis_summary
            
        except Exception as e:
            logger.error(f"Error analyzing agent performance: {str(e)}")
            return {"error": str(e)}

    async def analyze_all_agents(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Analyze performance of all active agents.
        
        Args:
            days_back: Number of days of data to analyze
            
        Returns:
            Summary of all agent analyses with system-wide optimizations
        """
        try:
            logger.info("Starting system-wide agent performance analysis")
            
            # Get all active agents
            active_agents = await self._get_active_agents(days_back)
            
            if not active_agents:
                logger.info("No active agents found for analysis")
                return {"message": "No active agents to analyze"}
            
            # Analyze each agent
            agent_analyses = {}
            system_issues = []
            system_recommendations = []
            
            for agent_id in active_agents:
                try:
                    analysis = await self.analyze_agent_performance(agent_id)
                    agent_analyses[agent_id] = analysis
                    
                    # Aggregate system-level issues
                    if "analysis_details" in analysis:
                        details = analysis["analysis_details"]
                        system_issues.extend(details.get("issues_identified", []))
                        system_recommendations.extend(details.get("optimization_opportunities", []))
                        
                except Exception as e:
                    logger.error(f"Error analyzing agent {agent_id}: {str(e)}")
                    agent_analyses[agent_id] = {"error": str(e)}
            
            # Identify system-wide optimization opportunities
            system_optimizations = await self._identify_system_optimizations(agent_analyses)
            
            # Generate overall system health score
            system_health = self._calculate_system_health(agent_analyses)
            
            summary = {
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "agents_analyzed": len(agent_analyses),
                "system_health_score": system_health,
                "system_optimizations": system_optimizations,
                "agent_analyses": agent_analyses,
                "next_analysis": (datetime.utcnow() + self.analysis_frequency).isoformat()
            }
            
            # Log system analysis
            if self.db_logger:
                await self.db_logger.log_event("system_performance_analysis", {
                    "agents_count": len(agent_analyses),
                    "system_health": system_health,
                    "optimizations_found": len(system_optimizations)
                })
            
            logger.info(f"System analysis complete: {len(agent_analyses)} agents, health score: {system_health:.2f}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in system-wide analysis: {str(e)}")
            return {"error": str(e)}

    async def _gather_agent_metrics(self, agent_id: str) -> Optional[AgentPerformanceMetrics]:
        """Gather comprehensive performance metrics for an agent from Supabase."""
        try:
            if not self.db_logger:
                logger.warning("No database logger available for metrics gathering")
                return None
            
            # Query agent metrics from last 30 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            
            # Get agent performance data
            agent_data = await self._query_agent_data(agent_id, start_date, end_date)
            
            if not agent_data:
                return None
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(agent_data)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error gathering metrics for {agent_id}: {str(e)}")
            return None

    async def _query_agent_data(self, agent_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Query agent data from Supabase."""
        try:
            # Get messages for this agent
            messages_query = f"""
            SELECT 
                m.*,
                c.user_id,
                c.channel_id
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.agent_type = '{agent_id}'
            AND m.timestamp >= '{start_date.isoformat()}'
            AND m.timestamp <= '{end_date.isoformat()}'
            ORDER BY m.timestamp DESC
            """
            
            # Get agent registry data if available
            registry_data = None
            if self.orchestrator and hasattr(self.orchestrator, 'agent_registry'):
                registry_data = self.orchestrator.agent_registry.get(agent_id, {})
            
            # Get token usage data
            token_query = f"""
            SELECT 
                date,
                SUM(total_tokens) as daily_tokens,
                SUM(estimated_cost) as daily_cost,
                COUNT(*) as daily_interactions
            FROM token_usage_detailed
            WHERE agent_type = '{agent_id}'
            AND date >= '{start_date.date()}'
            AND date <= '{end_date.date()}'
            GROUP BY date
            ORDER BY date
            """
            
            # Simulate data for now (in real implementation, would query Supabase)
            # This would be replaced with actual Supabase queries
            agent_data = {
                "agent_id": agent_id,
                "registry_data": registry_data or {},
                "messages": [],  # Would contain actual message data
                "token_usage": [],  # Would contain actual token usage data
                "interactions_count": 50,  # Example data
                "success_count": 42,
                "failed_count": 8,
                "total_tokens": 25000,
                "total_cost": 1.25,
                "avg_response_time": 2.3,
                "user_satisfaction_scores": [4.2, 4.5, 3.8, 4.1, 4.3]
            }
            
            return agent_data
            
        except Exception as e:
            logger.error(f"Error querying agent data: {str(e)}")
            return {}

    def _calculate_performance_metrics(self, agent_data: Dict[str, Any]) -> AgentPerformanceMetrics:
        """Calculate comprehensive performance metrics from raw data."""
        try:
            agent_id = agent_data.get("agent_id", "unknown")
            registry_data = agent_data.get("registry_data", {})
            
            # Basic counts
            total_interactions = agent_data.get("interactions_count", 0)
            successful_interactions = agent_data.get("success_count", 0)
            failed_interactions = agent_data.get("failed_count", 0)
            
            # Performance calculations
            success_rate = (successful_interactions / total_interactions) if total_interactions > 0 else 0.0
            avg_response_time = agent_data.get("avg_response_time", 0.0)
            total_tokens = agent_data.get("total_tokens", 0)
            total_cost = agent_data.get("total_cost", 0.0)
            
            avg_tokens_per_response = int(total_tokens / total_interactions) if total_interactions > 0 else 0
            avg_cost_per_interaction = total_cost / total_interactions if total_interactions > 0 else 0.0
            
            # User satisfaction
            satisfaction_scores = agent_data.get("user_satisfaction_scores", [])
            user_satisfaction_score = statistics.mean(satisfaction_scores) if satisfaction_scores else 0.0
            
            # Efficiency scores
            cost_efficiency_score = min(1.0, (success_rate * 2.0) / (avg_cost_per_interaction * 100 + 0.1))
            time_efficiency_score = min(1.0, success_rate / (avg_response_time + 0.1))
            
            # Configuration data
            current_temperature = registry_data.get("temperature", 0.4)
            current_max_tokens = registry_data.get("max_tokens", 500)
            system_prompt = registry_data.get("system_prompt", "")
            system_prompt_length = len(system_prompt)
            
            # Generate trend data (last 7 days)
            performance_trend = [success_rate + (i * 0.02 - 0.06) for i in range(7)]  # Simulated trend
            cost_trend = [avg_cost_per_interaction + (i * 0.001 - 0.003) for i in range(7)]  # Simulated trend
            
            metrics = AgentPerformanceMetrics(
                agent_id=agent_id,
                specialty=registry_data.get("specialty", "Unknown"),
                model_used=registry_data.get("model", "gpt-3.5-turbo"),
                total_interactions=total_interactions,
                successful_interactions=successful_interactions,
                failed_interactions=failed_interactions,
                avg_response_time=avg_response_time,
                avg_tokens_per_response=avg_tokens_per_response,
                avg_cost_per_interaction=avg_cost_per_interaction,
                success_rate=success_rate,
                user_satisfaction_score=user_satisfaction_score,
                escalation_rate=0.05,  # 5% escalation rate (would be calculated from data)
                current_temperature=current_temperature,
                current_max_tokens=current_max_tokens,
                system_prompt_length=system_prompt_length,
                cost_efficiency_score=cost_efficiency_score,
                time_efficiency_score=time_efficiency_score,
                performance_trend=performance_trend,
                cost_trend=cost_trend
            )
            
            # Store metrics
            self.agent_metrics[agent_id] = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return None

    async def _gather_recent_interactions(self, agent_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Gather recent interactions for analysis."""
        try:
            # In real implementation, would query recent interactions from Supabase
            # For now, return simulated data
            interactions = [
                {
                    "timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                    "user_message": f"Sample user message {i}",
                    "agent_response": f"Sample agent response {i}",
                    "success": i % 4 != 0,  # 75% success rate
                    "response_time": 2.0 + (i * 0.1),
                    "tokens_used": 150 + (i * 10),
                    "cost": 0.02 + (i * 0.001)
                }
                for i in range(limit)
            ]
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error gathering recent interactions: {str(e)}")
            return []

    async def _perform_agent_analysis(self, agent_id: str, metrics: AgentPerformanceMetrics, 
                                    interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform LLM-based analysis of agent performance."""
        try:
            # Prepare analysis data
            analysis_data = {
                "agent_id": agent_id,
                "specialty": metrics.specialty,
                "model_used": metrics.model_used,
                "temperature": metrics.current_temperature,
                "max_tokens": metrics.current_max_tokens,
                "prompt_length": metrics.system_prompt_length,
                "total_interactions": metrics.total_interactions,
                "success_rate": metrics.success_rate * 100,
                "avg_response_time": metrics.avg_response_time,
                "avg_cost": metrics.avg_cost_per_interaction,
                "user_satisfaction": metrics.user_satisfaction_score,
                "escalation_rate": metrics.escalation_rate * 100,
                "avg_tokens": metrics.avg_tokens_per_response,
                "cost_efficiency": metrics.cost_efficiency_score,
                "time_efficiency": metrics.time_efficiency_score,
                "performance_trend": metrics.performance_trend,
                "cost_trend": metrics.cost_trend,
                "recent_interactions": json.dumps(interactions[:5], indent=2),
                "peer_comparison": await self._get_peer_comparison(metrics)
            }
            
            # Run LLM analysis
            analysis_result = await self.analysis_chain.ainvoke(analysis_data)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return {"error": str(e)}

    async def _get_peer_comparison(self, metrics: AgentPerformanceMetrics) -> str:
        """Generate peer comparison data for context."""
        try:
            # Find similar agents (same specialty or model)
            similar_agents = []
            for agent_id, agent_metrics in self.agent_metrics.items():
                if (agent_metrics.specialty == metrics.specialty or 
                    agent_metrics.model_used == metrics.model_used) and agent_metrics.agent_id != metrics.agent_id:
                    similar_agents.append(agent_metrics)
            
            if not similar_agents:
                return "No similar agents found for comparison."
            
            # Calculate peer averages
            peer_success_rate = statistics.mean([a.success_rate for a in similar_agents])
            peer_response_time = statistics.mean([a.avg_response_time for a in similar_agents])
            peer_cost = statistics.mean([a.avg_cost_per_interaction for a in similar_agents])
            
            comparison = f"""
Peer Comparison (vs {len(similar_agents)} similar agents):
- Success Rate: {metrics.success_rate:.2%} (Peer Avg: {peer_success_rate:.2%})
- Response Time: {metrics.avg_response_time:.1f}s (Peer Avg: {peer_response_time:.1f}s)
- Cost per Interaction: ${metrics.avg_cost_per_interaction:.3f} (Peer Avg: ${peer_cost:.3f})
            """
            
            return comparison.strip()
            
        except Exception as e:
            logger.error(f"Error generating peer comparison: {str(e)}")
            return "Error generating peer comparison."

    async def _process_performance_issues(self, agent_id: str, issues_data: List[Dict[str, Any]]) -> List[str]:
        """Process and store performance issues identified by LLM."""
        try:
            issue_ids = []
            
            for issue_data in issues_data:
                issue_id = str(uuid.uuid4())
                
                # Map issue type
                issue_type = PerformanceIssueType.HIGH_COST  # Default
                issue_type_str = issue_data.get("type", "").lower()
                for pit in PerformanceIssueType:
                    if pit.value in issue_type_str:
                        issue_type = pit
                        break
                
                issue = PerformanceIssue(
                    id=issue_id,
                    agent_id=agent_id,
                    issue_type=issue_type,
                    severity=issue_data.get("severity", "medium"),
                    description=issue_data.get("description", ""),
                    impact_analysis=issue_data.get("impact", {}),
                    evidence=issue_data.get("evidence", []),
                    recommended_actions=issue_data.get("recommendations", []),
                    confidence=issue_data.get("confidence", 0.5)
                )
                
                self.performance_issues[issue_id] = issue
                issue_ids.append(issue_id)
                
                logger.info(f"Identified performance issue for {agent_id}: {issue.description}")
            
            return issue_ids
            
        except Exception as e:
            logger.error(f"Error processing performance issues: {str(e)}")
            return []

    async def _process_optimization_recommendations(self, agent_id: str, 
                                                 recommendations_data: List[Dict[str, Any]]) -> List[str]:
        """Process and store optimization recommendations."""
        try:
            recommendation_ids = []
            
            for rec_data in recommendations_data:
                rec_id = str(uuid.uuid4())
                
                # Map strategy type
                strategy = OptimizationStrategy.PROMPT_REFINEMENT  # Default
                strategy_str = rec_data.get("strategy", "").lower()
                for os in OptimizationStrategy:
                    if os.value in strategy_str:
                        strategy = os
                        break
                
                recommendation = OptimizationRecommendation(
                    id=rec_id,
                    agent_id=agent_id,
                    strategy=strategy,
                    title=rec_data.get("title", ""),
                    description=rec_data.get("description", ""),
                    expected_benefits=rec_data.get("benefits", {}),
                    implementation_steps=rec_data.get("steps", []),
                    risk_assessment=rec_data.get("risk", "low"),
                    priority=rec_data.get("priority", 3),
                    confidence=rec_data.get("confidence", 0.5)
                )
                
                self.optimization_recommendations[rec_id] = recommendation
                recommendation_ids.append(rec_id)
                
                logger.info(f"Generated optimization recommendation for {agent_id}: {recommendation.title}")
            
            return recommendation_ids
            
        except Exception as e:
            logger.error(f"Error processing optimization recommendations: {str(e)}")
            return []

    async def _generate_config_optimizations(self, agent_id: str, 
                                           metrics: AgentPerformanceMetrics) -> Dict[str, Any]:
        """Generate configuration optimization recommendations."""
        try:
            # Prepare configuration analysis data
            config_data = {
                "agent_id": agent_id,
                "current_temperature": metrics.current_temperature,
                "current_max_tokens": metrics.current_max_tokens,
                "current_model": metrics.model_used,
                "specialty": metrics.specialty,
                "success_rate": metrics.success_rate * 100,
                "quality_score": metrics.user_satisfaction_score * 20,  # Convert to 0-100 scale
                "avg_cost": metrics.avg_cost_per_interaction,
                "user_satisfaction": metrics.user_satisfaction_score,
                "task_analysis": await self._analyze_task_types(agent_id),
                "config_issues": await self._identify_config_issues(metrics),
                "benchmark_data": await self._get_benchmark_data(metrics.specialty)
            }
            
            # Run configuration optimization
            config_optimization = await self.config_chain.ainvoke(config_data)
            
            return config_optimization
            
        except Exception as e:
            logger.error(f"Error generating config optimizations: {str(e)}")
            return {}

    async def _analyze_task_types(self, agent_id: str) -> str:
        """Analyze the types of tasks this agent typically handles."""
        # In real implementation, would analyze actual task patterns
        return f"Agent {agent_id} typically handles technical analysis and problem-solving tasks."

    async def _identify_config_issues(self, metrics: AgentPerformanceMetrics) -> str:
        """Identify configuration-related issues."""
        issues = []
        
        if metrics.avg_cost_per_interaction > self.performance_baselines["cost_per_interaction_threshold"]:
            issues.append("High cost per interaction")
        
        if metrics.avg_response_time > self.performance_baselines["response_time_threshold"]:
            issues.append("Slow response times")
        
        if metrics.success_rate < self.performance_baselines["success_rate_threshold"]:
            issues.append("Below-target success rate")
        
        return "; ".join(issues) if issues else "No major configuration issues identified"

    async def _get_benchmark_data(self, specialty: str) -> str:
        """Get benchmark data for similar agent specialties."""
        # In real implementation, would query actual benchmark data
        return f"Benchmark data for {specialty} agents: avg success rate 87%, avg cost $0.018"

    async def _get_active_agents(self, days_back: int = 7) -> List[str]:
        """Get list of active agents from the past specified days."""
        try:
            # In real implementation, would query from database
            # For now, return agents from orchestrator registry
            if self.orchestrator and hasattr(self.orchestrator, 'agent_registry'):
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                active_agents = []
                
                for agent_id, config in self.orchestrator.agent_registry.items():
                    last_used = config.get("last_used", datetime.utcnow())
                    if isinstance(last_used, str):
                        last_used = datetime.fromisoformat(last_used.replace('Z', '+00:00'))
                    
                    if last_used > cutoff_date:
                        active_agents.append(agent_id)
                
                return active_agents
            
            # Fallback: return agents we have metrics for
            return list(self.agent_metrics.keys())
            
        except Exception as e:
            logger.error(f"Error getting active agents: {str(e)}")
            return []

    async def _identify_system_optimizations(self, agent_analyses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify system-wide optimization opportunities."""
        try:
            optimizations = []
            
            # Analyze for agent merging opportunities
            merge_opportunities = await self._identify_merge_opportunities(agent_analyses)
            optimizations.extend(merge_opportunities)
            
            # Analyze for agent splitting opportunities
            split_opportunities = await self._identify_split_opportunities(agent_analyses)
            optimizations.extend(split_opportunities)
            
            # Analyze for cost reduction opportunities
            cost_optimizations = await self._identify_cost_optimizations(agent_analyses)
            optimizations.extend(cost_optimizations)
            
            # Analyze for capability gaps
            capability_gaps = await self._identify_capability_gaps(agent_analyses)
            optimizations.extend(capability_gaps)
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Error identifying system optimizations: {str(e)}")
            return []

    async def _identify_merge_opportunities(self, agent_analyses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify agents that could be merged due to similar capabilities."""
        try:
            merge_opportunities = []
            
            # Group agents by specialty and performance
            agent_groups = defaultdict(list)
            
            for agent_id, analysis in agent_analyses.items():
                if "error" not in analysis and "analysis_details" in analysis:
                    details = analysis["analysis_details"]
                    specialty = details.get("specialty", "unknown")
                    agent_groups[specialty].append((agent_id, analysis))
            
            # Look for groups with multiple similar agents
            for specialty, agents in agent_groups.items():
                if len(agents) >= 2:
                    # Check if agents have similar performance and low individual usage
                    similar_agents = []
                    for agent_id, analysis in agents:
                        if analysis.get("performance_score", 0) > 0.6:  # Decent performance
                            similar_agents.append(agent_id)
                    
                    if len(similar_agents) >= 2:
                        merge_opportunities.append({
                            "type": "agent_merge",
                            "title": f"Merge {specialty} specialists",
                            "description": f"Consider merging {len(similar_agents)} similar {specialty} agents",
                            "affected_agents": similar_agents,
                            "expected_benefit": "Reduced operational complexity and cost",
                            "priority": 3
                        })
            
            return merge_opportunities
            
        except Exception as e:
            logger.error(f"Error identifying merge opportunities: {str(e)}")
            return []

    async def _identify_split_opportunities(self, agent_analyses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify agents that should be split into multiple specialists."""
        try:
            split_opportunities = []
            
            for agent_id, analysis in agent_analyses.items():
                if "error" not in analysis and "analysis_details" in analysis:
                    details = analysis["analysis_details"]
                    
                    # Look for high-usage agents with diverse task patterns
                    if (analysis.get("performance_score", 0) < 0.7 and  # Lower performance
                        details.get("total_interactions", 0) > 100):  # High usage
                        
                        split_opportunities.append({
                            "type": "agent_split",
                            "title": f"Split overloaded agent {agent_id}",
                            "description": f"Agent {agent_id} may benefit from splitting into focused specialists",
                            "affected_agents": [agent_id],
                            "expected_benefit": "Improved specialization and performance",
                            "priority": 4
                        })
            
            return split_opportunities
            
        except Exception as e:
            logger.error(f"Error identifying split opportunities: {str(e)}")
            return []

    async def _identify_cost_optimizations(self, agent_analyses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify cost optimization opportunities across the system."""
        try:
            cost_optimizations = []
            
            # Calculate total system cost
            total_cost = 0
            high_cost_agents = []
            
            for agent_id, analysis in agent_analyses.items():
                if "error" not in analysis and "analysis_details" in analysis:
                    details = analysis["analysis_details"]
                    agent_cost = details.get("avg_cost", 0) * details.get("total_interactions", 0)
                    total_cost += agent_cost
                    
                    # Identify high-cost agents
                    if details.get("avg_cost", 0) > self.performance_baselines["cost_per_interaction_threshold"]:
                        high_cost_agents.append(agent_id)
            
            if high_cost_agents:
                cost_optimizations.append({
                    "type": "cost_reduction",
                    "title": "Optimize high-cost agents",
                    "description": f"Optimize {len(high_cost_agents)} agents with above-average costs",
                    "affected_agents": high_cost_agents,
                    "expected_benefit": f"Potential 20-30% cost reduction (${total_cost * 0.25:.2f})",
                    "priority": 5
                })
            
            return cost_optimizations
            
        except Exception as e:
            logger.error(f"Error identifying cost optimizations: {str(e)}")
            return []

    async def _identify_capability_gaps(self, agent_analyses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify missing capabilities in the agent ecosystem."""
        try:
            capability_gaps = []
            
            # In real implementation, would analyze escalation patterns and unmet requests
            # For now, provide example capability gaps
            
            example_gaps = [
                {
                    "type": "capability_gap",
                    "title": "Missing Data Visualization Specialist",
                    "description": "High demand for data visualization tasks without dedicated agent",
                    "affected_agents": [],
                    "expected_benefit": "Improved user satisfaction for data tasks",
                    "priority": 3
                },
                {
                    "type": "capability_gap",
                    "title": "Missing DevOps Automation Specialist",
                    "description": "DevOps requests often escalated or handled suboptimally",
                    "affected_agents": [],
                    "expected_benefit": "Better DevOps support and reduced escalations",
                    "priority": 4
                }
            ]
            
            return example_gaps
            
        except Exception as e:
            logger.error(f"Error identifying capability gaps: {str(e)}")
            return []

    def _calculate_system_health(self, agent_analyses: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        try:
            if not agent_analyses:
                return 0.0
            
            valid_scores = []
            for analysis in agent_analyses.values():
                if "error" not in analysis:
                    score = analysis.get("performance_score", 0.0)
                    valid_scores.append(score)
            
            if not valid_scores:
                return 0.0
            
            # Calculate weighted average (could be more sophisticated)
            system_health = statistics.mean(valid_scores)
            
            return round(system_health, 3)
            
        except Exception as e:
            logger.error(f"Error calculating system health: {str(e)}")
            return 0.0

    def _is_recent_analysis(self, agent_id: str) -> bool:
        """Check if agent was analyzed recently."""
        try:
            for analysis in reversed(self.analysis_history):
                if (analysis.get("agent_id") == agent_id and 
                    datetime.fromisoformat(analysis["analysis_timestamp"]) > 
                    datetime.utcnow() - self.analysis_frequency):
                    return True
            return False
        except Exception:
            return False

    def _get_cached_analysis(self, agent_id: str) -> Dict[str, Any]:
        """Get most recent cached analysis for agent."""
        try:
            for analysis in reversed(self.analysis_history):
                if analysis.get("agent_id") == agent_id:
                    return analysis
            return {"error": "No cached analysis found"}
        except Exception:
            return {"error": "Error retrieving cached analysis"}

    def _start_periodic_analysis(self):
        """Start the periodic analysis task."""
        if self.analysis_task is not None:
            return  # Already running
        
        async def analysis_loop():
            while True:
                try:
                    await asyncio.sleep(self.analysis_frequency.total_seconds())
                    logger.info("Starting periodic agent performance analysis")
                    await self.analyze_all_agents()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in periodic analysis: {str(e)}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retry
        
        self.analysis_task = asyncio.create_task(analysis_loop())
        logger.info("Started periodic agent performance analysis")

    async def apply_optimization(self, optimization_id: str, auto_apply: bool = False) -> Dict[str, Any]:
        """Apply a specific optimization recommendation."""
        try:
            recommendation = self.optimization_recommendations.get(optimization_id)
            if not recommendation:
                return {"error": "Optimization not found"}
            
            logger.info(f"Applying optimization: {recommendation.title}")
            
            # Implementation would depend on optimization type
            if recommendation.strategy == OptimizationStrategy.TEMPERATURE_ADJUSTMENT:
                result = await self._apply_temperature_optimization(recommendation)
            elif recommendation.strategy == OptimizationStrategy.PROMPT_REFINEMENT:
                result = await self._apply_prompt_optimization(recommendation)
            elif recommendation.strategy == OptimizationStrategy.MODEL_CHANGE:
                result = await self._apply_model_optimization(recommendation)
            else:
                result = {"status": "queued", "message": "Optimization queued for manual review"}
            
            # Log the application
            if self.db_logger:
                await self.db_logger.log_event("optimization_applied", {
                    "optimization_id": optimization_id,
                    "agent_id": recommendation.agent_id,
                    "strategy": recommendation.strategy.value,
                    "auto_applied": auto_apply,
                    "result": result
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying optimization: {str(e)}")
            return {"error": str(e)}

    async def _apply_temperature_optimization(self, recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """Apply temperature optimization to an agent."""
        try:
            # In real implementation, would update agent configuration
            logger.info(f"Temperature optimization applied to {recommendation.agent_id}")
            return {"status": "applied", "message": "Temperature updated"}
        except Exception as e:
            return {"error": str(e)}

    async def _apply_prompt_optimization(self, recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """Apply prompt optimization to an agent."""
        try:
            # In real implementation, would update agent prompt
            logger.info(f"Prompt optimization applied to {recommendation.agent_id}")
            return {"status": "applied", "message": "Prompt updated"}
        except Exception as e:
            return {"error": str(e)}

    async def _apply_model_optimization(self, recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """Apply model optimization to an agent."""
        try:
            # In real implementation, would update agent model
            logger.info(f"Model optimization applied to {recommendation.agent_id}")
            return {"status": "applied", "message": "Model updated"}
        except Exception as e:
            return {"error": str(e)}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of agent performance analysis."""
        try:
            summary = {
                "total_agents_analyzed": len(self.agent_metrics),
                "active_issues": len(self.performance_issues),
                "pending_recommendations": len(self.optimization_recommendations),
                "last_analysis": self.last_analysis.isoformat() if self.last_analysis else None,
                "next_analysis": (datetime.utcnow() + self.analysis_frequency).isoformat(),
                "system_health": self._calculate_system_health_from_metrics(),
                "top_performing_agents": self._get_top_performers(),
                "agents_needing_attention": self._get_underperformers()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            return {"error": str(e)}

    def _calculate_system_health_from_metrics(self) -> float:
        """Calculate system health from stored metrics."""
        if not self.agent_metrics:
            return 0.0
        
        scores = [metrics.success_rate * metrics.cost_efficiency_score * metrics.time_efficiency_score 
                 for metrics in self.agent_metrics.values()]
        
        return statistics.mean(scores) if scores else 0.0

    def _get_top_performers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing agents."""
        try:
            agents = []
            for metrics in self.agent_metrics.values():
                score = metrics.success_rate * metrics.cost_efficiency_score * metrics.time_efficiency_score
                agents.append({
                    "agent_id": metrics.agent_id,
                    "specialty": metrics.specialty,
                    "performance_score": score,
                    "success_rate": metrics.success_rate
                })
            
            return sorted(agents, key=lambda x: x["performance_score"], reverse=True)[:limit]
            
        except Exception:
            return []

    def _get_underperformers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get agents that need attention."""
        try:
            agents = []
            for metrics in self.agent_metrics.values():
                issues = []
                if metrics.success_rate < self.performance_baselines["success_rate_threshold"]:
                    issues.append("Low success rate")
                if metrics.avg_cost_per_interaction > self.performance_baselines["cost_per_interaction_threshold"]:
                    issues.append("High cost")
                if metrics.avg_response_time > self.performance_baselines["response_time_threshold"]:
                    issues.append("Slow response")
                
                if issues:
                    agents.append({
                        "agent_id": metrics.agent_id,
                        "specialty": metrics.specialty,
                        "issues": issues,
                        "success_rate": metrics.success_rate
                    })
            
            return sorted(agents, key=lambda x: x["success_rate"])[:limit]
            
        except Exception:
            return []

    async def close(self):
        """Clean up resources and stop periodic tasks."""
        try:
            if self.analysis_task:
                self.analysis_task.cancel()
                try:
                    await self.analysis_task
                except asyncio.CancelledError:
                    pass
                self.analysis_task = None
            
            logger.info("Agent Performance Analyst closed")
            
        except Exception as e:
            logger.error(f"Error closing Agent Performance Analyst: {str(e)}") 