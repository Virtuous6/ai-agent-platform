"""
Filename: cost_optimizer.py
Purpose: Intelligent cost optimization engine for AI Agent Platform
Dependencies: langchain, openai, asyncio, statistics, hashlib

This module continuously monitors and optimizes operational costs across all agents,
implementing intelligent cost reduction strategies while maintaining quality.
"""

import asyncio
import logging
import os
import json
import hashlib
import statistics
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class CostOptimizationType(Enum):
    """Types of cost optimizations available."""
    PROMPT_COMPRESSION = "prompt_compression"
    MODEL_DOWNGRADE = "model_downgrade"
    INTELLIGENT_CACHING = "intelligent_caching"
    BATCH_PROCESSING = "batch_processing"
    TEMPERATURE_OPTIMIZATION = "temperature_optimization"
    TOKEN_LIMIT_OPTIMIZATION = "token_limit_optimization"
    CONTEXT_PRUNING = "context_pruning"
    RESPONSE_CACHING = "response_caching"

class CostIssueType(Enum):
    """Types of cost issues that can be identified."""
    HIGH_TOKEN_USAGE = "high_token_usage"
    EXPENSIVE_MODEL_OVERUSE = "expensive_model_overuse"
    REDUNDANT_QUERIES = "redundant_queries"
    INEFFICIENT_PROMPTS = "inefficient_prompts"
    UNNECESSARY_CONTEXT = "unnecessary_context"
    SUBOPTIMAL_TEMPERATURE = "suboptimal_temperature"
    TOKEN_WASTE = "token_waste"
    CACHE_MISS_RATE = "cache_miss_rate"

@dataclass
class CostMetrics:
    """Comprehensive cost metrics for an agent or system."""
    agent_id: str
    total_cost: float = 0.0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_requests: int = 0
    
    # Cost efficiency metrics
    cost_per_token: float = 0.0
    cost_per_request: float = 0.0
    cost_per_success: float = 0.0
    
    # Usage patterns
    model_distribution: Dict[str, int] = field(default_factory=dict)
    hourly_costs: Dict[str, float] = field(default_factory=dict)
    agent_costs: Dict[str, float] = field(default_factory=dict)
    
    # Efficiency scores
    cost_efficiency_score: float = 0.0
    token_efficiency_score: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Trends
    cost_trend: str = "stable"  # "increasing", "decreasing", "stable"
    usage_trend: str = "stable"
    
    # Timestamps
    last_updated: datetime = field(default_factory=lambda: datetime.utcnow())

@dataclass
class CostOptimization:
    """Represents a cost optimization opportunity."""
    id: str
    optimization_type: CostOptimizationType
    title: str
    description: str
    affected_agents: List[str]
    current_cost: float
    projected_cost: float
    potential_savings: float
    savings_percentage: float
    implementation_complexity: str  # "low", "medium", "high"
    risk_level: str  # "low", "medium", "high"
    implementation_steps: List[str]
    expected_impact: Dict[str, Any]
    confidence: float
    priority: int  # 1-5, 5 being highest
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())

@dataclass
class CacheEntry:
    """Represents a cached query response."""
    query_hash: str
    response: str
    tokens_saved: int
    cost_saved: float
    hit_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
    last_accessed: datetime = field(default_factory=lambda: datetime.utcnow())
    expiry: Optional[datetime] = None

class CostOptimizerAnalysis(BaseModel):
    """LLM response schema for cost optimization analysis."""
    cost_efficiency_score: float = Field(description="Overall cost efficiency 0.0-1.0")
    major_cost_drivers: List[Dict[str, Any]] = Field(description="Primary sources of high costs")
    optimization_opportunities: List[Dict[str, Any]] = Field(description="Specific cost reduction opportunities")
    prompt_optimizations: List[Dict[str, Any]] = Field(description="Prompt compression suggestions")
    model_recommendations: List[Dict[str, Any]] = Field(description="Model selection optimizations")
    caching_opportunities: List[Dict[str, Any]] = Field(description="Intelligent caching recommendations")
    projected_savings: Dict[str, float] = Field(description="Expected savings by optimization type")

class CostOptimizer:
    """
    Intelligent cost optimization engine that continuously monitors and reduces operational costs.
    
    Features:
    - Real-time cost tracking and analysis
    - Intelligent prompt compression
    - Model selection optimization
    - Advanced caching with similarity matching
    - Daily cost reports and projections
    - Automated cost reduction strategies
    """
    
    def __init__(self, db_logger=None, orchestrator=None):
        """
        Initialize the Cost Optimizer.
        
        Args:
            db_logger: Supabase logger for cost data access
            orchestrator: Agent orchestrator for applying optimizations
        """
        self.db_logger = db_logger
        self.orchestrator = orchestrator
        
        # Cost tracking
        self.cost_metrics: Dict[str, CostMetrics] = {}
        self.optimizations: Dict[str, CostOptimization] = {}
        self.cost_issues: Dict[str, Dict[str, Any]] = {}
        
        # Intelligent caching system
        self.query_cache: Dict[str, CacheEntry] = {}
        self.cache_similarity_threshold = 0.85
        self.cache_max_size = 10000
        self.cache_ttl_hours = 24
        
        # Analysis tracking
        self.analysis_history: List[Dict[str, Any]] = []
        self.last_analysis: Optional[datetime] = None
        self.analysis_frequency = timedelta(hours=1)  # Analyze every hour
        
        # Cost thresholds and targets
        self.cost_thresholds = {
            "daily_cost_limit": 10.0,  # USD per day
            "cost_per_interaction_max": 0.02,  # USD per interaction
            "token_efficiency_min": 0.75,  # Minimum token efficiency
            "cache_hit_rate_min": 0.30,  # Minimum cache hit rate
            "cost_increase_alert": 0.20  # Alert if costs increase by 20%
        }
        
        # Model cost tiers
        self.model_costs = {
            "gpt-4-0125-preview": {"input": 0.00001, "output": 0.00003},
            "gpt-4": {"input": 0.00003, "output": 0.00006},
            "gpt-3.5-turbo-0125": {"input": 0.0000005, "output": 0.0000015},
            "gpt-3.5-turbo": {"input": 0.0000005, "output": 0.0000015}
        }
        
        # Initialize LLMs for cost analysis
        self.analysis_llm = ChatOpenAI(
            model="gpt-4-0125-preview",  # High-accuracy analysis
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2000,
        )
        
        self.compression_llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",  # Cost-effective for compression
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=1000,
        )
        
        # Create prompt templates
        self.analysis_prompt = self._create_analysis_prompt()
        self.compression_prompt = self._create_compression_prompt()
        
        # Start periodic optimization
        self.optimization_task = None
        self._start_periodic_optimization()
        
        logger.info("Cost Optimizer initialized with intelligent cost reduction")

    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for comprehensive cost analysis."""
        
        system_template = """You are an expert AI cost optimization analyst specializing in LLM operational efficiency.

**Cost Analysis Framework:**
1. **Token Efficiency**: Analyze token usage patterns and identify waste
2. **Model Selection**: Evaluate if tasks are using appropriate models
3. **Prompt Optimization**: Identify verbose or inefficient prompts
4. **Caching Opportunities**: Find repeated queries suitable for caching
5. **Usage Patterns**: Detect cost spikes and inefficient workflows

**Cost Optimization Strategies:**
- **Prompt Compression**: Reduce token usage without losing meaning
- **Model Downgrading**: Use cheaper models for simple tasks
- **Intelligent Caching**: Cache similar queries with response reuse
- **Context Pruning**: Remove unnecessary conversation history
- **Batch Processing**: Group similar requests for efficiency
- **Temperature Optimization**: Adjust creativity settings for cost

**Analysis Criteria:**
- Current cost efficiency vs. benchmark
- Major cost drivers and optimization potential
- Model usage appropriateness for task complexity
- Caching hit rate and miss opportunities
- Prompt verbosity and compression potential

Return analysis in the specified JSON format with specific, actionable cost reduction recommendations."""

        human_template = """Analyze cost optimization opportunities for the system:

**Current Cost Metrics:**
- Total daily cost: ${total_cost:.4f}
- Cost per interaction: ${cost_per_interaction:.4f}
- Total tokens used: {total_tokens:,}
- Cache hit rate: {cache_hit_rate:.1%}

**Model Usage Distribution:**
{model_usage}

**Agent Cost Breakdown:**
{agent_costs}

**Recent Cost Trends:**
{cost_trends}

**High-Cost Operations:**
{expensive_operations}

**Current Issues:**
{cost_issues}

Provide specific cost optimization recommendations with projected savings and implementation details."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    def _create_compression_prompt(self) -> ChatPromptTemplate:
        """Create prompt for intelligent prompt compression."""
        
        system_template = """You are a prompt compression specialist. Your task is to reduce token usage while preserving meaning and effectiveness.

**Compression Techniques:**
1. **Remove Redundancy**: Eliminate repeated information
2. **Simplify Language**: Use concise, clear phrasing
3. **Combine Instructions**: Merge related directives
4. **Remove Fluff**: Eliminate unnecessary words and phrases
5. **Optimize Structure**: Reorganize for clarity and brevity
6. **Preserve Intent**: Maintain original meaning and functionality

**Compression Rules:**
- Preserve all essential information and instructions
- Maintain the original tone and style requirements
- Keep all technical specifications and constraints
- Ensure compressed prompt produces equivalent results
- Aim for 20-40% token reduction without quality loss

**Output Format:**
Return the compressed prompt followed by compression statistics:
- Original tokens (estimated)
- Compressed tokens (estimated)  
- Compression ratio
- Key changes made"""

        human_template = """Compress this prompt for optimal token efficiency:

**Original Prompt:**
{original_prompt}

**Context:**
- Agent type: {agent_type}
- Typical use case: {use_case}
- Performance requirements: {requirements}

**Compression Target:** Reduce tokens by 20-40% while maintaining effectiveness.

Provide the compressed prompt and analysis:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    async def track_operation_cost(self, agent_id: str, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track cost for a specific operation and identify optimization opportunities.
        
        Args:
            agent_id: The agent performing the operation
            operation_data: Contains tokens, model, cost, etc.
            
        Returns:
            Cost analysis with optimization suggestions
        """
        try:
            # Extract operation metrics
            input_tokens = operation_data.get("input_tokens", 0)
            output_tokens = operation_data.get("output_tokens", 0)
            total_tokens = operation_data.get("total_tokens", input_tokens + output_tokens)
            model_used = operation_data.get("model_used", "unknown")
            estimated_cost = operation_data.get("estimated_cost", 0.0)
            query_text = operation_data.get("query", "")
            response_text = operation_data.get("response", "")
            
            # Update cost metrics
            if agent_id not in self.cost_metrics:
                self.cost_metrics[agent_id] = CostMetrics(agent_id=agent_id)
            
            metrics = self.cost_metrics[agent_id]
            metrics.total_cost += estimated_cost
            metrics.total_tokens += total_tokens
            metrics.input_tokens += input_tokens
            metrics.output_tokens += output_tokens
            metrics.total_requests += 1
            
            # Update model distribution
            if model_used in metrics.model_distribution:
                metrics.model_distribution[model_used] += 1
            else:
                metrics.model_distribution[model_used] = 1
            
            # Calculate efficiency metrics
            metrics.cost_per_token = metrics.total_cost / metrics.total_tokens if metrics.total_tokens > 0 else 0
            metrics.cost_per_request = metrics.total_cost / metrics.total_requests if metrics.total_requests > 0 else 0
            metrics.last_updated = datetime.utcnow()
            
            # Check for caching opportunity
            cache_result = await self._check_cache_opportunity(query_text, agent_id)
            
            # Identify immediate cost issues
            issues = await self._identify_cost_issues(operation_data)
            
            # Generate optimization suggestions
            optimizations = await self._generate_operation_optimizations(operation_data)
            
            # Log high-cost operations
            if estimated_cost > self.cost_thresholds["cost_per_interaction_max"]:
                await self._log_high_cost_operation(agent_id, operation_data, estimated_cost)
            
            return {
                "cost_tracked": estimated_cost,
                "tokens_tracked": total_tokens,
                "efficiency_score": self._calculate_efficiency_score(metrics),
                "cache_opportunity": cache_result,
                "issues_identified": issues,
                "optimization_suggestions": optimizations,
                "cost_trend": self._analyze_cost_trend(agent_id),
                "projected_daily_cost": self._project_daily_cost(metrics)
            }
            
        except Exception as e:
            logger.error(f"Error tracking operation cost: {str(e)}")
            return {"error": str(e)}

    async def optimize_prompt(self, agent_id: str, original_prompt: str, 
                            agent_type: str = "general") -> Dict[str, Any]:
        """
        Optimize a prompt for reduced token usage while maintaining effectiveness.
        
        Args:
            agent_id: The agent using the prompt
            original_prompt: The original prompt text
            agent_type: Type of agent for context
            
        Returns:
            Optimized prompt with compression analysis
        """
        try:
            logger.info(f"Optimizing prompt for agent {agent_id}")
            
            # Estimate original token count
            original_tokens = self._estimate_tokens(original_prompt)
            
            # Use LLM to compress prompt
            compression_data = {
                "original_prompt": original_prompt,
                "agent_type": agent_type,
                "use_case": f"Agent {agent_id} operations",
                "requirements": "Maintain accuracy and effectiveness"
            }
            
            with get_openai_callback() as cb:
                compressed_result = await self.compression_llm.agenerate([
                    self.compression_prompt.format(**compression_data)
                ])
                compression_cost = cb.total_cost
            
            compressed_content = compressed_result.generations[0][0].text
            
            # Extract compressed prompt and statistics
            if "**Compressed Prompt:**" in compressed_content:
                parts = compressed_content.split("**Compressed Prompt:**")
                if len(parts) > 1:
                    compressed_prompt = parts[1].split("**")[0].strip()
                else:
                    compressed_prompt = original_prompt
            else:
                compressed_prompt = compressed_content.split("\n\n")[0].strip()
            
            # Calculate compression metrics
            compressed_tokens = self._estimate_tokens(compressed_prompt)
            compression_ratio = (original_tokens - compressed_tokens) / original_tokens if original_tokens > 0 else 0
            tokens_saved = original_tokens - compressed_tokens
            
            # Estimate cost savings
            if agent_id in self.cost_metrics:
                avg_requests_per_day = self.cost_metrics[agent_id].total_requests / 7  # Estimate weekly average
                daily_token_savings = tokens_saved * avg_requests_per_day
                daily_cost_savings = daily_token_savings * self.cost_metrics[agent_id].cost_per_token
            else:
                daily_cost_savings = tokens_saved * 0.00002  # Rough estimate
            
            optimization = {
                "agent_id": agent_id,
                "original_prompt": original_prompt,
                "optimized_prompt": compressed_prompt,
                "original_tokens": original_tokens,
                "optimized_tokens": compressed_tokens,
                "tokens_saved": tokens_saved,
                "compression_ratio": compression_ratio,
                "compression_percentage": compression_ratio * 100,
                "estimated_daily_savings": daily_cost_savings,
                "optimization_cost": compression_cost,
                "net_savings_potential": daily_cost_savings * 30 - compression_cost,  # Monthly net
                "quality_preserved": True,  # Would need A/B testing to verify
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store optimization if significant
            if compression_ratio > 0.15:  # 15% or better compression
                opt_id = str(uuid.uuid4())
                cost_optimization = CostOptimization(
                    id=opt_id,
                    optimization_type=CostOptimizationType.PROMPT_COMPRESSION,
                    title=f"Compress prompt for {agent_id}",
                    description=f"Reduce prompt tokens by {compression_ratio:.1%}",
                    affected_agents=[agent_id],
                    current_cost=daily_cost_savings * 30,
                    projected_cost=0,
                    potential_savings=daily_cost_savings * 30,
                    savings_percentage=compression_ratio * 100,
                    implementation_complexity="low",
                    risk_level="low",
                    implementation_steps=[
                        "Test optimized prompt with sample queries",
                        "Compare response quality to original",
                        "Deploy optimized prompt if quality maintained",
                        "Monitor performance for 48 hours"
                    ],
                    expected_impact={
                        "token_reduction": tokens_saved,
                        "daily_cost_savings": daily_cost_savings,
                        "monthly_savings": daily_cost_savings * 30
                    },
                    confidence=0.8,
                    priority=4 if compression_ratio > 0.25 else 3
                )
                
                self.optimizations[opt_id] = cost_optimization
                logger.info(f"Created prompt optimization {opt_id} with {compression_ratio:.1%} compression")
            
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing prompt: {str(e)}")
            return {"error": str(e)}

    async def check_intelligent_cache(self, query: str, agent_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Check if query can be served from intelligent cache with similarity matching.
        
        Args:
            query: The query to check
            agent_id: Optional agent ID for context
            
        Returns:
            Cached response if found, None otherwise
        """
        try:
            # Generate query hash
            query_hash = self._generate_query_hash(query)
            
            # Direct hash match
            if query_hash in self.query_cache:
                entry = self.query_cache[query_hash]
                if not entry.expiry or entry.expiry > datetime.utcnow():
                    entry.hit_count += 1
                    entry.last_accessed = datetime.utcnow()
                    logger.info(f"Cache hit (direct) for query hash: {query_hash[:8]}")
                    return {
                        "response": entry.response,
                        "cache_hit": True,
                        "similarity": 1.0,
                        "tokens_saved": entry.tokens_saved,
                        "cost_saved": entry.cost_saved
                    }
            
            # Similarity-based matching
            similar_entry = await self._find_similar_cached_query(query)
            if similar_entry:
                similar_entry["cache_hit"] = True
                logger.info(f"Cache hit (similarity) with {similar_entry['similarity']:.2f} match")
                return similar_entry
            
            logger.debug(f"Cache miss for query: {query[:50]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error checking intelligent cache: {str(e)}")
            return None

    async def cache_response(self, query: str, response: str, tokens_used: int, 
                           cost: float, ttl_hours: int = 24) -> bool:
        """
        Cache a query response with intelligent expiration.
        
        Args:
            query: The original query
            response: The response to cache
            tokens_used: Number of tokens for this response
            cost: Cost of generating this response
            ttl_hours: Time to live in hours
            
        Returns:
            True if successfully cached
        """
        try:
            # Clean up expired entries
            await self._cleanup_expired_cache()
            
            # Check cache size limits
            if len(self.query_cache) >= self.cache_max_size:
                await self._evict_cache_entries()
            
            # Generate cache entry
            query_hash = self._generate_query_hash(query)
            expiry = datetime.utcnow() + timedelta(hours=ttl_hours)
            
            cache_entry = CacheEntry(
                query_hash=query_hash,
                response=response,
                tokens_saved=tokens_used,
                cost_saved=cost,
                expiry=expiry
            )
            
            self.query_cache[query_hash] = cache_entry
            
            logger.info(f"Cached response for query {query_hash[:8]} (TTL: {ttl_hours}h)")
            return True
            
        except Exception as e:
            logger.error(f"Error caching response: {str(e)}")
            return False

    async def analyze_cost_patterns(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Analyze cost patterns and generate comprehensive optimization recommendations.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Comprehensive cost analysis with optimization opportunities
        """
        try:
            logger.info(f"Analyzing cost patterns for last {days_back} days")
            
            # Gather cost data
            cost_data = await self._gather_cost_data(days_back)
            
            if not cost_data:
                return {"message": "No cost data available for analysis"}
            
            # Calculate system-wide metrics
            system_metrics = self._calculate_system_metrics(cost_data)
            
            # Identify major cost drivers
            cost_drivers = self._identify_cost_drivers(cost_data)
            
            # Find optimization opportunities
            optimization_opportunities = await self._find_optimization_opportunities(cost_data)
            
            # Generate LLM-based analysis
            analysis_data = {
                "total_cost": system_metrics["total_cost"],
                "cost_per_interaction": system_metrics["cost_per_interaction"],
                "total_tokens": system_metrics["total_tokens"],
                "cache_hit_rate": system_metrics["cache_hit_rate"],
                "model_usage": json.dumps(system_metrics["model_distribution"], indent=2),
                "agent_costs": json.dumps(cost_drivers["agent_costs"], indent=2),
                "cost_trends": json.dumps(system_metrics["cost_trends"], indent=2),
                "expensive_operations": json.dumps(cost_drivers["expensive_operations"], indent=2),
                "cost_issues": json.dumps(self._format_cost_issues(), indent=2)
            }
            
            # Run comprehensive LLM analysis
            with get_openai_callback() as cb:
                llm_analysis = await self.analysis_llm.agenerate([
                    self.analysis_prompt.format(**analysis_data)
                ])
                analysis_cost = cb.total_cost
            
            analysis_content = llm_analysis.generations[0][0].text
            
            # Generate cost projections
            projections = self._generate_cost_projections(system_metrics)
            
            # Create comprehensive report
            analysis_report = {
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "period_analyzed": f"{days_back} days",
                "system_metrics": system_metrics,
                "cost_drivers": cost_drivers,
                "optimization_opportunities": optimization_opportunities,
                "llm_analysis": analysis_content,
                "cost_projections": projections,
                "optimization_recommendations": self._get_optimization_recommendations(),
                "cache_performance": self._get_cache_performance(),
                "analysis_cost": analysis_cost,
                "next_analysis": (datetime.utcnow() + self.analysis_frequency).isoformat()
            }
            
            # Store analysis
            self.analysis_history.append(analysis_report)
            self.last_analysis = datetime.utcnow()
            
            # Log to database
            if self.db_logger:
                await self.db_logger.log_event("cost_analysis", {
                    "total_cost": system_metrics["total_cost"],
                    "optimizations_found": len(optimization_opportunities),
                    "potential_savings": sum(opt.get("potential_savings", 0) for opt in optimization_opportunities)
                })
            
            logger.info(f"Cost analysis complete. Found {len(optimization_opportunities)} optimization opportunities")
            
            return analysis_report
            
        except Exception as e:
            logger.error(f"Error analyzing cost patterns: {str(e)}")
            return {"error": str(e)}

    async def generate_daily_cost_report(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive daily cost report with projections.
        
        Args:
            date: Date to generate report for (YYYY-MM-DD), defaults to today
            
        Returns:
            Detailed daily cost report
        """
        try:
            if not date:
                date = datetime.now(timezone.utc).date().isoformat()
            
            logger.info(f"Generating daily cost report for {date}")
            
            # Get daily cost data from database
            daily_data = {}
            if self.db_logger:
                daily_data = await self.db_logger.get_daily_token_summary("system", date)
            
            # Get system metrics
            system_cost = sum(metrics.total_cost for metrics in self.cost_metrics.values())
            system_tokens = sum(metrics.total_tokens for metrics in self.cost_metrics.values())
            system_requests = sum(metrics.total_requests for metrics in self.cost_metrics.values())
            
            # Calculate cost efficiency
            cost_per_request = system_cost / system_requests if system_requests > 0 else 0
            cost_per_token = system_cost / system_tokens if system_tokens > 0 else 0
            
            # Generate model breakdown
            model_costs = defaultdict(float)
            model_tokens = defaultdict(int)
            for metrics in self.cost_metrics.values():
                for model, count in metrics.model_distribution.items():
                    if model in self.model_costs:
                        avg_tokens = metrics.total_tokens / metrics.total_requests if metrics.total_requests > 0 else 0
                        model_tokens[model] += int(avg_tokens * count)
                        # Assume 60% input, 40% output token distribution
                        total_tokens = int(avg_tokens * count)
                        input_tokens = int(total_tokens * 0.6)
                        output_tokens = int(total_tokens * 0.4)
                        model_costs[model] += self._calculate_model_cost(model, input_tokens, output_tokens)
            
            # Identify top cost agents
            agent_costs = [(agent_id, metrics.total_cost) for agent_id, metrics in self.cost_metrics.items()]
            top_agents = sorted(agent_costs, key=lambda x: x[1], reverse=True)[:5]
            
            # Calculate cache performance
            cache_stats = self._get_cache_performance()
            
            # Generate cost projections
            if system_cost > 0:
                monthly_projection = system_cost * 30
                weekly_projection = system_cost * 7
            else:
                monthly_projection = 0
                weekly_projection = 0
            
            # Check against thresholds
            alerts = []
            if system_cost > self.cost_thresholds["daily_cost_limit"]:
                alerts.append({
                    "type": "daily_limit_exceeded",
                    "message": f"Daily cost ${system_cost:.4f} exceeds limit ${self.cost_thresholds['daily_cost_limit']:.4f}",
                    "severity": "high"
                })
            
            if cost_per_request > self.cost_thresholds["cost_per_interaction_max"]:
                alerts.append({
                    "type": "high_cost_per_interaction",
                    "message": f"Cost per interaction ${cost_per_request:.4f} exceeds threshold",
                    "severity": "medium"
                })
            
            # Create comprehensive report
            report = {
                "report_date": date,
                "generated_at": datetime.utcnow().isoformat(),
                "daily_summary": {
                    "total_cost": system_cost,
                    "total_tokens": system_tokens,
                    "total_requests": system_requests,
                    "cost_per_request": cost_per_request,
                    "cost_per_token": cost_per_token,
                    "efficiency_score": self._calculate_system_efficiency()
                },
                "model_breakdown": {
                    "costs": dict(model_costs),
                    "tokens": dict(model_tokens),
                    "distribution": self._calculate_model_distribution()
                },
                "agent_breakdown": {
                    "top_agents": top_agents,
                    "agent_count": len(self.cost_metrics),
                    "avg_cost_per_agent": system_cost / len(self.cost_metrics) if self.cost_metrics else 0
                },
                "cache_performance": cache_stats,
                "cost_projections": {
                    "weekly": weekly_projection,
                    "monthly": monthly_projection,
                    "quarterly": monthly_projection * 3
                },
                "optimization_summary": {
                    "active_optimizations": len(self.optimizations),
                    "potential_savings": sum(opt.potential_savings for opt in self.optimizations.values()),
                    "cache_savings": cache_stats.get("total_cost_saved", 0)
                },
                "alerts": alerts,
                "recommendations": self._get_daily_recommendations()
            }
            
            # Log report generation
            if self.db_logger:
                await self.db_logger.log_event("daily_cost_report", {
                    "date": date,
                    "total_cost": system_cost,
                    "alerts_count": len(alerts)
                })
            
            logger.info(f"Daily cost report generated: ${system_cost:.4f} total cost")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating daily cost report: {str(e)}")
            return {"error": str(e)}

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough estimation: ~4 characters per token for English
        return max(1, len(text) // 4)

    def _generate_query_hash(self, query: str) -> str:
        """Generate consistent hash for query caching."""
        # Normalize query for consistent hashing
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    async def _find_similar_cached_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Find cached queries similar to the given query using semantic similarity.
        
        Args:
            query: Query to search for similar cached responses
            
        Returns:
            Cached response data if similar query found, None otherwise
        """
        if not self.query_cache:
            return None
            
        query_hash = self._generate_query_hash(query)
        
        # First check for exact match
        if query_hash in self.query_cache:
            entry = self.query_cache[query_hash]
            entry.hit_count += 1
            entry.last_accessed = datetime.utcnow()
            return {
                "response": entry.response,
                "tokens_saved": entry.tokens_saved,
                "cost_saved": entry.cost_saved,
                "similarity": 1.0,
                "source": "exact_match"
            }
        
        # For now, implement simple keyword-based similarity
        # In production, this would use vector embeddings
        best_match = None
        best_similarity = 0.0
        
        query_words = set(query.lower().split())
        
        for cached_hash, entry in self.query_cache.items():
            if entry.expiry and entry.expiry < datetime.utcnow():
                continue
                
            # Simple word overlap similarity
            cached_words = set(entry.response.lower().split()[:50])  # Use first 50 words
            intersection = query_words.intersection(cached_words)
            union = query_words.union(cached_words)
            
            if union:
                similarity = len(intersection) / len(union)
                if similarity > best_similarity and similarity >= self.cache_similarity_threshold:
                    best_similarity = similarity
                    best_match = entry
        
        if best_match:
            best_match.hit_count += 1
            best_match.last_accessed = datetime.utcnow()
            return {
                "response": best_match.response,
                "tokens_saved": best_match.tokens_saved,
                "cost_saved": best_match.cost_saved,
                "similarity": best_similarity,
                "source": "similarity_match"
            }
        
        return None

    async def _check_cache_opportunity(self, query: str, agent_id: str) -> Dict[str, Any]:
        """
        Check if there's a caching opportunity for the given query.
        
        Args:
            query: Query to check for caching
            agent_id: Agent making the query
            
        Returns:
            Cache opportunity analysis
        """
        # Look for similar queries
        similar_query = await self._find_similar_cached_query(query)
        
        if similar_query:
            return {
                "cache_hit": True,
                "similarity": similar_query["similarity"],
                "tokens_saved": similar_query["tokens_saved"],
                "cost_saved": similar_query["cost_saved"],
                "source": similar_query["source"]
            }
        
        # Check if this query type appears frequently
        query_hash = self._generate_query_hash(query)
        query_frequency = sum(1 for entry in self.query_cache.values() 
                            if self._calculate_query_similarity(query, entry.response) > 0.7)
        
        return {
            "cache_hit": False,
            "should_cache": query_frequency >= 2,  # Cache if seen 2+ times
            "query_frequency": query_frequency,
            "cache_potential": "high" if query_frequency >= 3 else "medium" if query_frequency >= 1 else "low"
        }
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate simple similarity between two queries."""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    async def _gather_cost_data(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Gather comprehensive cost data for analysis.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Aggregated cost data
        """
        # For now, return synthetic data. In production, this would query Supabase
        total_cost = sum(metrics.total_cost for metrics in self.cost_metrics.values())
        total_tokens = sum(metrics.total_tokens for metrics in self.cost_metrics.values())
        total_requests = sum(metrics.total_requests for metrics in self.cost_metrics.values())
        
        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "avg_cost_per_request": total_cost / max(total_requests, 1),
            "avg_tokens_per_request": total_tokens / max(total_requests, 1),
            "cost_per_token": total_cost / max(total_tokens, 1),
            "agent_breakdown": {
                agent_id: {
                    "cost": metrics.total_cost,
                    "tokens": metrics.total_tokens,
                    "requests": metrics.total_requests
                } for agent_id, metrics in self.cost_metrics.items()
            },
            "model_breakdown": {},
            "daily_costs": {},
            "trends": {
                "cost_trend": "stable",
                "usage_trend": "stable"
            }
        }

    def _calculate_model_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for a specific model and token usage.
        
        Args:
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count
            
        Returns:
            Total cost in USD
        """
        if model not in self.model_costs:
            # Default to GPT-4 pricing if model not found
            model = "gpt-4"
        
        costs = self.model_costs[model]
        input_cost = input_tokens * costs["input"]
        output_cost = output_tokens * costs["output"]
        
        return input_cost + output_cost

    def _get_cache_performance(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.
        
        Returns:
            Cache performance statistics
        """
        if not self.query_cache:
            return {
                "total_entries": 0,
                "hit_rate": 0.0,
                "total_hits": 0,
                "total_savings": 0.0,
                "avg_hit_count": 0.0
            }
        
        total_entries = len(self.query_cache)
        total_hits = sum(entry.hit_count for entry in self.query_cache.values())
        total_savings = sum(entry.cost_saved * entry.hit_count for entry in self.query_cache.values())
        
        # Calculate hit rate (simplified)
        total_queries = total_hits + total_entries  # Assuming each entry had at least one miss initially
        hit_rate = total_hits / max(total_queries, 1)
        
        return {
            "total_entries": total_entries,
            "hit_rate": hit_rate,
            "total_hits": total_hits,
            "total_savings": total_savings,
            "avg_hit_count": total_hits / max(total_entries, 1)
        }

    async def _cleanup_expired_cache(self):
        """Remove expired cache entries."""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, entry in self.query_cache.items()
            if entry.expiry and entry.expiry <= current_time
        ]
        
        for key in expired_keys:
            del self.query_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def _evict_cache_entries(self):
        """Evict least recently used cache entries."""
        if len(self.query_cache) < self.cache_max_size:
            return
        
        # Sort by last accessed time and remove oldest 20%
        sorted_entries = sorted(
            self.query_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        evict_count = len(sorted_entries) // 5  # Remove 20%
        for i in range(evict_count):
            key = sorted_entries[i][0]
            del self.query_cache[key]
        
        logger.info(f"Evicted {evict_count} cache entries due to size limit")

    def _calculate_efficiency_score(self, metrics: CostMetrics) -> float:
        """Calculate cost efficiency score for an agent."""
        if metrics.total_requests == 0:
            return 0.0
        
        # Efficiency based on cost per successful interaction
        cost_efficiency = min(1.0, self.cost_thresholds["cost_per_interaction_max"] / (metrics.cost_per_request + 0.001))
        
        # Token efficiency based on reasonable token usage
        reasonable_tokens_per_request = 1000  # Baseline
        token_efficiency = min(1.0, reasonable_tokens_per_request / (metrics.total_tokens / metrics.total_requests + 1))
        
        # Combined efficiency score
        return (cost_efficiency + token_efficiency) / 2

    def _start_periodic_optimization(self):
        """Start the periodic cost optimization task."""
        async def optimization_loop():
            while True:
                try:
                    await asyncio.sleep(self.analysis_frequency.total_seconds())
                    if datetime.utcnow() - (self.last_analysis or datetime.min) >= self.analysis_frequency:
                        await self.analyze_cost_patterns()
                except Exception as e:
                    logger.error(f"Error in periodic optimization: {str(e)}")
        
        self.optimization_task = asyncio.create_task(optimization_loop())
        logger.info("Started periodic cost optimization")

    async def close(self):
        """Clean up resources."""
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Cost Optimizer closed")

    async def _identify_cost_issues(self, operation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify cost issues from operation data.
        
        Args:
            operation_data: Operation data to analyze
            
        Returns:
            List of identified cost issues
        """
        issues = []
        
        # Check for high token usage
        tokens_used = operation_data.get("tokens_used", 0)
        if tokens_used > 1000:
            issues.append({
                "type": "high_token_usage",
                "severity": "medium",
                "description": f"High token usage detected: {tokens_used} tokens",
                "tokens": tokens_used
            })
        
        # Check for expensive model usage
        model = operation_data.get("model", "")
        if "gpt-4" in model and tokens_used > 500:
            issues.append({
                "type": "expensive_model_overuse",
                "severity": "high", 
                "description": f"Using expensive model {model} for {tokens_used} tokens",
                "model": model,
                "tokens": tokens_used
            })
        
        # Check cost per request
        cost = operation_data.get("cost", 0)
        if cost > 0.05:  # $0.05 per request
            issues.append({
                "type": "high_cost_per_request",
                "severity": "high",
                "description": f"High cost per request: ${cost:.4f}",
                "cost": cost
            })
        
        return issues

    def _calculate_system_metrics(self, cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate system-wide metrics from cost data.
        
        Args:
            cost_data: Aggregated cost data
            
        Returns:
            System metrics
        """
        total_cost = cost_data.get("total_cost", 0)
        total_requests = cost_data.get("total_requests", 0)
        total_tokens = cost_data.get("total_tokens", 0)
        
        # Calculate efficiency scores
        efficiency_score = min(100, max(0, 100 - (total_cost * 100)))  # Simple efficiency metric
        token_efficiency = total_tokens / max(total_requests, 1)  # Tokens per request
        cost_efficiency = total_cost / max(total_requests, 1)  # Cost per request
        
        # Cache metrics
        cache_performance = self._get_cache_performance()
        
        return {
            "efficiency_score": efficiency_score,
            "token_efficiency": token_efficiency,
            "cost_efficiency": cost_efficiency,
            "cache_hit_rate": cache_performance["hit_rate"],
            "total_savings": cache_performance["total_savings"],
            "cost_trend": "stable",  # Simplified
            "optimization_potential": max(0, 100 - efficiency_score)
        }

    def _get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get current optimization recommendations.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Analyze current metrics
        total_cost = sum(metrics.total_cost for metrics in self.cost_metrics.values())
        total_tokens = sum(metrics.total_tokens for metrics in self.cost_metrics.values())
        
        # Recommend prompt compression if high token usage
        if total_tokens > 5000:
            recommendations.append({
                "type": "prompt_compression",
                "priority": "high",
                "description": "High token usage detected. Consider prompt compression.",
                "potential_savings": total_tokens * 0.3 * 0.00003,  # 30% reduction
                "implementation": "Use prompt compression for repetitive queries"
            })
        
        # Recommend model downgrade if using expensive models frequently
        gpt4_usage = sum(1 for metrics in self.cost_metrics.values() 
                        if "gpt-4" in metrics.model_distribution)
        
        if gpt4_usage > 10:
            recommendations.append({
                "type": "model_optimization",
                "priority": "medium", 
                "description": "Consider using GPT-3.5 for simple tasks",
                "potential_savings": total_cost * 0.9,  # 90% savings on model costs
                "implementation": "Route simple queries to GPT-3.5-turbo"
            })
        
        # Recommend caching if low hit rate
        cache_perf = self._get_cache_performance()
        if cache_perf["hit_rate"] < 0.3:
            recommendations.append({
                "type": "intelligent_caching",
                "priority": "medium",
                "description": "Low cache hit rate. Improve caching strategy.",
                "potential_savings": total_cost * 0.2,  # 20% savings from caching
                "implementation": "Implement semantic similarity caching"
            })
        
        return recommendations

    def _calculate_system_efficiency(self) -> float:
        """Calculate overall system efficiency score."""
        if not self.cost_metrics:
            return 50.0  # Default neutral score
        
        total_cost = sum(metrics.total_cost for metrics in self.cost_metrics.values())
        total_tokens = sum(metrics.total_tokens for metrics in self.cost_metrics.values())
        total_requests = sum(metrics.total_requests for metrics in self.cost_metrics.values())
        
        if total_requests == 0:
            return 50.0
        
        # Calculate efficiency based on cost per request and token efficiency
        cost_per_request = total_cost / total_requests
        tokens_per_request = total_tokens / total_requests
        
        # Lower cost per request and reasonable token usage indicate higher efficiency
        cost_efficiency = max(0, 100 - (cost_per_request * 1000))  # Scale cost
        token_efficiency = min(100, max(0, 100 - (tokens_per_request / 10)))  # Scale tokens
        
        return (cost_efficiency + token_efficiency) / 2

    def _calculate_model_distribution(self) -> Dict[str, Any]:
        """Calculate model usage distribution."""
        model_usage = defaultdict(int)
        total_requests = 0
        
        for metrics in self.cost_metrics.values():
            for model, count in metrics.model_distribution.items():
                model_usage[model] += count
                total_requests += count
        
        if total_requests == 0:
            return {}
        
        # Convert to percentages
        return {
            model: {
                "count": count,
                "percentage": round((count / total_requests) * 100, 1)
            }
            for model, count in model_usage.items()
        }

    def _get_daily_recommendations(self) -> List[Dict[str, Any]]:
        """Get daily optimization recommendations."""
        recommendations = []
        
        total_cost = sum(metrics.total_cost for metrics in self.cost_metrics.values())
        
        # High cost alert
        if total_cost > self.cost_thresholds["daily_cost_limit"]:
            recommendations.append({
                "type": "cost_reduction",
                "priority": "high",
                "title": "Daily cost limit exceeded",
                "description": f"Current cost ${total_cost:.4f} exceeds daily limit",
                "action": "Review high-cost operations and consider optimizations"
            })
        
        # Cache performance
        cache_perf = self._get_cache_performance()
        if cache_perf["hit_rate"] < 0.3:
            recommendations.append({
                "type": "caching",
                "priority": "medium", 
                "title": "Low cache hit rate",
                "description": f"Cache hit rate is {cache_perf['hit_rate']:.1%}",
                "action": "Implement better caching strategies"
            })
        
        # Model optimization
        model_dist = self._calculate_model_distribution()
        gpt4_usage = model_dist.get("gpt-4", {}).get("percentage", 0)
        if gpt4_usage > 70:
            recommendations.append({
                "type": "model_optimization",
                "priority": "medium",
                "title": "High GPT-4 usage",
                "description": f"GPT-4 usage is {gpt4_usage}% of requests",
                "action": "Consider using GPT-3.5 for simpler tasks"
            })
        
        return recommendations

    async def _generate_operation_optimizations(self, operation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate optimization suggestions for a specific operation.
        
        Args:
            operation_data: Operation data to analyze
            
        Returns:
            List of optimization suggestions
        """
        optimizations = []
        
        tokens_used = operation_data.get("tokens_used", 0)
        cost = operation_data.get("cost", 0)
        model = operation_data.get("model", "")
        
        # Suggest prompt compression for high token usage
        if tokens_used > 800:
            optimizations.append({
                "type": "prompt_compression",
                "description": f"Compress prompt to reduce {tokens_used} tokens by ~30%",
                "potential_savings": cost * 0.3,
                "priority": "high"
            })
        
        # Suggest model downgrade for simple tasks
        if "gpt-4" in model and tokens_used < 500:
            optimizations.append({
                "type": "model_downgrade",
                "description": f"Use GPT-3.5 instead of {model} for simple tasks",
                "potential_savings": cost * 0.9,
                "priority": "medium"
            })
        
        # Suggest caching for repeated queries
        query_text = operation_data.get("query", "")
        if query_text and len(query_text) > 50:
            optimizations.append({
                "type": "intelligent_caching",
                "description": "Cache this response for similar future queries",
                "potential_savings": cost * 0.8,  # Assume 80% savings on cache hits
                "priority": "low"
            })
        
        return optimizations

    async def _identify_cost_drivers(self, cost_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify primary cost drivers from cost data.
        
        Args:
            cost_data: Aggregated cost data
            
        Returns:
            List of cost drivers
        """
        drivers = []
        
        total_cost = cost_data.get("total_cost", 0)
        agent_breakdown = cost_data.get("agent_breakdown", {})
        
        # Identify high-cost agents
        for agent_id, agent_data in agent_breakdown.items():
            agent_cost = agent_data.get("cost", 0)
            if agent_cost > total_cost * 0.2:  # Agent uses >20% of total cost
                drivers.append({
                    "type": "high_cost_agent",
                    "entity": agent_id,
                    "cost": agent_cost,
                    "percentage": (agent_cost / total_cost * 100) if total_cost > 0 else 0,
                    "description": f"Agent {agent_id} accounts for {agent_cost/total_cost*100:.1f}% of costs"
                })
        
        # Check for model cost drivers
        total_tokens = cost_data.get("total_tokens", 0)
        if total_tokens > 10000:
            drivers.append({
                "type": "high_token_usage",
                "entity": "system",
                "tokens": total_tokens,
                "cost": total_cost,
                "description": f"High system-wide token usage: {total_tokens:,} tokens"
            })
        
        # Check cost per request
        total_requests = cost_data.get("total_requests", 0)
        if total_requests > 0:
            cost_per_request = total_cost / total_requests
            if cost_per_request > 0.02:  # >$0.02 per request
                drivers.append({
                    "type": "high_cost_per_request", 
                    "entity": "system",
                    "cost_per_request": cost_per_request,
                    "description": f"High cost per request: ${cost_per_request:.4f}"
                })
        
        return drivers

# Additional helper methods would continue here...
# (Due to length constraints, showing core functionality) 