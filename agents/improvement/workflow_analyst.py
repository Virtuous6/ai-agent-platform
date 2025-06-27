"""
Filename: workflow_analyst.py
Purpose: Intelligent workflow analysis engine for continuous improvement
Dependencies: langchain, openai, asyncio, logging, typing, supabase

This module analyzes completed workflows to extract patterns, identify optimizations,
and automatically create reusable runbook templates for improved efficiency.
"""

import asyncio
import logging
import os
import json
import uuid
import yaml
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of workflow analysis to perform."""
    PATTERN_EXTRACTION = "pattern_extraction"
    BOTTLENECK_IDENTIFICATION = "bottleneck_identification"
    SUCCESS_ANALYSIS = "success_analysis"
    OPTIMIZATION_DISCOVERY = "optimization_discovery"
    AGENT_GAP_ANALYSIS = "agent_gap_analysis"

class PatternStrength(Enum):
    """Strength/confidence levels for discovered patterns."""
    WEAK = "weak"           # 2-3 occurrences
    MODERATE = "moderate"   # 4-7 occurrences  
    STRONG = "strong"       # 8-15 occurrences
    VERY_STRONG = "very_strong"  # 16+ occurrences

class OptimizationType(Enum):
    """Types of optimizations that can be identified."""
    AGENT_REUSE = "agent_reuse"
    PARALLEL_EXECUTION = "parallel_execution"
    PROMPT_OPTIMIZATION = "prompt_optimization"
    COST_REDUCTION = "cost_reduction"
    RESPONSE_TIME = "response_time"
    WORKFLOW_AUTOMATION = "workflow_automation"

@dataclass
class WorkflowPattern:
    """Represents a discovered workflow pattern."""
    id: str
    name: str
    description: str
    trigger_keywords: List[str]
    typical_steps: List[str]
    required_agents: List[str]
    success_rate: float
    avg_duration: float  # minutes
    avg_cost: float  # USD
    frequency: int  # how often this pattern occurs
    strength: PatternStrength
    example_runs: List[str]  # workflow run IDs
    confidence: float  # 0.0-1.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class OptimizationOpportunity:
    """Represents an identified optimization opportunity."""
    id: str
    type: OptimizationType
    title: str
    description: str
    potential_benefit: str
    estimated_savings: Dict[str, float]  # {"cost": 0.25, "time": 5.0}
    affected_patterns: List[str]  # pattern IDs
    implementation_complexity: str  # "low", "medium", "high"
    priority: int  # 1-5, 5 being highest
    action_required: str
    confidence: float
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class AgentGap:
    """Represents an identified gap in agent capabilities."""
    id: str
    capability_gap: str
    frequency: int
    typical_requests: List[str]
    suggested_agent_type: str
    estimated_demand: Dict[str, float]  # {"requests_per_day": 15, "cost_per_month": 45}
    priority: int
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class WorkflowAnalysis(BaseModel):
    """LLM response schema for workflow analysis."""
    patterns_found: List[Dict[str, Any]] = Field(description="Discovered workflow patterns")
    bottlenecks: List[Dict[str, Any]] = Field(description="Identified bottlenecks")
    optimizations: List[Dict[str, Any]] = Field(description="Optimization opportunities")
    success_factors: List[str] = Field(description="Factors contributing to success")
    failure_factors: List[str] = Field(description="Common failure patterns")
    agent_gaps: List[Dict[str, Any]] = Field(description="Missing agent capabilities")

class RunbookGeneration(BaseModel):
    """LLM response schema for runbook generation."""
    runbook_name: str = Field(description="Name for the generated runbook")
    triggers: List[Dict[str, Any]] = Field(description="Trigger conditions")
    steps: List[Dict[str, Any]] = Field(description="Workflow steps")
    configuration: Dict[str, Any] = Field(description="Runbook configuration")
    success_criteria: List[str] = Field(description="Success criteria")

class WorkflowAnalyst:
    """
    Intelligent workflow analysis engine that learns from completed workflows
    to extract patterns, identify bottlenecks, and create optimization opportunities.
    """
    
    def __init__(self, db_logger=None, orchestrator=None):
        """
        Initialize the Workflow Analyst with analysis capabilities.
        
        Args:
            db_logger: Supabase logger for accessing workflow data
            orchestrator: Agent orchestrator for triggering specialist creation
        """
        self.db_logger = db_logger
        self.orchestrator = orchestrator
        
        # Pattern and optimization storage
        self.discovered_patterns: Dict[str, WorkflowPattern] = {}
        self.optimization_opportunities: Dict[str, OptimizationOpportunity] = {}
        self.agent_gaps: Dict[str, AgentGap] = {}
        
        # Analysis tracking
        self.analysis_history: List[Dict[str, Any]] = []
        self.last_analysis: Optional[datetime] = None
        self.analysis_frequency = timedelta(hours=6)  # Analyze every 6 hours
        
        # Initialize LLMs for different analysis types
        self.analysis_llm = ChatOpenAI(
            model="gpt-4-0125-preview",  # Advanced model for complex analysis
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=3000,
        )
        
        self.runbook_llm = ChatOpenAI(
            model="gpt-4-0125-preview",  # Advanced model for runbook generation
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2000,
        )
        
        self.optimization_llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",  # Faster model for optimization suggestions
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=1000,
        )
        
        # Create prompt templates
        self.analysis_prompt = self._create_analysis_prompt()
        self.runbook_prompt = self._create_runbook_prompt()
        self.optimization_prompt = self._create_optimization_prompt()
        
        # Create parsing chains
        self.analysis_parser = JsonOutputParser(pydantic_object=WorkflowAnalysis)
        self.runbook_parser = JsonOutputParser(pydantic_object=RunbookGeneration)
        
        self.analysis_chain = self.analysis_prompt | self.analysis_llm | self.analysis_parser
        self.runbook_chain = self.runbook_prompt | self.runbook_llm | self.runbook_parser
        
        # Start periodic analysis task
        self.analysis_task = None
        self._start_periodic_analysis()
        
        logger.info("Workflow Analyst initialized with intelligent pattern recognition")
    
    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for comprehensive workflow analysis."""
        
        system_template = """You are an expert workflow analyst for an AI Agent Platform.
Your role is to analyze completed workflows to discover patterns, identify bottlenecks, 
and suggest optimizations for continuous improvement.

**Analysis Framework:**

1. **Pattern Recognition:**
   - Identify recurring workflow sequences
   - Extract common trigger patterns
   - Find successful execution paths
   - Note resource usage patterns

2. **Bottleneck Identification:**
   - Find slow steps in workflows
   - Identify resource constraints
   - Spot agent assignment inefficiencies
   - Locate failure points

3. **Success Analysis:**
   - Determine what makes workflows successful
   - Identify optimal agent combinations
   - Find efficient execution patterns
   - Note user satisfaction factors

4. **Agent Gap Analysis:**
   - Identify requests that don't match existing agents
   - Find capability gaps in current agent roster
   - Suggest new specialist agent types
   - Estimate demand for missing capabilities

**Analysis Data Context:**
- Total workflows analyzed: {total_workflows}
- Time period: {time_period}
- Agent types available: {available_agents}
- Average success rate: {success_rate}

**Output Requirements:**
Always return valid JSON matching the required schema."""

        human_template = """Analyze the following workflow data:

**Completed Workflows:**
{workflow_data}

**Agent Performance Metrics:**
{agent_metrics}

**Recent Patterns:**
{existing_patterns}

Please provide comprehensive analysis including:
1. New patterns discovered
2. Bottlenecks identified  
3. Optimization opportunities
4. Success/failure factors
5. Agent capability gaps

Focus on actionable insights that can improve system performance."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_runbook_prompt(self) -> ChatPromptTemplate:
        """Create prompt for automatic runbook generation."""
        
        system_template = """You are a runbook generation specialist for an AI Agent Platform.
Your role is to create automated workflow runbooks from discovered patterns.

**Runbook Generation Principles:**
1. **Clear Triggers:** Define when the runbook should execute
2. **Step-by-Step Logic:** Break workflow into discrete steps
3. **Error Handling:** Include retry and fallback strategies
4. **Resource Management:** Specify agent requirements and limits
5. **Success Criteria:** Define measurable outcomes

**Available Step Actions:**
- validate_message: Validate user input
- route_to_agent: Assign to specific agent
- spawn_specialist: Create new specialist agent
- parallel_execution: Run multiple steps simultaneously
- conditional_logic: Branch based on conditions
- format_output: Format response for user

**Runbook Template Structure:**
```yaml
metadata:
  name: "pattern-based-runbook"
  description: "Generated from workflow pattern"
  
triggers:
  - condition: "message_contains"
    parameters:
      keywords: ["keyword1", "keyword2"]
      
steps:
  - id: "step1"
    action: "validate_message"
    description: "Step description"
    parameters: {{}}
    
configuration:
  timeout_seconds: 60
  max_retries: 3
```

Pattern Data: {pattern_data}
Success Examples: {success_examples}"""

        human_template = """Generate a runbook for this workflow pattern:

**Pattern Name:** {pattern_name}
**Description:** {pattern_description}
**Trigger Keywords:** {trigger_keywords}
**Typical Steps:** {typical_steps}
**Required Agents:** {required_agents}
**Success Rate:** {success_rate}%
**Average Duration:** {avg_duration} minutes

**Example Successful Workflows:**
{example_workflows}

Create a comprehensive runbook that automates this pattern:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_optimization_prompt(self) -> ChatPromptTemplate:
        """Create prompt for optimization opportunity identification."""
        
        system_template = """You are an optimization specialist for an AI Agent Platform.
Your role is to identify specific optimization opportunities from workflow analysis.

**Optimization Categories:**
1. **Agent Reuse:** Reduce agent switching overhead
2. **Parallel Execution:** Run independent tasks simultaneously  
3. **Prompt Optimization:** Improve prompt efficiency
4. **Cost Reduction:** Minimize token usage and model costs
5. **Response Time:** Reduce overall workflow duration
6. **Workflow Automation:** Convert manual patterns to automated runbooks

**Analysis Framework:**
- Identify high-frequency, high-cost patterns
- Find redundant or inefficient steps
- Spot opportunities for parallelization
- Detect agent assignment suboptimalities
- Calculate potential savings (time, cost, resources)

**Optimization Prioritization:**
- High frequency + High impact = Priority 5
- High frequency + Medium impact = Priority 4  
- Medium frequency + High impact = Priority 3
- Medium frequency + Medium impact = Priority 2
- Low frequency OR Low impact = Priority 1

Current Metrics: {current_metrics}
Resource Constraints: {resource_constraints}"""

        human_template = """Identify optimization opportunities from this data:

**Workflow Patterns:**
{patterns}

**Performance Bottlenecks:**
{bottlenecks}

**Resource Usage:**
{resource_usage}

**Current Pain Points:**
{pain_points}

Provide specific, actionable optimization recommendations with:
1. Clear benefit estimation
2. Implementation complexity assessment  
3. Priority ranking
4. Required actions"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    async def analyze_workflows(self, days_back: int = 7, 
                              force_analysis: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive workflow analysis on recent data.
        
        Args:
            days_back: Number of days of data to analyze
            force_analysis: Force analysis even if recently performed
            
        Returns:
            Analysis results with patterns, optimizations, and recommendations
        """
        try:
            # Check if analysis is needed
            if not force_analysis and self.last_analysis:
                time_since_last = datetime.utcnow() - self.last_analysis
                if time_since_last < self.analysis_frequency:
                    logger.info(f"Analysis performed {time_since_last} ago, skipping")
                    return {"status": "skipped", "reason": "recent_analysis"}
            
            logger.info(f"Starting workflow analysis for last {days_back} days...")
            
            # Gather data for analysis
            workflow_data = await self._gather_workflow_data(days_back)
            agent_metrics = await self._gather_agent_metrics(days_back)
            
            if not workflow_data:
                logger.warning("No workflow data found for analysis")
                return {"status": "no_data", "reason": "insufficient_data"}
            
            # Perform LLM-powered analysis
            analysis_result = await self._perform_llm_analysis(workflow_data, agent_metrics)
            
            # Process and store results
            patterns_found = await self._process_patterns(analysis_result.get("patterns_found", []))
            optimizations_found = await self._process_optimizations(analysis_result.get("optimizations", []))
            gaps_found = await self._process_agent_gaps(analysis_result.get("agent_gaps", []))
            
            # Generate runbooks for strong patterns
            runbooks_created = await self._generate_runbooks_for_patterns()
            
            # Trigger specialist creation for identified gaps
            specialists_created = await self._create_specialists_for_gaps()
            
            # Update analysis tracking
            self.last_analysis = datetime.utcnow()
            analysis_summary = {
                "timestamp": self.last_analysis.isoformat(),
                "days_analyzed": days_back,
                "workflows_processed": len(workflow_data),
                "patterns_discovered": len(patterns_found),
                "optimizations_identified": len(optimizations_found),
                "agent_gaps_found": len(gaps_found),
                "runbooks_created": runbooks_created,
                "specialists_created": specialists_created,
                "analysis_result": analysis_result
            }
            
            self.analysis_history.append(analysis_summary)
            
            # Log analysis completion
            if self.db_logger:
                await self.db_logger.log_event(
                    "workflow_analysis_completed",
                    analysis_summary,
                    "system"
                )
            
            logger.info(f"Workflow analysis completed: {len(patterns_found)} patterns, {len(optimizations_found)} optimizations")
            return analysis_summary
            
        except Exception as e:
            logger.error(f"Error performing workflow analysis: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _gather_workflow_data(self, days_back: int) -> List[Dict[str, Any]]:
        """Gather workflow execution data for analysis."""
        try:
            if not self.db_logger:
                return []
            
            # Query workflow runs from the new table
            from_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
            
            try:
                # Try new workflow_runs table first
                result = self.db_logger.client.table("workflow_runs").select("*").gte("created_at", from_date).execute()
                if result.data:
                    return result.data
            except:
                pass
            
            # Fallback to messages table analysis
            messages = await self.db_logger.client.table("messages").select(
                "user_id", "agent_type", "message_type", "processing_time_ms",
                "timestamp", "routing_confidence", "escalation_suggestion"
            ).gte("timestamp", from_date).execute()
            
            # Group by conversation patterns
            workflow_data = []
            # ... process messages into workflow patterns ...
            
            return workflow_data if workflow_data else messages.data
            
        except Exception as e:
            logger.error(f"Error gathering workflow data: {str(e)}")
            return []
    
    async def _gather_agent_metrics(self, days_back: int) -> List[Dict[str, Any]]:
        """Gather agent performance metrics for analysis."""
        try:
            if not self.db_logger:
                return []
            
            return await self.db_logger.get_agent_performance(days=days_back)
            
        except Exception as e:
            logger.error(f"Error gathering agent metrics: {str(e)}")
            return []
    
    async def _perform_llm_analysis(self, workflow_data: List[Dict[str, Any]], 
                                   agent_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform LLM-powered analysis of workflow data."""
        try:
            # Prepare data for LLM analysis
            total_workflows = len(workflow_data)
            success_rate = sum(w.get("success_rate", 0) for w in workflow_data) / total_workflows if total_workflows > 0 else 0
            available_agents = list(set(w.get("agent_type", "unknown") for w in workflow_data))
            
            analysis_result = await self.analysis_chain.ainvoke({
                "workflow_data": json.dumps(workflow_data[:20], indent=2),  # Limit data size
                "agent_metrics": json.dumps(agent_metrics[:10], indent=2),
                "existing_patterns": json.dumps([p.name for p in self.discovered_patterns.values()]),
                "total_workflows": total_workflows,
                "time_period": f"Last {len(workflow_data)} workflows",
                "available_agents": ", ".join(available_agents),
                "success_rate": f"{success_rate:.1%}"
            })
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error performing LLM analysis: {str(e)}")
            # Return fallback analysis
            return {
                "patterns_found": [],
                "bottlenecks": [],
                "optimizations": [],
                "success_factors": ["Successful agent routing", "Quick response times"],
                "failure_factors": ["Network timeouts", "Agent unavailability"],
                "agent_gaps": []
            }
    
    async def _process_patterns(self, patterns_data: List[Dict[str, Any]]) -> List[str]:
        """Process discovered patterns and store them."""
        pattern_ids = []
        
        for pattern_data in patterns_data:
            try:
                pattern_id = f"pattern_{uuid.uuid4().hex[:8]}"
                
                # Determine pattern strength based on frequency
                frequency = pattern_data.get("frequency", 1)
                if frequency >= 16:
                    strength = PatternStrength.VERY_STRONG
                elif frequency >= 8:
                    strength = PatternStrength.STRONG
                elif frequency >= 4:
                    strength = PatternStrength.MODERATE
                else:
                    strength = PatternStrength.WEAK
                
                pattern = WorkflowPattern(
                    id=pattern_id,
                    name=pattern_data.get("name", "Unnamed Pattern"),
                    description=pattern_data.get("description", ""),
                    trigger_keywords=pattern_data.get("trigger_keywords", []),
                    typical_steps=pattern_data.get("typical_steps", []),
                    required_agents=pattern_data.get("required_agents", []),
                    success_rate=pattern_data.get("success_rate", 0.0),
                    avg_duration=pattern_data.get("avg_duration", 0.0),
                    avg_cost=pattern_data.get("avg_cost", 0.0),
                    frequency=frequency,
                    strength=strength,
                    example_runs=pattern_data.get("example_runs", []),
                    confidence=pattern_data.get("confidence", 0.8)
                )
                
                self.discovered_patterns[pattern_id] = pattern
                pattern_ids.append(pattern_id)
                
                logger.info(f"Discovered pattern: {pattern.name} (strength: {strength.value})")
                
            except Exception as e:
                logger.error(f"Error processing pattern: {str(e)}")
        
        return pattern_ids
    
    async def _process_optimizations(self, optimizations_data: List[Dict[str, Any]]) -> List[str]:
        """Process optimization opportunities and store them."""
        optimization_ids = []
        
        for opt_data in optimizations_data:
            try:
                opt_id = f"opt_{uuid.uuid4().hex[:8]}"
                
                optimization = OptimizationOpportunity(
                    id=opt_id,
                    type=OptimizationType(opt_data.get("type", "workflow_automation")),
                    title=opt_data.get("title", "Optimization Opportunity"),
                    description=opt_data.get("description", ""),
                    potential_benefit=opt_data.get("potential_benefit", ""),
                    estimated_savings=opt_data.get("estimated_savings", {}),
                    affected_patterns=opt_data.get("affected_patterns", []),
                    implementation_complexity=opt_data.get("implementation_complexity", "medium"),
                    priority=opt_data.get("priority", 3),
                    action_required=opt_data.get("action_required", ""),
                    confidence=opt_data.get("confidence", 0.7)
                )
                
                self.optimization_opportunities[opt_id] = optimization
                optimization_ids.append(opt_id)
                
                logger.info(f"Identified optimization: {optimization.title} (priority: {optimization.priority})")
                
            except Exception as e:
                logger.error(f"Error processing optimization: {str(e)}")
        
        return optimization_ids
    
    async def _process_agent_gaps(self, gaps_data: List[Dict[str, Any]]) -> List[str]:
        """Process agent capability gaps and store them."""
        gap_ids = []
        
        for gap_data in gaps_data:
            try:
                gap_id = f"gap_{uuid.uuid4().hex[:8]}"
                
                gap = AgentGap(
                    id=gap_id,
                    capability_gap=gap_data.get("capability_gap", "Unknown Gap"),
                    frequency=gap_data.get("frequency", 1),
                    typical_requests=gap_data.get("typical_requests", []),
                    suggested_agent_type=gap_data.get("suggested_agent_type", ""),
                    estimated_demand=gap_data.get("estimated_demand", {}),
                    priority=gap_data.get("priority", 3)
                )
                
                self.agent_gaps[gap_id] = gap
                gap_ids.append(gap_id)
                
                logger.info(f"Identified agent gap: {gap.capability_gap} (frequency: {gap.frequency})")
                
            except Exception as e:
                logger.error(f"Error processing agent gap: {str(e)}")
        
        return gap_ids
    
    async def _generate_runbooks_for_patterns(self) -> int:
        """Generate runbooks for strong patterns."""
        runbooks_created = 0
        
        # Only create runbooks for strong patterns
        strong_patterns = [
            p for p in self.discovered_patterns.values() 
            if p.strength in [PatternStrength.STRONG, PatternStrength.VERY_STRONG] 
            and p.success_rate > 0.8
        ]
        
        for pattern in strong_patterns:
            try:
                # Create serializable pattern data
                pattern_dict = asdict(pattern)
                # Convert enum to string for JSON serialization
                pattern_dict['strength'] = pattern.strength.value
                pattern_dict['created_at'] = pattern.created_at.isoformat() if pattern.created_at else None
                
                runbook_spec = await self.runbook_chain.ainvoke({
                    "pattern_name": pattern.name,
                    "pattern_description": pattern.description,
                    "trigger_keywords": json.dumps(pattern.trigger_keywords),
                    "typical_steps": json.dumps(pattern.typical_steps),
                    "required_agents": json.dumps(pattern.required_agents),
                    "success_rate": pattern.success_rate * 100,
                    "avg_duration": pattern.avg_duration,
                    "example_workflows": "Example workflow execution logs",
                    "pattern_data": json.dumps(pattern_dict, indent=2),
                    "success_examples": f"Pattern seen {pattern.frequency} times with {pattern.success_rate:.1%} success"
                })
                
                # Save runbook to file
                await self._save_runbook(pattern, runbook_spec)
                runbooks_created += 1
                
                logger.info(f"Generated runbook for pattern: {pattern.name}")
                
            except Exception as e:
                logger.error(f"Error generating runbook for pattern {pattern.name}: {str(e)}")
        
        return runbooks_created
    
    async def _save_runbook(self, pattern: WorkflowPattern, runbook_spec: Dict[str, Any]):
        """Save generated runbook to the runbooks directory."""
        try:
            runbook_name = runbook_spec.get("runbook_name", pattern.name.lower().replace(" ", "-"))
            runbook_path = f"runbooks/active/{runbook_name}.yaml"
            
            # Create runbook content
            runbook_content = {
                "metadata": {
                    "name": runbook_name,
                    "version": "1.0.0",
                    "description": f"Auto-generated from pattern: {pattern.name}",
                    "author": "Workflow Analyst",
                    "created_date": datetime.utcnow().isoformat(),
                    "pattern_id": pattern.id,
                    "pattern_strength": pattern.strength.value,
                    "success_rate": pattern.success_rate,
                    "llm_context": f"Generated from {pattern.frequency} workflow instances with {pattern.success_rate:.1%} success rate"
                },
                "triggers": runbook_spec.get("triggers", []),
                "steps": runbook_spec.get("steps", []),
                "configuration": runbook_spec.get("configuration", {
                    "timeout_seconds": int(pattern.avg_duration * 60),
                    "max_retries": 3,
                    "log_execution": True,
                    "cache_results": True
                }),
                "outputs": {
                    "success": {
                        "message": f"Pattern '{pattern.name}' executed successfully",
                        "include_details": True,
                        "log_level": "info"
                    },
                    "failure": {
                        "message": f"Pattern '{pattern.name}' execution failed",
                        "include_error": True,
                        "log_level": "error"
                    }
                },
                "analytics": {
                    "track_usage": True,
                    "track_performance": True,
                    "track_errors": True,
                    "pattern_analytics": {
                        "original_frequency": pattern.frequency,
                        "original_success_rate": pattern.success_rate,
                        "creation_date": pattern.created_at.isoformat()
                    }
                }
            }
            
            # Write runbook file
            os.makedirs("runbooks/active", exist_ok=True)
            with open(runbook_path, 'w') as f:
                yaml.dump(runbook_content, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved runbook: {runbook_path}")
            
        except Exception as e:
            logger.error(f"Error saving runbook: {str(e)}")
    
    async def _create_specialists_for_gaps(self) -> int:
        """Create specialist agents for identified capability gaps."""
        specialists_created = 0
        
        if not self.orchestrator:
            return 0
        
        # Create specialists for high-priority gaps
        high_priority_gaps = [
            gap for gap in self.agent_gaps.values() 
            if gap.priority >= 4 and gap.frequency >= 5
        ]
        
        for gap in high_priority_gaps:
            try:
                # Create specialist agent for this capability gap
                specialist_id = await self.orchestrator.spawn_specialist_agent(
                    gap.suggested_agent_type,
                    {
                        "capability_gap": gap.capability_gap,
                        "typical_requests": gap.typical_requests,
                        "estimated_demand": gap.estimated_demand
                    }
                )
                
                if specialist_id:
                    specialists_created += 1
                    logger.info(f"Created specialist '{gap.suggested_agent_type}' for gap: {gap.capability_gap}")
                
            except Exception as e:
                logger.error(f"Error creating specialist for gap {gap.capability_gap}: {str(e)}")
        
        return specialists_created
    
    def _start_periodic_analysis(self):
        """Start periodic workflow analysis task."""
        
        async def analysis_loop():
            while True:
                try:
                    await asyncio.sleep(self.analysis_frequency.total_seconds())
                    await self.analyze_workflows()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in periodic analysis: {str(e)}")
                    await asyncio.sleep(3600)  # Wait 1 hour on error
        
        self.analysis_task = asyncio.create_task(analysis_loop())
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of analysis results."""
        return {
            "discovered_patterns": {
                "total": len(self.discovered_patterns),
                "by_strength": {
                    strength.value: len([p for p in self.discovered_patterns.values() if p.strength == strength])
                    for strength in PatternStrength
                },
                "patterns": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "strength": p.strength.value,
                        "frequency": p.frequency,
                        "success_rate": p.success_rate,
                        "avg_cost": p.avg_cost
                    }
                    for p in self.discovered_patterns.values()
                ]
            },
            "optimization_opportunities": {
                "total": len(self.optimization_opportunities),
                "by_priority": {
                    priority: len([o for o in self.optimization_opportunities.values() if o.priority == priority])
                    for priority in range(1, 6)
                },
                "by_type": {
                    opt_type.value: len([o for o in self.optimization_opportunities.values() if o.type == opt_type])
                    for opt_type in OptimizationType
                }
            },
            "agent_gaps": {
                "total": len(self.agent_gaps),
                "high_priority": len([g for g in self.agent_gaps.values() if g.priority >= 4]),
                "gaps": [
                    {
                        "id": g.id,
                        "capability_gap": g.capability_gap,
                        "frequency": g.frequency,
                        "priority": g.priority,
                        "suggested_agent": g.suggested_agent_type
                    }
                    for g in self.agent_gaps.values()
                ]
            },
            "analysis_history": {
                "total_analyses": len(self.analysis_history),
                "last_analysis": self.last_analysis.isoformat() if self.last_analysis else None,
                "next_analysis": (self.last_analysis + self.analysis_frequency).isoformat() if self.last_analysis else None
            }
        }
    
    def get_pattern(self, pattern_id: str) -> Optional[WorkflowPattern]:
        """Get a specific pattern by ID."""
        return self.discovered_patterns.get(pattern_id)
    
    def get_optimization(self, optimization_id: str) -> Optional[OptimizationOpportunity]:
        """Get a specific optimization by ID."""
        return self.optimization_opportunities.get(optimization_id)
    
    def get_top_patterns(self, limit: int = 10) -> List[WorkflowPattern]:
        """Get top patterns by frequency and success rate."""
        return sorted(
            self.discovered_patterns.values(),
            key=lambda p: (p.frequency * p.success_rate, p.strength.name),
            reverse=True
        )[:limit]
    
    def get_high_priority_optimizations(self, limit: int = 10) -> List[OptimizationOpportunity]:
        """Get highest priority optimization opportunities."""
        return sorted(
            self.optimization_opportunities.values(),
            key=lambda o: (o.priority, o.confidence),
            reverse=True
        )[:limit]
    
    async def close(self):
        """Clean up workflow analyst resources."""
        try:
            logger.info("Closing Workflow Analyst...")
            
            # Cancel periodic analysis task
            if self.analysis_task:
                self.analysis_task.cancel()
                try:
                    await self.analysis_task
                except asyncio.CancelledError:
                    pass
            
            # Close LLM connections
            for llm in [self.analysis_llm, self.runbook_llm, self.optimization_llm]:
                if hasattr(llm, 'client') and hasattr(llm.client, 'close'):
                    await llm.client.close()
            
            logger.info("Workflow Analyst closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing Workflow Analyst: {e}") 