"""
Filename: pattern_recognition.py
Purpose: Advanced pattern recognition system for user interactions and automation
Dependencies: langchain, openai, asyncio, logging, typing, supabase

This module monitors user interactions to identify recurring sequences, temporal patterns,
context patterns, and automatically creates runbooks and suggests workflow automations.
"""

import asyncio
import logging
import os
import json
import uuid
import yaml
from typing import Dict, Any, Optional, List, Tuple, Set, Union
from datetime import datetime, timedelta, time
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter, deque
import re
from statistics import mean, median

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of patterns that can be detected."""
    SEQUENCE = "sequence"           # A->B->C recurring sequences
    TEMPORAL = "temporal"           # Time-based patterns (daily, weekly)
    CONTEXTUAL = "contextual"       # When X happens, do Y
    LINGUISTIC = "linguistic"       # Similar language/phrasing patterns
    WORKFLOW = "workflow"           # Multi-step process patterns
    TRIGGER = "trigger"             # Event-driven patterns

class TemporalPeriod(Enum):
    """Temporal pattern periods."""
    HOURLY = "hourly"
    DAILY = "daily" 
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"

class PatternStrength(Enum):
    """Pattern strength based on frequency and consistency."""
    EMERGING = "emerging"           # 3-5 occurrences
    DEVELOPING = "developing"       # 6-10 occurrences
    ESTABLISHED = "established"     # 11-20 occurrences
    STRONG = "strong"               # 21+ occurrences

@dataclass
class InteractionEvent:
    """Represents a single user interaction event."""
    id: str
    user_id: str
    timestamp: datetime
    message: str
    context: Dict[str, Any]
    agent_used: str
    success: bool
    duration_ms: float
    tokens_used: int
    cost: float
    session_id: Optional[str] = None
    channel_id: Optional[str] = None
    thread_ts: Optional[str] = None

@dataclass
class RecognizedPattern:
    """Represents a recognized interaction pattern."""
    id: str
    type: PatternType
    name: str
    description: str
    trigger_conditions: List[str]
    sequence_steps: List[str]
    frequency: int
    strength: PatternStrength
    confidence: float
    users_affected: Set[str]
    temporal_info: Optional[Dict[str, Any]]
    automation_potential: float  # 0.0-1.0
    avg_duration_ms: float
    avg_cost: float
    success_rate: float
    examples: List[str]  # Event IDs
    created_at: datetime
    last_seen: datetime
    
    def __post_init__(self):
        if isinstance(self.users_affected, list):
            self.users_affected = set(self.users_affected)

@dataclass
class AutomationSuggestion:
    """Represents a suggested automation based on patterns."""
    id: str
    pattern_id: str
    title: str
    description: str
    automation_type: str  # "runbook", "shortcut", "scheduled", "triggered"
    estimated_time_saved: float  # minutes per execution
    estimated_cost_saved: float  # USD per execution
    implementation_complexity: str  # "low", "medium", "high"
    priority_score: float  # 0.0-1.0
    suggested_triggers: List[str]
    proposed_workflow: Dict[str, Any]
    user_benefit: str
    created_at: datetime

class PatternAnalysis(BaseModel):
    """LLM response schema for pattern analysis."""
    patterns_found: List[Dict[str, Any]] = Field(description="Identified interaction patterns")
    automation_opportunities: List[Dict[str, Any]] = Field(description="Suggested automations")
    temporal_insights: Dict[str, Any] = Field(description="Time-based pattern insights")
    user_behavior_trends: List[Dict[str, Any]] = Field(description="User behavior patterns")

class PatternRecognitionEngine:
    """
    Advanced pattern recognition system that monitors user interactions to identify
    recurring sequences, temporal patterns, and automation opportunities.
    """
    
    def __init__(self, db_logger=None, orchestrator=None):
        """
        Initialize the Pattern Recognition Engine.
        
        Args:
            db_logger: Supabase logger for accessing interaction data
            orchestrator: Agent orchestrator for automation deployment
        """
        self.db_logger = db_logger
        self.orchestrator = orchestrator
        
        # Pattern storage
        self.recognized_patterns: Dict[str, RecognizedPattern] = {}
        self.automation_suggestions: Dict[str, AutomationSuggestion] = {}
        
        # Interaction tracking
        self.interaction_buffer = deque()  # Recent interactions (manually managed size)
        self.user_sessions: Dict[str, List[InteractionEvent]] = defaultdict(list)
        self.temporal_buckets: Dict[str, List[InteractionEvent]] = defaultdict(list)
        
        # Analysis state
        self.last_analysis: Optional[datetime] = None
        self.analysis_frequency = timedelta(hours=2)  # Analyze every 2 hours
        self.min_pattern_frequency = 3  # Minimum occurrences to consider a pattern
        
        # Memory management
        self.max_memory_days = 30  # Keep patterns for 30 days
        self.max_interactions_memory = 100000  # Persist beyond this to DB
        
        # Initialize LLMs for different analysis types
        self.pattern_llm = ChatOpenAI(
            model="gpt-4-0125-preview",
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=3000,
        )
        
        self.automation_llm = ChatOpenAI(
            model="gpt-4-0125-preview", 
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2000,
        )
        
        self.temporal_llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=1500,
        )
        
        # Create prompt templates
        self.pattern_prompt = self._create_pattern_analysis_prompt()
        self.automation_prompt = self._create_automation_suggestion_prompt()
        self.temporal_prompt = self._create_temporal_analysis_prompt()
        
        # Create parsing chains
        self.pattern_parser = JsonOutputParser(pydantic_object=PatternAnalysis)
        self.pattern_chain = self.pattern_prompt | self.pattern_llm | self.pattern_parser
        
        # Start monitoring task
        self.monitoring_task = None
        self._start_continuous_monitoring()
        
        logger.info("Pattern Recognition Engine initialized with advanced analytics")
    
    def _create_pattern_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for pattern analysis."""
        
        system_template = """You are an expert pattern recognition analyst for an AI Agent Platform.
Your role is to analyze user interaction data to identify recurring patterns, sequences,
and automation opportunities that can improve user experience and system efficiency.

Focus on these pattern types:
1. SEQUENCE patterns: A→B→C recurring sequences in user interactions
2. TEMPORAL patterns: Time-based patterns (daily, weekly, specific hours)
3. CONTEXTUAL patterns: When X condition occurs, users typically do Y
4. LINGUISTIC patterns: Similar language/phrasing that indicates same intent
5. WORKFLOW patterns: Multi-step processes users repeat
6. TRIGGER patterns: Events that consistently lead to specific actions

Analyze the interaction data and identify patterns with:
- Clear trigger conditions
- Repeatable sequence steps  
- Automation potential (0.0-1.0 score)
- User impact and benefit
- Implementation complexity

Be specific about automation opportunities and practical next steps."""

        human_template = """Analyze these user interactions for patterns:

INTERACTION DATA:
{interaction_data}

TEMPORAL CONTEXT:
{temporal_context}

USER BEHAVIOR SUMMARY:
{user_summary}

Please identify:
1. Recurring interaction patterns with clear sequences
2. Temporal patterns (time-based behaviors)
3. Automation opportunities with high user benefit
4. User behavior trends and insights

Return analysis in the specified JSON format."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_automation_suggestion_prompt(self) -> ChatPromptTemplate:
        """Create prompt for automation suggestions."""
        
        system_template = """You are an automation specialist for an AI Agent Platform.
Your role is to create practical automation suggestions based on identified patterns.

For each pattern, suggest automations that:
1. Save significant user time and effort
2. Reduce repetitive tasks
3. Improve user experience  
4. Are technically feasible to implement
5. Have clear ROI in time/cost savings

Automation types to consider:
- RUNBOOK: Predefined workflows for common sequences
- SHORTCUT: Quick commands for frequent actions
- SCHEDULED: Time-based automated tasks
- TRIGGERED: Event-driven automations
- TEMPLATE: Reusable message/workflow templates

Be specific about implementation and user benefits."""

        human_template = """Create automation suggestions for these patterns:

IDENTIFIED PATTERNS:
{patterns}

USER CONTEXT:
{user_context}

SYSTEM CAPABILITIES:
{capabilities}

For each pattern, suggest specific automations with:
1. Clear automation type and triggers
2. Step-by-step implementation approach
3. Estimated time/cost savings
4. User experience improvements
5. Implementation complexity assessment

Return suggestions in the specified JSON format."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_temporal_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for temporal pattern analysis."""
        
        system_template = """You are a temporal pattern analyst for an AI Agent Platform.
Your role is to identify time-based patterns in user behavior that can be automated
or optimized based on timing.

Analyze for:
1. Daily patterns (specific hours when users are most active)
2. Weekly patterns (certain days for specific tasks)
3. Recurring schedules (monthly reports, weekly check-ins)
4. Seasonal patterns (if data spans sufficient time)
5. Context-time correlations (certain requests at certain times)

Suggest timing-based optimizations and scheduled automations."""

        human_template = """Analyze temporal patterns in this data:

TIMESTAMPED INTERACTIONS:
{timestamped_data}

TIME PERIOD: {time_period}

Identify patterns in:
1. Peak usage hours/days
2. Recurring time-based requests
3. Scheduling opportunities
4. Context-time correlations

Return temporal insights in JSON format."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ]) 

    async def record_interaction(self, event: InteractionEvent):
        """
        Record a new user interaction for pattern analysis.
        
        Args:
            event: The interaction event to record
        """
        try:
            # Add to buffer with size management
            self.interaction_buffer.append(event)
            if len(self.interaction_buffer) > 10000:  # Manual size management
                self.interaction_buffer.popleft()
            
            # Add to user session
            self.user_sessions[event.user_id].append(event)
            
            # Add to temporal buckets
            hour_key = f"{event.timestamp.hour:02d}:00"
            day_key = event.timestamp.strftime("%A")
            date_key = event.timestamp.strftime("%Y-%m-%d")
            
            self.temporal_buckets[f"hour_{hour_key}"].append(event)
            self.temporal_buckets[f"day_{day_key}"].append(event)
            self.temporal_buckets[f"date_{date_key}"].append(event)
            
            # Trigger real-time pattern check for emerging patterns
            await self._check_real_time_patterns(event)
            
            logger.debug(f"Recorded interaction: {event.id} from {event.user_id}")
            
        except Exception as e:
            logger.error(f"Error recording interaction: {str(e)}")
    
    async def _check_real_time_patterns(self, event: InteractionEvent):
        """Check for emerging patterns in real-time."""
        try:
            # Check recent user interactions for immediate sequences
            user_recent = self.user_sessions[event.user_id][-10:]  # Last 10 interactions
            
            if len(user_recent) >= 3:
                # Look for A→B→C sequences in recent interactions
                recent_sequence = [e.message[:50] for e in user_recent[-3:]]
                sequence_key = " → ".join(recent_sequence)
                
                # Check if this sequence has been seen before
                existing_patterns = [p for p in self.recognized_patterns.values() 
                                   if p.type == PatternType.SEQUENCE]
                
                for pattern in existing_patterns:
                    if any(seq in sequence_key for seq in pattern.sequence_steps):
                        # Update existing pattern
                        pattern.frequency += 1
                        pattern.last_seen = event.timestamp
                        pattern.examples.append(event.id)
                        
                        # Check if pattern strength changed
                        old_strength = pattern.strength
                        pattern.strength = self._calculate_pattern_strength(pattern.frequency)
                        
                        if pattern.strength != old_strength:
                            logger.info(f"Pattern {pattern.name} strength upgraded to {pattern.strength.value}")
                            
                            # Trigger automation suggestion if strong enough
                            if pattern.strength in [PatternStrength.ESTABLISHED, PatternStrength.STRONG]:
                                await self._suggest_automation_for_pattern(pattern)
                        
                        break
            
        except Exception as e:
            logger.error(f"Error in real-time pattern check: {str(e)}")
    
    async def analyze_patterns(self, days_back: int = 7, force_analysis: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive pattern analysis on recent interactions.
        
        Args:
            days_back: Number of days to analyze
            force_analysis: Force analysis even if recently performed
            
        Returns:
            Analysis results with patterns and suggestions
        """
        try:
            # Check if analysis needed
            if not force_analysis and self.last_analysis:
                time_since_last = datetime.utcnow() - self.last_analysis
                if time_since_last < self.analysis_frequency:
                    logger.info(f"Skipping analysis - last run {time_since_last.total_seconds()/3600:.1f}h ago")
                    return self._get_cached_analysis()
            
            logger.info(f"Starting pattern analysis for last {days_back} days")
            
            # Gather interaction data
            interactions = await self._gather_interaction_data(days_back)
            if len(interactions) < self.min_pattern_frequency:
                logger.info(f"Insufficient data for analysis: {len(interactions)} interactions")
                return {"status": "insufficient_data", "interactions_found": len(interactions)}
            
            # Prepare data for LLM analysis
            interaction_data = self._prepare_interaction_data(interactions)
            temporal_context = self._prepare_temporal_context(interactions)
            user_summary = self._prepare_user_summary(interactions)
            
            # Perform LLM-powered pattern analysis
            analysis_result = await self._perform_llm_pattern_analysis(
                interaction_data, temporal_context, user_summary
            )
            
            # Process and store results
            new_patterns = await self._process_pattern_results(analysis_result)
            automation_suggestions = await self._generate_automation_suggestions(new_patterns)
            
            # Update analysis timestamp
            self.last_analysis = datetime.utcnow()
            
            # Persist patterns to database
            await self._persist_patterns_to_db()
            
            result = {
                "status": "completed",
                "analysis_timestamp": self.last_analysis.isoformat(),
                "interactions_analyzed": len(interactions),
                "new_patterns_found": len(new_patterns),
                "automation_suggestions": len(automation_suggestions),
                "patterns": [asdict(p) for p in new_patterns],
                "suggestions": [asdict(s) for s in automation_suggestions]
            }
            
            logger.info(f"Pattern analysis completed: {len(new_patterns)} patterns, {len(automation_suggestions)} suggestions")
            return result
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_pattern_strength(self, frequency: int) -> PatternStrength:
        """Calculate pattern strength based on frequency."""
        if frequency >= 21:
            return PatternStrength.STRONG
        elif frequency >= 11:
            return PatternStrength.ESTABLISHED
        elif frequency >= 6:
            return PatternStrength.DEVELOPING
        else:
            return PatternStrength.EMERGING
    
    async def _suggest_automation_for_pattern(self, pattern: RecognizedPattern):
        """Suggest automation for a strong pattern."""
        try:
            if pattern.automation_potential > 0.7:
                suggestion = AutomationSuggestion(
                    id=str(uuid.uuid4()),
                    pattern_id=pattern.id,
                    title=f"Automate {pattern.name}",
                    description=f"Create automated workflow for {pattern.description}",
                    automation_type="runbook",
                    estimated_time_saved=pattern.avg_duration_ms / 60000,  # Convert to minutes
                    estimated_cost_saved=pattern.avg_cost * 0.5,  # 50% cost reduction
                    implementation_complexity="medium",
                    priority_score=pattern.automation_potential,
                    suggested_triggers=pattern.trigger_conditions,
                    proposed_workflow={
                        "steps": pattern.sequence_steps,
                        "agents": ["universal"],
                        "automation_level": "semi-automatic"
                    },
                    user_benefit=f"Save {pattern.avg_duration_ms/60000:.1f} minutes per execution",
                    created_at=datetime.utcnow()
                )
                
                self.automation_suggestions[suggestion.id] = suggestion
                logger.info(f"Created automation suggestion for pattern {pattern.name}")
                
        except Exception as e:
            logger.error(f"Error creating automation suggestion: {str(e)}")
    
    def _start_continuous_monitoring(self):
        """Start the continuous monitoring task."""
        async def monitoring_loop():
            while True:
                try:
                    await asyncio.sleep(self.analysis_frequency.total_seconds())
                    await self.analyze_patterns(force_analysis=False)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
        
        self.monitoring_task = asyncio.create_task(monitoring_loop())
        logger.info("Started continuous pattern monitoring")
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of all recognized patterns."""
        patterns_by_type = defaultdict(list)
        patterns_by_strength = defaultdict(int)
        
        for pattern in self.recognized_patterns.values():
            patterns_by_type[pattern.type.value].append(pattern.name)
            patterns_by_strength[pattern.strength.value] += 1
        
        return {
            "total_patterns": len(self.recognized_patterns),
            "patterns_by_type": dict(patterns_by_type),
            "patterns_by_strength": dict(patterns_by_strength),
            "automation_suggestions": len(self.automation_suggestions),
            "last_analysis": self.last_analysis.isoformat() if self.last_analysis else None,
            "interactions_in_buffer": len(self.interaction_buffer)
        }
    
    def get_top_patterns(self, limit: int = 10) -> List[RecognizedPattern]:
        """Get top patterns by frequency and automation potential."""
        sorted_patterns = sorted(
            self.recognized_patterns.values(),
            key=lambda p: (p.frequency, p.automation_potential),
            reverse=True
        )
        return sorted_patterns[:limit]
    
    def get_automation_suggestions(self, limit: int = 10) -> List[AutomationSuggestion]:
        """Get top automation suggestions by priority score."""
        sorted_suggestions = sorted(
            self.automation_suggestions.values(),
            key=lambda s: s.priority_score,
            reverse=True
        )
        return sorted_suggestions[:limit]
    
    async def close(self):
        """Cleanup resources."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Pattern Recognition Engine closed")
    
    def _get_cached_analysis(self) -> Dict[str, Any]:
        """Return cached analysis results."""
        return {
            "status": "cached",
            "last_analysis": self.last_analysis.isoformat() if self.last_analysis else None,
            "total_patterns": len(self.recognized_patterns),
            "automation_suggestions": len(self.automation_suggestions),
            "patterns": [asdict(p) for p in self.recognized_patterns.values()],
            "suggestions": [asdict(s) for s in self.automation_suggestions.values()]
        }
    
    async def _gather_interaction_data(self, days_back: int) -> List[InteractionEvent]:
        """Gather interaction data from buffer and database."""
        try:
            interactions = []
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Get from in-memory buffer first
            for event in self.interaction_buffer:
                if event.timestamp >= cutoff_date:
                    interactions.append(event)
            
            # Get additional data from database if available
            if self.db_logger:
                try:
                    # Query database for recent interactions (simplified)
                    db_interactions = await self._query_db_interactions(days_back)
                    
                    # Convert to InteractionEvent objects
                    for db_event in db_interactions:
                        event = self._convert_db_to_event(db_event)
                        if event and event.timestamp >= cutoff_date:
                            interactions.append(event)
                
                except Exception as e:
                    logger.warning(f"Could not fetch from database: {str(e)}")
            
            # Sort by timestamp
            interactions.sort(key=lambda x: x.timestamp)
            
            logger.info(f"Gathered {len(interactions)} interactions for analysis")
            return interactions
            
        except Exception as e:
            logger.error(f"Error gathering interaction data: {str(e)}")
            return []
    
    async def _query_db_interactions(self, days_back: int) -> List[Dict[str, Any]]:
        """Query database for recent interactions."""
        try:
            # This would use the Supabase logger to get message data
            # For now, return empty list - implement based on actual DB schema
            return []
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            return []
    
    def _convert_db_to_event(self, db_event: Dict[str, Any]) -> Optional[InteractionEvent]:
        """Convert database record to InteractionEvent."""
        try:
            # Convert database record to InteractionEvent
            # This would depend on the actual database schema
            return None  # Placeholder implementation
        except Exception as e:
            logger.error(f"Error converting DB event: {str(e)}")
            return None
    
    def _prepare_interaction_data(self, interactions: List[InteractionEvent]) -> str:
        """Prepare interaction data for LLM analysis."""
        try:
            data_summary = []
            
            for i, event in enumerate(interactions[-50:]):  # Last 50 interactions
                summary = {
                    "sequence": i + 1,
                    "timestamp": event.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "user": event.user_id[-4:],  # Last 4 chars for privacy
                    "message": event.message[:100],  # First 100 chars
                    "agent": event.agent_used,
                    "success": event.success,
                    "duration_ms": event.duration_ms,
                    "cost": event.cost
                }
                data_summary.append(summary)
            
            return json.dumps(data_summary, indent=2)
            
        except Exception as e:
            logger.error(f"Error preparing interaction data: {str(e)}")
            return "[]"
    
    def _prepare_temporal_context(self, interactions: List[InteractionEvent]) -> str:
        """Prepare temporal context for analysis."""
        try:
            temporal_analysis = {
                "time_range": {
                    "start": interactions[0].timestamp.isoformat() if interactions else None,
                    "end": interactions[-1].timestamp.isoformat() if interactions else None,
                    "total_interactions": len(interactions)
                },
                "hourly_distribution": {},
                "daily_distribution": {},
                "peak_hours": [],
                "peak_days": []
            }
            
            # Analyze hourly distribution
            hourly_counts = defaultdict(int)
            daily_counts = defaultdict(int)
            
            for event in interactions:
                hour = event.timestamp.hour
                day = event.timestamp.strftime("%A")
                hourly_counts[hour] += 1
                daily_counts[day] += 1
            
            temporal_analysis["hourly_distribution"] = dict(hourly_counts)
            temporal_analysis["daily_distribution"] = dict(daily_counts)
            
            # Find peak times
            if hourly_counts:
                max_hour_count = max(hourly_counts.values())
                temporal_analysis["peak_hours"] = [
                    f"{hour:02d}:00" for hour, count in hourly_counts.items() 
                    if count == max_hour_count
                ]
            
            if daily_counts:
                max_day_count = max(daily_counts.values())
                temporal_analysis["peak_days"] = [
                    day for day, count in daily_counts.items() 
                    if count == max_day_count
                ]
            
            return json.dumps(temporal_analysis, indent=2)
            
        except Exception as e:
            logger.error(f"Error preparing temporal context: {str(e)}")
            return "{}"
    
    def _prepare_user_summary(self, interactions: List[InteractionEvent]) -> str:
        """Prepare user behavior summary."""
        try:
            user_stats = defaultdict(lambda: {
                "interactions": 0,
                "avg_duration": 0,
                "total_cost": 0,
                "success_rate": 0,
                "agents_used": set(),
                "common_phrases": []
            })
            
            # Analyze per-user patterns
            for event in interactions:
                user_id = event.user_id
                stats = user_stats[user_id]
                
                stats["interactions"] += 1
                stats["avg_duration"] += event.duration_ms
                stats["total_cost"] += event.cost
                if event.success:
                    stats["success_rate"] += 1
                stats["agents_used"].add(event.agent_used)
                
                # Extract common phrases (simplified)
                words = event.message.lower().split()
                if len(words) > 2:
                    phrases = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
                    stats["common_phrases"].extend(phrases)
            
            # Calculate averages and convert sets to lists
            for user_id, stats in user_stats.items():
                if stats["interactions"] > 0:
                    stats["avg_duration"] /= stats["interactions"]
                    stats["success_rate"] = stats["success_rate"] / stats["interactions"]
                    stats["agents_used"] = list(stats["agents_used"])
                    
                    # Get most common phrases
                    phrase_counts = Counter(stats["common_phrases"])
                    stats["common_phrases"] = [
                        phrase for phrase, count in phrase_counts.most_common(5)
                    ]
            
            summary = {
                "total_users": len(user_stats),
                "user_patterns": dict(user_stats),
                "overall_stats": {
                    "total_interactions": len(interactions),
                    "avg_success_rate": mean([s["success_rate"] for s in user_stats.values()]) if user_stats else 0,
                    "avg_cost_per_interaction": mean([e.cost for e in interactions]) if interactions else 0
                }
            }
            
            return json.dumps(summary, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error preparing user summary: {str(e)}")
            return "{}"
    
    async def _perform_llm_pattern_analysis(self, interaction_data: str, 
                                          temporal_context: str, user_summary: str) -> Dict[str, Any]:
        """Perform LLM-powered pattern analysis."""
        try:
            # Use the pattern analysis chain
            result = await self.pattern_chain.ainvoke({
                "interaction_data": interaction_data,
                "temporal_context": temporal_context,
                "user_summary": user_summary
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM pattern analysis: {str(e)}")
            return {
                "patterns_found": [],
                "automation_opportunities": [],
                "temporal_insights": {},
                "user_behavior_trends": []
            }
    
    async def _process_pattern_results(self, analysis_result: Dict[str, Any]) -> List[RecognizedPattern]:
        """Process LLM analysis results into RecognizedPattern objects."""
        try:
            new_patterns = []
            
            for pattern_data in analysis_result.get("patterns_found", []):
                try:
                    pattern = RecognizedPattern(
                        id=str(uuid.uuid4()),
                        type=PatternType(pattern_data.get("type", "sequence")),
                        name=pattern_data.get("name", "Unnamed Pattern"),
                        description=pattern_data.get("description", ""),
                        trigger_conditions=pattern_data.get("trigger_conditions", []),
                        sequence_steps=pattern_data.get("sequence_steps", []),
                        frequency=pattern_data.get("frequency", 1),
                        strength=PatternStrength(pattern_data.get("strength", "emerging")),
                        confidence=pattern_data.get("confidence", 0.5),
                        users_affected=set(pattern_data.get("users_affected", [])),
                        temporal_info=pattern_data.get("temporal_info"),
                        automation_potential=pattern_data.get("automation_potential", 0.5),
                        avg_duration_ms=pattern_data.get("avg_duration_ms", 1000),
                        avg_cost=pattern_data.get("avg_cost", 0.01),
                        success_rate=pattern_data.get("success_rate", 1.0),
                        examples=pattern_data.get("examples", []),
                        created_at=datetime.utcnow(),
                        last_seen=datetime.utcnow()
                    )
                    
                    # Store pattern
                    self.recognized_patterns[pattern.id] = pattern
                    new_patterns.append(pattern)
                    
                except Exception as e:
                    logger.error(f"Error processing pattern: {str(e)}")
                    continue
            
            return new_patterns
            
        except Exception as e:
            logger.error(f"Error processing pattern results: {str(e)}")
            return []
    
    async def _generate_automation_suggestions(self, patterns: List[RecognizedPattern]) -> List[AutomationSuggestion]:
        """Generate automation suggestions for patterns."""
        try:
            new_suggestions = []
            
            for pattern in patterns:
                if pattern.automation_potential > 0.6:  # High automation potential
                    suggestion = AutomationSuggestion(
                        id=str(uuid.uuid4()),
                        pattern_id=pattern.id,
                        title=f"Automate {pattern.name}",
                        description=f"Create automated workflow for {pattern.description}",
                        automation_type=self._determine_automation_type(pattern),
                        estimated_time_saved=pattern.avg_duration_ms / 60000,  # Convert to minutes
                        estimated_cost_saved=pattern.avg_cost * 0.4,  # 40% cost reduction
                        implementation_complexity=self._assess_complexity(pattern),
                        priority_score=pattern.automation_potential * pattern.frequency / 10,
                        suggested_triggers=pattern.trigger_conditions,
                        proposed_workflow={
                            "steps": pattern.sequence_steps,
                            "agents": ["universal"],
                            "automation_level": "semi-automatic"
                        },
                        user_benefit=f"Save {pattern.avg_duration_ms/60000:.1f} minutes per execution",
                        created_at=datetime.utcnow()
                    )
                    
                    self.automation_suggestions[suggestion.id] = suggestion
                    new_suggestions.append(suggestion)
            
            return new_suggestions
            
        except Exception as e:
            logger.error(f"Error generating automation suggestions: {str(e)}")
            return []
    
    def _determine_automation_type(self, pattern: RecognizedPattern) -> str:
        """Determine the best automation type for a pattern."""
        if pattern.temporal_info and "recurring" in str(pattern.temporal_info):
            return "scheduled"
        elif pattern.type == PatternType.TRIGGER:
            return "triggered"
        elif pattern.frequency > 15:
            return "shortcut"
        else:
            return "runbook"
    
    def _assess_complexity(self, pattern: RecognizedPattern) -> str:
        """Assess implementation complexity for a pattern."""
        if len(pattern.sequence_steps) <= 3:
            return "low"
        elif len(pattern.sequence_steps) <= 6:
            return "medium"
        else:
            return "high"
    
    async def _persist_patterns_to_db(self):
        """Persist patterns to database for long-term storage."""
        try:
            if self.db_logger:
                # This would persist patterns to Supabase
                # Implementation depends on database schema
                pass
            
        except Exception as e:
            logger.error(f"Error persisting patterns to DB: {str(e)}") 