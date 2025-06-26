"""
Filename: error_recovery.py
Purpose: Intelligent error recovery agent for self-healing AI system
Dependencies: langchain, openai, asyncio, logging, typing, supabase

This module analyzes every error in the system, identifies patterns and root causes,
creates recovery strategies, and automatically applies fixes to prevent future errors.
"""

import asyncio
import logging
import os
import json
import uuid
import traceback
import inspect
from typing import Dict, Any, Optional, List, Tuple, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import hashlib
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"           # Minor issues, system continues
    MEDIUM = "medium"     # Moderate impact, may affect performance
    HIGH = "high"         # Significant impact, feature disruption
    CRITICAL = "critical" # System failure, immediate attention required

class ErrorCategory(Enum):
    """Categories of errors for classification."""
    SYSTEM = "system"           # System-level errors (OS, network, database)
    LLM = "llm"                # LLM-related errors (API limits, model issues)
    AGENT = "agent"            # Agent-specific errors (logic, processing)
    USER = "user"              # User-input related errors
    INTEGRATION = "integration" # Integration/communication errors
    RESOURCE = "resource"       # Resource-related errors (memory, disk, tokens)
    SECURITY = "security"       # Security-related errors
    UNKNOWN = "unknown"         # Unclassified errors

class RecoveryStatus(Enum):
    """Status of recovery strategy application."""
    NOT_ATTEMPTED = "not_attempted"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"

@dataclass
class ErrorEvent:
    """Represents a single error occurrence."""
    id: str
    timestamp: datetime
    error_type: str
    error_message: str
    stack_trace: str
    component: str  # Which component/agent/module
    context: Dict[str, Any]
    severity: ErrorSeverity
    category: ErrorCategory
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    environment_info: Optional[Dict[str, Any]] = None

@dataclass
class ErrorPattern:
    """Represents an identified error pattern."""
    id: str
    pattern_name: str
    description: str
    error_signature: str  # Hash of common error characteristics
    occurrences: int
    first_seen: datetime
    last_seen: datetime
    affected_components: Set[str]
    common_contexts: List[Dict[str, Any]]
    severity_distribution: Dict[ErrorSeverity, int]
    category: ErrorCategory
    root_cause_analysis: str
    confidence: float  # 0.0-1.0 confidence in pattern identification
    
    def __post_init__(self):
        if isinstance(self.affected_components, list):
            self.affected_components = set(self.affected_components)

@dataclass
class RecoveryStrategy:
    """Represents a recovery strategy for an error pattern."""
    id: str
    pattern_id: str
    strategy_name: str
    description: str
    strategy_type: str  # "retry", "fallback", "reset", "escalate", "ignore"
    implementation: Dict[str, Any]  # Code/configuration for strategy
    preconditions: List[str]
    success_rate: float
    average_recovery_time: float  # seconds
    side_effects: List[str]
    priority: int  # 1-10, higher = more preferred
    created_at: datetime
    last_applied: Optional[datetime] = None
    application_count: int = 0

@dataclass
class PreventiveMeasure:
    """Represents a preventive measure to avoid errors."""
    id: str
    measure_name: str
    description: str
    target_patterns: List[str]  # Pattern IDs this prevents
    implementation_type: str  # "config_change", "code_change", "monitoring", "validation"
    implementation_details: Dict[str, Any]
    estimated_reduction: float  # 0.0-1.0 expected error reduction
    implementation_complexity: str  # "low", "medium", "high"
    maintenance_overhead: str  # "low", "medium", "high"
    created_at: datetime

class ErrorAnalysis(BaseModel):
    """LLM response schema for error analysis."""
    error_category: str = Field(description="Categorization of the error")
    severity_assessment: str = Field(description="Severity level assessment")
    root_cause: str = Field(description="Identified root cause")
    contributing_factors: List[str] = Field(description="Factors that contributed to the error")
    impact_analysis: Dict[str, Any] = Field(description="Analysis of error impact")
    pattern_indicators: List[str] = Field(description="Indicators this might be part of a pattern")

class RecoveryPlan(BaseModel):
    """LLM response schema for recovery strategy generation."""
    recovery_strategies: List[Dict[str, Any]] = Field(description="Proposed recovery strategies")
    preventive_measures: List[Dict[str, Any]] = Field(description="Preventive measures to avoid recurrence")
    monitoring_recommendations: List[str] = Field(description="Monitoring recommendations")
    escalation_triggers: List[str] = Field(description="Conditions that should trigger escalation")

class ErrorRecoveryAgent:
    """
    Intelligent Error Recovery Agent that learns from failures to create a self-healing AI system.
    
    Capabilities:
    - Real-time error analysis and classification
    - Pattern recognition across error types
    - Automatic recovery strategy generation and application
    - Preventive measure recommendations
    - Knowledge base of error solutions
    - Self-improving recovery effectiveness
    """
    
    def __init__(self, db_logger=None, orchestrator=None):
        """
        Initialize the Error Recovery Agent.
        
        Args:
            db_logger: Supabase logger for error persistence
            orchestrator: Agent orchestrator for system coordination
        """
        self.db_logger = db_logger
        self.orchestrator = orchestrator
        
        # Error tracking and storage
        self.error_events: List[ErrorEvent] = []
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.preventive_measures: Dict[str, PreventiveMeasure] = {}
        
        # Pattern recognition
        self.error_signatures: Dict[str, str] = {}  # signature -> pattern_id
        self.component_error_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Recovery tracking
        self.active_recoveries: Dict[str, Dict[str, Any]] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.max_error_history = 10000
        self.pattern_detection_threshold = 3  # Minimum occurrences for pattern
        self.auto_recovery_enabled = True
        self.analysis_frequency = timedelta(hours=1)  # Analyze errors hourly
        
        # Initialize LLMs for error analysis
        self.analysis_llm = ChatOpenAI(
            model="gpt-4-0125-preview",
            temperature=0.1,  # Low temperature for consistent analysis
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2000,
        )
        
        self.recovery_llm = ChatOpenAI(
            model="gpt-4-0125-preview",
            temperature=0.2,  # Slightly higher for creative solutions
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2500,
        )
        
        self.classification_llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0.0,  # Deterministic classification
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=500,
        )
        
        # Create prompt templates
        self.analysis_prompt = self._create_error_analysis_prompt()
        self.recovery_prompt = self._create_recovery_generation_prompt()
        self.classification_prompt = self._create_classification_prompt()
        
        # Create parsing chains
        self.analysis_parser = JsonOutputParser(pydantic_object=ErrorAnalysis)
        self.recovery_parser = JsonOutputParser(pydantic_object=RecoveryPlan)
        
        self.analysis_chain = self.analysis_prompt | self.analysis_llm | self.analysis_parser
        self.recovery_chain = self.recovery_prompt | self.recovery_llm | self.recovery_parser
        
        # Start monitoring and analysis tasks
        self.monitoring_task = None
        self.analysis_task = None
        self._start_error_monitoring()
        
        logger.info("Error Recovery Agent initialized with intelligent error analysis")
    
    def _create_error_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for error analysis."""
        
        system_template = """You are an expert error analysis specialist for an AI Agent Platform.
Your role is to analyze errors, identify root causes, assess severity, and detect patterns
that could indicate systemic issues requiring attention.

Focus on:
1. CATEGORIZATION: Classify errors into system, LLM, agent, user, integration, resource, or security categories
2. SEVERITY ASSESSMENT: Determine impact level (low/medium/high/critical)
3. ROOT CAUSE ANALYSIS: Identify the underlying cause, not just symptoms
4. CONTRIBUTING FACTORS: Find environmental or contextual factors that contributed
5. PATTERN INDICATORS: Look for signs this error might be part of a larger pattern
6. IMPACT ANALYSIS: Assess the broader impact on users and system functionality

Be thorough but concise. Focus on actionable insights."""

        human_template = """Analyze this error event:

ERROR DETAILS:
Type: {error_type}
Message: {error_message}
Component: {component}
Stack Trace: {stack_trace}

CONTEXT:
{context_info}

ENVIRONMENT:
{environment_info}

RECENT SIMILAR ERRORS:
{similar_errors}

Please provide a comprehensive analysis including categorization, severity assessment,
root cause identification, contributing factors, and pattern indicators.

Return your analysis in the specified JSON format."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_recovery_generation_prompt(self) -> ChatPromptTemplate:
        """Create prompt for recovery strategy generation."""
        
        system_template = """You are an expert recovery strategy specialist for an AI Agent Platform.
Your role is to create effective recovery strategies and preventive measures for error patterns.

For each error pattern, design strategies that:
1. IMMEDIATE RECOVERY: Quick fixes to restore functionality
2. ROBUST HANDLING: Long-term solutions that prevent recurrence  
3. GRACEFUL DEGRADATION: Fallback options when primary recovery fails
4. PREVENTIVE MEASURES: Changes to avoid the error entirely
5. MONITORING: Early warning systems to detect issues before they become errors

Strategy types to consider:
- RETRY: Attempt the operation again (with backoff, limits)
- FALLBACK: Switch to alternative approach/service
- RESET: Clear state and restart component
- ESCALATE: Alert human operators
- IGNORE: Continue operation (for non-critical errors)
- CIRCUIT_BREAKER: Temporarily disable failing component

Be specific about implementation and provide clear success criteria."""

        human_template = """Create recovery strategies for this error pattern:

PATTERN DETAILS:
Name: {pattern_name}
Description: {pattern_description}
Category: {category}
Occurrences: {occurrences}
Affected Components: {affected_components}

ROOT CAUSE ANALYSIS:
{root_cause}

COMMON CONTEXTS:
{common_contexts}

SEVERITY DISTRIBUTION:
{severity_distribution}

Create comprehensive recovery strategies with:
1. Multiple recovery options (immediate, fallback, escalation)
2. Preventive measures to avoid future occurrences
3. Monitoring recommendations for early detection
4. Clear implementation guidance

Return strategies in the specified JSON format."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_classification_prompt(self) -> ChatPromptTemplate:
        """Create prompt for quick error classification."""
        
        system_template = """You are an error classification specialist. Quickly categorize errors into:

CATEGORIES:
- system: OS, network, database, infrastructure issues
- llm: LLM API errors, model issues, token limits
- agent: Agent logic errors, processing failures  
- user: Invalid user input, permission issues
- integration: Service communication failures
- resource: Memory, disk, CPU, quota issues
- security: Authentication, authorization, security violations
- unknown: Cannot be classified

SEVERITY LEVELS:
- low: Minor issues, system continues normally
- medium: Moderate impact, degraded performance
- high: Significant impact, feature disruption
- critical: System failure, immediate attention required

Be consistent and accurate in classification."""

        human_template = """Classify this error:

Error Type: {error_type}
Message: {error_message}
Component: {component}

Return: category,severity"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    async def record_error(self, error: Exception, component: str, 
                          context: Optional[Dict[str, Any]] = None,
                          user_id: Optional[str] = None,
                          severity: Optional[ErrorSeverity] = None) -> str:
        """
        Record an error event for analysis and potential recovery.
        
        Args:
            error: The exception that occurred
            component: Component/module where error occurred
            context: Additional context information
            user_id: User associated with the error (if applicable)
            severity: Override severity assessment
            
        Returns:
            Error event ID
        """
        try:
            # Create error event
            error_id = str(uuid.uuid4())
            
            # Extract error information
            error_type = type(error).__name__
            error_message = str(error)
            stack_trace = traceback.format_exc() if hasattr(error, '__traceback__') else ""
            
            # Quick classification if severity not provided
            if severity is None:
                severity = await self._classify_error_severity(error_type, error_message, component)
            
            # Categorize error
            category = await self._classify_error_category(error_type, error_message, component)
            
            # Gather environment info
            environment_info = self._gather_environment_info()
            
            # Create error event
            error_event = ErrorEvent(
                id=error_id,
                timestamp=datetime.utcnow(),
                error_type=error_type,
                error_message=error_message,
                stack_trace=stack_trace,
                component=component,
                context=context or {},
                severity=severity,
                category=category,
                user_id=user_id,
                environment_info=environment_info
            )
            
            # Store error event
            self.error_events.append(error_event)
            
            # Manage memory
            if len(self.error_events) > self.max_error_history:
                self.error_events = self.error_events[-self.max_error_history:]
            
            # Update component statistics
            self.component_error_stats[component][error_type] += 1
            
            # Check for immediate pattern match
            await self._check_immediate_pattern_match(error_event)
            
            # Attempt automatic recovery if enabled
            if self.auto_recovery_enabled:
                asyncio.create_task(self._attempt_automatic_recovery(error_event))
            
            # Persist to database
            await self._persist_error_to_db(error_event)
            
            logger.warning(f"Error recorded: {error_type} in {component} [{error_id}]")
            return error_id
            
        except Exception as e:
            logger.error(f"Failed to record error: {str(e)}")
            return ""
    
    async def _classify_error_severity(self, error_type: str, error_message: str, component: str) -> ErrorSeverity:
        """Quickly classify error severity."""
        try:
            # Simple heuristic-based classification
            error_lower = error_message.lower()
            
            # Critical errors
            if any(keyword in error_lower for keyword in ["system", "database", "connection", "memory", "disk"]):
                return ErrorSeverity.CRITICAL
            
            # High severity errors
            if any(keyword in error_lower for keyword in ["timeout", "permission", "authentication", "authorization"]):
                return ErrorSeverity.HIGH
            
            # Medium severity errors
            if any(keyword in error_lower for keyword in ["rate limit", "quota", "retry", "temporary"]):
                return ErrorSeverity.MEDIUM
            
            # Default to low for unclassified
            return ErrorSeverity.LOW
            
        except Exception as e:
            logger.error(f"Error classifying severity: {str(e)}")
            return ErrorSeverity.MEDIUM
    
    async def _classify_error_category(self, error_type: str, error_message: str, component: str) -> ErrorCategory:
        """Classify error into category."""
        try:
            error_lower = error_message.lower()
            
            # LLM-related errors
            if any(keyword in error_lower for keyword in ["openai", "api", "token", "model", "completion"]):
                return ErrorCategory.LLM
            
            # System-level errors
            if any(keyword in error_lower for keyword in ["connection", "network", "database", "file", "disk"]):
                return ErrorCategory.SYSTEM
            
            # Resource errors
            if any(keyword in error_lower for keyword in ["memory", "quota", "limit", "resource"]):
                return ErrorCategory.RESOURCE
            
            # Security errors
            if any(keyword in error_lower for keyword in ["permission", "auth", "security", "forbidden"]):
                return ErrorCategory.SECURITY
            
            # User input errors
            if any(keyword in error_lower for keyword in ["validation", "input", "format", "syntax"]):
                return ErrorCategory.USER
            
            # Integration errors
            if any(keyword in error_lower for keyword in ["timeout", "service", "endpoint", "api"]):
                return ErrorCategory.INTEGRATION
            
            # Agent-specific errors
            if "agent" in component.lower():
                return ErrorCategory.AGENT
            
            return ErrorCategory.UNKNOWN
            
        except Exception as e:
            logger.error(f"Error classifying category: {str(e)}")
            return ErrorCategory.UNKNOWN
    
    def _gather_environment_info(self) -> Dict[str, Any]:
        """Gather current environment information."""
        try:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "python_version": os.sys.version,
                "platform": os.name,
                "memory_usage": self._get_memory_usage(),
                "active_tasks": len(asyncio.all_tasks()) if hasattr(asyncio, 'all_tasks') else 0
            }
        except Exception as e:
            logger.error(f"Error gathering environment info: {str(e)}")
            return {}
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            return {
                "rss": process.memory_info().rss,
                "vms": process.memory_info().vms,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    async def _check_immediate_pattern_match(self, error_event: ErrorEvent):
        """Check if error matches an existing pattern."""
        try:
            # Generate error signature
            signature = self._generate_error_signature(error_event)
            
            # Check if pattern exists
            if signature in self.error_signatures:
                pattern_id = self.error_signatures[signature]
                pattern = self.error_patterns[pattern_id]
                
                # Update pattern with new occurrence
                pattern.occurrences += 1
                pattern.last_seen = error_event.timestamp
                pattern.affected_components.add(error_event.component)
                
                # Update severity distribution
                if error_event.severity in pattern.severity_distribution:
                    pattern.severity_distribution[error_event.severity] += 1
                else:
                    pattern.severity_distribution[error_event.severity] = 1
                
                logger.info(f"Error matched existing pattern: {pattern.pattern_name} ({pattern.occurrences} occurrences)")
                
            else:
                # Check if we should create a new pattern
                await self._detect_new_pattern(error_event, signature)
                
        except Exception as e:
            logger.error(f"Error checking pattern match: {str(e)}")
    
    def _generate_error_signature(self, error_event: ErrorEvent) -> str:
        """Generate a signature for error pattern matching."""
        try:
            # Combine key characteristics for signature
            signature_components = [
                error_event.error_type,
                error_event.component,
                error_event.category.value,
                # Simplified error message (remove dynamic parts)
                self._normalize_error_message(error_event.error_message)
            ]
            
            signature_string = "|".join(signature_components)
            return hashlib.md5(signature_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating signature: {str(e)}")
            return str(uuid.uuid4())
    
    def _normalize_error_message(self, message: str) -> str:
        """Normalize error message by removing dynamic parts."""
        try:
            # Remove common dynamic parts
            normalized = re.sub(r'\d+', 'NUM', message)  # Replace numbers
            normalized = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', 'UUID', normalized)  # UUIDs
            normalized = re.sub(r'/[^\s]+', '/PATH', normalized)  # File paths
            normalized = re.sub(r'at \d{4}-\d{2}-\d{2}', 'at DATE', normalized)  # Dates
            
            return normalized.lower().strip()
            
        except Exception as e:
            logger.error(f"Error normalizing message: {str(e)}")
            return message.lower().strip()
    
    async def _detect_new_pattern(self, error_event: ErrorEvent, signature: str):
        """Detect if this error should start a new pattern."""
        try:
            # Count similar errors in recent history
            similar_count = 0
            similar_events = []
            
            current_time = datetime.utcnow()
            lookback_window = current_time - timedelta(hours=24)  # Look back 24 hours
            
            for event in reversed(self.error_events[-100:]):  # Check recent events
                if (event.timestamp >= lookback_window and
                    event.error_type == error_event.error_type and
                    event.component == error_event.component and
                    event.category == error_event.category):
                    
                    similar_count += 1
                    similar_events.append(event)
            
            # Create pattern if threshold met
            if similar_count >= self.pattern_detection_threshold:
                await self._create_error_pattern(error_event, signature, similar_events)
                
        except Exception as e:
            logger.error(f"Error detecting new pattern: {str(e)}")
    
    async def _create_error_pattern(self, error_event: ErrorEvent, signature: str, similar_events: List[ErrorEvent]):
        """Create a new error pattern."""
        try:
            pattern_id = str(uuid.uuid4())
            
            # Generate pattern name
            pattern_name = f"{error_event.error_type} in {error_event.component}"
            
            # Calculate severity distribution
            severity_dist = defaultdict(int)
            for event in similar_events:
                severity_dist[event.severity] += 1
            
            # Extract common contexts
            common_contexts = []
            for event in similar_events[-5:]:  # Take last 5 contexts
                if event.context:
                    common_contexts.append(event.context)
            
            # Create pattern
            pattern = ErrorPattern(
                id=pattern_id,
                pattern_name=pattern_name,
                description=f"Recurring {error_event.error_type} errors in {error_event.component}",
                error_signature=signature,
                occurrences=len(similar_events),
                first_seen=similar_events[0].timestamp if similar_events else error_event.timestamp,
                last_seen=error_event.timestamp,
                affected_components={error_event.component},
                common_contexts=common_contexts,
                severity_distribution=dict(severity_dist),
                category=error_event.category,
                root_cause_analysis="",  # Will be filled by LLM analysis
                confidence=0.8  # High confidence for frequent patterns
            )
            
            # Store pattern
            self.error_patterns[pattern_id] = pattern
            self.error_signatures[signature] = pattern_id
            
            # Trigger deep analysis
            asyncio.create_task(self._analyze_pattern_with_llm(pattern))
            
            logger.info(f"New error pattern created: {pattern_name} [{pattern_id}]")
            
        except Exception as e:
            logger.error(f"Error creating pattern: {str(e)}")
    
    async def _analyze_pattern_with_llm(self, pattern: ErrorPattern):
        """Perform deep LLM analysis of error pattern."""
        try:
            # Get recent events for this pattern
            pattern_events = []
            for event in self.error_events:
                if self._generate_error_signature(event) == pattern.error_signature:
                    pattern_events.append(event)
            
            if not pattern_events:
                return
            
            # Prepare analysis data
            analysis_data = {
                "error_type": pattern_events[0].error_type,
                "error_message": pattern_events[0].error_message,
                "component": pattern_events[0].component,
                "stack_trace": pattern_events[0].stack_trace,
                "context_info": json.dumps(pattern.common_contexts, indent=2),
                "environment_info": json.dumps(pattern_events[0].environment_info or {}, indent=2),
                "similar_errors": self._format_similar_errors(pattern_events)
            }
            
            # Perform LLM analysis
            analysis_result = await self.analysis_chain.ainvoke(analysis_data)
            
            # Update pattern with analysis
            pattern.root_cause_analysis = analysis_result["root_cause"]
            pattern.confidence = min(pattern.confidence + 0.1, 1.0)  # Increase confidence
            
            # Generate recovery strategies
            await self._generate_recovery_strategies(pattern, analysis_result)
            
            logger.info(f"Pattern analysis completed: {pattern.pattern_name}")
            
        except Exception as e:
            logger.error(f"Error analyzing pattern with LLM: {str(e)}")
    
    def _format_similar_errors(self, events: List[ErrorEvent]) -> str:
        """Format similar errors for LLM analysis."""
        try:
            formatted = []
            for event in events[-5:]:  # Last 5 events
                formatted.append(f"- {event.timestamp}: {event.error_message} ({event.severity.value})")
            return "\n".join(formatted)
        except Exception as e:
            logger.error(f"Error formatting similar errors: {str(e)}")
            return "No similar errors found"
    
    async def _generate_recovery_strategies(self, pattern: ErrorPattern, analysis: Dict[str, Any]):
        """Generate recovery strategies for a pattern."""
        try:
            # Prepare strategy generation data
            strategy_data = {
                "pattern_name": pattern.pattern_name,
                "pattern_description": pattern.description,
                "category": pattern.category.value,
                "occurrences": pattern.occurrences,
                "affected_components": list(pattern.affected_components),
                "root_cause": pattern.root_cause_analysis,
                "common_contexts": json.dumps(pattern.common_contexts, indent=2),
                "severity_distribution": {k.value: v for k, v in pattern.severity_distribution.items()}
            }
            
            # Generate strategies with LLM
            recovery_plan = await self.recovery_chain.ainvoke(strategy_data)
            
            # Create recovery strategy objects
            for strategy_data in recovery_plan["recovery_strategies"]:
                strategy = RecoveryStrategy(
                    id=str(uuid.uuid4()),
                    pattern_id=pattern.id,
                    strategy_name=strategy_data.get("name", "Unknown Strategy"),
                    description=strategy_data.get("description", ""),
                    strategy_type=strategy_data.get("type", "retry"),
                    implementation=strategy_data.get("implementation", {}),
                    preconditions=strategy_data.get("preconditions", []),
                    success_rate=0.0,  # Will be updated based on results
                    average_recovery_time=0.0,
                    side_effects=strategy_data.get("side_effects", []),
                    priority=strategy_data.get("priority", 5),
                    created_at=datetime.utcnow()
                )
                
                self.recovery_strategies[strategy.id] = strategy
            
            # Create preventive measures
            for measure_data in recovery_plan["preventive_measures"]:
                measure = PreventiveMeasure(
                    id=str(uuid.uuid4()),
                    measure_name=measure_data.get("name", "Unknown Measure"),
                    description=measure_data.get("description", ""),
                    target_patterns=[pattern.id],
                    implementation_type=measure_data.get("implementation_type", "monitoring"),
                    implementation_details=measure_data.get("implementation", {}),
                    estimated_reduction=measure_data.get("estimated_reduction", 0.5),
                    implementation_complexity=measure_data.get("complexity", "medium"),
                    maintenance_overhead=measure_data.get("maintenance", "low"),
                    created_at=datetime.utcnow()
                )
                
                self.preventive_measures[measure.id] = measure
            
            logger.info(f"Recovery strategies generated for pattern: {pattern.pattern_name}")
            
        except Exception as e:
            logger.error(f"Error generating recovery strategies: {str(e)}")
    
    async def _attempt_automatic_recovery(self, error_event: ErrorEvent):
        """Attempt automatic recovery for an error."""
        try:
            # Find matching recovery strategies
            signature = self._generate_error_signature(error_event)
            
            if signature not in self.error_signatures:
                logger.debug(f"No recovery strategies found for error: {error_event.error_type}")
                return
            
            pattern_id = self.error_signatures[signature]
            pattern = self.error_patterns[pattern_id]
            
            # Get available strategies for this pattern
            available_strategies = [
                strategy for strategy in self.recovery_strategies.values()
                if strategy.pattern_id == pattern_id
            ]
            
            if not available_strategies:
                logger.debug(f"No recovery strategies available for pattern: {pattern.pattern_name}")
                return
            
            # Sort strategies by priority and success rate
            available_strategies.sort(key=lambda s: (s.priority, s.success_rate), reverse=True)
            
            # Try each strategy in order
            for strategy in available_strategies:
                try:
                    recovery_id = str(uuid.uuid4())
                    
                    # Record recovery attempt
                    self.active_recoveries[recovery_id] = {
                        "strategy_id": strategy.id,
                        "error_id": error_event.id,
                        "start_time": datetime.utcnow(),
                        "status": RecoveryStatus.IN_PROGRESS
                    }
                    
                    # Apply recovery strategy
                    success = await self._apply_recovery_strategy(strategy, error_event, recovery_id)
                    
                    if success:
                        # Update strategy success rate
                        strategy.application_count += 1
                        strategy.success_rate = (strategy.success_rate * (strategy.application_count - 1) + 1) / strategy.application_count
                        strategy.last_applied = datetime.utcnow()
                        
                        logger.info(f"Recovery successful: {strategy.strategy_name} for {error_event.error_type}")
                        break
                    else:
                        # Update failure rate
                        strategy.application_count += 1
                        strategy.success_rate = (strategy.success_rate * (strategy.application_count - 1)) / strategy.application_count
                        
                        logger.warning(f"Recovery failed: {strategy.strategy_name} for {error_event.error_type}")
                
                except Exception as e:
                    logger.error(f"Error applying recovery strategy: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error in automatic recovery: {str(e)}")
    
    async def _apply_recovery_strategy(self, strategy: RecoveryStrategy, error_event: ErrorEvent, recovery_id: str) -> bool:
        """Apply a specific recovery strategy."""
        try:
            implementation = strategy.implementation
            strategy_type = strategy.strategy_type.lower()
            
            if strategy_type == "retry":
                return await self._apply_retry_strategy(implementation, error_event)
            elif strategy_type == "fallback":
                return await self._apply_fallback_strategy(implementation, error_event)
            elif strategy_type == "reset":
                return await self._apply_reset_strategy(implementation, error_event)
            elif strategy_type == "escalate":
                return await self._apply_escalation_strategy(implementation, error_event)
            elif strategy_type == "ignore":
                return True  # Always "succeeds" by ignoring
            elif strategy_type == "circuit_breaker":
                return await self._apply_circuit_breaker_strategy(implementation, error_event)
            else:
                logger.warning(f"Unknown recovery strategy type: {strategy_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying recovery strategy: {str(e)}")
            return False
    
    async def _apply_retry_strategy(self, implementation: Dict[str, Any], error_event: ErrorEvent) -> bool:
        """Apply retry recovery strategy."""
        try:
            max_retries = implementation.get("max_retries", 3)
            backoff_delay = implementation.get("backoff_delay", 1.0)
            
            logger.info(f"Applying retry strategy: max_retries={max_retries}")
            
            for attempt in range(max_retries):
                try:
                    # Wait with exponential backoff
                    if attempt > 0:
                        await asyncio.sleep(backoff_delay * (2 ** (attempt - 1)))
                    
                    # Here we would retry the original operation
                    # For now, we simulate success/failure
                    if attempt >= max_retries - 1:  # Last attempt
                        return True  # Simulate success
                    
                except Exception as retry_error:
                    logger.warning(f"Retry attempt {attempt + 1} failed: {str(retry_error)}")
                    continue
            
            return False
            
        except Exception as e:
            logger.error(f"Error in retry strategy: {str(e)}")
            return False
    
    async def _apply_fallback_strategy(self, implementation: Dict[str, Any], error_event: ErrorEvent) -> bool:
        """Apply fallback recovery strategy."""
        try:
            fallback_component = implementation.get("fallback_component")
            fallback_method = implementation.get("fallback_method")
            
            logger.info(f"Applying fallback strategy: component={fallback_component}")
            
            # Here we would switch to the fallback implementation
            # For now, we simulate success
            return True
            
        except Exception as e:
            logger.error(f"Error in fallback strategy: {str(e)}")
            return False
    
    async def _apply_reset_strategy(self, implementation: Dict[str, Any], error_event: ErrorEvent) -> bool:
        """Apply reset recovery strategy."""
        try:
            reset_scope = implementation.get("reset_scope", "component")
            
            logger.info(f"Applying reset strategy: scope={reset_scope}")
            
            if reset_scope == "component":
                # Reset the specific component
                if self.orchestrator:
                    # Signal orchestrator to restart component
                    await self.orchestrator.restart_component(error_event.component)
            elif reset_scope == "agent":
                # Reset specific agent
                if self.orchestrator:
                    await self.orchestrator.restart_agent(error_event.component)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in reset strategy: {str(e)}")
            return False
    
    async def _apply_escalation_strategy(self, implementation: Dict[str, Any], error_event: ErrorEvent) -> bool:
        """Apply escalation recovery strategy."""
        try:
            escalation_level = implementation.get("escalation_level", "admin")
            notification_method = implementation.get("notification_method", "log")
            
            logger.critical(f"Escalating error: {error_event.error_type} in {error_event.component}")
            
            # For now, we just log the escalation
            # In a real system, this would send alerts, notifications, etc.
            
            return True
            
        except Exception as e:
            logger.error(f"Error in escalation strategy: {str(e)}")
            return False
    
    async def _apply_circuit_breaker_strategy(self, implementation: Dict[str, Any], error_event: ErrorEvent) -> bool:
        """Apply circuit breaker recovery strategy."""
        try:
            failure_threshold = implementation.get("failure_threshold", 5)
            timeout_duration = implementation.get("timeout_duration", 300)  # 5 minutes
            
            logger.info(f"Applying circuit breaker: component={error_event.component}")
            
            # Here we would implement circuit breaker logic
            # For now, we simulate temporary disabling
            
            return True
            
        except Exception as e:
            logger.error(f"Error in circuit breaker strategy: {str(e)}")
            return False
    
    def _start_error_monitoring(self):
        """Start error monitoring and analysis tasks."""
        try:
            if self.monitoring_task is None or self.monitoring_task.done():
                self.monitoring_task = asyncio.create_task(self._error_monitoring_loop())
            
            if self.analysis_task is None or self.analysis_task.done():
                self.analysis_task = asyncio.create_task(self._error_analysis_loop())
                
            logger.info("Error monitoring and analysis tasks started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring tasks: {str(e)}")
    
    async def _error_monitoring_loop(self):
        """Continuous error monitoring loop."""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                # Clean up old error events
                current_time = datetime.utcnow()
                cutoff_time = current_time - timedelta(days=7)  # Keep 7 days
                
                self.error_events = [
                    event for event in self.error_events
                    if event.timestamp >= cutoff_time
                ]
                
                # Clean up completed recoveries
                completed_recoveries = []
                for recovery_id, recovery in self.active_recoveries.items():
                    if recovery["status"] in [RecoveryStatus.SUCCESS, RecoveryStatus.FAILED]:
                        if current_time - recovery["start_time"] > timedelta(hours=1):
                            completed_recoveries.append(recovery_id)
                            self.recovery_history.append(recovery)
                
                for recovery_id in completed_recoveries:
                    del self.active_recoveries[recovery_id]
                
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
    
    async def _error_analysis_loop(self):
        """Continuous error analysis loop."""
        try:
            while True:
                await asyncio.sleep(3600)  # Analyze every hour
                
                # Analyze recent error patterns
                await self._analyze_recent_patterns()
                
                # Update pattern confidence
                await self._update_pattern_confidence()
                
                # Generate new preventive measures
                await self._suggest_preventive_measures()
                
        except Exception as e:
            logger.error(f"Error in analysis loop: {str(e)}")
    
    async def _analyze_recent_patterns(self):
        """Analyze recent error patterns for insights."""
        try:
            recent_time = datetime.utcnow() - timedelta(hours=1)
            recent_errors = [
                event for event in self.error_events
                if event.timestamp >= recent_time
            ]
            
            if len(recent_errors) < 3:
                return  # Not enough recent errors
            
            # Look for new patterns in recent errors
            error_groups = defaultdict(list)
            for error in recent_errors:
                signature = self._generate_error_signature(error)
                error_groups[signature].append(error)
            
            # Find frequent error groups that aren't patterns yet
            for signature, errors in error_groups.items():
                if len(errors) >= self.pattern_detection_threshold and signature not in self.error_signatures:
                    await self._create_error_pattern(errors[0], signature, errors)
            
            logger.info(f"Analyzed {len(recent_errors)} recent errors")
            
        except Exception as e:
            logger.error(f"Error analyzing recent patterns: {str(e)}")
    
    async def _update_pattern_confidence(self):
        """Update confidence scores for existing patterns."""
        try:
            current_time = datetime.utcnow()
            
            for pattern in self.error_patterns.values():
                # Decrease confidence for patterns not seen recently
                days_since_last = (current_time - pattern.last_seen).days
                
                if days_since_last > 7:
                    pattern.confidence = max(0.1, pattern.confidence - 0.1)
                elif days_since_last > 3:
                    pattern.confidence = max(0.3, pattern.confidence - 0.05)
                elif pattern.occurrences > 10:
                    pattern.confidence = min(1.0, pattern.confidence + 0.05)
            
            logger.debug("Updated pattern confidence scores")
            
        except Exception as e:
            logger.error(f"Error updating pattern confidence: {str(e)}")
    
    async def _suggest_preventive_measures(self):
        """Suggest new preventive measures based on patterns."""
        try:
            # Find patterns without preventive measures
            patterns_without_measures = []
            for pattern in self.error_patterns.values():
                has_measures = any(
                    pattern.id in measure.target_patterns
                    for measure in self.preventive_measures.values()
                )
                if not has_measures and pattern.confidence > 0.7:
                    patterns_without_measures.append(pattern)
            
            logger.info(f"Found {len(patterns_without_measures)} patterns needing preventive measures")
            
        except Exception as e:
            logger.error(f"Error suggesting preventive measures: {str(e)}")
    
    async def _persist_error_to_db(self, error_event: ErrorEvent):
        """Persist error event to database."""
        try:
            if self.db_logger:
                await self.db_logger.log_error(
                    error_type=error_event.error_type,
                    error_message=error_event.error_message,
                    component=error_event.component,
                    severity=error_event.severity.value,
                    category=error_event.category.value,
                    context=error_event.context,
                    user_id=error_event.user_id,
                    stack_trace=error_event.stack_trace
                )
                
        except Exception as e:
            logger.error(f"Error persisting to database: {str(e)}")
    
    async def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        try:
            current_time = datetime.utcnow()
            
            # Time-based statistics
            last_24h = current_time - timedelta(hours=24)
            last_week = current_time - timedelta(days=7)
            
            recent_errors = [e for e in self.error_events if e.timestamp >= last_24h]
            weekly_errors = [e for e in self.error_events if e.timestamp >= last_week]
            
            # Component statistics
            component_stats = defaultdict(int)
            for error in weekly_errors:
                component_stats[error.component] += 1
            
            # Severity distribution
            severity_stats = defaultdict(int)
            for error in weekly_errors:
                severity_stats[error.severity.value] += 1
            
            # Category distribution  
            category_stats = defaultdict(int)
            for error in weekly_errors:
                category_stats[error.category.value] += 1
            
            # Pattern statistics
            pattern_stats = {
                "total_patterns": len(self.error_patterns),
                "active_patterns": len([p for p in self.error_patterns.values() if p.confidence > 0.5]),
                "recovery_strategies": len(self.recovery_strategies),
                "preventive_measures": len(self.preventive_measures)
            }
            
            # Recovery statistics
            recovery_stats = {
                "active_recoveries": len(self.active_recoveries),
                "successful_recoveries": len([r for r in self.recovery_history if r.get("status") == RecoveryStatus.SUCCESS]),
                "failed_recoveries": len([r for r in self.recovery_history if r.get("status") == RecoveryStatus.FAILED])
            }
            
            return {
                "error_counts": {
                    "total_errors": len(self.error_events),
                    "last_24h": len(recent_errors),
                    "last_week": len(weekly_errors)
                },
                "component_breakdown": dict(component_stats),
                "severity_distribution": dict(severity_stats),
                "category_distribution": dict(category_stats),
                "pattern_statistics": pattern_stats,
                "recovery_statistics": recovery_stats,
                "top_error_types": self._get_top_error_types(weekly_errors)
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def _get_top_error_types(self, errors: List[ErrorEvent]) -> List[Dict[str, Any]]:
        """Get top error types from error list."""
        try:
            error_counts = Counter(error.error_type for error in errors)
            return [
                {"error_type": error_type, "count": count}
                for error_type, count in error_counts.most_common(10)
            ]
        except Exception as e:
            logger.error(f"Error getting top error types: {str(e)}")
            return []
    
    async def close(self):
        """Clean up resources."""
        try:
            # Cancel monitoring tasks
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
                
            if self.analysis_task and not self.analysis_task.done():
                self.analysis_task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(
                self.monitoring_task,
                self.analysis_task,
                return_exceptions=True
            )
            
            logger.info("Error Recovery Agent closed")
            
        except Exception as e:
            logger.error(f"Error closing Error Recovery Agent: {str(e)}") 