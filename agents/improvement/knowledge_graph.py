"""
Filename: knowledge_graph.py
Purpose: Knowledge Graph Builder for connecting all learnings into a searchable network
Dependencies: networkx, langchain, openai, asyncio, logging, typing, supabase

This module builds relationships between problems, solutions, and patterns to create
a searchable graph of all system knowledge, identify gaps, and enable cross-agent learning.
"""

import asyncio
import logging
import os
import json
import uuid
import yaml
from typing import Dict, Any, Optional, List, Tuple, Set, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter, deque
import re
from statistics import mean, median
import hashlib

import networkx as nx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    PROBLEM = "problem"
    SOLUTION = "solution"
    PATTERN = "pattern"
    AGENT = "agent"
    WORKFLOW = "workflow"
    USER_REQUEST = "user_request"
    ERROR = "error"
    OPTIMIZATION = "optimization"
    FEEDBACK = "feedback"
    COST_SAVING = "cost_saving"

class RelationshipType(Enum):
    """Types of relationships between nodes."""
    SOLVES = "solves"                   # Solution -> Problem
    CAUSES = "causes"                   # Problem -> Error
    CONTAINS = "contains"               # Pattern -> Solution/Problem
    IMPLEMENTS = "implements"           # Agent -> Solution
    TRIGGERS = "triggers"               # User Request -> Workflow
    OPTIMIZES = "optimizes"             # Optimization -> Workflow/Agent
    LEARNS_FROM = "learns_from"         # Agent -> Pattern/Error
    SIMILAR_TO = "similar_to"           # Any -> Any (similarity)
    DEPENDS_ON = "depends_on"           # Solution -> Agent/Resource
    IMPROVES = "improves"               # Feedback -> Solution/Workflow
    REDUCES = "reduces"                 # Cost Saving -> Agent/Workflow

class KnowledgeStrength(Enum):
    """Strength of knowledge connections."""
    WEAK = "weak"           # 1-2 occurrences, low confidence
    MODERATE = "moderate"   # 3-5 occurrences, medium confidence
    STRONG = "strong"       # 6-10 occurrences, high confidence
    VERY_STRONG = "very_strong"  # 11+ occurrences, very high confidence

@dataclass
class KnowledgeNode:
    """Represents a node in the knowledge graph."""
    id: str
    type: NodeType
    title: str
    description: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    confidence: float  # 0.0-1.0
    frequency: int     # How often this knowledge appears
    last_seen: datetime
    created_at: datetime
    tags: Set[str]
    
    def __post_init__(self):
        if isinstance(self.tags, list):
            self.tags = set(self.tags)

@dataclass
class KnowledgeRelationship:
    """Represents a relationship between nodes in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: KnowledgeStrength
    confidence: float
    evidence: List[str]  # Evidence supporting this relationship
    metadata: Dict[str, Any]
    created_at: datetime
    last_updated: datetime

@dataclass
class KnowledgeGap:
    """Represents an identified gap in the knowledge graph."""
    id: str
    gap_type: str  # "missing_solution", "missing_agent", "broken_path", "isolated_knowledge"
    description: str
    affected_areas: List[str]
    severity: str  # "low", "medium", "high", "critical"
    suggested_actions: List[str]
    potential_benefits: Dict[str, float]
    priority_score: float
    created_at: datetime

@dataclass
class LearningPath:
    """Represents a path from problem to solution through the graph."""
    id: str
    source_problem: str
    target_solution: str
    path_nodes: List[str]
    path_relationships: List[str]
    path_length: int
    confidence: float
    efficiency_score: float  # How good this path is
    alternative_paths: List[List[str]]
    created_at: datetime

class GraphAnalysis(BaseModel):
    """LLM response schema for graph analysis."""
    knowledge_gaps: List[Dict[str, Any]] = Field(description="Identified knowledge gaps")
    relationship_suggestions: List[Dict[str, Any]] = Field(description="Suggested new relationships")
    node_improvements: List[Dict[str, Any]] = Field(description="Node enhancement suggestions")
    learning_opportunities: List[Dict[str, Any]] = Field(description="Cross-agent learning opportunities")
    optimization_paths: List[Dict[str, Any]] = Field(description="Optimal knowledge paths")

class SpecialtyRecommendation(BaseModel):
    """LLM response schema for agent specialty recommendations."""
    recommended_specialties: List[Dict[str, Any]] = Field(description="Recommended new agent specialties")
    justification: str = Field(description="Why these specialties are needed")
    expected_impact: Dict[str, Any] = Field(description="Expected impact of these specialties")

class KnowledgeGraphBuilder:
    """
    Knowledge Graph Builder that connects all learnings into a searchable network.
    Builds relationships between problems, solutions, patterns, and enables cross-agent learning.
    """
    
    def __init__(self, db_logger=None, orchestrator=None):
        """
        Initialize the Knowledge Graph Builder.
        
        Args:
            db_logger: Supabase logger for persistence
            orchestrator: Agent orchestrator for specialist creation
        """
        self.db_logger = db_logger
        self.orchestrator = orchestrator
        
        # Graph storage
        self.knowledge_graph = nx.MultiDiGraph()  # Directed graph with multiple edges
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.relationships: Dict[str, KnowledgeRelationship] = {}
        self.gaps: Dict[str, KnowledgeGap] = {}
        self.learning_paths: Dict[str, LearningPath] = {}
        
        # Analysis state
        self.last_analysis: Optional[datetime] = None
        self.analysis_frequency = timedelta(hours=4)  # Analyze every 4 hours
        self.last_graph_update: Optional[datetime] = None
        
        # Graph metrics
        self.graph_metrics = {
            "nodes": 0,
            "edges": 0,
            "density": 0.0,
            "avg_clustering": 0.0,
            "connected_components": 0,
            "avg_path_length": 0.0,
            "knowledge_coverage": 0.0
        }
        
        # Initialize LLMs for different analysis types
        self.analysis_llm = ChatOpenAI(
            model="gpt-4-0125-preview",
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=3000,
        )
        
        self.relationship_llm = ChatOpenAI(
            model="gpt-4-0125-preview",
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2000,
        )
        
        self.specialty_llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0.4,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=1500,
        )
        
        # Create prompt templates
        self.analysis_prompt = self._create_analysis_prompt()
        self.relationship_prompt = self._create_relationship_prompt()
        self.specialty_prompt = self._create_specialty_prompt()
        
        # Create parsing chains
        self.analysis_parser = JsonOutputParser(pydantic_object=GraphAnalysis)
        self.specialty_parser = JsonOutputParser(pydantic_object=SpecialtyRecommendation)
        
        self.analysis_chain = self.analysis_prompt | self.analysis_llm | self.analysis_parser
        self.specialty_chain = self.specialty_prompt | self.specialty_llm | self.specialty_parser
        
        # Start continuous learning task
        self.learning_task = None
        self._start_continuous_learning()
        
        logger.info("Knowledge Graph Builder initialized with cross-agent learning capabilities")
    
    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for knowledge graph analysis."""
        
        system_template = """You are an expert knowledge graph analyst for an AI Agent Platform.
Your role is to analyze the knowledge graph to identify gaps, suggest relationships,
and find opportunities for cross-agent learning and system optimization.

ANALYSIS CAPABILITIES:
- Knowledge gap identification (missing solutions, broken paths, isolated knowledge)
- Relationship suggestion (connect related problems, solutions, patterns)
- Learning opportunity discovery (cross-agent knowledge sharing)
- Optimization path finding (most efficient problem-to-solution routes)

GRAPH STRUCTURE:
- Nodes: problems, solutions, patterns, agents, workflows, user_requests, errors, optimizations
- Relationships: solves, causes, contains, implements, triggers, optimizes, learns_from, similar_to

Focus on finding actionable insights that can improve the AI system's capabilities
and knowledge connectivity. Prioritize gaps that affect user experience or system efficiency.

Respond with a JSON object containing your analysis."""

        human_template = """Analyze this knowledge graph for improvement opportunities:

GRAPH METRICS:
{graph_metrics}

RECENT NODES (last 7 days):
{recent_nodes}

RELATIONSHIP PATTERNS:
{relationship_patterns}

ISOLATED KNOWLEDGE:
{isolated_knowledge}

FREQUENT PROBLEMS WITHOUT SOLUTIONS:
{unsolved_problems}

Please identify:
1. Critical knowledge gaps that need attention
2. Missing relationships between existing knowledge
3. Opportunities for cross-agent learning
4. Optimal paths for common problem-solution patterns
5. Areas where new agent specialties would help

Focus on actionable insights with high impact potential."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_relationship_prompt(self) -> ChatPromptTemplate:
        """Create prompt for relationship extraction."""
        
        system_template = """You are an expert at identifying relationships between knowledge elements.
Analyze the provided content to extract meaningful relationships that should be
connected in the knowledge graph.

RELATIONSHIP TYPES:
- solves: Solution addresses Problem
- causes: Problem leads to Error
- contains: Pattern includes Solution/Problem
- implements: Agent provides Solution
- triggers: User Request starts Workflow
- optimizes: Optimization improves Workflow/Agent
- learns_from: Agent gains knowledge from Pattern/Error
- similar_to: Elements share characteristics
- depends_on: Solution requires Agent/Resource
- improves: Feedback enhances Solution/Workflow
- reduces: Cost Saving affects Agent/Workflow

Return identified relationships with confidence scores."""

        human_template = """Extract relationships from this knowledge:

SOURCE CONTENT:
{source_content}

TARGET CONTENT:
{target_content}

CONTEXT:
{context}

Identify all meaningful relationships between these knowledge elements.
Include confidence scores (0.0-1.0) for each relationship."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_specialty_prompt(self) -> ChatPromptTemplate:
        """Create prompt for agent specialty recommendations."""
        
        system_template = """You are an expert at identifying gaps in AI agent capabilities.
Analyze the knowledge graph to recommend new agent specialties that would
fill critical gaps and improve system capabilities.

CURRENT AGENT TYPES:
- General agents (conversational, broad knowledge)
- Technical agents (coding, system administration)
- Research agents (information gathering, analysis)
- Universal agents (configurable specialists)

ANALYSIS FOCUS:
- Knowledge gaps without suitable agents
- High-frequency problems lacking specialized solutions
- Cross-domain knowledge that needs dedicated expertise
- User requests that consistently require escalation

Recommend practical, high-impact agent specialties."""

        human_template = """Analyze these knowledge gaps for agent specialty recommendations:

KNOWLEDGE GAPS:
{knowledge_gaps}

UNSOLVED PROBLEMS:
{unsolved_problems}

USER REQUEST PATTERNS:
{user_patterns}

AGENT PERFORMANCE GAPS:
{agent_gaps}

CROSS-DOMAIN KNOWLEDGE:
{cross_domain}

Recommend new agent specialties that would:
1. Address critical knowledge gaps
2. Solve frequent unsolved problems
3. Handle common user request patterns
4. Fill agent performance gaps
5. Bridge cross-domain knowledge

Include justification and expected impact for each recommendation."""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    async def add_knowledge_node(self, node_type: NodeType, title: str, description: str,
                                content: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None,
                                confidence: float = 0.8, tags: Optional[Set[str]] = None) -> str:
        """
        Add a new knowledge node to the graph.
        
        Args:
            node_type: Type of knowledge node
            title: Human-readable title
            description: Detailed description
            content: Node content data
            metadata: Additional metadata
            confidence: Confidence in this knowledge (0.0-1.0)
            tags: Associated tags
            
        Returns:
            Node ID if successful
        """
        try:
            node_id = str(uuid.uuid4())
            
            # Create knowledge node
            node = KnowledgeNode(
                id=node_id,
                type=node_type,
                title=title,
                description=description,
                content=content,
                metadata=metadata or {},
                confidence=confidence,
                frequency=1,
                last_seen=datetime.utcnow(),
                created_at=datetime.utcnow(),
                tags=tags or set()
            )
            
            # Add to storage
            self.nodes[node_id] = node
            self.knowledge_graph.add_node(node_id, **asdict(node))
            
            # Update metrics
            await self._update_graph_metrics()
            
            logger.info(f"Added knowledge node: {node_type.value} - {title}")
            return node_id
            
        except Exception as e:
            logger.error(f"Error adding knowledge node: {str(e)}")
            return None
    
    async def add_relationship(self, source_id: str, target_id: str, relationship_type: RelationshipType,
                              confidence: float = 0.8, evidence: Optional[List[str]] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a relationship between knowledge nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship
            confidence: Confidence in this relationship (0.0-1.0)
            evidence: Evidence supporting this relationship
            metadata: Additional metadata
            
        Returns:
            Relationship ID if successful
        """
        try:
            # Validate nodes exist
            if source_id not in self.nodes or target_id not in self.nodes:
                logger.error(f"Cannot create relationship: missing nodes {source_id} -> {target_id}")
                return None
            
            relationship_id = str(uuid.uuid4())
            
            # Determine strength based on confidence and evidence
            strength = self._calculate_relationship_strength(confidence, evidence)
            
            # Create relationship
            relationship = KnowledgeRelationship(
                id=relationship_id,
                source_id=source_id,
                target_id=target_id,
                relationship_type=relationship_type,
                strength=strength,
                confidence=confidence,
                evidence=evidence or [],
                metadata=metadata or {},
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            # Add to storage
            self.relationships[relationship_id] = relationship
            self.knowledge_graph.add_edge(
                source_id, target_id,
                key=relationship_id,
                **asdict(relationship)
            )
            
            # Update metrics
            await self._update_graph_metrics()
            
            logger.info(f"Added relationship: {relationship_type.value} ({source_id} -> {target_id})")
            return relationship_id
            
        except Exception as e:
            logger.error(f"Error adding relationship: {str(e)}")
            return None
    
    def _calculate_relationship_strength(self, confidence: float, evidence: Optional[List[str]]) -> KnowledgeStrength:
        """Calculate relationship strength based on confidence and evidence."""
        evidence_count = len(evidence) if evidence else 0
        
        # Combine confidence and evidence count
        combined_score = confidence + (evidence_count * 0.1)
        
        if combined_score >= 0.9:
            return KnowledgeStrength.VERY_STRONG
        elif combined_score >= 0.7:
            return KnowledgeStrength.STRONG
        elif combined_score >= 0.5:
            return KnowledgeStrength.MODERATE
        else:
            return KnowledgeStrength.WEAK
    
    async def find_shortest_path(self, problem_id: str, solution_type: Optional[str] = None) -> Optional[LearningPath]:
        """
        Find shortest path from a problem to solutions.
        
        Args:
            problem_id: ID of the problem node
            solution_type: Optional filter for solution type
            
        Returns:
            LearningPath if path found, None otherwise
        """
        try:
            # Find all solution nodes
            solution_nodes = [
                node_id for node_id, node in self.nodes.items()
                if node.type == NodeType.SOLUTION and
                (solution_type is None or solution_type in node.tags)
            ]
            
            if not solution_nodes:
                logger.warning(f"No solution nodes found for type: {solution_type}")
                return None
            
            best_path = None
            best_score = 0.0
            alternative_paths = []
            
            # Find paths to all solutions
            for solution_id in solution_nodes:
                try:
                    # Find shortest path
                    path = nx.shortest_path(
                        self.knowledge_graph,
                        source=problem_id,
                        target=solution_id
                    )
                    
                    if path:
                        # Calculate path efficiency
                        efficiency = self._calculate_path_efficiency(path)
                        
                        if efficiency > best_score:
                            if best_path:
                                alternative_paths.append(best_path)
                            best_path = path
                            best_score = efficiency
                        else:
                            alternative_paths.append(path)
                            
                except nx.NetworkXNoPath:
                    continue
            
            if not best_path:
                logger.warning(f"No path found from problem {problem_id} to solutions")
                return None
            
            # Create learning path
            path_id = str(uuid.uuid4())
            learning_path = LearningPath(
                id=path_id,
                source_problem=problem_id,
                target_solution=best_path[-1],
                path_nodes=best_path,
                path_relationships=self._extract_path_relationships(best_path),
                path_length=len(best_path) - 1,
                confidence=best_score,
                efficiency_score=best_score,
                alternative_paths=alternative_paths[:5],  # Keep top 5 alternatives
                created_at=datetime.utcnow()
            )
            
            self.learning_paths[path_id] = learning_path
            
            logger.info(f"Found learning path: {len(best_path)} nodes, efficiency: {best_score:.2f}")
            return learning_path
            
        except Exception as e:
            logger.error(f"Error finding shortest path: {str(e)}")
            return None
    
    def _calculate_path_efficiency(self, path: List[str]) -> float:
        """Calculate efficiency score for a knowledge path."""
        if len(path) < 2:
            return 0.0
        
        total_confidence = 0.0
        relationship_count = 0
        
        # Calculate average relationship confidence
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]
            
            # Find relationship between these nodes
            if self.knowledge_graph.has_edge(source_id, target_id):
                edge_data = self.knowledge_graph.get_edge_data(source_id, target_id)
                for key, rel_data in edge_data.items():
                    total_confidence += rel_data.get('confidence', 0.5)
                    relationship_count += 1
        
        if relationship_count == 0:
            return 0.0
        
        avg_confidence = total_confidence / relationship_count
        
        # Penalize longer paths
        length_penalty = 1.0 / (1.0 + (len(path) - 2) * 0.1)
        
        return avg_confidence * length_penalty
    
    def _extract_path_relationships(self, path: List[str]) -> List[str]:
        """Extract relationship IDs along a path."""
        relationships = []
        
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]
            
            if self.knowledge_graph.has_edge(source_id, target_id):
                edge_data = self.knowledge_graph.get_edge_data(source_id, target_id)
                for key, rel_data in edge_data.items():
                    relationships.append(key)
                    break  # Take first relationship
        
        return relationships
    
    async def identify_knowledge_gaps(self) -> List[KnowledgeGap]:
        """
        Identify gaps in the knowledge graph.
        
        Returns:
            List of identified knowledge gaps
        """
        try:
            gaps = []
            
            # Find isolated nodes
            isolated_nodes = [
                node_id for node_id in self.knowledge_graph.nodes()
                if self.knowledge_graph.degree(node_id) == 0
            ]
            
            if isolated_nodes:
                gap = KnowledgeGap(
                    id=str(uuid.uuid4()),
                    gap_type="isolated_knowledge",
                    description=f"Found {len(isolated_nodes)} isolated knowledge nodes",
                    affected_areas=[self.nodes[node_id].type.value for node_id in isolated_nodes[:10]],
                    severity="medium",
                    suggested_actions=[
                        "Connect isolated nodes to relevant knowledge",
                        "Review node creation process for better integration"
                    ],
                    potential_benefits={"knowledge_connectivity": 0.3, "system_efficiency": 0.2},
                    priority_score=0.6,
                    created_at=datetime.utcnow()
                )
                gaps.append(gap)
            
            # Find problems without solutions
            problem_nodes = [
                node_id for node_id, node in self.nodes.items()
                if node.type == NodeType.PROBLEM
            ]
            
            unsolved_problems = []
            for problem_id in problem_nodes:
                # Check if there's a path to any solution
                solution_nodes = [
                    node_id for node_id, node in self.nodes.items()
                    if node.type == NodeType.SOLUTION
                ]
                
                has_solution = False
                for solution_id in solution_nodes:
                    try:
                        nx.shortest_path(self.knowledge_graph, problem_id, solution_id)
                        has_solution = True
                        break
                    except nx.NetworkXNoPath:
                        continue
                
                if not has_solution:
                    unsolved_problems.append(problem_id)
            
            if unsolved_problems:
                gap = KnowledgeGap(
                    id=str(uuid.uuid4()),
                    gap_type="missing_solution",
                    description=f"Found {len(unsolved_problems)} problems without solution paths",
                    affected_areas=[self.nodes[problem_id].title for problem_id in unsolved_problems[:5]],
                    severity="high" if len(unsolved_problems) > 10 else "medium",
                    suggested_actions=[
                        "Create solutions for frequent problems",
                        "Identify agent gaps for unsolved problem domains"
                    ],
                    potential_benefits={"user_satisfaction": 0.5, "problem_resolution": 0.7},
                    priority_score=0.8,
                    created_at=datetime.utcnow()
                )
                gaps.append(gap)
            
            # Find broken knowledge paths
            weak_connections = [
                rel_id for rel_id, rel in self.relationships.items()
                if rel.strength == KnowledgeStrength.WEAK and
                rel.confidence < 0.3
            ]
            
            if weak_connections:
                gap = KnowledgeGap(
                    id=str(uuid.uuid4()),
                    gap_type="broken_path",
                    description=f"Found {len(weak_connections)} weak knowledge connections",
                    affected_areas=["knowledge_connectivity"],
                    severity="low",
                    suggested_actions=[
                        "Strengthen weak relationships with more evidence",
                        "Review and validate low-confidence connections"
                    ],
                    potential_benefits={"knowledge_reliability": 0.4},
                    priority_score=0.4,
                    created_at=datetime.utcnow()
                )
                gaps.append(gap)
            
            # Store gaps
            for gap in gaps:
                self.gaps[gap.id] = gap
            
            logger.info(f"Identified {len(gaps)} knowledge gaps")
            return gaps
            
        except Exception as e:
            logger.error(f"Error identifying knowledge gaps: {str(e)}")
            return []
    
    async def suggest_agent_specialties(self) -> List[Dict[str, Any]]:
        """
        Suggest new agent specialties based on knowledge gaps.
        
        Returns:
            List of agent specialty recommendations
        """
        try:
            # Gather gap analysis data
            gaps = await self.identify_knowledge_gaps()
            
            # Prepare context for LLM
            gap_summary = self._prepare_gap_summary(gaps)
            unsolved_problems = self._get_unsolved_problems()
            user_patterns = await self._get_user_request_patterns()
            agent_gaps = await self._get_agent_performance_gaps()
            cross_domain = self._get_cross_domain_knowledge()
            
            # Generate specialty recommendations
            analysis_result = await self.specialty_chain.ainvoke({
                "knowledge_gaps": gap_summary,
                "unsolved_problems": unsolved_problems,
                "user_patterns": user_patterns,
                "agent_gaps": agent_gaps,
                "cross_domain": cross_domain
            })
            
            recommendations = analysis_result.get("recommended_specialties", [])
            
            logger.info(f"Generated {len(recommendations)} agent specialty recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error suggesting agent specialties: {str(e)}")
            return []
    
    def _prepare_gap_summary(self, gaps: List[KnowledgeGap]) -> str:
        """Prepare knowledge gaps summary for LLM analysis."""
        if not gaps:
            return "No significant knowledge gaps identified."
        
        summary = "IDENTIFIED KNOWLEDGE GAPS:\n\n"
        for gap in sorted(gaps, key=lambda x: x.priority_score, reverse=True)[:10]:
            summary += f"- {gap.gap_type.upper()}: {gap.description}\n"
            summary += f"  Severity: {gap.severity}, Priority: {gap.priority_score:.2f}\n"
            summary += f"  Affected Areas: {', '.join(gap.affected_areas[:3])}\n\n"
        
        return summary
    
    def _get_unsolved_problems(self) -> str:
        """Get summary of unsolved problems."""
        problem_nodes = [
            node for node in self.nodes.values()
            if node.type == NodeType.PROBLEM
        ]
        
        if not problem_nodes:
            return "No unresolved problems found."
        
        # Sort by frequency (most common problems first)
        sorted_problems = sorted(problem_nodes, key=lambda x: x.frequency, reverse=True)
        
        summary = "FREQUENT UNSOLVED PROBLEMS:\n\n"
        for problem in sorted_problems[:10]:
            summary += f"- {problem.title} (frequency: {problem.frequency})\n"
            summary += f"  {problem.description}\n\n"
        
        return summary
    
    async def _get_user_request_patterns(self) -> str:
        """Get user request patterns from the database."""
        try:
            if not self.db_logger:
                return "User request patterns not available."
            
            # Query recent messages for patterns
            recent_messages = await self.db_logger.client.table("messages").select(
                "content, agent_type, message_type, created_at"
            ).gte(
                "created_at", 
                (datetime.utcnow() - timedelta(days=30)).isoformat()
            ).eq("message_type", "user_message").limit(500).execute()
            
            if not recent_messages.data:
                return "No recent user messages found."
            
            # Analyze patterns (simplified)
            summary = "USER REQUEST PATTERNS:\n\n"
            summary += f"- Total user messages: {len(recent_messages.data)}\n"
            summary += "- Common request types will be analyzed for gaps\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting user request patterns: {str(e)}")
            return "User request patterns not available."
    
    async def _get_agent_performance_gaps(self) -> str:
        """Get agent performance gaps from the database."""
        try:
            if not self.db_logger:
                return "Agent performance data not available."
            
            # Query agent metrics for gaps
            agent_metrics = await self.db_logger.client.table("agent_metrics").select(
                "agent_name, success_rate, error_count, escalation_count"
            ).gte(
                "last_updated",
                (datetime.utcnow() - timedelta(days=7)).isoformat()
            ).execute()
            
            if not agent_metrics.data:
                return "No recent agent metrics found."
            
            summary = "AGENT PERFORMANCE GAPS:\n\n"
            
            # Find agents with low success rates or high escalations
            for metric in agent_metrics.data:
                if metric.get("success_rate", 1.0) < 0.8 or metric.get("escalation_count", 0) > 5:
                    summary += f"- {metric['agent_name']}: "
                    summary += f"Success rate: {metric.get('success_rate', 0):.1%}, "
                    summary += f"Escalations: {metric.get('escalation_count', 0)}\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting agent performance gaps: {str(e)}")
            return "Agent performance data not available."
    
    def _get_cross_domain_knowledge(self) -> str:
        """Get cross-domain knowledge analysis."""
        # Analyze node types and their connections
        domain_connections = defaultdict(set)
        
        for rel in self.relationships.values():
            source_type = self.nodes[rel.source_id].type.value
            target_type = self.nodes[rel.target_id].type.value
            domain_connections[source_type].add(target_type)
        
        summary = "CROSS-DOMAIN KNOWLEDGE:\n\n"
        for domain, connected_domains in domain_connections.items():
            summary += f"- {domain}: connected to {', '.join(connected_domains)}\n"
        
        return summary
    
    async def _update_graph_metrics(self):
        """Update graph metrics for analysis."""
        try:
            self.graph_metrics = {
                "nodes": self.knowledge_graph.number_of_nodes(),
                "edges": self.knowledge_graph.number_of_edges(),
                "density": nx.density(self.knowledge_graph) if self.knowledge_graph.number_of_nodes() > 1 else 0.0,
                "connected_components": nx.number_weakly_connected_components(self.knowledge_graph),
                "avg_path_length": 0.0,  # Will calculate if needed
                "knowledge_coverage": self._calculate_knowledge_coverage()
            }
            
            self.last_graph_update = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating graph metrics: {str(e)}")
    
    def _calculate_knowledge_coverage(self) -> float:
        """Calculate how well the knowledge graph covers different domains."""
        node_types = set(node.type.value for node in self.nodes.values())
        total_types = len(NodeType)
        
        return len(node_types) / total_types if total_types > 0 else 0.0
    
    async def analyze_graph(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the knowledge graph.
        
        Returns:
            Analysis results with gaps, suggestions, and optimizations
        """
        try:
            # Update metrics first
            await self._update_graph_metrics()
            
            # Identify gaps
            gaps = await self.identify_knowledge_gaps()
            
            # Prepare analysis context
            recent_nodes = self._get_recent_nodes(days=7)
            relationship_patterns = self._analyze_relationship_patterns()
            isolated_knowledge = self._find_isolated_knowledge()
            unsolved_problems = self._get_unsolved_problems()
            
            # Perform LLM analysis
            analysis_result = await self.analysis_chain.ainvoke({
                "graph_metrics": json.dumps(self.graph_metrics, indent=2),
                "recent_nodes": recent_nodes,
                "relationship_patterns": relationship_patterns,
                "isolated_knowledge": isolated_knowledge,
                "unsolved_problems": unsolved_problems
            })
            
            # Get agent specialty recommendations
            specialty_recommendations = await self.suggest_agent_specialties()
            
            analysis_summary = {
                "graph_metrics": self.graph_metrics,
                "knowledge_gaps": [asdict(gap) for gap in gaps],
                "llm_analysis": analysis_result,
                "specialty_recommendations": specialty_recommendations,
                "learning_paths": len(self.learning_paths),
                "last_analysis": datetime.utcnow().isoformat(),
                "recommendations_summary": {
                    "critical_gaps": len([g for g in gaps if g.severity == "critical"]),
                    "new_specialties": len(specialty_recommendations),
                    "improvement_opportunities": len(analysis_result.get("optimization_paths", []))
                }
            }
            
            self.last_analysis = datetime.utcnow()
            
            # Persist analysis to database
            await self._persist_analysis_to_db(analysis_summary)
            
            logger.info("Completed comprehensive knowledge graph analysis")
            return analysis_summary
            
        except Exception as e:
            logger.error(f"Error analyzing knowledge graph: {str(e)}")
            return {"error": str(e), "last_analysis": None}
    
    def _get_recent_nodes(self, days: int) -> str:
        """Get summary of recently added nodes."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_nodes = [
            node for node in self.nodes.values()
            if node.created_at >= cutoff_date
        ]
        
        if not recent_nodes:
            return "No recent nodes added."
        
        summary = f"RECENT NODES (last {days} days):\n\n"
        for node in sorted(recent_nodes, key=lambda x: x.created_at, reverse=True)[:20]:
            summary += f"- {node.type.value.upper()}: {node.title}\n"
            summary += f"  Confidence: {node.confidence:.2f}, Frequency: {node.frequency}\n\n"
        
        return summary
    
    def _analyze_relationship_patterns(self) -> str:
        """Analyze patterns in relationships."""
        relationship_counts = Counter(rel.relationship_type.value for rel in self.relationships.values())
        strength_counts = Counter(rel.strength.value for rel in self.relationships.values())
        
        summary = "RELATIONSHIP PATTERNS:\n\n"
        summary += "Relationship Types:\n"
        for rel_type, count in relationship_counts.most_common(10):
            summary += f"- {rel_type}: {count}\n"
        
        summary += "\nRelationship Strengths:\n"
        for strength, count in strength_counts.most_common():
            summary += f"- {strength}: {count}\n"
        
        return summary
    
    def _find_isolated_knowledge(self) -> str:
        """Find isolated knowledge nodes."""
        isolated = []
        for node_id, node in self.nodes.items():
            if self.knowledge_graph.degree(node_id) == 0:
                isolated.append(node)
        
        if not isolated:
            return "No isolated knowledge found."
        
        summary = f"ISOLATED KNOWLEDGE ({len(isolated)} nodes):\n\n"
        for node in isolated[:10]:
            summary += f"- {node.type.value.upper()}: {node.title}\n"
            summary += f"  Created: {node.created_at.strftime('%Y-%m-%d')}\n\n"
        
        return summary
    
    async def _persist_analysis_to_db(self, analysis_data: Dict[str, Any]):
        """Persist analysis results to Supabase."""
        try:
            if not self.db_logger:
                return
            
            # Store in a knowledge_graph_analysis table (would need migration)
            analysis_record = {
                "id": str(uuid.uuid4()),
                "analysis_data": analysis_data,
                "graph_metrics": self.graph_metrics,
                "node_count": len(self.nodes),
                "relationship_count": len(self.relationships),
                "gap_count": len([g for g in self.gaps.values() if g.created_at >= datetime.utcnow() - timedelta(hours=24)]),
                "created_at": datetime.utcnow().isoformat()
            }
            
            # For now, log the event since the table doesn't exist yet
            await self.db_logger.log_event(
                "knowledge_graph_analysis",
                analysis_record,
                user_id="system"
            )
            
            logger.info("Persisted knowledge graph analysis to database")
            
        except Exception as e:
            logger.error(f"Error persisting analysis to database: {str(e)}")
    
    def _start_continuous_learning(self):
        """Start continuous learning task."""
        async def learning_loop():
            while True:
                try:
                    await asyncio.sleep(self.analysis_frequency.total_seconds())
                    
                    # Perform periodic analysis
                    if (not self.last_analysis or 
                        datetime.utcnow() - self.last_analysis >= self.analysis_frequency):
                        
                        await self.analyze_graph()
                        
                except Exception as e:
                    logger.error(f"Error in knowledge graph learning loop: {str(e)}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retrying
        
        self.learning_task = asyncio.create_task(learning_loop())
    
    async def integrate_workflow_data(self, workflow_data: Dict[str, Any]):
        """
        Integrate workflow analysis data into the knowledge graph.
        
        Args:
            workflow_data: Data from workflow analysis or other improvement agents
        """
        try:
            # Extract patterns and create nodes
            if "patterns" in workflow_data:
                for pattern_data in workflow_data["patterns"]:
                    await self.add_knowledge_node(
                        NodeType.PATTERN,
                        pattern_data.get("name", "Unknown Pattern"),
                        pattern_data.get("description", ""),
                        pattern_data,
                        confidence=pattern_data.get("confidence", 0.8),
                        tags=set(pattern_data.get("tags", []))
                    )
            
            # Extract problems and solutions
            if "problems" in workflow_data:
                for problem_data in workflow_data["problems"]:
                    await self.add_knowledge_node(
                        NodeType.PROBLEM,
                        problem_data.get("title", "Unknown Problem"),
                        problem_data.get("description", ""),
                        problem_data,
                        confidence=problem_data.get("confidence", 0.8)
                    )
            
            if "solutions" in workflow_data:
                for solution_data in workflow_data["solutions"]:
                    await self.add_knowledge_node(
                        NodeType.SOLUTION,
                        solution_data.get("title", "Unknown Solution"),
                        solution_data.get("description", ""),
                        solution_data,
                        confidence=solution_data.get("confidence", 0.8)
                    )
            
            logger.info("Integrated workflow data into knowledge graph")
            
        except Exception as e:
            logger.error(f"Error integrating workflow data: {str(e)}")
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge graph state."""
        return {
            "nodes": len(self.nodes),
            "relationships": len(self.relationships),
            "gaps": len(self.gaps),
            "learning_paths": len(self.learning_paths),
            "graph_metrics": self.graph_metrics,
            "last_analysis": self.last_analysis.isoformat() if self.last_analysis else None,
            "node_types": {
                node_type.value: len([n for n in self.nodes.values() if n.type == node_type])
                for node_type in NodeType
            },
            "relationship_types": {
                rel_type.value: len([r for r in self.relationships.values() if r.relationship_type == rel_type])
                for rel_type in RelationshipType
            }
        }
    
    async def search_knowledge(self, query: str, node_types: Optional[List[NodeType]] = None,
                              limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph for relevant information.
        
        Args:
            query: Search query
            node_types: Optional filter by node types
            limit: Maximum results to return
            
        Returns:
            List of matching knowledge nodes
        """
        try:
            query_lower = query.lower()
            results = []
            
            for node in self.nodes.values():
                # Filter by node types if specified
                if node_types and node.type not in node_types:
                    continue
                
                # Calculate relevance score
                score = 0.0
                
                # Title match (higher weight)
                if query_lower in node.title.lower():
                    score += 2.0
                
                # Description match
                if query_lower in node.description.lower():
                    score += 1.0
                
                # Tag match
                for tag in node.tags:
                    if query_lower in tag.lower():
                        score += 1.5
                
                # Content match (lower weight)
                content_str = json.dumps(node.content).lower()
                if query_lower in content_str:
                    score += 0.5
                
                # Weight by confidence and frequency
                score *= (node.confidence * 0.8 + min(node.frequency / 10.0, 1.0) * 0.2)
                
                if score > 0:
                    results.append({
                        "node": asdict(node),
                        "relevance_score": score
                    })
            
            # Sort by relevance score and limit results
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching knowledge graph: {str(e)}")
            return []
    
    async def close(self):
        """Clean up resources."""
        if self.learning_task:
            self.learning_task.cancel()
            try:
                await self.learning_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Knowledge Graph Builder closed") 