"""
Dynamic Prompt Manager

Enables agents to load and optimize prompts dynamically from Supabase.
Supports A/B testing, performance tracking, and learning-driven improvements.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from storage.supabase import SupabaseLogger

logger = logging.getLogger(__name__)

class ComplexityLevel(Enum):
    """Complexity levels for prompt selection."""
    SIMPLE = "simple"
    MEDIUM = "medium" 
    COMPLEX = "complex"
    ADAPTIVE = "adaptive"  # Adapts based on context

@dataclass
class PromptTemplate:
    """Represents a dynamic prompt template."""
    template_id: str
    agent_type: str
    specialty: Optional[str]
    complexity_level: ComplexityLevel
    system_prompt: str
    tool_decision_guidance: str
    communication_style: str
    tool_selection_criteria: str
    performance_score: float
    usage_count: int
    test_group: str
    created_at: datetime

class DynamicPromptManager:
    """
    Manages dynamic, learning-driven agent prompts from Supabase.
    
    Features:
    - Dynamic prompt loading based on agent type and context
    - Performance tracking and optimization
    - A/B testing capabilities
    - Learning-driven prompt improvements
    """
    
    def __init__(self, supabase_logger: Optional[SupabaseLogger] = None):
        """Initialize the dynamic prompt manager."""
        self.supabase_logger = supabase_logger or SupabaseLogger()
        self.prompt_cache: Dict[str, PromptTemplate] = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        self.last_cache_update = {}
        
        logger.info("Dynamic Prompt Manager initialized")
    
    async def get_optimal_prompt(self, 
                                agent_type: str,
                                specialty: Optional[str] = None,
                                complexity_level: ComplexityLevel = ComplexityLevel.MEDIUM,
                                context: Optional[Dict[str, Any]] = None) -> PromptTemplate:
        """
        Get the optimal prompt template for given parameters.
        
        Args:
            agent_type: Type of agent (e.g., 'universal', 'specialized')
            specialty: Agent specialty (e.g., 'research', 'technical')
            complexity_level: Complexity of the task
            context: Additional context for prompt selection
            
        Returns:
            PromptTemplate with optimal prompt configuration
        """
        try:
            # Create cache key
            cache_key = f"{agent_type}_{specialty}_{complexity_level.value}"
            
            # Check cache first
            if self._is_cache_valid(cache_key):
                logger.debug(f"Using cached prompt for {cache_key}")
                return self.prompt_cache[cache_key]
            
            # Determine test group (could be user-specific or random)
            test_group = self._determine_test_group(context)
            
            # Query Supabase for optimal prompt
            result = self.supabase_logger.client.rpc(
                'get_optimal_prompt',
                {
                    'p_agent_type': agent_type,
                    'p_specialty': specialty,
                    'p_complexity_level': complexity_level.value,
                    'p_test_group': test_group
                }
            ).execute()
            
            if result.data and len(result.data) > 0:
                prompt_data = result.data[0]
                prompt_template = PromptTemplate(
                    template_id=prompt_data['template_id'],
                    agent_type=agent_type,
                    specialty=specialty,
                    complexity_level=complexity_level,
                    system_prompt=prompt_data['system_prompt'],
                    tool_decision_guidance=prompt_data['tool_decision_guidance'] or "",
                    communication_style=prompt_data['communication_style'] or "",
                    tool_selection_criteria=prompt_data['tool_selection_criteria'] or "",
                    performance_score=float(prompt_data['performance_score'] or 0.5),
                    usage_count=0,  # Will be updated separately
                    test_group=test_group,
                    created_at=datetime.utcnow()
                )
                
                # Cache the result
                self.prompt_cache[cache_key] = prompt_template
                self.last_cache_update[cache_key] = datetime.utcnow()
                
                logger.info(f"✅ Loaded dynamic prompt for {agent_type}/{specialty} "
                           f"(score: {prompt_template.performance_score:.2f})")
                
                return prompt_template
            else:
                logger.warning(f"No prompt found in database, using fallback for {cache_key}")
                return self._create_fallback_prompt(agent_type, specialty, complexity_level)
                
        except Exception as e:
            logger.error(f"Error loading dynamic prompt: {e}")
            return self._create_fallback_prompt(agent_type, specialty, complexity_level)
    
    async def log_prompt_performance(self,
                                   template_id: str,
                                   user_message: str,
                                   complexity_score: float,
                                   tool_decision_accuracy: float,
                                   response_quality_score: float,
                                   tools_invoked: List[str] = None,
                                   mcp_solutions_found: int = 0,
                                   user_satisfaction: Optional[float] = None) -> str:
        """
        Log the performance of a prompt template.
        
        Args:
            template_id: ID of the prompt template used
            user_message: Original user message (preview)
            complexity_score: Complexity score of the query
            tool_decision_accuracy: How well tools were selected (0.0-1.0)
            response_quality_score: Quality of the response (0.0-1.0)
            tools_invoked: List of tools that were used
            mcp_solutions_found: Number of MCP solutions found
            user_satisfaction: User satisfaction score if available
            
        Returns:
            Log entry ID
        """
        try:
            # Use the stored procedure to log performance
            result = self.supabase_logger.client.rpc(
                'log_prompt_performance',
                {
                    'p_template_id': template_id,
                    'p_user_message_preview': user_message[:200],  # Truncate for privacy
                    'p_complexity_score': complexity_score,
                    'p_tool_decision_accuracy': tool_decision_accuracy,
                    'p_response_quality_score': response_quality_score,
                    'p_tools_invoked': tools_invoked or [],
                    'p_mcp_solutions_found': mcp_solutions_found
                }
            ).execute()
            
            if result.data:
                log_id = result.data[0] if isinstance(result.data, list) else result.data
                
                # Also log user satisfaction if provided
                if user_satisfaction is not None:
                    await self._update_user_satisfaction(log_id, user_satisfaction)
                
                logger.debug(f"📊 Logged prompt performance: {template_id} "
                           f"(tool_accuracy: {tool_decision_accuracy:.2f}, "
                           f"quality: {response_quality_score:.2f})")
                
                # Clear cache to force refresh of performance scores
                self._invalidate_cache()
                
                return str(log_id)
            else:
                logger.error("Failed to log prompt performance - no result returned")
                return ""
                
        except Exception as e:
            logger.error(f"Error logging prompt performance: {e}")
            return ""
    
    async def generate_optimization_insights(self, agent_type: str, specialty: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate insights for prompt optimization based on performance data.
        
        Args:
            agent_type: Type of agent to analyze
            specialty: Specific specialty to analyze
            
        Returns:
            List of optimization insights
        """
        try:
            # Query performance logs for analysis
            query = self.supabase_logger.client.table("prompt_performance_logs")\
                .select("""
                    *,
                    agent_prompt_templates(agent_type, specialty, system_prompt, tool_decision_guidance)
                """)\
                .gte("created_at", (datetime.utcnow() - timedelta(days=30)).isoformat())
            
            if agent_type:
                query = query.eq("agent_prompt_templates.agent_type", agent_type)
            if specialty:
                query = query.eq("agent_prompt_templates.specialty", specialty)
            
            result = query.execute()
            
            if not result.data:
                return []
            
            insights = []
            
            # Analyze tool decision patterns
            tool_insights = self._analyze_tool_decision_patterns(result.data)
            insights.extend(tool_insights)
            
            # Analyze communication effectiveness
            communication_insights = self._analyze_communication_patterns(result.data)
            insights.extend(communication_insights)
            
            # Analyze complexity handling
            complexity_insights = self._analyze_complexity_handling(result.data)
            insights.extend(complexity_insights)
            
            logger.info(f"Generated {len(insights)} optimization insights for {agent_type}/{specialty}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating optimization insights: {e}")
            return []
    
    async def create_prompt_variant(self,
                                  base_template_id: str,
                                  variant_name: str,
                                  modifications: Dict[str, str],
                                  test_group: str = "variant_a") -> str:
        """
        Create a new prompt variant for A/B testing.
        
        Args:
            base_template_id: ID of the base template to modify
            variant_name: Name for the new variant
            modifications: Dictionary of fields to modify
            test_group: Test group identifier
            
        Returns:
            New template ID
        """
        try:
            # Get base template
            base_result = self.supabase_logger.client.table("agent_prompt_templates")\
                .select("*")\
                .eq("id", base_template_id)\
                .execute()
            
            if not base_result.data:
                raise ValueError(f"Base template {base_template_id} not found")
            
            base_template = base_result.data[0]
            
            # Create new template with modifications
            new_template = base_template.copy()
            new_template.update(modifications)
            new_template.update({
                'prompt_version': f"{base_template['prompt_version']}_variant_{variant_name}",
                'test_group': test_group,
                'performance_score': 0.5,  # Start with neutral score
                'usage_count': 0,
                'created_at': datetime.utcnow().isoformat(),
                'created_by': 'prompt_optimization_system'
            })
            
            # Remove ID so a new one is generated
            del new_template['id']
            
            # Insert new variant
            result = self.supabase_logger.client.table("agent_prompt_templates")\
                .insert(new_template)\
                .execute()
            
            if result.data:
                new_id = result.data[0]['id']
                logger.info(f"✅ Created prompt variant {variant_name}: {new_id}")
                return new_id
            else:
                raise Exception("Failed to create prompt variant")
                
        except Exception as e:
            logger.error(f"Error creating prompt variant: {e}")
            raise
    
    def _determine_test_group(self, context: Optional[Dict[str, Any]]) -> str:
        """Determine which test group to use for prompt selection."""
        if not context:
            return 'control'
        
        # Could implement user-based assignment, random assignment, etc.
        user_id = context.get('user_id', '')
        
        # Simple hash-based assignment for consistent user experience
        if user_id:
            hash_val = hash(user_id) % 100
            if hash_val < 20:  # 20% get variant_a
                return 'variant_a'
            elif hash_val < 30:  # 10% get variant_b  
                return 'variant_b'
        
        return 'control'  # 70% get control
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached prompt is still valid."""
        if cache_key not in self.prompt_cache:
            return False
        
        last_update = self.last_cache_update.get(cache_key)
        if not last_update:
            return False
        
        age_seconds = (datetime.utcnow() - last_update).total_seconds()
        return age_seconds < self.cache_ttl
    
    def _invalidate_cache(self):
        """Invalidate the entire prompt cache."""
        self.prompt_cache.clear()
        self.last_cache_update.clear()
    
    def _create_fallback_prompt(self, agent_type: str, specialty: Optional[str], complexity_level: ComplexityLevel) -> PromptTemplate:
        """Create a fallback prompt when database is unavailable."""
        
        # Basic fallback prompts
        fallback_prompts = {
            'universal': {
                'system_prompt': 'You are a versatile AI assistant with access to various tools and capabilities.',
                'tool_decision_guidance': 'Assess if the task requires external tools. Use MCP tools when available. Be transparent about tool usage.',
                'communication_style': 'Professional yet approachable.',
                'tool_selection_criteria': 'Prioritize existing tools over custom solutions.'
            },
            'research': {
                'system_prompt': 'You are a research specialist with access to information gathering tools.',
                'tool_decision_guidance': 'Use web search for current information. Verify sources.',
                'communication_style': 'Analytical and thorough.',
                'tool_selection_criteria': 'Prioritize real-time data sources.'
            }
        }
        
        prompt_config = fallback_prompts.get(agent_type, fallback_prompts['universal'])
        
        return PromptTemplate(
            template_id='fallback',
            agent_type=agent_type,
            specialty=specialty,
            complexity_level=complexity_level,
            system_prompt=prompt_config['system_prompt'],
            tool_decision_guidance=prompt_config['tool_decision_guidance'],
            communication_style=prompt_config['communication_style'],
            tool_selection_criteria=prompt_config['tool_selection_criteria'],
            performance_score=0.5,
            usage_count=0,
            test_group='fallback',
            created_at=datetime.utcnow()
        )
    
    def _analyze_tool_decision_patterns(self, performance_data: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze tool decision patterns to generate insights."""
        insights = []
        
        # Example analysis: Low tool decision accuracy
        low_accuracy_sessions = [p for p in performance_data if p.get('tool_decision_accuracy', 1.0) < 0.7]
        
        if len(low_accuracy_sessions) > len(performance_data) * 0.3:  # More than 30% low accuracy
            insights.append({
                'insight_type': 'tool_selection_improvement',
                'insight_description': 'Tool decision accuracy is below optimal. Consider enhancing tool selection criteria in prompts.',
                'confidence_score': 0.8,
                'suggested_prompt_changes': {
                    'tool_decision_guidance': 'Enhanced tool selection framework with clearer decision trees'
                },
                'sample_count': len(low_accuracy_sessions)
            })
        
        return insights
    
    def _analyze_communication_patterns(self, performance_data: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze communication effectiveness patterns."""
        insights = []
        
        # Example analysis: Response quality issues
        low_quality_responses = [p for p in performance_data if p.get('response_quality_score', 1.0) < 0.6]
        
        if len(low_quality_responses) > len(performance_data) * 0.25:  # More than 25% low quality
            insights.append({
                'insight_type': 'communication_optimization',
                'insight_description': 'Response quality could be improved. Consider adjusting communication style prompts.',
                'confidence_score': 0.75,
                'suggested_prompt_changes': {
                    'communication_style': 'More structured and detailed responses with clear explanations'
                },
                'sample_count': len(low_quality_responses)
            })
        
        return insights
    
    def _analyze_complexity_handling(self, performance_data: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze how well different complexity levels are handled."""
        insights = []
        
        # Group by complexity score
        high_complexity = [p for p in performance_data if p.get('complexity_score', 0.5) > 0.7]
        low_performance_high_complexity = [p for p in high_complexity 
                                         if p.get('response_quality_score', 1.0) < 0.6]
        
        if len(low_performance_high_complexity) > len(high_complexity) * 0.4:  # 40% of high complexity queries underperform
            insights.append({
                'insight_type': 'complexity_handling_improvement',
                'insight_description': 'High complexity queries are underperforming. Consider specialized prompts for complex scenarios.',
                'confidence_score': 0.85,
                'suggested_prompt_changes': {
                    'system_prompt': 'Enhanced instructions for handling complex, multi-step requests'
                },
                'sample_count': len(low_performance_high_complexity)
            })
        
        return insights
    
    async def _update_user_satisfaction(self, log_id: str, satisfaction_score: float):
        """Update user satisfaction score for a logged performance entry."""
        try:
            self.supabase_logger.client.table("prompt_performance_logs")\
                .update({'user_satisfaction_score': satisfaction_score})\
                .eq('id', log_id)\
                .execute()
        except Exception as e:
            logger.error(f"Error updating user satisfaction: {e}")


# Global instance for easy access
_prompt_manager = None

def get_prompt_manager() -> DynamicPromptManager:
    """Get the global prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = DynamicPromptManager()
    return _prompt_manager 