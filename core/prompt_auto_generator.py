"""
Prompt Auto-Generator

Automatically creates and optimizes prompts for agents based on their specialty and performance data.
Integrates with the DynamicPromptManager for learning-driven prompt evolution.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from storage.supabase import SupabaseLogger

logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    """Template for generating prompts."""
    agent_type: str
    specialty: str
    system_prompt: str
    tool_decision_guidance: str
    communication_style: str
    tool_selection_criteria: str
    complexity_level: str = "medium"

class PromptAutoGenerator:
    """
    Automatically generates and optimizes prompts for agents.
    
    Features:
    - Auto-generate prompts for new agent types
    - Optimize existing prompts based on performance data  
    - A/B test different prompt variations
    - Learn from successful patterns across agents
    """
    
    def __init__(self, supabase_logger: Optional[SupabaseLogger] = None, prompt_manager=None):
        """Initialize the prompt auto-generator."""
        self.supabase_logger = supabase_logger or SupabaseLogger()
        self.prompt_manager = prompt_manager
        
        # LLM for prompt generation
        self.prompt_llm = ChatOpenAI(
            model="gpt-4-0125-preview",  # Use GPT-4 for better prompt generation
            temperature=0.3,  # Lower temperature for more consistent prompts
            max_tokens=1000
        )
        
        # Template for prompt generation
        self.generation_prompt = self._create_generation_prompt()
        
        logger.info("✅ PromptAutoGenerator initialized")
    
    async def generate_prompt_for_agent(self, 
                                      agent_type: str,
                                      specialty: str,
                                      complexity_level: str = "medium",
                                      context: Optional[Dict[str, Any]] = None) -> PromptTemplate:
        """
        Generate a new prompt for an agent type and specialty.
        
        Args:
            agent_type: Type of agent (e.g., 'universal', 'specialized')
            specialty: Agent specialty (e.g., 'technical', 'research', 'customer_support')
            complexity_level: Target complexity level
            context: Additional context for prompt generation
            
        Returns:
            Generated PromptTemplate
        """
        try:
            # Check if prompt already exists and is performing well
            existing_prompt = await self._get_existing_prompt(agent_type, specialty, complexity_level)
            if existing_prompt and existing_prompt.get('performance_score', 0) > 0.8:
                logger.info(f"🎯 Using high-performing existing prompt for {agent_type}/{specialty}")
                return self._convert_to_template(existing_prompt, agent_type, specialty, complexity_level)
            
            # Gather context for prompt generation
            generation_context = await self._gather_generation_context(agent_type, specialty, context)
            
            # Generate the prompt using LLM
            generation_chain = self.generation_prompt | self.prompt_llm
            
            response = await generation_chain.ainvoke({
                "agent_type": agent_type,
                "specialty": specialty,
                "complexity_level": complexity_level,
                "context": json.dumps(generation_context, indent=2),
                "existing_patterns": await self._get_successful_patterns(specialty)
            })
            
            # Parse the generated prompt
            prompt_data = self._parse_generated_prompt(response.content)
            
            # Create template
            template = PromptTemplate(
                agent_type=agent_type,
                specialty=specialty,
                system_prompt=prompt_data.get('system_prompt', ''),
                tool_decision_guidance=prompt_data.get('tool_decision_guidance', ''),
                communication_style=prompt_data.get('communication_style', ''),
                tool_selection_criteria=prompt_data.get('tool_selection_criteria', ''),
                complexity_level=complexity_level
            )
            
            # Save to database
            await self._save_generated_prompt(template)
            
            logger.info(f"✅ Generated new prompt for {agent_type}/{specialty} ({complexity_level})")
            return template
            
        except Exception as e:
            logger.error(f"Failed to generate prompt for {agent_type}/{specialty}: {e}")
            return self._create_fallback_template(agent_type, specialty, complexity_level)
    
    async def optimize_existing_prompt(self, 
                                     template_id: str,
                                     performance_data: Dict[str, Any]) -> Optional[PromptTemplate]:
        """
        Optimize an existing prompt based on performance data.
        
        Args:
            template_id: ID of the prompt template to optimize
            performance_data: Performance metrics and feedback
            
        Returns:
            Optimized PromptTemplate or None if optimization failed
        """
        try:
            # Get current prompt
            current_prompt = await self._get_prompt_by_id(template_id)
            if not current_prompt:
                logger.warning(f"Prompt {template_id} not found for optimization")
                return None
            
            # Analyze performance issues
            optimization_insights = await self._analyze_performance_issues(current_prompt, performance_data)
            
            # Generate optimized version
            optimized_prompt = await self._generate_optimized_version(current_prompt, optimization_insights)
            
            # Save as new variant for A/B testing
            await self._save_prompt_variant(optimized_prompt, template_id)
            
            logger.info(f"✅ Created optimized variant for prompt {template_id}")
            return optimized_prompt
            
        except Exception as e:
            logger.error(f"Failed to optimize prompt {template_id}: {e}")
            return None
    
    async def auto_evolve_prompts(self) -> List[Dict[str, Any]]:
        """
        Automatically evolve prompts based on performance data.
        
        Returns:
            List of evolution results
        """
        evolution_results = []
        
        try:
            # Get underperforming prompts
            underperforming_prompts = await self._get_underperforming_prompts()
            
            for prompt_data in underperforming_prompts:
                try:
                    # Generate improvement suggestions
                    improvements = await self._generate_improvement_suggestions(prompt_data)
                    
                    # Create improved version
                    improved_template = await self._create_improved_version(prompt_data, improvements)
                    
                    if improved_template:
                        evolution_results.append({
                            'original_id': prompt_data['id'],
                            'improved_id': improved_template.get('id'),
                            'agent_type': prompt_data['agent_type'],
                            'specialty': prompt_data['specialty'],
                            'improvements': improvements
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to evolve prompt {prompt_data.get('id')}: {e}")
            
            logger.info(f"✅ Auto-evolved {len(evolution_results)} prompts")
            return evolution_results
            
        except Exception as e:
            logger.error(f"Failed to auto-evolve prompts: {e}")
            return []
    
    def _create_generation_prompt(self) -> ChatPromptTemplate:
        """Create the prompt for generating agent prompts."""
        template = """You are an expert AI prompt engineer. Create an optimized prompt template for an AI agent.

**Agent Details:**
- Agent Type: {agent_type}
- Specialty: {specialty}  
- Complexity Level: {complexity_level}

**Context:**
{context}

**Successful Patterns:**
{existing_patterns}

**Requirements:**
1. Create a system prompt that guides the agent to excel in their specialty
2. Include tool decision guidance for smart MCP/tool usage
3. Define communication style appropriate for the specialty
4. Specify tool selection criteria

**Output Format (JSON):**
{{
    "system_prompt": "Clear, comprehensive system prompt...",
    "tool_decision_guidance": "Specific guidance for tool selection...",
    "communication_style": "How the agent should communicate...",
    "tool_selection_criteria": "Criteria for choosing tools..."
}}

Focus on making prompts that lead to:
- Better tool selection accuracy
- Higher user satisfaction
- More efficient responses
- Clear, actionable outputs"""

        return ChatPromptTemplate.from_template(template)
    
    async def _gather_generation_context(self, agent_type: str, specialty: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Gather context for prompt generation."""
        generation_context = {
            'agent_type': agent_type,
            'specialty': specialty,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if context:
            generation_context.update(context)
        
        # Add any domain-specific context
        if specialty == 'technical':
            generation_context['focus_areas'] = ['code debugging', 'system optimization', 'technical problem solving']
        elif specialty == 'research':
            generation_context['focus_areas'] = ['information gathering', 'analysis', 'synthesis']
        elif specialty == 'customer_support':
            generation_context['focus_areas'] = ['problem resolution', 'empathy', 'clear communication']
        
        return generation_context
    
    async def _get_existing_prompt(self, agent_type: str, specialty: str, complexity_level: str) -> Optional[Dict]:
        """Get existing prompt if it exists and performs well."""
        try:
            if not self.supabase_logger or not hasattr(self.supabase_logger, 'client'):
                return None
                
            result = self.supabase_logger.client.table("agent_prompt_templates")\
                .select("*")\
                .eq("agent_type", agent_type)\
                .eq("specialty", specialty)\
                .eq("complexity_level", complexity_level)\
                .eq("is_active", True)\
                .order("performance_score", desc=True)\
                .limit(1)\
                .execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
                
        except Exception as e:
            logger.warning(f"Failed to get existing prompt: {e}")
        
        return None
    
    async def _get_successful_patterns(self, specialty: str) -> str:
        """Get successful patterns from high-performing prompts."""
        try:
            if not self.supabase_logger or not hasattr(self.supabase_logger, 'client'):
                return "No patterns available"
                
            result = self.supabase_logger.client.table("agent_prompt_templates")\
                .select("system_prompt, tool_decision_guidance, performance_score")\
                .eq("specialty", specialty)\
                .gte("performance_score", 0.8)\
                .eq("is_active", True)\
                .limit(3)\
                .execute()
            
            if result.data:
                patterns = []
                for prompt in result.data:
                    patterns.append(f"Score: {prompt['performance_score']:.2f} - {prompt['tool_decision_guidance'][:100]}...")
                return "\n".join(patterns)
                
        except Exception as e:
            logger.warning(f"Failed to get successful patterns: {e}")
        
        return "No patterns available"
    
    def _parse_generated_prompt(self, response_content: str) -> Dict[str, str]:
        """Parse the LLM-generated prompt response."""
        try:
            # Clean up the response
            content = response_content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            # Parse JSON
            parsed = json.loads(content.strip())
            
            # Validate required fields
            required_fields = ['system_prompt', 'tool_decision_guidance', 'communication_style', 'tool_selection_criteria']
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = f"Default {field.replace('_', ' ')} for the agent."
            
            return parsed
            
        except Exception as e:
            logger.warning(f"Failed to parse generated prompt: {e}")
            return {
                'system_prompt': 'You are a helpful AI assistant.',
                'tool_decision_guidance': 'Use available tools when appropriate.',
                'communication_style': 'Be clear and helpful.',
                'tool_selection_criteria': 'Choose tools that best solve the user\'s needs.'
            }
    
    async def _save_generated_prompt(self, template: PromptTemplate) -> str:
        """Save generated prompt to database."""
        try:
            if not self.supabase_logger or not hasattr(self.supabase_logger, 'client'):
                logger.warning("No Supabase connection for saving prompt")
                return "fallback_id"
            
            result = self.supabase_logger.client.table("agent_prompt_templates").insert({
                "agent_type": template.agent_type,
                "specialty": template.specialty,
                "complexity_level": template.complexity_level,
                "system_prompt": template.system_prompt,
                "tool_decision_guidance": template.tool_decision_guidance,
                "communication_style": template.communication_style,
                "tool_selection_criteria": template.tool_selection_criteria,
                "performance_score": 0.5,  # Start with neutral score
                "is_active": True,
                "test_group": "auto_generated",
                "created_by": "prompt_auto_generator"
            }).execute()
            
            if result.data and len(result.data) > 0:
                template_id = result.data[0]['id']
                logger.info(f"✅ Saved generated prompt with ID: {template_id}")
                return template_id
            
        except Exception as e:
            logger.error(f"Failed to save generated prompt: {e}")
        
        return "fallback_id"
    
    def _convert_to_template(self, prompt_data: Dict, agent_type: str, specialty: str, complexity_level: str) -> PromptTemplate:
        """Convert database record to PromptTemplate."""
        return PromptTemplate(
            agent_type=agent_type,
            specialty=specialty,
            system_prompt=prompt_data.get('system_prompt', ''),
            tool_decision_guidance=prompt_data.get('tool_decision_guidance', ''),
            communication_style=prompt_data.get('communication_style', ''),
            tool_selection_criteria=prompt_data.get('tool_selection_criteria', ''),
            complexity_level=complexity_level
        )
    
    def _create_fallback_template(self, agent_type: str, specialty: str, complexity_level: str) -> PromptTemplate:
        """Create a fallback template when generation fails."""
        return PromptTemplate(
            agent_type=agent_type,
            specialty=specialty,
            system_prompt=f"You are a helpful {specialty} AI assistant specialized in {agent_type} tasks.",
            tool_decision_guidance="Use available tools when they can help solve the user's request efficiently.",
            communication_style="Be clear, helpful, and professional in your responses.",
            tool_selection_criteria="Choose tools that directly address the user's needs and provide accurate results.",
            complexity_level=complexity_level
        )
    
    # Additional methods for optimization and evolution would go here...
    async def _get_underperforming_prompts(self) -> List[Dict[str, Any]]:
        """Get prompts that are underperforming."""
        # Implementation would query Supabase for prompts with low performance scores
        return []
    
    async def _generate_improvement_suggestions(self, prompt_data: Dict) -> List[str]:
        """Generate suggestions for improving a prompt."""
        # Implementation would analyze performance data and generate suggestions
        return []
    
    async def _create_improved_version(self, prompt_data: Dict, improvements: List[str]) -> Optional[Dict]:
        """Create an improved version of a prompt."""
        # Implementation would apply improvements and create new prompt variant
        return None 