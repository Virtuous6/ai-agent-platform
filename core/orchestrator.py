"""
Filename: orchestrator.py
Purpose: Simplified orchestrator - the heart of the AI agent platform
Dependencies: asyncio, logging, typing, uuid

This is the central orchestrator that manages agent spawning, routing, learning, and PROMPT MANAGEMENT.
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Enhanced orchestrator that manages agent lifecycle, communication, and DYNAMIC PROMPTS.
    
    Features:
    - Dynamic agent spawning with auto-generated prompts
    - Prompt performance tracking and optimization
    - Agent-specific prompt evolution
    - Auto-generation of prompts for new agent types
    """
    
    def __init__(self, storage=None, event_bus=None):
        """Initialize the orchestrator with essential dependencies."""
        self.agents = {}  # Active agents (max 50)
        self.agent_configs = {}  # Agent configurations
        self.agent_prompts = {}  # Cache for agent-specific prompts
        self.storage = storage
        self.event_bus = event_bus
        self.max_active_agents = 50
        
        # Initialize core components
        self.tracker = None  # Will be set by dependency injection
        self.learner = None  # Will be set by dependency injection
        
        # Prompt management
        self.prompt_manager = None  # Will be initialized later
        self.prompt_auto_generator = None  # For creating new prompts
        
        # Agent cleanup tracking
        self.agent_last_used = {}
        self.cleanup_task = None
        
        logger.info("🎯 Enhanced Orchestrator initialized with dynamic prompt management")
    
    async def initialize_prompt_system(self):
        """Initialize the dynamic prompt management system."""
        try:
            from core.dynamic_prompt_manager import get_prompt_manager
            from core.prompt_auto_generator import PromptAutoGenerator
            
            self.prompt_manager = get_prompt_manager()
            self.prompt_auto_generator = PromptAutoGenerator(
                supabase_logger=self.storage,
                prompt_manager=self.prompt_manager
            )
            
            logger.info("✅ Dynamic prompt system initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize prompt system: {e}")
    
    def set_components(self, tracker, learner):
        """Set tracker and learner components (dependency injection)."""
        self.tracker = tracker
        self.learner = learner
    
    async def process(self, message: str, context: Dict[str, Any]) -> str:
        """
        Main entry point for processing messages.
        Tracks everything, routes to agents, and learns from results.
        """
        # 1. Start workflow tracking
        run_id = str(uuid.uuid4())
        user_id = context.get("user_id", "anonymous")
        
        if self.tracker:
            await self.tracker.start_workflow(run_id, message, context)
        
        # 2. Check for special commands
        if message.startswith("/"):
            return await self._handle_command(message, user_id, context, run_id)
        
        try:
            # 3. Determine intent (simple classification)
            intent = await self._classify_intent(message)
            
            # 4. Get or spawn appropriate agent
            agent = await self._get_or_spawn_agent(intent, context)
            
            # 5. Process with agent
            result = await agent.process(message, context)
            
            # 6. Track completion
            if self.tracker:
                await self.tracker.complete_workflow(run_id, {
                    "response": result,
                    "agent_type": intent,
                    "duration": (datetime.utcnow() - datetime.utcnow()).total_seconds()
                })
            
            # 7. Async learning (non-blocking)
            if self.learner:
                asyncio.create_task(self.learner.analyze_interaction(run_id))
            
            return result.get("response", result) if isinstance(result, dict) else result
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            
            # Track error
            if self.tracker:
                await self.tracker.track_error(run_id, str(e))
            
            return f"I encountered an error: {str(e)}. Let me try a different approach."
    
    async def _handle_command(self, message: str, user_id: str, 
                             context: Dict[str, Any], run_id: str) -> str:
        """Handle special commands like /improve, /save-workflow, etc."""
        parts = message.split(maxsplit=1)
        command = parts[0][1:]  # Remove the /
        args = parts[1] if len(parts) > 1 else ""
        
        # Handle learning/feedback commands
        if command in ["improve", "save-workflow", "list-workflows"]:
            if self.learner:
                result = await self.learner.process_user_feedback(command, user_id, args, context)
                return result.get("message", "Command processed")
        
        # Handle system commands
        elif command == "status":
            return self._get_system_status()
        elif command == "agents":
            return self._list_active_agents()
        else:
            return f"Unknown command: /{command}"
    
    async def _classify_intent(self, message: str) -> str:
        """
        Simple intent classification.
        In production, this could use embeddings or a classifier.
        """
        message_lower = message.lower()
        
        # Simple keyword-based classification
        if any(word in message_lower for word in ["code", "debug", "error", "fix", "implement"]):
            return "technical"
        elif any(word in message_lower for word in ["research", "find", "search", "information"]):
            return "research"
        else:
            return "general"
    
    async def _get_or_spawn_agent(self, intent: str, context: Dict[str, Any]) -> Any:
        """Get existing agent or spawn new one based on intent with AUTO-GENERATED PROMPTS."""
        user_id = context.get("user_id", "anonymous")
        agent_key = f"{user_id}:{intent}"
        
        # Check if agent exists and is active
        if agent_key in self.agents:
            self.agent_last_used[agent_key] = datetime.utcnow()
            logger.info(f"🔄 Reusing existing agent: {agent_key}")
            return self.agents[agent_key]
        
        # Check agent limit
        if len(self.agents) >= self.max_active_agents:
            await self._cleanup_inactive_agents()
        
        logger.info(f"🚀 Spawning new agent for {intent}...")
        
        # 🎯 STEP 1: Auto-generate custom prompt for this agent
        custom_prompt = await self._generate_agent_prompt(intent, context)
        
        # 🏗️ STEP 2: Load base agent configuration
        config = await self._load_agent_config(intent)
        
        # Import UniversalAgent dynamically to avoid circular imports
        from core.agent import UniversalAgent
        
        # 🚀 STEP 3: Spawn agent with generated prompt
        agent = UniversalAgent(
            specialty=intent,
            system_prompt=custom_prompt.get('system_prompt', config.get("system_prompt", "")),
            temperature=config.get("temperature", 0.7),
            model_name=config.get("model", "gpt-3.5-turbo"),
            max_tokens=config.get("max_tokens", 500),
            agent_id=agent_key,
            supabase_logger=self.storage,
            event_bus=self.event_bus
        )
        
        # Store the custom prompt data for this agent
        self.agent_prompts[agent_key] = custom_prompt
        
        self.agents[agent_key] = agent
        self.agent_last_used[agent_key] = datetime.utcnow()
        
        # Emit spawn event with prompt info
        if self.event_bus:
            await self.event_bus.publish(
                "agent_spawned",
                {
                    "agent_id": agent_key, 
                    "intent": intent,
                    "custom_prompt_generated": True,
                    "prompt_id": custom_prompt.get('template_id')
                },
                source="orchestrator"
            )
        
        logger.info(f"✅ Spawned new agent: {agent_key} with custom AI-generated prompt")
        return agent
    
    async def _generate_agent_prompt(self, intent: str, context: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a custom prompt for a new agent using LLM.
        
        Args:
            intent: Agent specialty/intent (e.g., 'technical', 'research')
            context: User context for customization
            
        Returns:
            Generated prompt components
        """
        try:
            # Initialize prompt auto-generator if not done yet
            if not self.prompt_auto_generator:
                await self.initialize_prompt_system()
            
            # If still no generator, use fallback
            if not self.prompt_auto_generator:
                logger.warning("No prompt auto-generator available, using fallback")
                return self._create_fallback_prompt(intent, context)
            
            # Assess complexity based on context
            complexity_level = self._assess_spawn_complexity(intent, context)
            
            # Generate custom prompt
            logger.info(f"🎯 Generating custom prompt for {intent} agent (complexity: {complexity_level})...")
            
            prompt_template = await self.prompt_auto_generator.generate_prompt_for_agent(
                agent_type="universal",
                specialty=intent,
                complexity_level=complexity_level,
                context={
                    "user_id": context.get("user_id"),
                    "user_context": context.get("user_context", ""),
                    "spawn_reason": f"User needs {intent} assistance",
                    "platform": "ai-agent-platform"
                }
            )
            
            # Convert template to dict for easy access
            custom_prompt = {
                'system_prompt': prompt_template.system_prompt,
                'tool_decision_guidance': prompt_template.tool_decision_guidance,
                'communication_style': prompt_template.communication_style,
                'tool_selection_criteria': prompt_template.tool_selection_criteria,
                'complexity_level': complexity_level,
                'generated_at': datetime.utcnow().isoformat(),
                'template_id': getattr(prompt_template, 'template_id', None)
            }
            
            logger.info(f"✅ Generated custom prompt for {intent} agent")
            return custom_prompt
            
        except Exception as e:
            logger.error(f"Failed to generate custom prompt for {intent}: {e}")
            return self._create_fallback_prompt(intent, context)
    
    def _assess_spawn_complexity(self, intent: str, context: Dict[str, Any]) -> str:
        """Assess complexity level for agent spawning."""
        # Simple heuristics for spawn complexity
        if intent in ['technical', 'optimization', 'analysis']:
            return 'complex'
        elif intent in ['general', 'help']:
            return 'simple'
        else:
            return 'medium'
    
    def _create_fallback_prompt(self, intent: str, context: Dict[str, Any]) -> Dict[str, str]:
        """Create fallback prompt when LLM generation fails."""
        return {
            'system_prompt': f"""You are a specialized {intent} AI assistant powered by the AI Agent Platform.

**Your Specialty:** {intent.title()} assistance and problem-solving

**Core Capabilities:**
- Expert knowledge in {intent} domain
- Tool-aware decision making
- Clear, actionable responses
- User-focused solutions

**Tool Decision Framework:**
1. **Assess Need**: Does this require real-time data or external resources?
2. **MCP-First**: Check for existing MCP tools that can help
3. **Gap Detection**: If no suitable tools exist, suggest MCP setup
4. **User Guidance**: Be transparent about tool usage and limitations

**Communication Style:**
- Be professional yet approachable
- Provide clear, step-by-step guidance
- Explain your reasoning
- Ask clarifying questions when needed

Current context: {context.get('user_context', 'General assistance')}""",
            'tool_decision_guidance': f"""For {intent} tasks:
1. Always check if existing tools can solve the problem
2. Prioritize MCP solutions for external data needs
3. If tools are missing, guide users on setting them up
4. Be transparent about tool limitations""",
            'communication_style': f"Professional {intent} expert who explains concepts clearly and provides actionable solutions.",
            'tool_selection_criteria': f"Choose tools that best serve {intent} objectives and provide accurate, relevant results.",
            'complexity_level': 'medium',
            'generated_at': datetime.utcnow().isoformat(),
            'template_id': 'fallback'
        }
    
    async def _load_agent_config(self, intent: str) -> Dict[str, Any]:
        """Load agent configuration from Supabase storage."""
        # Check cache first
        if intent in self.agent_configs:
            return self.agent_configs[intent]
        
        # Load from Supabase if storage is available
        if self.storage and hasattr(self.storage, 'client'):
            try:
                result = self.storage.client.table("agents")\
                    .select("*")\
                    .eq("name", intent)\
                    .eq("is_active", True)\
                    .execute()
                
                if result.data and len(result.data) > 0:
                    agent_data = result.data[0]
                    config = {
                        "model": agent_data.get("model", "gpt-3.5-turbo"),
                        "temperature": float(agent_data.get("temperature", 0.7)),
                        "max_tokens": int(agent_data.get("max_tokens", 500)),
                        "system_prompt": agent_data.get("system_prompt", "You are a helpful AI assistant.")
                    }
                    
                    # Cache the configuration
                    self.agent_configs[intent] = config
                    logger.info(f"✅ Loaded agent config for '{intent}' from Supabase")
                    return config
                else:
                    logger.warning(f"⚠️ No configuration found in Supabase for agent '{intent}'")
            
            except Exception as e:
                logger.error(f"❌ Error loading agent config from Supabase: {e}")
        
        # Fallback to minimal defaults only if Supabase fails
        logger.info(f"📋 Using fallback configuration for '{intent}'")
        fallback_config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 500,
            "system_prompt": f"You are a helpful {intent} AI assistant."
        }
        
        self.agent_configs[intent] = fallback_config
        return fallback_config
    
    async def _cleanup_inactive_agents(self):
        """Remove agents that haven't been used recently."""
        cutoff_time = datetime.utcnow()
        inactive_threshold = 3600  # 1 hour
        
        agents_to_remove = []
        for agent_key, last_used in self.agent_last_used.items():
            if (cutoff_time - last_used).total_seconds() > inactive_threshold:
                agents_to_remove.append(agent_key)
        
        for agent_key in agents_to_remove:
            del self.agents[agent_key]
            del self.agent_last_used[agent_key]
            logger.info(f"Cleaned up inactive agent: {agent_key}")
    
    def _get_system_status(self) -> str:
        """Get current system status."""
        status = f"""
**System Status**
• Active Agents: {len(self.agents)}/{self.max_active_agents}
• Storage: {'Connected' if self.storage else 'Disconnected'}
• Event Bus: {'Active' if self.event_bus else 'Inactive'}
• Tracker: {'Active' if self.tracker else 'Inactive'}
• Learner: {'Active' if self.learner else 'Inactive'}
        """
        return status.strip()
    
    def _list_active_agents(self) -> str:
        """List currently active agents."""
        if not self.agents:
            return "No active agents"
        
        agent_list = "**Active Agents:**\n"
        for agent_key in self.agents:
            user_id, intent = agent_key.split(":", 1)
            agent_list += f"• {intent} agent for user {user_id[:8]}...\n"
        
        return agent_list
    
    async def start_periodic_cleanup(self):
        """Start periodic cleanup of inactive agents."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_inactive_agents()
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Orchestrator shutting down...")
        
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Close all agents
        for agent in self.agents.values():
            if hasattr(agent, 'close'):
                await agent.close()
        
        self.agents.clear()
        logger.info("Orchestrator shutdown complete")
    
    async def update_agent_prompt(self, agent_key: str, prompt_updates: Dict[str, str]) -> bool:
        """
        Update an active agent's prompt.
        
        Args:
            agent_key: Key of the agent to update
            prompt_updates: Dictionary of prompt components to update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if agent_key not in self.agents:
                logger.warning(f"Agent {agent_key} not found for prompt update")
                return False
            
            agent = self.agents[agent_key]
            current_prompt = self.agent_prompts.get(agent_key, {})
            
            # Update stored prompt data
            current_prompt.update(prompt_updates)
            current_prompt['updated_at'] = datetime.utcnow().isoformat()
            self.agent_prompts[agent_key] = current_prompt
            
            # Update the agent's system prompt
            if 'system_prompt' in prompt_updates:
                agent.system_prompt = prompt_updates['system_prompt']
                # Regenerate the main prompt template
                agent.main_prompt = agent._create_main_prompt()
            
            # Save updated prompt to database
            if self.prompt_auto_generator:
                await self._save_prompt_update(agent_key, current_prompt)
            
            logger.info(f"✅ Updated prompt for agent {agent_key}")
            
            # Emit update event
            if self.event_bus:
                await self.event_bus.publish(
                    "agent_prompt_updated",
                    {
                        "agent_id": agent_key,
                        "updates": list(prompt_updates.keys()),
                        "updated_at": current_prompt['updated_at']
                    },
                    source="orchestrator"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update prompt for agent {agent_key}: {e}")
            return False
    
    async def evolve_agent_prompts(self) -> Dict[str, Any]:
        """
        Evolve prompts for all active agents based on performance data.
        
        Returns:
            Summary of evolution results
        """
        evolution_summary = {
            'agents_processed': 0,
            'prompts_evolved': 0,
            'improvements_found': 0,
            'errors': []
        }
        
        try:
            if not self.prompt_auto_generator:
                await self.initialize_prompt_system()
            
            if not self.prompt_auto_generator:
                evolution_summary['errors'].append("No prompt auto-generator available")
                return evolution_summary
            
            # Evolve prompts for all active agents
            for agent_key, agent in self.agents.items():
                try:
                    evolution_summary['agents_processed'] += 1
                    
                    # Get performance data for this agent
                    performance_data = await self._get_agent_performance_data(agent_key)
                    
                    # Check if prompt needs evolution
                    if self._should_evolve_prompt(performance_data):
                        # Generate evolved prompt
                        current_prompt = self.agent_prompts.get(agent_key, {})
                        evolved_prompt = await self.prompt_auto_generator.optimize_existing_prompt(
                            current_prompt.get('template_id', 'fallback'),
                            performance_data
                        )
                        
                        if evolved_prompt:
                            # Apply evolved prompt
                            prompt_updates = {
                                'system_prompt': evolved_prompt.system_prompt,
                                'tool_decision_guidance': evolved_prompt.tool_decision_guidance,
                                'communication_style': evolved_prompt.communication_style,
                                'tool_selection_criteria': evolved_prompt.tool_selection_criteria
                            }
                            
                            success = await self.update_agent_prompt(agent_key, prompt_updates)
                            if success:
                                evolution_summary['prompts_evolved'] += 1
                                evolution_summary['improvements_found'] += 1
                        
                except Exception as e:
                    error_msg = f"Failed to evolve prompt for {agent_key}: {e}"
                    evolution_summary['errors'].append(error_msg)
                    logger.warning(error_msg)
            
            logger.info(f"🧬 Prompt evolution complete: {evolution_summary}")
            return evolution_summary
            
        except Exception as e:
            logger.error(f"Failed to evolve agent prompts: {e}")
            evolution_summary['errors'].append(str(e))
            return evolution_summary
    
    async def get_agent_prompt_status(self, agent_key: str) -> Optional[Dict[str, Any]]:
        """
        Get the current prompt status for an agent.
        
        Args:
            agent_key: Key of the agent
            
        Returns:
            Prompt status information or None if not found
        """
        if agent_key not in self.agents:
            return None
        
        current_prompt = self.agent_prompts.get(agent_key, {})
        
        return {
            'agent_key': agent_key,
            'specialty': agent_key.split(':')[-1],
            'prompt_generated_at': current_prompt.get('generated_at'),
            'prompt_updated_at': current_prompt.get('updated_at'),
            'template_id': current_prompt.get('template_id'),
            'complexity_level': current_prompt.get('complexity_level'),
            'has_custom_prompt': bool(current_prompt),
            'prompt_length': len(current_prompt.get('system_prompt', '')),
            'last_used': self.agent_last_used.get(agent_key)
        }
    
    async def _save_prompt_update(self, agent_key: str, prompt_data: Dict[str, str]):
        """Save prompt update to database."""
        try:
            if self.storage and hasattr(self.storage, 'client'):
                # Log the prompt update
                self.storage.client.table("agent_prompt_updates").insert({
                    "agent_key": agent_key,
                    "prompt_data": json.dumps(prompt_data),
                    "updated_at": datetime.utcnow().isoformat(),
                    "source": "orchestrator_update"
                }).execute()
                
        except Exception as e:
            logger.warning(f"Failed to save prompt update to database: {e}")
    
    async def _get_agent_performance_data(self, agent_key: str) -> Dict[str, Any]:
        """Get performance data for an agent."""
        # Placeholder - would get actual performance metrics
        return {
            'response_quality': 0.8,
            'user_satisfaction': 0.75,
            'tool_accuracy': 0.9,
            'recent_interactions': 10
        }
    
    def _should_evolve_prompt(self, performance_data: Dict[str, Any]) -> bool:
        """Determine if a prompt should be evolved based on performance."""
        # Simple criteria - evolve if performance is below threshold
        avg_performance = (
            performance_data.get('response_quality', 0.8) +
            performance_data.get('user_satisfaction', 0.8) +
            performance_data.get('tool_accuracy', 0.8)
        ) / 3
        
        return avg_performance < 0.8 and performance_data.get('recent_interactions', 0) > 5 