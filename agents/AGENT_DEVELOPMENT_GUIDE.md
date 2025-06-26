# Agent Development Guide

> **Complete guide for building LLM-powered agents with proper lifecycle management**

## 🚀 Quick Start Checklist

### Before You Begin
- [ ] Read `agents/README.llm.md` for architecture overview
- [ ] Understand LLM integration patterns and cost implications
- [ ] Have OpenAI API key configured in environment
- [ ] Review existing agents for patterns and examples

### Agent Creation Steps
1. **[Create Directory Structure](#1-directory-structure)**
2. **[Implement Agent Class](#2-agent-implementation)**  
3. **[Add Lifecycle Management](#3-lifecycle-management)**
4. **[Create Documentation](#4-documentation)**
5. **[Integration & Testing](#5-integration--testing)**

---

## 1. Directory Structure

```bash
# Create new agent directory
mkdir agents/{agent_name}
cd agents/{agent_name}

# Create required files
touch __init__.py
touch {agent_name}_agent.py
touch README.llm.md

# Example: Creating a "translation" agent
mkdir agents/translation
touch agents/translation/__init__.py
touch agents/translation/translation_agent.py
touch agents/translation/README.llm.md
```

## 2. Agent Implementation

### 2.1 Basic Agent Template

```python
"""
Filename: {agent_name}_agent.py
Purpose: LLM-powered {agent_name} agent for {purpose}
Dependencies: langchain, openai, asyncio, logging, typing

This module is part of the AI Agent Platform.
See README.llm.md in this directory for context.
"""

import asyncio
import logging
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class {AgentName}Domain(Enum):
    """Domains the {agent_name} agent handles."""
    DOMAIN_1 = "domain_1"
    DOMAIN_2 = "domain_2"
    # Add specific domains for your agent

class {AgentName}Agent:
    """
    LLM-powered {agent_name} agent specialized for {purpose}.
    
    Uses ChatOpenAI with {domain}-specific prompts to provide {capabilities}.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125", temperature: float = 0.4):
        """Initialize the LLM-powered {AgentName} Agent."""
        
        # Initialize the LLM with appropriate temperature for your use case
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,  # Adjust based on creativity needs
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=800,  # Adjust based on response length needs
        )
        
        # Initialize secondary LLM if needed (e.g., for classification)
        self.classification_llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,  # Lower temperature for classification tasks
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=200,
        )
        
        # Initialize prompt templates
        self.main_prompt = self._create_main_prompt()
        self.classification_prompt = self._create_classification_prompt()
        
        # Initialize state tracking
        self.interaction_history = []
        
        logger.info(f"{AgentName} Agent initialized with model: {model_name}")
    
    def _create_main_prompt(self) -> ChatPromptTemplate:
        """Create the main conversation prompt template."""
        
        system_template = """You are the {AgentName} Agent for an AI Agent Platform - a specialized expert in {domain}. Your expertise and approach:

**Your Role:**
- {Role description}
- {Key responsibilities}
- {Primary capabilities}

**Your Expertise Areas:**
- **{Area 1}**: {Description}
- **{Area 2}**: {Description}
- **{Area 3}**: {Description}

**Your Personality:**
- {Personality trait 1}
- {Personality trait 2}
- {Communication style}
- Uses relevant emojis for clarity

**Your Approach:**
1. **{Step 1}**: {Description}
2. **{Step 2}**: {Description}
3. **{Step 3}**: {Description}
4. **{Step 4}**: {Description}

Current conversation context: {context}
Recent conversation history: {history}"""

        human_template = """{Agent} request: {message}

Please provide expert {domain} assistance. {Specific instructions for response format}"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_classification_prompt(self) -> ChatPromptTemplate:
        """Create domain classification prompt template (optional)."""
        
        system_template = """You are a {domain} classifier. Analyze requests to categorize them into specific {domain} domains.

**Available Domains:**
1. **{Domain 1}** - {Description}
2. **{Domain 2}** - {Description}

**Your Task:**
Analyze the request and determine the most appropriate domain.

Return your analysis in this exact JSON format:
{{
    "domain": "{domain_1|domain_2}",
    "confidence": float,
    "reasoning": "brief explanation"
}}"""

        human_template = """{Agent} request: "{message}"

Classify this request:"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a {domain} request using specialized LLM intelligence.
        
        Args:
            message: {Agent} request content
            context: Conversation and {domain} context
            
        Returns:
            Dictionary containing {domain} response and metadata
        """
        try:
            logger.info(f"{AgentName} Agent processing: '{message[:50]}...'")
            
            # Classify domain (if applicable)
            domain = self._classify_domain(message)
            
            # Assess complexity or other attributes
            complexity = self._assess_complexity(message, context)
            
            # Prepare conversation history context
            history_context = self._format_conversation_history(context.get("conversation_history", []))
            
            # Generate response with cost tracking
            try:
                with get_openai_callback() as cb:
                    response = await self._generate_response(
                        message, context, history_context, domain, complexity
                    )
                tokens_used = cb.total_tokens
                cost = cb.total_cost
            except Exception as e:
                logger.warning(f"OpenAI callback failed, proceeding without tracking: {e}")
                response = await self._generate_response(
                    message, context, history_context, domain, complexity
                )
                tokens_used = 0
                cost = 0.0
            
            # Log the interaction
            interaction_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "message_preview": message[:100],
                "domain": domain.value if domain else "unknown",
                "complexity": complexity,
                "user_id": context.get("user_id"),
                "channel_id": context.get("channel_id"),
                "tokens_used": tokens_used,
                "cost": cost
            }
            self.interaction_history.append(interaction_log)
            
            return {
                "response": response,
                "domain": domain.value if domain else "unknown",
                "complexity": complexity,
                "confidence": 0.85,  # Adjust based on your agent's confidence
                "tokens_used": tokens_used,
                "processing_cost": cost,
                "metadata": {
                    "agent_type": "{agent_name}",
                    "model_used": self.llm.model_name,
                    "temperature": self.llm.temperature,
                    "specialization": "{domain}_specialist"
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing {domain} request: {str(e)}")
            return {
                "response": f"I apologize, but I'm experiencing difficulties processing your {domain} request. Please try rephrasing your question and I'll provide expert {domain} assistance.",
                "domain": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _classify_domain(self, message: str) -> Optional[{AgentName}Domain]:
        """Classify the domain of the request (implement based on your needs)."""
        # Implement domain classification logic
        # This could use keywords, patterns, or even LLM classification
        return {AgentName}Domain.DOMAIN_1  # Default return
    
    def _assess_complexity(self, message: str, context: Dict[str, Any]) -> str:
        """Assess complexity level of the request."""
        # Implement complexity assessment logic
        return "medium"  # Default return
    
    async def _generate_response(self, message: str, context: Dict[str, Any], 
                               history_context: str, domain: Optional[{AgentName}Domain],
                               complexity: str) -> str:
        """Generate the main response using the LLM."""
        
        main_chain = self.main_prompt | self.llm
        
        response = await main_chain.ainvoke({
            "message": message,
            "context": self._format_context(context),
            "history": history_context
        })
        
        response_text = response.content
        
        # Add agent-specific footer
        emoji = "🔧"  # Choose appropriate emoji for your agent
        response_text += f"\n\n{emoji} *{AgentName} Agent - {domain.value.title() if domain else 'General'} Specialist*"
        
        return response_text
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for context."""
        if not history:
            return f"No previous {domain} conversation history."
        
        formatted_history = []
        for item in history[-3:]:  # Last 3 messages for context
            role = "User" if item.get("message_type") == "user_message" else f"{AgentName} Agent"
            content = item.get("content", "")[:150]
            formatted_history.append(f"{role}: {content}")
        
        return "\n".join(formatted_history)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information for the prompt."""
        context_parts = []
        
        if context.get("channel_type"):
            context_parts.append(f"Channel type: {context['channel_type']}")
        
        if context.get("user_id"):
            context_parts.append(f"User: {context['user_id']}")
        
        if context.get("is_thread"):
            context_parts.append(f"{AgentName} discussion thread")
        
        return ", ".join(context_parts) if context_parts else f"{AgentName} consultation session"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about interactions handled."""
        if not self.interaction_history:
            return {"total_interactions": 0}
        
        total_tokens = sum(log.get("tokens_used", 0) for log in self.interaction_history)
        total_cost = sum(log.get("cost", 0) for log in self.interaction_history)
        
        # Calculate domain distribution
        domain_counts = {}
        complexity_counts = {}
        
        for log in self.interaction_history:
            domain = log.get("domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            complexity = log.get("complexity", "unknown")
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        return {
            "total_interactions": len(self.interaction_history),
            "total_tokens_used": total_tokens,
            "total_cost": total_cost,
            "domain_distribution": domain_counts,
            "complexity_distribution": complexity_counts
        }
    
    async def close(self):
        """
        ⚠️ MANDATORY: Close the agent and cleanup resources.
        
        This method ensures proper cleanup of HTTP connections
        used by the OpenAI clients when shutting down.
        """
        try:
            logger.info(f"Closing {AgentName} Agent connections...")
            
            # Close ALL LLM clients
            llm_clients = [
                ("Main LLM", self.llm),
                ("Classification LLM", self.classification_llm),
                # Add any additional LLM clients here
            ]
            
            for client_name, client in llm_clients:
                if client and hasattr(client, 'client') and hasattr(client.client, 'close'):
                    await client.client.close()
                    logger.debug(f"Closed {client_name}")
            
            # Clear state to help garbage collection
            self.interaction_history.clear()
            
            logger.info(f"{AgentName} Agent connections closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing {AgentName} Agent: {e}")
```

### 2.2 Package Init File

```python
# agents/{agent_name}/__init__.py
"""
{AgentName} Agent Package

LLM-powered {domain} agent for {capabilities}.
Uses {temperature} temperature for {reasoning}.
"""

from .{agent_name}_agent import {AgentName}Agent

__all__ = ['{AgentName}Agent']
```

## 3. Lifecycle Management

### 3.1 ✅ Lifecycle Verification Checklist

- [ ] **`async def close()` method implemented**
- [ ] **All LLM clients closed in `close()` method**
- [ ] **Exception handling in cleanup**
- [ ] **State clearing for garbage collection**
- [ ] **Proper logging during cleanup**

### 3.2 Testing Lifecycle Management

```python
# Test lifecycle management (add to your test suite)
import pytest
import asyncio

@pytest.mark.asyncio
async def test_agent_lifecycle():
    """Test proper agent lifecycle management."""
    
    # Initialize agent
    agent = {AgentName}Agent()
    
    # Verify initialization
    assert agent.llm is not None
    assert agent.classification_llm is not None
    assert len(agent.interaction_history) == 0
    
    # Process a test message
    response = await agent.process_message("test message", {})
    assert "response" in response
    
    # Verify cleanup
    await agent.close()
    
    # Verify state is cleared
    assert len(agent.interaction_history) == 0
```

## 4. Documentation

### 4.1 README.llm.md Template

```markdown
# {AgentName} Agent - LLM-Powered {Domain} Specialist

## Purpose
LLM-powered agent specialized for {domain} tasks using ChatGPT with {domain}-specific prompts and expertise.

## Agent Configuration

### LLM Settings
- **Model**: gpt-3.5-turbo-0125 (default)
- **Temperature**: {temperature} ({reasoning for temperature})
- **Max Tokens**: {max_tokens} ({reasoning for token limit})
- **Specialization**: {domain} expertise with {specific capabilities}

### Domain Classification
The agent handles {number} main domains:
1. **{Domain 1}**: {Description and capabilities}
2. **{Domain 2}**: {Description and capabilities}

## Agent Capabilities

### Core Features
- {Feature 1 with description}
- {Feature 2 with description}
- {Feature 3 with description}

### Response Patterns
- **{Pattern 1}**: {When and how used}
- **{Pattern 2}**: {When and how used}

### Integration Points
- **Tools**: {Any external tools or APIs}
- **Escalation**: {When to escalate to other agents}
- **Context**: {How context is used}

## Usage Examples

### Basic Usage
```python
from agents.{agent_name} import {AgentName}Agent

agent = {AgentName}Agent()
response = await agent.process_message("example request", context)
await agent.close()  # Always close when done
```

### Context Integration
```python
context = {
    "user_id": "user123",
    "channel_id": "channel456", 
    "conversation_history": [...],
    "domain_specific_data": {...}
}

response = await agent.process_message("request", context)
```

## Performance Metrics

### Token Usage
- Average tokens per response: ~{estimate}
- Cost per interaction: ~${estimate}
- Response time: ~{estimate}ms

### Quality Metrics
- Domain classification accuracy: {percentage}%
- User satisfaction (estimated): {percentage}%
- Escalation rate: {percentage}%

## Development Notes

### Prompt Engineering
- {Note about system prompts}
- {Note about response formatting}
- {Note about domain-specific instructions}

### Error Handling
- {How errors are handled}
- {Fallback mechanisms}
- {Logging and debugging}

### Future Enhancements
- {Planned feature 1}
- {Planned feature 2}
- {Integration opportunities}
```

### 4.2 Update Parent __init__.py

```python
# agents/__init__.py - Add your new agent
"""
AI Agents Package

This package contains specialized LLM-powered agents for different domains:
- General Agent: Conversational interactions with higher temperature
- Technical Agent: Precise technical responses with lower temperature  
- Research Agent: Analytical research responses with balanced temperature
- {AgentName} Agent: {Description}
"""

from .general.general_agent import GeneralAgent
from .research.research_agent import ResearchAgent
from .technical.technical_agent import TechnicalAgent
from .{agent_name}.{agent_name}_agent import {AgentName}Agent

__all__ = ['GeneralAgent', 'ResearchAgent', 'TechnicalAgent', '{AgentName}Agent']
```

## 5. Integration & Testing

### 5.1 Orchestrator Integration

```python
# In orchestrator/agent_orchestrator.py
class AgentOrchestrator:
    def __init__(self, ..., {agent_name}_agent: Optional[Any] = None):
        # Add to initialization
        self.{agent_name}_agent = {agent_name}_agent
        
    def _initialize_routing_rules(self):
        # Add routing rules for your agent
        rules = {...}
        rules[AgentType.{AGENT_NAME}] = {
            "keywords": ["keyword1", "keyword2", ...],
            "patterns": [r"pattern1", r"pattern2", ...],
            "priority": {priority_number}
        }
        return rules
    
    def _initialize_agent_capabilities(self):
        # Add capabilities description
        capabilities = {...}
        capabilities[AgentType.{AGENT_NAME}] = {
            "name": "{AgentName} Agent",
            "description": "{Description}",
            "specialties": ["{specialty1}", "{specialty2}"],
            "confidence_threshold": {threshold},
            "max_concurrent": {max_concurrent}
        }
        return capabilities
    
    async def close(self):
        # Add to cleanup
        agents_to_close = [
            # ... existing agents ...
            ("{AgentName} Agent", self.{agent_name}_agent)
        ]
        # ... rest of cleanup logic
```

### 5.2 Add to AgentType Enum

```python
# In orchestrator/agent_orchestrator.py
class AgentType(Enum):
    """Available agent types in the system."""
    GENERAL = "general"
    TECHNICAL = "technical" 
    RESEARCH = "research"
    {AGENT_NAME} = "{agent_name}"  # Add your agent
```

### 5.3 Testing Checklist

- [ ] **Unit Tests**: Test message processing and response generation
- [ ] **Lifecycle Tests**: Test initialization and cleanup
- [ ] **Integration Tests**: Test with orchestrator routing
- [ ] **Performance Tests**: Verify token usage and costs
- [ ] **Error Handling Tests**: Test fallback mechanisms

### 5.4 Manual Testing

```bash
# Test the agent directly
cd ai-agent-platform
source venv/bin/activate
python

# In Python REPL
from agents.{agent_name} import {AgentName}Agent
import asyncio

async def test_agent():
    agent = {AgentName}Agent()
    
    response = await agent.process_message("test request", {
        "user_id": "test_user",
        "channel_id": "test_channel"
    })
    
    print(f"Response: {response['response']}")
    print(f"Tokens used: {response['tokens_used']}")
    print(f"Cost: ${response['processing_cost']:.4f}")
    
    # Always test cleanup
    await agent.close()
    print("Agent closed successfully")

# Run test
asyncio.run(test_agent())
```

## 6. Production Deployment

### 6.1 Environment Configuration

```bash
# Add any agent-specific environment variables
{AGENT_NAME}_AGENT_TEMPERATURE=0.4
{AGENT_NAME}_AGENT_MAX_TOKENS=800
{AGENT_NAME}_AGENT_ENABLED=true
```

### 6.2 Monitoring Setup

- [ ] Add agent to performance monitoring
- [ ] Set up cost tracking for the new agent
- [ ] Configure alerting for errors or high costs
- [ ] Add to health check endpoints

### 6.3 Documentation Updates

- [ ] Update main README.md with new agent
- [ ] Update DEVELOPMENT.md with agent-specific notes
- [ ] Add to .cursor/.cursorrules if needed
- [ ] Update any API documentation

## 🚨 Common Pitfalls

### ❌ **Lifecycle Management Failures**
- **Forgetting `async def close()`** → Resource leaks
- **Not closing all LLM clients** → HTTP connection exhaustion
- **Missing error handling in cleanup** → Cleanup failures

### ❌ **Integration Issues**
- **Not updating AgentType enum** → Routing failures
- **Missing __init__.py exports** → Import errors
- **Incorrect orchestrator integration** → Agent not accessible

### ❌ **Performance Problems**
- **Wrong temperature setting** → Poor response quality
- **No token tracking** → Unexpected costs
- **Missing conversation history** → Poor context continuity

### ❌ **Testing Oversights**
- **Not testing lifecycle management** → Runtime failures
- **Missing error condition tests** → Poor error handling
- **No performance validation** → Cost surprises

## ✅ Success Criteria

Your agent is ready for production when:

- [ ] **All lifecycle management implemented and tested**
- [ ] **Proper integration with orchestrator and routing**
- [ ] **Documentation complete and accurate**
- [ ] **Performance meets expectations**
- [ ] **Error handling comprehensive**
- [ ] **Cost tracking functional**
- [ ] **Manual testing successful**
- [ ] **Unit tests passing**

---

**For complete implementation details, see the full template in agents/README.llm.md** 