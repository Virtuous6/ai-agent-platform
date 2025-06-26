# Agents Directory - LLM-Powered Architecture

## Purpose
Contains specialized AI agents that handle different types of user requests. Each agent is **LLM-powered** using OpenAI's ChatGPT with domain-specific prompts and expertise.

## Agent Architecture Philosophy

### From Rules to Intelligence
All agents have been transformed from rule-based pattern matching to **intelligent LLM-powered systems**:

- **Before**: Regex patterns, static templates, simple keyword routing
- **After**: ChatOpenAI integration, dynamic responses, intelligent escalation assessment
- **Benefit**: Natural conversation, contextual understanding, adaptive expertise

### Unified LLM Framework
Each agent follows the same LLM-powered pattern:
1. **Specialized Prompts**: Domain-specific system prompts and expertise
2. **Intelligent Classification**: Automatic domain/type classification
3. **Context Awareness**: Conversation history and user context integration
4. **Structured Responses**: Consistent metadata and performance tracking
5. **Graceful Fallback**: Robust error handling with keyword-based backup

## Agent Types

### General Agent (`general/`) 🤖
**LLM-Powered Conversational Specialist**
- **Role**: Primary interface for everyday conversations and general assistance
- **Expertise**: Natural conversation, general knowledge, intelligent escalation
- **Temperature**: 0.7 (creative but focused for conversation)
- **Specialization**: Warm, helpful personality with escalation intelligence
- **Key Features**: 
  - Contextual conversation continuity
  - Intelligent escalation to specialists
  - Conversation history awareness
  - Slack-optimized communication style

### Technical Agent (`technical/`) 👨‍💻
**LLM-Powered Programming & Systems Specialist**
- **Role**: Expert technical support for programming, debugging, and infrastructure
- **Expertise**: Programming languages, DevOps, system administration, code review
- **Temperature**: 0.3 (precise for technical accuracy)
- **Specialization**: Technical domains with user level adaptation
- **Key Features**:
  - Automatic technical domain classification (8 domains)
  - User skill level assessment (beginner/intermediate/advanced)
  - Code examples with syntax highlighting
  - Tool integration recommendations

### Research Agent (`research/`) 🔬
**LLM-Powered Analysis & Research Specialist**
- **Role**: Comprehensive research methodology and strategic analysis
- **Expertise**: Market research, competitive analysis, data insights, academic research
- **Temperature**: 0.4 (balanced analytical creativity)
- **Specialization**: Research types with methodology frameworks
- **Key Features**:
  - Research type classification (8 research areas)
  - Methodology design and recommendations
  - Complexity assessment and scoping
  - Structured research deliverables

## 🔄 Agent Lifecycle Management

### Critical Lifecycle Patterns

All agents MUST follow proper lifecycle management to ensure:
- **Resource Cleanup**: HTTP connections and API clients properly closed
- **Memory Management**: Conversation history and state properly managed  
- **Cost Optimization**: LLM usage tracked and connections reused efficiently
- **Graceful Shutdown**: No hanging connections or resource leaks

### 1. Agent Initialization Pattern

```python
class AgentTemplate:
    """Template for LLM-powered agent with proper lifecycle management."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125", temperature: float = 0.4):
        """Initialize the agent with proper resource management."""
        
        # Initialize primary LLM client
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=500
        )
        
        # Initialize secondary LLM clients if needed
        self.classification_llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,  # Lower temp for classification
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=200
        )
        
        # Initialize prompts and state
        self.main_prompt = self._create_main_prompt()
        self.classification_prompt = self._create_classification_prompt()
        self.interaction_history = []
        
        logger.info(f"Agent initialized with model: {model_name}")
```

### 2. Resource Management Pattern

```python
async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Process message with proper resource tracking."""
    try:
        # Track token usage and costs
        with get_openai_callback() as cb:
            response = await self._generate_response(message, context)
        
        # Log resource usage
        self.interaction_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "tokens_used": cb.total_tokens,
            "cost": cb.total_cost,
            "message_preview": message[:100]
        })
        
        return {
            "response": response,
            "tokens_used": cb.total_tokens,
            "processing_cost": cb.total_cost,
            "metadata": {"model_used": self.llm.model_name}
        }
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return {"response": "Error occurred", "error": str(e)}
```

### 3. **MANDATORY** Cleanup Pattern

```python
async def close(self):
    """
    Close the agent and cleanup resources.
    
    ⚠️ CRITICAL: This method MUST be implemented by every agent.
    Failure to implement proper cleanup leads to:
    - HTTP connection leaks
    - Memory leaks 
    - API client resource exhaustion
    - Potential system instability
    """
    try:
        logger.info("Closing Agent connections...")
        
        # Close ALL LLM clients
        llm_clients = [
            ("Main LLM", self.llm),
            ("Classification LLM", self.classification_llm),
            # Add any other LLM clients here
        ]
        
        for client_name, client in llm_clients:
            if client and hasattr(client, 'client') and hasattr(client.client, 'close'):
                await client.client.close()
                logger.debug(f"Closed {client_name}")
        
        # Clear state to help garbage collection
        self.interaction_history.clear()
        
        logger.info("Agent connections closed successfully")
        
    except Exception as e:
        logger.warning(f"Error closing Agent: {e}")
```

### 4. Agent Building Checklist

When creating a new agent, ensure:

#### ✅ **Initialization Requirements**
- [ ] Proper `__init__` method with model configuration
- [ ] Environment variable validation (OPENAI_API_KEY)
- [ ] LLM clients initialized with appropriate temperatures
- [ ] Prompt templates created and validated
- [ ] Interaction history tracking initialized
- [ ] Logging configuration included

#### ✅ **Processing Requirements**  
- [ ] `async def process_message()` method implemented
- [ ] Token usage tracking with `get_openai_callback()`
- [ ] Comprehensive error handling and fallback responses
- [ ] Context integration for conversation continuity
- [ ] Metadata and performance tracking
- [ ] Response validation and formatting

#### ✅ **Lifecycle Requirements**
- [ ] **`async def close()` method implemented** (MANDATORY)
- [ ] All LLM clients properly closed in `close()` method
- [ ] State cleanup included (history, caches, etc.)
- [ ] Error handling in cleanup process
- [ ] Logging for successful and failed cleanup

#### ✅ **Integration Requirements**
- [ ] Agent exported in `__init__.py` files
- [ ] README.llm.md documentation created
- [ ] Added to orchestrator routing logic
- [ ] Proper import statements and package structure
- [ ] Unit tests for lifecycle management

### 5. Orchestrator Integration Pattern

```python
# In orchestrator/agent_orchestrator.py
async def close(self):
    """Close all agents properly."""
    agents_to_close = [
        ("General Agent", self.general_agent),
        ("Technical Agent", self.technical_agent), 
        ("Research Agent", self.research_agent),
        ("New Agent", self.new_agent)  # Add new agents here
    ]
    
    for agent_name, agent in agents_to_close:
        if agent and hasattr(agent, 'close'):
            try:
                await agent.close()
                logger.info(f"Closed {agent_name}")
            except Exception as e:
                logger.warning(f"Error closing {agent_name}: {e}")
```

### 6. Startup and Shutdown Sequence

#### **Startup Sequence:**
1. **Environment Validation** - Check API keys and configuration
2. **Agent Initialization** - Create agents with LLM clients
3. **Orchestrator Setup** - Register agents with orchestrator
4. **Health Checks** - Verify all agents are responsive
5. **Service Start** - Begin processing requests

#### **Shutdown Sequence:**
1. **Signal Handling** - Catch shutdown signals (Ctrl+C, SIGTERM)
2. **Request Completion** - Allow current requests to finish
3. **Agent Cleanup** - Call `close()` on all agents in reverse order
4. **Orchestrator Cleanup** - Close orchestrator and routing
5. **Connection Cleanup** - Close HTTP sessions and database connections
6. **Task Cancellation** - Cancel any remaining asyncio tasks

## LLM Integration Details

### Shared LLM Configuration
```python
# All agents use consistent LLM setup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Agent-specific temperature tuning
temperatures = {
    "general": 0.7,     # Creative conversation
    "technical": 0.3,   # Technical precision  
    "research": 0.4     # Analytical balance
}
```

### Prompt Engineering Strategy
Each agent implements specialized prompt templates:

1. **System Prompts**: Define role, expertise, personality, and approach
2. **Context Integration**: Include conversation history and user context
3. **Domain Guidance**: Specific instructions for technical/research domains
4. **Response Format**: Consistent structure and metadata expectations

### Intelligence Features

#### Domain Classification
- **Technical Agent**: 8 technical domains (programming, debugging, infrastructure, etc.)
- **Research Agent**: 8 research types (market research, competitive analysis, etc.)
- **General Agent**: Conversation types with escalation assessment

#### Adaptive Responses
- **User Level Detection**: Beginner, intermediate, advanced adaptation
- **Complexity Assessment**: High, medium, low scope evaluation
- **Context Continuity**: Conversation history integration
- **Tool Recommendations**: External tool usage suggestions

## Agent Selection & Routing

### Intelligent Orchestration
The orchestrator uses both **keyword matching** and **LLM-powered classification**:

1. **Explicit Mentions**: @technical, @research, @general
2. **Keyword Scoring**: Domain-specific keyword matching
3. **Pattern Recognition**: Regex patterns for complex requests
4. **Confidence Thresholds**: Minimum confidence for agent selection
5. **Fallback Logic**: General agent for unclear requests

### Routing Priority
```python
AgentType.TECHNICAL: priority=1    # Highest priority for technical requests
AgentType.RESEARCH: priority=2     # Medium priority for research requests  
AgentType.GENERAL: priority=3      # Fallback for general conversations
```

## Performance & Analytics

### Token Tracking
All agents monitor LLM usage:
- **Token Consumption**: Per interaction tracking
- **Cost Analysis**: OpenAI API cost monitoring
- **Performance Metrics**: Response time and quality indicators

### Quality Metrics
- **Domain Classification Accuracy**: How well requests are categorized
- **Response Appropriateness**: Match between request and response complexity
- **User Satisfaction**: Implicit feedback from conversation patterns
- **Escalation Effectiveness**: Success rate of agent-to-agent handoffs

### Analytics Dashboard Data
```python
# Example agent statistics
{
    "total_interactions": 245,
    "total_tokens_used": 89430,
    "total_cost": 1.67,
    "domain_distribution": {...},
    "user_level_distribution": {...},
    "average_response_time": "1.2s"
}
```

## Development Patterns

### Agent Implementation Template
```python
class SpecializedAgent:
    def __init__(self, model_name="gpt-3.5-turbo-0125", temperature=0.4):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.main_prompt = self._create_main_prompt()
        self.classification_prompt = self._create_classification_prompt()
        
    async def process_message(self, message: str, context: Dict[str, Any]):
        # 1. Classify domain/type
        # 2. Assess complexity/level  
        # 3. Generate response with context
        # 4. Return structured result with metadata
        
    async def close(self):
        # ⚠️ MANDATORY: Cleanup all resources
        pass
```

### Error Handling Strategy
- **LLM Failures**: Graceful fallback to keyword-based responses
- **API Limits**: Rate limiting and quota management
- **Validation**: Response quality checks and filtering
- **Logging**: Comprehensive error tracking and diagnostics

### Extensibility Points
- **Custom Prompts**: Domain-specific prompt engineering
- **Tool Integration**: External API and service connections
- **Model Selection**: Easy switching between OpenAI models
- **Metrics Collection**: Custom analytics and performance tracking

## Agent Lifecycle Management

### Initialization
1. **Environment Setup**: API keys and configuration validation
2. **Prompt Loading**: System and user prompt template creation
3. **LLM Connection**: OpenAI API connection establishment
4. **Context Preparation**: Conversation history and state access

### Runtime Operation
1. **Request Classification**: Automatic domain/type identification
2. **Context Assembly**: History and metadata preparation
3. **LLM Invocation**: Structured prompt execution
4. **Response Processing**: Metadata extraction and formatting
5. **State Updates**: Conversation history and analytics logging

### Monitoring & Maintenance
- **Performance Tracking**: Token usage and cost monitoring
- **Quality Assessment**: Response appropriateness evaluation
- **Error Analysis**: Failure pattern identification and resolution
- **Model Updates**: Easy migration to newer OpenAI models

## Future Enhancements

### Advanced LLM Features
- **Function Calling**: Structured tool usage integration
- **Retrieval Augmentation**: Knowledge base and document integration
- **Multi-Modal**: Image and document analysis capabilities
- **Custom Fine-Tuning**: Domain-specific model optimization

### Platform Integration
- **Real-Time Collaboration**: Multi-agent coordination and handoffs
- **Workflow Automation**: Complex task decomposition and execution
- **Knowledge Management**: Persistent learning and expertise accumulation
- **Performance Optimization**: Intelligent caching and response optimization

## Configuration Management

### Environment Variables
```bash
# Shared configuration
OPENAI_API_KEY=sk-your-key-here

# Agent-specific tuning
GENERAL_AGENT_TEMPERATURE=0.7
TECHNICAL_AGENT_TEMPERATURE=0.3
RESEARCH_AGENT_TEMPERATURE=0.4

# Performance settings
MAX_TOKENS_PER_RESPONSE=800
CONTEXT_HISTORY_LIMIT=3
```

### Model Selection Strategy
- **Development**: gpt-3.5-turbo-0125 (cost-effective, fast)
- **Production**: gpt-4 option for higher quality when needed
- **Specialized**: Domain-specific fine-tuned models for expertise areas

## 🚨 Critical Reminders

### **MANDATORY Lifecycle Management**
- **Every agent MUST implement `async def close()`**
- **All LLM clients MUST be closed in the `close()` method**
- **Orchestrator MUST call `close()` on all agents during shutdown**
- **Slack bot MUST call orchestrator `close()` during shutdown**

### **Resource Management**
- **Track token usage and costs for every LLM interaction**
- **Implement proper error handling and fallback mechanisms**
- **Use appropriate temperatures for different agent types**
- **Clear interaction history and state during cleanup**

### **Development Guidelines**
- **Follow the agent building checklist completely**
- **Test lifecycle management in development**
- **Monitor resource usage and cleanup in production**
- **Document any new patterns or requirements**

This LLM-powered architecture transforms the AI Agent Platform from a rule-based system into an intelligent, adaptive platform capable of natural conversation and expert-level assistance across multiple domains while maintaining proper resource management and system stability. 