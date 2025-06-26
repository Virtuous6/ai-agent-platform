# AI Agent Platform - LLM-Powered

An **intelligent, LLM-powered platform** that orchestrates specialized AI agents through Slack, using OpenAI's ChatGPT with domain-specific expertise and runbook-driven workflows for maximum flexibility and intelligence.

## 🧠 Overview

The AI Agent Platform is built around **LLM-powered intelligence** where each agent uses OpenAI's ChatGPT with specialized prompts to provide expert-level assistance. The platform combines the power of large language models with structured workflows, making it both intelligent and maintainable.

### 🚀 Key Features

- **🤖 LLM-Powered Agents**: ChatGPT-driven specialized agents with domain expertise
- **💬 Slack-First Interface**: Natural conversation interface through Slack
- **🧠 Intelligent Routing**: Smart request classification and agent selection
- **📋 Runbook-Driven Logic**: YAML workflows that enhance LLM decision-making
- **📚 Self-Documenting**: Every component includes README.llm.md for LLM understanding
- **⚡ Async Architecture**: Built for high concurrency and performance
- **📊 Comprehensive Analytics**: Full interaction tracking with LLM cost monitoring

## 🏗️ LLM-Powered Architecture

```
User → Slack Bot → Orchestrator → LLM-Powered Agents → MCP Tools → External Services
                      ↓              ↓
                 Runbook Engine → ChatGPT Integration
                      ↓              ↓
                 State Management ← LLM Analytics → Supabase Database
```

### 🤖 Intelligent Agents

#### General Agent 🤖
**LLM-Powered Conversational Specialist**
- **Intelligence**: ChatGPT with conversation prompts (temperature: 0.7)
- **Capabilities**: Natural conversation, context awareness, intelligent escalation
- **Features**: Conversation continuity, escalation assessment, Slack optimization

#### Technical Agent 👨‍💻  
**LLM-Powered Programming & Systems Expert**
- **Intelligence**: ChatGPT with technical expertise prompts (temperature: 0.3)
- **Capabilities**: Programming, debugging, infrastructure, code review
- **Features**: 8 technical domains, user level adaptation, code examples

#### Research Agent 🔬
**LLM-Powered Analysis & Research Specialist**
- **Intelligence**: ChatGPT with research methodology prompts (temperature: 0.4)
- **Capabilities**: Market research, competitive analysis, data insights
- **Features**: 8 research types, methodology design, structured deliverables

### 🧠 Intelligence Features

- **🎯 Domain Classification**: Automatic intelligent categorization of requests
- **📈 Adaptive Responses**: User skill level detection and complexity assessment
- **🔄 Context Continuity**: Conversation history integration across agent handoffs
- **⚙️ Tool Integration**: Smart recommendations for external tool usage
- **🛡️ Graceful Fallback**: Robust error handling with keyword-based backup

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Slack workspace with bot permissions
- Supabase project
- **OpenAI API key** (required for LLM-powered agents)

### Installation

1. **Clone and setup virtual environment**:
   ```bash
   git clone <repository-url>
   cd ai-agent-platform
   
   # Create and activate virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Run setup script**:
   ```bash
   python setup.py
   ```
   This will:
   - Create a `.env` file with auto-generated secrets
   - Verify all dependencies are installed
   - Show detailed Slack app setup instructions

3. **Configure your credentials** (⚠️ **Required for LLM functionality**):
   ```bash
   # Edit the .env file with your credentials
   OPENAI_API_KEY=sk-your-openai-key-here  # Required for LLM agents
   SLACK_BOT_TOKEN=xoxb-your-slack-token
   SUPABASE_URL=your-supabase-url
   SUPABASE_KEY=your-supabase-key
   ```

4. **Run the LLM-powered bot**:
   ```bash
   python slack_interface/slack_bot.py
   ```

### 🤖 Agent Capabilities Demo

Once running, test the intelligent agents:

```
# General conversation (General Agent)
@botname Hello! How are you doing today?

# Technical assistance (Technical Agent)  
@botname Help me debug this Python error in my Flask application

# Research requests (Research Agent)
@botname Analyze the competitive landscape for our new AI product
```

## 📊 LLM Analytics & Performance

### 🔍 Token Tracking
- **Real-time Monitoring**: Track OpenAI API usage per agent and interaction
- **Cost Analysis**: Monitor LLM costs with detailed breakdowns
- **Performance Metrics**: Response times, token efficiency, quality indicators

### 📈 Intelligence Metrics
- **Domain Classification Accuracy**: How well agents categorize requests
- **Escalation Effectiveness**: Success rate of agent-to-agent handoffs
- **User Adaptation**: Appropriateness of response complexity to user level
- **Conversation Quality**: Context continuity and satisfaction indicators

## 🏗️ Core Components

- **`slack_interface/`** - Slack bot with LLM response formatting
- **`orchestrator/`** - Intelligent request routing with LLM-powered classification  
- **`agents/`** - LLM-powered specialized agents (General, Technical, Research)
- **`runbooks/`** - YAML workflows that enhance LLM decision-making
- **`tools/`** - MCP (Model Context Protocol) tool implementations
- **`state/`** - Conversation context with LLM interaction history
- **`database/`** - Database integration with LLM analytics (Supabase)

## 🧠 LLM Configuration

### Model Selection Strategy
```python
# Agent-specific temperature tuning for optimal performance
temperatures = {
    "general": 0.7,     # Creative conversation
    "technical": 0.3,   # Technical precision  
    "research": 0.4     # Analytical balance
}

# Model options
models = {
    "development": "gpt-3.5-turbo-0125",  # Cost-effective, fast
    "production": "gpt-4",                # Higher quality option
}
```

### Intelligent Features
- **Automatic Domain Classification**: Technical domains, research types, conversation types
- **User Level Adaptation**: Beginner, intermediate, advanced response tuning
- **Context Integration**: Conversation history and user preferences
- **Tool Recommendations**: Smart suggestions for external tool usage

## 📚 Documentation Philosophy

This project follows a **LLM-first documentation approach**:

- Every directory has a `README.llm.md` file explaining its purpose to LLMs
- Business logic is documented in YAML runbooks that LLMs can understand
- Code comments assume LLM readers who need context about the intelligent behavior
- Changes must update documentation to reflect LLM capabilities

### Key Documentation Files

- **`.llm/PROJECT_CONTEXT.md`** - LLM-powered architecture and intelligence principles
- **`{component}/README.llm.md`** - Component-specific LLM integration documentation
- **`agents/README.llm.md`** - Detailed LLM agent capabilities and configuration

## 🔧 Development

### Adding New LLM-Powered Functionality

1. **Understand LLM Integration**: Check `.llm/PROJECT_CONTEXT.md` and agent READMEs
2. **Consider Intelligence Enhancement**: Can this benefit from LLM reasoning?
3. **Follow LLM Patterns**: Use ChatOpenAI, structured prompts, fallback handling
4. **Update LLM Documentation**: Modify README.llm.md files with intelligence details

### Creating New LLM-Powered Agents

```bash
# 1. Create agent directory
mkdir agents/new_agent

# 2. Create LLM-specific documentation
echo "# New LLM-Powered Agent Documentation" > agents/new_agent/README.llm.md

# 3. Implement LLM integration
cat > agents/new_agent/agent.py << 'EOF'
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class NewAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.4)
        self.prompt = self._create_specialized_prompt()
    
    def _create_specialized_prompt(self):
        # Domain-specific LLM prompt engineering
        pass
EOF

# 4. Update agents/README.llm.md with new LLM capabilities
```

## 🧪 Testing LLM Integration

```bash
# Test LLM agent functionality
pytest tests/agents/          # LLM agent integration tests
pytest tests/llm_integration/ # LLM API and response tests
pytest tests/orchestrator/    # Intelligent routing tests

# Test with different OpenAI models
OPENAI_MODEL=gpt-4 pytest tests/agents/
OPENAI_MODEL=gpt-3.5-turbo pytest tests/agents/

# Monitor LLM costs during testing
pytest --cov=agents tests/ --llm-cost-tracking
```

## 💰 LLM Cost Management

### Cost Optimization Strategies
- **Smart Token Usage**: Efficient prompt engineering and context management
- **Model Selection**: Balance between quality (GPT-4) and cost (GPT-3.5-turbo)
- **Caching**: Avoid redundant LLM calls for similar requests
- **Fallback Mechanisms**: Keyword-based responses when LLM is unavailable

### Monitoring & Alerts
- **Real-time Cost Tracking**: Per-agent and per-user LLM usage monitoring
- **Budget Alerts**: Automatic notifications for cost thresholds
- **Usage Analytics**: Token efficiency and cost optimization insights

## 🤝 Contributing

1. **Understand LLM Architecture**: Read `.llm/PROJECT_CONTEXT.md` and agent documentation
2. **Test LLM Integration**: Ensure new features work with ChatGPT prompts
3. **Follow LLM Patterns**: Use async LLM calls, structured prompts, error handling
4. **Update LLM Documentation**: Include intelligence features in README.llm.md files
5. **Monitor LLM Costs**: Add cost tracking for new LLM functionality

## 🆘 Support

- **LLM Integration Help**: Check `agents/README.llm.md` for prompt engineering guidance
- **Testing LLM Features**: Review `tests/agents/` for LLM integration examples
- **Cost Optimization**: See monitoring dashboards for LLM usage patterns
- **Agent Capabilities**: All README.llm.md files contain LLM-specific guidance

---

**🧠 Built with LLMs at the Core** - This platform leverages OpenAI's ChatGPT to provide intelligent, context-aware assistance across multiple specialized domains. Every component is designed to work seamlessly with large language models while maintaining cost efficiency and reliability. 