# AI Agent Platform 2.0 - Thin Engine + Supabase Storage

A self-improving, multi-agent AI platform that learns from every interaction. **Your git repo becomes a thin, powerful engine while Supabase stores everything dynamic.**

## 🚀 Architecture: Thin Engine + Database Intelligence

**✅ NO MORE:** Config files, static workflows, deployments for changes  
**✅ NOW:** Dynamic configs, database workflows, real-time updates

- **Git Repo** = Lean execution engine (~200 line orchestrator)
- **Supabase** = All intelligence, configs, patterns, and analytics

## 🚀 What's New in 2.0

- **80% Less Code**: Removed over-engineering, kept the gold
- **Thin Engine**: Your code is now a lean execution engine
- **Supabase Storage**: All configs, patterns, and intelligence in database
- **Zero Deployments**: Update agents, prompts, workflows via database
- **Dynamic Agent Spawning**: Agents created on-demand from database configs
- **True Multi-Agent**: Agents collaborate through events
- **Self-Improving**: Every interaction tracked and analyzed in Supabase
- **Real-time Updates**: Change anything without restarting

## 📁 Architecture

```
ai-agent-platform/
├── core/                    # Heart of the system
│   ├── agent.py            # UniversalAgent - one class, infinite configs
│   ├── orchestrator.py     # Simple 200-line orchestrator
│   ├── events.py           # Perfect event bus (unchanged)
│   └── workflow.py         # Multi-agent workflow engine
│
├── evolution/              # Self-improvement
│   ├── tracker.py         # Tracks every workflow
│   ├── learner.py         # Learns from patterns & feedback
│   └── memory.py          # Vector store for context
│
├── tools/                  # MCP integrations
│   ├── mcp_manager.py     # MCP connection management
│   ├── registry.py        # Tool registry
│   └── cards/             # MCP implementations
│       ├── serper.py      # Web search
│       ├── supabase.py    # Database
│       └── github.py      # Code integration
│
├── adapters/              # Platform adapters
│   ├── base.py           # Abstract adapter
│   └── slack.py          # Slack integration
│
├── storage/               # Persistence
│   ├── supabase.py       # Database logger
│   └── configs/          # Agent configurations
│       ├── general.yaml
│       ├── technical.yaml
│       └── research.yaml
│
├── main.py               # Clean entry point
└── config.yaml          # System configuration
```

## 🎯 Key Concepts

### 1. Universal Agent
Instead of multiple agent classes, we have ONE configurable agent:
```python
# Old way: Multiple classes
class GeneralAgent: ...
class TechnicalAgent: ...
class ResearchAgent: ...

# New way: One class, config-driven
agent = UniversalAgent(config=load_config("technical"))
```

### 2. Dynamic Spawning
Agents are created on-demand based on context:
```python
# Orchestrator automatically spawns the right agent
response = await orchestrator.process("Debug this code", context)
# ^ Spawns technical agent automatically
```

### 3. Event-Driven Everything
Agents communicate through events:
```python
# Agent publishes completion
await event_bus.publish("task_completed", data)

# Other agents can subscribe and react
await event_bus.subscribe("task_completed", handle_completion)
```

### 4. Continuous Learning
Every interaction is analyzed:
```python
# Automatic pattern recognition
run_id = track_workflow(message, context)
asyncio.create_task(learner.analyze_interaction(run_id))
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repo
git clone https://github.com/yourusername/ai-agent-platform.git
cd ai-agent-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy example env
cp config/env.example .env

# Edit .env with your keys
OPENAI_API_KEY=your-key
SUPABASE_URL=your-url
SUPABASE_KEY=your-key
SLACK_BOT_TOKEN=your-token  # Optional
```

### 3. Run the Platform

**CLI Mode (for testing):**
```bash
python main.py cli
```

**Slack Mode:**
```bash
python main.py slack
```

**Or just run (auto-detects mode):**
```bash
python main.py
```

## 💬 Using the Platform

### Basic Commands
- `/improve` - Improve your last workflow
- `/save-workflow [name]` - Save current workflow  
- `/list-workflows` - List your saved workflows
- `/status` - Show system status
- `/agents` - List active agents

### Example Interactions

**Simple Question:**
```
You: What is machine learning?
Agent: Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed...
```

**Technical Task:**
```
You: Debug this Python code: def fib(n): return fib(n-1) + fib(n-2)
Agent: I see the issue with your Fibonacci function. It's missing base cases...
```

**Multi-Agent Workflow:**
```
You: Research quantum computing and create a technical summary
Agent: [Research agent gathers information]
Agent: [Technical agent creates summary]
Agent: Here's your technical summary of quantum computing...
```

### Improving Workflows

```
You: Find the latest AI news
Agent: Here are the latest AI developments...

You: /improve make it focus on LLMs only
Agent: ✅ Workflow improved! Future searches will prioritize LLM news.

You: /save-workflow LLM News Tracker
Agent: ✅ Workflow saved!
```

## 🧠 How It Learns

1. **Pattern Recognition**: Frequently used message patterns become workflows
2. **User Feedback**: `/improve` commands directly enhance capabilities
3. **Performance Tracking**: Successful workflows are prioritized
4. **Agent Evolution**: New specialist agents spawn based on demand

## 🔧 Configuration

### Agent Configuration (storage/configs/)
```yaml
name: "Technical Expert"
model: "gpt-3.5-turbo"
temperature: 0.3
system_prompt: |
  You are a technical expert...
triggers:
  - "code"
  - "debug"
  - "error"
```

### System Configuration (config.yaml)
```yaml
agents:
  max_active: 50
  cleanup_interval: 3600
  
learning:
  pattern_threshold: 3
  analysis_interval: 21600
```

## 🛠️ Development

### Adding New Agent Types
1. Create config in `storage/configs/newtype.yaml`
2. Add intent classification in `orchestrator._classify_intent()`
3. That's it! The system handles the rest

### Adding New Tools (MCP)
1. Create card in `tools/cards/newtool.py`
2. Register in `tools/registry.py`
3. Tools are automatically available to all agents

### Creating Workflows
```python
workflow_engine.register_workflow("my_workflow", {
    "steps": [
        {"agent_type": "research", "prompt": "Research {topic}"},
        {"agent_type": "technical", "prompt": "Analyze {research_results}"}
    ]
})
```

## 📊 Monitoring

- **Supabase Dashboard**: View all interactions, patterns, and performance
- **Event Stream**: Real-time event monitoring
- **System Status**: `/status` command shows active agents and health

## 🤝 Contributing

We love contributions! The codebase is now simple and approachable:

1. Fork the repository
2. Create your feature branch
3. Make your changes (keep it simple!)
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

Built with:
- OpenAI GPT for agent intelligence
- Supabase for persistence
- Model Context Protocol (MCP) for tools
- Slack SDK for communication

---

**Remember**: Simplicity is the ultimate sophistication. This platform proves that less code can do more when designed thoughtfully. 