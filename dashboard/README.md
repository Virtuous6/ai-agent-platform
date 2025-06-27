# 🖥️ AI Agent Platform Dashboard

A retro terminal-style dashboard for monitoring your self-improving AI agent platform in real-time.

## ✨ Features

- **Real-time System Monitoring** - Live system health, agent performance, and cost analytics
- **Retro Terminal UI** - Classic ASCII art and color-coded interface
- **Supabase Integration** - Direct connection to your database for live logs
- **Agent Intelligence** - AI-readable data formats for autonomous monitoring
- **Event Streaming** - Real-time event bus integration
- **Cost Analytics** - Financial optimization tracking and alerts
- **Modular Architecture** - Lightweight, componentized design

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Your AI Agent Platform environment variables set up
- Supabase connection configured

### Launch Dashboard

```bash
# From project root
python dashboard/launch_dashboard.py

# Or as a module
python -m dashboard.launch_dashboard
```

### Navigation

- **[1]** - System Overview
- **[2]** - Agent Monitoring  
- **[3]** - Cost Analytics
- **[4]** - Event Stream
- **[5]** - Real-time Logs
- **[R]** - Force Refresh
- **[Q]** - Quit

## 📊 Dashboard Views

### System Overview
- Overall system health score
- Active agent statistics
- Cost efficiency metrics
- Recent activity feed
- ROI tracking

### Agent Monitoring
- Active agent list with performance metrics
- Success rates and cost per interaction
- Agent spawning activity
- Cache hit rates and efficiency

### Cost Analytics
- Daily/monthly cost projections
- Optimization opportunities
- Cost trend analysis
- Efficiency scoring

### Event Stream
- Real-time system events
- Color-coded event types
- Source tracking
- Event filtering

### Real-time Logs
- Live Supabase log streaming
- Conversation and message logs
- Error tracking and summaries
- Log level filtering

## 🤖 AI-First Design

The dashboard is designed as a **first-class citizen** for AI agents:

### Machine-Readable API

```python
from dashboard.terminal_dashboard import TerminalDashboard

# Initialize dashboard
dashboard = TerminalDashboard(db_logger, orchestrator, event_bus)

# Get AI-consumable data
data = await dashboard.get_api_data()

# Example response structure
{
    "system_health": {
        "overall_score": 0.85,
        "performance_score": 0.92,
        "cost_efficiency": 0.78
    },
    "agent_ecosystem": {
        "active_count": 45,
        "success_rate": 0.91,
        "cache_hit_rate": 0.67
    },
    "alerts": [
        {
            "level": "warning",
            "type": "cost_increase",
            "suggested_action": "analyze_performance",
            "confidence": 0.83
        }
    ],
    "recommendations": [
        {
            "type": "spawn_specialist",
            "specialty": "data_analysis",
            "priority": 4,
            "expected_roi": 2.1
        }
    ]
}
```

### Autonomous Decision Support

The dashboard provides actionable intelligence for AI agents:

- **Spawning Decisions** - Which specialists to create based on demand patterns
- **Resource Allocation** - Where to focus improvement efforts
- **Cost Optimization** - Which optimizations to prioritize by ROI
- **Pattern Recognition** - When to automate workflows
- **Risk Assessment** - Confidence scores for all recommendations

## 🛠️ Architecture

### Components

```
dashboard/
├── __init__.py                 # Main module exports
├── terminal_dashboard.py       # Core dashboard orchestrator
├── launch_dashboard.py         # Platform integration launcher
├── styles/
│   ├── ascii_art.py           # Retro ASCII art and graphics
│   └── colors.py              # Terminal color schemes
├── utils/
│   ├── real_time_updater.py   # Data gathering from all sources
│   └── data_formatter.py      # Data formatting utilities
└── components/
    ├── system_health.py        # System health monitoring
    ├── agent_monitor.py        # Agent performance tracking
    ├── cost_analytics.py       # Financial monitoring
    ├── event_stream.py         # Real-time event display
    └── logs_viewer.py          # Supabase log streaming
```

### Integration Points

- **Supabase Logger** - Real-time database log streaming
- **Agent Orchestrator** - System metrics and agent statistics
- **Event Bus** - Live event streaming and notifications
- **Improvement Orchestrator** - Self-improvement cycle monitoring
- **Cost Analytics** - Financial optimization tracking

## 🎨 Customization

### Color Schemes

```python
# Available retro color schemes
schemes = [
    "green_terminal",    # Classic green on black
    "amber_terminal",    # Retro amber on black  
    "blue_terminal",     # Cool blue theme
    "matrix"            # Matrix-style green
]

# Initialize with custom scheme
colors = TerminalColors(scheme="matrix")
```

### Data Sources

The dashboard automatically connects to your platform components but can be extended:

```python
# Add custom data sources
dashboard.updater.custom_sources = {
    "my_service": my_service_connector,
    "external_api": external_api_client
}
```

## 📝 Real-time Logs

The dashboard streams live logs directly from Supabase:

- **Conversation Logs** - All user interactions
- **Message Logs** - Individual message processing
- **System Events** - Agent spawning, optimizations, errors
- **Performance Metrics** - Response times, success rates, costs

### Log Sources

- `conversations` table - User interaction tracking
- `messages` table - Individual message processing
- `events` table - System event logging
- Custom application logs via Supabase logger

## 🔧 Troubleshooting

### Common Issues

**Dashboard won't start:**
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $SUPABASE_URL
echo $SUPABASE_ANON_KEY

# Verify Python path
python -c "import dashboard; print('Dashboard module found')"
```

**No data displayed:**
- Verify Supabase connection
- Check that agents are running
- Ensure event bus is initialized

**Curses/Terminal issues:**
- Use a proper terminal (not IDE console)
- Ensure terminal supports colors
- Try resizing terminal window

### Fallback Mode

If curses is unavailable, the dashboard falls back to simple text output:

```bash
# Simple dashboard without curses
python dashboard/launch_dashboard.py
```

## 📈 Performance

The dashboard is designed to be lightweight:

- **Memory usage**: < 50MB
- **CPU usage**: < 5% on updates
- **Network**: Minimal API calls to Supabase
- **Refresh rate**: 2 seconds (configurable)

## 🔮 Future Enhancements

- **Web Dashboard** - Browser-based version
- **Mobile View** - Responsive mobile interface  
- **Custom Widgets** - Pluggable dashboard components
- **Historical Analytics** - Trend analysis and forecasting
- **Alert System** - Configurable notifications
- **Export Functions** - Data export and reporting

---

Built with ❤️ for the AI Agent Platform. Embracing that retro terminal aesthetic! 🚀 