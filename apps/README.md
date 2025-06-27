# Applications

This folder contains the main application servers and entry points:

## Slack Bot
- `start_bot.py` - Slack bot interface for the AI agent platform

## Web Servers
- `web_server.py` - Basic web interface and API server (health/metrics)
- `web_dashboard.py` - **Full web dashboard with real Supabase data** ⭐

## Running Applications

### Slack Bot
```bash
python apps/start_bot.py
```

### Web Dashboard (Recommended)
```bash
# Full dashboard with Supabase integration
python apps/web_dashboard.py
```

### Basic Web Server  
```bash
# Basic health/status endpoints only
python apps/web_server.py
```

## Dashboard Features

The **Web Dashboard** (`web_dashboard.py`) provides:

- 📊 **Real-time Overview** - System health, costs, activity metrics
- 💬 **Messages View** - Live messages from Supabase `messages` table
- ⚡ **Workflows** - Runbook executions with status tracking
- 🔌 **MCP Connections** - Manage external service connections
- 🗣️ **Conversations** - User conversation sessions
- 💰 **Cost Analytics** - Real spending data and optimizations
- ➕ **Interactive MCP Creation** - Add new connections via web UI

### Dashboard URLs
- **Main Dashboard**: `http://localhost:5000`
- **Health Check**: `http://localhost:5000/health`
- **API Endpoints**: `http://localhost:5000/api/*`

## Configuration

Both applications require environment variables to be set. See the main [README.md](../README.md) and [config/](../config/) folder for setup instructions.

## Alternative Interfaces

For different interface preferences:

### Terminal Dashboard
```bash
python dashboard/launch_dashboard.py
```

### Web Dashboard (This folder)
```bash
python apps/web_dashboard.py
```

## Quick Start

**For monitoring and management**, use the web dashboard:
```bash
python apps/web_dashboard.py
```

**For Slack integration**, run the bot:
```bash 
python apps/start_bot.py
``` 