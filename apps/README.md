# Applications

This folder contains the main application servers and entry points:

## Slack Bot
- `start_bot.py` - Slack bot interface for the AI agent platform

## Web Server
- `web_server.py` - Web interface and API server

## Running Applications

### Slack Bot
```bash
python apps/start_bot.py
```

### Web Server  
```bash
python apps/web_server.py
```

## Configuration

Both applications require environment variables to be set. See the main [README.md](../README.md) and [config/](../config/) folder for setup instructions.

## Dashboard Alternative

For a terminal-based interface, you can also use:
```bash
python dashboard/launch_dashboard.py
``` 