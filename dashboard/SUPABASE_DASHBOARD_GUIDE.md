# 📊 Supabase Dashboard Integration Guide

The AI Agent Platform dashboard has been updated to display **real data from Supabase** instead of mock data. This provides live monitoring of your system's operations.

## 🆕 New Features

### Real Data Integration
- **Messages**: Live messages from the `messages` table
- **Workflow Runs**: Active runbook executions from `runbook_executions` table  
- **MCP Connections**: User's external service connections from `mcp_connections` table
- **Conversations**: User conversations from `conversations` table
- **Cost Analytics**: Real cost data from `daily_token_summary` and cost tables

### Enhanced Navigation
```
[1] Overview    [2] Agents     [3] Costs
[4] Events      [5] Logs       [6] Messages
[7] Workflows   [8] MCPs       [9] Conversations
[A] Add MCP     [R] Refresh    [Q] Quit
```

## 🚀 Quick Start

### Option 1: Demo Script (Recommended)
```bash
# Run the dashboard demo with sample data
python dashboard/demo_updated_dashboard.py
```

### Option 2: Direct Launch
```bash
# Launch dashboard directly
python dashboard/launch_dashboard.py
```

## 📋 Dashboard Views

### 1. Overview (Key 1)
- Real-time system health metrics
- Today's and weekly costs from Supabase
- Active agent counts
- ROI and patterns discovered
- Recent activity feed

### 2. Messages (Key 6) 
**NEW VIEW** - Shows recent messages from Supabase:
- Message ID, role (user/assistant), content preview
- User ID, timestamp, token count
- Success rate and average tokens
- Color-coded by role and errors

### 3. Workflows (Key 7)
**NEW VIEW** - Shows runbook executions:
- Workflow ID, runbook name, status
- Duration, user, cost, agents/tools used
- Success rates and performance metrics
- Color-coded by execution status

### 4. MCP Connections (Key 8)
**NEW VIEW** - Shows user's external connections:
- Connection name, type, status
- Success rate, tools count, last used
- Total executions and monthly cost
- **Press 'A' to add new MCP connection**

### 5. Conversations (Key 9)
**NEW VIEW** - Shows user conversations:
- Conversation ID, user, status
- Start time, message count, total cost
- Active vs completed conversations

### 6. Cost Analytics (Key 3)
**ENHANCED** - Real cost data from Supabase:
- Today's actual cost and tokens
- Weekly cost trends
- Monthly projections
- Cost optimization opportunities from `cost_optimizations` table

## 🔧 Data Sources

The dashboard pulls from these Supabase tables:

| View | Primary Table | Additional Tables |
|------|---------------|-------------------|
| Messages | `messages` | `conversations` |
| Workflows | `runbook_executions` | - |
| MCP Connections | `mcp_connections` | `mcp_run_cards` |
| Conversations | `conversations` | - |
| Cost Analytics | `daily_token_summary` | `cost_optimizations`, `token_usage` |
| Events | `events` (fallback to messages) | - |

## 🎛️ Interactive Features

### Adding MCP Connections
1. Navigate to MCP Connections view (Key 8)
2. Press 'A' to add new connection
3. Follow prompts to select MCP type and configure

### Real-time Updates
- Dashboard refreshes every 2 seconds
- Press 'R' to force immediate refresh
- All data comes live from Supabase

### Navigation
- Number keys (1-9) switch between views
- 'Q' to quit
- 'A' to add MCP (when in MCP view)
- 'R' to refresh data

## 📊 Sample Data

The demo script creates sample data including:
- Sample conversation with messages
- Sample runbook execution
- Sample MCP connection (GitHub)
- Cost tracking data

## 🛠️ Technical Details

### Components
- `SupabaseDataViewer`: Fetches and formats real Supabase data
- `RealTimeUpdater`: Manages data refresh and caching
- `TerminalDashboard`: Renders the terminal interface

### Data Flow
```
Supabase Tables → SupabaseDataViewer → RealTimeUpdater → TerminalDashboard → User
```

### Error Handling
- Graceful fallback to mock data if Supabase unavailable
- Connection error indicators in each view
- Detailed error logging

## 🔍 Troubleshooting

### Dashboard Not Starting
```bash
# Check Supabase connection
python -c "from database.supabase_logger import SupabaseLogger; SupabaseLogger()"
```

### No Data Showing
1. Check environment variables (`SUPABASE_URL`, `SUPABASE_KEY`)
2. Ensure database migrations are applied
3. Run demo script to create sample data

### Terminal Issues
- Ensure running in proper terminal (not IDE terminal)
- Check if terminal supports curses
- Try different terminal applications

## 🎯 Use Cases

### Development Monitoring
- Track message flow and conversation patterns
- Monitor workflow execution success rates
- Watch cost accumulation in real-time

### Production Operations  
- Monitor system health and performance
- Track MCP connection status and usage
- Identify cost optimization opportunities

### Debugging
- View recent error messages
- Track failed workflow executions
- Monitor MCP connection health

## 📈 Metrics Dashboard Provides

### Performance Metrics
- Message success rates
- Workflow completion rates  
- Agent utilization
- MCP connection health

### Cost Metrics
- Daily/weekly/monthly costs
- Cost per message/workflow
- Optimization opportunities
- ROI tracking

### Usage Metrics
- Active conversations
- MCP tool usage
- Pattern discovery
- User activity

---

**Ready to explore your AI platform's real-time data?** 

Run `python dashboard/demo_updated_dashboard.py` to get started! 🚀 