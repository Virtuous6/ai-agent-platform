# 🌐 Web Dashboard Guide

The AI Agent Platform now has a **beautiful web dashboard** that displays real-time data from your Supabase database. This provides a modern, user-friendly alternative to the terminal dashboard.

## 🚀 Quick Start

### Option 1: Demo with Sample Data (Recommended)
```bash
python apps/demo_web_dashboard.py
```

### Option 2: Direct Launch
```bash
python apps/web_dashboard.py
```

Then open your browser to: **http://localhost:5000**

## 📊 Dashboard Features

### 🏠 Overview Tab
The main dashboard shows:
- **System Health**: Overall platform health with color-coded indicators
- **Cost Metrics**: Today's cost, weekly cost, monthly projections
- **Activity Stats**: Total messages, active MCPs, active conversations
- **Recent Activity**: Live feed of messages and workflow executions

### 💬 Messages Tab
Real-time view of the `messages` table:
- Message content previews with role indicators (user/assistant)
- User IDs, timestamps, token counts
- Success rates and error indicators
- Color-coded status badges

### ⚡ Workflows Tab
Live monitoring of `runbook_executions`:
- Workflow IDs, runbook names, execution status
- Duration tracking, user attribution
- Cost per execution, agents used
- Success/failure rates with visual indicators

### 🔌 MCP Connections Tab ⭐
Interactive MCP management:
- **View all connections**: Name, type, status, success rates
- **Add new MCPs**: Click "➕ Add Connection" button
- **Monitor usage**: Tools count, last used, monthly costs
- **Health status**: Connection status with real-time updates

### 🗣️ Conversations Tab
User session monitoring:
- Conversation IDs, user attribution, session status
- Start times, message counts, total costs
- Active vs completed conversations
- Real-time session tracking

### 💰 Costs Tab
Comprehensive cost analytics:
- **Real-time costs**: Today, weekly, monthly projections
- **Efficiency scores**: Platform cost optimization metrics
- **Optimizations**: Actionable cost-saving recommendations
- **Trend analysis**: Cost patterns and projections

## 🎮 Interactive Features

### Adding MCP Connections
1. Navigate to the **🔌 MCP Connections** tab
2. Click **"➕ Add Connection"** button
3. Select connection type from dropdown (pulls from `mcp_run_cards` table)
4. Fill in connection details:
   - **Connection Name**: Unique identifier
   - **User ID**: Owner of the connection
   - **Description**: Optional details
5. Click **"Create Connection"**
6. Connection appears immediately in the table

### Real-time Updates
- Dashboard refreshes every **30 seconds** automatically
- Connection status indicator shows live Supabase connectivity
- "Last updated" timestamp shows when data was refreshed
- Tab switching loads fresh data immediately

### Navigation
- **Tab-based interface**: Click tabs to switch views
- **Responsive design**: Works on desktop, tablet, mobile
- **Status indicators**: Color-coded badges throughout
- **Loading states**: Visual feedback during data fetching

## 🎨 User Interface

### Modern Design
- **Dark theme**: Professional appearance for technical users
- **Color coding**: Green (success), red (error), yellow (warning), blue (info)
- **Hover effects**: Interactive elements respond to mouse
- **Typography**: Clean, readable fonts optimized for data display

### Responsive Layout
- **Grid-based metrics**: Automatically adjusts to screen size
- **Scrollable tables**: Handle large datasets gracefully
- **Mobile-friendly**: Touch-optimized interface
- **Accessibility**: Keyboard navigation support

### Status Indicators
- **Connection status**: Animated dot showing Supabase connectivity
- **Health indicators**: Visual system health representation
- **Status badges**: Color-coded labels for all statuses
- **Progress indicators**: Loading states and data refresh timing

## 🔧 Technical Details

### Architecture
```
Browser → Flask Web Server → SupabaseDataViewer → Supabase Database
```

### API Endpoints
- `GET /` - Main dashboard page
- `GET /api/overview` - System overview data
- `GET /api/messages` - Messages table data
- `GET /api/workflows` - Workflow runs data
- `GET /api/mcp-connections` - MCP connections data
- `GET /api/conversations` - Conversations data
- `GET /api/costs` - Cost analytics data
- `GET /api/mcp-types` - Available MCP types for creation
- `POST /api/mcp-connections` - Create new MCP connection
- `GET /health` - Health check endpoint

### Data Sources
| Dashboard View | Primary Supabase Table | Additional Tables |
|---------------|------------------------|-------------------|
| Overview | Multiple | `messages`, `runbook_executions`, `mcp_connections` |
| Messages | `messages` | `conversations` |
| Workflows | `runbook_executions` | - |
| MCP Connections | `mcp_connections` | `mcp_run_cards` |
| Conversations | `conversations` | - |
| Costs | `daily_token_summary` | `cost_optimizations`, `token_usage` |

### Real-time Features
- **Auto-refresh**: Every 30 seconds when connected
- **Connection monitoring**: Live Supabase connection status
- **Error handling**: Graceful fallbacks when data unavailable
- **Caching**: Efficient data retrieval and display

## 🚫 Comparison: Web vs Terminal Dashboard

| Feature | Web Dashboard | Terminal Dashboard |
|---------|---------------|-------------------|
| **Interface** | Modern web UI | ASCII art terminal |
| **Navigation** | Click tabs | Keyboard numbers |
| **MCP Management** | Interactive forms | Key commands |
| **Data Display** | Tables & cards | Text-based tables |
| **Real-time Updates** | 30-second refresh | 2-second refresh |
| **Accessibility** | Mouse & keyboard | Keyboard only |
| **Platform** | Any web browser | Terminal required |
| **Deployment** | Web server | Local terminal |

## 🌟 Use Cases

### Development & Testing
- Monitor message flow during development
- Track workflow execution success rates
- Test MCP connection creation and management
- Observe real-time cost accumulation

### Production Operations
- Monitor system health and performance
- Manage MCP connections across teams
- Track cost optimization opportunities
- Analyze user conversation patterns

### Team Collaboration
- **Shared monitoring**: Multiple team members can view same dashboard
- **Remote access**: Web-based access from anywhere
- **Visual reports**: Easy-to-understand metrics and charts
- **Interactive management**: Team members can add MCPs

### Executive Reporting
- **High-level metrics**: System health and cost at a glance
- **Visual appeal**: Professional dashboard suitable for presentations
- **Real-time data**: Always current information
- **Cost tracking**: Clear spending visibility

## 🛠️ Customization

### Theming
The dashboard uses CSS custom properties for easy theming:
```css
:root {
    --primary-color: #6366f1;      /* Main brand color */
    --success-color: #10b981;      /* Success indicators */
    --error-color: #ef4444;        /* Error indicators */
    --bg-primary: #0f172a;         /* Main background */
    --text-primary: #f8fafc;       /* Main text color */
}
```

### Extending Features
- Add new tabs by updating `dashboard.html` and `dashboard.js`
- Create new API endpoints in `web_dashboard.py`
- Add new data sources via `SupabaseDataViewer`
- Customize styling in `dashboard.css`

## 🔍 Troubleshooting

### Dashboard Won't Start
```bash
# Check Supabase connection
python -c "from database.supabase_logger import SupabaseLogger; SupabaseLogger()"

# Verify Flask installation
pip install flask

# Check environment variables
echo $SUPABASE_URL
echo $SUPABASE_KEY
```

### No Data Showing
1. **Check Supabase**: Verify environment variables are set
2. **Run migrations**: Ensure database tables exist
3. **Create sample data**: Run demo script with sample data option
4. **Check browser console**: Look for JavaScript errors

### Connection Issues
- **Red status indicator**: Supabase connection failed
- **Yellow status indicator**: Connecting/retrying
- **Green status indicator**: Connected successfully

### Performance Issues
- **Large datasets**: Tables automatically limit results
- **Slow loading**: Check Supabase region and connection
- **Memory usage**: Dashboard designed for efficiency

## 📱 Mobile Support

The web dashboard is fully responsive:
- **Tablet view**: Optimized layout for tablets
- **Mobile view**: Touch-friendly interface
- **Responsive tables**: Horizontal scrolling for large tables
- **Touch gestures**: Native mobile interactions

---

## 🎯 Getting Started Checklist

- [ ] Start the web dashboard: `python apps/demo_web_dashboard.py`
- [ ] Open browser to: `http://localhost:5000`
- [ ] Check connection status (should be green)
- [ ] Explore the Overview tab
- [ ] Try creating an MCP connection
- [ ] Monitor real-time data updates
- [ ] Bookmark for daily use!

**Ready to monitor your AI platform with style?** 🚀

The web dashboard provides everything you need to manage and monitor your AI Agent Platform with a beautiful, professional interface! 