# 🚀 MCP Integration Quick Start Guide

## Overview
The MCP (Model Context Protocol) integration adds powerful external tool capabilities to your AI Agent Platform. Users can connect to databases, APIs, and services directly through Slack commands, giving agents access to real-world data and actions.

## 🏗️ Architecture Summary

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Slack User    │───▶│   MCP Commands   │───▶│  Connection Manager │
│   /mcp connect  │    │   (Slack Bot)    │    │   (Supabase DB)     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                │                          │
                                ▼                          ▼
                       ┌──────────────────┐    ┌─────────────────────┐
                       │   Run Cards      │    │  Credential Store   │
                       │   (Templates)    │    │   (.env / Secrets)  │
                       └──────────────────┘    └─────────────────────┘
                                │                          │
                                ▼                          ▼
                       ┌──────────────────┐    ┌─────────────────────┐
                       │   Tool Registry  │───▶│   Security Sandbox  │
                       │   (Agent Tools)  │    │   (Safe Execution)  │
                       └──────────────────┘    └─────────────────────┘
```

## 📋 Implementation Checklist

### Phase 1: Database Setup (5 minutes)
- [ ] Run the MCP migration: `database/migrations/003_mcp_integration.sql`
- [ ] Verify tables created: `mcp_connections`, `mcp_tool_usage`, `mcp_run_cards`, etc.
- [ ] Test database connectivity

### Phase 2: Core Integration (10 minutes)
- [ ] Add MCP imports to your Slack bot
- [ ] Initialize MCP commands handler
- [ ] Register MCP command handlers
- [ ] Test `/mcp help` command

### Phase 3: First Connection (5 minutes)
- [ ] Test `/mcp connect supabase` 
- [ ] Verify connection storage
- [ ] Test `/mcp list` and `/mcp tools`
- [ ] Validate agent tool access

### Phase 4: Production Ready (15 minutes)
- [ ] Configure credential storage for your environment
- [ ] Set up security monitoring
- [ ] Add usage analytics
- [ ] Test error handling

**Total Implementation Time: ~35 minutes**

## 🔧 Step-by-Step Implementation

### Step 1: Database Migration

Run the MCP migration in your Supabase dashboard:

```sql
-- Copy and run: database/migrations/003_mcp_integration.sql
-- This creates all MCP tables and analytics views
```

### Step 2: Install Dependencies

```bash
# Optional: For cloud credential storage
pip install google-cloud-secret-manager  # GCP
pip install boto3  # AWS
```

### Step 3: Integrate with Slack Bot

Add to your `slack_interface/slack_bot.py`:

```python
# 1. Add import at the top
from mcp.integration_example import MCPIntegratedSlackBot

# 2. In AIAgentSlackBot.__init__(), after existing setup:
try:
    self.mcp_integration = MCPIntegratedSlackBot(self.supabase_logger)
    self.mcp_integration.register_mcp_handlers(self.app)
    logger.info("🔌 MCP integration enabled")
except Exception as e:
    logger.warning(f"MCP integration disabled: {str(e)}")
    self.mcp_integration = None

# 3. In _handle_message(), before orchestrator call:
if self.mcp_integration:
    context = await self.mcp_integration.enhance_agent_with_mcp_tools(
        None, user_id, context
    )
```

### Step 4: Test the Integration

1. **Test Help Command:**
   ```
   /mcp help
   ```
   Should show MCP command documentation.

2. **Test Connection:**
   ```
   /mcp connect supabase
   ```
   Should open a modal for Supabase credentials.

3. **Test List:**
   ```
   /mcp list
   ```
   Should show user's connections (empty initially).

## 💻 Local Development Setup

### Environment Variables

Add to your `.env` file:

```bash
# MCP Configuration
MCP_SANDBOX_ENABLED=true
MCP_MAX_CONNECTIONS_PER_USER=10
MCP_TOOL_TIMEOUT_SECONDS=30

# Example MCP connection (for testing)
MCP_CRED_MCP_TEST_SUPABASE_1234567890=eyJ1cmwiOiJodHRwczovL3Rlc3QucHJvamVjdC5zdXBhYmFzZS5jbyIsInNlcnZpY2Vfcm9sZV9rZXkiOiJleUpoYkdOcGIycGhkek0uLi4ifQ==
```

### Testing with Supabase

1. **Get Supabase Credentials:**
   - Project URL: `https://your-project.supabase.co`
   - Service Role Key: From Settings > API

2. **Create Test Connection:**
   ```
   /mcp connect supabase
   # Fill in modal with your credentials
   ```

3. **Test Tools:**
   ```
   /mcp tools my_database
   # Should show: list_tables, execute_sql, get_schema, etc.
   ```

## 🔐 Production Deployment

### Credential Storage by Environment

**Local Development (.env):**
```bash
# Credentials stored base64-encoded in .env
MCP_CRED_[CONNECTION_NAME]=[BASE64_CREDENTIALS]
```

**Google Cloud Platform:**
```python
# Automatically detects GCP and uses Secret Manager
# Requires GOOGLE_CLOUD_PROJECT environment variable
```

**AWS:**
```python
# Automatically detects AWS and uses Secrets Manager  
# Requires AWS credentials configured
```

**Docker/Kubernetes:**
```yaml
# Mount secrets as files or environment variables
env:
  - name: MCP_CRED_PROD_DB
    valueFrom:
      secretKeyRef:
        name: mcp-credentials
        key: prod_db_creds
```

### Security Configuration

```python
# In mcp/__init__.py, adjust security settings:
SECURITY_CONFIG = {
    'max_execution_time_seconds': 30,
    'memory_limit_mb': 128,
    'network_restrictions': ['localhost', 'private_ranges'],
    'allowed_protocols': ['https', 'wss'],
    'parameter_validation_enabled': True
}
```

## 📊 Monitoring and Analytics

### Built-in Analytics

```sql
-- Connection performance
SELECT * FROM mcp_connection_analytics;

-- Tool popularity  
SELECT * FROM mcp_tool_popularity;

-- User usage summary
SELECT * FROM user_mcp_summary;
```

### Slack Analytics

```
/mcp analytics
```

Shows user's MCP usage statistics:
- Total connections
- Tool executions
- Success rates
- Tokens saved

## 🛠️ Adding New Run Cards

Create new service integrations:

```python
# mcp/run_cards/github_card.py
class GitHubRunCard:
    def __init__(self):
        self.card_name = "github_integration"
        self.mcp_type = "github"
        self.required_credentials = [
            {"key": "access_token", "label": "GitHub Token"}
        ]
        self.available_tools = [
            "search_repos", "get_issues", "create_issue"
        ]
```

Register in `mcp/slack_interface/mcp_commands.py`:

```python
self.run_cards = {
    'supabase': SupabaseRunCard(),
    'github': GitHubRunCard(),  # Add new card
    'slack': SlackRunCard()
}
```

## 🎯 User Experience Flow

### Quick Setup (30 seconds)
1. User: `/mcp connect supabase`
2. Modal opens with form
3. User enters URL and service key  
4. System tests connection
5. Success: "✅ Connected! 8 database tools now available"

### Using Tools (Seamless)
1. User: "Show me all users from my database"
2. Agent recognizes database intent
3. Agent uses `list_tables` and `execute_sql` tools automatically
4. Real data returned with AI analysis
5. Usage tracked for analytics

## 🚨 Troubleshooting

### Common Issues

**1. Migration Fails:**
```sql
-- Check if tables exist
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' AND table_name LIKE 'mcp_%';
```

**2. Credentials Not Found:**
```bash
# Check .env file has base64 encoded credentials
grep "MCP_CRED_" .env
```

**3. Connection Test Fails:**
```python
# Enable debug logging
import logging
logging.getLogger('mcp').setLevel(logging.DEBUG)
```

**4. Tools Not Available to Agents:**
```python
# Check context enhancement
logger.info(f"MCP tools in context: {context.get('mcp_tools_available', [])}")
```

### Support Resources

- **GitHub Issues:** Report bugs and feature requests
- **Documentation:** Full API documentation in `MCP_INTEGRATION_PLAN.md`
- **Examples:** See `mcp/integration_example.py` for complete examples

## 🎉 What's Next?

After successful integration:

1. **Add More Services:** GitHub, Slack, MongoDB, etc.
2. **Custom Tools:** Build your own API integrations
3. **Advanced Security:** Role-based access, audit logs
4. **Workflow Automation:** Chain tools together for complex tasks
5. **Team Management:** Share connections across team members

**The MCP integration transforms your AI agents from conversational helpers into powerful automation tools with real-world capabilities!** 