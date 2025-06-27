# 🔌 MCP (Model Context Protocol) Integration Plan
## Complete Architecture for Dynamic Tool Integration

## 🎯 **Overview**
Add MCP capability to enable agents to dynamically connect to external tools, databases, and services through a secure, user-managed interface.

## 🏗️ **Architecture Components**

### 1. **MCP Connection Manager**
```
mcp/
├── __init__.py
├── connection_manager.py      # Core MCP connection handling
├── credential_store.py        # Secure credential management  
├── tool_registry.py          # Dynamic tool discovery
├── security_sandbox.py       # Isolation and security
├── run_cards/               # Pre-built connection templates
│   ├── supabase_card.py
│   ├── github_card.py
│   ├── slack_card.py
│   └── custom_card.py
└── slack_interface/         # Slack UI for MCP management
    ├── mcp_commands.py
    ├── connection_modal.py
    └── tool_browser.py
```

### 2. **Database Schema (Supabase)**
```sql
-- MCP Connections
CREATE TABLE mcp_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    connection_name TEXT NOT NULL,
    mcp_type TEXT NOT NULL, -- 'supabase', 'github', 'slack', 'custom'
    connection_config JSONB NOT NULL,
    credential_reference TEXT, -- Reference to secure credential storage
    status TEXT DEFAULT 'active',
    tools_available JSONB, -- List of available tools from this MCP
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used TIMESTAMPTZ,
    usage_count INTEGER DEFAULT 0,
    UNIQUE(user_id, connection_name)
);

-- MCP Tool Usage Tracking
CREATE TABLE mcp_tool_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connection_id UUID REFERENCES mcp_connections(id),
    agent_id TEXT,
    tool_name TEXT,
    execution_time_ms FLOAT,
    success BOOLEAN,
    error_details JSONB,
    tokens_saved INTEGER, -- Tools can save LLM tokens
    executed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Run Cards (Pre-built Templates)
CREATE TABLE mcp_run_cards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    card_name TEXT UNIQUE NOT NULL,
    mcp_type TEXT NOT NULL,
    description TEXT,
    config_template JSONB,
    required_credentials JSONB,
    available_tools JSONB,
    popularity_score INTEGER DEFAULT 0,
    created_by TEXT DEFAULT 'system',
    is_public BOOLEAN DEFAULT true
);
```

## 🔐 **Credential Management Strategy**

### **Environment-Aware Credential Storage**
```python
class CredentialManager:
    """Handles credentials for local vs hosted environments."""
    
    def __init__(self):
        self.environment = self._detect_environment()
        self.storage = self._init_storage()
    
    def _detect_environment(self):
        if os.path.exists('.env'):
            return 'local'
        elif os.getenv('GOOGLE_CLOUD_PROJECT'):
            return 'gcp'
        elif os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
            return 'aws'
        else:
            return 'hosted'
    
    def store_credential(self, connection_name: str, credentials: Dict):
        if self.environment == 'local':
            return self._store_local_env(connection_name, credentials)
        elif self.environment == 'gcp':
            return self._store_gcp_secret_manager(connection_name, credentials)
        elif self.environment == 'aws':
            return self._store_aws_secrets_manager(connection_name, credentials)
        else:
            return self._store_supabase_vault(connection_name, credentials)
```

### **Local Development (.env)**
```bash
# MCP Connections - Local Development
MCP_SUPABASE_URL=https://your-project.supabase.co
MCP_SUPABASE_KEY=your-supabase-key
MCP_GITHUB_TOKEN=ghp_your-github-token
MCP_SLACK_BOT_TOKEN=xoxb-your-slack-token

# MCP Configuration
MCP_SANDBOX_ENABLED=true
MCP_MAX_CONNECTIONS_PER_USER=10
MCP_TOOL_TIMEOUT_SECONDS=30
```

### **Production Hosting**
- **GCP**: Secret Manager integration
- **AWS**: Secrets Manager integration  
- **Docker**: Mounted secrets volume
- **Kubernetes**: Secret objects

## 💬 **Slack Interface for MCP Management**

### **Slash Commands**
```
/mcp connect [service]     # Quick connect with run card
/mcp list                  # Show user's connections
/mcp tools [connection]    # Browse available tools
/mcp disconnect [name]     # Remove connection
/mcp test [connection]     # Test connection health
```

### **Interactive Connection Modal**
```python
@app.command("/mcp")
async def mcp_command(ack, command, client):
    await ack()
    
    if command['text'].startswith('connect'):
        await show_connection_modal(client, command['user_id'])
    elif command['text'] == 'list':
        await show_connections_list(client, command['user_id'])
    elif command['text'].startswith('tools'):
        await show_tools_browser(client, command['user_id'])

async def show_connection_modal(client, user_id):
    modal = {
        "type": "modal",
        "title": {"type": "plain_text", "text": "Connect MCP Service"},
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Choose a service to connect:"}
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🗄️ Supabase"},
                        "action_id": "connect_supabase"
                    },
                    {
                        "type": "button", 
                        "text": {"type": "plain_text", "text": "🐙 GitHub"},
                        "action_id": "connect_github"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "💬 Slack"},
                        "action_id": "connect_slack"
                    }
                ]
            }
        ]
    }
    await client.views_open(trigger_id=trigger_id, view=modal)
```

## 🃏 **Run Cards for Quick Tool Access**

### **Supabase Run Card**
```python
class SupabaseRunCard:
    """Pre-built Supabase MCP connection."""
    
    def __init__(self):
        self.name = "Supabase Database"
        self.description = "Connect to Supabase for database operations"
        self.required_credentials = ['url', 'service_role_key']
        self.available_tools = [
            'list_tables', 'execute_sql', 'get_schema',
            'insert_data', 'update_data', 'delete_data'
        ]
    
    async def quick_setup(self, user_id: str, credentials: Dict):
        """One-click setup for Supabase connection."""
        connection = MCPConnection(
            user_id=user_id,
            name=f"supabase_{user_id}",
            mcp_type="supabase",
            config={
                'url': credentials['url'],
                'features': ['database', 'auth', 'storage']
            }
        )
        
        # Test connection
        if await self.test_connection(connection):
            await self.register_tools(connection)
            return connection
        else:
            raise ConnectionError("Failed to connect to Supabase")
```

### **GitHub Run Card**
```python
class GitHubRunCard:
    """Pre-built GitHub MCP connection."""
    
    def __init__(self):
        self.name = "GitHub Integration"
        self.description = "Connect to GitHub for repository operations"
        self.available_tools = [
            'search_repos', 'get_issues', 'create_issue',
            'get_pull_requests', 'create_pr', 'get_file_content'
        ]
```

## 🔧 **Tool Integration with Agents**

### **Dynamic Tool Loading**
```python
class MCPToolRegistry:
    """Manages available tools from MCP connections."""
    
    async def get_user_tools(self, user_id: str) -> List[Tool]:
        """Get all available tools for a user."""
        connections = await self.get_active_connections(user_id)
        tools = []
        
        for connection in connections:
            connection_tools = await self.load_connection_tools(connection)
            tools.extend(connection_tools)
        
        return tools
    
    async def execute_tool(self, tool_name: str, parameters: Dict, 
                          user_id: str, agent_id: str) -> Dict:
        """Execute an MCP tool securely."""
        # Find the connection that provides this tool
        connection = await self.find_tool_connection(tool_name, user_id)
        
        # Security check
        if not await self.check_permissions(user_id, connection, tool_name):
            raise PermissionError(f"Access denied to {tool_name}")
        
        # Execute with timeout and sandboxing
        result = await self.sandbox_execute(
            connection, tool_name, parameters, timeout=30
        )
        
        # Log usage
        await self.log_tool_usage(connection.id, agent_id, tool_name, result)
        
        return result
```

### **Agent Integration**
```python
class MCPEnabledAgent(UniversalAgent):
    """Agent with MCP tool capabilities."""
    
    async def process_message(self, message: str, context: Dict) -> Dict:
        # Get user's available tools
        user_tools = await self.mcp_registry.get_user_tools(context['user_id'])
        
        # Add tools to LLM context
        enhanced_context = context.copy()
        enhanced_context['available_tools'] = [
            {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.parameters
            } for tool in user_tools
        ]
        
        # Process with tool awareness
        response = await super().process_message(message, enhanced_context)
        
        # Execute any tool calls
        if 'tool_calls' in response:
            tool_results = await self.execute_tool_calls(
                response['tool_calls'], context['user_id']
            )
            response['tool_results'] = tool_results
        
        return response
```

## 🛡️ **Security & Sandboxing**

### **Connection Security**
```python
class MCPSecuritySandbox:
    """Secure execution environment for MCP tools."""
    
    def __init__(self):
        self.max_execution_time = 30  # seconds
        self.memory_limit = 128  # MB
        self.network_restrictions = ['localhost', 'private_ranges']
    
    async def sandbox_execute(self, connection: MCPConnection, 
                            tool_name: str, parameters: Dict) -> Dict:
        """Execute tool in secure sandbox."""
        
        # Validate parameters
        self.validate_parameters(tool_name, parameters)
        
        # Create isolated execution context
        with self.create_sandbox() as sandbox:
            # Set resource limits
            sandbox.set_memory_limit(self.memory_limit)
            sandbox.set_network_policy(self.network_restrictions)
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self.execute_tool(connection, tool_name, parameters),
                timeout=self.max_execution_time
            )
            
            return result
    
    def validate_parameters(self, tool_name: str, parameters: Dict):
        """Validate tool parameters for security."""
        # SQL injection protection
        if tool_name in ['execute_sql', 'query_database']:
            self.validate_sql_safety(parameters.get('query', ''))
        
        # Path traversal protection
        if 'file_path' in parameters:
            self.validate_file_path(parameters['file_path'])
        
        # URL validation
        if 'url' in parameters:
            self.validate_url_safety(parameters['url'])
```

## 📊 **Usage Analytics & Optimization**

### **Tool Usage Analytics**
```python
class MCPAnalytics:
    """Analytics for MCP tool usage."""
    
    async def get_usage_stats(self, user_id: str) -> Dict:
        """Get user's MCP usage statistics."""
        return {
            'total_connections': await self.count_connections(user_id),
            'tools_used_today': await self.count_daily_usage(user_id),
            'most_used_tools': await self.get_popular_tools(user_id),
            'tokens_saved': await self.calculate_token_savings(user_id),
            'success_rate': await self.calculate_success_rate(user_id)
        }
    
    async def suggest_optimizations(self, user_id: str) -> List[str]:
        """Suggest MCP optimizations."""
        suggestions = []
        
        # Suggest new connections based on usage patterns
        patterns = await self.analyze_usage_patterns(user_id)
        if patterns['database_queries'] > 10:
            suggestions.append("Consider adding a direct database connection")
        
        return suggestions
```

## 🚀 **Implementation Phases**

### **Phase 1: Foundation (Week 1)**
- [ ] Core MCP Connection Manager
- [ ] Basic credential storage (local .env support)
- [ ] Supabase schema setup
- [ ] Simple Slack commands

### **Phase 2: Run Cards (Week 2)**  
- [ ] Supabase run card
- [ ] GitHub run card
- [ ] Connection testing & health checks
- [ ] Basic tool registry

### **Phase 3: Agent Integration (Week 3)**
- [ ] Tool loading into agents
- [ ] Secure tool execution
- [ ] Usage tracking and analytics
- [ ] Error handling & recovery

### **Phase 4: Advanced Features (Week 4)**
- [ ] Security sandboxing
- [ ] Multi-environment credential management
- [ ] Advanced Slack UI (modals, buttons)
- [ ] Performance optimization

### **Phase 5: Production (Week 5)**
- [ ] Hosted environment support (GCP/AWS secrets)
- [ ] Load testing & scaling
- [ ] Documentation & user guides
- [ ] Monitoring & alerting

## 🔗 **Integration Points**

### **With Existing Platform**
- **Agent System**: Tools become available to all agents
- **Goal Orchestrator**: MCP tools can be used in goal execution
- **Cost Tracking**: Tool usage costs tracked alongside LLM costs
- **Approval System**: Expensive tools require human approval
- **Memory System**: Tool results stored in vector memory

### **External Integrations**
- **Supabase**: Database operations, auth, storage
- **GitHub**: Repository management, issue tracking
- **Slack**: Workspace integration, user management
- **Custom APIs**: RESTful services, GraphQL endpoints
- **Databases**: PostgreSQL, MongoDB, Redis

## 📋 **User Experience Flow**

### **Quick Setup**
1. User types `/mcp connect supabase` in Slack
2. Modal appears with Supabase run card
3. User enters URL and service key
4. System tests connection automatically
5. Tools become available to all user's agents
6. Confirmation: "✅ Connected! 8 database tools now available"

### **Using Tools**
1. User asks agent: "Show me all users from my database"
2. Agent recognizes database query intent
3. Agent uses available `list_tables` and `execute_sql` tools
4. Results returned with usage tracking
5. User gets real data with agent analysis

This architecture provides a complete MCP integration that's secure, user-friendly, and production-ready! Want me to start implementing any specific component? 