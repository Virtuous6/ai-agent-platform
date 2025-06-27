# Supabase Runbook Management Guide

## 🎯 Overview

The AI Agent Platform now uses **Supabase for global runbook storage** instead of YAML files. This provides better performance, intelligent querying, analytics, and centralized management.

## 🚀 Migration from YAML to Supabase

### What Changed
- **Before**: Runbooks stored as YAML files in `runbooks/active/`
- **After**: Runbooks stored in Supabase database with intelligent querying
- **Benefits**: Better performance, analytics, version control, and trigger matching

### Migration Steps

1. **Set up Supabase credentials**:
   ```bash
   export SUPABASE_URL="your_supabase_url"
   export SUPABASE_SERVICE_ROLE_KEY="your_service_role_key"
   ```

2. **Run the database migration**:
   ```sql
   -- Run in Supabase SQL editor
   \i database/migrations/004_global_runbooks.sql
   ```

3. **Migrate existing runbooks**:
   ```bash
   python scripts/migrate_runbooks_to_supabase.py
   ```

4. **Verify migration**:
   ```bash
   python scripts/migrate_runbooks_to_supabase.py --verify-only
   ```

## 📊 Database Schema

### Core Tables

**`global_runbooks`** - Main runbook storage
```sql
CREATE TABLE global_runbooks (
    id UUID PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    version TEXT DEFAULT '1.0.0',
    description TEXT,
    definition JSONB NOT NULL,        -- Runbook YAML as JSON
    category TEXT DEFAULT 'general',
    tags TEXT[],
    priority INTEGER,
    status TEXT DEFAULT 'active',
    usage_count INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 0.0,
    agent_compatibility TEXT[],       -- ['general', 'technical', 'research']
    llm_context TEXT,
    -- ... timestamps and metadata
);
```

**`runbook_triggers`** - Trigger conditions
```sql
CREATE TABLE runbook_triggers (
    id UUID PRIMARY KEY,
    runbook_id UUID REFERENCES global_runbooks(id),
    condition_type TEXT NOT NULL,     -- 'message_contains', 'agent_mention', etc.
    parameters JSONB DEFAULT '{}',
    priority INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE
);
```

**`runbook_versions`** - Version history
```sql
CREATE TABLE runbook_versions (
    id UUID PRIMARY KEY,
    runbook_id UUID REFERENCES global_runbooks(id),
    version TEXT NOT NULL,
    definition JSONB NOT NULL,
    changes_summary TEXT,
    deployed_at TIMESTAMPTZ,
    is_current BOOLEAN DEFAULT FALSE
);
```

## 🔧 Using the Runbook Manager

### Basic Operations

```python
from runbooks.manager import get_runbook_manager

# Initialize
runbook_manager = await get_runbook_manager()

# Get a specific runbook
runbook = await runbook_manager.get_runbook("answer-question")

# List all runbooks
runbooks = await runbook_manager.list_runbooks()

# Find matching runbooks for a message
matches = await runbook_manager.find_matching_runbooks(
    message="How do I debug Python code?",
    agent_type="technical"
)

# Create a new runbook
success = await runbook_manager.create_runbook(
    name="new-workflow",
    definition=runbook_dict,
    category="technical",
    agent_compatibility=["technical", "general"]
)
```

### Intelligent Trigger Matching

The system automatically finds the best runbook for each message:

```python
# The database function handles intelligent matching
result = client.rpc("find_matching_runbooks", {
    "message_text": "What is machine learning?",
    "agent_type": "research",
    "user_context": json.dumps({})
})
```

## 📈 Analytics and Performance

### Built-in Analytics Views

**Runbook Performance Dashboard**:
```sql
SELECT * FROM runbook_performance_dashboard;
-- Shows usage, success rates, trigger counts
```

**Execution Analytics**:
```sql
SELECT * FROM runbook_execution_analytics;
-- Correlates stored stats with actual execution data
```

### Performance Monitoring

```python
# Get analytics
analytics = await runbook_manager.get_analytics()

print(f"Total runbooks: {analytics['total_runbooks']}")
print(f"Average success rate: {analytics['performance_summary']['avg_success_rate']:.1%}")
print(f"Most used: {analytics['performance_summary']['most_used']}")
```

## 🤖 Integration with AI Agents

### Orchestrator Integration

The orchestrator now automatically selects runbooks from Supabase:

```python
# In agent_orchestrator.py
async def _select_runbook_for_message(self, message: str, context: Dict[str, Any]) -> str:
    runbook_manager = await get_runbook_manager()
    
    # Find best matching runbook
    matches = await runbook_manager.find_matching_runbooks(
        message=message,
        agent_type=context.get("preferred_agent"),
        user_context=context
    )
    
    if matches:
        return matches[0][0].name  # Best match
    
    return "answer-question"  # Fallback
```

### LangGraph Workflow Engine

Workflows are now loaded from Supabase:

```python
# Load from Supabase instead of YAML files
workflow = await workflow_engine.load_runbook_from_supabase("answer-question")
```

## 🛠️ Management Commands

### Migration Script Usage

```bash
# Full migration
python scripts/migrate_runbooks_to_supabase.py

# Force overwrite existing
python scripts/migrate_runbooks_to_supabase.py --force

# Verify only (no migration)
python scripts/migrate_runbooks_to_supabase.py --verify-only

# Show analytics
python scripts/migrate_runbooks_to_supabase.py --analytics
```

### Database Functions

**Update usage statistics**:
```sql
SELECT update_runbook_usage('answer-question', true, 45.2);
```

**Get active runbooks**:
```sql
SELECT * FROM get_active_runbooks('user_interaction');
```

**Find matching runbooks**:
```sql
SELECT * FROM find_matching_runbooks('How do I debug code?', 'technical');
```

## 🔍 Troubleshooting

### Common Issues

1. **"Runbook manager not initialized"**
   - Ensure Supabase credentials are set
   - Run `await initialize_runbook_manager(url, key)`

2. **"No runbooks found"**
   - Check migration completed: `--verify-only`
   - Verify runbook status is 'active'

3. **"LangGraph workflow loading failed"**
   - Check runbook definition structure
   - Verify agent compatibility arrays

### Health Check

```python
# Check runbook manager health
status = await runbook_manager.health_check()
print(f"Database accessible: {status['database_accessible']}")
print(f"Total runbooks: {status['total_runbooks']}")
```

## 🚀 Future Enhancements

### Planned Features
- **User-specific runbooks**: Personal workflows per user
- **Runbook templates**: Standardized patterns for creation
- **A/B testing**: Compare runbook variations
- **Auto-generation**: Create runbooks from successful patterns
- **Visual editor**: GUI for runbook creation and editing

### Advanced Analytics
- **Performance predictions**: ML-based success rate forecasting
- **Usage optimization**: Automatic trigger refinement
- **Cost analytics**: Token usage and cost tracking per runbook
- **User satisfaction correlation**: Link runbook usage to feedback

## 📚 API Reference

### SupabaseRunbookManager

```python
class SupabaseRunbookManager:
    async def get_runbook(name: str) -> Optional[GlobalRunbook]
    async def list_runbooks(category: str = None, agent_type: str = None) -> List[GlobalRunbook]
    async def find_matching_runbooks(message: str, agent_type: str = None, user_context: Dict = None) -> List[Tuple[GlobalRunbook, float]]
    async def create_runbook(name: str, definition: Dict, ...) -> bool
    async def update_runbook(name: str, ...) -> bool
    async def record_execution(runbook_name: str, success: bool, execution_time: float) -> None
    async def get_analytics() -> Dict[str, Any]
    async def migrate_yaml_runbook(yaml_path: str) -> bool
    async def health_check() -> Dict[str, Any]
```

### GlobalRunbook DataClass

```python
@dataclass
class GlobalRunbook:
    id: str
    name: str
    version: str
    description: str
    definition: Dict[str, Any]  # YAML content as dict
    category: str
    tags: List[str]
    priority: int
    status: str
    llm_context: Optional[str]
    agent_compatibility: List[str]
    usage_count: int
    success_rate: float
    triggers: List[RunbookTrigger]
```

---

🎉 **Congratulations!** You now have intelligent, scalable runbook management with Supabase. The system provides better performance, analytics, and will enable user-specific runbooks in the future. 