# 🧪 MCP Integration Testing Guide

## Test the complete MCP (Model Context Protocol) integration we just built in Supabase project: `pfvlmoybzjkajubzlnsx`

---

## 🎯 **TESTING PROMPT**

*Use this prompt with your AI agent to comprehensively test the MCP integration:*

---

**"I want to test the MCP (Model Context Protocol) integration we just implemented. Please help me validate all components are working correctly. Here's what I need you to test:**

### **Phase 1: Database Schema Validation**
1. **Verify all MCP tables exist and have correct structure**
2. **Check that indexes are created for performance**
3. **Validate foreign key relationships between tables**
4. **Confirm triggers are active and working**

### **Phase 2: Seed Data Testing**
1. **List all available MCP run cards**
2. **Show the tools available for each service (Supabase, GitHub, Slack, PostgreSQL)**
3. **Verify the tool counts match what was seeded**
4. **Check popularity scores and categories are correct**

### **Phase 3: Simulate MCP Usage Workflow**
1. **Create a test MCP connection** for a user
2. **Log some simulated tool usage** with different success/failure scenarios
3. **Test the automatic trigger** that updates connection usage counts
4. **Verify cost tracking** and token savings calculations

### **Phase 4: Analytics & Monitoring**
1. **Query the MCP connection health view**
2. **Test the cost analytics view** with date ranges
3. **Generate usage insights** for optimization recommendations
4. **Check security logging** for audit trails

### **Phase 5: Performance & Security**
1. **Test bulk data operations** to ensure indexes work
2. **Validate security constraints** (user isolation, data validation)
3. **Test credential storage options** (simulate different environments)
4. **Verify cache hit tracking** and performance metrics

### **Phase 6: Real-World Scenarios**
1. **Simulate a power user** with multiple connections and heavy usage
2. **Test cost optimization** scenarios with different token patterns
3. **Generate actionable insights** based on usage patterns
4. **Test connection health monitoring** and failure handling

### **Expected Outcomes:**
- All database operations should complete successfully
- Analytics views should return meaningful data
- Triggers should automatically update related tables
- Security logging should capture all relevant events
- Cost tracking should provide accurate estimates
- The system should scale to handle multiple users and connections

**Please run through each phase systematically and report any issues or successful validations. Show me the actual data and queries you're running to verify everything works as designed.**"

---

## 🔧 **Quick Test Commands**

If you want to run specific tests yourself, here are key SQL queries:

### Test Tables & Structure
```sql
-- Check all MCP tables exist
SELECT table_name, table_comment 
FROM information_schema.tables 
WHERE table_name LIKE 'mcp_%' 
ORDER BY table_name;
```

### Test Seed Data
```sql
-- Verify run cards loaded correctly
SELECT service_name, display_name, tool_count, category 
FROM mcp_run_cards 
ORDER BY popularity_score DESC;
```

### Test Analytics Views
```sql
-- Test connection health view
SELECT * FROM mcp_connection_health LIMIT 5;

-- Test cost analytics view  
SELECT * FROM mcp_cost_analytics LIMIT 5;
```

### Simulate Usage & Test Triggers
```sql
-- Create test connection
INSERT INTO mcp_connections (user_id, service_name, connection_name, mcp_server_url)
VALUES ('test_user_123', 'supabase', 'My Test DB', 'supabase://test');

-- Simulate tool usage (this will trigger automatic updates)
INSERT INTO mcp_tool_usage (
    connection_id, user_id, tool_name, success, 
    input_tokens, output_tokens, estimated_cost
) 
SELECT id, 'test_user_123', 'execute_sql', true, 150, 50, 0.02
FROM mcp_connections WHERE connection_name = 'My Test DB';

-- Verify trigger worked (should show total_tool_calls = 1)
SELECT connection_name, total_tool_calls, last_used 
FROM mcp_connections 
WHERE connection_name = 'My Test DB';
```

---

## 🎯 **Success Criteria**

Your MCP integration is working correctly if:

✅ All 5 MCP tables exist with proper structure  
✅ 4 run cards loaded with correct tool counts  
✅ Analytics views return data without errors  
✅ Triggers automatically update usage counters  
✅ Security logging captures events  
✅ Cost tracking calculates estimates  
✅ Performance indexes speed up queries  

---

## 🚀 **Next Steps After Testing**

Once testing confirms everything works:

1. **Integrate MCP commands** into your Slack bot
2. **Set up credential storage** for your environment
3. **Configure cost limits** and monitoring alerts
4. **Deploy to production** with confidence
5. **Train users** on `/mcp connect` commands

**Your AI agents are now ready to connect to real-world services! 🔥** 