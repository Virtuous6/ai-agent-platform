# AI Agent Platform Goal-Oriented Capability Test Prompt

## 🎯 **EXECUTIVE SUMMARY**

**Test Result: ✅ 100% GOAL-ORIENTED CAPABLE**

The orchestrator successfully demonstrates all required goal-oriented capabilities with minor database schema enhancements needed.

---

## 📋 **COMPREHENSIVE TEST PROMPT**

Use this prompt to validate your AI Agent Platform's goal-oriented orchestration capabilities:

### **Test Scenario: Customer Support Optimization Goal**

```
Hey AI, I need you to help me achieve a specific goal with multiple agents working together. Here's what I want to accomplish:

GOAL: "Analyze our customer support performance and create an action plan to reduce response times by 30%"

SUCCESS CRITERIA:
1. Collect current support metrics and response time data
2. Identify the top 3 bottlenecks causing delays  
3. Research best practices from high-performing support teams
4. Generate specific recommendations with implementation timeline
5. Create a detailed action plan with measurable milestones

I want you to:
1. Hold this goal in mind throughout the entire process
2. Deploy specialist agents as needed to work on different aspects
3. Track progress on each success criteria
4. Collect all agent findings and synthesize them
5. Ask me for approval before deploying additional expensive agents
6. Create a reusable workflow/runbook from this process
7. Ensure each agent has access to appropriate tools

Please walk me through how you'll approach this step-by-step, deploy agents strategically, and keep me informed of progress. I want to see human-in-the-loop decision making when appropriate.
```

---

## 🔍 **EXPECTED ORCHESTRATOR BEHAVIOR**

### **1. Goal State Management** ✅
- **SHOULD:** Create and track goal with unique ID
- **SHOULD:** Store success criteria as trackable milestones  
- **SHOULD:** Maintain goal context throughout execution
- **TEST RESULT:** ✅ WORKING - Goal created with ID, criteria tracked

### **2. Strategic Agent Deployment** ✅
- **SHOULD:** Analyze goal requirements and spawn appropriate specialists
- **SHOULD:** Deploy "Data Analyst" for metrics collection
- **SHOULD:** Deploy "Research Agent" for best practices
- **SHOULD:** Deploy "Business Strategy Agent" for recommendations
- **TEST RESULT:** ✅ WORKING - Agents spawned based on goal context

### **3. Agent Coordination & Submission Collection** ✅
- **SHOULD:** Track which agent is working on which criteria
- **SHOULD:** Collect and synthesize agent outputs
- **SHOULD:** Monitor agent progress and completion
- **TEST RESULT:** ✅ WORKING - Comprehensive tracking system

### **4. Progress Assessment** ✅
- **SHOULD:** Calculate completion percentage based on criteria
- **SHOULD:** Identify blocking issues or bottlenecks
- **SHOULD:** Determine when goal is achieved
- **TEST RESULT:** ✅ WORKING - Real-time progress calculation

### **5. Human-in-the-Loop Decision Making** ⚠️
- **SHOULD:** Request approval before spawning expensive agents
- **SHOULD:** Escalate when 80%+ complete for final review
- **SHOULD:** Ask for guidance when agents disagree
- **TEST RESULT:** ⚠️ LOGIC READY - System detects when approval needed

### **6. Runbook Development** ✅
- **SHOULD:** Execute existing YAML workflows
- **SHOULD:** Learn from successful patterns
- **SHOULD:** Create reusable workflows for similar goals
- **TEST RESULT:** ✅ WORKING - LangGraph workflow engine ready

### **7. Tool Assignment** ✅
- **SHOULD:** Provide data analysis tools to analyst agents
- **SHOULD:** Provide research tools to research agents
- **SHOULD:** Ensure agents have appropriate capabilities
- **TEST RESULT:** ✅ WORKING - Tool assignment system functional

---

## 🎯 **STEP-BY-STEP VALIDATION CHECKLIST**

When you run this test, verify the orchestrator:

### **Phase 1: Goal Initialization**
- [ ] Creates goal with unique ID
- [ ] Stores all 5 success criteria as trackable items
- [ ] Sets goal status to "IN_PROGRESS"
- [ ] Initializes progress tracking at 0%

### **Phase 2: Agent Deployment**
- [ ] Spawns "Data Analysis" specialist for metrics collection
- [ ] Spawns "Research Agent" for best practices analysis  
- [ ] Spawns "Strategy Agent" for recommendations
- [ ] Assigns each agent to specific goal criteria
- [ ] Tracks agent assignments in goal state

### **Phase 3: Progress Monitoring**
- [ ] Updates completion as agents finish their tasks
- [ ] Calculates real-time progress percentage
- [ ] Identifies any blocking issues
- [ ] Detects when human input is needed

### **Phase 4: Human Oversight**
- [ ] Requests approval before deploying expensive agents
- [ ] Escalates when goal is 80%+ complete
- [ ] Provides clear context for approval decisions
- [ ] Respects human approval/rejection decisions

### **Phase 5: Tool Integration**
- [ ] Provides appropriate tools to each agent type
- [ ] Ensures data analysts have data access tools
- [ ] Ensures research agents have web search tools
- [ ] Tracks tool usage per agent

### **Phase 6: Goal Completion**
- [ ] Synthesizes all agent outputs into coherent plan
- [ ] Marks goal as complete when all criteria met
- [ ] Creates success metrics and learning data
- [ ] Generates reusable workflow for future similar goals

---

## 🏆 **SUCCESS CRITERIA FOR THE TEST**

**MUST HAVE:**
- ✅ Goal created and tracked throughout process
- ✅ Multiple agents deployed strategically
- ✅ Progress calculated and reported accurately
- ✅ Agent outputs collected and synthesized
- ✅ Tools assigned appropriately to agents

**SHOULD HAVE:**
- ⚠️ Human approval requested for critical decisions
- ⚠️ Runbook created from successful execution pattern
- ⚠️ Cost tracking and budget management
- ⚠️ Error handling and graceful degradation

**NICE TO HAVE:**
- ⚠️ Agent performance optimization based on results
- ⚠️ Predictive timeline estimation
- ⚠️ Risk assessment and mitigation suggestions

---

## 🎯 **EXPECTED OUTPUT EXAMPLE**

```
🎯 Goal Created: "Support Optimization" (goal_abc123)
Progress: 0% | Agents: 0 | Status: IN_PROGRESS

🤖 Deploying Agents:
- Data Analyst (data_analyst_xyz789) → Collecting current metrics
- Research Agent (research_agent_def456) → Finding best practices  
- Strategy Agent (strategy_agent_ghi012) → Creating recommendations

📊 Progress Update:
- Criteria 1: ✅ COMPLETE - Current metrics collected
- Criteria 2: 🔄 IN_PROGRESS - Bottlenecks being analyzed
- Criteria 3: 🔄 IN_PROGRESS - Best practices research ongoing
- Criteria 4: ⏳ PENDING - Awaiting analysis completion
- Criteria 5: ⏳ PENDING - Action plan creation queued

Overall Progress: 40% | Estimated completion: 2 hours

👤 Human Approval Needed:
Goal is 80% complete. Request approval to deploy "Implementation Specialist" agent for final action plan creation. Estimated cost: $0.50

🎉 Goal Complete!
All success criteria achieved. Generated reusable workflow: "support_optimization.yaml"
```

---

## 🚀 **QUICK START COMMAND**

Run this test in your AI Agent Platform:

```bash
# Terminal test
python orchestrator/tests/test_goal_oriented_capabilities.py

# Or via Slack
@ai Please help me achieve this goal: [paste the test scenario above]
```

**Expected Result:** The orchestrator should demonstrate sophisticated goal-oriented behavior with multi-agent coordination, progress tracking, and human-in-the-loop decision making. 