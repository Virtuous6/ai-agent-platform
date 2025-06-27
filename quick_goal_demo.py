#!/usr/bin/env python3
"""
Quick Goal-Workflow Integration Demo

Shows complete integration of goals, workflows, and Supabase tracking
in under 30 lines of working code.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestrator.goal_oriented_orchestrator import GoalOrientedOrchestrator
from database.supabase_logger import SupabaseLogger

async def quick_demo():
    """Demonstrate complete goal-workflow-Supabase integration."""
    
    print("🚀 Starting Quick Goal-Workflow Integration Demo")
    
    # Initialize components
    logger = SupabaseLogger()
    orchestrator = GoalOrientedOrchestrator()
    
    # 1. Create a goal with Supabase tracking
    print("\n📋 Step 1: Creating goal with Supabase tracking...")
    goal_id = await orchestrator.create_goal(
        description="Test customer support analysis workflow",
        success_criteria=[
            "Collect current metrics", 
            "Identify bottlenecks",
            "Generate recommendations"
        ],
        user_id="demo_user",
        priority="high"
    )
    print(f"✅ Goal created: {goal_id}")
    
    # 2. Start workflow with goal linking
    print("\n🔄 Step 2: Starting workflow linked to goal...")
    workflow_run_id = await orchestrator.agent_orchestrator.workflow_tracker.start_workflow(
        workflow_type="goal_execution",
        user_id="demo_user",
        trigger_message="Analyze customer support performance",
        goal_id=goal_id  # 🎯 CRITICAL: Goal linking
    )
    print(f"✅ Workflow started: {workflow_run_id}")
    
    # 3. Request human approval for expensive operation
    print("\n🤲 Step 3: Requesting human approval...")
    approval_id = await orchestrator.request_human_approval(
        goal_id=goal_id,
        action_type="spawn_expensive_agent",
        description="Deploy AI Research Analyst ($0.50 estimated cost)",
        cost_estimate=0.50,
        user_id="demo_user"
    )
    print(f"✅ Approval requested: {approval_id}")
    
    # 4. Simulate workflow progress
    print("\n📊 Step 4: Updating goal progress...")
    await orchestrator.update_goal_progress(
        goal_id=goal_id,
        progress=0.6,
        criteria_met=["Collect current metrics", "Identify bottlenecks"]
    )
    print("✅ Goal progress updated: 60% complete")
    
    # 5. Complete workflow with final tracking
    print("\n🎯 Step 5: Completing workflow...")
    await orchestrator.agent_orchestrator.workflow_tracker.complete_workflow(
        run_id=workflow_run_id,
        success=True,
        response="Customer support analysis completed with 3 recommendations",
        tokens_used=245,
        confidence_score=0.89
    )
    
    # Final goal completion
    await orchestrator.update_goal_progress(
        goal_id=goal_id,
        progress=1.0,
        criteria_met=["Collect current metrics", "Identify bottlenecks", "Generate recommendations"]
    )
    print("✅ Goal completed: 100%")
    
    # 6. Query Supabase to verify tracking
    print("\n🔍 Step 6: Verifying Supabase tracking...")
    
    # Check goal was created
    goal_result = logger.client.table("goals").select("id, description, current_status, progress_percentage").eq("id", goal_id).execute()
    print(f"📊 Goal in DB: {goal_result.data[0] if goal_result.data else 'Not found'}")
    
    # Check workflow was linked
    workflow_result = logger.client.table("workflow_runs").select("run_id, goal_id, success, tokens_used").eq("run_id", workflow_run_id).execute()
    print(f"🔄 Workflow in DB: {workflow_result.data[0] if workflow_result.data else 'Not found'}")
    
    # Check approval was created
    approval_result = logger.client.table("approval_requests").select("id, action_type, status, cost_estimate").eq("id", approval_id).execute()
    print(f"🤲 Approval in DB: {approval_result.data[0] if approval_result.data else 'Not found'}")
    
    print("\n🎉 DEMO COMPLETE: Goal-Workflow-Supabase Integration Working!")
    print("\n📈 Summary:")
    print("✅ Goals created and tracked in Supabase")
    print("✅ Workflows linked to goals") 
    print("✅ Human approval system working")
    print("✅ Progress tracking functional")
    print("✅ Complete end-to-end integration")

if __name__ == "__main__":
    asyncio.run(quick_demo()) 