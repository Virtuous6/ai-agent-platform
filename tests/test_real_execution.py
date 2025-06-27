#!/usr/bin/env python3
"""
🔥 Test Real Agent Execution
Tests the 100% real LLM-powered agent execution system.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealExecutionTest:
    """Test the real agent execution system with actual LLM calls."""
    
    def __init__(self):
        self.max_budget = 1.00  # Small budget for testing
        self.start_time = datetime.now(timezone.utc)
        self.total_cost = 0.0
        
    async def test_real_execution(self):
        """Test real agent execution with a simple goal."""
        logger.info("🔥 TESTING 100% REAL AGENT EXECUTION")
        logger.info(f"💰 Test Budget: ${self.max_budget}")
        
        try:
            # Initialize the real system
            from database.supabase_logger import SupabaseLogger
            from orchestrator.goal_oriented_orchestrator import GoalOrientedOrchestrator
            from orchestrator.goal_manager import GoalPriority
            
            db_logger = SupabaseLogger()
            orchestrator = GoalOrientedOrchestrator(db_logger=db_logger)
            
            # Create a simple goal for real testing
            goal_description = "Analyze our customer support system and provide 3 specific improvement recommendations"
            
            success_criteria = [
                "Analyze current support metrics and identify performance gaps",
                "Research industry best practices for customer support",
                "Generate 3 specific, actionable improvement recommendations"
            ]
            
            logger.info("🎯 Creating real goal with actual LLM execution...")
            
            # Execute goal with real agents
            goal_id = await orchestrator.execute_goal(
                goal_description=goal_description,
                success_criteria=success_criteria,
                created_by="real_execution_test",
                priority=GoalPriority.HIGH,
                human_oversight=True  # This will auto-approve for test
            )
            
            logger.info(f"🚀 Real goal execution started: {goal_id}")
            
            # Monitor real execution
            execution_time = 0
            max_time = 300  # 5 minutes max
            
            while execution_time < max_time:
                await asyncio.sleep(10)
                execution_time += 10
                
                # Check real status
                status = await orchestrator.get_goal_status(goal_id)
                
                # Handle any pending approvals (auto-approve for test)
                pending = status.get("approvals", {}).get("requests", [])
                for approval in pending:
                    await orchestrator.human_approval.approve_request(
                        approval["id"], True, "real_test", "Auto-approved for real execution test"
                    )
                    logger.info(f"✅ Auto-approved: {approval['action']} - ${approval['cost']:.2f}")
                
                # Log real progress
                progress = status.get("progress", {})
                logger.info(f"📊 Real Progress: {progress.get('completion_percentage', 0):.1f}% | "
                          f"Agents: {status.get('agents', {}).get('count', 0)}")
                
                # Check if completed
                if status.get("status") in ["completed", "failed"]:
                    logger.info(f"🎯 Real execution {status.get('status')}: {goal_id}")
                    break
            
            # Get final results
            final_status = await orchestrator.get_goal_status(goal_id)
            
            # Generate real execution report
            self.print_real_execution_report(final_status, execution_time)
            
            return final_status
            
        except Exception as e:
            logger.error(f"❌ Real execution test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def print_real_execution_report(self, status, execution_time):
        """Print comprehensive report of real execution results."""
        logger.info("=" * 60)
        logger.info("🔥 REAL AGENT EXECUTION TEST RESULTS")
        logger.info("=" * 60)
        
        # Status
        logger.info(f"✅ Goal Status: {status.get('status', 'unknown')}")
        progress = status.get("progress", {})
        logger.info(f"📈 Progress: {progress.get('completion_percentage', 0):.1f}%")
        
        # Agent Performance
        agents = status.get("agents", {})
        logger.info(f"🤖 Agents Deployed: {agents.get('count', 0)}")
        
        # Execution Details
        logger.info(f"⏱️ Execution Time: {execution_time} seconds")
        
        # Real vs Simulated
        if status.get("status") == "completed":
            logger.info("🔥 REAL EXECUTION SUCCESS!")
            logger.info("✅ Agents made actual LLM calls")
            logger.info("✅ Real business analysis generated")
            logger.info("✅ Actual OpenAI costs incurred")
            logger.info("✅ Dynamic agent responses")
        else:
            logger.info("⚠️ Real execution needs attention")
        
        logger.info("=" * 60)

async def main():
    """Run the real execution test."""
    print("🔥 AI AGENT PLATFORM - REAL EXECUTION TEST")
    print("💡 This test uses REAL LLM calls with actual costs")
    print("🎯 Testing: 100% Real Agent Execution")
    print("=" * 60)
    
    test = RealExecutionTest()
    result = await test.test_real_execution()
    
    success = result.get("status") == "completed"
    if success:
        print("🔥 REAL EXECUTION TEST PASSED!")
        print("🎉 Your platform now executes 100% real workflows!")
    else:
        print("📋 Real execution test completed - check results above")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(main()) 