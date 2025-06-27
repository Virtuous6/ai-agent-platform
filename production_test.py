#!/usr/bin/env python3
"""
🎯 AI Agent Platform - $2.50 Production Test
Tests the complete system with real OpenAI costs.
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

class ProductionTest:
    """Execute comprehensive production test with real costs."""
    
    def __init__(self):
        self.max_budget = float(os.getenv("MAX_GOAL_COST", "2.50"))
        self.start_time = datetime.now(timezone.utc)
        self.total_cost = 0.0
        self.approval_requests = []
        
        # Validate environment variables
        self.validate_environment()
        
    def validate_environment(self):
        """Validate required environment variables are loaded."""
        required_vars = {
            "OPENAI_API_KEY": "OpenAI API key for LLM calls",
            "SUPABASE_URL": "Supabase database URL", 
            "SUPABASE_KEY": "Supabase API key"
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                missing_vars.append(f"{var} ({description})")
            else:
                # Mask sensitive values in logs
                masked_value = value[:8] + "..." if len(value) > 8 else "***"
                logger.info(f"✅ {var}: {masked_value}")
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        logger.info("✅ All environment variables loaded from .env file")
        
    async def run_comprehensive_test(self):
        """Execute the complete comprehensive business test."""
        logger.info("🎯 STARTING $2.50 PRODUCTION TEST")
        logger.info(f"💰 Budget: ${self.max_budget}")
        logger.info("🔑 Loading credentials from .env file")
        
        try:
            # Initialize components
            orchestrator, db_logger = await self.setup_system()
            
            # Create complex goal
            goal_id = await self.create_business_goal(orchestrator)
            
            # Monitor execution
            result = await self.monitor_execution(orchestrator, goal_id)
            
            # Generate report
            report = await self.generate_report(orchestrator, goal_id, result)
            
            logger.info("🎉 PRODUCTION TEST COMPLETED!")
            return report
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def setup_system(self):
        """Initialize the system components."""
        logger.info("🔧 Initializing system...")
        
        from database.supabase_logger import SupabaseLogger
        from orchestrator.goal_oriented_orchestrator import GoalOrientedOrchestrator
        
        db_logger = SupabaseLogger()
        orchestrator = GoalOrientedOrchestrator(db_logger=db_logger)
        
        # Test database connection
        health = db_logger.health_check()
        if health.get("healthy"):
            logger.info("✅ Supabase connection verified")
        else:
            logger.warning(f"⚠️ Database health check: {health}")
        
        return orchestrator, db_logger
    
    async def create_business_goal(self, orchestrator):
        """Create the complex business turnaround goal."""
        logger.info("📋 Creating complex business goal...")
        
        goal_description = """Analyze our struggling e-commerce platform and create a comprehensive 90-day turnaround strategy to increase revenue by 40% while reducing operational costs by 25%."""
        
        success_criteria = [
            "Technical Analysis: Audit tech stack, identify bottlenecks, security issues, scalability problems. Provide specific recommendations with costs.",
            "Market Research: Research top 5 competitors, analyze trends, identify opportunities, benchmark pricing with current data.",
            "Customer Analytics: Analyze behavior patterns, identify churn reasons, determine profitable segments, recommend retention strategies.",
            "Financial Assessment: Create projections, cost-benefit analysis, ROI calculations, risk assessments for each recommendation.",
            "Implementation Planning: Develop 90-day roadmap with milestones, resources, timeline, success metrics."
        ]
        
        # Import GoalPriority properly
        from orchestrator.goal_manager import GoalPriority
        
        goal_id = await orchestrator.execute_goal(
            goal_description=goal_description,
            success_criteria=success_criteria,
            created_by="production_test",
            priority=GoalPriority.HIGH,
            human_oversight=True
        )
        
        logger.info(f"🎯 Goal created: {goal_id}")
        logger.info(f"📊 Success criteria: {len(success_criteria)} comprehensive requirements")
        return goal_id
    
    async def monitor_execution(self, orchestrator, goal_id):
        """Monitor goal execution with real-time approvals."""
        logger.info("👀 Monitoring execution...")
        
        for i in range(60):  # Monitor for up to 10 minutes
            try:
                status = await orchestrator.get_goal_status(goal_id)
                
                # Handle approvals
                pending = status.get("approvals", {}).get("requests", [])
                for approval in pending:
                    await self.handle_approval(orchestrator, approval)
                
                # Check completion
                if status.get("status") in ["completed", "failed"]:
                    logger.info(f"🎯 Goal {status.get('status')}")
                    break
                
                # Log progress
                progress = status.get("progress", {})
                agents = status.get("agents", {})
                logger.info(f"📊 Progress: {progress.get('completion_percentage', 0):.1f}% | "
                          f"Agents: {agents.get('count', 0)} | "
                          f"Approvals: {len(pending)}")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                break
        
        return await orchestrator.get_goal_status(goal_id)
    
    async def handle_approval(self, orchestrator, approval):
        """Handle approval requests automatically."""
        approval_id = approval["id"]
        cost = approval["cost"]
        action = approval["action"]
        urgency = approval.get("urgency", "medium")
        
        logger.info(f"🤝 APPROVAL REQUEST: {approval_id}")
        logger.info(f"   Action: {action}")
        logger.info(f"   Cost: ${cost:.2f}")
        logger.info(f"   Urgency: {urgency}")
        
        # Check budget before approving
        projected_cost = self.total_cost + cost
        
        if projected_cost <= self.max_budget:
            await orchestrator.human_approval.approve_request(
                approval_id, True, "production_test", f"Approved - ${cost:.2f} within budget"
            )
            self.total_cost += cost
            logger.info(f"✅ APPROVED - Total cost now: ${self.total_cost:.2f}")
        else:
            await orchestrator.human_approval.approve_request(
                approval_id, False, "production_test", f"Budget exceeded - would be ${projected_cost:.2f}"
            )
            logger.info(f"❌ REJECTED - Would exceed ${self.max_budget} budget")
        
        self.approval_requests.append({
            "id": approval_id,
            "cost": cost,
            "action": action,
            "approved": projected_cost <= self.max_budget,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def generate_report(self, orchestrator, goal_id, final_status):
        """Generate test execution report."""
        execution_time = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        report = {
            "test_results": {
                "goal_id": goal_id,
                "status": final_status.get("status"),
                "completion": final_status.get("progress", {}).get("completion_percentage", 0),
                "execution_time": execution_time,
                "total_cost": self.total_cost,
                "budget_used": (self.total_cost / self.max_budget) * 100,
                "approvals_handled": len(self.approval_requests),
                "success": final_status.get("status") == "completed"
            },
            "system_performance": {
                "agents_deployed": final_status.get("agents", {}).get("count", 0),
                "approval_system": len(self.approval_requests) > 0,
                "cost_tracking": self.total_cost <= self.max_budget,
                "multi_agent_coordination": final_status.get("agents", {}).get("count", 0) > 2
            }
        }
        
        # Print comprehensive summary
        logger.info("=" * 60)
        logger.info("📊 COMPREHENSIVE PRODUCTION TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Goal Status: {report['test_results']['status']}")
        logger.info(f"📈 Completion: {report['test_results']['completion']:.1f}%")
        logger.info(f"💰 Total Cost: ${report['test_results']['total_cost']:.2f}/${self.max_budget}")
        logger.info(f"📊 Budget Used: {report['test_results']['budget_used']:.1f}%")
        logger.info(f"🤖 Agents Deployed: {report['system_performance']['agents_deployed']}")
        logger.info(f"🤝 Approvals Handled: {len(self.approval_requests)}")
        logger.info(f"⏱️ Execution Time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")
        
        # Success determination
        test_passed = (
            report['test_results']['success'] and
            report['test_results']['total_cost'] <= self.max_budget and
            report['system_performance']['agents_deployed'] >= 3 and
            len(self.approval_requests) >= 2
        )
        
        if test_passed:
            logger.info("🎉 PRODUCTION TEST PASSED!")
            logger.info("✅ All systems validated and operational")
        else:
            logger.info("⚠️ Production test completed with issues")
            
        logger.info("=" * 60)
        
        return report

async def main():
    """Main test entry point."""
    print("🚀 AI AGENT PLATFORM - COMPREHENSIVE PRODUCTION TEST")
    print("💰 Budget: $2.50 | Expected Duration: 8-12 minutes")
    print("🎯 Testing: Complete System Integration with Real Costs")
    print("🔑 Loading credentials from .env file")
    print("=" * 60)
    
    test = ProductionTest()
    result = await test.run_comprehensive_test()
    
    success = result.get("test_results", {}).get("success", False)
    if success:
        print("🎉 COMPREHENSIVE TEST PASSED!")
        return 0
    else:
        print("⚠️ Test completed with issues - check logs above")
        return 1

if __name__ == "__main__":
    result = asyncio.run(main()) 