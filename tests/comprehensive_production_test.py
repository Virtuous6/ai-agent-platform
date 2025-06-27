#!/usr/bin/env python3
"""
🎯 COMPREHENSIVE PRODUCTION TEST
AI Agent Platform - Full $2.50 Production Test

This script executes the complete complex business goal with:
- Real OpenAI API costs
- Multi-agent coordination
- Human approval workflows  
- Cost tracking and budgets
- Pattern recognition
- Error recovery
- Runbook creation

COST BUDGET: $2.50 maximum
EXPECTED DURATION: 10-15 minutes
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'production_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

class ProductionTestRunner:
    """Comprehensive production test runner with real-time monitoring."""
    
    def __init__(self):
        """Initialize test runner with environment validation."""
        self.validate_environment()
        self.start_time = datetime.now(timezone.utc)
        self.max_budget = float(os.getenv("MAX_GOAL_COST", "2.50"))
        self.cost_alert_threshold = float(os.getenv("COST_ALERT_THRESHOLD", "2.00"))
        self.auto_stop_at_budget = os.getenv("AUTO_STOP_AT_BUDGET", "true").lower() == "true"
        
        # Test execution tracking
        self.total_cost = 0.0
        self.tokens_used = 0
        self.agents_spawned = []
        self.approval_requests = []
        self.workflow_runs = []
        self.errors_encountered = []
        
        logger.info("🚀 Production Test Runner Initialized")
        logger.info(f"💰 Budget: ${self.max_budget} | Alert: ${self.cost_alert_threshold}")
    
    def validate_environment(self):
        """Validate all required environment variables and configurations."""
        required_vars = [
            "OPENAI_API_KEY",
            "SUPABASE_URL", 
            "SUPABASE_KEY"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        logger.info("✅ Environment validation passed")
    
    async def execute_comprehensive_test(self) -> Dict[str, Any]:
        """Execute the complete comprehensive business goal test."""
        
        try:
            logger.info("🎯 STARTING COMPREHENSIVE PRODUCTION TEST")
            logger.info("=" * 80)
            
            # Initialize core components
            orchestrator, db_logger = await self.initialize_components()
            
            # Create the complex business goal
            goal_id = await self.create_complex_goal(orchestrator)
            
            # Monitor execution in real-time
            result = await self.monitor_goal_execution(orchestrator, goal_id)
            
            # Generate comprehensive test report
            report = await self.generate_test_report(orchestrator, goal_id, db_logger, result)
            
            logger.info("🎯 COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
            return report
            
        except Exception as e:
            error_msg = f"Critical error in comprehensive test: {str(e)}"
            logger.error(error_msg)
            self.errors_encountered.append({
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "critical_failure"
            })
            
            # Generate error report
            return await self.generate_error_report(e)
    
    async def initialize_components(self):
        """Initialize all system components for testing."""
        logger.info("🔧 Initializing system components...")
        
        # Initialize database logger
        from database.supabase_logger import SupabaseLogger
        db_logger = SupabaseLogger()
        
        # Initialize goal-oriented orchestrator
        from orchestrator.goal_oriented_orchestrator import GoalOrientedOrchestrator
        orchestrator = GoalOrientedOrchestrator(db_logger=db_logger)
        
        # Test database connection
        health = db_logger.health_check()
        if not health.get("healthy", False):
            raise RuntimeError(f"Database health check failed: {health}")
        
        logger.info("✅ All components initialized successfully")
        return orchestrator, db_logger
    
    async def create_complex_goal(self, orchestrator) -> str:
        """Create the complex business turnaround goal."""
        logger.info("📋 Creating complex business goal...")
        
        # Complex goal description
        goal_description = """Analyze our struggling e-commerce platform and create a comprehensive 90-day turnaround strategy to increase revenue by 40% while reducing operational costs by 25%. This requires multi-domain expertise coordination."""
        
        # Comprehensive success criteria (tests all agent types)
        success_criteria = [
            "Conduct comprehensive technical audit: Analyze current tech stack, identify performance bottlenecks, security vulnerabilities, and scalability issues. Provide specific technical recommendations with implementation costs.",
            
            "Perform competitive market research: Research top 5 competitors, analyze industry trends, identify market opportunities, and benchmark pricing strategy. Include current market data and competitor analysis.",
            
            "Execute customer behavior analysis: Analyze customer patterns, identify churn reasons, determine most profitable segments, and recommend retention strategies with data-driven insights.",
            
            "Create detailed financial projections: Develop financial models, cost-benefit analysis of proposed changes, ROI calculations, and risk assessments for each recommendation.",
            
            "Design 90-day implementation roadmap: Create detailed action plan with specific milestones, resource requirements, timeline dependencies, and success metrics with accountability measures.",
            
            "Develop risk mitigation framework: Identify potential risks, create contingency plans, establish monitoring systems, and design rollback procedures for each initiative.",
            
            "Build success measurement system: Design KPI framework, establish baseline metrics, create reporting dashboards, and implement continuous monitoring systems."
        ]
        
        # Create goal with human oversight enabled
        goal_id = await orchestrator.execute_goal(
            goal_description=goal_description,
            success_criteria=success_criteria,
            created_by="production_test_user",
            priority=orchestrator.goal_manager.GoalPriority.HIGH,
            human_oversight=True
        )
        
        logger.info(f"🎯 Complex goal created: {goal_id}")
        logger.info(f"📊 Success criteria: {len(success_criteria)} comprehensive requirements")
        
        return goal_id
    
    async def monitor_goal_execution(self, orchestrator, goal_id: str) -> Dict[str, Any]:
        """Monitor goal execution with real-time cost tracking and approvals."""
        logger.info("👀 Starting real-time goal monitoring...")
        
        execution_start = datetime.now(timezone.utc)
        max_execution_time = 900  # 15 minutes maximum
        check_interval = 5  # Check every 5 seconds
        
        while True:
            try:
                # Check execution time limit
                elapsed = (datetime.now(timezone.utc) - execution_start).total_seconds()
                if elapsed > max_execution_time:
                    logger.warning("⏰ Maximum execution time reached, stopping test")
                    break
                
                # Get current goal status
                status = await orchestrator.get_goal_status(goal_id)
                
                # Check for pending approvals
                pending_approvals = status.get("approvals", {}).get("requests", [])
                if pending_approvals:
                    await self.handle_approval_requests(orchestrator, pending_approvals)
                
                # Monitor costs from database
                await self.check_cost_limits(orchestrator.db_logger)
                
                # Check if goal is completed
                if status.get("status") in ["completed", "failed"]:
                    logger.info(f"🎯 Goal execution {status.get('status')}: {goal_id}")
                    break
                
                # Log progress
                progress = status.get("progress", {})
                logger.info(f"📊 Progress: {progress.get('completion_percentage', 0):.1f}% | "
                          f"Agents: {status.get('agents', {}).get('count', 0)} | "
                          f"Approvals: {len(pending_approvals)}")
                
                # Wait before next check
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring goal execution: {e}")
                await asyncio.sleep(check_interval)
        
        # Get final status
        final_status = await orchestrator.get_goal_status(goal_id)
        
        logger.info("✅ Goal execution monitoring completed")
        return final_status
    
    async def handle_approval_requests(self, orchestrator, pending_approvals: List[Dict]):
        """Handle human approval requests with cost considerations."""
        for approval in pending_approvals:
            try:
                approval_id = approval["id"]
                action = approval["action"]
                cost = approval["cost"]
                urgency = approval["urgency"]
                
                logger.info(f"🤝 APPROVAL REQUEST: {approval_id}")
                logger.info(f"   Action: {action}")
                logger.info(f"   Cost: ${cost:.2f}")
                logger.info(f"   Urgency: {urgency}")
                
                # Check if this would exceed budget
                total_projected_cost = self.total_cost + cost
                
                if total_projected_cost > self.max_budget:
                    logger.warning(f"💰 REJECTING: Would exceed budget (${total_projected_cost:.2f} > ${self.max_budget})")
                    await orchestrator.human_approval.approve_request(
                        approval_id, 
                        approved=False, 
                        responder="production_test",
                        response=f"Rejected: Would exceed ${self.max_budget} budget"
                    )
                else:
                    logger.info(f"✅ APPROVING: Within budget (${total_projected_cost:.2f} <= ${self.max_budget})")
                    await orchestrator.human_approval.approve_request(
                        approval_id,
                        approved=True,
                        responder="production_test", 
                        response=f"Approved for production test - cost ${cost:.2f}"
                    )
                    
                    # Track the approval
                    self.approval_requests.append({
                        "approval_id": approval_id,
                        "action": action,
                        "cost": cost,
                        "approved": total_projected_cost <= self.max_budget,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
                # Small delay to simulate human decision time
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error handling approval {approval.get('id', 'unknown')}: {e}")
    
    async def check_cost_limits(self, db_logger):
        """Check current costs against limits."""
        try:
            # Get today's token summary for production test user
            today = datetime.now(timezone.utc).date().isoformat()
            summary = await db_logger.get_daily_token_summary("production_test_user", today)
            
            current_cost = summary.get("total_cost", 0.0)
            current_tokens = summary.get("total_tokens", 0)
            
            # Update tracking
            self.total_cost = current_cost
            self.tokens_used = current_tokens
            
            # Check alert threshold
            if current_cost >= self.cost_alert_threshold:
                logger.warning(f"💰 COST ALERT: ${current_cost:.2f} >= ${self.cost_alert_threshold}")
            
            # Check budget limit
            if current_cost >= self.max_budget and self.auto_stop_at_budget:
                logger.error(f"🛑 BUDGET EXCEEDED: ${current_cost:.2f} >= ${self.max_budget} - STOPPING TEST")
                raise RuntimeError(f"Budget limit exceeded: ${current_cost:.2f}")
            
        except Exception as e:
            logger.error(f"Error checking cost limits: {e}")
    
    async def generate_test_report(self, orchestrator, goal_id: str, db_logger, final_status: Dict) -> Dict[str, Any]:
        """Generate comprehensive test execution report."""
        logger.info("📊 Generating comprehensive test report...")
        
        execution_time = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        # Get workflow data
        workflows = await orchestrator.workflow_tracker.get_workflows_for_goal(goal_id)
        
        # Get token analytics
        token_analytics = await db_logger.get_token_usage_analytics("production_test_user", days=1)
        
        # Get spawned agents data
        try:
            spawned_agents = db_logger.client.table("spawned_agents").select("*").execute()
            agents_data = spawned_agents.data if spawned_agents.data else []
        except:
            agents_data = []
        
        report = {
            "test_metadata": {
                "test_name": "Comprehensive Business Turnaround Goal",
                "executed_at": self.start_time.isoformat(),
                "execution_time_seconds": execution_time,
                "budget_limit": self.max_budget,
                "cost_alert_threshold": self.cost_alert_threshold,
                "auto_stop_enabled": self.auto_stop_at_budget
            },
            
            "goal_execution": {
                "goal_id": goal_id,
                "final_status": final_status.get("status"),
                "completion_percentage": final_status.get("progress", {}).get("completion_percentage", 0),
                "criteria_completed": final_status.get("progress", {}).get("completed_criteria", 0),
                "total_criteria": final_status.get("progress", {}).get("total_criteria", 0),
                "agents_deployed": final_status.get("agents", {}).get("count", 0)
            },
            
            "cost_analysis": {
                "total_cost": self.total_cost,
                "budget_used_percentage": (self.total_cost / self.max_budget) * 100,
                "total_tokens": self.tokens_used,
                "cost_per_token": self.total_cost / max(self.tokens_used, 1),
                "remained_under_budget": self.total_cost <= self.max_budget,
                "token_breakdown": token_analytics
            },
            
            "agent_coordination": {
                "approval_requests_handled": len(self.approval_requests),
                "agents_spawned": len(agents_data),
                "approval_details": self.approval_requests,
                "spawned_agent_details": agents_data
            },
            
            "workflow_tracking": {
                "workflows_executed": len(workflows),
                "workflow_details": workflows
            },
            
            "system_performance": {
                "errors_encountered": len(self.errors_encountered),
                "error_details": self.errors_encountered,
                "test_stability": len(self.errors_encountered) == 0
            },
            
            "test_results": {
                "overall_success": (
                    final_status.get("status") == "completed" and
                    self.total_cost <= self.max_budget and
                    len(self.errors_encountered) == 0
                ),
                "goal_achievement": final_status.get("status") == "completed",
                "budget_compliance": self.total_cost <= self.max_budget,
                "error_free_execution": len(self.errors_encountered) == 0,
                "approval_system_tested": len(self.approval_requests) > 0,
                "multi_agent_coordination": len(agents_data) > 3
            },
            
            "lessons_learned": {
                "cost_efficiency": f"${self.total_cost:.2f} for {len(agents_data)} agents and {len(workflows)} workflows",
                "execution_efficiency": f"{execution_time:.1f} seconds for complex multi-domain analysis",
                "system_scalability": f"Handled {len(self.approval_requests)} approval requests seamlessly",
                "error_resilience": f"System stability: {len(self.errors_encountered)} errors in complex execution"
            }
        }
        
        # Save report to file
        report_filename = f"production_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Test report saved: {report_filename}")
        
        # Print summary
        self.print_test_summary(report)
        
        return report
    
    async def generate_error_report(self, error: Exception) -> Dict[str, Any]:
        """Generate error report if test fails."""
        execution_time = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        error_report = {
            "test_metadata": {
                "test_name": "Comprehensive Business Turnaround Goal",
                "executed_at": self.start_time.isoformat(),
                "execution_time_seconds": execution_time,
                "test_result": "FAILED"
            },
            "error_details": {
                "primary_error": str(error),
                "error_type": type(error).__name__,
                "all_errors": self.errors_encountered
            },
            "partial_results": {
                "total_cost": self.total_cost,
                "tokens_used": self.tokens_used,
                "approval_requests": len(self.approval_requests),
                "agents_spawned": len(self.agents_spawned)
            }
        }
        
        # Save error report
        error_filename = f"production_test_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_filename, 'w') as f:
            json.dump(error_report, f, indent=2, default=str)
        
        logger.error(f"❌ Error report saved: {error_filename}")
        return error_report
    
    def print_test_summary(self, report: Dict[str, Any]):
        """Print a comprehensive test summary."""
        logger.info("=" * 80)
        logger.info("🎯 COMPREHENSIVE PRODUCTION TEST SUMMARY")
        logger.info("=" * 80)
        
        # Test Results
        results = report["test_results"]
        logger.info(f"✅ Overall Success: {results['overall_success']}")
        logger.info(f"🎯 Goal Achievement: {results['goal_achievement']}")
        logger.info(f"💰 Budget Compliance: {results['budget_compliance']}")
        logger.info(f"🛡️ Error-Free Execution: {results['error_free_execution']}")
        
        # Performance Metrics  
        cost = report["cost_analysis"]
        logger.info(f"\n💰 COST ANALYSIS:")
        logger.info(f"   Total Cost: ${cost['total_cost']:.2f} / ${report['test_metadata']['budget_limit']}")
        logger.info(f"   Budget Used: {cost['budget_used_percentage']:.1f}%")
        logger.info(f"   Total Tokens: {cost['total_tokens']:,}")
        logger.info(f"   Cost per Token: ${cost['cost_per_token']:.6f}")
        
        # Agent Coordination
        agents = report["agent_coordination"]
        logger.info(f"\n🤖 AGENT COORDINATION:")
        logger.info(f"   Agents Spawned: {agents['agents_spawned']}")
        logger.info(f"   Approval Requests: {agents['approval_requests_handled']}")
        
        # Goal Execution
        goal = report["goal_execution"]
        logger.info(f"\n🎯 GOAL EXECUTION:")
        logger.info(f"   Completion: {goal['completion_percentage']:.1f}%")
        logger.info(f"   Criteria Met: {goal['criteria_completed']}/{goal['total_criteria']}")
        logger.info(f"   Agents Deployed: {goal['agents_deployed']}")
        
        # System Performance
        performance = report["system_performance"]
        logger.info(f"\n🛡️ SYSTEM PERFORMANCE:")
        logger.info(f"   Errors: {performance['errors_encountered']}")
        logger.info(f"   Stability: {performance['test_stability']}")
        
        # Execution Time
        execution_time = report["test_metadata"]["execution_time_seconds"]
        logger.info(f"\n⏱️ EXECUTION TIME: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")
        
        logger.info("=" * 80)
        
        # Success celebration or failure note
        if results["overall_success"]:
            logger.info("🎉 COMPREHENSIVE TEST PASSED - ALL SYSTEMS OPERATIONAL!")
        else:
            logger.info("⚠️ Test completed with issues - check detailed report")
        
        logger.info("=" * 80)

async def main():
    """Main entry point for comprehensive production test."""
    print("🚀 STARTING COMPREHENSIVE AI AGENT PLATFORM PRODUCTION TEST")
    print("💰 Budget: $2.50 | Expected Duration: 10-15 minutes")
    print("🎯 Testing: Goal Orchestration, Multi-Agent Coordination, Cost Tracking, Approvals")
    print("=" * 80)
    
    try:
        # Initialize and run test
        test_runner = ProductionTestRunner()
        result = await test_runner.execute_comprehensive_test()
        
        # Final status
        if result.get("test_results", {}).get("overall_success", False):
            print("\n🎉 COMPREHENSIVE PRODUCTION TEST COMPLETED SUCCESSFULLY!")
            print("✅ All systems validated and operational")
            return 0
        else:
            print("\n⚠️ Production test completed with issues")
            print("📊 Check detailed report for analysis")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    
    # Set event loop policy for Windows compatibility
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run the comprehensive test
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 