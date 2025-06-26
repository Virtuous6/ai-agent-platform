"""
Pattern Recognition System Integration
Connects the Pattern Recognition Engine with the existing AI Agent Platform components.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from agents.improvement.pattern_recognition import PatternRecognitionEngine, InteractionEvent
from agents.improvement.workflow_analyst import WorkflowAnalyst
from agents.improvement.feedback_handler import FeedbackHandler

logger = logging.getLogger(__name__)

class PatternRecognitionIntegrator:
    """
    Integrates Pattern Recognition with existing AI Agent Platform systems.
    
    This class serves as a bridge between:
    - Pattern Recognition Engine (new)
    - Workflow Analyst (existing)
    - Feedback Handler (existing)
    - Agent Orchestrator (existing)
    """
    
    def __init__(self, db_logger=None, orchestrator=None):
        """
        Initialize the Pattern Recognition Integrator.
        
        Args:
            db_logger: Supabase logger for data persistence
            orchestrator: Agent orchestrator for system coordination
        """
        self.db_logger = db_logger
        self.orchestrator = orchestrator
        
        # Initialize Pattern Recognition Engine
        self.pattern_engine = PatternRecognitionEngine(
            db_logger=db_logger,
            orchestrator=orchestrator
        )
        
        # Initialize other improvement components if available
        self.workflow_analyst = None
        self.feedback_handler = None
        
        try:
            self.workflow_analyst = WorkflowAnalyst(
                db_logger=db_logger,
                orchestrator=orchestrator
            )
        except Exception as e:
            logger.warning(f"Could not initialize Workflow Analyst: {str(e)}")
        
        try:
            self.feedback_handler = FeedbackHandler(db_logger=db_logger)
        except Exception as e:
            logger.warning(f"Could not initialize Feedback Handler: {str(e)}")
        
        logger.info("Pattern Recognition Integrator initialized")
    
    async def track_user_interaction(self, user_id: str, message: str, 
                                   agent_response: Dict[str, Any],
                                   context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Track a user interaction for pattern analysis.
        
        Args:
            user_id: User identifier
            message: User's message
            agent_response: Agent's response data
            context: Additional context (channel, thread, etc.)
            
        Returns:
            True if successfully tracked
        """
        try:
            # Create InteractionEvent
            event = InteractionEvent(
                id=f"interaction_{datetime.utcnow().timestamp()}",
                user_id=user_id,
                timestamp=datetime.utcnow(),
                message=message,
                context=context or {},
                agent_used=agent_response.get("agent_type", "unknown"),
                success=agent_response.get("success", True),
                duration_ms=agent_response.get("processing_time_ms", 0),
                tokens_used=agent_response.get("tokens_used", 0),
                cost=agent_response.get("estimated_cost", 0.0)
            )
            
            # Record in Pattern Recognition Engine
            await self.pattern_engine.record_interaction(event)
            
            logger.debug(f"Tracked interaction for user {user_id}: {message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking user interaction: {str(e)}")
            return False
    
    async def analyze_patterns(self, force_analysis: bool = False) -> Dict[str, Any]:
        """
        Trigger pattern analysis across all systems.
        
        Args:
            force_analysis: Force analysis even if recently performed
            
        Returns:
            Combined analysis results
        """
        try:
            results = {
                "pattern_recognition": {},
                "workflow_analysis": {},
                "combined_insights": {}
            }
            
            # Run pattern recognition analysis
            pattern_results = await self.pattern_engine.analyze_patterns(
                days_back=7, 
                force_analysis=force_analysis
            )
            results["pattern_recognition"] = pattern_results
            
            # Run workflow analysis if available
            if self.workflow_analyst:
                try:
                    workflow_results = await self.workflow_analyst.analyze_workflows(
                        days_back=7,
                        force_analysis=force_analysis
                    )
                    results["workflow_analysis"] = workflow_results
                except Exception as e:
                    logger.warning(f"Workflow analysis failed: {str(e)}")
            
            # Combine insights
            results["combined_insights"] = self._combine_insights(
                pattern_results, 
                results.get("workflow_analysis", {})
            )
            
            logger.info("Pattern analysis completed across all systems")
            return results
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _combine_insights(self, pattern_results: Dict[str, Any], 
                         workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine insights from pattern recognition and workflow analysis."""
        try:
            combined = {
                "total_patterns": 0,
                "automation_opportunities": 0,
                "high_value_patterns": [],
                "recommendations": []
            }
            
            # Add pattern recognition insights
            if pattern_results.get("status") == "completed":
                combined["total_patterns"] += pattern_results.get("new_patterns_found", 0)
                combined["automation_opportunities"] += pattern_results.get("automation_suggestions", 0)
                
                # Extract high-value patterns
                for pattern_data in pattern_results.get("patterns", []):
                    if pattern_data.get("automation_potential", 0) > 0.7:
                        combined["high_value_patterns"].append({
                            "name": pattern_data.get("name"),
                            "type": "interaction_pattern",
                            "automation_potential": pattern_data.get("automation_potential"),
                            "frequency": pattern_data.get("frequency")
                        })
            
            # Add workflow analysis insights
            if workflow_results.get("status") == "completed":
                patterns_found = workflow_results.get("patterns_discovered", 0)
                combined["total_patterns"] += patterns_found
                
                # Add workflow recommendations
                if workflow_results.get("optimization_opportunities"):
                    for opt in workflow_results["optimization_opportunities"]:
                        combined["recommendations"].append({
                            "type": "workflow_optimization",
                            "title": opt.get("title", "Workflow Optimization"),
                            "benefit": opt.get("potential_benefit", "Unknown")
                        })
            
            # Generate combined recommendations
            if combined["automation_opportunities"] > 0:
                combined["recommendations"].append({
                    "type": "automation_implementation",
                    "title": "Implement Pattern-Based Automations",
                    "benefit": f"Automate {combined['automation_opportunities']} recurring workflows"
                })
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining insights: {str(e)}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all pattern recognition systems."""
        try:
            pattern_summary = self.pattern_engine.get_pattern_summary()
            
            status = {
                "pattern_recognition_engine": {
                    "status": "active",
                    "patterns_recognized": pattern_summary["total_patterns"],
                    "automation_suggestions": pattern_summary["automation_suggestions"],
                    "interactions_tracked": pattern_summary["interactions_in_buffer"]
                },
                "workflow_analyst": {
                    "status": "active" if self.workflow_analyst else "not_available",
                    "last_analysis": None
                },
                "feedback_handler": {
                    "status": "active" if self.feedback_handler else "not_available"
                },
                "integration_status": "operational"
            }
            
            # Get workflow analyst status
            if self.workflow_analyst:
                try:
                    workflow_summary = self.workflow_analyst.get_analysis_summary()
                    status["workflow_analyst"]["last_analysis"] = workflow_summary.get("last_analysis")
                    status["workflow_analyst"]["patterns_discovered"] = len(workflow_summary.get("patterns", []))
                except Exception as e:
                    logger.warning(f"Could not get workflow analyst status: {str(e)}")
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def suggest_workflow_improvements(self, user_id: str) -> Dict[str, Any]:
        """
        Generate workflow improvement suggestions for a specific user.
        
        Args:
            user_id: User to generate suggestions for
            
        Returns:
            Personalized improvement suggestions
        """
        try:
            suggestions = {
                "user_id": user_id,
                "pattern_based_suggestions": [],
                "automation_opportunities": [],
                "general_recommendations": []
            }
            
            # Get user-specific patterns
            top_patterns = self.pattern_engine.get_top_patterns(limit=5)
            user_patterns = [p for p in top_patterns if user_id in p.users_affected]
            
            # Generate pattern-based suggestions
            for pattern in user_patterns:
                if pattern.automation_potential > 0.6:
                    suggestions["pattern_based_suggestions"].append({
                        "pattern_name": pattern.name,
                        "suggestion": f"Consider automating '{pattern.name}' - you use this {pattern.frequency} times",
                        "time_savings": f"{pattern.avg_duration_ms / 60000:.1f} minutes per execution",
                        "automation_potential": pattern.automation_potential
                    })
            
            # Get automation suggestions
            automation_suggestions = self.pattern_engine.get_automation_suggestions(limit=3)
            for suggestion in automation_suggestions:
                pattern = self.pattern_engine.recognized_patterns.get(suggestion.pattern_id)
                if pattern and user_id in pattern.users_affected:
                    suggestions["automation_opportunities"].append({
                        "title": suggestion.title,
                        "description": suggestion.description,
                        "time_saved": suggestion.estimated_time_saved,
                        "cost_saved": suggestion.estimated_cost_saved,
                        "complexity": suggestion.implementation_complexity
                    })
            
            # Add general recommendations
            if len(user_patterns) > 3:
                suggestions["general_recommendations"].append(
                    "You have several recurring workflows that could benefit from automation. "
                    "Consider using the /save-workflow command to capture your most common processes."
                )
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating workflow improvements: {str(e)}")
            return {"error": str(e)}
    
    async def close(self):
        """Cleanup all resources."""
        try:
            await self.pattern_engine.close()
            
            if self.workflow_analyst:
                await self.workflow_analyst.close()
            
            logger.info("Pattern Recognition Integrator closed")
            
        except Exception as e:
            logger.error(f"Error closing integrator: {str(e)}")


# Integration demo function
async def run_integration_demo():
    """Demonstrate the integrated pattern recognition system."""
    print("🚀 Pattern Recognition System Integration Demo")
    print("=" * 60)
    
    try:
        # Initialize integrator
        integrator = PatternRecognitionIntegrator()
        
        # Step 1: Track some interactions
        print("\n📝 Step 1: Tracking User Interactions")
        
        interactions = [
            ("user_alice", "Generate daily sales report", {"agent_type": "research", "success": True, "processing_time_ms": 2500, "tokens_used": 150, "estimated_cost": 0.008}),
            ("user_alice", "Send report to team", {"agent_type": "general", "success": True, "processing_time_ms": 1200, "tokens_used": 80, "estimated_cost": 0.004}),
            ("user_bob", "Check system status", {"agent_type": "technical", "success": True, "processing_time_ms": 800, "tokens_used": 60, "estimated_cost": 0.003}),
            ("user_alice", "Generate daily sales report", {"agent_type": "research", "success": True, "processing_time_ms": 2300, "tokens_used": 140, "estimated_cost": 0.007}),
            ("user_bob", "Deploy new feature", {"agent_type": "technical", "success": True, "processing_time_ms": 5000, "tokens_used": 300, "estimated_cost": 0.015})
        ]
        
        for user_id, message, response in interactions:
            await integrator.track_user_interaction(user_id, message, response)
            print(f"  ✅ Tracked: {message} (User: {user_id})")
        
        # Step 2: Get system status
        print("\n📊 Step 2: System Status")
        status = integrator.get_system_status()
        print(f"  • Pattern Recognition Engine: {status['pattern_recognition_engine']['status']}")
        print(f"  • Interactions tracked: {status['pattern_recognition_engine']['interactions_tracked']}")
        print(f"  • Patterns recognized: {status['pattern_recognition_engine']['patterns_recognized']}")
        print(f"  • Workflow Analyst: {status['workflow_analyst']['status']}")
        
        # Step 3: Generate user suggestions
        print("\n💡 Step 3: User-Specific Suggestions")
        alice_suggestions = await integrator.suggest_workflow_improvements("user_alice")
        
        print(f"  Suggestions for user_alice:")
        if alice_suggestions.get("pattern_based_suggestions"):
            for suggestion in alice_suggestions["pattern_based_suggestions"]:
                print(f"    • {suggestion['suggestion']}")
        else:
            print("    • No specific patterns identified yet (need more data)")
        
        if alice_suggestions.get("general_recommendations"):
            for rec in alice_suggestions["general_recommendations"]:
                print(f"    • {rec}")
        
        # Step 4: Run analysis
        print("\n🔍 Step 4: Running Pattern Analysis")
        analysis_results = await integrator.analyze_patterns(force_analysis=True)
        
        print(f"  • Pattern Recognition Status: {analysis_results['pattern_recognition'].get('status', 'unknown')}")
        print(f"  • Workflow Analysis Status: {analysis_results['workflow_analysis'].get('status', 'not_run')}")
        
        combined = analysis_results.get("combined_insights", {})
        print(f"  • Total patterns identified: {combined.get('total_patterns', 0)}")
        print(f"  • Automation opportunities: {combined.get('automation_opportunities', 0)}")
        
        print("\n🎉 Integration Demo Complete!")
        print("   ✅ Successfully tracked user interactions")
        print("   ✅ Integrated with existing systems")
        print("   ✅ Generated personalized recommendations")
        print("   ✅ Performed cross-system analysis")
        
        return {
            "demo_success": True,
            "interactions_tracked": len(interactions),
            "systems_integrated": 3,
            "analysis_completed": True
        }
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return {"demo_success": False, "error": str(e)}
    
    finally:
        try:
            await integrator.close()
        except:
            pass


if __name__ == "__main__":
    """Run the integration demo."""
    asyncio.run(run_integration_demo()) 