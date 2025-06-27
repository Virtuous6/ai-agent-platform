#!/usr/bin/env python3
"""
🔥 Simple Real Agent Execution Test
Direct test of real LLM calls through spawned agents.
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

class SimpleRealExecutionTest:
    """Test real agent execution directly."""
    
    def __init__(self):
        self.total_cost = 0.0
        
    async def test_direct_agent_execution(self):
        """Test real agent execution directly without complex workflows."""
        logger.info("🔥 TESTING DIRECT REAL AGENT EXECUTION")
        
        try:
            # Initialize components
            from orchestrator.agent_orchestrator import AgentOrchestrator
            from database.supabase_logger import SupabaseLogger
            
            db_logger = SupabaseLogger()
            orchestrator = AgentOrchestrator(db_logger=db_logger)
            
            logger.info("✅ System initialized")
            
            # Test 1: Spawn a real specialist agent
            logger.info("🤖 Phase 1: Spawning Real Specialist Agent")
            
            agent_id = await orchestrator.spawn_specialist_agent(
                specialty="Business Analysis Specialist",
                parent_context={"test": "real_execution"},
                temperature=0.3,
                max_tokens=500
            )
            
            if not agent_id:
                raise Exception("Failed to spawn agent")
                
            logger.info(f"✅ Spawned real agent: {agent_id}")
            
            # Test 2: Load and execute the real agent
            logger.info("📞 Phase 2: Loading and Executing Real Agent")
            
            # Load the agent
            agent = await orchestrator.get_or_load_agent(agent_id)
            
            if not agent:
                raise Exception(f"Failed to load agent {agent_id}")
                
            logger.info(f"✅ Loaded agent successfully")
            
            # Test 3: Make a real LLM call
            logger.info("🔥 Phase 3: Making Real LLM Call")
            
            test_message = """Analyze this business scenario and provide 3 specific recommendations:
            
            A small e-commerce company is struggling with:
            - 30% cart abandonment rate
            - Average order processing time of 3 days  
            - Customer complaints about slow support response
            
            Provide 3 actionable recommendations to improve their business performance."""
            
            context = {
                "goal": "business_analysis_test",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info("🤖 Executing real agent with business analysis request...")
            
            # 🔥 THIS IS THE REAL LLM EXECUTION
            agent_response = await agent.process_message(
                message=test_message,
                context=context
            )
            
            # Process the real response
            if agent_response:
                response_text = agent_response.get("response", "No response")
                metadata = agent_response.get("metadata", {})
                tokens_used = metadata.get("tokens_used", 0)
                model_used = metadata.get("model_used", "unknown")
                
                logger.info("🎉 REAL LLM EXECUTION SUCCESSFUL!")
                logger.info(f"📝 Response length: {len(response_text)} characters")
                logger.info(f"💰 Tokens used: {tokens_used}")
                logger.info(f"🤖 Model: {model_used}")
                
                # Show excerpt of real response
                logger.info("📋 Real Agent Response (excerpt):")
                logger.info(f"   {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
                
                # Calculate real cost
                cost = self.calculate_cost(tokens_used, model_used)
                self.total_cost += cost
                logger.info(f"💵 Real cost: ${cost:.4f}")
                
                return {
                    "success": True,
                    "agent_id": agent_id,
                    "response": response_text,
                    "tokens_used": tokens_used,
                    "model_used": model_used,
                    "cost": cost,
                    "response_length": len(response_text)
                }
            else:
                logger.error("❌ Agent returned no response")
                return {"success": False, "error": "No response from agent"}
                
        except Exception as e:
            logger.error(f"❌ Direct execution test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate real OpenAI costs."""
        pricing = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "gpt-3.5-turbo-0125": 0.0005
        }
        
        rate = pricing.get(model, 0.002)
        return (tokens / 1000) * rate
    
    def print_execution_summary(self, result):
        """Print summary of real execution test."""
        logger.info("=" * 60)
        logger.info("🔥 REAL AGENT EXECUTION SUMMARY")
        logger.info("=" * 60)
        
        if result.get("success"):
            logger.info("✅ Status: SUCCESS - Real LLM execution works!")
            logger.info(f"🤖 Agent ID: {result['agent_id']}")
            logger.info(f"📝 Response Length: {result['response_length']} characters")
            logger.info(f"💰 Tokens Used: {result['tokens_used']}")
            logger.info(f"🤖 Model: {result['model_used']}")
            logger.info(f"💵 Cost: ${result['cost']:.4f}")
            logger.info("🎯 REAL EXECUTION CONFIRMED!")
        else:
            logger.info("❌ Status: FAILED")
            logger.info(f"🚨 Error: {result.get('error', 'Unknown error')}")
        
        logger.info("=" * 60)

async def main():
    """Run the simple real execution test."""
    print("🔥 SIMPLE REAL AGENT EXECUTION TEST")
    print("💡 Direct test of real LLM calls")
    print("🎯 Bypassing complex workflows")
    print("=" * 60)
    
    test = SimpleRealExecutionTest()
    result = await test.test_direct_agent_execution()
    
    test.print_execution_summary(result)
    
    if result.get("success"):
        print("🎉 REAL EXECUTION CONFIRMED!")
        print("✅ Your agents can make actual LLM calls")
        print("🔥 100% real execution capability validated")
    else:
        print("⚠️ Real execution test had issues")
        print(f"📋 Error: {result.get('error', 'Unknown')}")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(main()) 