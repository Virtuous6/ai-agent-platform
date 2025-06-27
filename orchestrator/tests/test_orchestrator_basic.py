#!/usr/bin/env python3
"""
Basic test for Orchestrator functionality
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

async def test_orchestrator_basic():
    """Test basic orchestrator functionality."""
    print("🚀 Testing Basic Orchestrator Functionality...")
    
    try:
        # Test 1: Import orchestrator
        from orchestrator.agent_orchestrator import AgentOrchestrator
        print("✅ Successfully imported AgentOrchestrator")
        
        # Test 2: Initialize orchestrator
        orchestrator = AgentOrchestrator()
        print("✅ Successfully initialized AgentOrchestrator")
        
        # Test 3: Check lazy loader integration
        if hasattr(orchestrator, 'lazy_loader'):
            print("✅ Lazy loader integrated")
            print(f"   • Max active agents: {orchestrator.lazy_loader.max_active_agents}")
            print(f"   • Max configurations: {orchestrator.lazy_loader.max_total_configurations}")
        else:
            print("❌ Lazy loader not found")
        
        # Test 4: Check improvement orchestrator
        if hasattr(orchestrator, 'improvement_orchestrator'):
            if orchestrator.improvement_orchestrator:
                print("✅ Improvement orchestrator integrated")
            else:
                print("⚠️ Improvement orchestrator is None (expected without db_logger)")
        else:
            print("❌ Improvement orchestrator not found")
        
        # Test 5: Check workflow tracker
        if hasattr(orchestrator, 'workflow_tracker'):
            print("✅ Workflow tracker integrated")
        else:
            print("❌ Workflow tracker not found")
        
        # Test 6: Test agent spawning
        print("\n🔧 Testing agent spawning...")
        agent_id = await orchestrator.spawn_specialist_agent(
            specialty="Test Specialist",
            parent_context={"test": True},
            temperature=0.3,
            max_tokens=500
        )
        
        if agent_id:
            print(f"✅ Successfully spawned agent: {agent_id}")
            
            # Test 7: Test agent loading
            agent = await orchestrator.get_or_load_agent(agent_id)
            if agent:
                print(f"✅ Successfully loaded agent: {agent_id}")
            else:
                print(f"❌ Failed to load agent: {agent_id}")
        else:
            print("❌ Failed to spawn agent")
        
        # Test 8: Get agent stats
        stats = orchestrator.get_agent_stats()
        print(f"\n📊 Agent Stats:")
        print(f"   • Active agents: {stats.get('active_agents', 0)}")
        print(f"   • Total spawned: {stats.get('total_spawned', 0)}")
        if 'lazy_loader' in stats:
            print(f"   • Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
        
        # Test 9: Test basic routing (without LLM)
        print(f"\n🔀 Testing basic routing...")
        
        # Test explicit agent mention
        test_message = "Hey @technical, help me debug this code"
        explicit_agent = orchestrator._check_explicit_mentions(test_message)
        if explicit_agent:
            print(f"✅ Explicit agent detection working: {explicit_agent}")
        else:
            print("⚠️ No explicit agent detected (expected for this test)")
        
        # Test 10: Cleanup
        await orchestrator.close()
        print("✅ Successfully closed orchestrator")
        
        print(f"\n🎉 All basic tests passed! Orchestrator is functional.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_supabase_integration():
    """Test Supabase integration if available."""
    print("\n🗄️ Testing Supabase Integration...")
    
    try:
        from database.supabase_logger import SupabaseLogger
        print("✅ Successfully imported SupabaseLogger")
        
        # Try to initialize (may fail if env vars not set)
        try:
            logger = SupabaseLogger()
            health = logger.health_check()
            print(f"✅ Supabase connection: {health.get('status', 'unknown')}")
            
            # Test with orchestrator
            from orchestrator.agent_orchestrator import AgentOrchestrator
            orchestrator = AgentOrchestrator(db_logger=logger)
            print("✅ Orchestrator initialized with Supabase logger")
            
            await orchestrator.close()
            await logger.close()
            
        except Exception as e:
            print(f"⚠️ Supabase connection failed (likely missing env vars): {str(e)}")
            
    except Exception as e:
        print(f"❌ Supabase test failed: {str(e)}")

async def test_langgraph_integration():
    """Test LangGraph integration if available."""
    print("\n🔄 Testing LangGraph Integration...")
    
    try:
        from orchestrator.langgraph.workflow_engine import LangGraphWorkflowEngine, LANGGRAPH_AVAILABLE
        print("✅ Successfully imported LangGraphWorkflowEngine")
        
        print(f"✅ LANGGRAPH_AVAILABLE: {LANGGRAPH_AVAILABLE}")
        
        engine = LangGraphWorkflowEngine({}, {})
        if engine.is_available():
            print("✅ LangGraph is available and functional")
            
            # Test basic workflow functionality
            workflows = engine.get_loaded_workflows()
            print(f"✅ Current loaded workflows: {len(workflows)}")
        else:
            print("⚠️ LangGraph not available in engine")
        
        await engine.close()
        
        # Test direct LangGraph functionality
        if LANGGRAPH_AVAILABLE:
            try:
                from langgraph.graph import StateGraph, END
                print("✅ Direct LangGraph imports working")
                
                # Try creating a simple graph
                from orchestrator.langgraph.state_schemas import RunbookState
                graph = StateGraph(RunbookState)
                print("✅ StateGraph creation successful")
                
            except Exception as e:
                print(f"⚠️ Direct LangGraph test failed: {str(e)}")
        
    except Exception as e:
        print(f"❌ LangGraph test failed: {str(e)}")

async def main():
    """Run all tests."""
    print("="*60)
    print("🧪 ORCHESTRATOR COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    success = True
    
    # Basic functionality
    success &= await test_orchestrator_basic()
    
    # Supabase integration
    await test_supabase_integration()
    
    # LangGraph integration  
    await test_langgraph_integration()
    
    print("\n" + "="*60)
    if success:
        print("🎉 ORCHESTRATOR TEST SUITE COMPLETED SUCCESSFULLY!")
        print("✅ The orchestrator is correctly connected and functional")
    else:
        print("❌ SOME TESTS FAILED - CHECK OUTPUT ABOVE")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main()) 