#!/usr/bin/env python3
"""
Agent Lifecycle Verification Script

This script verifies that all agents implement proper lifecycle management
including the mandatory async def close() method.

Usage:
    python agents/verify_lifecycle.py
"""

import asyncio
import inspect
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_agent_lifecycle(agent_class, agent_name: str) -> dict:
    """Check if an agent implements proper lifecycle management."""
    
    results = {
        "agent_name": agent_name,
        "has_close_method": False,
        "close_is_async": False,
        "has_init": False,
        "issues": []
    }
    
    # Check for __init__ method
    if hasattr(agent_class, '__init__'):
        results["has_init"] = True
    else:
        results["issues"].append("Missing __init__ method")
    
    # Check for close method
    if hasattr(agent_class, 'close'):
        results["has_close_method"] = True
        
        # Check if close method is async
        if inspect.iscoroutinefunction(agent_class.close):
            results["close_is_async"] = True
        else:
            results["issues"].append("close() method is not async")
    else:
        results["issues"].append("Missing close() method - CRITICAL")
    
    # Check method signatures
    try:
        if hasattr(agent_class, 'process_message'):
            sig = inspect.signature(agent_class.process_message)
            if 'message' not in sig.parameters or 'context' not in sig.parameters:
                results["issues"].append("process_message() missing required parameters")
        else:
            results["issues"].append("Missing process_message() method")
    except Exception as e:
        results["issues"].append(f"Error checking process_message: {e}")
    
    return results

async def test_agent_lifecycle(agent_class, agent_name: str) -> dict:
    """Test actual lifecycle of an agent."""
    
    test_results = {
        "agent_name": agent_name,
        "initialization_success": False,
        "close_success": False,
        "errors": []
    }
    
    agent_instance = None
    
    try:
        # Test initialization
        agent_instance = agent_class()
        test_results["initialization_success"] = True
        print(f"  ✅ {agent_name} initialized successfully")
        
        # Test close method if it exists
        if hasattr(agent_instance, 'close'):
            await agent_instance.close()
            test_results["close_success"] = True
            print(f"  ✅ {agent_name} closed successfully")
        else:
            test_results["errors"].append("No close() method to test")
            print(f"  ❌ {agent_name} has no close() method")
            
    except Exception as e:
        test_results["errors"].append(str(e))
        print(f"  ❌ {agent_name} lifecycle test failed: {e}")
    
    return test_results

async def main():
    """Main verification function."""
    
    print("🔄 Agent Lifecycle Verification")
    print("=" * 50)
    
    # Import all agents
    agents_to_check = []
    
    try:
        from agents.general.general_agent import GeneralAgent
        agents_to_check.append((GeneralAgent, "GeneralAgent"))
    except ImportError as e:
        print(f"❌ Could not import GeneralAgent: {e}")
    
    try:
        from agents.technical.technical_agent import TechnicalAgent
        agents_to_check.append((TechnicalAgent, "TechnicalAgent"))
    except ImportError as e:
        print(f"❌ Could not import TechnicalAgent: {e}")
    
    try:
        from agents.research.research_agent import ResearchAgent
        agents_to_check.append((ResearchAgent, "ResearchAgent"))
    except ImportError as e:
        print(f"❌ Could not import ResearchAgent: {e}")
    
    if not agents_to_check:
        print("❌ No agents found to check!")
        return
    
    print(f"📊 Found {len(agents_to_check)} agents to verify\n")
    
    # Check agent interfaces
    print("🔍 Checking Agent Interfaces...")
    interface_results = []
    
    for agent_class, agent_name in agents_to_check:
        print(f"\n📋 Checking {agent_name}:")
        result = check_agent_lifecycle(agent_class, agent_name)
        interface_results.append(result)
        
        if result["has_close_method"] and result["close_is_async"]:
            print(f"  ✅ Has proper async close() method")
        elif result["has_close_method"]:
            print(f"  ⚠️  Has close() method but not async")
        else:
            print(f"  ❌ Missing close() method")
        
        if result["issues"]:
            for issue in result["issues"]:
                print(f"  ❗ {issue}")
    
    # Test actual lifecycle
    print(f"\n🧪 Testing Agent Lifecycle...")
    lifecycle_results = []
    
    for agent_class, agent_name in agents_to_check:
        print(f"\n🔄 Testing {agent_name} lifecycle:")
        
        # Skip lifecycle test if missing close method
        interface_result = next(r for r in interface_results if r["agent_name"] == agent_name)
        if not interface_result["has_close_method"]:
            print(f"  ⏭️  Skipping lifecycle test (no close method)")
            continue
            
        result = await test_agent_lifecycle(agent_class, agent_name)
        lifecycle_results.append(result)
    
    # Summary
    print(f"\n📊 Verification Summary")
    print("=" * 30)
    
    total_agents = len(agents_to_check)
    agents_with_close = sum(1 for r in interface_results if r["has_close_method"])
    agents_with_async_close = sum(1 for r in interface_results if r["has_close_method"] and r["close_is_async"])
    successful_tests = sum(1 for r in lifecycle_results if r["initialization_success"] and r["close_success"])
    
    print(f"Total agents checked: {total_agents}")
    print(f"Agents with close() method: {agents_with_close}/{total_agents}")
    print(f"Agents with async close(): {agents_with_async_close}/{total_agents}")
    print(f"Successful lifecycle tests: {successful_tests}/{len(lifecycle_results)}")
    
    # Critical issues
    critical_issues = []
    for result in interface_results:
        if not result["has_close_method"]:
            critical_issues.append(f"{result['agent_name']}: Missing close() method")
        elif not result["close_is_async"]:
            critical_issues.append(f"{result['agent_name']}: close() method not async")
    
    if critical_issues:
        print(f"\n🚨 Critical Issues:")
        for issue in critical_issues:
            print(f"  ❌ {issue}")
        print(f"\n⚠️  Agents with critical issues may cause resource leaks!")
    else:
        print(f"\n✅ All agents have proper lifecycle management!")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if agents_with_async_close == total_agents:
        print("  ✅ All agents implement proper lifecycle management")
    else:
        print("  📖 Review agents/README.llm.md for lifecycle management patterns")
        print("  📝 Follow agents/AGENT_DEVELOPMENT_GUIDE.md for new agents")
        print("  🔧 Add missing close() methods to existing agents")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Verification interrupted by user")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        sys.exit(1) 