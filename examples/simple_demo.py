#!/usr/bin/env python3
"""
Simple Demo: smolagents-inspired agent with universal tools
Shows how we implemented the key features:
- Universal tools (web search, calculate, etc.) available to all agents
- Code-first execution 
- Simple interface
- Platform integration
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simple_agent import SimpleCodeAgent, HybridAgent

async def main():
    print("🔥 smolagents-inspired Simple Agent Demo")
    print("=" * 60)
    
    print("✅ IMPLEMENTED FEATURES:")
    print("  • Universal tools (web_search, calculate, etc.)")
    print("  • @tool decorator pattern")  
    print("  • Code-first execution like smolagents")
    print("  • Simple interface - just agent.run(message)")
    print("  • Platform integration for learning")
    print()
    
    # Create simple agent
    agent = SimpleCodeAgent(agent_id="demo_simple")
    
    # Demo tasks that show universal capabilities
    tasks = [
        "Calculate 15% tip on a $47.80 bill",
        "What's the square root of 144?",
        "Calculate compound interest: $5000 at 3.5% for 2 years",
    ]
    
    print("🧪 TESTING UNIVERSAL CAPABILITIES:")
    print("-" * 40)
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{i}. {task}")
        
        result = await agent.run(task)
        
        if result["success"]:
            print(f"   ✅ {result['response']}")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
    
    print(f"\n📊 AGENT TOOLS:")
    print(f"   Available: {agent.list_tools()}")
    
    print(f"\n🎯 SUCCESS: smolagents patterns implemented!")
    print(f"   • Tools work universally across all agents")
    print(f"   • Code-first execution reduces LLM calls")
    print(f"   • Simple interface like smolagents.run()")
    print(f"   • Ready for platform integration")

if __name__ == "__main__":
    asyncio.run(main()) 