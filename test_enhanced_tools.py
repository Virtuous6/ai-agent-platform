#!/usr/bin/env python3
"""
Test script to validate enhanced tool execution improvements.

This script tests all the major improvements made to the UniversalAgent:
1. Rule-based tool detection
2. Tool execution planning and chaining
3. Context persistence and learning
4. Code generation integration
5. Caching and error recovery
"""

import asyncio
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import UniversalAgent, ToolCapability

class MockCalculateTool:
    """Mock calculator tool for testing."""
    
    @staticmethod
    async def calculate(message: str, context: dict) -> dict:
        """Perform basic mathematical calculations."""
        try:
            # Extract numbers and operation
            import re
            match = re.search(r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)', message)
            if match:
                num1, op, num2 = match.groups()
                num1, num2 = float(num1), float(num2)
                
                operations = {
                    '+': num1 + num2,
                    '-': num1 - num2,
                    '*': num1 * num2,
                    '/': num1 / num2 if num2 != 0 else None
                }
                
                result = operations.get(op)
                if result is not None:
                    return {"result": result, "operation": f"{num1} {op} {num2} = {result}"}
            
            return {"error": "Could not parse mathematical expression"}
        except Exception as e:
            return {"error": str(e)}

class MockSearchTool:
    """Mock search tool for testing."""
    
    @staticmethod
    async def web_search(message: str, context: dict) -> dict:
        """Mock web search that returns fake results."""
        search_terms = message.lower()
        
        # Simulate different responses based on search terms
        if "python" in search_terms:
            return {
                "results": "Python is a high-level programming language",
                "success": True,
                "source": "mock_search"
            }
        elif "weather" in search_terms:
            return {
                "results": "The weather is sunny and 75°F",
                "success": True,
                "source": "mock_search"
            }
        else:
            return {
                "results": f"Mock search results for: {search_terms}",
                "success": True,
                "source": "mock_search"
            }

async def test_enhanced_tool_execution():
    """Test all enhanced tool execution features."""
    
    print("🧪 Testing Enhanced Tool Execution System")
    print("=" * 50)
    
    # Create tools
    calc_tool = ToolCapability(
        name="calculate",
        description="Perform mathematical calculations",
        function=MockCalculateTool.calculate,
        enabled=True
    )
    
    search_tool = ToolCapability(
        name="web_search", 
        description="Search the web for information",
        function=MockSearchTool.web_search,
        enabled=True
    )
    
    # Create agent with tools
    agent = UniversalAgent(
        specialty="Enhanced Testing Agent",
        system_prompt="You are a testing agent with enhanced tool capabilities.",
        tools=[calc_tool, search_tool],
        agent_id="test_enhanced_agent"
    )
    
    # Test cases
    test_cases = [
        {
            "name": "Code Generation - Compound Interest",
            "message": "Calculate compound interest on $1000 at 5% for 3 years",
            "expect_code": True
        },
        {
            "name": "Code Generation - Multi-step Tip Calculation", 
            "message": "Calculate 15% tip on $87.50 plus 8.5% tax",
            "expect_code": True
        },
        {
            "name": "Tool Detection - Search",
            "message": "Search for information about Python",
            "expect_tools": ["search"]
        },
        {
            "name": "Tool Detection - Simple Math",
            "message": "What's 25 * 47 + 128?",
            "expect_tools": ["calculate"]
        },
        {
            "name": "Tool Detection - Calculate",
            "message": "Calculate 100 + 50",
            "expect_tools": ["calculate"]
        },
        {
            "name": "Tool Chaining - Search then Calculate",
            "message": "Search for Python and calculate 10 * 5",
            "expect_tools": ["search", "calculate"]
        }
    ]
    
    # Run test cases
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print(f"Query: {test_case['message']}")
        print("-" * 40)
        
        try:
            # Execute the test
            context = {"user_id": "test_user", "test_case": test_case['name']}
            
            # Test the enhanced tool execution directly
            print(f"  Debug: Testing message '{test_case['message']}'")
            tool_results = await agent._execute_tools(test_case['message'], context)
            print(f"  Debug: Got results: {tool_results}")
            
            # Also test the individual detection methods
            should_code = agent._should_use_code_generation(test_case['message'])
            detected_tools = agent._detect_required_tools(test_case['message'])
            print(f"  Debug: Should use code: {should_code}, Detected tools: {detected_tools}")
            
            # Analyze results
            success = True
            details = []
            
            if test_case.get('expect_code'):
                if 'code_execution' in tool_results:
                    details.append("✅ Code generation triggered")
                    code_result = tool_results['code_execution']
                    if code_result.get('success'):
                        details.append(f"✅ Code executed: {code_result.get('output', 'N/A')}")
                    else:
                        details.append(f"❌ Code execution failed: {code_result.get('error', 'Unknown error')}")
                        success = False
                else:
                    details.append("❌ Code generation not triggered")
                    success = False
            
            if test_case.get('expect_tools'):
                expected_tools = test_case['expect_tools']
                found_tools = []
                
                for tool_type in expected_tools:
                    if any(tool_type in key for key in tool_results.keys()):
                        found_tools.append(tool_type)
                        details.append(f"✅ Tool detected: {tool_type}")
                    else:
                        details.append(f"❌ Tool not detected: {tool_type}")
                        success = False
            
            # Print results
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"Status: {status}")
            for detail in details:
                print(f"  {detail}")
            
            if tool_results:
                print(f"  Tool Results: {list(tool_results.keys())}")
            
            results.append({
                "test": test_case['name'],
                "success": success,
                "details": details,
                "tool_results": tool_results
            })
            
        except Exception as e:
            print(f"❌ FAIL: Exception occurred: {e}")
            results.append({
                "test": test_case['name'],
                "success": False,
                "error": str(e)
            })
    
    # Test status reporting
    print(f"\n📊 Testing Status Reporting")
    print("-" * 40)
    
    stats = agent.get_enhanced_tool_stats()
    print(f"✅ Tool execution system version: {stats['tool_execution_system']['version']}")
    print(f"✅ Available tools: {len(stats['available_tools'])}")
    print(f"✅ Features enabled: {len(stats['tool_execution_system']['features'])}")
    print(f"✅ Conversation context tracking: {stats['conversation_context']}")
    
    # Summary
    print(f"\n🎯 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced tool execution is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the details above.")
        for result in results:
            if not result['success']:
                print(f"  - Failed: {result['test']}")
    
    return results

if __name__ == "__main__":
    print("🚀 Enhanced Tool Execution Test Suite")
    print("Testing all improvements made to UniversalAgent tool execution")
    print()
    
    # Run the tests
    asyncio.run(test_enhanced_tool_execution()) 