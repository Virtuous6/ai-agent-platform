#!/usr/bin/env python3
"""
Test script for QueryComplexityRouter

Tests the automatic complexity assessment functionality to ensure
it correctly routes different types of queries.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_complexity_router():
    """Test the QueryComplexityRouter with various query types."""
    
    # Import the router
    from orchestrator.query_complexity_router import QueryComplexityRouter, ComplexityLevel
    
    # Initialize the router
    router = QueryComplexityRouter()
    
    # Test queries of different complexity levels
    test_queries = [
        # Simple queries (should be 0.0-0.3)
        {
            "message": "What is Python?",
            "expected_level": ComplexityLevel.SIMPLE,
            "description": "Simple definition question"
        },
        {
            "message": "Explain what REST APIs are",
            "expected_level": ComplexityLevel.SIMPLE,
            "description": "Simple explanation request"
        },
        {
            "message": "Define machine learning",
            "expected_level": ComplexityLevel.SIMPLE,
            "description": "Simple definition"
        },
        
        # Medium queries (should be 0.3-0.7)
        {
            "message": "Analyze this code for performance issues",
            "expected_level": ComplexityLevel.MEDIUM,
            "description": "Analysis task requiring tools"
        },
        {
            "message": "Research the best JavaScript frameworks for 2024",
            "expected_level": ComplexityLevel.MEDIUM,
            "description": "Research task"
        },
        {
            "message": "Compare React vs Vue.js performance",
            "expected_level": ComplexityLevel.MEDIUM,
            "description": "Comparison analysis"
        },
        
        # Complex queries (should be 0.7-1.0)
        {
            "message": "Build a full-stack web application with user authentication",
            "expected_level": ComplexityLevel.COMPLEX,
            "description": "Multi-step creation task"
        },
        {
            "message": "Deploy a microservices architecture on AWS with monitoring",
            "expected_level": ComplexityLevel.COMPLEX,
            "description": "Complex deployment task"
        },
        {
            "message": "Create a complete marketing strategy for a SaaS startup",
            "expected_level": ComplexityLevel.COMPLEX,
            "description": "Multi-domain coordination task"
        }
    ]
    
    print("🧠 Testing QueryComplexityRouter")
    print("=" * 60)
    
    correct_assessments = 0
    total_assessments = len(test_queries)
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"Query: \"{test_case['message']}\"")
        
        try:
            # Assess complexity
            analysis = await router.assess_complexity(
                message=test_case['message'],
                context={"user_id": "test_user"}
            )
            
            # Check if assessment matches expected level
            correct = analysis.level == test_case['expected_level']
            if correct:
                correct_assessments += 1
                status = "✅ CORRECT"
            else:
                status = "❌ INCORRECT"
            
            print(f"Expected: {test_case['expected_level'].value}")
            print(f"Actual: {analysis.level.value} (score: {analysis.score:.2f}) {status}")
            print(f"Strategy: {analysis.strategy.value}")
            print(f"Reasoning: {analysis.reasoning}")
            print(f"Indicators: {', '.join(analysis.indicators)}")
            print(f"Estimated steps: {analysis.estimated_steps}")
            print(f"Requires tools: {analysis.requires_tools}")
            print(f"Requires multiple agents: {analysis.requires_multiple_agents}")
            print(f"Estimated cost: ${analysis.estimated_cost:.3f}")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY")
    print(f"Correct assessments: {correct_assessments}/{total_assessments}")
    print(f"Accuracy: {(correct_assessments/total_assessments)*100:.1f}%")
    
    # Test quick complexity check
    print(f"\n🚀 Testing quick complexity check...")
    for query in ["What is AI?", "Build an app", "Analyze this data"]:
        score, level = await router.quick_complexity_check(query)
        print(f"'{query}' -> {level.value} ({score:.2f})")
    
    # Get assessment statistics
    print(f"\n📈 Assessment Statistics:")
    stats = router.get_assessment_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\n✅ QueryComplexityRouter test completed!")
    return correct_assessments / total_assessments

async def test_orchestrator_integration():
    """Test the QueryComplexityRouter integration with AgentOrchestrator."""
    
    print(f"\n🔗 Testing AgentOrchestrator Integration")
    print("=" * 60)
    
    try:
        from orchestrator.agent_orchestrator import AgentOrchestrator
        
        # Initialize orchestrator
        orchestrator = AgentOrchestrator()
        
        # Test complexity assessment method
        test_queries = [
            "What is Python?",
            "Analyze this code for bugs", 
            "Build a complete e-commerce platform"
        ]
        
        for query in test_queries:
            print(f"\n📝 Testing: \"{query}\"")
            
            # Test direct complexity assessment
            analysis = await orchestrator.assess_query_complexity(query)
            print(f"Complexity: {analysis.level.value} ({analysis.score:.2f})")
            print(f"Strategy: {analysis.strategy.value}")
            
        # Test routing statistics
        print(f"\n📊 Routing Statistics:")
        stats = orchestrator.get_routing_stats()
        if "complexity_assessment" in stats:
            complexity_stats = stats["complexity_assessment"]
            print(f"Total assessments: {complexity_stats.get('total_assessments', 0)}")
            if "complexity_distribution" in complexity_stats:
                dist = complexity_stats["complexity_distribution"]
                print(f"Distribution: Simple={dist['simple']}, Medium={dist['medium']}, Complex={dist['complex']}")
        
        print(f"✅ Orchestrator integration test completed!")
        
        # Clean up
        await orchestrator.close()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Integration test error: {e}")

async def main():
    """Run all tests."""
    print("🧪 Starting QueryComplexityRouter Tests")
    print("=" * 80)
    
    try:
        # Test 1: Basic router functionality
        accuracy = await test_complexity_router()
        
        # Test 2: Orchestrator integration
        await test_orchestrator_integration()
        
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS COMPLETED!")
        print(f"Overall accuracy: {accuracy*100:.1f}%")
        
        if accuracy >= 0.7:  # 70% accuracy threshold
            print("✅ QueryComplexityRouter is working correctly!")
        else:
            print("⚠️ QueryComplexityRouter may need tuning.")
            
    except Exception as e:
        print(f"❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 