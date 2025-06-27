"""
Quick Agent Integration Review

Tests all agents for proper integration with platform components.
"""

import sys
import os
sys.path.append('/Users/josephsanchez/Documents/ai-agent-platform')

from agents.general.general_agent import GeneralAgent
from agents.research.research_agent import ResearchAgent
from agents.technical.technical_agent import TechnicalAgent
from agents.universal_agent import UniversalAgent

def test_agent_integrations():
    """Test agent integrations with platform components."""
    
    print("🔍 Agent Integration Review\n")
    
    # Required components for proper integration
    required_components = [
        'supabase_logger',
        'vector_store', 
        'event_bus',
        'workflow_tracker',
        'tool_registry'
    ]
    
    agents_to_test = [
        ("GeneralAgent", GeneralAgent),
        ("ResearchAgent", ResearchAgent), 
        ("TechnicalAgent", TechnicalAgent),
        ("UniversalAgent", lambda: UniversalAgent("Test", "Test prompt"))
    ]
    
    results = {}
    
    for agent_name, agent_class in agents_to_test:
        print(f"Testing {agent_name}...")
        
        try:
            agent = agent_class()
            missing = []
            
            for component in required_components:
                if not hasattr(agent, component):
                    missing.append(component)
            
            results[agent_name] = {
                'missing': missing,
                'integration_score': len(required_components) - len(missing),
                'total_required': len(required_components)
            }
            
            if missing:
                print(f"  ❌ Missing: {', '.join(missing)}")
            else:
                print(f"  ✅ All integrations present")
                
        except Exception as e:
            print(f"  💥 Error: {e}")
            results[agent_name] = {
                'error': str(e),
                'integration_score': 0,
                'total_required': len(required_components)
            }
    
    # Summary
    print(f"\n📊 Integration Summary:")
    total_agents = len(results)
    fully_integrated = sum(1 for r in results.values() if r.get('integration_score', 0) == len(required_components))
    
    print(f"  Total agents: {total_agents}")
    print(f"  Fully integrated: {fully_integrated}")
    print(f"  Need fixes: {total_agents - fully_integrated}")
    
    print(f"\n❌ Issues Found:")
    for agent_name, result in results.items():
        if result.get('missing'):
            print(f"  {agent_name}: missing {len(result['missing'])}/{result['total_required']} components")
        elif result.get('error'):
            print(f"  {agent_name}: {result['error']}")
    
    return results

if __name__ == "__main__":
    test_agent_integrations() 