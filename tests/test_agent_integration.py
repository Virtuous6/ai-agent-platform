"""
Comprehensive Agent Integration Tests

Tests all agents for proper integration with:
- Supabase logging
- Memory/Vector store
- Event bus
- Goals system
- Database
- Orchestrator
- Tools registry
"""

import asyncio
import pytest
import os
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Import agents
from agents.general.general_agent import GeneralAgent
from agents.research.research_agent import ResearchAgent
from agents.technical.technical_agent import TechnicalAgent
from agents.universal_agent import UniversalAgent
from agents.improvement.workflow_analyst import WorkflowAnalyst

# Import platform components
from database.supabase_logger import SupabaseLogger
from memory.vector_store import VectorMemoryStore
from events.event_bus import EventBus
from orchestrator.agent_orchestrator import AgentOrchestrator

class TestAgentIntegration:
    """Test suite for agent platform integration."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        return {
            'supabase_logger': Mock(spec=SupabaseLogger),
            'vector_store': Mock(spec=VectorMemoryStore),
            'event_bus': Mock(spec=EventBus),
            'orchestrator': Mock(spec=AgentOrchestrator)
        }
    
    def test_general_agent_missing_integrations(self, mock_components):
        """Test that GeneralAgent is missing required integrations."""
        agent = GeneralAgent()
        
        # Check for missing integrations
        assert not hasattr(agent, 'supabase_logger'), "GeneralAgent should not have supabase_logger yet"
        assert not hasattr(agent, 'vector_store'), "GeneralAgent should not have vector_store yet"
        assert not hasattr(agent, 'event_bus'), "GeneralAgent should not have event_bus yet"
        assert not hasattr(agent, 'workflow_tracker'), "GeneralAgent should not have workflow_tracker yet"
    
    def test_research_agent_missing_integrations(self, mock_components):
        """Test that ResearchAgent is missing required integrations."""
        agent = ResearchAgent()
        
        # Check for missing integrations
        assert not hasattr(agent, 'supabase_logger'), "ResearchAgent should not have supabase_logger yet"
        assert not hasattr(agent, 'vector_store'), "ResearchAgent should not have vector_store yet"
        assert not hasattr(agent, 'event_bus'), "ResearchAgent should not have event_bus yet"
        assert not hasattr(agent, 'web_search_tool'), "ResearchAgent should not have web_search_tool yet"
    
    def test_technical_agent_missing_integrations(self, mock_components):
        """Test that TechnicalAgent is missing required integrations."""
        agent = TechnicalAgent()
        
        # Check for missing integrations
        assert not hasattr(agent, 'supabase_logger'), "TechnicalAgent should not have supabase_logger yet"
        assert not hasattr(agent, 'vector_store'), "TechnicalAgent should not have vector_store yet"
        assert not hasattr(agent, 'event_bus'), "TechnicalAgent should not have event_bus yet"
        assert not hasattr(agent, 'code_analysis_tools'), "TechnicalAgent should not have code_analysis_tools yet"
    
    def test_universal_agent_missing_integrations(self, mock_components):
        """Test that UniversalAgent is missing required integrations."""
        agent = UniversalAgent(
            specialty="Test Specialist",
            system_prompt="Test prompt"
        )
        
        # Check for missing integrations
        assert not hasattr(agent, 'supabase_logger'), "UniversalAgent should not have supabase_logger yet"
        assert not hasattr(agent, 'vector_store'), "UniversalAgent should not have vector_store yet"
        assert not hasattr(agent, 'event_bus'), "UniversalAgent should not have event_bus yet"
        assert not hasattr(agent, 'workflow_tracker'), "UniversalAgent should not have workflow_tracker yet"
    
    @pytest.mark.asyncio
    async def test_agent_supabase_logging_missing(self, mock_components):
        """Test that agents don't log to Supabase."""
        agent = GeneralAgent()
        
        # Mock message processing
        with patch.object(agent.llm, 'agenerate', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = Mock(generations=[[Mock(text="Test response")]])
            
            result = await agent.process_message("test message", {})
            
            # Verify no Supabase logging occurred
            assert 'conversation_id' not in result, "Agent should not be logging to Supabase yet"
            assert 'message_id' not in result, "Agent should not be logging message IDs yet"
    
    @pytest.mark.asyncio
    async def test_agent_event_publishing_missing(self, mock_components):
        """Test that agents don't publish events."""
        agent = ResearchAgent()
        
        # Process a message
        with patch.object(agent.llm, 'agenerate', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = Mock(generations=[[Mock(text="Test response")]])
            
            result = await agent.process_message("research request", {})
            
            # Verify no events were published
            assert 'events_published' not in result, "Agent should not be publishing events yet"
    
    @pytest.mark.asyncio
    async def test_agent_memory_storage_missing(self, mock_components):
        """Test that agents don't store memories."""
        agent = TechnicalAgent()
        
        # Process a message
        with patch.object(agent.llm, 'agenerate', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = Mock(generations=[[Mock(text="Test response")]])
            
            result = await agent.process_message("technical question", {})
            
            # Verify no memory storage
            assert 'memory_stored' not in result, "Agent should not be storing memories yet"
            assert 'context_retrieved' not in result, "Agent should not be retrieving context yet"

@pytest.mark.asyncio
async def test_run_agent_integration_tests():
    """Run all agent integration tests."""
    print("\n🧪 Running Agent Integration Tests...")
    
    test_suite = TestAgentIntegration()
    mock_components = test_suite.mock_components()
    
    # Test each agent type
    agents_to_test = [
        ("GeneralAgent", GeneralAgent),
        ("ResearchAgent", ResearchAgent), 
        ("TechnicalAgent", TechnicalAgent),
        ("UniversalAgent", lambda: UniversalAgent("Test", "Test prompt"))
    ]
    
    results = {
        "total_tests": 0,
        "failed_integrations": [],
        "missing_components": {},
        "recommendations": []
    }
    
    for agent_name, agent_class in agents_to_test:
        print(f"\n  Testing {agent_name}...")
        
        try:
            agent = agent_class()
            results["total_tests"] += 1
            
            # Check for required integrations
            missing = []
            required_components = [
                'supabase_logger',
                'vector_store', 
                'event_bus',
                'workflow_tracker'
            ]
            
            for component in required_components:
                if not hasattr(agent, component):
                    missing.append(component)
            
            if missing:
                results["failed_integrations"].append(agent_name)
                results["missing_components"][agent_name] = missing
                print(f"    ❌ Missing: {', '.join(missing)}")
            else:
                print(f"    ✅ All integrations present")
                
        except Exception as e:
            print(f"    💥 Error testing {agent_name}: {e}")
            results["failed_integrations"].append(agent_name)
    
    # Generate recommendations
    if results["failed_integrations"]:
        results["recommendations"] = [
            "Add Supabase logging integration to all agents",
            "Connect agents to event bus for communication",
            "Integrate vector memory store for context",
            "Add workflow tracking capabilities",
            "Create unified agent base class with integrations"
        ]
    
    print(f"\n📊 Integration Test Results:")
    print(f"  Total agents tested: {results['total_tests']}")
    print(f"  Failed integrations: {len(results['failed_integrations'])}")
    
    if results["missing_components"]:
        print(f"\n❌ Missing Components by Agent:")
        for agent, missing in results["missing_components"].items():
            print(f"  {agent}: {', '.join(missing)}")
    
    if results["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in results["recommendations"]:
            print(f"  • {rec}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_run_agent_integration_tests()) 