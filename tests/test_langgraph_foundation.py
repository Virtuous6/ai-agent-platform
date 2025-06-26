"""
LangGraph Foundation Tests

Tests for the basic LangGraph integration without requiring actual LLM calls.
Uses mocks for cost-effective testing.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import sys
import os

# Add project root to path for testing
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Test imports with proper error handling
try:
    from orchestrator import AgentOrchestrator
    from orchestrator.langgraph import LangGraphWorkflowEngine
    from orchestrator.langgraph.state_schemas import RunbookState, AgentState, ToolState
    from orchestrator.langgraph.runbook_converter import RunbookToGraphConverter
    LANGGRAPH_FOUNDATION_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_FOUNDATION_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Skip entire test suite if foundation not available
pytestmark = pytest.mark.skipif(
    not LANGGRAPH_FOUNDATION_AVAILABLE, 
    reason=f"LangGraph foundation not available: {IMPORT_ERROR if not LANGGRAPH_FOUNDATION_AVAILABLE else ''}"
)

class TestFoundationStructure:
    """Test that the foundation package structure is correct."""
    
    def test_package_imports(self):
        """Test that all foundation packages import correctly."""
        # Test state schemas
        assert RunbookState is not None
        assert AgentState is not None
        assert ToolState is not None
        
        # Test workflow engine
        assert LangGraphWorkflowEngine is not None
        
        # Test runbook converter
        assert RunbookToGraphConverter is not None
        
        # Test orchestrator integration
        assert AgentOrchestrator is not None
    
    def test_state_schema_structure(self):
        """Test that state schemas have required fields."""
        # Check RunbookState required fields
        runbook_annotations = RunbookState.__annotations__
        required_runbook_fields = [
            'user_id', 'user_message', 'current_step', 
            'agent_responses', 'final_response', 'confidence_score'
        ]
        
        for field in required_runbook_fields:
            assert field in runbook_annotations, f"RunbookState missing {field}"
        
        # Check AgentState required fields  
        agent_annotations = AgentState.__annotations__
        required_agent_fields = [
            'agent_type', 'agent_name', 'input_message', 
            'response', 'confidence'
        ]
        
        for field in required_agent_fields:
            assert field in agent_annotations, f"AgentState missing {field}"
    
    def test_orchestrator_langgraph_methods(self):
        """Test that orchestrator has LangGraph integration methods."""
        orchestrator = AgentOrchestrator()
        
        # Test required methods exist
        assert hasattr(orchestrator, 'process_with_langgraph')
        assert hasattr(orchestrator, '_select_runbook_for_message')
        
        # Test methods are callable
        assert callable(orchestrator.process_with_langgraph)
        assert callable(orchestrator._select_runbook_for_message)

class TestLangGraphWorkflowEngine:
    """Test LangGraph workflow engine functionality."""
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agent instances for testing."""
        general_agent = AsyncMock()
        general_agent.process_message.return_value = {
            'response': 'Mock general response',
            'confidence': 0.8,
            'tokens_used': 100
        }
        
        technical_agent = AsyncMock()
        technical_agent.process_message.return_value = {
            'response': 'Mock technical response', 
            'confidence': 0.9,
            'tokens_used': 150
        }
        
        return {
            'general': general_agent,
            'technical': technical_agent,
            'research': AsyncMock()
        }
    
    @pytest.fixture
    def mock_tools(self):
        """Create mock tool instances for testing."""
        web_search_tool = AsyncMock()
        web_search_tool.search.return_value = {
            'results': [{'title': 'Test Result', 'url': 'test.com'}],
            'query': 'test query'
        }
        
        return {
            'web_search': web_search_tool
        }
    
    @pytest.fixture
    def mock_supabase_logger(self):
        """Create mock Supabase logger for testing."""
        logger = Mock()
        logger.client = Mock()
        logger.client.table.return_value.insert.return_value.execute.return_value = Mock(
            data=[{'id': 'test-execution-id'}]
        )
        logger.client.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock()
        return logger
    
    @pytest.fixture
    def workflow_engine(self, mock_agents, mock_tools, mock_supabase_logger):
        """Create workflow engine with mocked dependencies."""
        return LangGraphWorkflowEngine(mock_agents, mock_tools, mock_supabase_logger)
    
    def test_workflow_engine_initialization(self, workflow_engine):
        """Test workflow engine initializes correctly."""
        assert workflow_engine.agents is not None
        assert workflow_engine.tools is not None
        assert workflow_engine.active_workflows == {}
        assert workflow_engine.supabase_logger is not None
    
    def test_is_available_method(self, workflow_engine):
        """Test availability check method."""
        # Should return True/False based on LangGraph availability
        availability = workflow_engine.is_available()
        assert isinstance(availability, bool)
    
    def test_get_loaded_workflows(self, workflow_engine):
        """Test getting list of loaded workflows."""
        # Initially should be empty
        workflows = workflow_engine.get_loaded_workflows()
        assert workflows == []
        
        # Add a mock workflow
        workflow_engine.active_workflows['test-workflow'] = Mock()
        workflows = workflow_engine.get_loaded_workflows()
        assert 'test-workflow' in workflows
    
    @pytest.mark.asyncio
    async def test_workflow_engine_cleanup(self, workflow_engine):
        """Test workflow engine cleans up properly."""
        # Add some mock workflows
        workflow_engine.active_workflows['test1'] = Mock()
        workflow_engine.active_workflows['test2'] = Mock()
        
        # Clean up
        await workflow_engine.close()
        
        # Should be empty after cleanup
        assert workflow_engine.active_workflows == {}
    
    @pytest.mark.asyncio
    async def test_load_runbook_workflow_unavailable(self, mock_agents, mock_tools):
        """Test workflow loading when LangGraph is unavailable."""
        # Create engine that reports unavailable
        engine = LangGraphWorkflowEngine(mock_agents, mock_tools)
        
        # Mock the availability check to return False
        with patch.object(engine, 'is_available', return_value=False):
            result = await engine.load_runbook_workflow('test.yaml')
            assert result is None
    
    @pytest.mark.asyncio
    async def test_execute_workflow_unavailable(self, workflow_engine):
        """Test workflow execution when LangGraph is unavailable."""
        # Mock the availability check to return False
        with patch.object(workflow_engine, 'is_available', return_value=False):
            with pytest.raises(ValueError, match="LangGraph not available"):
                await workflow_engine.execute_workflow('test', {})
    
    @pytest.mark.asyncio
    async def test_execute_workflow_not_loaded(self, workflow_engine):
        """Test workflow execution with unloaded workflow."""
        # Mock availability as True
        with patch.object(workflow_engine, 'is_available', return_value=True):
            with pytest.raises(ValueError, match="Workflow 'nonexistent' not loaded"):
                await workflow_engine.execute_workflow('nonexistent', {})

class TestRunbookConverter:
    """Test runbook to LangGraph conversion functionality."""
    
    @pytest.fixture
    def converter(self):
        """Create runbook converter instance."""
        return RunbookToGraphConverter()
    
    @pytest.fixture
    def sample_runbook_data(self):
        """Create sample runbook data for testing."""
        return {
            'metadata': {
                'name': 'test-runbook',
                'version': '1.0.0'
            },
            'steps': [
                {
                    'id': 'validate_input',
                    'action': 'validate_message',
                    'description': 'Validate user input'
                },
                {
                    'id': 'process_request',
                    'action': 'invoke_agent',
                    'parameters': {'agent': 'general'}
                },
                {
                    'id': 'format_response',
                    'action': 'format_output',
                    'parameters': {'format_type': 'markdown'}
                }
            ],
            'outputs': {
                'success': {'message': 'Completed'}
            }
        }
    
    def test_converter_initialization(self, converter):
        """Test converter initializes with node builders."""
        assert converter.node_builders is not None
        assert 'analyze_message' in converter.node_builders
        assert 'invoke_agent' in converter.node_builders
        assert 'invoke_tool' in converter.node_builders
        assert 'format_output' in converter.node_builders
    
    @pytest.mark.asyncio 
    async def test_load_runbook_file_not_found(self, converter):
        """Test loading non-existent runbook file."""
        with pytest.raises(ValueError, match="Runbook file not found"):
            await converter._load_runbook('nonexistent.yaml')
    
    @pytest.mark.asyncio
    async def test_load_runbook_valid_structure(self, converter, sample_runbook_data, tmp_path):
        """Test loading valid runbook structure."""
        # Create temporary runbook file
        import yaml
        runbook_file = tmp_path / 'test_runbook.yaml'
        with open(runbook_file, 'w') as f:
            yaml.dump(sample_runbook_data, f)
        
        # Load runbook
        loaded_data = await converter._load_runbook(str(runbook_file))
        
        assert loaded_data['metadata']['name'] == 'test-runbook'
        assert len(loaded_data['steps']) == 3
        assert loaded_data['steps'][0]['id'] == 'validate_input'
    
    @pytest.mark.asyncio
    async def test_load_runbook_missing_steps(self, converter, tmp_path):
        """Test loading runbook without steps section."""
        # Create invalid runbook
        import yaml
        invalid_runbook = {'metadata': {'name': 'invalid'}}
        runbook_file = tmp_path / 'invalid_runbook.yaml'
        with open(runbook_file, 'w') as f:
            yaml.dump(invalid_runbook, f)
        
        # Should raise error
        with pytest.raises(ValueError, match="Runbook missing 'steps' section"):
            await converter._load_runbook(str(runbook_file))
    
    @pytest.mark.asyncio
    async def test_convert_runbook_unavailable(self, converter):
        """Test conversion when LangGraph is unavailable."""
        # Mock LangGraph as unavailable
        with patch('orchestrator.langgraph.runbook_converter.LANGGRAPH_AVAILABLE', False):
            result = await converter.convert_runbook_to_graph('test.yaml', {}, {})
            assert result is None

class TestOrchestratorIntegration:
    """Test orchestrator integration with LangGraph."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        return AgentOrchestrator()
    
    @pytest.mark.asyncio
    async def test_select_runbook_for_message(self, orchestrator):
        """Test runbook selection logic."""
        # Test question patterns
        question_msg = "What is the capital of France?"
        context = {}
        runbook = await orchestrator._select_runbook_for_message(question_msg, context)
        assert runbook == 'answer-question'
        
        # Test technical patterns
        tech_msg = "I have a bug in my code"
        runbook = await orchestrator._select_runbook_for_message(tech_msg, context)
        assert runbook == 'technical-support'
        
        # Test research patterns
        research_msg = "Please analyze this data"
        runbook = await orchestrator._select_runbook_for_message(research_msg, context)
        assert runbook == 'research-task'
        
        # Test default fallback
        general_msg = "Hello there"
        runbook = await orchestrator._select_runbook_for_message(general_msg, context)
        assert runbook == 'answer-question'
    
    @pytest.mark.asyncio
    async def test_process_with_langgraph_unavailable(self, orchestrator):
        """Test LangGraph processing when unavailable."""
        # Mock the workflow engine to be unavailable
        with patch.object(orchestrator, '_workflow_engine', None):
            with patch('orchestrator.langgraph.workflow_engine.LANGGRAPH_AVAILABLE', False):
                # Should fall back to standard routing
                with patch.object(orchestrator, 'route_request') as mock_route:
                    mock_route.return_value = {'response': 'fallback response'}
                    
                    result = await orchestrator.process_with_langgraph("test message", {})
                    
                    assert mock_route.called
                    assert result['response'] == 'fallback response'
    
    @pytest.mark.asyncio
    async def test_process_with_langgraph_workflow_failure(self, orchestrator):
        """Test LangGraph processing when workflow loading fails."""
        # Mock workflow engine
        mock_engine = Mock()
        mock_engine.is_available.return_value = True
        mock_engine.load_runbook_workflow.return_value = None  # Simulate failure
        
        orchestrator._workflow_engine = mock_engine
        
        # Should fall back to standard routing
        with patch.object(orchestrator, 'route_request') as mock_route:
            mock_route.return_value = {'response': 'fallback response'}
            
            result = await orchestrator.process_with_langgraph("test message", {})
            
            assert mock_route.called
            assert result['response'] == 'fallback response'

class TestFoundationEndToEnd:
    """End-to-end tests for foundation functionality."""
    
    @pytest.mark.asyncio
    async def test_foundation_graceful_fallback(self):
        """Test that foundation gracefully falls back when dependencies missing."""
        # This test should pass even if LangGraph isn't installed
        
        # Test orchestrator creation
        orchestrator = AgentOrchestrator()
        assert orchestrator is not None
        
        # Test that LangGraph methods exist but handle unavailability
        assert hasattr(orchestrator, 'process_with_langgraph')
        
        # Test fallback behavior
        with patch.object(orchestrator, 'route_request') as mock_route:
            mock_route.return_value = {'response': 'standard routing response'}
            
            result = await orchestrator.process_with_langgraph("test", {})
            assert 'response' in result
    
    def test_package_structure_completeness(self):
        """Test that all expected packages and modules are available."""
        # Test core orchestrator
        from orchestrator import AgentOrchestrator, AgentType
        assert AgentOrchestrator is not None
        assert AgentType is not None
        
        # Test LangGraph integration (should not fail even if LangGraph unavailable)
        try:
            from orchestrator.langgraph import LangGraphWorkflowEngine
            from orchestrator.langgraph.state_schemas import RunbookState
            from orchestrator.langgraph.runbook_converter import RunbookToGraphConverter
        except ImportError:
            # This is acceptable - LangGraph may not be installed yet
            pass
    
    @pytest.mark.asyncio
    async def test_orchestrator_lifecycle_with_langgraph(self):
        """Test orchestrator lifecycle including LangGraph cleanup."""
        orchestrator = AgentOrchestrator()
        
        # Initialize with mock workflow engine
        mock_engine = AsyncMock()
        orchestrator._workflow_engine = mock_engine
        
        # Test close method
        await orchestrator.close()
        
        # Verify workflow engine was closed
        mock_engine.close.assert_called_once()

class TestFoundationPerformance:
    """Performance and resource management tests."""
    
    def test_import_performance(self):
        """Test that imports are reasonably fast."""
        import time
        
        start_time = time.time()
        
        # Re-import modules to test performance
        try:
            from orchestrator.langgraph import LangGraphWorkflowEngine
            from orchestrator.langgraph.state_schemas import RunbookState
        except ImportError:
            # Skip if not available
            pytest.skip("LangGraph not available for performance testing")
        
        import_time = time.time() - start_time
        
        # Should import in under 1 second
        assert import_time < 1.0, f"Imports took {import_time:.2f}s, should be < 1.0s"
    
    def test_memory_usage_reasonable(self):
        """Test that foundation doesn't use excessive memory."""
        import sys
        
        # Get initial memory usage
        initial_modules = len(sys.modules)
        
        # Import foundation
        try:
            from orchestrator.langgraph import LangGraphWorkflowEngine
            from orchestrator.langgraph.state_schemas import RunbookState, AgentState, ToolState
        except ImportError:
            pytest.skip("LangGraph not available for memory testing")
        
        # Check module count increase
        final_modules = len(sys.modules)
        modules_added = final_modules - initial_modules
        
        # Should not add excessive modules (arbitrary limit of 50)
        assert modules_added < 50, f"Added {modules_added} modules, should be < 50"

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"]) 