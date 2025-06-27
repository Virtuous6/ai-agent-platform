"""
Test suite for FeedbackHandler
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from feedback_handler import FeedbackHandler, UserWorkflow, UserFeedback, FeedbackType

async def test_basic_functionality():
    """Basic functionality test."""
    print("🧪 Testing FeedbackHandler basic functionality...")
    
    # Create handler
    handler = FeedbackHandler()
    
    # Test command processing
    result = await handler.process_feedback_command(
        command="list-workflows",
        user_id="test_user",
        message_content="/list-workflows",
        context={}
    )
    
    print(f"✅ List workflows result: {result['success']}")
    print(f"📝 Message: {result['message'][:100]}...")
    
    # Test feedback storage
    result = await handler.process_feedback_command(
        command="feedback",
        user_id="test_user",
        message_content="/suggest This is a test feedback",
        context={"conversation_id": "test"}
    )
    
    print(f"✅ Feedback result: {result['success']}")
    print(f"📝 Feedback ID: {result.get('feedback_id', 'None')}")
    
    # Test improve command (without workflow)
    result = await handler.process_feedback_command(
        command="improve",
        user_id="test_user",
        message_content="/improve Make this faster",
        context={}
    )
    
    print(f"✅ Improve (no workflow) result: {result['success']}")
    print(f"📝 Expected failure message: {result['message'][:60]}...")
    
    # Test save workflow command
    result = await handler.process_feedback_command(
        command="save-workflow",
        user_id="test_user",
        message_content="/save-workflow Test Workflow",
        context={"conversation_id": "test"}
    )
    
    print(f"✅ Save workflow result: {result['success']}")
    print(f"📝 Workflow saved: {'workflow_id' in result}")
    
    print("🎉 Basic functionality tests completed!")
    
    # Show summary
    print(f"\n📊 Summary:")
    print(f"  Workflows: {len(handler.user_workflows)}")
    print(f"  Feedback: {len(handler.user_feedback)}")
    
    await handler.close()

class TestFeedbackHandler:
    """Test cases for FeedbackHandler functionality."""
    
    @pytest.fixture
    async def handler(self):
        """Create FeedbackHandler instance for testing."""
        mock_db_logger = Mock()
        mock_orchestrator = Mock()
        handler = FeedbackHandler(
            db_logger=mock_db_logger,
            orchestrator=mock_orchestrator
        )
        return handler
    
    @pytest.mark.asyncio
    async def test_improve_command_no_workflow(self, handler):
        """Test /improve command when no workflow exists."""
        result = await handler.process_feedback_command(
            command="improve",
            user_id="test_user",
            message_content="/improve Make this faster",
            context={}
        )
        
        assert result["success"] is False
        assert "No recent workflow found" in result["message"]
    
    @pytest.mark.asyncio
    async def test_improve_command_with_workflow(self, handler):
        """Test /improve command with existing workflow."""
        # Mock existing workflow
        handler._extract_last_workflow = AsyncMock(return_value={
            "name": "Test Workflow",
            "description": "A test workflow",
            "steps": [{"step": 1, "action": "Test action", "agent": "general"}],
            "agents_used": ["general"],
            "triggers": ["test"],
            "success_criteria": ["completed"]
        })
        
        result = await handler.process_feedback_command(
            command="improve",
            user_id="test_user",
            message_content="/improve Make this faster and more efficient",
            context={"conversation_id": "test_conv"}
        )
        
        assert result["success"] is True
        assert "Workflow Improved!" in result["message"]
        assert "workflow_id" in result
    
    @pytest.mark.asyncio
    async def test_save_workflow_command(self, handler):
        """Test /save-workflow command."""
        # Mock workflow creation
        handler._create_workflow_from_conversation = AsyncMock(return_value={
            "name": "My Saved Workflow",
            "description": "User saved workflow",
            "steps": [{"step": 1, "action": "Test", "agent": "general"}],
            "agents_used": ["general"],
            "triggers": ["user request"],
            "success_criteria": ["completed"]
        })
        
        result = await handler.process_feedback_command(
            command="save-workflow",
            user_id="test_user",
            message_content="/save-workflow My Awesome Workflow",
            context={"conversation_id": "test_conv"}
        )
        
        assert result["success"] is True
        assert "Workflow Saved!" in result["message"]
        assert "My Awesome Workflow" in result["message"]
        assert "workflow_id" in result
    
    @pytest.mark.asyncio
    async def test_list_workflows_empty(self, handler):
        """Test /list-workflows command with no workflows."""
        result = await handler.process_feedback_command(
            command="list-workflows",
            user_id="test_user",
            message_content="/list-workflows",
            context={}
        )
        
        assert result["success"] is True
        assert "No Saved Workflows" in result["message"]
    
    @pytest.mark.asyncio
    async def test_list_workflows_with_data(self, handler):
        """Test /list-workflows command with existing workflows."""
        # Add some test workflows
        test_workflow = UserWorkflow(
            id="test_id",
            user_id="test_user",
            name="Test Workflow",
            description="A test workflow for testing",
            steps=[{"step": 1, "action": "Test"}],
            triggers=["test"],
            agents_used=["general"],
            success_criteria=["success"],
            version="1.0.0",
            usage_count=5,
            success_rate=0.9
        )
        handler.user_workflows["test_id"] = test_workflow
        
        result = await handler.process_feedback_command(
            command="list-workflows",
            user_id="test_user",
            message_content="/list-workflows",
            context={}
        )
        
        assert result["success"] is True
        assert "Your Saved Workflows" in result["message"]
        assert "Test Workflow" in result["message"]
        assert "✅" in result["message"]  # Success indicator
    
    @pytest.mark.asyncio
    async def test_feedback_command(self, handler):
        """Test /suggest command."""
        result = await handler.process_feedback_command(
            command="feedback",
            user_id="test_user",
            message_content="/suggest The system could be faster",
            context={"conversation_id": "test_conv"}
        )
        
        assert result["success"] is True
        assert "Feedback Received!" in result["message"]
        assert "feedback_id" in result
    
    @pytest.mark.asyncio
    async def test_unknown_command(self, handler):
        """Test unknown command handling."""
        result = await handler.process_feedback_command(
            command="unknown",
            user_id="test_user",
            message_content="/unknown test",
            context={}
        )
        
        assert result["success"] is False
        assert "Unknown command" in result["message"]
    
    def test_extract_improvement_request(self, handler):
        """Test improvement request extraction."""
        result = handler._extract_improvement_request("/improve Make this faster")
        assert result == "Make this faster"
        
        result = handler._extract_improvement_request("/improve")
        assert "better and more efficient" in result
    
    def test_extract_workflow_name(self, handler):
        """Test workflow name extraction."""
        result = handler._extract_workflow_name("/save-workflow My Workflow")
        assert result == "My Workflow"
        
        result = handler._extract_workflow_name("/save-workflow")
        assert result is None
    
    def test_extract_feedback_content(self, handler):
        """Test feedback content extraction."""
        result = handler._extract_feedback_content("/suggest This is my feedback")
        assert result == "This is my feedback"
    
    @pytest.mark.asyncio
    async def test_save_new_workflow(self, handler):
        """Test saving a new workflow."""
        workflow_data = {
            "name": "Test Workflow",
            "description": "Test description",
            "steps": [{"step": 1, "action": "test"}],
            "triggers": ["test"],
            "agents_used": ["general"],
            "success_criteria": ["success"]
        }
        
        workflow_id = await handler._save_new_workflow(workflow_data, "test_user")
        
        assert workflow_id in handler.user_workflows
        assert handler.user_workflows[workflow_id].name == "Test Workflow"
        assert handler.user_workflows[workflow_id].version == "1.0.0"
        assert handler.last_workflow_by_user["test_user"] == workflow_id
    
    @pytest.mark.asyncio
    async def test_store_feedback(self, handler):
        """Test storing user feedback."""
        feedback_id = await handler._store_feedback(
            "test_user", 
            "Test feedback content",
            {"conversation_id": "test_conv"}
        )
        
        assert feedback_id in handler.user_feedback
        assert handler.user_feedback[feedback_id].content == "Test feedback content"
        assert handler.user_feedback[feedback_id].status == "pending"
    
    def test_format_improvement_response(self, handler):
        """Test improvement response formatting."""
        improved_workflow = {
            "name": "Improved Workflow",
            "improvements_made": ["Added error handling", "Improved efficiency"],
            "steps": [{"step": 1}, {"step": 2}]
        }
        
        response = handler._format_improvement_response(improved_workflow)
        
        assert "Workflow Improved!" in response
        assert "Improved Workflow" in response
        assert "Added error handling" in response
        assert "Improved efficiency" in response
        assert "2" in response  # Number of steps
    
    def test_get_user_workflows(self, handler):
        """Test getting workflows for a user."""
        # Add test workflows
        workflow1 = UserWorkflow(
            id="id1", user_id="user1", name="Workflow 1", description="Test",
            steps=[], triggers=[], agents_used=[], success_criteria=[], version="1.0.0"
        )
        workflow2 = UserWorkflow(
            id="id2", user_id="user2", name="Workflow 2", description="Test",
            steps=[], triggers=[], agents_used=[], success_criteria=[], version="1.0.0"
        )
        workflow3 = UserWorkflow(
            id="id3", user_id="user1", name="Workflow 3", description="Test",
            steps=[], triggers=[], agents_used=[], success_criteria=[], version="1.0.0"
        )
        
        handler.user_workflows = {"id1": workflow1, "id2": workflow2, "id3": workflow3}
        
        user1_workflows = handler.get_user_workflows("user1")
        assert len(user1_workflows) == 2
        assert all(w.user_id == "user1" for w in user1_workflows)
    
    def test_get_user_feedback(self, handler):
        """Test getting feedback for a user."""
        # Add test feedback
        feedback1 = UserFeedback(
            id="f1", user_id="user1", feedback_type=FeedbackType.IMPROVEMENT_REQUEST,
            content="Test feedback 1", priority=3, status="pending"
        )
        feedback2 = UserFeedback(
            id="f2", user_id="user2", feedback_type=FeedbackType.PERFORMANCE_FEEDBACK,
            content="Test feedback 2", priority=3, status="pending"
        )
        
        handler.user_feedback = {"f1": feedback1, "f2": feedback2}
        
        user1_feedback = handler.get_user_feedback("user1")
        assert len(user1_feedback) == 1
        assert user1_feedback[0].content == "Test feedback 1"

if __name__ == "__main__":
    # Run tests
    asyncio.run(test_basic_functionality()) 