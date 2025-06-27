"""
Integration test for FeedbackHandler - demonstrates complete workflow
"""

import asyncio
from feedback_handler import FeedbackHandler

async def demo_feedback_system():
    """Demonstrate the complete feedback system workflow."""
    print("🚀 **USER FEEDBACK SYSTEM DEMO**")
    print("="*50)
    
    # Initialize handler
    handler = FeedbackHandler()
    
    print("\n1️⃣ **Testing /list-workflows (empty)**")
    result = await handler.process_feedback_command(
        command="list-workflows",
        user_id="demo_user",
        message_content="/list-workflows",
        context={}
    )
    print(f"✅ Success: {result['success']}")
    print(f"📝 Response:\n{result['message']}")
    
    print("\n2️⃣ **Testing /save-workflow**")
    result = await handler.process_feedback_command(
        command="save-workflow",
        user_id="demo_user", 
        message_content="/save-workflow My First Workflow",
        context={"conversation_id": "demo_conv_1"}
    )
    print(f"✅ Success: {result['success']}")
    print(f"📝 Response:\n{result['message']}")
    saved_workflow_id = result.get('workflow_id')
    
    print("\n3️⃣ **Testing /suggest**")
    result = await handler.process_feedback_command(
        command="feedback",
        user_id="demo_user",
        message_content="/suggest This system is amazing! Could be faster though.",
        context={"conversation_id": "demo_conv_1"}
    )
    print(f"✅ Success: {result['success']}")
    print(f"📝 Response:\n{result['message']}")
    
    print("\n4️⃣ **Testing /improve (with workflow)**")
    # First set up a workflow in context
    result = await handler.process_feedback_command(
        command="improve",
        user_id="demo_user",
        message_content="/improve Add error handling and make it 50% faster",
        context={"conversation_id": "demo_conv_1"}
    )
    print(f"✅ Success: {result['success']}")
    print(f"📝 Response:\n{result['message']}")
    
    print("\n5️⃣ **Testing /list-workflows (with data)**")
    result = await handler.process_feedback_command(
        command="list-workflows",
        user_id="demo_user",
        message_content="/list-workflows",
        context={}
    )
    print(f"✅ Success: {result['success']}")
    print(f"📝 Response:\n{result['message']}")
    
    print("\n6️⃣ **System Analytics**")
    user_workflows = handler.get_user_workflows("demo_user")
    user_feedback = handler.get_user_feedback("demo_user")
    
    print(f"📊 **User Metrics:**")
    print(f"  • Total Workflows: {len(user_workflows)}")
    print(f"  • Total Feedback: {len(user_feedback)}")
    print(f"  • Last Workflow: {handler.last_workflow_by_user.get('demo_user', 'None')}")
    
    print(f"\n🗂️ **Workflow Details:**")
    for workflow in user_workflows:
        print(f"  • {workflow.name} (v{workflow.version})")
        print(f"    📝 {workflow.description}")
        print(f"    🎯 {len(workflow.steps)} steps")
        print(f"    🤖 Agents: {', '.join(workflow.agents_used)}")
    
    print(f"\n💬 **Feedback Details:**")
    for feedback in user_feedback:
        print(f"  • {feedback.feedback_type.value}")
        print(f"    📝 {feedback.content[:60]}...")
        print(f"    📊 Priority: {feedback.priority}, Status: {feedback.status}")
    
    print("\n7️⃣ **Command Help**")
    result = await handler.process_feedback_command(
        command="unknown",
        user_id="demo_user",
        message_content="/unknown",
        context={}
    )
    print(f"✅ Help shown: {not result['success']}")
    print(f"📝 Help Response:\n{result['message']}")
    
    print("\n🎉 **DEMO COMPLETED!**")
    print("="*50)
    print("🌟 **Key Features Demonstrated:**")
    print("  ✅ Workflow saving and versioning")
    print("  ✅ Workflow improvement through natural language")
    print("  ✅ User feedback collection and storage")
    print("  ✅ Workflow listing and management")
    print("  ✅ Command help and error handling")
    print("  ✅ User analytics and metrics")
    
    await handler.close()

if __name__ == "__main__":
    asyncio.run(demo_feedback_system()) 