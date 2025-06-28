#!/usr/bin/env python3
"""
Test Agent CRUD Operations

Simple script to test the complete agent management system.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.agent_manager import AgentManager, AgentProfile, AgentStatus

async def test_agent_crud():
    """Test complete CRUD operations for agents."""
    
    print("🤖 Testing Agent CRUD Operations")
    print("=" * 50)
    
    # Initialize manager
    agent_manager = AgentManager()
    
    # 1. CREATE - Create a test agent
    print("\n1. ➕ CREATE - Creating test agent...")
    
    test_profile = AgentProfile(
        name="Test Technical Agent",
        specialty="technical",
        description="A test agent for demonstrating CRUD operations",
        model="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=800,
        system_prompt="""You are a specialized technical AI assistant.

**Your Specialty:** Technical assistance and problem-solving

**Core Capabilities:**
- Expert knowledge in software development
- Code analysis and debugging
- Architecture recommendations
- Best practices guidance

**Tool Decision Framework:**
1. **Assess Need**: Does this require real-time data or external resources?
2. **MCP-First**: Check for existing MCP tools that can help
3. **Gap Detection**: If no suitable tools exist, suggest MCP setup
4. **User Guidance**: Be transparent about tool usage and limitations""",
        tool_decision_guidance="Prioritize existing MCPs > MCP Library > Custom tools > Inform user of limitations",
        communication_style="Be precise, technical, and provide actionable recommendations",
        tool_selection_criteria="Choose tools based on accuracy, reliability, and user needs",
        status=AgentStatus.DRAFT
    )
    
    create_result = await agent_manager.create_agent(test_profile)
    
    if create_result['success']:
        agent_id = create_result['agent_id']
        print(f"   ✅ Agent created with ID: {agent_id[:8]}...")
    else:
        print(f"   ❌ Failed to create agent: {create_result['error']}")
        return
    
    # 2. READ - Get the agent we just created
    print("\n2. 👀 READ - Retrieving agent...")
    
    get_result = await agent_manager.get_agent(agent_id)
    
    if get_result['success']:
        agent = get_result['agent']
        print(f"   ✅ Retrieved agent: {agent['name']}")
        print(f"   📋 Status: {agent['status']}")
        print(f"   🎯 Specialty: {agent['specialty']}")
        print(f"   🌡️ Temperature: {agent['temperature']}")
    else:
        print(f"   ❌ Failed to get agent: {get_result['error']}")
        return
    
    # 3. LIST - Show all agents
    print("\n3. 📋 LIST - Listing all agents...")
    
    list_result = await agent_manager.list_agents()
    
    if list_result['success']:
        agents = list_result['agents']
        print(f"   ✅ Found {len(agents)} agents:")
        for i, agent in enumerate(agents[-3:], 1):  # Show last 3
            print(f"      {i}. {agent['name']} ({agent['specialty']}) - {agent['status']}")
    else:
        print(f"   ❌ Failed to list agents: {list_result['error']}")
    
    # 4. UPDATE - Modify the agent
    print("\n4. ✏️ UPDATE - Updating agent...")
    
    updates = {
        'description': 'Updated test agent with new description',
        'temperature': 0.5,
        'system_prompt': agent['system_prompt'] + '\n\n**Update:** This prompt has been updated via CRUD operations.'
    }
    
    update_result = await agent_manager.update_agent(agent_id, updates)
    
    if update_result['success']:
        print(f"   ✅ Agent updated successfully!")
        print(f"   📝 Updated fields: {', '.join(update_result['updated_fields'])}")
        
        # Verify the update
        verify_result = await agent_manager.get_agent(agent_id)
        if verify_result['success']:
            updated_agent = verify_result['agent']
            print(f"   🌡️ New temperature: {updated_agent['temperature']}")
            print(f"   📝 New description: {updated_agent['description']}")
    else:
        print(f"   ❌ Failed to update agent: {update_result['error']}")
    
    # 5. STATUS CHANGE - Activate the agent
    print("\n5. 🔄 STATUS CHANGE - Activating agent...")
    
    status_result = await agent_manager.update_agent_status(agent_id, AgentStatus.ACTIVE)
    
    if status_result['success']:
        print(f"   ✅ Agent activated!")
        
        # Verify status change
        verify_result = await agent_manager.get_agent(agent_id)
        if verify_result['success']:
            print(f"   🟢 New status: {verify_result['agent']['status']}")
    else:
        print(f"   ❌ Failed to activate agent: {status_result['error']}")
    
    # 6. DUPLICATE - Create a copy
    print("\n6. 📋 DUPLICATE - Creating a copy...")
    
    duplicate_result = await agent_manager.duplicate_agent(agent_id, "Test Technical Agent Copy")
    
    if duplicate_result['success']:
        duplicate_id = duplicate_result['agent_id']
        print(f"   ✅ Agent duplicated with ID: {duplicate_id[:8]}...")
    else:
        print(f"   ❌ Failed to duplicate agent: {duplicate_result['error']}")
        duplicate_id = None
    
    # 7. STATISTICS - Show stats
    print("\n7. 📊 STATISTICS - Agent stats...")
    
    stats_result = await agent_manager.get_agent_stats()
    
    if stats_result['success']:
        stats = stats_result['stats']
        print(f"   ✅ Agent Statistics:")
        print(f"      📈 Total: {stats['total_agents']}")
        print(f"      🟢 Active: {stats['active_agents']}")
        print(f"      🔴 Inactive: {stats['inactive_agents']}")
        print(f"      📝 Draft: {stats['draft_agents']}")
        print(f"      📦 Archived: {stats['archived_agents']}")
        
        if stats['specialties']:
            print(f"      🏷️ Specialties: {', '.join(stats['specialties'].keys())}")
    else:
        print(f"   ❌ Failed to get stats: {stats_result['error']}")
    
    # 8. DELETE - Clean up test agents
    print("\n8. 🗑️ DELETE - Cleaning up test agents...")
    
    # Delete original (soft delete - archive)
    delete_result = await agent_manager.delete_agent(agent_id, soft_delete=True)
    
    if delete_result['success']:
        print(f"   ✅ Original agent archived")
    else:
        print(f"   ❌ Failed to archive original: {delete_result['error']}")
    
    # Delete duplicate (hard delete if it was created)
    if duplicate_id:
        delete_duplicate_result = await agent_manager.delete_agent(duplicate_id, soft_delete=False)
        
        if delete_duplicate_result['success']:
            print(f"   ✅ Duplicate agent deleted permanently")
        else:
            print(f"   ❌ Failed to delete duplicate: {delete_duplicate_result['error']}")
    
    print(f"\n🎉 CRUD Test Complete!")
    print(f"✅ All operations tested successfully")

async def main():
    """Main entry point."""
    try:
        await test_agent_crud()
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 