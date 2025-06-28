#!/usr/bin/env python3
"""
Agent Management CLI

Command-line interface for full CRUD operations on AI agents.
View, create, edit, delete agents with their prompts and settings.
"""

import asyncio
import json
import sys
import os
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent_manager import AgentManager, AgentProfile, AgentStatus

class AgentCLI:
    """CLI interface for agent management."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.agent_manager = AgentManager()
    
    async def run(self):
        """Run the CLI interface."""
        print("🤖 AI Agent Management CLI")
        print("=" * 50)
        
        while True:
            print("\nChoose an action:")
            print("1. 👥 List all agents")
            print("2. 👀 View agent details")
            print("3. ➕ Create new agent")
            print("4. ✏️  Edit agent")
            print("5. 🔄 Change agent status")
            print("6. 📋 Duplicate agent")
            print("7. 🗑️  Delete agent")
            print("8. 📊 Agent statistics")
            print("9. 🚪 Exit")
            
            choice = input("\nEnter your choice (1-9): ").strip()
            
            try:
                if choice == "1":
                    await self.list_agents()
                elif choice == "2":
                    await self.view_agent()
                elif choice == "3":
                    await self.create_agent()
                elif choice == "4":
                    await self.edit_agent()
                elif choice == "5":
                    await self.change_status()
                elif choice == "6":
                    await self.duplicate_agent()
                elif choice == "7":
                    await self.delete_agent()
                elif choice == "8":
                    await self.show_stats()
                elif choice == "9":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def list_agents(self):
        """List all agents."""
        print("\n👥 All Agents")
        print("-" * 30)
        
        # Get filters
        status_filter = input("Filter by status (leave empty for all): ").strip()
        specialty_filter = input("Filter by specialty (leave empty for all): ").strip()
        
        filters = {
            'status': status_filter if status_filter else None,
            'specialty': specialty_filter if specialty_filter else None,
            'limit': 50
        }
        
        result = await self.agent_manager.list_agents(**filters)
        
        if not result['success']:
            print(f"❌ Failed to load agents: {result['error']}")
            return
        
        agents = result['agents']
        
        if not agents:
            print("📭 No agents found.")
            return
        
        print(f"\nFound {len(agents)} agents:")
        print()
        
        for i, agent in enumerate(agents, 1):
            status_emoji = {
                'active': '🟢',
                'inactive': '🔴', 
                'draft': '📝',
                'archived': '📦'
            }.get(agent['status'], '❓')
            
            print(f"{i:2}. {status_emoji} {agent['name']}")
            print(f"    ID: {agent['id'][:8]}...")
            print(f"    Specialty: {agent['specialty']}")
            print(f"    Status: {agent['status']}")
            print(f"    Model: {agent['model']} (temp: {agent['temperature']})")
            print(f"    Interactions: {agent.get('total_interactions', 0)}")
            print(f"    Updated: {agent['updated_at'][:19] if agent['updated_at'] else 'N/A'}")
            print()
    
    async def view_agent(self):
        """View detailed agent information."""
        print("\n👀 View Agent Details")
        print("-" * 25)
        
        agent_id = input("Enter agent ID (first 8 characters): ").strip()
        if not agent_id:
            print("❌ Agent ID required.")
            return
        
        # Find agent by partial ID
        agents_result = await self.agent_manager.list_agents(limit=100)
        if not agents_result['success']:
            print(f"❌ Failed to load agents: {agents_result['error']}")
            return
        
        # Find matching agent
        matching_agent = None
        for agent in agents_result['agents']:
            if agent['id'].startswith(agent_id):
                matching_agent = agent
                break
        
        if not matching_agent:
            print(f"❌ No agent found with ID starting with '{agent_id}'")
            return
        
        # Display full agent details
        agent = matching_agent
        print(f"\n🤖 Agent: {agent['name']}")
        print("=" * 50)
        
        print(f"📋 Basic Information:")
        print(f"  • ID: {agent['id']}")
        print(f"  • Name: {agent['name']}")
        print(f"  • Specialty: {agent['specialty']}")
        print(f"  • Description: {agent.get('description', 'No description')}")
        print(f"  • Status: {agent['status']}")
        print(f"  • Created: {agent.get('created_at', 'Unknown')[:19]}")
        print(f"  • Updated: {agent.get('updated_at', 'Unknown')[:19]}")
        
        print(f"\n⚙️ Configuration:")
        print(f"  • Model: {agent['model']}")
        print(f"  • Temperature: {agent['temperature']}")
        print(f"  • Max Tokens: {agent['max_tokens']}")
        print(f"  • Complexity: {agent.get('complexity_level', 'medium')}")
        
        print(f"\n📊 Performance:")
        print(f"  • Total Interactions: {agent.get('total_interactions', 0)}")
        print(f"  • Success Rate: {agent.get('success_rate', 0):.1%}")
        print(f"  • Avg Response Time: {agent.get('avg_response_time', 0):.2f}s")
        print(f"  • User Satisfaction: {agent.get('user_satisfaction', 0):.1%}")
        
        # Show prompts
        show_prompts = input("\n👁️ View prompts? (y/n): ").strip().lower()
        if show_prompts == 'y':
            print(f"\n📝 System Prompt:")
            print("-" * 20)
            print(agent['system_prompt'])
            
            if agent.get('tool_decision_guidance'):
                print(f"\n🔧 Tool Decision Guidance:")
                print("-" * 30)
                print(agent['tool_decision_guidance'])
            
            if agent.get('communication_style'):
                print(f"\n💬 Communication Style:")
                print("-" * 25)
                print(agent['communication_style'])
    
    async def create_agent(self):
        """Create a new agent."""
        print("\n➕ Create New Agent")
        print("-" * 20)
        
        # Collect basic information
        name = input("Agent name: ").strip()
        if not name:
            print("❌ Agent name is required.")
            return
        
        specialty = input("Specialty (e.g., technical, research, customer_support): ").strip()
        if not specialty:
            print("❌ Specialty is required.")
            return
        
        description = input("Description (optional): ").strip()
        
        # Configuration
        print("\n⚙️ Configuration:")
        model = input("Model (default: gpt-3.5-turbo): ").strip() or "gpt-3.5-turbo"
        
        try:
            temperature = float(input("Temperature (0.0-2.0, default: 0.7): ").strip() or "0.7")
            if temperature < 0.0 or temperature > 2.0:
                print("❌ Temperature must be between 0.0 and 2.0")
                return
        except ValueError:
            print("❌ Invalid temperature value")
            return
        
        try:
            max_tokens = int(input("Max tokens (default: 500): ").strip() or "500")
            if max_tokens < 1 or max_tokens > 4000:
                print("❌ Max tokens must be between 1 and 4000")
                return
        except ValueError:
            print("❌ Invalid max tokens value")
            return
        
        # System prompt
        print("\n📝 System Prompt:")
        print("Enter the system prompt (required). End with a line containing only '---':")
        
        system_prompt_lines = []
        while True:
            line = input()
            if line.strip() == "---":
                break
            system_prompt_lines.append(line)
        
        system_prompt = "\n".join(system_prompt_lines).strip()
        if not system_prompt:
            print("❌ System prompt is required.")
            return
        
        # Optional advanced prompts
        print("\n🔧 Advanced Configuration (optional):")
        
        add_advanced = input("Add tool decision guidance? (y/n): ").strip().lower()
        tool_decision_guidance = ""
        if add_advanced == 'y':
            print("Enter tool decision guidance. End with '---':")
            guidance_lines = []
            while True:
                line = input()
                if line.strip() == "---":
                    break
                guidance_lines.append(line)
            tool_decision_guidance = "\n".join(guidance_lines).strip()
        
        # Create agent profile
        profile = AgentProfile(
            name=name,
            specialty=specialty,
            description=description,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            tool_decision_guidance=tool_decision_guidance,
            status=AgentStatus.DRAFT
        )
        
        # Create agent
        result = await self.agent_manager.create_agent(profile)
        
        if result['success']:
            print(f"\n✅ Agent '{name}' created successfully!")
            print(f"   Agent ID: {result['agent_id']}")
            
            # Option to activate
            activate = input("\n🚀 Activate agent now? (y/n): ").strip().lower()
            if activate == 'y':
                activate_result = await self.agent_manager.update_agent_status(
                    result['agent_id'], 
                    AgentStatus.ACTIVE
                )
                if activate_result['success']:
                    print("✅ Agent activated!")
                else:
                    print(f"❌ Failed to activate: {activate_result['error']}")
        else:
            print(f"❌ Failed to create agent: {result['error']}")
    
    async def edit_agent(self):
        """Edit an existing agent."""
        print("\n✏️ Edit Agent")
        print("-" * 15)
        
        agent_id = input("Enter agent ID (first 8 characters): ").strip()
        if not agent_id:
            print("❌ Agent ID required.")
            return
        
        # Find agent
        agents_result = await self.agent_manager.list_agents(limit=100)
        if not agents_result['success']:
            print(f"❌ Failed to load agents: {agents_result['error']}")
            return
        
        matching_agent = None
        for agent in agents_result['agents']:
            if agent['id'].startswith(agent_id):
                matching_agent = agent
                break
        
        if not matching_agent:
            print(f"❌ No agent found with ID starting with '{agent_id}'")
            return
        
        agent = matching_agent
        print(f"\n📝 Editing: {agent['name']}")
        
        # Show editable fields
        print("\nWhat would you like to edit?")
        print("1. Name")
        print("2. Description") 
        print("3. Temperature")
        print("4. Max Tokens")
        print("5. System Prompt")
        print("6. Tool Decision Guidance")
        print("7. Multiple fields")
        
        choice = input("\nChoose (1-7): ").strip()
        
        updates = {}
        
        if choice == "1":
            new_name = input(f"New name (current: {agent['name']}): ").strip()
            if new_name:
                updates['name'] = new_name
        
        elif choice == "2":
            new_desc = input(f"New description (current: {agent.get('description', 'None')}): ").strip()
            updates['description'] = new_desc
        
        elif choice == "3":
            try:
                new_temp = float(input(f"New temperature (current: {agent['temperature']}): ").strip())
                if 0.0 <= new_temp <= 2.0:
                    updates['temperature'] = new_temp
                else:
                    print("❌ Temperature must be between 0.0 and 2.0")
                    return
            except ValueError:
                print("❌ Invalid temperature value")
                return
        
        elif choice == "4":
            try:
                new_tokens = int(input(f"New max tokens (current: {agent['max_tokens']}): ").strip())
                if 1 <= new_tokens <= 4000:
                    updates['max_tokens'] = new_tokens
                else:
                    print("❌ Max tokens must be between 1 and 4000")
                    return
            except ValueError:
                print("❌ Invalid max tokens value")
                return
        
        elif choice == "5":
            print("Enter new system prompt. End with '---':")
            prompt_lines = []
            while True:
                line = input()
                if line.strip() == "---":
                    break
                prompt_lines.append(line)
            new_prompt = "\n".join(prompt_lines).strip()
            if new_prompt:
                updates['system_prompt'] = new_prompt
        
        elif choice == "6":
            print("Enter new tool decision guidance. End with '---':")
            guidance_lines = []
            while True:
                line = input()
                if line.strip() == "---":
                    break
                guidance_lines.append(line)
            new_guidance = "\n".join(guidance_lines).strip()
            updates['tool_decision_guidance'] = new_guidance
        
        elif choice == "7":
            print("Multiple field editing not implemented yet. Please edit one field at a time.")
            return
        
        else:
            print("❌ Invalid choice")
            return
        
        if not updates:
            print("❌ No updates provided")
            return
        
        # Apply updates
        result = await self.agent_manager.update_agent(agent['id'], updates)
        
        if result['success']:
            print(f"✅ Agent updated successfully!")
            print(f"   Updated fields: {', '.join(result['updated_fields'])}")
        else:
            print(f"❌ Failed to update agent: {result['error']}")
    
    async def change_status(self):
        """Change agent status."""
        print("\n🔄 Change Agent Status")
        print("-" * 25)
        
        agent_id = input("Enter agent ID (first 8 characters): ").strip()
        if not agent_id:
            print("❌ Agent ID required.")
            return
        
        # Find agent
        agents_result = await self.agent_manager.list_agents(limit=100)
        if not agents_result['success']:
            print(f"❌ Failed to load agents: {agents_result['error']}")
            return
        
        matching_agent = None
        for agent in agents_result['agents']:
            if agent['id'].startswith(agent_id):
                matching_agent = agent
                break
        
        if not matching_agent:
            print(f"❌ No agent found with ID starting with '{agent_id}'")
            return
        
        agent = matching_agent
        print(f"\n🤖 Agent: {agent['name']}")
        print(f"Current status: {agent['status']}")
        
        print("\nAvailable statuses:")
        statuses = list(AgentStatus)
        for i, status in enumerate(statuses, 1):
            print(f"{i}. {status.value}")
        
        try:
            choice = int(input(f"\nChoose new status (1-{len(statuses)}): ").strip())
            if 1 <= choice <= len(statuses):
                new_status = statuses[choice - 1]
                
                result = await self.agent_manager.update_agent_status(agent['id'], new_status)
                
                if result['success']:
                    print(f"✅ Status changed to: {new_status.value}")
                else:
                    print(f"❌ Failed to change status: {result['error']}")
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Invalid input")
    
    async def duplicate_agent(self):
        """Duplicate an agent."""
        print("\n📋 Duplicate Agent")
        print("-" * 20)
        
        agent_id = input("Enter agent ID to duplicate (first 8 characters): ").strip()
        if not agent_id:
            print("❌ Agent ID required.")
            return
        
        new_name = input("Enter name for the new agent: ").strip()
        if not new_name:
            print("❌ New agent name required.")
            return
        
        # Find source agent
        agents_result = await self.agent_manager.list_agents(limit=100)
        if not agents_result['success']:
            print(f"❌ Failed to load agents: {agents_result['error']}")
            return
        
        source_agent = None
        for agent in agents_result['agents']:
            if agent['id'].startswith(agent_id):
                source_agent = agent
                break
        
        if not source_agent:
            print(f"❌ No agent found with ID starting with '{agent_id}'")
            return
        
        result = await self.agent_manager.duplicate_agent(source_agent['id'], new_name)
        
        if result['success']:
            print(f"✅ Agent duplicated successfully!")
            print(f"   Original: {source_agent['name']}")
            print(f"   New: {new_name}")
            print(f"   New ID: {result['agent_id']}")
        else:
            print(f"❌ Failed to duplicate agent: {result['error']}")
    
    async def delete_agent(self):
        """Delete an agent."""
        print("\n🗑️ Delete Agent")
        print("-" * 17)
        
        agent_id = input("Enter agent ID to delete (first 8 characters): ").strip()
        if not agent_id:
            print("❌ Agent ID required.")
            return
        
        # Find agent
        agents_result = await self.agent_manager.list_agents(limit=100)
        if not agents_result['success']:
            print(f"❌ Failed to load agents: {agents_result['error']}")
            return
        
        target_agent = None
        for agent in agents_result['agents']:
            if agent['id'].startswith(agent_id):
                target_agent = agent
                break
        
        if not target_agent:
            print(f"❌ No agent found with ID starting with '{agent_id}'")
            return
        
        print(f"\n⚠️ You are about to delete: {target_agent['name']}")
        print("Choose deletion type:")
        print("1. Soft delete (archive)")
        print("2. Hard delete (permanent)")
        
        choice = input("Choose (1-2): ").strip()
        
        if choice == "1":
            soft_delete = True
        elif choice == "2":
            soft_delete = False
            confirm = input("⚠️ Permanent deletion cannot be undone. Type 'DELETE' to confirm: ").strip()
            if confirm != "DELETE":
                print("❌ Deletion cancelled.")
                return
        else:
            print("❌ Invalid choice")
            return
        
        result = await self.agent_manager.delete_agent(target_agent['id'], soft_delete=soft_delete)
        
        if result['success']:
            action = "archived" if soft_delete else "deleted permanently"
            print(f"✅ Agent {action}!")
        else:
            print(f"❌ Failed to delete agent: {result['error']}")
    
    async def show_stats(self):
        """Show agent statistics."""
        print("\n📊 Agent Statistics")
        print("-" * 22)
        
        result = await self.agent_manager.get_agent_stats()
        
        if not result['success']:
            print(f"❌ Failed to load statistics: {result['error']}")
            return
        
        stats = result['stats']
        
        print(f"📈 Overview:")
        print(f"  • Total Agents: {stats['total_agents']}")
        print(f"  • Active: {stats['active_agents']}")
        print(f"  • Inactive: {stats['inactive_agents']}")
        print(f"  • Draft: {stats['draft_agents']}")
        print(f"  • Archived: {stats['archived_agents']}")
        
        print(f"\n🎯 Performance:")
        print(f"  • Total Interactions: {stats['total_interactions']}")
        if stats['total_agents'] > 0:
            print(f"  • Average Success Rate: {stats['avg_success_rate']:.1%}")
            print(f"  • Average Temperature: {stats['avg_temperature']:.2f}")
        
        if stats['specialties']:
            print(f"\n🏷️ Specialties:")
            for specialty, count in stats['specialties'].items():
                print(f"  • {specialty}: {count}")
        
        if stats['models']:
            print(f"\n🤖 Models:")
            for model, count in stats['models'].items():
                print(f"  • {model}: {count}")

async def main():
    """Main entry point."""
    cli = AgentCLI()
    await cli.run()

if __name__ == "__main__":
    asyncio.run(main()) 