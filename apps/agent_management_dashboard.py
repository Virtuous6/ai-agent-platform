"""
Agent Management Dashboard

Web interface for full CRUD operations on AI agents.
View, create, edit, delete agents with their prompts, settings, and performance data.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

# Import our agent management system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent_manager import AgentManager, AgentProfile, AgentStatus
from storage.supabase import SupabaseLogger

logger = logging.getLogger(__name__)

class AgentManagementDashboard:
    """Web dashboard for complete agent management."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.agent_manager = AgentManager()
        
        # Initialize session state
        if 'selected_agent' not in st.session_state:
            st.session_state.selected_agent = None
        if 'editing_agent' not in st.session_state:
            st.session_state.editing_agent = False
        if 'agents_data' not in st.session_state:
            st.session_state.agents_data = []
    
    def run(self):
        """Run the dashboard."""
        st.set_page_config(
            page_title="AI Agent Management",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🤖 AI Agent Management Dashboard")
        st.markdown("**Complete CRUD operations for your AI agents**")
        
        # Navigation menu
        with st.sidebar:
            selected = option_menu(
                "Agent Management",
                ["📊 Overview", "👥 All Agents", "➕ Create Agent", "⚙️ Settings"],
                icons=['graph-up', 'people', 'plus-circle', 'gear'],
                menu_icon="robot",
                default_index=0
            )
        
        # Route to appropriate page
        if selected == "📊 Overview":
            self.show_overview()
        elif selected == "👥 All Agents":
            self.show_all_agents()
        elif selected == "➕ Create Agent":
            self.show_create_agent()
        elif selected == "⚙️ Settings":
            self.show_settings()
    
    def show_overview(self):
        """Show agent overview and statistics."""
        st.header("📊 Agent Overview")
        
        # Load agent stats
        with st.spinner("Loading agent statistics..."):
            stats_result = asyncio.run(self.agent_manager.get_agent_stats())
        
        if stats_result['success']:
            stats = stats_result['stats']
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Agents", stats['total_agents'])
            with col2:
                st.metric("Active Agents", stats['active_agents'])
            with col3:
                st.metric("Total Interactions", stats['total_interactions'])
            with col4:
                if stats['total_agents'] > 0:
                    st.metric("Avg Success Rate", f"{stats['avg_success_rate']:.1%}")
                else:
                    st.metric("Avg Success Rate", "0%")
            
            # Agent status breakdown
            st.subheader("Agent Status Distribution")
            status_data = {
                'Active': stats['active_agents'],
                'Inactive': stats['inactive_agents'], 
                'Draft': stats['draft_agents'],
                'Archived': stats['archived_agents']
            }
            
            # Remove zero values for cleaner chart
            status_data = {k: v for k, v in status_data.items() if v > 0}
            
            if status_data:
                st.bar_chart(status_data)
            else:
                st.info("No agents created yet. Go to 'Create Agent' to get started!")
            
            # Specialties breakdown
            if stats['specialties']:
                st.subheader("Agent Specialties")
                specialties_df = pd.DataFrame(
                    list(stats['specialties'].items()),
                    columns=['Specialty', 'Count']
                )
                st.dataframe(specialties_df, use_container_width=True)
            
            # Models breakdown
            if stats['models']:
                st.subheader("Models in Use")
                models_df = pd.DataFrame(
                    list(stats['models'].items()),
                    columns=['Model', 'Count']
                )
                st.dataframe(models_df, use_container_width=True)
        
        else:
            st.error(f"Failed to load statistics: {stats_result['error']}")
    
    def show_all_agents(self):
        """Show all agents with CRUD operations."""
        st.header("👥 All Agents")
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_filter = st.selectbox(
                "Filter by Status",
                options=["All", "active", "inactive", "draft", "archived"],
                index=0
            )
        
        with col2:
            specialty_filter = st.text_input("Filter by Specialty")
        
        with col3:
            search_filter = st.text_input("Search Name/Description")
        
        with col4:
            if st.button("🔄 Refresh"):
                st.session_state.agents_data = []
        
        # Load agents
        with st.spinner("Loading agents..."):
            filters = {
                'status': None if status_filter == "All" else status_filter,
                'specialty': specialty_filter if specialty_filter else None,
                'search': search_filter if search_filter else None,
                'limit': 100
            }
            
            agents_result = asyncio.run(self.agent_manager.list_agents(**filters))
        
        if agents_result['success']:
            agents = agents_result['agents']
            
            if not agents:
                st.info("No agents found matching your criteria.")
                return
            
            # Display agents table
            st.subheader(f"Found {len(agents)} agents")
            
            # Prepare data for display
            display_data = []
            for agent in agents:
                display_data.append({
                    'ID': agent['id'][:8] + '...',
                    'Name': agent['name'],
                    'Specialty': agent['specialty'],
                    'Status': agent['status'],
                    'Model': agent['model'],
                    'Temperature': agent['temperature'],
                    'Interactions': agent.get('total_interactions', 0),
                    'Success Rate': f"{agent.get('success_rate', 0):.1%}",
                    'Last Updated': agent['updated_at'][:19] if agent['updated_at'] else 'N/A'
                })
            
            # Show as DataFrame with selection
            df = pd.DataFrame(display_data)
            
            # Agent selection and actions
            for i, agent in enumerate(agents):
                with st.expander(f"🤖 {agent['name']} ({agent['specialty']}) - {agent['status'].upper()}", expanded=False):
                    
                    # Agent details
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Basic Info:**")
                        st.write(f"• **ID:** {agent['id']}")
                        st.write(f"• **Specialty:** {agent['specialty']}")
                        st.write(f"• **Description:** {agent.get('description', 'No description')}")
                        st.write(f"• **Status:** {agent['status']}")
                        st.write(f"• **Created:** {agent.get('created_at', 'Unknown')[:19]}")
                    
                    with col2:
                        st.write("**Configuration:**")
                        st.write(f"• **Model:** {agent['model']}")
                        st.write(f"• **Temperature:** {agent['temperature']}")
                        st.write(f"• **Max Tokens:** {agent['max_tokens']}")
                        st.write(f"• **Complexity:** {agent.get('complexity_level', 'medium')}")
                    
                    # Performance metrics
                    st.write("**Performance:**")
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    with perf_col1:
                        st.metric("Interactions", agent.get('total_interactions', 0))
                    with perf_col2:
                        st.metric("Success Rate", f"{agent.get('success_rate', 0):.1%}")
                    with perf_col3:
                        st.metric("Avg Response Time", f"{agent.get('avg_response_time', 0):.2f}s")
                    
                    # Show prompts
                    if st.button(f"👁️ View Prompts", key=f"view_prompts_{i}"):
                        st.text_area("System Prompt", agent['system_prompt'], height=150, key=f"system_prompt_view_{i}")
                        if agent.get('tool_decision_guidance'):
                            st.text_area("Tool Decision Guidance", agent['tool_decision_guidance'], height=100, key=f"tool_guidance_view_{i}")
                    
                    # Action buttons
                    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
                    
                    with action_col1:
                        if st.button(f"✏️ Edit", key=f"edit_{i}"):
                            st.session_state.selected_agent = agent
                            st.session_state.editing_agent = True
                            st.rerun()
                    
                    with action_col2:
                        new_status = "inactive" if agent['status'] == 'active' else "active"
                        if st.button(f"🔄 {'Deactivate' if agent['status'] == 'active' else 'Activate'}", key=f"toggle_status_{i}"):
                            result = asyncio.run(
                                self.agent_manager.update_agent_status(
                                    agent['id'], 
                                    AgentStatus(new_status)
                                )
                            )
                            if result['success']:
                                st.success(f"Agent {new_status}!")
                                st.rerun()
                            else:
                                st.error(f"Failed to update status: {result['error']}")
                    
                    with action_col3:
                        if st.button(f"📋 Duplicate", key=f"duplicate_{i}"):
                            new_name = f"{agent['name']} Copy"
                            result = asyncio.run(
                                self.agent_manager.duplicate_agent(agent['id'], new_name)
                            )
                            if result['success']:
                                st.success(f"Agent duplicated as '{new_name}'!")
                                st.rerun()
                            else:
                                st.error(f"Failed to duplicate: {result['error']}")
                    
                    with action_col4:
                        if st.button(f"🗑️ Archive", key=f"archive_{i}"):
                            if st.button(f"❗ Confirm Archive", key=f"confirm_archive_{i}"):
                                result = asyncio.run(
                                    self.agent_manager.delete_agent(agent['id'], soft_delete=True)
                                )
                                if result['success']:
                                    st.success("Agent archived!")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to archive: {result['error']}")
            
        else:
            st.error(f"Failed to load agents: {agents_result['error']}")
        
        # Show edit form if editing
        if st.session_state.editing_agent and st.session_state.selected_agent:
            self.show_edit_agent_form()
    
    def show_edit_agent_form(self):
        """Show agent editing form."""
        agent = st.session_state.selected_agent
        
        st.header(f"✏️ Edit Agent: {agent['name']}")
        
        with st.form("edit_agent_form"):
            # Basic info
            st.subheader("Basic Information")
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Agent Name", value=agent['name'])
                specialty = st.text_input("Specialty", value=agent['specialty'])
            
            with col2:
                description = st.text_area("Description", value=agent.get('description', ''))
                status = st.selectbox(
                    "Status",
                    options=[s.value for s in AgentStatus],
                    index=[s.value for s in AgentStatus].index(agent['status'])
                )
            
            # Configuration
            st.subheader("Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                model = st.selectbox(
                    "Model",
                    options=["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4", "gpt-4-0125-preview"],
                    index=["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4", "gpt-4-0125-preview"].index(
                        agent['model'] if agent['model'] in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4", "gpt-4-0125-preview"] else "gpt-3.5-turbo"
                    )
                )
            
            with col2:
                temperature = st.slider("Temperature", 0.0, 2.0, value=float(agent['temperature']), step=0.1)
            
            with col3:
                max_tokens = st.number_input("Max Tokens", min_value=1, max_value=4000, value=int(agent['max_tokens']))
            
            # Prompts
            st.subheader("Prompts")
            system_prompt = st.text_area(
                "System Prompt",
                value=agent['system_prompt'],
                height=200,
                help="The main instruction that defines the agent's role and behavior"
            )
            
            tool_decision_guidance = st.text_area(
                "Tool Decision Guidance",
                value=agent.get('tool_decision_guidance', ''),
                height=150,
                help="Instructions for how the agent should decide when and which tools to use"
            )
            
            communication_style = st.text_area(
                "Communication Style",
                value=agent.get('communication_style', ''),
                height=100,
                help="How the agent should communicate with users"
            )
            
            tool_selection_criteria = st.text_area(
                "Tool Selection Criteria",
                value=agent.get('tool_selection_criteria', ''),
                height=100,
                help="Criteria for selecting the best tools for tasks"
            )
            
            # Form buttons
            col1, col2 = st.columns(2)
            
            with col1:
                submitted = st.form_submit_button("💾 Save Changes", type="primary")
            
            with col2:
                cancelled = st.form_submit_button("❌ Cancel")
            
            if submitted:
                # Prepare updates
                updates = {
                    'name': name,
                    'specialty': specialty,
                    'description': description,
                    'status': status,
                    'model': model,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'system_prompt': system_prompt,
                    'tool_decision_guidance': tool_decision_guidance,
                    'communication_style': communication_style,
                    'tool_selection_criteria': tool_selection_criteria
                }
                
                # Update agent
                result = asyncio.run(
                    self.agent_manager.update_agent(agent['id'], updates)
                )
                
                if result['success']:
                    st.success("✅ Agent updated successfully!")
                    st.session_state.editing_agent = False
                    st.session_state.selected_agent = None
                    st.rerun()
                else:
                    st.error(f"❌ Failed to update agent: {result['error']}")
            
            if cancelled:
                st.session_state.editing_agent = False
                st.session_state.selected_agent = None
                st.rerun()
    
    def show_create_agent(self):
        """Show create agent form."""
        st.header("➕ Create New Agent")
        
        with st.form("create_agent_form"):
            # Basic info
            st.subheader("Basic Information")
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Agent Name*", placeholder="e.g., Technical Assistant")
                specialty = st.text_input("Specialty*", placeholder="e.g., technical, research, customer_support")
            
            with col2:
                description = st.text_area("Description", placeholder="Brief description of what this agent does")
            
            # Configuration
            st.subheader("Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                model = st.selectbox(
                    "Model",
                    options=["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4", "gpt-4-0125-preview"],
                    index=0
                )
            
            with col2:
                temperature = st.slider("Temperature", 0.0, 2.0, value=0.7, step=0.1)
            
            with col3:
                max_tokens = st.number_input("Max Tokens", min_value=1, max_value=4000, value=500)
            
            # Prompts
            st.subheader("Prompts")
            system_prompt = st.text_area(
                "System Prompt*",
                placeholder="You are a helpful AI assistant specialized in...",
                height=200,
                help="The main instruction that defines the agent's role and behavior"
            )
            
            # Optional advanced prompts
            with st.expander("📝 Advanced Prompt Configuration", expanded=False):
                tool_decision_guidance = st.text_area(
                    "Tool Decision Guidance",
                    placeholder="When deciding which tools to use...",
                    height=150
                )
                
                communication_style = st.text_area(
                    "Communication Style",
                    placeholder="Be professional, clear, and helpful...",
                    height=100
                )
                
                tool_selection_criteria = st.text_area(
                    "Tool Selection Criteria", 
                    placeholder="Choose tools based on...",
                    height=100
                )
            
            # Submit button
            submitted = st.form_submit_button("🚀 Create Agent", type="primary")
            
            if submitted:
                # Validate required fields
                if not name or not specialty or not system_prompt:
                    st.error("❌ Please fill in all required fields (marked with *)")
                    return
                
                # Create agent profile
                profile = AgentProfile(
                    name=name,
                    specialty=specialty,
                    description=description,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    tool_decision_guidance=tool_decision_guidance or "",
                    communication_style=communication_style or "",
                    tool_selection_criteria=tool_selection_criteria or "",
                    status=AgentStatus.DRAFT  # Start as draft
                )
                
                # Create agent
                result = asyncio.run(self.agent_manager.create_agent(profile))
                
                if result['success']:
                    st.success(f"✅ Agent '{name}' created successfully!")
                    st.balloons()
                    
                    # Option to activate immediately
                    if st.button("🚀 Activate Agent Now"):
                        activate_result = asyncio.run(
                            self.agent_manager.update_agent_status(
                                result['agent_id'], 
                                AgentStatus.ACTIVE
                            )
                        )
                        if activate_result['success']:
                            st.success("Agent activated!")
                else:
                    st.error(f"❌ Failed to create agent: {result['error']}")
    
    def show_settings(self):
        """Show settings and system information."""
        st.header("⚙️ Settings")
        
        st.subheader("System Information")
        st.write("**Agent Management System v1.0**")
        st.write("Complete CRUD operations for AI agents with dynamic prompt management.")
        
        st.subheader("Database Status")
        try:
            # Test database connection
            result = asyncio.run(self.agent_manager.list_agents(limit=1))
            if result['success']:
                st.success("✅ Database connection: OK")
            else:
                st.error(f"❌ Database connection: {result['error']}")
        except Exception as e:
            st.error(f"❌ Database connection: {str(e)}")

def main():
    """Main entry point."""
    dashboard = AgentManagementDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 