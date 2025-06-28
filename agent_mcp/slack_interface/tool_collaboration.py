"""
Slack Interface for Tool Collaboration

Enables users to respond to tool creation requests and provide the information
needed to build missing tools that agents have requested.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from mcp.dynamic_tool_builder import DynamicToolBuilder
from agents.universal_agent import UniversalAgent

logger = logging.getLogger(__name__)

class ToolCollaborationInterface:
    """
    Slack interface for tool collaboration between users and agents.
    
    Handles:
    1. Notifying users about tool creation requests
    2. Collecting user input for tool building  
    3. Showing tool creation progress
    4. Completing tasks once tools are ready
    """
    
    def __init__(self, 
                 slack_app: AsyncApp,
                 dynamic_tool_builder: DynamicToolBuilder):
        """Initialize tool collaboration interface."""
        self.slack_app = slack_app
        self.dynamic_tool_builder = dynamic_tool_builder
        self.active_agents: Dict[str, UniversalAgent] = {}
        
        # Register Slack command handlers
        self._register_handlers()
        
        logger.info("🤝 Tool Collaboration Interface initialized")
    
    def register_agent(self, agent: UniversalAgent):
        """Register an enhanced agent for tool collaboration."""
        self.active_agents[agent.agent_id] = agent
        logger.info(f"Registered agent for tool collaboration: {agent.specialty}")
    
    def _register_handlers(self):
        """Register Slack command and interaction handlers."""
        
        @self.slack_app.command("/tool-status")
        async def handle_tool_status(ack, body, say):
            await ack()
            await self._show_tool_status(body, say)
        
        @self.slack_app.command("/tool-help")
        async def handle_tool_help(ack, body, say):
            await ack()
            await self._show_collaboration_requests(body, say)
        
        @self.slack_app.action("provide_tool_info")
        async def handle_provide_tool_info(ack, body, say):
            await ack()
            await self._show_tool_input_modal(body)
        
        @self.slack_app.view("tool_input_modal")
        async def handle_tool_input_submission(ack, body, say):
            await ack()
            await self._process_tool_input(body, say)
        
        @self.slack_app.action("dismiss_tool_request")
        async def handle_dismiss_request(ack, body, say):
            await ack()
            await self._dismiss_tool_request(body, say)
    
    async def notify_tool_request(self, user_id: str, request_data: Dict[str, Any]):
        """
        Notify user about a new tool creation request.
        
        This is called when an agent detects a tool gap and needs user help.
        """
        try:
            client = self.slack_app.client
            
            # Create notification blocks
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🛠️ Agent Needs Your Help Building a Tool!"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Agent:* {request_data.get('agent_specialty', 'Unknown')}\n*Tool Needed:* {request_data.get('capability_needed', 'Unknown capability')}\n*Why:* {request_data.get('description', 'No description')}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Your Original Request:*\n> {request_data.get('original_message', 'No message')[:200]}..."
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "Help Build Tool"
                            },
                            "action_id": "provide_tool_info",
                            "value": request_data.get('request_id', ''),
                            "style": "primary"
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "Not Now"
                            },
                            "action_id": "dismiss_tool_request",
                            "value": request_data.get('request_id', '')
                        }
                    ]
                }
            ]
            
            # Send notification
            await client.chat_postMessage(
                channel=user_id,
                text="🛠️ Agent needs help building a tool",
                blocks=blocks
            )
            
            logger.info(f"Sent tool request notification to user {user_id}")
            
        except Exception as e:
            logger.error(f"Error sending tool request notification: {e}")
    
    async def _show_tool_status(self, body: Dict[str, Any], say):
        """Show status of all tool requests for the user."""
        try:
            user_id = body['user_id']
            
            # Get pending requests for this user
            pending_requests = []
            for agent in self.active_agents.values():
                request = await agent.check_mcp_tool_requests_status(user_id)
                if request:
                    pending_requests.append({
                        **request,
                        'agent_specialty': agent.specialty
                    })
            
            if not pending_requests:
                await say("✅ No pending tool requests!")
                return
            
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🔧 Tool Requests Status ({len(pending_requests)})"
                    }
                },
                {"type": "divider"}
            ]
            
            for request in pending_requests:
                blocks.extend([
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Tool:* {request['tool_name']}"
                            },
                            {
                                "type": "mrkdwn", 
                                "text": f"*Agent:* {request['agent_specialty']}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Status:* {request.get('priority', 'Medium')} priority"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Help Needed:* {', '.join(request['help_needed'])}"
                            }
                        ]
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "Provide Information"
                                },
                                "action_id": "provide_tool_info",
                                "value": request['request_id'],
                                "style": "primary"
                            }
                        ]
                    },
                    {"type": "divider"}
                ])
            
            await say(blocks=blocks)
            
        except Exception as e:
            logger.error(f"Error showing tool status: {e}")
            await say("❌ Error retrieving tool status")
    
    async def _show_collaboration_requests(self, body: Dict[str, Any], say):
        """Show detailed collaboration requests for the user."""
        try:
            user_id = body['user_id']
            
            # Get collaboration request from dynamic tool builder
            collaboration_request = await self.dynamic_tool_builder.get_user_collaboration_request(user_id)
            
            if not collaboration_request:
                await say("✅ No collaboration requests pending!")
                return
            
            # Create detailed help message
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🤝 Tool Building Collaboration"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Tool Name:* {collaboration_request['tool_name']}\n*Description:* {collaboration_request['description']}"
                    }
                }
            ]
            
            if 'original_task' in collaboration_request:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Your Original Request:*\n> {collaboration_request['original_task']['message'][:300]}..."
                    }
                })
            
            # Show what help is needed
            help_needed = collaboration_request['help_needed']
            help_text = "I need your help with:\n"
            
            for item in help_needed:
                if item == "api_credentials":
                    help_text += "• API credentials or connection information\n"
                elif item == "tool_specification": 
                    help_text += "• Detailed specification of how the tool should work\n"
                elif item == "testing_data":
                    help_text += "• Sample data to test the tool with\n"
                else:
                    help_text += f"• {item}\n"
            
            blocks.extend([
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": help_text
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "Provide Information"
                            },
                            "action_id": "provide_tool_info",
                            "value": collaboration_request['request_id'],
                            "style": "primary"
                        }
                    ]
                }
            ])
            
            await say(blocks=blocks)
            
        except Exception as e:
            logger.error(f"Error showing collaboration requests: {e}")
            await say("❌ Error retrieving collaboration requests")
    
    async def _show_tool_input_modal(self, body: Dict[str, Any]):
        """Show modal for user to provide tool building information."""
        try:
            trigger_id = body['trigger_id']
            request_id = body['actions'][0]['value']
            
            # Get collaboration request details
            user_id = body['user']['id']
            collaboration_request = await self.dynamic_tool_builder.get_user_collaboration_request(user_id)
            
            if not collaboration_request or collaboration_request['request_id'] != request_id:
                return
            
            # Build modal based on what help is needed
            blocks = []
            help_needed = collaboration_request['help_needed']
            
            if "api_credentials" in help_needed:
                blocks.extend([
                    {
                        "type": "input",
                        "block_id": "api_credentials_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "api_url",
                            "placeholder": {
                                "type": "plain_text",
                                "text": "https://api.example.com"
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "API URL"
                        }
                    },
                    {
                        "type": "input",
                        "block_id": "api_key_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "api_key",
                            "placeholder": {
                                "type": "plain_text", 
                                "text": "your-api-key-here"
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "API Key (optional)"
                        },
                        "optional": True
                    }
                ])
            
            if "tool_specification" in help_needed:
                blocks.append({
                    "type": "input",
                    "block_id": "specification_block",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "specification",
                        "multiline": True,
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Describe how the tool should work, what parameters it needs, etc."
                        }
                    },
                    "label": {
                        "type": "plain_text",
                        "text": "Tool Specification"
                    }
                })
            
            if "testing_data" in help_needed:
                blocks.append({
                    "type": "input",
                    "block_id": "testing_block",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "test_data",
                        "multiline": True,
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Sample data or examples to test the tool with"
                        }
                    },
                    "label": {
                        "type": "plain_text",
                        "text": "Testing Data"
                    },
                    "optional": True
                })
            
            modal = {
                "type": "modal",
                "callback_id": "tool_input_modal",
                "title": {
                    "type": "plain_text",
                    "text": "Build Tool"
                },
                "submit": {
                    "type": "plain_text",
                    "text": "Submit"
                },
                "close": {
                    "type": "plain_text",
                    "text": "Cancel"
                },
                "private_metadata": request_id,
                "blocks": blocks
            }
            
            client = self.slack_app.client
            await client.views_open(
                trigger_id=trigger_id,
                view=modal
            )
            
        except Exception as e:
            logger.error(f"Error showing tool input modal: {e}")
    
    async def _process_tool_input(self, body: Dict[str, Any], say):
        """Process user input for tool building."""
        try:
            request_id = body['view']['private_metadata']
            values = body['view']['state']['values']
            
            # Extract user input
            user_input = {}
            
            # API credentials
            if 'api_credentials_block' in values:
                api_url = values['api_credentials_block']['api_url']['value']
                api_key = values.get('api_key_block', {}).get('api_key', {}).get('value')
                
                user_input['api_credentials'] = {
                    'url': api_url,
                    'key': api_key
                }
            
            # Tool specification
            if 'specification_block' in values:
                spec = values['specification_block']['specification']['value']
                user_input['tool_specification'] = {
                    'user_description': spec
                }
            
            # Testing data
            if 'testing_block' in values:
                test_data = values['testing_block']['test_data']['value']
                if test_data:
                    user_input['testing_data'] = test_data
            
            # Submit to dynamic tool builder
            result = await self.dynamic_tool_builder.handle_user_input(request_id, user_input)
            
            if result['success']:
                # Notify user of success
                user_id = body['user']['id']
                client = self.slack_app.client
                
                await client.chat_postMessage(
                    channel=user_id,
                    text="✅ Thanks! I'm now building your tool...",
                    blocks=[
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "✅ *Tool building started!*\n\nI've received your information and I'm now building the tool. I'll let you know when it's ready and complete your original task!"
                            }
                        }
                    ]
                )
                
                # Monitor tool building progress
                asyncio.create_task(self._monitor_tool_progress(request_id, user_id))
                
            else:
                await say(f"❌ Error processing your input: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Error processing tool input: {e}")
            await say("❌ Error processing your tool information")
    
    async def _monitor_tool_progress(self, request_id: str, user_id: str):
        """Monitor tool building progress and notify user when complete."""
        try:
            client = self.slack_app.client
            
            # Check progress every 30 seconds for up to 10 minutes
            for _ in range(20):
                await asyncio.sleep(30)
                
                request = self.dynamic_tool_builder.active_requests.get(request_id)
                if not request:
                    break
                
                if request.status.value == "completed":
                    # Tool is ready - notify user and complete task
                    await client.chat_postMessage(
                        channel=user_id,
                        text="🎉 Tool ready! Completing your task...",
                        blocks=[
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": "🎉 *Tool is ready!*\n\nI've successfully built your tool and I'm now completing your original task..."
                                }
                            }
                        ]
                    )
                    
                    # Find agent and complete task
                    for agent in self.active_agents.values():
                        result = await agent.handle_tool_ready(request_id)
                        if result:
                            # Send final result
                            await client.chat_postMessage(
                                channel=user_id,
                                text="✅ Task completed!",
                                blocks=[
                                    {
                                        "type": "section", 
                                        "text": {
                                            "type": "mrkdwn",
                                            "text": f"✅ *Task completed with new tool!*\n\n{result['response']}"
                                        }
                                    }
                                ]
                            )
                            break
                    
                    break
                    
                elif request.status.value == "failed":
                    # Tool building failed
                    await client.chat_postMessage(
                        channel=user_id,
                        text="❌ Tool building failed",
                        blocks=[
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": "❌ *Tool building failed*\n\nI wasn't able to build the tool successfully. Let me try to help you with a different approach."
                                }
                            }
                        ]
                    )
                    break
            
        except Exception as e:
            logger.error(f"Error monitoring tool progress: {e}")
    
    async def _dismiss_tool_request(self, body: Dict[str, Any], say):
        """Handle user dismissing a tool request."""
        request_id = body['actions'][0]['value']
        await say(f"👍 Tool request {request_id} dismissed. Feel free to ask me anything else!")
    
    async def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get statistics about tool collaboration."""
        total_requests = len(self.dynamic_tool_builder.active_requests)
        pending_collaboration = len([
            r for r in self.dynamic_tool_builder.active_requests.values()
            if r.status.value == "user_input_needed"
        ])
        
        return {
            "total_tool_requests": total_requests,
            "pending_user_collaboration": pending_collaboration,
            "active_agents": len(self.active_agents),
            "agent_specialties": [agent.specialty for agent in self.active_agents.values()]
        } 