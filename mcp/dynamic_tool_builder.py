"""
Dynamic Tool Builder

Enables agents to request, build, and validate new MCP tools with user collaboration.
When agents identify missing capabilities, they can build tools, find them, or get user help.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from database.supabase_logger import SupabaseLogger
from events.event_bus import EventBus, EventType
from .tool_registry import MCPToolRegistry
from .security_sandbox import MCPSecuritySandbox
from .credential_store import CredentialManager
from .mcp_discovery_engine import MCPDiscoveryEngine, MCPMatch

logger = logging.getLogger(__name__)

class ToolRequestStatus(Enum):
    """Status of tool requests."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    BUILDING = "building"
    TESTING = "testing"
    USER_INPUT_NEEDED = "user_input_needed"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ToolGap:
    """Represents a detected tool gap."""
    gap_id: str
    agent_id: str
    capability_needed: str
    description: str
    user_message: str
    context: Dict[str, Any]
    detected_at: datetime
    priority: float  # 0.0-1.0
    suggested_solutions: List[str]
    mcp_solutions: List[MCPMatch] = None  # Available MCP solutions
    
@dataclass
class ToolRequest:
    """Represents a request for a new tool."""
    request_id: str
    gap_id: str
    agent_id: str
    user_id: str
    tool_name: str
    tool_description: str
    tool_specification: Dict[str, Any]
    status: ToolRequestStatus
    user_collaboration_needed: List[str]
    created_at: datetime
    completed_at: Optional[datetime] = None
    validation_results: Optional[Dict[str, Any]] = None

class DynamicToolBuilder:
    """
    Enables agents to request, build, and validate new tools dynamically.
    
    Process:
    1. Agent detects capability gap
    2. System tries to build tool automatically
    3. If needed, requests user collaboration
    4. Validates and deploys tool
    5. Makes available to all agents
    """
    
    def __init__(self, 
                 supabase_logger: SupabaseLogger,
                 tool_registry: MCPToolRegistry,
                 event_bus: EventBus,
                 security_sandbox: MCPSecuritySandbox,
                 connection_manager=None):
        """Initialize the dynamic tool builder."""
        self.supabase_logger = supabase_logger
        self.tool_registry = tool_registry
        self.event_bus = event_bus
        self.security_sandbox = security_sandbox
        self.credential_manager = CredentialManager()
        
        # Initialize MCP-first discovery engine
        self.mcp_discovery = MCPDiscoveryEngine(tool_registry, connection_manager)
        
        # Active requests and gaps
        self.active_gaps: Dict[str, ToolGap] = {}
        self.active_requests: Dict[str, ToolRequest] = {}
        
        # Tool templates for common patterns
        self.tool_templates = self._load_tool_templates()
        
        logger.info("🔧 Dynamic Tool Builder initialized with MCP-first approach")
    
    async def detect_tool_gap(self, agent_id: str, message: str, 
                            context: Dict[str, Any]) -> Optional[ToolGap]:
        """
        Detect if the agent needs a tool it doesn't have.
        
        MCP-FIRST APPROACH: Prioritizes finding MCP solutions before building custom tools.
        
        This is the key method that agents call when they encounter
        a task they can't complete with existing tools.
        """
        try:
            # Step 1: Analyze message for capability needs
            capabilities_needed = await self._analyze_capability_needs(message, context)
            
            if not capabilities_needed:
                return None
            
            # Step 2: Check if agent already has these capabilities
            available_tools = await self.tool_registry.get_available_tools()
            available_capabilities = [tool['description'] for tool in available_tools]
            
            for capability in capabilities_needed:
                if not self._capability_exists(capability, available_capabilities):
                    
                    # Step 3: Search for MCP solutions FIRST
                    logger.info(f"🔍 Searching for MCP solutions for: {capability['name']}")
                    mcp_solutions = await self.mcp_discovery.find_mcp_solutions(
                        capability['name'], 
                        capability['description'], 
                        {"message": message, "context": context}
                    )
                    
                    # Create tool gap with MCP solutions
                    gap = ToolGap(
                        gap_id=str(uuid.uuid4()),
                        agent_id=agent_id,
                        capability_needed=capability['name'],
                        description=capability['description'],
                        user_message=message,
                        context=context,
                        detected_at=datetime.utcnow(),
                        priority=capability.get('priority', 0.5),
                        suggested_solutions=capability.get('solutions', []),
                        mcp_solutions=mcp_solutions
                    )
                    
                    self.active_gaps[gap.gap_id] = gap
                    
                    # Determine best solution approach
                    solution_type = "mcp" if mcp_solutions else "custom"
                    best_solution = mcp_solutions[0] if mcp_solutions else None
                    
                    # Publish gap detection event with MCP info
                    if self.event_bus:
                        await self.event_bus.publish(
                            EventType.TOOL_GAP_DETECTED,
                            {
                                "gap_id": gap.gap_id,
                                "agent_id": agent_id,
                                "capability": capability['name'],
                                "priority": gap.priority,
                                "suggested_solutions": gap.suggested_solutions,
                                "mcp_solutions_found": len(mcp_solutions),
                                "best_solution_type": solution_type,
                                "best_mcp_solution": {
                                    "name": best_solution.capability.name,
                                    "match_type": best_solution.match_type,
                                    "match_score": best_solution.match_score
                                } if best_solution else None
                            },
                            source=agent_id
                        )
                    
                    if mcp_solutions:
                        logger.info(f"📦 Found {len(mcp_solutions)} MCP solutions for '{capability['name']}'")
                        logger.info(f"   Best: {mcp_solutions[0].capability.name} (score: {mcp_solutions[0].match_score:.2f})")
                    else:
                        logger.info(f"🔧 No MCP solutions found for '{capability['name']}' - will build custom tool")
                    
                    return gap
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting tool gap: {e}")
            return None
    
    async def request_tool_creation(self, gap: ToolGap, user_id: str) -> str:
        """
        Create a request to build a new tool for the detected gap.
        
        Returns:
            request_id for tracking progress
        """
        try:
            # Generate tool specification
            tool_spec = await self._generate_tool_specification(gap)
            
            # Create tool request
            request = ToolRequest(
                request_id=str(uuid.uuid4()),
                gap_id=gap.gap_id,
                agent_id=gap.agent_id,
                user_id=user_id,
                tool_name=self._generate_tool_name(gap.capability_needed),
                tool_description=gap.description,
                tool_specification=tool_spec,
                status=ToolRequestStatus.ANALYZING,
                user_collaboration_needed=[],
                created_at=datetime.utcnow()
            )
            
            self.active_requests[request.request_id] = request
            
            # Log to database
            await self._log_tool_request(request)
            
            # Start tool building process
            asyncio.create_task(self._process_tool_request(request))
            
            logger.info(f"🛠️ Tool creation requested: {request.tool_name} for gap {gap.gap_id}")
            return request.request_id
            
        except Exception as e:
            logger.error(f"Error creating tool request: {e}")
            raise
    
    async def get_user_collaboration_request(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get pending collaboration requests for a user.
        
        This is what the Slack interface would call to show users
        what help is needed with tool creation.
        """
        pending_requests = []
        
        for request in self.active_requests.values():
            if (request.user_id == user_id and 
                request.status == ToolRequestStatus.USER_INPUT_NEEDED):
                
                pending_requests.append({
                    "request_id": request.request_id,
                    "tool_name": request.tool_name,
                    "description": request.tool_description,
                    "help_needed": request.user_collaboration_needed,
                    "created_at": request.created_at.isoformat(),
                    "priority": self.active_gaps.get(request.gap_id, ToolGap(
                        "", "", "", "", "", {}, datetime.utcnow(), 0.5, []
                    )).priority
                })
        
        return pending_requests[0] if pending_requests else None
    
    async def handle_user_input(self, request_id: str, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle user input for tool creation.
        
        This is called when users provide the information needed
        to complete tool creation.
        """
        request = self.active_requests.get(request_id)
        if not request:
            return {"success": False, "error": "Request not found"}
        
        try:
            # Process different types of user input
            if "api_credentials" in user_input:
                request.tool_specification["credentials"] = user_input["api_credentials"]
            
            if "tool_specification" in user_input:
                request.tool_specification.update(user_input["tool_specification"])
            
            if "testing_data" in user_input:
                request.tool_specification["test_data"] = user_input["testing_data"]
            
            # Clear user collaboration needed
            request.user_collaboration_needed = []
            request.status = ToolRequestStatus.BUILDING
            
            # Continue building process
            asyncio.create_task(self._process_tool_request(request))
            
            return {"success": True, "message": "User input processed successfully"}
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return {"success": False, "error": str(e)}
    
    async def _analyze_capability_needs(self, message: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze message to determine what capabilities are needed.
        
        This uses LLM to understand what external tools/APIs would help.
        """
        from langchain_openai import ChatOpenAI
        
        analysis_prompt = f"""Analyze this user message to identify what capabilities/tools would be needed:

User Message: "{message}"

Look for needs like:
- API calls to external services
- Database operations  
- File processing
- Web scraping
- Complex calculations
- Data transformations

Return JSON array of capabilities:
[
  {{
    "name": "capability_name",
    "description": "what this capability does",
    "priority": 0.8,
    "type": "api|database|file_processing|web_scraping|calculation|other",
    "solutions": ["suggestion1", "suggestion2"]
  }}
]

Only return capabilities that require external tools. Return empty array if none needed."""

        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
            response = await llm.ainvoke(analysis_prompt)
            
            # Parse JSON response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.strip("```json").strip("```").strip()
            elif content.startswith("```"):
                content = content.strip("```").strip()
            
            capabilities = json.loads(content)
            return capabilities if isinstance(capabilities, list) else []
            
        except Exception as e:
            logger.warning(f"Capability analysis failed: {e}")
            return []
    
    def _capability_exists(self, capability: Dict[str, Any], available_capabilities: List[str]) -> bool:
        """Check if a capability already exists in available tools."""
        capability_name = capability['name'].lower()
        capability_desc = capability['description'].lower()
        
        for available in available_capabilities:
            available_lower = available.lower()
            if (capability_name in available_lower or 
                any(word in available_lower for word in capability_desc.split()[:3])):
                return True
        
        return False
    
    async def _generate_tool_specification(self, gap: ToolGap) -> Dict[str, Any]:
        """Generate a detailed specification for the needed tool."""
        from langchain_openai import ChatOpenAI
        
        spec_prompt = f"""Generate a tool specification for this capability:

Capability: {gap.capability_needed}
Description: {gap.description}
User Context: {gap.user_message}
Priority: {gap.priority}

Generate a tool specification with:
{{
  "function_name": "tool_function_name",
  "parameters": {{
    "param1": {{"type": "string", "description": "param description", "required": true}}
  }},
  "output_format": {{
    "success": "boolean",
    "result": "any", 
    "error": "string"
  }},
  "implementation": {{
    "approach": "how to implement this",
    "apis_needed": ["api1", "api2"],
    "complexity": "low|medium|high"
  }},
  "template": "rest_api|database|file_processing|calculation"
}}"""

        try:
            llm = ChatOpenAI(model="gpt-4", temperature=0.3)
            response = await llm.ainvoke(spec_prompt)
            
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.strip("```json").strip("```").strip()
            elif content.startswith("```"):
                content = content.strip("```").strip()
            
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Tool specification generation failed: {e}")
            return self._get_default_tool_spec(gap)
    
    async def _process_tool_request(self, request: ToolRequest):
        """Process a tool creation request through the MCP-first pipeline."""
        try:
            gap = self.active_gaps.get(request.gap_id)
            
            # Step 1: Try MCP solutions FIRST
            if gap and gap.mcp_solutions:
                logger.info(f"🚀 Attempting MCP solutions for {request.tool_name}")
                mcp_success = await self._attempt_mcp_solutions(request, gap.mcp_solutions)
                if mcp_success:
                    return
            
            # Step 2: Try to find existing custom tool
            existing_tool = await self._search_existing_tools(request)
            if existing_tool:
                await self._complete_request_with_existing_tool(request, existing_tool)
                return
            
            # Step 3: Try to build custom tool automatically
            request.status = ToolRequestStatus.BUILDING
            built_tool = await self._attempt_automatic_build(request)
            
            if built_tool:
                # Step 4: Test the custom tool
                request.status = ToolRequestStatus.TESTING
                validation_results = await self._validate_tool(built_tool)
                
                if validation_results['success']:
                    # Step 5: Deploy custom tool
                    await self._deploy_tool(request, built_tool)
                    await self._complete_request(request)
                    return
                else:
                    logger.warning(f"Tool validation failed: {validation_results}")
            
            # Step 6: Request user collaboration (MCP setup or custom tool help)
            await self._request_user_collaboration(request)
            
        except Exception as e:
            logger.error(f"Error processing tool request {request.request_id}: {e}")
            request.status = ToolRequestStatus.FAILED
            await self._notify_failure(request, str(e))
    
    def _load_tool_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load tool templates for common patterns."""
        return {
            "rest_api": {
                "template": """
async def {function_name}(**kwargs):
    import httpx
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url=kwargs.get('url', '{default_url}'),
                params=kwargs.get('params', {{}})
            )
            response.raise_for_status()
            
            return {{
                "success": True,
                "result": response.json(),
                "status_code": response.status_code
            }}
            
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "result": None
        }}
""",
                "complexity": "medium"
            },
            
            "database": {
                "template": """
async def {function_name}(**kwargs):
    try:
        # Database operation implementation
        result = await execute_database_operation(kwargs)
        
        return {{
            "success": True,
            "result": result
        }}
        
    except Exception as e:
        return {{
            "success": False,
            "error": str(e)
        }}
""",
                "complexity": "high"
            },
            
            "calculation": {
                "template": """
async def {function_name}(**kwargs):
    try:
        # Calculation implementation
        result = perform_calculation(kwargs)
        
        return {{
            "success": True,
            "result": result
        }}
        
    except Exception as e:
        return {{
            "success": False,
            "error": str(e)
        }}
""",
                "complexity": "low"
            }
        }
    
    # Helper methods
    def _generate_tool_name(self, capability: str) -> str:
        """Generate a tool name from capability description."""
        name = capability.lower().replace(' ', '_').replace('-', '_')
        name = ''.join(c for c in name if c.isalnum() or c == '_')
        return f"{name}_tool"
    
    def _get_default_tool_spec(self, gap: ToolGap) -> Dict[str, Any]:
        """Get default tool specification."""
        return {
            "function_name": self._generate_tool_name(gap.capability_needed),
            "parameters": {"input": {"type": "string", "required": True}},
            "output_format": {"success": "boolean", "result": "any"},
            "implementation": {"approach": "custom", "complexity": "medium"},
            "template": "rest_api"
        }
    
    async def _attempt_mcp_solutions(self, request: ToolRequest, mcp_solutions: List[MCPMatch]) -> bool:
        """
        Attempt to use MCP solutions for the tool need.
        
        Returns True if successful, False if user collaboration needed.
        """
        try:
            for mcp_match in mcp_solutions[:3]:  # Try top 3 solutions
                mcp = mcp_match.capability
                logger.info(f"📦 Trying MCP solution: {mcp.name}")
                
                # Check if MCP is already connected
                if mcp_match.match_type == "exact_existing":
                    logger.info(f"✅ MCP {mcp.name} already connected and available")
                    await self._complete_request_with_mcp(request, mcp_match)
                    return True
                
                # Check if we can auto-connect this MCP
                if await self._can_auto_connect_mcp(mcp):
                    logger.info(f"🔌 Auto-connecting MCP: {mcp.name}")
                    connection_success = await self._auto_connect_mcp(mcp)
                    if connection_success:
                        await self._complete_request_with_mcp(request, mcp_match)
                        return True
                
                # If we can't auto-connect, add to user collaboration needs
                logger.info(f"👤 MCP {mcp.name} requires user setup")
                await self._add_mcp_to_collaboration_request(request, mcp_match)
            
            # If we reach here, all MCPs need user collaboration
            return False
            
        except Exception as e:
            logger.error(f"Error attempting MCP solutions: {e}")
            return False
    
    async def _can_auto_connect_mcp(self, mcp: 'MCPCapability') -> bool:
        """Check if an MCP can be auto-connected without user input."""
        # Check if all required credentials are already stored
        for requirement in mcp.setup_requirements:
            if not await self.credential_manager.has_credential(mcp.mcp_id, requirement):
                return False
        return True
    
    async def _auto_connect_mcp(self, mcp: 'MCPCapability') -> bool:
        """Attempt to auto-connect an MCP using stored credentials."""
        try:
            # Retrieve stored credentials
            credentials = {}
            for requirement in mcp.setup_requirements:
                cred = await self.credential_manager.get_credential(mcp.mcp_id, requirement)
                if cred:
                    credentials[requirement] = cred
            
            # Attempt connection through tool registry
            connection_result = await self.tool_registry.connect_mcp(
                mcp.mcp_id, 
                mcp.name, 
                credentials
            )
            
            if connection_result['success']:
                logger.info(f"✅ Successfully auto-connected MCP: {mcp.name}")
                return True
            else:
                logger.warning(f"❌ Failed to auto-connect MCP {mcp.name}: {connection_result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error auto-connecting MCP {mcp.mcp_id}: {e}")
            return False
    
    async def _complete_request_with_mcp(self, request: ToolRequest, mcp_match: MCPMatch):
        """Complete a tool request using an MCP solution."""
        request.status = ToolRequestStatus.COMPLETED
        request.completed_at = datetime.utcnow()
        request.validation_results = {
            "success": True,
            "solution_type": "mcp",
            "mcp_used": mcp_match.capability.name,
            "match_type": mcp_match.match_type,
            "match_score": mcp_match.match_score
        }
        
        # Publish success event
        if self.event_bus:
            await self.event_bus.publish(
                EventType.TOOL_CREATED,
                {
                    "request_id": request.request_id,
                    "agent_id": request.agent_id,
                    "tool_name": request.tool_name,
                    "solution_type": "mcp",
                    "mcp_name": mcp_match.capability.name,
                    "setup_time_ms": (datetime.utcnow() - request.created_at).total_seconds() * 1000
                },
                source=request.agent_id
            )
        
        logger.info(f"✅ Tool request completed with MCP: {mcp_match.capability.name}")
    
    async def _add_mcp_to_collaboration_request(self, request: ToolRequest, mcp_match: MCPMatch):
        """Add MCP setup requirements to user collaboration request."""
        setup_guide = await self.mcp_discovery.suggest_mcp_setup(mcp_match)
        
        collaboration_item = {
            "type": "mcp_setup",
            "mcp_name": mcp_match.capability.name,
            "description": mcp_match.capability.description,
            "setup_requirements": mcp_match.setup_needed,
            "setup_steps": setup_guide["setup_steps"],
            "documentation": setup_guide.get("documentation"),
            "estimated_time": setup_guide.get("estimated_setup_time", "5-10 minutes")
        }
        
        request.user_collaboration_needed.append(collaboration_item)
    
    async def _search_existing_tools(self, request: ToolRequest) -> Optional[Dict[str, Any]]:
        """Search for existing tools that might satisfy the need."""
        available_tools = await self.tool_registry.get_available_tools()
        capability_keywords = request.tool_description.lower().split()
        
        for tool in available_tools:
            tool_desc = tool['description'].lower()
            if any(keyword in tool_desc for keyword in capability_keywords[:3]):
                return tool
        
        return None
    
    async def _attempt_automatic_build(self, request: ToolRequest) -> Optional[Dict[str, Any]]:
        """Attempt to automatically build the tool from specification."""
        spec = request.tool_specification
        template_type = spec.get('template', 'rest_api')
        
        template = self.tool_templates.get(template_type)
        if not template:
            return None
        
        try:
            # Generate basic tool code from template
            tool_code = template['template'].format(
                function_name=spec.get('function_name', 'generated_tool'),
                default_url='https://api.example.com'
            )
            
            return {
                "name": request.tool_name,
                "description": request.tool_description,
                "function_code": tool_code,
                "parameters": spec.get('parameters', {}),
                "created_by": "dynamic_builder"
            }
            
        except Exception as e:
            logger.error(f"Automatic tool building failed: {e}")
            return None
    
    async def _validate_tool(self, tool_def: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a built tool before deployment."""
        try:
            # Basic syntax validation
            try:
                compile(tool_def['function_code'], '<string>', 'exec')
                return {"success": True, "message": "Tool validation passed"}
            except SyntaxError as e:
                return {"success": False, "error": f"Syntax error: {str(e)}"}
            
        except Exception as e:
            return {"success": False, "error": f"Validation error: {str(e)}"}
    
    async def _request_user_collaboration(self, request: ToolRequest):
        """Request user help with tool creation."""
        request.status = ToolRequestStatus.USER_INPUT_NEEDED
        
        # Determine what help is needed
        help_needed = []
        spec = request.tool_specification
        impl = spec.get('implementation', {})
        
        if impl.get('apis_needed'):
            help_needed.append("api_credentials")
        
        if impl.get('complexity') == 'high':
            help_needed.append("tool_specification")
        
        help_needed.append("testing_data")
        
        request.user_collaboration_needed = help_needed
        
        # Publish event for user notification
        if self.event_bus:
            await self.event_bus.publish(
                EventType.USER_COLLABORATION_NEEDED,
                {
                    "request_id": request.request_id,
                    "user_id": request.user_id,
                    "tool_name": request.tool_name,
                    "help_needed": help_needed,
                    "priority": "medium"
                },
                source="dynamic_tool_builder"
            )
        
        logger.info(f"👥 User collaboration requested for tool: {request.tool_name}")
    
    async def _log_tool_request(self, request: ToolRequest):
        """Log tool request to database."""
        try:
            await self.supabase_logger.client.table("tool_requests").insert({
                "request_id": request.request_id,
                "gap_id": request.gap_id,
                "agent_id": request.agent_id,
                "user_id": request.user_id,
                "tool_name": request.tool_name,
                "tool_description": request.tool_description,
                "status": request.status.value,
                "created_at": request.created_at.isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Error logging tool request: {e}")
    
    # Placeholder implementations for complete interface
    async def _complete_request_with_existing_tool(self, request: ToolRequest, existing_tool: Dict[str, Any]):
        """Complete request using existing tool."""
        request.status = ToolRequestStatus.COMPLETED
        logger.info(f"✅ Completed request {request.request_id} with existing tool")
    
    async def _deploy_tool(self, request: ToolRequest, tool_def: Dict[str, Any]):
        """Deploy tool to registry."""
        await self.tool_registry.register_tool(
            connection_id="dynamic_builder",
            tool_name=tool_def['name'],
            description=tool_def['description'],
            parameters=tool_def['parameters'],
            execution_function=lambda **kwargs: {"success": True, "result": "Tool executed"}
        )
        logger.info(f"🚀 Deployed tool: {tool_def['name']}")
    
    async def _complete_request(self, request: ToolRequest):
        """Complete tool request."""
        request.status = ToolRequestStatus.COMPLETED
        request.completed_at = datetime.utcnow()
        logger.info(f"✅ Completed tool request: {request.tool_name}")
    
    async def _notify_failure(self, request: ToolRequest, error: str):
        """Notify of request failure."""
        logger.error(f"❌ Tool request failed: {request.tool_name} - {error}")
        
        if self.event_bus:
            await self.event_bus.publish(
                EventType.TOOL_REQUEST_FAILED,
                {
                    "request_id": request.request_id,
                    "tool_name": request.tool_name,
                    "error": error
                },
                source="dynamic_tool_builder"
            ) 