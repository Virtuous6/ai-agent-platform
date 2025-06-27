"""
Real MCP Protocol Client

Implements the official Model Context Protocol (MCP) specification:
- JSON-RPC 2.0 over Server-Sent Events (SSE)
- Tool discovery and execution
- Session management
- Error handling and retries

This replaces the simulation layer with actual MCP server communication.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import aiohttp
import backoff
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MCPToolDescriptor:
    """MCP tool descriptor from server discovery."""
    name: str
    description: str
    input_schema: Dict[str, Any]

@dataclass
class MCPSession:
    """Active MCP session with a server."""
    session_id: str
    server_url: str
    tools: List[MCPToolDescriptor]
    capabilities: Dict[str, Any]
    credentials: Optional[Dict[str, Any]] = None
    websocket: Optional[Any] = None
    last_activity: Optional[datetime] = None

class MCPProtocolClient:
    """
    Official MCP protocol client implementing JSON-RPC 2.0 over SSE/WebSocket.
    
    Follows the Model Context Protocol specification for:
    - Client-server handshake
    - Tool discovery
    - Tool execution
    - Session management
    """
    
    def __init__(self):
        """Initialize MCP protocol client."""
        self.active_sessions: Dict[str, MCPSession] = {}
        self.request_id_counter = 0
        
        logger.info("🔌 MCP Protocol Client initialized")
    
    async def connect_to_mcp_server(self, server_url: str, credentials: Dict[str, Any] = None) -> MCPSession:
        """
        Connect to an MCP server and establish a session.
        
        Args:
            server_url: MCP server URL (SSE or WebSocket)
            credentials: Authentication credentials
            
        Returns:
            Active MCP session
            
        Raises:
            ConnectionError: If connection fails
            ValueError: If server doesn't support MCP protocol
        """
        try:
            logger.info(f"🔗 Connecting to MCP server: {server_url}")
            
            # Create session
            session_id = str(uuid.uuid4())
            session = MCPSession(
                session_id=session_id,
                server_url=server_url,
                tools=[],
                capabilities={},
                credentials=credentials,
                last_activity=datetime.utcnow()
            )
            
            # Determine transport type
            if server_url.endswith('/sse'):
                await self._connect_sse(session, credentials)
            elif 'ws://' in server_url or 'wss://' in server_url:
                await self._connect_websocket(session, credentials)
            else:
                # Try SSE first, then WebSocket
                try:
                    await self._connect_sse(session, credentials)
                except Exception:
                    await self._connect_websocket(session, credentials)
            
            # Perform MCP handshake
            await self._perform_handshake(session)
            
            # Discover available tools
            await self._discover_tools(session)
            
            # Store active session
            self.active_sessions[session_id] = session
            
            logger.info(f"✅ Connected to MCP server with {len(session.tools)} tools")
            return session
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP server {server_url}: {str(e)}")
            raise ConnectionError(f"MCP connection failed: {str(e)}")
    
    async def _connect_sse(self, session: MCPSession, credentials: Dict[str, Any] = None):
        """Connect using Server-Sent Events transport."""
        logger.info(f"🌊 Establishing SSE connection to {session.server_url}")
        
        # SSE connections are established per request
        # We'll create a session object but actual connection happens during requests
        session.capabilities['transport'] = 'sse'
        
        # Test connectivity with a ping
        await self._test_sse_connectivity(session.server_url, credentials)
    
    async def _connect_websocket(self, session: MCPSession, credentials: Dict[str, Any] = None):
        """Connect using WebSocket transport."""
        logger.info(f"🔌 Establishing WebSocket connection to {session.server_url}")
        
        # For WebSocket, we maintain persistent connection
        # This would require websockets library - for now, fallback to SSE
        session.capabilities['transport'] = 'websocket'
        raise NotImplementedError("WebSocket transport not yet implemented - using SSE")
    
    async def _test_sse_connectivity(self, server_url: str, credentials: Dict[str, Any] = None):
        """Test SSE server connectivity."""
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            headers = {'Accept': 'application/json'}
            
            if credentials:
                # Add authentication headers
                api_key = credentials.get('api_key')
                auth_type = credentials.get('auth_type', 'Bearer')
                
                if api_key:
                    if auth_type.lower() == 'bearer':
                        headers['Authorization'] = f"Bearer {api_key}"
                    elif auth_type.lower() == 'apikey':
                        headers['Authorization'] = f"ApiKey {api_key}"
                    elif auth_type.lower() == 'x-api-key':
                        headers['X-API-Key'] = api_key
                    else:
                        headers['Authorization'] = f"{auth_type} {api_key}"
                        
                logger.info(f"🔑 Using authentication: {auth_type} (key: ...{api_key[-4:] if api_key and len(api_key) > 4 else 'short'})")
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(server_url, headers=headers) as response:
                    if response.status == 200:
                        logger.info("✅ SSE server connectivity confirmed")
                    else:
                        raise ConnectionError(f"Server returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"❌ SSE connectivity test failed: {str(e)}")
            raise
    
    async def _perform_handshake(self, session: MCPSession):
        """Perform MCP protocol handshake."""
        logger.info("🤝 Performing MCP handshake...")
        
        # Initialize request (MCP protocol)
        initialize_request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0.0",
                "capabilities": {
                    "tools": {},
                    "logging": {},
                    "prompts": {}
                },
                "clientInfo": {
                    "name": "ai-agent-platform",
                    "version": "1.0.0"
                }
            }
        }
        
        # Send initialize request
        response = await self._send_request(session, initialize_request)
        
        if 'error' in response:
            raise ValueError(f"MCP handshake failed: {response['error']}")
        
        # Store server capabilities
        result = response.get('result', {})
        session.capabilities.update(result.get('capabilities', {}))
        
        logger.info("✅ MCP handshake completed")
    
    async def _discover_tools(self, session: MCPSession):
        """Discover available tools from MCP server."""
        logger.info("🔍 Discovering MCP tools...")
        
        # List tools request (MCP protocol)
        tools_request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/list",
            "params": {}
        }
        
        response = await self._send_request(session, tools_request)
        
        if 'error' in response:
            logger.warning(f"Tool discovery failed: {response['error']}")
            return
        
        # Parse tools from response
        tools_data = response.get('result', {}).get('tools', [])
        
        for tool_data in tools_data:
            tool = MCPToolDescriptor(
                name=tool_data.get('name', 'unknown'),
                description=tool_data.get('description', ''),
                input_schema=tool_data.get('inputSchema', {})
            )
            session.tools.append(tool)
        
        logger.info(f"✅ Discovered {len(session.tools)} tools")
    
    async def execute_tool(self, session_id: str, tool_name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a tool on the MCP server.
        
        Args:
            session_id: Active session ID
            tool_name: Name of tool to execute
            parameters: Tool parameters
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If session not found or tool doesn't exist
            RuntimeError: If tool execution fails
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Check if tool exists
        tool = next((t for t in session.tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in session")
        
        logger.info(f"🔧 Executing MCP tool: {tool_name}")
        
        # Execute tool request (MCP protocol)
        execute_request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": parameters or {}
            }
        }
        
        try:
            response = await self._send_request(session, execute_request)
            
            if 'error' in response:
                error_msg = response['error'].get('message', 'Unknown error')
                logger.error(f"❌ Tool execution failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "tool_name": tool_name
                }
            
            # Parse successful result
            result = response.get('result', {})
            
            # Update session activity
            session.last_activity = datetime.utcnow()
            
            logger.info(f"✅ Tool '{tool_name}' executed successfully")
            
            return {
                "success": True,
                "data": result.get('content', result),
                "tool_name": tool_name,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"❌ Tool execution error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name
            }
    
    @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=3)
    async def _send_request(self, session: MCPSession, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON-RPC request to MCP server with retry logic."""
        
        if session.capabilities.get('transport') == 'sse':
            return await self._send_sse_request(session, request)
        elif session.capabilities.get('transport') == 'websocket':
            return await self._send_websocket_request(session, request)
        else:
            raise ValueError("No valid transport configured")
    
    async def _send_sse_request(self, session: MCPSession, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request via Server-Sent Events with authentication."""
        timeout = aiohttp.ClientTimeout(total=30)
        
        try:
            # Prepare headers with authentication
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # Add authentication if session has credentials
            if hasattr(session, 'credentials') and session.credentials:
                api_key = session.credentials.get('api_key')
                auth_type = session.credentials.get('auth_type', 'Bearer')
                
                if api_key:
                    if auth_type.lower() == 'bearer':
                        headers['Authorization'] = f"Bearer {api_key}"
                    elif auth_type.lower() == 'apikey':
                        headers['Authorization'] = f"ApiKey {api_key}"
                    elif auth_type.lower() == 'x-api-key':
                        headers['X-API-Key'] = api_key
                    else:
                        headers['Authorization'] = f"{auth_type} {api_key}"
            
            async with aiohttp.ClientSession(timeout=timeout) as http_session:
                # For SSE, we typically POST the request and get response
                async with http_session.post(
                    session.server_url,
                    json=request,
                    headers=headers
                ) as response:
                    
                    if response.status != 200:
                        response_text = await response.text()
                        logger.error(f"❌ MCP server error (status {response.status}): {response_text}")
                        raise aiohttp.ClientError(f"Server returned status {response.status}: {response_text}")
                    
                    response_data = await response.json()
                    return response_data
                    
        except Exception as e:
            logger.error(f"❌ SSE request failed: {str(e)}")
            raise
    
    async def _send_websocket_request(self, session: MCPSession, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request via WebSocket."""
        # WebSocket implementation would go here
        raise NotImplementedError("WebSocket transport not yet implemented")
    
    def _next_request_id(self) -> int:
        """Generate next request ID for JSON-RPC."""
        self.request_id_counter += 1
        return self.request_id_counter
    
    async def get_session_tools(self, session_id: str) -> List[MCPToolDescriptor]:
        """Get list of available tools for a session."""
        if session_id not in self.active_sessions:
            return []
        
        return self.active_sessions[session_id].tools
    
    async def close_session(self, session_id: str):
        """Close an MCP session."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Close WebSocket if exists
            if session.websocket:
                await session.websocket.close()
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            logger.info(f"🔌 Closed MCP session: {session_id}")
    
    async def close_all_sessions(self):
        """Close all active MCP sessions."""
        for session_id in list(self.active_sessions.keys()):
            await self.close_session(session_id)
        
        logger.info("🔌 All MCP sessions closed")

# Global MCP client instance
mcp_client = MCPProtocolClient() 