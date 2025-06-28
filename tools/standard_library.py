#!/usr/bin/env python3
"""
Standard Tools Library
Universal capabilities available to all agents - inspired by smolagents simplicity.
"""

import requests
import json
import os
from typing import Dict, Any, List, Optional
from functools import wraps

def tool(func):
    """
    Simple @tool decorator like smolagents.
    Marks functions as agent tools with automatic discovery.
    """
    func._is_tool = True
    func._tool_name = func.__name__
    func._tool_description = func.__doc__ or f"Tool: {func.__name__}"
    return func

# =============================================================================
# 🌐 WEB CAPABILITIES - Universal like smolagents
# =============================================================================

@tool
def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo.
    Universal web search capability available to all agents.
    
    Args:
        query: Search query
        num_results: Number of results to return
    
    Returns:
        Dict with search results
    """
    try:
        # Use DuckDuckGo Instant Answer API
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract useful information
        results = {
            "query": query,
            "answer": data.get('Answer', ''),
            "abstract": data.get('Abstract', ''),
            "definition": data.get('Definition', ''),
            "related_topics": [
                {
                    "text": topic.get('Text', ''),
                    "url": topic.get('FirstURL', '')
                }
                for topic in data.get('RelatedTopics', [])[:num_results]
                if isinstance(topic, dict) and topic.get('Text')
            ],
            "infobox": data.get('Infobox', {}),
            "source": "DuckDuckGo"
        }
        
        return {"success": True, "results": results}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool
def visit_webpage(url: str) -> Dict[str, Any]:
    """
    Visit a webpage and extract text content.
    
    Args:
        url: URL to visit
    
    Returns:
        Dict with webpage content
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Simple text extraction (could be enhanced with BeautifulSoup)
        content = response.text
        
        # Basic cleanup
        if len(content) > 10000:
            content = content[:10000] + "..."
        
        return {
            "success": True,
            "url": url,
            "content": content,
            "status_code": response.status_code
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "url": url}

# =============================================================================
# 🔧 UTILITY CAPABILITIES
# =============================================================================

@tool
def calculate(expression: str) -> Dict[str, Any]:
    """
    Safely evaluate mathematical expressions.
    
    Args:
        expression: Mathematical expression to evaluate
    
    Returns:
        Dict with calculation result
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set('0123456789+-*/().,e ')
        if not all(c in allowed_chars for c in expression.replace(' ', '')):
            return {"success": False, "error": "Invalid characters in expression"}
        
        # Evaluate safely
        result = eval(expression)
        
        return {
            "success": True,
            "expression": expression,
            "result": result
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool  
def read_file(file_path: str) -> Dict[str, Any]:
    """
    Read a text file.
    
    Args:
        file_path: Path to file
    
    Returns:
        Dict with file content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "success": True,
            "file_path": file_path,
            "content": content,
            "size": len(content)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool
def write_file(file_path: str, content: str) -> Dict[str, Any]:
    """
    Write content to a file.
    
    Args:
        file_path: Path to file
        content: Content to write
    
    Returns:
        Dict with operation result
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "file_path": file_path,
            "bytes_written": len(content.encode('utf-8'))
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# =============================================================================
# 🎯 TOOL DISCOVERY & EXECUTION
# =============================================================================

def get_standard_tools() -> List[Dict[str, Any]]:
    """Get all available standard tools."""
    tools = []
    
    # Discover all functions with @tool decorator in this module
    import inspect
    current_module = inspect.getmodule(inspect.currentframe())
    
    for name, func in inspect.getmembers(current_module, inspect.isfunction):
        if hasattr(func, '_is_tool'):
            # Extract parameter info
            sig = inspect.signature(func)
            parameters = {}
            
            for param_name, param in sig.parameters.items():
                param_info = {
                    "type": param.annotation.__name__ if param.annotation != param.empty else "any",
                    "required": param.default == param.empty
                }
                if param.default != param.empty:
                    param_info["default"] = param.default
                
                parameters[param_name] = param_info
            
            tools.append({
                "name": func._tool_name,
                "description": func._tool_description,
                "function": func,
                "parameters": parameters
            })
    
    return tools

def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """Execute a standard tool by name."""
    tools = get_standard_tools()
    
    for tool in tools:
        if tool["name"] == tool_name:
            try:
                result = tool["function"](**kwargs)
                return result
            except Exception as e:
                return {"success": False, "error": f"Tool execution failed: {e}"}
    
    return {"success": False, "error": f"Tool '{tool_name}' not found"}

# =============================================================================
# 🚀 STANDARD TOOL AGENT (Simplified)
# =============================================================================

class StandardAgent:
    """
    Simple agent with universal tool capabilities.
    Inspired by smolagents simplicity.
    """
    
    def __init__(self, agent_id: str = "standard_agent"):
        self.agent_id = agent_id
        self.tools = {tool["name"]: tool["function"] for tool in get_standard_tools()}
        
        print(f"🔧 StandardAgent initialized with {len(self.tools)} universal tools:")
        for tool_name in self.tools.keys():
            print(f"   • {tool_name}")
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name."""
        if tool_name in self.tools:
            try:
                return self.tools[tool_name](**kwargs)
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            return {"success": False, "error": f"Tool '{tool_name}' not available"}
    
    def list_tools(self) -> List[str]:
        """List available tools."""
        return list(self.tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a specific tool."""
        tools = get_standard_tools()
        for tool in tools:
            if tool["name"] == tool_name:
                return {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
        return {"error": f"Tool '{tool_name}' not found"}

# =============================================================================
# 🧪 DEMO
# =============================================================================

if __name__ == "__main__":
    print("🔧 Standard Tools Library Demo")
    print("=" * 40)
    
    # Create agent
    agent = StandardAgent()
    
    # Test web search
    print("\n🌐 Testing web search...")
    search_result = agent.execute_tool("web_search", query="Python programming", num_results=3)
    print(f"Success: {search_result.get('success')}")
    if search_result.get('success'):
        results = search_result.get('results', {})
        print(f"Answer: {results.get('answer', 'N/A')}")
        print(f"Abstract: {results.get('abstract', 'N/A')[:100]}...")
    
    # Test calculation
    print("\n🔢 Testing calculation...")
    calc_result = agent.execute_tool("calculate", expression="10 + 20 * 3")
    print(f"Result: {calc_result}")
    
    # List all tools
    print(f"\n📋 Available tools: {agent.list_tools()}") 