#!/usr/bin/env python3
"""
Simple Code-First Agent

"""

import asyncio
import uuid
import time
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Import standard tools directly
from tools.standard_library import get_standard_tools, StandardAgent

class SimpleCodeAgent:
    """
    Code-first agent with standard tools.
    
    Philosophy:
    - Universal tools available by default (web_search, calculate, etc.)
    - Code-first execution
    - Simple interface, powerful capabilities
    - Zero complex dependencies
    """
    
    def __init__(self, 
                 agent_id: str = None,
                 temperature: float = 0.4,
                 model_name: str = "gpt-3.5-turbo-0125"):
        
        self.agent_id = agent_id or f"simple_code_{uuid.uuid4().hex[:8]}"
        self.temperature = temperature
        self.model_name = model_name
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize standard tools
        self.standard_agent = StandardAgent(self.agent_id)
        self.tools = self.standard_agent.tools
        
        # Create code execution prompt
        self.prompt = self._create_code_prompt()
        
        print(f"🤖 SimpleCodeAgent '{self.agent_id}' ready with {len(self.tools)} tools")
    
    def _create_code_prompt(self) -> ChatPromptTemplate:
        """Create prompt that enables code-first tool usage."""
        
        # Get tool descriptions
        tools_info = []
        for tool_name, tool_func in self.tools.items():
            doc = tool_func.__doc__ or "No description"
            first_line = doc.split('\n')[0].strip()
            tools_info.append(f"- {tool_name}: {first_line}")
        
        tools_list = "\n".join(tools_info)
        
        system_template = f"""You are a helpful AI agent that can write and execute Python code to solve problems.

**Available Tools:**
{tools_list}

**Instructions:**
1. Write Python code to solve the user's request
2. Use the available tools when needed
3. Always wrap your code in ```python code blocks
4. Be direct - write executable code, not explanations

**Examples:**

User: "What's 5 + 3?"
```python
result = calculate("5 + 3")
print(f"5 + 3 = {{{{result['result']}}}}")
```

User: "Search for Python info"
```python
search_result = web_search("Python programming language")
if search_result["success"]:
    print(f"Answer: {{{{search_result['results']['answer']}}}}")
    print(f"Abstract: {{{{search_result['results']['abstract']}}}}")
```

**Important:**
- Always use ```python code blocks
- Write executable Python code only
- No explanations outside code blocks
- Use tools for calculations, searches, etc.

User: {{message}}

```python"""

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{message}")
        ])
    
    async def run(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the agent with a message.
        Simple interface for universal tool access.
        """
        run_id = str(uuid.uuid4())
        start_time = time.time()
        context = context or {}
        
        try:
            # Generate code using LLM
            print(f"🧠 Thinking about: {message}")
            
            prompt_input = {"message": message}
            chain = self.prompt | self.llm
            
            llm_response = await chain.ainvoke(prompt_input)
            generated_code = llm_response.content
            
            # Extract just the Python code from the response
            code_to_execute = self._extract_python_code(generated_code)
            print(f"📝 Generated code:\n{code_to_execute}")
            
            # Execute the code
            execution_result = await self._execute_code(code_to_execute, message)
            
            # Track performance
            processing_time = time.time() - start_time
            
            return {
                "response": execution_result.get("output", ""),
                "success": execution_result.get("success", False),
                "code_generated": code_to_execute,
                "execution_result": execution_result,
                "run_id": run_id,
                "processing_time": processing_time,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            error_result = {
                "response": f"Error: {str(e)}",
                "success": False,
                "error": str(e),
                "run_id": run_id,
                "agent_id": self.agent_id
            }
            
            return error_result
    
    def _extract_python_code(self, llm_response: str) -> str:
        """Extract Python code from LLM response."""
        import re
        
        # Clean the response
        response = llm_response.strip()
        
        # If response starts with ```python, extract everything after it
        if response.startswith('```python'):
            # Remove ```python and any trailing ```
            code = response[9:].strip()  # Remove ```python
            if code.endswith('```'):
                code = code[:-3].strip()  # Remove trailing ```
            return code
        
        # Try to find ```python code blocks
        python_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if python_blocks:
            return python_blocks[0].strip()
        
        # Try to find any ``` code blocks
        code_blocks = re.findall(r'```\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # Look for lines that look like executable Python code
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip explanatory text
            if any(phrase in stripped.lower() for phrase in ['to calculate', 'let\'s', 'we can', 'follow these steps', 'step by step']):
                continue
                
            # Look for actual Python code patterns
            if (stripped.startswith(('result =', 'calc_result =', 'search_result =', 'tip =', 'total =', 'answer =', 'print(')) or
                'calculate(' in stripped or 'web_search(' in stripped or
                stripped.startswith(('if ', 'for ', 'while ', 'def ', 'import ', 'from '))):
                in_code = True
                code_lines.append(line)
            elif in_code and (stripped.startswith('    ') or stripped == ''):
                # Continue if we're in a code block (indented lines or empty lines)
                code_lines.append(line)
            elif stripped and not any(phrase in stripped.lower() for phrase in ['calculate', 'step', 'result']):
                # Stop if we hit non-code text
                in_code = False
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        # Fallback: if it looks like simple math, make it a calculation
        if any(op in response for op in ['+', '-', '*', '/', '%']) and len(response.split()) < 10:
            return f'result = calculate("{response.strip()}")\nprint(f"Result: {{result[\'result\']}}")'
        
        # Last resort: return the whole response and hope for the best
        return response.strip()
    
    async def _execute_code(self, code: str, original_message: str) -> Dict[str, Any]:
        """
        Execute generated code with tool access.
        Safe execution environment.
        """
        try:
            # Create execution environment with tools
            exec_globals = {
                "__builtins__": {"print": print, "len": len, "str": str, "int": int, "float": float},
                # Add all tools to execution environment
                **self.tools,
                # Add utility functions
                "range": range,
                "enumerate": enumerate,
                "zip": zip
            }
            
            # Capture output
            output_lines = []
            
            def capture_print(*args, **kwargs):
                output_lines.append(" ".join(str(arg) for arg in args))
            
            exec_globals["print"] = capture_print
            
            # Execute the code
            exec(code, exec_globals)
            
            output = "\n".join(output_lines) if output_lines else "Code executed successfully"
            
            return {
                "success": True,
                "output": output,
                "code": code
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": f"Execution error: {str(e)}",
                "error": str(e),
                "code": code
            }
    
    def list_tools(self) -> List[str]:
        """List available tools."""
        return list(self.tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get info about a specific tool."""
        return self.standard_agent.get_tool_info(tool_name)

# =============================================================================
# 🔗 PLATFORM INTEGRATION
# =============================================================================

class HybridAgent(SimpleCodeAgent):
    """
    Hybrid agent that combines SimpleCodeAgent with platform features.
    
    Best of both worlds:
    - Simple tool execution interface
    - Platform learning and improvement capabilities
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Add platform integrations
        self.conversation_history = []
        self.performance_metrics = {
            "total_runs": 0,
            "successful_runs": 0,
            "average_processing_time": 0.0,
            "tools_used": {},
            "code_execution_success_rate": 0.0
        }
        
        print(f"🔗 HybridAgent initialized with platform integration")
    
    async def run(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced run with platform learning."""
        
        # Execute with simple code agent
        result = await super().run(message, context)
        
        # Update metrics
        self._update_metrics(result)
        
        # Store conversation
        self.conversation_history.append({
            "message": message,
            "response": result.get("response", ""),
            "success": result.get("success", False),
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": result.get("run_id")
        })
        
        # Keep history manageable
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
        
        return result
    
    def _update_metrics(self, result: Dict[str, Any]):
        """Update performance metrics."""
        self.performance_metrics["total_runs"] += 1
        
        if result.get("success"):
            self.performance_metrics["successful_runs"] += 1
        
        # Update success rate
        total = self.performance_metrics["total_runs"]
        successful = self.performance_metrics["successful_runs"]
        self.performance_metrics["code_execution_success_rate"] = successful / total if total > 0 else 0
        
        # Update average processing time
        current_time = result.get("processing_time", 0)
        current_avg = self.performance_metrics["average_processing_time"]
        self.performance_metrics["average_processing_time"] = (
            (current_avg * (total - 1) + current_time) / total
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self.agent_id,
            "tools_available": list(self.tools.keys()),
            "performance": self.performance_metrics,
            "conversation_count": len(self.conversation_history),
            "agent_type": "hybrid_code_first"
        }

# =============================================================================
# 🧪 DEMO & TESTING
# =============================================================================

async def demo_simple_agent():
    """Demo the simple code-first agent."""
    print("🚀 SimpleCodeAgent Demo")
    print("=" * 40)
    
    # Create agent
    agent = SimpleCodeAgent()
    
    # Test cases
    test_cases = [
        "What's 25 * 47 + 128?",
        "Calculate the compound interest on $1000 at 5% for 3 years using the formula A = P(1 + r)^t",
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case}")
        print("-" * 50)
        
        result = await agent.run(test_case)
        
        print(f"Success: {result['success']}")
        print(f"Response: {result['response']}")
        if not result['success'] and 'error' in result:
            print(f"Error: {result['error']}")

async def demo_hybrid_agent():
    """Demo the hybrid agent with platform features."""
    print("\n🔗 HybridAgent Demo")
    print("=" * 40)
    
    # Create hybrid agent
    agent = HybridAgent(agent_id="demo_hybrid")
    
    # Run a few tasks
    tasks = [
        "Calculate 15% tip on a $87.50 bill",
        "What's 123 * 456?"
    ]
    
    for task in tasks:
        print(f"\n📋 Task: {task}")
        result = await agent.run(task)
        print(f"✅ Response: {result['response']}")
    
    # Show stats
    print(f"\n📊 Agent Stats:")
    stats = agent.get_stats()
    print(f"  • Success Rate: {stats['performance']['code_execution_success_rate']:.1%}")
    print(f"  • Avg Processing Time: {stats['performance']['average_processing_time']:.2f}s")
    print(f"  • Total Runs: {stats['performance']['total_runs']}")

if __name__ == "__main__":
    print("🤖 Simple Code-First Agent")
    print("=" * 50)
    
    asyncio.run(demo_simple_agent())
    asyncio.run(demo_hybrid_agent()) 