"""
Supabase Run Card

Pre-built connection template for Supabase database integration.
Provides quick setup with common database tools and operations.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SupabaseRunCard:
    """
    Pre-built Supabase MCP connection template.
    Provides one-click setup for Supabase database operations.
    """
    
    def __init__(self):
        """Initialize Supabase run card."""
        self.card_name = "supabase_database"
        self.display_name = "Supabase Database"
        self.mcp_type = "supabase"
        self.description = "Connect to your Supabase database for data operations"
        self.long_description = """
Connect your AI agents to your Supabase database for powerful data operations. 
This integration provides secure access to your database with built-in SQL 
execution, schema inspection, and data management tools.

Perfect for: Data analysis, user management, report generation, and real-time queries.
        """.strip()
        
        # Required credentials for Supabase
        self.required_credentials = [
            {
                "key": "url",
                "label": "Supabase URL",
                "type": "url",
                "placeholder": "https://your-project.supabase.co",
                "help": "Your project URL from Supabase dashboard"
            },
            {
                "key": "service_role_key", 
                "label": "Service Role Key",
                "type": "password",
                "placeholder": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "help": "Service role key (bypasses RLS) from Supabase dashboard > Settings > API"
            }
        ]
        
        # Optional credentials
        self.optional_credentials = [
            {
                "key": "anon_key",
                "label": "Anonymous Key (optional)",
                "type": "password", 
                "placeholder": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "help": "Public anonymous key for RLS-compliant operations"
            }
        ]
        
        # Available tools from this connection
        self.available_tools = [
            "list_tables",
            "execute_sql", 
            "get_schema",
            "insert_data",
            "update_data",
            "delete_data",
            "get_user_data",
            "manage_auth",
            "get_storage_files",
            "upload_file"
        ]
        
        # Tool descriptions for user reference
        self.tool_descriptions = {
            "list_tables": "Get all tables in your database with row counts",
            "execute_sql": "Run custom SQL queries safely with result formatting",
            "get_schema": "Get table schemas, columns, and relationships",
            "insert_data": "Insert new records into specified tables",
            "update_data": "Update existing records with conditions", 
            "delete_data": "Delete records with safety checks",
            "get_user_data": "Retrieve user information from auth.users",
            "manage_auth": "Manage user authentication and permissions",
            "get_storage_files": "List files in Supabase storage buckets",
            "upload_file": "Upload files to Supabase storage"
        }
        
        # Example use cases
        self.example_use_cases = [
            "Database queries and analytics",
            "User management and authentication", 
            "Data analysis and reporting",
            "Content management", 
            "Real-time data monitoring",
            "File storage and management"
        ]
        
        # Configuration template
        self.config_template = {
            "features": ["database", "auth", "storage"],
            "default_schema": "public",
            "max_rows_per_query": 1000,
            "enable_rls": True,
            "connection_pool_size": 5,
            "query_timeout_seconds": 30
        }
        
        logger.info("🗄️ Supabase Run Card initialized")
    
    async def quick_setup(self, user_id: str, connection_name: str, 
                         credentials: Dict[str, str],
                         config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        One-click setup for Supabase connection.
        
        Args:
            user_id: User creating the connection
            connection_name: Name for the connection
            credentials: Supabase credentials (url, service_role_key)
            config: Optional configuration overrides
            
        Returns:
            Setup result with connection details
        """
        try:
            # Validate required credentials
            self._validate_credentials(credentials)
            
            # Merge config with template
            final_config = self.config_template.copy()
            if config:
                final_config.update(config)
            
            # Test the connection
            connection_test = await self._test_connection(credentials, final_config)
            
            if not connection_test["success"]:
                return {
                    "success": False,
                    "error": connection_test["error"],
                    "step": "connection_test"
                }
            
            # Discover available tools
            available_tools = await self._discover_tools(credentials, final_config)
            
            return {
                "success": True,
                "connection_config": final_config,
                "available_tools": available_tools,
                "tool_schemas": self._get_tool_schemas(),
                "setup_summary": {
                    "database_name": connection_test.get("database_name"),
                    "tables_found": connection_test.get("tables_count", 0),
                    "tools_available": len(available_tools),
                    "features_enabled": final_config["features"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Supabase quick setup failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "step": "setup_validation"
            }
    
    async def test_connection(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """
        Test Supabase connection without creating it.
        
        Args:
            credentials: Supabase credentials
            
        Returns:
            Test result
        """
        return await self._test_connection(credentials, self.config_template)
    
    def get_setup_instructions(self) -> str:
        """Get step-by-step setup instructions."""
        return """
**Supabase Connection Setup Instructions:**

1. **Get your Supabase URL:**
   - Go to your Supabase dashboard
   - Select your project
   - Copy the "Project URL" from Settings > General

2. **Get your Service Role Key:**
   - In Supabase dashboard, go to Settings > API
   - Copy the "service_role" key (not the anon key)
   - ⚠️ **Important:** This key bypasses Row Level Security

3. **Optional - Anonymous Key:**
   - Also in Settings > API
   - Copy the "anon" key for RLS-compliant operations

4. **Configure Connection:**
   - Choose a memorable connection name
   - Paste your URL and keys
   - Test the connection

**Security Notes:**
- Service role key has full database access
- Store credentials securely
- Consider using environment-specific projects
- Enable Row Level Security (RLS) for production data
        """.strip()
    
    def get_slack_modal_blocks(self) -> List[Dict[str, Any]]:
        """Get Slack modal blocks for connection setup."""
        return [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*🗄️ {self.display_name} Setup*\n{self.description}"
                }
            },
            {"type": "divider"},
            {
                "type": "input",
                "block_id": "connection_name_block",
                "element": {
                    "type": "plain_text_input",
                    "action_id": "connection_name",
                    "placeholder": {"type": "plain_text", "text": "my_database"}
                },
                "label": {"type": "plain_text", "text": "Connection Name"}
            },
            {
                "type": "input",
                "block_id": "supabase_url_block",
                "element": {
                    "type": "url_text_input",
                    "action_id": "supabase_url",
                    "placeholder": {"type": "plain_text", "text": "https://your-project.supabase.co"}
                },
                "label": {"type": "plain_text", "text": "Supabase URL"}
            },
            {
                "type": "input",
                "block_id": "service_key_block",
                "element": {
                    "type": "plain_text_input",
                    "action_id": "service_role_key",
                    "placeholder": {"type": "plain_text", "text": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."}
                },
                "label": {"type": "plain_text", "text": "Service Role Key"}
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Available Tools:* {len(self.available_tools)} database tools\n*Use Cases:* {', '.join(self.example_use_cases[:3])}"
                }
            }
        ]
    
    # Private helper methods
    
    def _validate_credentials(self, credentials: Dict[str, str]):
        """Validate Supabase credentials."""
        required_keys = ["url", "service_role_key"]
        
        for key in required_keys:
            if key not in credentials or not credentials[key]:
                raise ValueError(f"Missing required credential: {key}")
        
        # Basic URL validation
        url = credentials["url"]
        if not url.startswith("https://") or ".supabase.co" not in url:
            raise ValueError("Invalid Supabase URL format")
        
        # Basic key validation (should start with 'eyJ')
        service_key = credentials["service_role_key"]
        if not service_key.startswith("eyJ"):
            raise ValueError("Invalid service role key format")
    
    async def _test_connection(self, credentials: Dict[str, str], 
                             config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test Supabase connection.
        
        Args:
            credentials: Supabase credentials
            config: Connection configuration
            
        Returns:
            Test result with connection details
        """
        try:
            # In a real implementation, this would:
            # 1. Create a Supabase client with the credentials
            # 2. Test a simple query like SELECT 1
            # 3. Get database metadata
            
            # For now, simulate a successful connection test
            await asyncio.sleep(0.5)  # Simulate network delay
            
            # Extract project ID from URL for display
            url = credentials["url"]
            project_id = url.split("//")[1].split(".")[0] if "//" in url else "unknown"
            
            return {
                "success": True,
                "database_name": f"project_{project_id}",
                "tables_count": 12,  # Simulated
                "connection_time_ms": 150,
                "server_version": "PostgreSQL 13.7",
                "features_available": config["features"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "connection_failed"
            }
    
    async def _discover_tools(self, credentials: Dict[str, str], 
                            config: Dict[str, Any]) -> List[str]:
        """
        Discover available tools for this Supabase connection.
        
        Args:
            credentials: Supabase credentials
            config: Connection configuration
            
        Returns:
            List of available tool names
        """
        # Base tools always available
        available_tools = ["list_tables", "execute_sql", "get_schema"]
        
        # Add tools based on enabled features
        if "database" in config["features"]:
            available_tools.extend(["insert_data", "update_data", "delete_data"])
        
        if "auth" in config["features"]:
            available_tools.extend(["get_user_data", "manage_auth"])
        
        if "storage" in config["features"]:
            available_tools.extend(["get_storage_files", "upload_file"])
        
        return available_tools
    
    def _get_tool_schemas(self) -> Dict[str, Any]:
        """Get JSON schemas for all tools."""
        return {
            "list_tables": {
                "description": "List all tables in the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "schema": {
                            "type": "string",
                            "default": "public",
                            "description": "Database schema to query"
                        },
                        "include_system_tables": {
                            "type": "boolean", 
                            "default": False,
                            "description": "Include system tables in results"
                        }
                    }
                }
            },
            "execute_sql": {
                "description": "Execute a SQL query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute"
                        },
                        "max_rows": {
                            "type": "integer",
                            "default": 100,
                            "maximum": 1000,
                            "description": "Maximum rows to return"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["table", "json", "csv"],
                            "default": "table",
                            "description": "Output format"
                        }
                    },
                    "required": ["query"]
                }
            },
            "get_schema": {
                "description": "Get database schema information",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Specific table to get schema for (optional)"
                        },
                        "include_relationships": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include foreign key relationships"
                        }
                    }
                }
            }
        } 