"""
MCP Discovery Engine

Discovers and matches existing MCPs (Model Context Protocol) tools before
building custom solutions. Prioritizes MCP-first approach for tool creation.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Add Supabase integration
from database.supabase_logger import SupabaseLogger

logger = logging.getLogger(__name__)

class MCPType(Enum):
    """Types of MCPs available."""
    DATABASE = "database"
    API_SERVICE = "api_service"
    SOFTWARE_INTEGRATION = "software_integration"
    FILE_SYSTEM = "file_system"
    COMMUNICATION = "communication"
    ANALYTICS = "analytics"
    CUSTOM = "custom"

@dataclass
class MCPCapability:
    """Represents an MCP capability."""
    mcp_id: str
    name: str
    description: str
    mcp_type: MCPType
    supported_operations: List[str]
    api_patterns: List[str]  # URL patterns this MCP can handle
    software_names: List[str]  # Software names this MCP works with
    setup_requirements: List[str]  # What user needs to provide
    confidence_score: float  # How well it matches the need
    connection_url: Optional[str] = None
    documentation_url: Optional[str] = None
    popularity_score: int = 0
    is_core: bool = False  # True for file-based core MCPs

@dataclass
class MCPMatch:
    """Represents a match between user need and available MCP."""
    capability: MCPCapability
    match_type: str  # "exact", "partial", "api_compatible", "software_compatible"
    match_score: float  # 0.0-1.0
    setup_needed: List[str]  # What setup is required
    suggested_parameters: Dict[str, Any]

class MCPDiscoveryEngine:
    """
    Discovers and matches MCPs for user requests using hybrid approach:
    1. Core MCPs from file structure (always available)
    2. Library MCPs from Supabase (community/extended)
    3. User's existing connected MCPs
    """
    
    def __init__(self, mcp_registry=None, connection_manager=None, supabase_logger=None):
        """Initialize MCP discovery engine with hybrid approach."""
        self.mcp_registry = mcp_registry
        self.connection_manager = connection_manager
        self.supabase_logger = supabase_logger or SupabaseLogger()
        
        # Load MCPs from both sources
        self.known_mcps = []
        self._load_mcps_hybrid()
        
        # API and software pattern databases
        self.api_patterns = self._load_api_patterns()
        self.software_patterns = self._load_software_patterns()
        
        logger.info(f"🔍 MCP Discovery Engine initialized with {len(self.known_mcps)} MCPs (hybrid approach)")
    
    def _load_mcps_hybrid(self):
        """Load MCPs from both core files and Supabase library."""
        # Load core MCPs (essential, always available)
        core_mcps = self._load_core_mcps()
        
        # Load library MCPs from Supabase (extended/community)
        try:
            library_mcps = self._load_library_mcps_from_supabase()
        except Exception as e:
            logger.warning(f"Failed to load library MCPs from Supabase: {e}")
            library_mcps = []
        
        # Combine and deduplicate
        all_mcps = core_mcps + library_mcps
        self.known_mcps = self._deduplicate_mcps(all_mcps)
        
        logger.info(f"✅ Loaded {len(core_mcps)} core MCPs + {len(library_mcps)} library MCPs")
    
    def _load_core_mcps(self) -> List[MCPCapability]:
        """Load essential MCPs that are actually implemented (from file structure)."""
        return [
            # Serper Web Search - Real implementation in serper_card.py
            MCPCapability(
                mcp_id="serper",
                name="Serper Web Search MCP",
                description="Real-time web search results using Serper.dev API with Google search data",
                mcp_type=MCPType.API_SERVICE,
                supported_operations=["web_search", "news_search", "images_search", "places_search"],
                api_patterns=["serper.dev", "api.serper.dev"],
                software_names=["serper", "web search", "google search"],
                setup_requirements=["serper_api_key"],
                confidence_score=0.95,
                documentation_url="https://serper.dev/api-documentation",
                popularity_score=95,
                is_core=True
            ),
            
            # Supabase Database - Real implementation in supabase_card.py  
            MCPCapability(
                mcp_id="supabase_core",
                name="Supabase Database MCP",
                description="Complete Supabase database operations including tables, auth, storage",
                mcp_type=MCPType.DATABASE,
                supported_operations=["list_tables", "execute_sql", "get_schema", "insert_data", "update_data", "delete_data", "manage_auth", "upload_file"],
                api_patterns=["*.supabase.co", "supabase.com"],
                software_names=["supabase"],
                setup_requirements=["supabase_url", "service_role_key"],
                confidence_score=0.95,
                documentation_url="https://supabase.com/docs",
                popularity_score=100,
                is_core=True
            )
        ]
    
    def _load_library_mcps_from_supabase(self) -> List[MCPCapability]:
        """Load MCP library from Supabase run cards table."""
        try:
            # Query the mcp_run_cards table directly
            result = self.supabase_logger.client.table("mcp_run_cards").select(
                "service_name, display_name, description, category, "
                "available_tools, required_credentials, optional_credentials, "
                "documentation_url, popularity_score, default_server_url"
            ).eq("is_active", True).order("popularity_score", desc=True).execute()
            
            if not result.data:
                logger.warning("No MCP run cards found in database")
                return []
            
            library_mcps = []
            for row in result.data:
                # Convert database row to MCPCapability
                mcp = self._db_row_to_mcp_capability(row)
                if mcp:
                    library_mcps.append(mcp)
            
            logger.info(f"✅ Loaded {len(library_mcps)} MCPs from Supabase library")
            return library_mcps
            
        except Exception as e:
            logger.error(f"Failed to load library MCPs from Supabase: {e}")
            return self._fallback_library_mcps()
    
    def _db_row_to_mcp_capability(self, row: Dict[str, Any]) -> Optional[MCPCapability]:
        """Convert database row to MCPCapability object."""
        try:
            # Map database categories to MCPType
            category_map = {
                'database': MCPType.DATABASE,
                'api': MCPType.API_SERVICE,
                'communication': MCPType.COMMUNICATION,
                'storage': MCPType.FILE_SYSTEM,
                'analytics': MCPType.ANALYTICS,
                'general': MCPType.CUSTOM
            }
            
            # Extract tool names from available_tools JSON
            tools = row.get('available_tools', [])
            if isinstance(tools, list):
                operations = [tool.get('name', '') for tool in tools if isinstance(tool, dict)]
            else:
                operations = []
            
            # Determine API patterns and software names based on service
            api_patterns, software_names = self._determine_patterns_from_service(row['service_name'])
            
            return MCPCapability(
                mcp_id=row['service_name'],
                name=row['display_name'],
                description=row['description'],
                mcp_type=category_map.get(row.get('category', 'general'), MCPType.CUSTOM),
                supported_operations=operations,
                api_patterns=api_patterns,
                software_names=software_names,
                setup_requirements=row.get('required_credentials', []),
                confidence_score=0.85,  # High confidence for vetted library MCPs
                connection_url=row.get('default_server_url'),
                documentation_url=row.get('documentation_url'),
                popularity_score=row.get('popularity_score', 0),
                is_core=False  # Library MCPs are not core
            )
            
        except Exception as e:
            logger.error(f"Error converting DB row to MCP: {e}")
            return None
    
    def _determine_patterns_from_service(self, service_name: str) -> Tuple[List[str], List[str]]:
        """Determine API patterns and software names from service name."""
        pattern_map = {
            'supabase': (['*.supabase.co', 'supabase.com'], ['supabase']),
            'github': (['api.github.com', 'github.com'], ['github']),
            'slack': (['api.slack.com', 'slack.com'], ['slack']),
            'postgres': (['postgresql://', 'postgres://'], ['postgresql', 'postgres']),
            'weather': (['api.openweathermap.org', 'weather.'], ['weather', 'openweather']),
            'sheets': (['sheets.googleapis.com'], ['google sheets', 'sheets'])
        }
        
        return pattern_map.get(service_name, ([], [service_name]))
    
    def _fallback_library_mcps(self) -> List[MCPCapability]:
        """No fallback MCPs - only show real MCPs from Supabase."""
        logger.warning("⚠️ Supabase unavailable - only core MCPs will be available")
        return []
    
    def _deduplicate_mcps(self, mcps: List[MCPCapability]) -> List[MCPCapability]:
        """Remove duplicate MCPs, preferring core over library."""
        seen = {}
        for mcp in mcps:
            if mcp.mcp_id not in seen or mcp.is_core:
                seen[mcp.mcp_id] = mcp
        return list(seen.values())

    async def refresh_library_mcps(self):
        """Refresh library MCPs from Supabase (for dynamic updates)."""
        try:
            library_mcps = self._load_library_mcps_from_supabase()
            
            # Update known_mcps while preserving core MCPs
            core_mcps = [mcp for mcp in self.known_mcps if mcp.is_core]
            self.known_mcps = self._deduplicate_mcps(core_mcps + library_mcps)
            
            logger.info(f"🔄 Refreshed library MCPs: {len(library_mcps)} available")
            
        except Exception as e:
            logger.error(f"Failed to refresh library MCPs: {e}")
    
    async def add_mcp_to_library(self, mcp: MCPCapability, user_id: str = None) -> bool:
        """
        Add a new MCP to the Supabase library for future use.
        
        Args:
            mcp: MCP capability to add
            user_id: User who is adding the MCP (optional)
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Prepare data for Supabase
            insert_data = {
                'card_name': mcp.mcp_id,
                'display_name': mcp.name,
                'description': mcp.description,
                'category': mcp.mcp_type.value,
                'mcp_type': mcp.mcp_type.value,
                'available_tools': [{'name': op} for op in mcp.supported_operations],
                'required_credentials': mcp.setup_requirements,
                'optional_credentials': [],
                'documentation_url': mcp.documentation_url,
                'default_server_url': mcp.connection_url or '',
                'popularity_score': mcp.popularity_score,
                'tags': mcp.software_names,
                'is_public': True,
                'is_active': True,
                'created_by': user_id,
                'is_community_contributed': user_id is not None
            }
            
            # Insert using Supabase client
            result = self.supabase_logger.client.table("mcp_run_cards").insert(insert_data).execute()
            
            if result.data:
                logger.info(f"✅ Added new MCP to library: {mcp.name}")
                
                # Refresh the local cache
                await self.refresh_library_mcps()
                return True
            else:
                logger.error(f"❌ Failed to insert MCP: {mcp.name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error adding MCP to library: {e}")
            return False
    
    async def update_mcp_popularity(self, mcp_id: str, increment: int = 1) -> bool:
        """
        Update MCP popularity score when it's used.
        
        Args:
            mcp_id: MCP identifier
            increment: Amount to increase popularity by
            
        Returns:
            True if updated successfully
        """
        try:
            result = self.supabase_logger.client.table("mcp_run_cards").update({
                'popularity_score': f"popularity_score + {increment}",
                'last_used_at': 'NOW()'
            }).eq('card_name', mcp_id).execute()
            
            if result.data:
                logger.info(f"📈 Updated popularity for MCP: {mcp_id}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating MCP popularity: {e}")
            return False

    async def find_mcp_solutions(self, capability_needed: str, description: str, 
                                context: Dict[str, Any]) -> List[MCPMatch]:
        """
        Find MCP solutions for a needed capability.
        
        Args:
            capability_needed: The capability description
            description: Detailed description of what's needed
            context: Additional context (user message, etc.)
            
        Returns:
            List of MCP matches ordered by relevance
        """
        try:
            logger.info(f"🔍 Searching for MCP solutions for: {capability_needed}")
            
            matches = []
            
            # Step 1: Check existing connected MCPs
            existing_matches = await self._check_existing_mcps(capability_needed, description)
            matches.extend(existing_matches)
            
            # Step 2: Check available MCPs by capability
            capability_matches = await self._match_by_capability(capability_needed, description)
            matches.extend(capability_matches)
            
            # Step 3: Check for API-compatible MCPs
            api_matches = await self._match_by_api_patterns(description, context)
            matches.extend(api_matches)
            
            # Step 4: Check for software-specific MCPs
            software_matches = await self._match_by_software(description, context)
            matches.extend(software_matches)
            
            # Remove duplicates and sort by match score
            unique_matches = self._deduplicate_matches(matches)
            sorted_matches = sorted(unique_matches, key=lambda m: m.match_score, reverse=True)
            
            logger.info(f"✅ Found {len(sorted_matches)} MCP solutions")
            for match in sorted_matches[:3]:  # Log top 3
                logger.info(f"   📦 {match.capability.name} ({match.match_type}, score: {match.match_score:.2f})")
            
            return sorted_matches
            
        except Exception as e:
            logger.error(f"Error finding MCP solutions: {e}")
            return []
    
    async def _check_existing_mcps(self, capability_needed: str, description: str) -> List[MCPMatch]:
        """Check if any existing connected MCPs can handle this capability."""
        matches = []
        
        if not self.mcp_registry:
            return matches
        
        try:
            # Get currently available tools from MCP registry
            available_tools = await self.mcp_registry.get_available_tools()
            
            for tool in available_tools:
                # Check if tool description matches capability
                similarity_score = self._calculate_similarity(
                    capability_needed + " " + description,
                    tool.get('description', '')
                )
                
                if similarity_score > 0.6:  # Good match
                    # Find the MCP capability for this tool
                    mcp_capability = self._find_mcp_for_tool(tool)
                    
                    if mcp_capability:
                        matches.append(MCPMatch(
                            capability=mcp_capability,
                            match_type="exact_existing",
                            match_score=similarity_score,
                            setup_needed=[],  # Already connected
                            suggested_parameters=self._extract_tool_parameters(tool)
                        ))
                        
        except Exception as e:
            logger.warning(f"Error checking existing MCPs: {e}")
        
        return matches
    
    async def _match_by_capability(self, capability_needed: str, description: str) -> List[MCPMatch]:
        """Match against known MCP capabilities."""
        matches = []
        
        capability_text = f"{capability_needed} {description}".lower()
        
        for mcp in self.known_mcps:
            # Calculate match score based on description and operations
            desc_score = self._calculate_similarity(capability_text, mcp.description.lower())
            ops_score = max([
                self._calculate_similarity(capability_text, op.lower()) 
                for op in mcp.supported_operations
            ] + [0.0])
            
            match_score = max(desc_score, ops_score * 0.8)
            
            if match_score > 0.5:  # Reasonable match
                match_type = "exact" if match_score > 0.8 else "partial"
                
                matches.append(MCPMatch(
                    capability=mcp,
                    match_type=match_type,
                    match_score=match_score,
                    setup_needed=mcp.setup_requirements,
                    suggested_parameters=self._suggest_mcp_parameters(mcp, capability_text)
                ))
        
        return matches
    
    async def _match_by_api_patterns(self, description: str, context: Dict[str, Any]) -> List[MCPMatch]:
        """Find MCPs that can work with detected APIs."""
        matches = []
        
        # Extract potential API URLs/services from description and context
        api_indicators = self._extract_api_indicators(description, context)
        
        for indicator in api_indicators:
            # Find MCPs that can handle this API pattern
            compatible_mcps = self._find_api_compatible_mcps(indicator)
            
            for mcp in compatible_mcps:
                matches.append(MCPMatch(
                    capability=mcp,
                    match_type="api_compatible",
                    match_score=0.7,  # Good but requires setup
                    setup_needed=mcp.setup_requirements + [f"API connection: {indicator}"],
                    suggested_parameters={"api_endpoint": indicator}
                ))
        
        return matches
    
    async def _match_by_software(self, description: str, context: Dict[str, Any]) -> List[MCPMatch]:
        """Find MCPs for specific software integrations."""
        matches = []
        
        # Extract software names from description
        mentioned_software = self._extract_software_mentions(description, context)
        
        for software in mentioned_software:
            # Find MCPs that integrate with this software
            compatible_mcps = self._find_software_compatible_mcps(software)
            
            for mcp in compatible_mcps:
                matches.append(MCPMatch(
                    capability=mcp,
                    match_type="software_compatible", 
                    match_score=0.8,  # High relevance for software integration
                    setup_needed=mcp.setup_requirements + [f"Access to {software}"],
                    suggested_parameters={"software": software}
                ))
        
        return matches
    
    def _load_api_patterns(self) -> Dict[str, List[str]]:
        """Load patterns to identify API types."""
        return {
            "weather": ["weather", "forecast", "climate", "temperature"],
            "finance": ["stock", "market", "price", "trading", "financial"],
            "social": ["twitter", "facebook", "instagram", "social"],
            "communication": ["email", "sms", "message", "chat"],
            "database": ["database", "sql", "query", "table"],
            "file": ["file", "storage", "document", "upload"],
            "analytics": ["analytics", "metrics", "tracking", "data"]
        }
    
    def _load_software_patterns(self) -> Dict[str, List[str]]:
        """Load patterns to identify software mentions."""
        return {
            "github": ["github", "git", "repository", "repo", "code"],
            "slack": ["slack", "team chat", "workspace"],
            "google_sheets": ["google sheets", "spreadsheet", "excel", "sheets"],
            "supabase": ["supabase", "database", "backend"],
            "weather": ["weather", "forecast", "temperature", "climate"],
            "serper": ["search", "web search", "google search", "find", "lookup", "research"]
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simplified)."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_api_indicators(self, description: str, context: Dict[str, Any]) -> List[str]:
        """Extract potential API indicators from text."""
        indicators = []
        text = description.lower()
        
        # Look for service names
        service_patterns = ["weather", "stocks", "github", "slack", "database", "sheets", "email"]
        for service in service_patterns:
            if service in text:
                indicators.append(service)
        
        return list(set(indicators))
    
    def _extract_software_mentions(self, description: str, context: Dict[str, Any]) -> List[str]:
        """Extract software names mentioned in text."""
        software_mentions = []
        text = description.lower()
        
        for software, patterns in self.software_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    software_mentions.append(software)
                    break
        
        return list(set(software_mentions))
    
    def _find_api_compatible_mcps(self, api_indicator: str) -> List[MCPCapability]:
        """Find MCPs compatible with an API indicator."""
        compatible = []
        
        for mcp in self.known_mcps:
            # Check if any API pattern matches
            for pattern in mcp.api_patterns:
                if pattern.replace("*", "") in api_indicator or api_indicator in pattern:
                    compatible.append(mcp)
                    break
            
            # Also check software names
            if api_indicator in mcp.software_names:
                compatible.append(mcp)
        
        return compatible
    
    def _find_software_compatible_mcps(self, software: str) -> List[MCPCapability]:
        """Find MCPs compatible with specific software."""
        compatible = []
        
        for mcp in self.known_mcps:
            if software in mcp.software_names:
                compatible.append(mcp)
        
        return compatible
    
    def _find_mcp_for_tool(self, tool: Dict[str, Any]) -> Optional[MCPCapability]:
        """Find MCP capability that corresponds to a tool."""
        tool_name = tool.get('tool_name', '').lower()
        
        for mcp in self.known_mcps:
            if mcp.name.lower() in tool_name or any(op in tool_name for op in mcp.supported_operations):
                return mcp
        
        return None
    
    def _extract_tool_parameters(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from existing tool."""
        return {
            "tool_id": tool.get('tool_id'),
            "existing_tool": True,
            "description": tool.get('description')
        }
    
    def _suggest_mcp_parameters(self, mcp: MCPCapability, capability_text: str) -> Dict[str, Any]:
        """Suggest parameters for MCP setup."""
        params = {
            "mcp_type": mcp.mcp_type.value,
            "setup_requirements": mcp.setup_requirements
        }
        
        # Add specific suggestions based on MCP type
        if mcp.mcp_type == MCPType.API_SERVICE:
            params["suggested_auth"] = "api_key"
        elif mcp.mcp_type == MCPType.DATABASE:
            params["suggested_auth"] = "connection_string"
        
        return params
    
    def _deduplicate_matches(self, matches: List[MCPMatch]) -> List[MCPMatch]:
        """Remove duplicate matches, keeping the best one."""
        seen_mcps = {}
        
        for match in matches:
            mcp_id = match.capability.mcp_id
            if mcp_id not in seen_mcps or match.match_score > seen_mcps[mcp_id].match_score:
                seen_mcps[mcp_id] = match
        
        return list(seen_mcps.values())
    
    async def suggest_mcp_setup(self, mcp_match: MCPMatch) -> Dict[str, Any]:
        """Generate setup instructions for an MCP."""
        mcp = mcp_match.capability
        
        setup_guide = {
            "mcp_name": mcp.name,
            "description": mcp.description,
            "setup_steps": [],
            "required_info": mcp_match.setup_needed,
            "documentation": mcp.documentation_url,
            "estimated_setup_time": "5-10 minutes"
        }
        
        # Generate specific setup steps
        for requirement in mcp.setup_requirements:
            if "api_key" in requirement.lower():
                setup_guide["setup_steps"].append(f"Obtain API key for {mcp.name}")
            elif "url" in requirement.lower():
                setup_guide["setup_steps"].append(f"Provide service URL for {mcp.name}")
            elif "token" in requirement.lower():
                setup_guide["setup_steps"].append(f"Generate access token for {mcp.name}")
            else:
                setup_guide["setup_steps"].append(f"Configure {requirement}")
        
        setup_guide["setup_steps"].append("Test connection and validate access")
        
        return setup_guide 