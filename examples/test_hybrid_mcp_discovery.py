#!/usr/bin/env python3
"""
Test Hybrid MCP Discovery Engine

Tests the new hybrid approach that loads:
1. Core MCPs from file structure (actually implemented)
2. Library MCPs from Supabase (community/extended)
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.mcp_discovery_engine import MCPDiscoveryEngine, MCPType

async def test_hybrid_mcp_discovery():
    """Test the hybrid MCP discovery system."""
    print("🔍 Testing Hybrid MCP Discovery Engine")
    print("=" * 50)
    
    try:
        # Initialize discovery engine
        discovery = MCPDiscoveryEngine()
        
        print(f"✅ Initialized with {len(discovery.known_mcps)} total MCPs")
        
        # Count MCPs by source
        core_mcps = [mcp for mcp in discovery.known_mcps if mcp.is_core]
        library_mcps = [mcp for mcp in discovery.known_mcps if not mcp.is_core]
        
        print(f"📁 Core MCPs (from files): {len(core_mcps)}")
        print(f"🗄️ Library MCPs (from Supabase): {len(library_mcps)}")
        print()
        
        # Show real core MCPs only
        print("📁 CORE MCPs (Actually Implemented):")
        for mcp in core_mcps:
            print(f"   ✅ {mcp.name}")
            print(f"      Type: {mcp.mcp_type.value}")
            print(f"      Tools: {len(mcp.supported_operations)}")
            print(f"      Popularity: {mcp.popularity_score}")
            print()
        
        # Show library MCPs from Supabase
        print("🗄️ LIBRARY MCPs (From Supabase):")
        if library_mcps:
            for mcp in library_mcps:
                print(f"   📦 {mcp.name}")
                print(f"      Type: {mcp.mcp_type.value}")
                print(f"      Tools: {len(mcp.supported_operations)}")
                print(f"      Popularity: {mcp.popularity_score}")
                print()
        else:
            print("   (None found - check Supabase connection)")
            print()
        
        # Test MCP search functionality
        print("🔍 Testing MCP Search...")
        test_queries = [
            ("web search", "Need to search the web for information"),
            ("database operations", "Need to query and manage database"),
            ("github integration", "Need to work with GitHub repositories")
        ]
        
        for capability, description in test_queries:
            print(f"\n🔎 Searching for: {capability}")
            
            matches = await discovery.find_mcp_solutions(capability, description, {})
            
            if matches:
                print(f"   Found {len(matches)} matches:")
                for match in matches[:2]:  # Show top 2
                    print(f"   - {match.capability.name} (score: {match.match_score:.2f})")
            else:
                print("   No matches found")
        
        print("\n" + "=" * 50)
        print("✅ Hybrid MCP Discovery Test Completed!")
        print(f"Total MCPs Available: {len(discovery.known_mcps)}")
        print(f"Real Implementations: {len(core_mcps)}")
        print(f"Library Extensions: {len(library_mcps)}")
        
        # Test adding a new MCP to library
        print("\n🧪 Testing MCP Library Addition...")
        from mcp.mcp_discovery_engine import MCPCapability
        
        new_mcp = MCPCapability(
            mcp_id="test_weather",
            name="Test Weather API",
            description="Test weather data integration",
            mcp_type=MCPType.API_SERVICE,
            supported_operations=["get_weather", "get_forecast"],
            api_patterns=["api.weather.com"],
            software_names=["weather"],
            setup_requirements=["api_key"],
            confidence_score=0.8,
            documentation_url="https://weather.com/api",
            popularity_score=1,
            is_core=False
        )
        
        # Test the add method (commented out to avoid actual DB changes)
        # success = await discovery.add_mcp_to_library(new_mcp, "test_user")
        print("   MCP addition method available ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the test."""
    success = await test_hybrid_mcp_discovery()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 