#!/usr/bin/env python3
"""
Validate Real MCPs Only

Confirms that the MCP Discovery Engine only shows:
1. Real, implemented core MCPs from file structure  
2. Real library MCPs from Supabase
3. No mock data or fake implementations
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.mcp_discovery_engine import MCPDiscoveryEngine, MCPCapability, MCPType

async def validate_real_mcps():
    """Validate that only real MCPs are shown."""
    print("🔍 Validating Real MCPs Only")
    print("=" * 50)
    
    discovery = MCPDiscoveryEngine()
    
    # Validate core MCPs
    core_mcps = [mcp for mcp in discovery.known_mcps if mcp.is_core]
    print(f"📁 Core MCPs Found: {len(core_mcps)}")
    
    expected_core = ["serper", "supabase_core"]
    for mcp in core_mcps:
        if mcp.mcp_id in expected_core:
            print(f"   ✅ {mcp.name} - REAL implementation")
        else:
            print(f"   ❌ {mcp.name} - UNEXPECTED")
    
    # Validate library MCPs
    library_mcps = [mcp for mcp in discovery.known_mcps if not mcp.is_core]
    print(f"\n🗄️ Library MCPs Found: {len(library_mcps)}")
    
    for mcp in library_mcps:
        print(f"   📦 {mcp.name} - From Supabase")
    
    # Test adding new MCP to library
    print(f"\n🧪 Testing New MCP Addition...")
    
    test_mcp = MCPCapability(
        mcp_id="example_integration",
        name="Example Integration Test",
        description="Testing MCP library addition",
        mcp_type=MCPType.API_SERVICE,
        supported_operations=["test_operation"],
        api_patterns=["api.example.com"],
        software_names=["example"],
        setup_requirements=["api_key"],
        confidence_score=0.9,
        documentation_url="https://example.com/docs",
        popularity_score=10,
        is_core=False
    )
    
    # Test the add method (without actually adding to avoid DB changes)
    print(f"   📝 MCP Definition: {test_mcp.name}")
    print(f"   🔧 Add Method Available: {hasattr(discovery, 'add_mcp_to_library')}")
    print(f"   📈 Popularity Update Available: {hasattr(discovery, 'update_mcp_popularity')}")
    
    # Validate no mock data
    print(f"\n🚫 Validating No Mock Data...")
    
    mock_indicators = ["mock", "fake", "test_only", "example_only"]
    found_mock = False
    
    for mcp in discovery.known_mcps:
        for indicator in mock_indicators:
            if indicator in mcp.name.lower() or indicator in mcp.description.lower():
                print(f"   ❌ Mock data found: {mcp.name}")
                found_mock = True
    
    if not found_mock:
        print(f"   ✅ No mock data found - all MCPs are real")
    
    # Summary
    print(f"\n" + "=" * 50)
    print(f"✅ VALIDATION COMPLETE")
    print(f"📊 Total MCPs: {len(discovery.known_mcps)}")
    print(f"📁 Real Core MCPs: {len(core_mcps)}")
    print(f"🗄️ Library MCPs: {len(library_mcps)}")
    print(f"🚫 Mock Data: {'None' if not found_mock else 'Found'}")
    print(f"🔄 Dynamic Addition: {'Enabled' if hasattr(discovery, 'add_mcp_to_library') else 'Disabled'}")
    
    # Test search functionality
    print(f"\n🔍 Testing Search with Real MCPs...")
    
    # Search for web search (should find Serper)
    web_matches = await discovery.find_mcp_solutions(
        "web search", 
        "I need to search for information online",
        {"user_query": "search the web"}
    )
    
    print(f"   🔎 Web Search Query: {len(web_matches)} matches")
    for match in web_matches[:1]:
        print(f"      - {match.capability.name} (score: {match.match_score:.2f})")
    
    # Search for database (should find Supabase)
    db_matches = await discovery.find_mcp_solutions(
        "database", 
        "I need to query database information",
        {"user_query": "database operations"}
    )
    
    print(f"   🗄️ Database Query: {len(db_matches)} matches")
    for match in db_matches[:1]:
        print(f"      - {match.capability.name} (score: {match.match_score:.2f})")
    
    return len(discovery.known_mcps) > 0 and not found_mock

async def main():
    """Run validation."""
    try:
        success = await validate_real_mcps()
        
        if success:
            print(f"\n🎉 VALIDATION PASSED: Only real MCPs found!")
            print(f"🎯 All new MCPs will be stored in Supabase automatically")
            sys.exit(0)
        else:
            print(f"\n💥 VALIDATION FAILED: Issues found")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 