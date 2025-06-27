#!/usr/bin/env python3
"""
🚀 Run Updated Production Test

Convenience script to run the comprehensive production test with the new:
- Clean MCP Architecture (no mock data)
- Hybrid approach (core + library MCPs)
- Real implementations only
- Dynamic MCP addition validation
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def main():
    """Run the updated production test."""
    print("🎯 Starting Updated Production Test...")
    print("📦 New Features Being Tested:")
    print("   ✅ Clean MCP Architecture")
    print("   ✅ No Mock Data")
    print("   ✅ Hybrid Core + Library MCPs")
    print("   ✅ Real Implementations Only")
    print("   ✅ Dynamic MCP Addition")
    print()
    
    # Import and run the production test
    from tests.production_test import main as run_production_test
    
    try:
        result = await run_production_test()
        
        if result == 0:
            print("\n🎉 UPDATED PRODUCTION TEST PASSED!")
            print("✅ All new MCP architecture features working!")
        else:
            print("\n⚠️ Production test had issues - check output above")
            
        return result
        
    except Exception as e:
        print(f"\n❌ Test execution error: {e}")
        return 1

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result) 