#!/usr/bin/env python3
"""
Simple test runner for AI-First Architecture Integration Test

This script helps run the comprehensive AI ecosystem test with proper
environment variable checking and setup.
"""

import os
import sys
import asyncio
from datetime import datetime

def check_environment():
    """Check if required environment variables are set."""
    print("🔍 Checking environment variables...")
    
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key for LLM integration',
        'SUPABASE_URL': 'Supabase project URL for database features', 
        'SUPABASE_KEY': 'Supabase API key for database features'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append((var, description))
            print(f"❌ {var}: Not set ({description})")
        else:
            masked_value = f"{os.getenv(var)[:8]}..." if len(os.getenv(var)) > 8 else "***"
            print(f"✅ {var}: {masked_value}")
    
    if missing_vars:
        print("\n⚠️  Missing environment variables detected:")
        for var, desc in missing_vars:
            print(f"   {var}: {desc}")
        
        if 'OPENAI_API_KEY' in [var for var, _ in missing_vars]:
            print("\n❌ OPENAI_API_KEY is required to run the test.")
            print("   Please set it with: export OPENAI_API_KEY='your-api-key-here'")
            return False
        else:
            print("\n⚠️  Some features may not work properly without all variables.")
            print("   Test will continue with limited functionality.")
    
    print("✅ Environment check completed\n")
    return True

async def run_ai_test():
    """Run the AI-first architecture integration test."""
    print("🚀 Starting AI-First Architecture Integration Test")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print(f"Python version: {sys.version}")
    print()
    
    # Check environment
    if not check_environment():
        return False
    
    try:
        # Import and run the test
        from ai_ecosystem_integration_test import AIEcosystemIntegrationTest
        
        test_suite = AIEcosystemIntegrationTest()
        results = await test_suite.run_comprehensive_test()
        
        # Print final summary
        print("\n🎯 FINAL TEST SUMMARY")
        print("=" * 40)
        print(f"Test ID: {results['test_id']}")
        print(f"Overall Success: {'✅ PASS' if results['overall_success'] else '❌ FAIL'}")
        print(f"Success Rate: {results['summary']['success_rate']:.1%}")
        print(f"AI Ecosystem Health: {results['summary']['ai_ecosystem_health'].upper()}")
        
        return results['overall_success']
        
    except ImportError as e:
        print(f"❌ Failed to import test module: {e}")
        print("   Make sure ai_ecosystem_integration_test.py is in the current directory")
        return False
    
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def main():
    """Main entry point."""
    try:
        success = asyncio.run(run_ai_test())
        exit_code = 0 if success else 1
        
        print(f"\n🏁 Test completed with exit code: {exit_code}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 