#!/usr/bin/env python3
"""
Quick Serper MCP Connection Script

This script helps you quickly connect your Serper.dev API key to the platform.
It tests the connection and sets up the MCP for immediate use.
"""

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.run_cards.serper_card import SerperMCP

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def connect_serper(api_key: str):
    """Connect and test Serper MCP with the provided API key."""
    
    print("🔍 Connecting to Serper Web Search API...")
    print("="*50)
    
    try:
        # Initialize Serper MCP
        serper = SerperMCP(api_key)
        
        # Test the connection
        print("🧪 Testing API connection...")
        test_result = await serper.test_connection()
        
        if test_result["success"]:
            print("✅ Serper API connection successful!")
            print(f"   API Key: {'*' * 10}{api_key[-5:]}")
            print(f"   Status: {test_result['message']}")
        else:
            print("❌ Serper API connection failed!")
            print(f"   Error: {test_result['error']}")
            return False
        
        # Test web search
        print("\n🔍 Testing web search functionality...")
        search_result = await serper.web_search("latest AI news", num_results=3)
        
        if search_result.get("success"):
            print("✅ Web search test successful!")
            print(f"   Query: 'latest AI news'")
            print(f"   Results found: {len(search_result.get('organic', []))}")
            
            # Show sample results
            for i, result in enumerate(search_result.get('organic', [])[:2]):
                print(f"   {i+1}. {result.get('title', 'No title')}")
                print(f"      {result.get('link', 'No link')}")
        else:
            print("❌ Web search test failed!")
            print(f"   Error: {search_result.get('error')}")
            return False
        
        # Test news search
        print("\n📰 Testing news search functionality...")
        news_result = await serper.news_search("artificial intelligence", num_results=2)
        
        if news_result.get("success"):
            print("✅ News search test successful!")
            print(f"   Query: 'artificial intelligence'")
            print(f"   News articles found: {len(news_result.get('news', []))}")
        else:
            print("⚠️ News search test failed (this is optional)")
        
        print("\n🎉 Serper MCP is ready to use!")
        print("\n📋 Available capabilities:")
        print("   • Web search - Real-time Google search results")
        print("   • News search - Latest news articles")  
        print("   • Image search - Find images")
        print("   • Places search - Local business and location data")
        
        print("\n💡 Usage in agents:")
        print("   When you ask agents to 'search for X' or 'find information about Y',")
        print("   they will automatically detect the need for web search and use Serper!")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    """Main function to handle API key input and connection."""
    
    print("🚀 Serper MCP Connection Setup")
    print("=" * 50)
    
    # Check for environment variable first
    api_key = os.getenv("SERPER_API_KEY")
    
    if api_key:
        print(f"✅ Found API key in environment: {'*' * 10}{api_key[-5:]}")
    else:
        print("🔑 Please enter your Serper.dev API key:")
        print("   (Get one free at: https://serper.dev/)")
        api_key = input("API Key: ").strip()
        
        if not api_key:
            print("❌ No API key provided. Exiting.")
            return
    
    # Validate API key format (basic check)
    if len(api_key) < 10:
        print("⚠️ API key seems too short. Please check it.")
        return
    
    # Connect to Serper
    try:
        success = asyncio.run(connect_serper(api_key))
        
        if success:
            print("\n✅ Setup completed successfully!")
            print("\n🔧 To use with the platform:")
            print("   1. Set environment variable: export SERPER_API_KEY='your_key'")
            print("   2. Restart your agents/platform")
            print("   3. Ask agents to search for things!")
            
            print(f"\n💾 Save this for your .env file:")
            print(f"SERPER_API_KEY='{api_key}'")
            
        else:
            print("\n❌ Setup failed. Please check your API key and try again.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 