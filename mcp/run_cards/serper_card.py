"""
Serper Web Search MCP Card

Provides web search capabilities using the Serper.dev API.
This MCP connects to Serper's Google Search API for real-time search results.
"""

import asyncio
import logging
import json
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SerperMCP:
    """
    Serper Web Search MCP implementation.
    
    Provides access to:
    - Web search results
    - News search
    - Image search
    - Video search
    - Places search
    """
    
    def __init__(self, api_key: str):
        """Initialize Serper MCP with API key."""
        self.api_key = api_key
        self.base_url = "https://google.serper.dev"
        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        logger.info("🔍 Serper Web Search MCP initialized")
    
    async def web_search(self, query: str, num_results: int = 10, 
                        country: str = "us", language: str = "en") -> Dict[str, Any]:
        """
        Perform web search using Serper API.
        
        Args:
            query: Search query
            num_results: Number of results to return (max 100)
            country: Country code for localized results
            language: Language code for results
            
        Returns:
            Search results with organic results, answer box, etc.
        """
        try:
            search_data = {
                "q": query,
                "num": min(num_results, 100),
                "gl": country,
                "hl": language
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    headers=self.headers,
                    json=search_data
                )
                response.raise_for_status()
                
                results = response.json()
                
                # Process and format results
                formatted_results = await self._format_search_results(results, "web")
                
                logger.info(f"🔍 Web search completed: '{query}' returned {len(formatted_results.get('organic', []))} results")
                return formatted_results
                
        except httpx.HTTPError as e:
            logger.error(f"Serper API error: {e}")
            return {"error": f"Search API error: {str(e)}", "success": False}
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {"error": f"Search failed: {str(e)}", "success": False}
    
    async def news_search(self, query: str, num_results: int = 10, 
                         time_range: str = "1d") -> Dict[str, Any]:
        """
        Search for news articles.
        
        Args:
            query: News search query
            num_results: Number of results to return
            time_range: Time range (1h, 1d, 1w, 1m, 1y)
            
        Returns:
            News search results
        """
        try:
            search_data = {
                "q": query,
                "num": min(num_results, 100),
                "tbs": f"qdr:{time_range}"
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/news",
                    headers=self.headers,
                    json=search_data
                )
                response.raise_for_status()
                
                results = response.json()
                formatted_results = await self._format_search_results(results, "news")
                
                logger.info(f"📰 News search completed: '{query}' returned {len(formatted_results.get('news', []))} articles")
                return formatted_results
                
        except Exception as e:
            logger.error(f"News search error: {e}")
            return {"error": f"News search failed: {str(e)}", "success": False}
    
    async def images_search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """
        Search for images.
        
        Args:
            query: Image search query
            num_results: Number of results to return
            
        Returns:
            Image search results
        """
        try:
            search_data = {
                "q": query,
                "num": min(num_results, 100)
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/images",
                    headers=self.headers,
                    json=search_data
                )
                response.raise_for_status()
                
                results = response.json()
                formatted_results = await self._format_search_results(results, "images")
                
                logger.info(f"🖼️ Image search completed: '{query}' returned {len(formatted_results.get('images', []))} images")
                return formatted_results
                
        except Exception as e:
            logger.error(f"Image search error: {e}")
            return {"error": f"Image search failed: {str(e)}", "success": False}
    
    async def places_search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """
        Search for places and local businesses.
        
        Args:
            query: Places search query
            num_results: Number of results to return
            
        Returns:
            Places search results
        """
        try:
            search_data = {
                "q": query,
                "num": min(num_results, 100)
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/places",
                    headers=self.headers,
                    json=search_data
                )
                response.raise_for_status()
                
                results = response.json()
                formatted_results = await self._format_search_results(results, "places")
                
                logger.info(f"📍 Places search completed: '{query}' returned {len(formatted_results.get('places', []))} places")
                return formatted_results
                
        except Exception as e:
            logger.error(f"Places search error: {e}")
            return {"error": f"Places search failed: {str(e)}", "success": False}
    
    async def _format_search_results(self, raw_results: Dict[str, Any], 
                                   search_type: str) -> Dict[str, Any]:
        """Format search results for consistent output."""
        formatted = {
            "success": True,
            "search_type": search_type,
            "timestamp": datetime.utcnow().isoformat(),
            "query": raw_results.get("searchParameters", {}).get("q", ""),
            "total_results": raw_results.get("searchInformation", {}).get("totalResults", 0)
        }
        
        # Add answer box if present
        if "answerBox" in raw_results:
            formatted["answer_box"] = raw_results["answerBox"]
        
        # Add knowledge graph if present
        if "knowledgeGraph" in raw_results:
            formatted["knowledge_graph"] = raw_results["knowledgeGraph"]
        
        # Format specific result types
        if search_type == "web":
            formatted["organic"] = raw_results.get("organic", [])
            formatted["people_also_ask"] = raw_results.get("peopleAlsoAsk", [])
            formatted["related_searches"] = raw_results.get("relatedSearches", [])
            
        elif search_type == "news":
            formatted["news"] = raw_results.get("news", [])
            
        elif search_type == "images":
            formatted["images"] = raw_results.get("images", [])
            
        elif search_type == "places":
            formatted["places"] = raw_results.get("places", [])
            
        return formatted
    
    async def search(self, query: str, search_type: str = "web", **kwargs) -> Dict[str, Any]:
        """
        Universal search method that routes to appropriate search type.
        
        Args:
            query: Search query
            search_type: Type of search (web, news, images, places)
            **kwargs: Additional search parameters
            
        Returns:
            Search results
        """
        search_methods = {
            "web": self.web_search,
            "news": self.news_search,
            "images": self.images_search,
            "places": self.places_search
        }
        
        method = search_methods.get(search_type, self.web_search)
        return await method(query, **kwargs)
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test the Serper API connection."""
        try:
            test_result = await self.web_search("test", num_results=1)
            
            if test_result.get("success", False):
                return {
                    "success": True,
                    "message": "Serper API connection successful",
                    "api_key_valid": True
                }
            else:
                return {
                    "success": False,
                    "message": "Serper API connection failed",
                    "error": test_result.get("error", "Unknown error")
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": "Serper API connection test failed",
                "error": str(e)
            }

# Tool functions for MCP integration
async def serper_web_search(query: str, num_results: int = 10, **kwargs) -> Dict[str, Any]:
    """Web search tool function."""
    # This would get the API key from credential store in production
    api_key = kwargs.get('api_key', 'your_serper_api_key_here')
    
    serper = SerperMCP(api_key)
    return await serper.web_search(query, num_results, **kwargs)

async def serper_news_search(query: str, num_results: int = 10, **kwargs) -> Dict[str, Any]:
    """News search tool function."""
    api_key = kwargs.get('api_key', 'your_serper_api_key_here')
    
    serper = SerperMCP(api_key)
    return await serper.news_search(query, num_results, **kwargs)

# MCP tool registry entries
SERPER_TOOLS = [
    {
        "tool_id": "serper_web_search",
        "tool_name": "Serper Web Search",
        "description": "Search the web using Serper.dev API for real-time results",
        "function": serper_web_search,
        "parameters": {
            "query": {"type": "string", "required": True, "description": "Search query"},
            "num_results": {"type": "integer", "required": False, "description": "Number of results (max 100)"},
            "country": {"type": "string", "required": False, "description": "Country code for localized results"},
            "language": {"type": "string", "required": False, "description": "Language code for results"}
        }
    },
    {
        "tool_id": "serper_news_search", 
        "tool_name": "Serper News Search",
        "description": "Search for news articles using Serper.dev API",
        "function": serper_news_search,
        "parameters": {
            "query": {"type": "string", "required": True, "description": "News search query"},
            "num_results": {"type": "integer", "required": False, "description": "Number of results"},
            "time_range": {"type": "string", "required": False, "description": "Time range (1h, 1d, 1w, 1m, 1y)"}
        }
    }
] 