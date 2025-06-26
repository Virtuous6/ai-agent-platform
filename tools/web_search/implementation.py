"""
Filename: implementation.py
Purpose: Web search tool with intelligent dual-engine routing
Dependencies: httpx, aiohttp, beautifulsoup4, pydantic

This module is part of the AI Agent Platform.
See README.llm.md in this directory for context.
"""

import os
import re
import json
import time
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from urllib.parse import urlparse, quote_plus

import httpx
import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)

# Configuration
SEARCH_ROUTING_CONFIG = {
    "local": "serp",
    "product": "serp", 
    "map": "serp",
    "snippet": "serp",
    "FAQ": "serp",
    "shopping": "serp",
    "general": "lobstr",
    "research": "lobstr",
    "news": "lobstr",
    "academic": "lobstr"
}

# Credible domains for scoring
CREDIBLE_DOMAINS = {
    "wikipedia.org": 0.95,
    "github.com": 0.90,
    "stackoverflow.com": 0.88,
    "docs.python.org": 0.92,
    "mozilla.org": 0.90,
    "w3.org": 0.95,
    "ieee.org": 0.93,
    "nature.com": 0.94,
    "arxiv.org": 0.85,
    "medium.com": 0.75,
    "reddit.com": 0.65,
    "quora.com": 0.60
}

class SearchEngineError(Exception):
    """Engine-specific API failure"""
    pass

class RateLimitError(Exception):
    """API quota exceeded"""
    pass

class InvalidQueryError(Exception):
    """Malformed search query"""
    pass

class NetworkError(Exception):
    """Connection timeout or failure"""
    pass

class SearchResult(BaseModel):
    """Individual search result model"""
    title: str
    url: str
    snippet: Optional[str] = None
    source: str = Field(..., pattern="^(lobstr|serp)$")
    credibility_score: float = Field(0.5, ge=0, le=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchMetadata(BaseModel):
    """Search execution metadata"""
    query_processed: str
    engine_used: str = Field(..., pattern="^(lobstr|serp)$")
    routing_reason: str
    total_results: int = 0
    search_time_ms: float
    cache_hit: bool = False

class WebSearchResponse(BaseModel):
    """Complete web search response"""
    results: List[SearchResult]
    search_metadata: SearchMetadata
    summary: Optional[str] = None
    related_queries: List[str] = Field(default_factory=list)

class SearchCache:
    """Simple in-memory cache for search results"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl_seconds = ttl_seconds
    
    def _get_cache_key(self, query: str, **kwargs) -> str:
        """Generate cache key from query and parameters"""
        cache_data = {"query": query, **kwargs}
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached result if not expired"""
        key = self._get_cache_key(query, **kwargs)
        if key in self.cache:
            cached_item = self.cache[key]
            if time.time() - cached_item["timestamp"] < self.ttl_seconds:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_item["data"]
            else:
                del self.cache[key]
        return None
    
    def set(self, query: str, data: Dict[str, Any], **kwargs):
        """Store result in cache"""
        key = self._get_cache_key(query, **kwargs)
        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }
        logger.debug(f"Cached result for query: {query[:50]}...")

class WebSearchTool:
    """
    Intelligent web search tool with dual-engine routing.
    
    Routes queries to optimal search engines based on query type:
    - Lobstr.io for general research and broad queries
    - SERP API for structured data and precision tasks
    """
    
    def __init__(self):
        """Initialize the web search tool with configuration."""
        self.lobstr_api_key = os.getenv("LOBSTR_API_KEY")
        self.serp_api_key = os.getenv("SERP_API_KEY") 
        self.default_results = int(os.getenv("WEB_SEARCH_DEFAULT_RESULTS", "10"))
        self.timeout_seconds = int(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "30"))
        self.cache_ttl = int(os.getenv("WEB_SEARCH_CACHE_TTL", "3600"))
        
        self.cache = SearchCache(ttl_seconds=self.cache_ttl)
        
        # HTTP clients
        self.http_client = httpx.AsyncClient(timeout=self.timeout_seconds)
        
        logger.info("Web Search Tool initialized")
    
    async def execute(self, query: str, max_results: int = None, 
                     search_type: Optional[str] = None, include_snippets: bool = True,
                     location: Optional[str] = None, language: str = "en",
                     safe_search: bool = True) -> Dict[str, Any]:
        """
        Main search execution with intelligent routing.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            search_type: Force specific search type for routing
            include_snippets: Include content snippets in results
            location: Geographic location for local searches
            language: Language preference (ISO 639-1 code)
            safe_search: Enable safe search filtering
            
        Returns:
            Dictionary containing search results and metadata
        """
        start_time = time.time()
        
        try:
            # Validate and clean query
            cleaned_query = self._clean_query(query)
            if not cleaned_query:
                raise InvalidQueryError("Query is empty or invalid")
            
            # Set defaults
            max_results = max_results or self.default_results
            
            # Check cache first
            cache_params = {
                "max_results": max_results,
                "search_type": search_type,
                "location": location,
                "language": language
            }
            
            cached_result = self.cache.get(cleaned_query, **cache_params)
            if cached_result:
                cached_result["search_metadata"]["cache_hit"] = True
                return cached_result
            
            # Determine search engine routing
            engine, routing_reason = self._route_query(cleaned_query, search_type)
            
            # Execute search with selected engine
            if engine == "serp":
                if not self.serp_api_key:
                    logger.warning("SERP API key not found, falling back to Lobstr")
                    engine = "lobstr"
                    routing_reason = "serp_unavailable_fallback"
                else:
                    results = await self._search_with_serp(
                        cleaned_query, max_results, location, language, safe_search
                    )
            
            if engine == "lobstr":
                if not self.lobstr_api_key:
                    logger.warning("Lobstr API key not found, using web scraping fallback")
                    results = await self._search_fallback(cleaned_query, max_results)
                else:
                    results = await self._search_with_lobstr(
                        cleaned_query, max_results, language, safe_search
                    )
            
            # Process and enhance results
            processed_results = await self._process_results(results, include_snippets)
            
            # Generate summary if requested
            summary = await self._generate_summary(processed_results) if len(processed_results) > 0 else None
            
            # Calculate search time
            search_time_ms = (time.time() - start_time) * 1000
            
            # Build response
            response_data = {
                "results": [result.dict() for result in processed_results],
                "search_metadata": {
                    "query_processed": cleaned_query,
                    "engine_used": engine,
                    "routing_reason": routing_reason,
                    "total_results": len(processed_results),
                    "search_time_ms": search_time_ms,
                    "cache_hit": False
                },
                "summary": summary,
                "related_queries": self._generate_related_queries(cleaned_query)
            }
            
            # Cache the result
            self.cache.set(cleaned_query, response_data, **cache_params)
            
            return response_data
            
        except InvalidQueryError:
            raise
        except (RateLimitError, NetworkError, SearchEngineError) as e:
            logger.error(f"Search error: {str(e)}")
            # Try fallback engine or cached results
            return await self._handle_search_error(e, cleaned_query, start_time)
        except Exception as e:
            logger.error(f"Unexpected search error: {str(e)}")
            raise SearchEngineError(f"Search failed: {str(e)}")
    
    def _clean_query(self, query: str) -> str:
        """Clean and validate search query"""
        if not query or not isinstance(query, str):
            return ""
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Basic security: remove potential injection attempts
        cleaned = re.sub(r'[<>"\';]', '', cleaned)
        
        # Limit length
        if len(cleaned) > 500:
            cleaned = cleaned[:500]
        
        return cleaned
    
    def _route_query(self, query: str, forced_type: Optional[str] = None) -> tuple[str, str]:
        """
        Determine optimal search engine based on query analysis.
        
        Args:
            query: Cleaned search query
            forced_type: Override automatic routing
            
        Returns:
            Tuple of (engine, reasoning)
        """
        if forced_type:
            if forced_type in SEARCH_ROUTING_CONFIG:
                engine = SEARCH_ROUTING_CONFIG[forced_type]
                return engine, f"forced_type_{forced_type}"
            else:
                logger.warning(f"Unknown forced search type: {forced_type}")
        
        query_lower = query.lower()
        
        # Local search indicators
        local_patterns = [
            r'\bnear me\b', r'\bnearby\b', r'\bin [a-zA-Z\s,]+\b',
            r'\b(restaurant|store|shop|business|office|clinic|hospital)\b',
            r'\b(address|location|directions|hours)\b'
        ]
        
        for pattern in local_patterns:
            if re.search(pattern, query_lower):
                return "serp", "local_search_detected"
        
        # Product/shopping indicators  
        product_patterns = [
            r'\b(price|cost|buy|purchase|sale|discount|deal)\b',
            r'\b(review|rating|compare|vs|versus)\b',
            r'\b(product|item|shop|store|amazon|ebay)\b'
        ]
        
        for pattern in product_patterns:
            if re.search(pattern, query_lower):
                return "serp", "product_search_detected"
        
        # Snippet/FAQ indicators
        snippet_patterns = [
            r'^(what|how|why|when|where|who)\b',
            r'\b(define|definition|meaning|explain)\b',
            r'\?(what|how|why|when|where)\b'
        ]
        
        for pattern in snippet_patterns:
            if re.search(pattern, query_lower):
                return "serp", "snippet_query_detected"
        
        # Academic/research indicators (prefer Lobstr for comprehensive results)
        research_patterns = [
            r'\b(research|study|analysis|paper|journal|academic)\b',
            r'\b(trends|statistics|data|report|survey)\b',
            r'\b(documentation|docs|tutorial|guide|best practices)\b'
        ]
        
        for pattern in research_patterns:
            if re.search(pattern, query_lower):
                return "lobstr", "research_query_detected"
        
        # Default to Lobstr for general queries
        return "lobstr", "general_query_default"
    
    async def _search_with_lobstr(self, query: str, max_results: int, 
                                 language: str, safe_search: bool) -> List[Dict[str, Any]]:
        """Search using Lobstr.io API"""
        try:
            url = "https://api.lobstr.io/search"
            headers = {
                "Authorization": f"Bearer {self.lobstr_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": query,
                "max_results": max_results,
                "language": language,
                "safe_search": safe_search
            }
            
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
                
                if response.status_code == 429:
                    raise RateLimitError("Lobstr API rate limit exceeded")
                elif response.status_code != 200:
                    raise SearchEngineError(f"Lobstr API error: {response.status_code}")
                
                data = response.json()
                return data.get("results", [])
                
        except httpx.TimeoutException:
            raise NetworkError("Lobstr API request timeout")
        except Exception as e:
            raise SearchEngineError(f"Lobstr search failed: {str(e)}")
    
    async def _search_with_serp(self, query: str, max_results: int,
                               location: Optional[str], language: str, 
                               safe_search: bool) -> List[Dict[str, Any]]:
        """Search using SERP API"""
        try:
            url = "https://serpapi.com/search"
            params = {
                "api_key": self.serp_api_key,
                "engine": "google",
                "q": query,
                "num": max_results,
                "hl": language
            }
            
            if location:
                params["location"] = location
            
            if safe_search:
                params["safe"] = "active"
            
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url, params=params)
                
                if response.status_code == 429:
                    raise RateLimitError("SERP API rate limit exceeded")
                elif response.status_code != 200:
                    raise SearchEngineError(f"SERP API error: {response.status_code}")
                
                data = response.json()
                
                # Extract organic results
                organic_results = data.get("organic_results", [])
                results = []
                
                for item in organic_results[:max_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "serp"
                    })
                
                return results
                
        except httpx.TimeoutException:
            raise NetworkError("SERP API request timeout")
        except Exception as e:
            raise SearchEngineError(f"SERP search failed: {str(e)}")
    
    async def _search_fallback(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback search using web scraping (DuckDuckGo)"""
        logger.info("Using fallback search method")
        
        try:
            # Simple DuckDuckGo scraping as fallback
            search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
            
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(search_url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })
                
                if response.status_code != 200:
                    raise SearchEngineError(f"Fallback search failed: {response.status_code}")
                
                soup = BeautifulSoup(response.content, 'html.parser')
                results = []
                
                # Extract search results from DuckDuckGo HTML
                for result_div in soup.find_all('div', class_='result__body')[:max_results]:
                    title_elem = result_div.find('h2', class_='result__title')
                    snippet_elem = result_div.find('div', class_='result__snippet')
                    
                    if title_elem and title_elem.find('a'):
                        title = title_elem.get_text().strip()
                        url = title_elem.find('a').get('href', '')
                        snippet = snippet_elem.get_text().strip() if snippet_elem else ""
                        
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "source": "fallback"
                        })
                
                return results
                
        except Exception as e:
            logger.error(f"Fallback search failed: {str(e)}")
            return []
    
    async def _process_results(self, raw_results: List[Dict[str, Any]], 
                              include_snippets: bool) -> List[SearchResult]:
        """Process and enhance raw search results"""
        processed = []
        seen_urls = set()
        
        for result in raw_results:
            # Skip duplicates
            url = result.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            # Calculate credibility score
            credibility_score = self._calculate_credibility_score(url)
            
            # Extract domain and metadata
            domain = urlparse(url).netloc.lower()
            metadata = {
                "domain": domain,
                "content_type": self._detect_content_type(result.get("title", "")),
            }
            
            # Build processed result
            processed_result = SearchResult(
                title=result.get("title", ""),
                url=url,
                snippet=result.get("snippet", "") if include_snippets else None,
                source=result.get("source", "unknown"),
                credibility_score=credibility_score,
                metadata=metadata
            )
            
            processed.append(processed_result)
        
        # Sort by credibility score
        processed.sort(key=lambda x: x.credibility_score, reverse=True)
        
        return processed
    
    def _calculate_credibility_score(self, url: str) -> float:
        """Calculate credibility score based on domain and other factors"""
        try:
            domain = urlparse(url).netloc.lower()
            
            # Remove www prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check against known credible domains
            for credible_domain, score in CREDIBLE_DOMAINS.items():
                if domain == credible_domain or domain.endswith('.' + credible_domain):
                    return score
            
            # Default scoring based on domain characteristics
            if domain.endswith('.edu'):
                return 0.85
            elif domain.endswith('.gov'):
                return 0.90
            elif domain.endswith('.org'):
                return 0.75
            elif domain.endswith('.com'):
                return 0.60
            else:
                return 0.50
                
        except Exception:
            return 0.50
    
    def _detect_content_type(self, title: str) -> str:
        """Detect content type based on title patterns"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['video', 'watch', 'youtube']):
            return "video"
        elif any(word in title_lower for word in ['tutorial', 'guide', 'how to']):
            return "tutorial"
        elif any(word in title_lower for word in ['news', 'breaking', 'report']):
            return "news"
        elif any(word in title_lower for word in ['paper', 'journal', 'research']):
            return "academic"
        elif any(word in title_lower for word in ['documentation', 'docs', 'api']):
            return "documentation"
        else:
            return "article"
    
    async def _generate_summary(self, results: List[SearchResult]) -> Optional[str]:
        """Generate AI summary of search results"""
        if not results:
            return None
        
        # Simple extractive summary for now
        # In production, this could use an LLM for better summaries
        snippets = [r.snippet for r in results[:3] if r.snippet]
        
        if snippets:
            combined_text = " ".join(snippets)
            # Truncate to reasonable length
            if len(combined_text) > 500:
                combined_text = combined_text[:500] + "..."
            
            return f"Key findings: {combined_text}"
        
        return None
    
    def _generate_related_queries(self, original_query: str) -> List[str]:
        """Generate related search queries"""
        # Simple related query generation
        # In production, this could use more sophisticated methods
        related = []
        
        words = original_query.lower().split()
        
        if len(words) > 1:
            # Add variations
            related.append(f"{original_query} tutorial")
            related.append(f"{original_query} examples")
            related.append(f"how to {original_query}")
            
        return related[:3]  # Limit to 3 suggestions
    
    async def _handle_search_error(self, error: Exception, query: str, 
                                  start_time: float) -> Dict[str, Any]:
        """Handle search errors with fallback strategies"""
        search_time_ms = (time.time() - start_time) * 1000
        
        # Try to return cached results if available
        cached_result = self.cache.get(query)
        if cached_result:
            logger.info("Returning cached results due to search error")
            cached_result["search_metadata"]["cache_hit"] = True
            cached_result["search_metadata"]["error_fallback"] = str(error)
            return cached_result
        
        # Return empty results with error information
        return {
            "results": [],
            "search_metadata": {
                "query_processed": query,
                "engine_used": "error",
                "routing_reason": "error_fallback",
                "total_results": 0,
                "search_time_ms": search_time_ms,
                "cache_hit": False,
                "error": str(error)
            },
            "summary": None,
            "related_queries": []
        }
    
    async def close(self):
        """Clean up resources"""
        if hasattr(self, 'http_client'):
            await self.http_client.aclose()

# MCP Tool Registration
async def web_search(query: str, **kwargs) -> Dict[str, Any]:
    """
    MCP tool function for web search.
    
    Args:
        query: Search query string
        **kwargs: Additional parameters (max_results, search_type, etc.)
        
    Returns:
        Dictionary containing search results and metadata
    """
    tool = WebSearchTool()
    try:
        return await tool.execute(query, **kwargs)
    finally:
        await tool.close()

# Tool registry entry
TOOL_DEFINITION = {
    "name": "web_search",
    "function": web_search,
    "description": "Intelligent web search with dual-engine routing",
    "parameters": {
        "query": {"type": "string", "required": True},
        "max_results": {"type": "integer", "default": 10},
        "search_type": {"type": "string", "optional": True},
        "include_snippets": {"type": "boolean", "default": True},
        "location": {"type": "string", "optional": True},
        "language": {"type": "string", "default": "en"},
        "safe_search": {"type": "boolean", "default": True}
    }
} 