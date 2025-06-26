"""
Test suite for the Web Search Tool implementation.

Tests cover:
- Query routing logic
- Response validation
- Error handling
- Caching behavior
- Credibility scoring
"""

import os
import sys
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from tools.web_search.implementation import (
    WebSearchTool, 
    SearchCache, 
    SearchResult,
    WebSearchResponse,
    SearchMetadata,
    InvalidQueryError,
    SearchEngineError,
    RateLimitError,
    NetworkError
)


class TestSearchCache:
    """Test cases for the SearchCache class"""
    
    def test_cache_initialization(self):
        cache = SearchCache(ttl_seconds=1800)
        assert cache.ttl_seconds == 1800
        assert cache.cache == {}
    
    def test_cache_key_generation(self):
        cache = SearchCache()
        key1 = cache._get_cache_key("test query", max_results=10)
        key2 = cache._get_cache_key("test query", max_results=10)
        key3 = cache._get_cache_key("test query", max_results=5)
        
        assert key1 == key2  # Same parameters should generate same key
        assert key1 != key3  # Different parameters should generate different key
    
    def test_cache_set_and_get(self):
        cache = SearchCache()
        test_data = {"results": [{"title": "Test", "url": "https://test.com"}]}
        
        # Store data
        cache.set("test query", test_data)
        
        # Retrieve data
        retrieved = cache.get("test query")
        assert retrieved == test_data
    
    def test_cache_miss(self):
        cache = SearchCache()
        result = cache.get("nonexistent query")
        assert result is None


class TestWebSearchTool:
    """Test cases for the WebSearchTool class"""
    
    @pytest.fixture
    def search_tool(self):
        """Create a WebSearchTool instance for testing"""
        with patch.dict(os.environ, {
            'LOBSTR_API_KEY': 'test_lobstr_key',
            'SERP_API_KEY': 'test_serp_key',
            'WEB_SEARCH_DEFAULT_RESULTS': '10',
            'WEB_SEARCH_TIMEOUT_SECONDS': '30',
            'WEB_SEARCH_CACHE_TTL': '3600'
        }):
            return WebSearchTool()
    
    def test_initialization(self, search_tool):
        """Test tool initialization with environment variables"""
        assert search_tool.lobstr_api_key == 'test_lobstr_key'
        assert search_tool.serp_api_key == 'test_serp_key'
        assert search_tool.default_results == 10
        assert search_tool.timeout_seconds == 30
        assert search_tool.cache_ttl == 3600
    
    def test_query_cleaning(self, search_tool):
        """Test query cleaning and validation"""
        # Valid queries
        assert search_tool._clean_query("python programming") == "python programming"
        assert search_tool._clean_query("  spaces  ") == "spaces"
        
        # Invalid queries
        assert search_tool._clean_query("") == ""
        assert search_tool._clean_query("   ") == ""
    
    def test_query_routing_general(self, search_tool):
        """Test query routing for general searches"""
        engine, reason = search_tool._route_query("python programming tutorial")
        assert engine == "lobstr"
        # The query contains "tutorial" which triggers research pattern
        assert "research" in reason.lower() or "general" in reason.lower()
    
    def test_query_routing_local(self, search_tool):
        """Test query routing for local searches"""
        test_cases = [
            "restaurants near me",
            "coffee shops in San Francisco", 
            "pizza shop nearby",
            "hotels near me"
        ]
        
        for query in test_cases:
            engine, reason = search_tool._route_query(query)
            assert engine == "serp"
            assert "local" in reason.lower()
    
    def test_query_routing_product(self, search_tool):
        """Test query routing for product searches"""
        test_cases = [
            "iPhone 15 price",
            "buy laptops online", 
            "running shoes cost",
            "MacBook Pro review"  # Singular "review" triggers product pattern
        ]
        
        for query in test_cases:
            engine, reason = search_tool._route_query(query)
            assert engine == "serp"
            assert "product" in reason.lower()
    
    def test_credibility_scoring(self, search_tool):
        """Test credibility score calculation"""
        # High credibility domains
        assert search_tool._calculate_credibility_score("https://wikipedia.org/article") >= 0.9
        assert search_tool._calculate_credibility_score("https://github.com/user/repo") >= 0.85
        
        # Medium credibility domains
        assert 0.6 <= search_tool._calculate_credibility_score("https://medium.com/article") <= 0.8
        
        # Unknown domains get default score
        score = search_tool._calculate_credibility_score("https://unknown-site.com")
        assert 0.4 <= score <= 0.6
    
    def test_content_type_detection(self, search_tool):
        """Test content type detection from titles"""
        assert search_tool._detect_content_type("Python Tutorial - Learn Programming") == "tutorial"
        assert search_tool._detect_content_type("How to Build a Web App") == "tutorial"
        assert search_tool._detect_content_type("Understanding Machine Learning") == "article"  # Default
        assert search_tool._detect_content_type("Latest News in Tech") == "news"
        assert search_tool._detect_content_type("Random Blog Post") == "article"  # Default
    
    def test_related_queries_generation(self, search_tool):
        """Test related query generation"""
        related = search_tool._generate_related_queries("python programming")
        
        assert len(related) <= 5
        assert all(isinstance(q, str) for q in related)
        # Should contain relevant variations
        assert any("tutorial" in q.lower() for q in related)
    
    @pytest.mark.asyncio
    async def test_execute_invalid_query(self, search_tool):
        """Test execution with invalid query"""
        with pytest.raises(InvalidQueryError):
            await search_tool.execute("")
    
    @pytest.mark.asyncio
    async def test_execute_with_cache_hit(self, search_tool):
        """Test execution when cache returns a result"""
        # Mock cache to return a result
        mock_result = {
            "results": [{"title": "Test", "url": "https://test.com"}],
            "search_metadata": {
                "query_processed": "test query",
                "engine_used": "lobstr",
                "routing_reason": "general search",
                "total_results": 1,
                "search_time_ms": 100.0,
                "cache_hit": False
            }
        }
        
        search_tool.cache.set("test query", mock_result)
        
        result = await search_tool.execute("test query")
        assert result["search_metadata"]["cache_hit"] is True
    
    @pytest.mark.asyncio
    async def test_lobstr_search_success(self, search_tool):
        """Test successful Lobstr API search"""
        mock_response = {
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com",
                    "snippet": "Test snippet content"
                }
            ]
        }
        
        # Mock the entire AsyncClient context manager
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            mock_response_obj = Mock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_client.post.return_value = mock_response_obj
            
            results = await search_tool._search_with_lobstr("test query", 10, "en", True)
            
            assert len(results) == 1
            assert results[0]["title"] == "Test Result"
            assert results[0]["url"] == "https://example.com"
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self, search_tool):
        """Test error handling during search execution"""
        with patch.object(search_tool, '_search_with_lobstr', side_effect=NetworkError("Connection failed")):
            with patch.object(search_tool, '_search_fallback', return_value=[]):
                result = await search_tool.execute("test query")
                
                # Should return error response with metadata containing error
                assert "search_metadata" in result
                assert "error" in result["search_metadata"]
                assert result["search_metadata"]["error"] == "Connection failed"


class TestSearchModels:
    """Test Pydantic models for validation"""
    
    def test_search_result_validation(self):
        """Test SearchResult model validation"""
        # Valid result
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            source="lobstr",
            credibility_score=0.8
        )
        
        assert result.title == "Test Title"
        assert result.credibility_score == 0.8
    
    def test_search_result_invalid_source(self):
        """Test SearchResult with invalid source"""
        with pytest.raises(ValueError):
            SearchResult(
                title="Test",
                url="https://example.com", 
                source="invalid_source",
                credibility_score=0.5
            )
    
    def test_search_metadata_validation(self):
        """Test SearchMetadata model validation"""
        metadata = SearchMetadata(
            query_processed="test query",
            engine_used="lobstr",
            routing_reason="general search",
            total_results=5,
            search_time_ms=150.0
        )
        
        assert metadata.engine_used == "lobstr"
        assert metadata.total_results == 5
    
    def test_web_search_response_validation(self):
        """Test WebSearchResponse model validation"""
        search_result = SearchResult(
            title="Test",
            url="https://example.com",
            source="lobstr"
        )
        
        metadata = SearchMetadata(
            query_processed="test",
            engine_used="lobstr", 
            routing_reason="general",
            search_time_ms=100.0
        )
        
        response = WebSearchResponse(
            results=[search_result],
            search_metadata=metadata,
            summary="Test summary",
            related_queries=["related query 1"]
        )
        
        assert len(response.results) == 1
        assert response.summary == "Test summary"
        assert len(response.related_queries) == 1


@pytest.mark.asyncio
async def test_web_search_function():
    """Test the main web_search function"""
    with patch('tools.web_search.implementation.WebSearchTool') as MockTool:
        mock_instance = AsyncMock()
        mock_instance.execute.return_value = {"results": [], "search_metadata": {}}
        MockTool.return_value = mock_instance
        
        from tools.web_search.implementation import web_search
        
        result = await web_search("test query", max_results=5)
        
        mock_instance.execute.assert_called_once_with("test query", max_results=5)
        assert "results" in result


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"]) 