# Web Search Tool

## Purpose
Intelligent web search tool that routes queries to optimal search engines based on query type and requirements. Implements dual-strategy approach for comprehensive search capabilities.

## Search Engines

### Primary Engines
- **Lobstr.io** 🔍 - Default search agent for general research and wide queries
  - Fast response times
  - High free tier quota
  - Great baseline for most searches
- **SERP API** 🎯 - Precision tasks requiring structured data
  - Rich structured results
  - LangChain native integration
  - Local business info, product pricing, maps, FAQs

### Routing Strategy
The tool automatically selects the optimal search engine based on query characteristics:

```python
# Precision tasks → SERP API
if query_type in ["local", "product", "map", "snippet", "FAQ", "shopping"]:
    use_serp_api()
    
# General research → Lobstr.io  
else:
    use_lobstr_io()
```

## Tool Capabilities

### Core Functions
- `web_search()` - Main search function with intelligent routing
- `search_with_lobstr()` - Direct Lobstr.io search for general queries
- `search_with_serp()` - Direct SERP API search for structured data
- `analyze_query_type()` - Query classification for optimal routing

### Advanced Features
- Content summarization for long results
- Duplicate result filtering
- Source credibility scoring
- Automatic follow-up scraping when needed
- Result caching and memory storage

## MCP Tool Schema

### Input Parameters
- `query` (required): Search query string
- `max_results` (optional): Maximum number of results (default: 10)
- `search_type` (optional): Force specific search type ("general", "local", "product", etc.)
- `include_snippets` (optional): Include content snippets (default: true)
- `location` (optional): Geographic location for local searches

### Output Format
```json
{
  "results": [
    {
      "title": "Result title",
      "url": "https://example.com",
      "snippet": "Content preview...",
      "source": "lobstr|serp",
      "credibility_score": 0.85,
      "metadata": {
        "domain": "example.com",
        "date_published": "2024-01-15",
        "content_type": "article"
      }
    }
  ],
  "search_metadata": {
    "query_processed": "processed query",
    "engine_used": "lobstr",
    "routing_reason": "general_query",
    "total_results": 150,
    "search_time_ms": 342
  },
  "summary": "AI-generated summary of key findings..."
}
```

## Usage Examples

### General Research
```python
result = await web_search_tool.execute({
    "query": "artificial intelligence trends 2024",
    "max_results": 5
})
# Routes to Lobstr.io for broad research
```

### Local Business Search
```python
result = await web_search_tool.execute({
    "query": "best pizza restaurants near me",
    "location": "San Francisco, CA",
    "search_type": "local"
})
# Routes to SERP API for structured local results
```

### Product Research
```python
result = await web_search_tool.execute({
    "query": "iPhone 15 Pro price comparison",
    "search_type": "product"
})
# Routes to SERP API for structured pricing data
```

### Technical Documentation
```python
result = await web_search_tool.execute({
    "query": "Python asyncio best practices",
    "max_results": 8
})
# Routes to Lobstr.io for comprehensive technical content
```

## Integration with Agents

### Agent Discovery
Tools are automatically discovered through the MCP registry:
```python
# Tool registration
@register_tool("web_search")
async def web_search(query: str, **kwargs) -> Dict[str, Any]:
    return await WebSearchTool().execute(query, **kwargs)
```

### Usage in Agent Workflows
```python
# In agent processing
search_results = await self.tools.web_search(
    query=user_query,
    max_results=5
)

# Process and respond
summary = await self.summarize_search_results(search_results)
return f"Based on my search: {summary}"
```

## Configuration

### Environment Variables
```bash
# Required for Lobstr.io
LOBSTR_API_KEY=your_lobstr_key

# Required for SERP API  
SERP_API_KEY=your_serp_key

# Optional settings
WEB_SEARCH_DEFAULT_RESULTS=10
WEB_SEARCH_TIMEOUT_SECONDS=30
WEB_SEARCH_CACHE_TTL=3600
```

### Search Engine Priorities
Default routing can be customized:
```python
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
```

## Error Handling

### Fallback Strategy
- Primary engine failure → Automatic fallback to secondary
- Rate limiting → Queue management with exponential backoff
- Network timeouts → Cached results when available
- Invalid queries → Query refinement suggestions

### Error Types
- `SearchEngineError` - Engine-specific failures
- `RateLimitError` - API quota exceeded  
- `InvalidQueryError` - Malformed search query
- `NetworkError` - Connection issues

## Performance

### Caching Strategy
- Query result caching (1 hour TTL)
- Popular query pre-fetching
- Intelligent cache invalidation
- Memory-efficient storage

### Metrics Tracking
- Search latency by engine
- Success/failure rates
- Query type distribution
- Cache hit ratios

## Security

### Input Validation
- Query sanitization against injection
- Rate limiting per user/agent
- Malicious URL filtering
- Content safety scoring

### Data Privacy
- No query logging in production
- Encrypted API communications
- Minimal metadata collection
- GDPR compliance ready

## Testing

### Test Coverage
- Unit tests for each search engine
- Integration tests with real APIs
- Mock testing for error conditions
- Performance benchmarking
- Query routing accuracy tests

### Test Commands
```bash
# Run all tests
python -m pytest tools/web_search/tests/

# Test specific engine
python -m pytest tools/web_search/tests/test_lobstr.py

# Performance tests
python -m pytest tools/web_search/tests/test_performance.py
``` 