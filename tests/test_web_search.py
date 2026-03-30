"""Tests for Web Search-Grounded Agents.

Given/When/Then Rules:
  - GIVEN a search query → WHEN searching → THEN returns SearchResult with real URLs
  - GIVEN search results → WHEN building LLM context → THEN each has title, url, snippet, domain
  - GIVEN same query twice → WHEN using cache → THEN second call returns cached results
  - GIVEN web_search_enabled=False → WHEN executing worker → THEN no search (LLM-only)
  - GIVEN search results in context → WHEN LLM responds with [Source: URL] → THEN sources parsed
  - GIVEN DuckDuckGo fails → WHEN worker searches → THEN graceful fallback to LLM-only
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock


class TestSearchResult:
    """GIVEN a SearchResult
    WHEN created with valid data
    THEN all fields are accessible"""

    def test_search_result_has_required_fields(self):
        from liquid_swarm.web_search import SearchResult
        result = SearchResult(
            title="AI Market Report 2026",
            url="https://statista.com/ai-market-2026",
            snippet="The global AI market is projected to reach...",
            source="statista.com",
        )
        assert result.title == "AI Market Report 2026"
        assert result.url == "https://statista.com/ai-market-2026"
        assert result.snippet.startswith("The global AI market")
        assert result.source == "statista.com"

    def test_search_result_extracts_domain(self):
        from liquid_swarm.web_search import SearchResult
        result = SearchResult(
            title="Test",
            url="https://www.reuters.com/technology/ai-growth",
            snippet="...",
            source="reuters.com",
        )
        assert result.source == "reuters.com"


class TestBuildSearchContext:
    """GIVEN search results from web
    WHEN building the LLM context string
    THEN it contains all results formatted for the LLM"""

    def test_context_contains_all_results(self):
        from liquid_swarm.web_search import SearchResult, build_search_context
        results = [
            SearchResult(
                title="AI Market 2026",
                url="https://statista.com/ai",
                snippet="Market is $250B",
                source="statista.com",
            ),
            SearchResult(
                title="Tech Trends",
                url="https://reuters.com/tech",
                snippet="AI dominates tech",
                source="reuters.com",
            ),
        ]
        context = build_search_context(results)
        assert "statista.com" in context
        assert "reuters.com" in context
        assert "https://statista.com/ai" in context
        assert "Market is $250B" in context

    def test_empty_results_returns_none_message(self):
        from liquid_swarm.web_search import build_search_context
        context = build_search_context([])
        assert "no search results" in context.lower() or context == ""


class TestSearchCache:
    """GIVEN the same query searched twice
    WHEN using the search cache
    THEN the second call returns cached results without new HTTP request"""

    def test_cache_returns_same_results(self):
        from liquid_swarm.web_search import SearchCache, SearchResult
        cache = SearchCache()
        results = [
            SearchResult(title="Test", url="https://example.com", snippet="...", source="example.com")
        ]
        cache.put("AI market 2026", results)
        cached = cache.get("AI market 2026")
        assert cached is not None
        assert len(cached) == 1
        assert cached[0].url == "https://example.com"

    def test_cache_miss_returns_none(self):
        from liquid_swarm.web_search import SearchCache
        cache = SearchCache()
        assert cache.get("unknown query") is None

    def test_cache_is_case_insensitive(self):
        from liquid_swarm.web_search import SearchCache, SearchResult
        cache = SearchCache()
        results = [
            SearchResult(title="Test", url="https://example.com", snippet="...", source="example.com")
        ]
        cache.put("AI Market", results)
        assert cache.get("ai market") is not None


class TestSourceParsing:
    """GIVEN LLM output with [Source: URL] tags
    WHEN parsing sources
    THEN real URLs are extracted"""

    def test_parse_single_source(self):
        from liquid_swarm.web_search import parse_sources
        text = "The AI market grew 25% [Source: https://statista.com/ai-2026]"
        sources = parse_sources(text)
        assert len(sources) == 1
        assert sources[0] == "https://statista.com/ai-2026"

    def test_parse_multiple_sources(self):
        from liquid_swarm.web_search import parse_sources
        text = (
            "Growth is 25% [Source: https://statista.com/ai]. "
            "Revenue hit $300B [Source: https://reuters.com/tech]."
        )
        sources = parse_sources(text)
        assert len(sources) == 2
        assert "https://statista.com/ai" in sources
        assert "https://reuters.com/tech" in sources

    def test_parse_no_sources(self):
        from liquid_swarm.web_search import parse_sources
        text = "This is a plain response without any sources."
        sources = parse_sources(text)
        assert len(sources) == 0

    def test_parse_deduplicates_sources(self):
        from liquid_swarm.web_search import parse_sources
        text = (
            "Fact A [Source: https://example.com]. "
            "Fact B [Source: https://example.com]."
        )
        sources = parse_sources(text)
        assert len(sources) == 1


class TestSearchEngineFactory:
    """GIVEN different environment configurations
    WHEN creating a search engine
    THEN the correct engine type is returned"""

    def test_default_engine_is_duckduckgo(self):
        from liquid_swarm.web_search import get_search_engine
        engine = get_search_engine()
        assert engine.__class__.__name__ == "DuckDuckGoSearchEngine"

    @patch.dict("os.environ", {"TAVILY_API_KEY": "tvly-test-key"})
    def test_tavily_engine_when_key_set(self):
        from liquid_swarm.web_search import get_search_engine
        engine = get_search_engine()
        assert engine.__class__.__name__ == "TavilySearchEngine"


class TestSearchFallback:
    """GIVEN DuckDuckGo search fails (rate limited, network error)
    WHEN worker tries to search
    THEN it falls back gracefully without crashing"""

    @pytest.mark.asyncio
    async def test_search_failure_returns_empty_list(self):
        from liquid_swarm.web_search import DuckDuckGoSearchEngine
        engine = DuckDuckGoSearchEngine()

        # Mock the internal _search to raise an exception
        with patch.object(engine, '_raw_search', side_effect=Exception("Rate limited")):
            results = await engine.search("test query")
            assert results == []
