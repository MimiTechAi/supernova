"""Web Search Engine — Real-time grounding for Swarm Agents.

Provides web search capabilities so workers can fetch real, current data
instead of relying on LLM training data cutoffs.

Architecture:
    SearchEngine (abstract) → DuckDuckGoSearchEngine (default, free)
                             → TavilySearchEngine (optional, paid)

Each worker calls: search(query) → list[SearchResult]
Then feeds results into the LLM as grounding context.

Copyright 2026 MiMi Tech Ai UG, Bad Liebenzell, Germany.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single web search result with verified metadata."""
    title: str
    url: str
    snippet: str
    source: str  # domain name, e.g. "reuters.com"


# ── Search Cache ─────────────────────────────────────────────────────────────

class SearchCache:
    """In-memory cache for search results within a single run.

    Prevents duplicate searches (same sub-query = same results).
    Cache is case-insensitive.
    """

    def __init__(self):
        self._cache: dict[str, list[SearchResult]] = {}

    def get(self, query: str) -> list[SearchResult] | None:
        key = query.strip().lower()
        return self._cache.get(key)

    def put(self, query: str, results: list[SearchResult]) -> None:
        key = query.strip().lower()
        self._cache[key] = results

    def clear(self) -> None:
        self._cache.clear()


# ── Search Engine Protocol ───────────────────────────────────────────────────

class SearchEngine(Protocol):
    """Protocol for web search engines."""

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        ...


# ── DuckDuckGo Search (Default — Free, No API Key) ──────────────────────────

class DuckDuckGoSearchEngine:
    """Free web search via DuckDuckGo — no API key needed.

    Uses the duckduckgo-search library for web + news results.
    Rate-limited to avoid throttling (max 3 concurrent searches).
    """

    def __init__(self, max_concurrent: int = 3):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search DuckDuckGo for web + news results.

        Returns up to max_results SearchResults with real URLs.
        Falls back gracefully to empty list on any error.
        """
        try:
            async with self._semaphore:
                results = await asyncio.to_thread(self._raw_search, query, max_results)
                return results
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed for '{query}': {e}")
            return []

    def _raw_search(self, query: str, max_results: int) -> list[SearchResult]:
        """Synchronous DuckDuckGo search (run in thread pool)."""
        from duckduckgo_search import DDGS

        results: list[SearchResult] = []
        seen_urls: set[str] = set()

        with DDGS() as ddgs:
            # Web results
            web_count = max(1, max_results - 2)
            for r in ddgs.text(query, max_results=web_count):
                url = r.get("href", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=url,
                        snippet=r.get("body", ""),
                        source=_extract_domain(url),
                    ))

            # News results (fresher data)
            news_count = max(1, max_results - len(results))
            try:
                for r in ddgs.news(query, max_results=news_count):
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append(SearchResult(
                            title=r.get("title", ""),
                            url=url,
                            snippet=r.get("body", ""),
                            source=_extract_domain(url),
                        ))
            except Exception:
                pass  # News search is optional enhancement

        return results[:max_results]


# ── Tavily Search (Optional — Paid, More Reliable) ───────────────────────────

class TavilySearchEngine:
    """Tavily AI search — designed for AI agents.

    Requires TAVILY_API_KEY. More reliable for production use.
    See: https://tavily.com
    """

    def __init__(self, api_key: str | None = None, max_concurrent: int = 5):
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search via Tavily API."""
        try:
            import httpx

            async with self._semaphore:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(
                        "https://api.tavily.com/search",
                        json={
                            "api_key": self._api_key,
                            "query": query,
                            "max_results": max_results,
                            "search_depth": "basic",
                            "include_answer": False,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()

            results = []
            for r in data.get("results", [])[:max_results]:
                url = r.get("url", "")
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=url,
                    snippet=r.get("content", ""),
                    source=_extract_domain(url),
                ))
            return results

        except Exception as e:
            logger.warning(f"Tavily search failed for '{query}': {e}")
            return []


# ── Factory ──────────────────────────────────────────────────────────────────

def get_search_engine() -> SearchEngine:
    """Return the best available search engine based on environment.

    - If TAVILY_API_KEY is set → TavilySearchEngine (more reliable)
    - Otherwise → DuckDuckGoSearchEngine (free, no key needed)
    """
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key:
        return TavilySearchEngine(api_key=tavily_key)

    return DuckDuckGoSearchEngine()


# ── Context Builder ──────────────────────────────────────────────────────────

def build_search_context(results: list[SearchResult]) -> str:
    """Format search results into LLM-readable context.

    Each result is formatted as:
        [1] Title — domain.com
        URL: https://...
        Snippet: ...
    """
    if not results:
        return "No search results were found for this query."

    lines = ["WEB SEARCH RESULTS:"]
    for i, r in enumerate(results, 1):
        lines.append(f"\n[{i}] {r.title} — {r.source}")
        lines.append(f"    URL: {r.url}")
        lines.append(f"    Snippet: {r.snippet}")

    lines.append("\n---")
    lines.append(
        "INSTRUCTIONS: Only use information from the search results above. "
        "Cite each fact with [Source: URL]. Do NOT invent sources."
    )

    return "\n".join(lines)


# ── Source Parser ────────────────────────────────────────────────────────────

def parse_sources(text: str) -> list[str]:
    """Extract [Source: URL] citations from LLM output.

    Returns deduplicated list of URLs.
    """
    pattern = r'\[Source:\s*(https?://[^\]\s]+)\s*\]'
    matches = re.findall(pattern, text, re.IGNORECASE)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for url in matches:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    """Extract clean domain from URL.

    'https://www.reuters.com/tech/ai' → 'reuters.com'
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        # Remove 'www.' prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return url
