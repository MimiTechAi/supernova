import asyncio
from langchain_core.tools import tool
from liquid_swarm.web_search import get_search_engine

@tool
async def web_search_tool(query: str) -> str:
    """Useful to search the internet for current events, real-time facts, and market data.
    Input should be a specific search query.
    Returns excerpts from the top web results.
    """
    engine = get_search_engine()
    results = await engine.search(query, max_results=3)
    
    if not results:
        return "No search results found."
        
    lines = ["WEB SEARCH RESULTS:"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title} ({r.source})")
        lines.append(f"URL: {r.url}")
        lines.append(f"Snippet: {r.snippet}\n")
    return "\n".join(lines)
