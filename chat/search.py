from duckduckgo_search import DDGS
from tavily import TavilyClient
import os

def search_live_web(query: str, max_results: int = 3, provider: str = "duckduckgo") -> str:
    """Fetches real-time web context for queries not covered by local corpus."""
    context_snippets = []
    
    if provider == "duckduckgo":
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            for res in results:
                context_snippets.append(f"Source: {res['title']}\nSnippet: {res['body']}")
                
    elif provider == "tavily":
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(query=query, max_results=max_results)
        for res in response.get("results", []):
            context_snippets.append(f"Source: {res['title']}\nContent: {res['content']}")

    return "\n\n".join(context_snippets) if context_snippets else "No live web data found."