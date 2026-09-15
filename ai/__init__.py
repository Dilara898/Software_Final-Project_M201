"""
AI module for Topic 4 — Async Research Assistant.

Public surface
--------------
async fetch_wikipedia(query, *, max_results=3, client=None) -> list[Source]
async fetch_arxiv(query, *, max_results=3, client=None)     -> list[Source]
async fetch_web(query, *, max_results=3, provider=None, client=None) -> list[Source]
    Async source fetchers. Pass an `httpx.AsyncClient` via `client` to share
    connections across calls.

class WebSearchProvider (abstract)
class TavilyProvider     (concrete)
class SerperProvider     (concrete)
class DuckDuckGoProvider (concrete)
get_web_search_provider() -> WebSearchProvider
    Pluggable web-search backend. Selected by WEB_SEARCH_PROVIDER env var.

synthesize(question, sources, *, llm=None) -> AnswerWithCitations
    LLM synthesizes a cited answer from a list of sources.

Schemas: Source, Citation, AnswerWithCitations.
"""

from ai.schemas import Source, Citation, AnswerWithCitations
from ai.sources import (
    fetch_wikipedia, fetch_arxiv, fetch_web,
    WebSearchProvider, TavilyProvider, SerperProvider, DuckDuckGoProvider,
    get_web_search_provider,
)
from ai.synthesizer import synthesize

__all__ = [
    "Source", "Citation", "AnswerWithCitations",
    "fetch_wikipedia", "fetch_arxiv", "fetch_web",
    "WebSearchProvider", "TavilyProvider", "SerperProvider", "DuckDuckGoProvider",
    "get_web_search_provider",
    "synthesize",
]
