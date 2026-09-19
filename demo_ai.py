"""Demo harness for the Topic 4 AI module.

Two modes:

  python demo_ai.py             # uses real LLM + sources from env (API keys required)
  python demo_ai.py --offline   # uses fake providers + canned sources (no network)

The demo:
  1. Loads research questions from data/research_questions.json
  2. Fetches sources concurrently from Wikipedia, arXiv, and the web
     (via asyncio.gather) — or returns canned sources in offline mode
  3. Synthesizes a cited answer with the LLM
  4. Prints the answer + numbered references

Demonstrates the parallel fetch pattern students must use in their pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ai import (
    Source, AnswerWithCitations,
    fetch_wikipedia, fetch_arxiv, fetch_web,
    synthesize,
)
from ai.providers.base import LLMProvider, ProviderError
from ai.sources import WebSearchProvider


# --- offline fakes --------------------------------------------------------

class _OfflineLLM(LLMProvider):
    """Constructs a templated answer that cites the provided sources."""

    def complete(self, prompt: str, *, json_schema=None, max_tokens: int = 1024) -> str:
        # The number of sources is implicit in the prompt — count "[N]" markers
        # in the source block.
        import re
        n = len(re.findall(r"^\[(\d+)\]", prompt, re.MULTILINE))
        if n == 0:
            return "I cannot answer from the available sources."
        # Fabricate a plausible answer that cites the first up-to-3 sources.
        cited = ", ".join(f"[{i}]" for i in range(1, min(n, 3) + 1))
        return (
            f"Based on the available sources, here is a synthesized answer "
            f"that draws from multiple references {cited}. The sources broadly "
            f"agree on the main points; differences in emphasis are noted in "
            f"each reference [1]."
        )


class _OfflineSources:
    """Returns canned Source lists for the demo, keyed by query keywords."""

    DB: dict[str, list[Source]] = {
        "photosynthesis": [
            Source(
                title="Photosynthesis",
                url="https://en.wikipedia.org/wiki/Photosynthesis",
                snippet="Photosynthesis is a process used by plants and other "
                        "organisms to convert light energy into chemical energy.",
                origin="wikipedia",
            ),
            Source(
                title="Light-Dependent Reactions of Photosynthesis",
                url="https://arxiv.org/abs/1706.03762",
                snippet="A review of the light-dependent reactions of "
                        "photosynthesis and their role in oxygen evolution.",
                origin="arxiv",
            ),
            Source(
                title="How Plants Make Food",
                url="https://example.com/plants",
                snippet="Plants use chlorophyll to absorb sunlight and produce "
                        "glucose from carbon dioxide and water.",
                origin="web",
            ),
        ],
        "transformer": [
            Source(
                title="Transformer (machine learning model)",
                url="https://en.wikipedia.org/wiki/Transformer_(machine_learning)",
                snippet="A transformer is a deep learning model that adopts the "
                        "mechanism of self-attention.",
                origin="wikipedia",
            ),
            Source(
                title="Attention Is All You Need",
                url="https://arxiv.org/abs/1706.03762",
                snippet="We propose a new simple network architecture, the "
                        "Transformer, based solely on attention mechanisms.",
                origin="arxiv",
            ),
        ],
    }

    @classmethod
    def fetch(cls, query: str) -> list[Source]:
        q = query.lower()
        for keyword, sources in cls.DB.items():
            if keyword in q:
                return sources
        # Generic fallback so unknown queries still return something useful.
        return [
            Source(
                title="Generic reference",
                url="https://example.com/generic",
                snippet=f"A general overview of: {query}",
                origin="web",
            )
        ]


# --- live parallel fetcher ------------------------------------------------

async def fetch_all_sources_live(question: str) -> list[Source]:
    """Run the three fetchers concurrently and combine their results."""
    import httpx  # type: ignore

    async with httpx.AsyncClient(timeout=15.0) as client:
        results = await asyncio.gather(
            fetch_wikipedia(question, max_results=2, client=client),
            fetch_arxiv(question, max_results=2, client=client),
            fetch_web(question, max_results=3, client=client),
            return_exceptions=True,
        )

    combined: list[Source] = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  ! one source failed: {r}", file=sys.stderr)
            continue
        combined.extend(r)
    return combined


# --- offline parallel fetcher --------------------------------------------

async def fetch_all_sources_offline(question: str) -> list[Source]:
    # Even in offline mode, demonstrate the parallel pattern.
    async def _one(q):
        return _OfflineSources.fetch(q)
    results = await asyncio.gather(_one(question), _one(question), _one(question))
    # Dedupe by URL across the three "fetchers".
    seen: set[str] = set()
    out: list[Source] = []
    for batch in results:
        for s in batch:
            if s.url in seen:
                continue
            seen.add(s.url)
            out.append(s)
    return out


# --- main ----------------------------------------------------------------

def render(answer: AnswerWithCitations) -> str:
    out = [f"Q: {answer.question}", "", f"A: {answer.answer}", ""]
    if answer.citations:
        out.append("References:")
        for c in answer.citations:
            out.append(f"  [{c.index}] ({c.source.origin}) {c.source.title}")
            out.append(f"      {c.source.url}")
    return "\n".join(out)


async def run_one(question: str, offline: bool, llm: LLMProvider | None) -> None:
    print(f"=" * 72)
    print(f"Researching: {question}\n")

    if offline:
        sources = await fetch_all_sources_offline(question)
    else:
        sources = await fetch_all_sources_live(question)

    if not sources:
        print("  ! No sources retrieved.")
        return

    print(f"  retrieved {len(sources)} sources "
          f"({sum(1 for s in sources if s.origin == 'wikipedia')} wiki, "
          f"{sum(1 for s in sources if s.origin == 'arxiv')} arxiv, "
          f"{sum(1 for s in sources if s.origin == 'web')} web)\n")

    try:
        answer = synthesize(question, sources, llm=llm)
    except (ProviderError, ValueError) as e:
        print(f"  ! synthesis failed: {e}", file=sys.stderr)
        return
    print(render(answer))
    print()


async def run_demo(offline: bool, limit: int) -> None:
    here = Path(__file__).parent
    qfile = here / "data" / "research_questions.json"
    if not qfile.exists():
        print(f"!! Missing {qfile}", file=sys.stderr)
        sys.exit(2)

    questions = json.loads(qfile.read_text())["questions"][:limit]
    llm = _OfflineLLM() if offline else None

    for q in questions:
        await run_one(q["text"], offline=offline, llm=llm)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline", action="store_true",
                   help="Use fake LLM and canned sources (no API keys, no network).")
    p.add_argument("--limit", type=int, default=2,
                   help="How many questions to run (default: 2).")
    args = p.parse_args()
    asyncio.run(run_demo(offline=args.offline, limit=args.limit))


if __name__ == "__main__":
    main()
