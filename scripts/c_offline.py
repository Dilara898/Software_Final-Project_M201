"""Explicit offline doubles for C's demonstrations; never selected in live mode."""

import asyncio

import httpx

from ai.providers.base import ProviderError
from ai.schemas import AnswerWithCitations, Citation, Source
from researcher.concurrency.models import SessionRecord


def offline_client() -> httpx.AsyncClient:
    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Offline demonstration attempted HTTP")

    return httpx.AsyncClient(transport=httpx.MockTransport(forbidden), trust_env=False)


class OfflineService:
    """Synthetic latencies and cited text, clearly labelled as simulation."""

    def __init__(self, failed: tuple[str, ...] = ()) -> None:
        self.failed = failed
        self.synthesis_calls = 0

    async def fetch(
        self, source: str, query: str, client: httpx.AsyncClient
    ) -> list[Source]:
        await asyncio.sleep({"wikipedia": 0.04, "arxiv": 0.08, "web": 0.04}[source])
        if source in self.failed:
            raise ProviderError("Simulated source outage")
        return [
            Source(
                title=f"Simulated {source}",
                url=f"https://example.invalid/{source}",
                snippet=f"Offline fixture for {query}",
                origin=source,
            )
        ]

    async def synthesize(
        self, question: str, sources: list[Source]
    ) -> AnswerWithCitations:
        self.synthesis_calls += 1
        return AnswerWithCitations(
            question=question,
            answer="Offline simulation using "
            + ", ".join(f"[{i}]" for i in range(1, len(sources) + 1)),
            citations=[
                Citation(index=i, source=source) for i, source in enumerate(sources, 1)
            ],
        )


class OfflineHistory:
    def __init__(self) -> None:
        self.records: list[SessionRecord] = []

    async def save(self, record: SessionRecord) -> int:
        self.records.append(record)
        return len(self.records)
