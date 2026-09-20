"""Research workflow over injected ports; no configuration or database imports."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Sequence
from typing import Literal
from uuid import uuid4

import httpx

from ai.providers.base import ProviderError
from ai.schemas import AnswerWithCitations, Source
from researcher.concurrency.contracts import (
    HistoryStore,
    StorageUnavailableError,
    SynthesisService,
    UpstreamDataError,
)
from researcher.concurrency.models import (
    SOURCE_NAMES,
    CollectionResult,
    OperationWarning,
    ResearchResult,
    ResearchTimings,
    SessionRecord,
)
from researcher.concurrency.orchestrator import SourceOrchestrator, positive_seconds

log = logging.getLogger(__name__)
_CITATIONS = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class OrchestrationError(Exception):
    """C's public error boundary; D can map it to the shared CLI error family."""


class NoSourcesError(OrchestrationError):
    def __init__(self, collection: CollectionResult) -> None:
        super().__init__(
            "No usable sources were returned; synthesis was not attempted."
        )
        self.collection = collection


class SynthesisError(OrchestrationError):
    """The synthesis operation failed or exceeded its caller deadline."""


class InvalidAnswerError(OrchestrationError):
    """The answer does not satisfy the citation contract."""


class ResearchTimeoutError(OrchestrationError):
    """The overall request budget was exceeded."""


def validate_answer(
    answer: AnswerWithCitations,
    question: str,
    sources: list[Source],
    *,
    require_citations: bool = True,
) -> None:
    """Check mapping, not factual truth; do not rewrite the model's text."""
    if not isinstance(answer, AnswerWithCitations):
        raise TypeError("SynthesisService must return AnswerWithCitations")
    if not answer.answer.strip() or answer.question.strip() != question.strip():
        raise InvalidAnswerError(
            "The answer is empty or belongs to a different question."
        )
    try:
        referenced = {
            int(value.strip())
            for match in _CITATIONS.finditer(answer.answer)
            for value in match.group(1).split(",")
        }
    except ValueError as exc:
        raise InvalidAnswerError("A numeric source reference is invalid.") from exc
    indices = [citation.index for citation in answer.citations]
    if require_citations and not referenced:
        raise InvalidAnswerError("The answer has no numeric source references.")
    if len(indices) != len(set(indices)) or set(indices) != referenced:
        raise InvalidAnswerError("Answer markers and reference list do not agree.")
    for citation in answer.citations:
        if not 1 <= citation.index <= len(sources):
            raise InvalidAnswerError("A citation points outside the source list.")
        if citation.source != sources[citation.index - 1]:
            raise InvalidAnswerError("A citation points to the wrong source.")


class ResearchOrchestrator:
    """Collect, synthesize and persist, using one event loop and external owners.

    SynthesisService must be async. If B wraps a synchronous SDK in a thread,
    these caller deadlines cannot forcibly terminate that underlying thread.
    History timeouts are 'unknown' because the insert may already have committed.
    """

    def __init__(
        self,
        collector: SourceOrchestrator,
        synthesis: SynthesisService,
        history: HistoryStore,
        *,
        synthesis_timeout_seconds: float = 20,
        history_timeout_seconds: float = 2,
        research_timeout_seconds: float = 45,
        require_citations: bool = True,
    ) -> None:
        self._collector, self._synthesis, self._history = collector, synthesis, history
        self._synthesis_timeout = positive_seconds(
            synthesis_timeout_seconds, "synthesis_timeout_seconds"
        )
        self._history_timeout = positive_seconds(
            history_timeout_seconds, "history_timeout_seconds"
        )
        self._research_timeout = positive_seconds(
            research_timeout_seconds, "research_timeout_seconds"
        )
        self._require_citations = require_citations

    async def research(
        self,
        question: str,
        wanted: Sequence[str] = SOURCE_NAMES,
        *,
        use_cache: bool = True,
    ) -> ResearchResult:
        """Produce a cited answer; persistence outages become visible warnings."""
        request_id = uuid4().hex
        log.info("research_started", extra={"request_id": request_id})
        try:
            async with asyncio.timeout(self._research_timeout):
                return await self._run(question, wanted, use_cache, request_id)
        except TimeoutError as exc:
            raise ResearchTimeoutError(
                "The overall research deadline was exceeded."
            ) from exc

    async def _run(
        self,
        question: str,
        wanted: Sequence[str],
        use_cache: bool,
        request_id: str,
    ) -> ResearchResult:
        started = time.perf_counter()
        bundle = await self._collector.collect(
            question, wanted, use_cache=use_cache, request_id=request_id
        )
        sources = bundle.sources
        if not sources:
            raise NoSourcesError(bundle)
        warnings = list(bundle.warnings)
        for outcome in bundle.outcomes:
            if outcome.status not in ("ok", "empty"):
                warnings.append(
                    OperationWarning(
                        code=outcome.error_code or "source_empty",
                        stage="collection",
                        source=outcome.name,
                        message=f"{outcome.name}: {outcome.status}",
                    )
                )
        synthesis_started = time.perf_counter()
        try:
            async with asyncio.timeout(self._synthesis_timeout):
                answer = await self._synthesis.synthesize(question, list(sources))
        except TimeoutError as exc:
            raise SynthesisError("The synthesis caller deadline was exceeded.") from exc
        except (ProviderError, httpx.HTTPError, UpstreamDataError) as exc:
            raise SynthesisError(
                "The synthesis provider could not return an answer."
            ) from exc
        synthesis_ms = (time.perf_counter() - synthesis_started) * 1000
        validate_answer(
            answer, question, sources, require_citations=self._require_citations
        )
        answer_ready_ms = (time.perf_counter() - started) * 1000
        record = SessionRecord(
            request_id=request_id,
            question=question,
            answer=answer.answer,
            sources_used=bundle.used,
            sources_failed=bundle.failed,
            duration_ms=int(answer_ready_ms),
        )
        history_started = time.perf_counter()
        history_status: Literal["saved", "failed", "unknown"] = "saved"
        session_id: int | None = None
        try:
            async with asyncio.timeout(self._history_timeout):
                session_id = await self._history.save(record)
            if (
                isinstance(session_id, bool)
                or not isinstance(session_id, int)
                or session_id < 1
            ):
                raise TypeError(
                    "HistoryStore.save must return a positive integer session id"
                )
        except (TimeoutError, StorageUnavailableError, ConnectionError) as exc:
            history_status = "unknown" if isinstance(exc, TimeoutError) else "failed"
            code = (
                "history_write_unknown"
                if history_status == "unknown"
                else "history_write_failed"
            )
            warnings.append(
                OperationWarning(
                    code=code,
                    stage="history",
                    message="The answer could not be confirmed in history.",
                )
            )
            log.warning(code, extra={"request_id": request_id, "stage": "history"})
        history_ms = (time.perf_counter() - history_started) * 1000
        timings = ResearchTimings(
            collection_ms=bundle.duration_ms,
            synthesis_ms=synthesis_ms,
            answer_ready_ms=answer_ready_ms,
            history_ms=history_ms,
            total_ms=(time.perf_counter() - started) * 1000,
        )
        log.info(
            "research_completed",
            extra={
                "request_id": request_id,
                "duration_ms": timings.total_ms,
                "history_status": history_status,
            },
        )
        return ResearchResult(
            request_id=request_id,
            answer=answer,
            bundle=bundle,
            timings=timings,
            warnings=warnings,
            history_status=history_status,
            session_id=session_id,
        )


async def research(
    question: str,
    wanted: Sequence[str] = SOURCE_NAMES,
    *,
    orchestrator: ResearchOrchestrator,
    use_cache: bool = True,
) -> ResearchResult:
    """Small functional entry point for D; dependencies are explicitly composed."""
    return await orchestrator.research(question, wanted, use_cache=use_cache)
