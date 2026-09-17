"""Full research flow using fake ports, without A/B/D production files."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from ai.providers.base import ProviderError
from ai.schemas import AnswerWithCitations, Citation, Source
from researcher.concurrency.contracts import (
    HistoryCallbackAdapter,
    StorageUnavailableError,
)
from researcher.concurrency.orchestrator import SourceOrchestrator
from researcher.concurrency.research import (
    InvalidAnswerError,
    NoSourcesError,
    ResearchOrchestrator,
    ResearchTimeoutError,
    SynthesisError,
    research,
    validate_answer,
)


def sample(name):
    return Source(
        title=name,
        url=f"https://example.invalid/{name}",
        snippet="evidence",
        origin=name,
    )


@pytest_asyncio.fixture
async def ports():
    def forbidden(request):
        raise AssertionError("network forbidden")

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
        service, history = AsyncMock(), AsyncMock()

        async def fetch(name, question, client):
            return [sample(name)]

        async def synthesize(question, sources):
            return AnswerWithCitations(
                question=question,
                answer="Supported [1].",
                citations=[Citation(index=1, source=sources[0])],
            )

        service.fetch.side_effect = fetch
        service.synthesize.side_effect = synthesize
        history.save.return_value = 10
        collector = SourceOrchestrator(
            service, client=client, timeout_seconds=1, max_concurrency=3
        )
        yield collector, service, history


@pytest.mark.asyncio
async def test_complete_research_has_typed_result_and_one_history_write(ports):
    collector, service, history = ports
    result = await research(
        "question", orchestrator=ResearchOrchestrator(collector, service, history)
    )
    assert result.answer.answer == "Supported [1]."
    assert result.history_status == "saved" and result.session_id == 10
    assert result.bundle.used == ["wikipedia", "arxiv", "web"]
    assert result.cache_stats.bypasses == 3
    record = history.save.await_args.args[0]
    assert record.request_id == result.request_id == result.bundle.request_id
    assert record.duration_ms == int(result.timings.answer_ready_ms)
    assert (
        result.timings.total_ms
        >= result.timings.answer_ready_ms
        >= result.timings.collection_ms
    )
    service.synthesize.assert_awaited_once()
    history.save.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [1, 2, 3])
async def test_partial_and_all_source_failures(ports, failed):
    collector, service, history = ports

    async def fetch(name, question, client):
        if name in ["wikipedia", "arxiv", "web"][:failed]:
            raise ProviderError("offline")
        return [sample(name)]

    service.fetch.side_effect = fetch
    worker = ResearchOrchestrator(collector, service, history)
    if failed == 3:
        with pytest.raises(NoSourcesError) as caught:
            await worker.research("q")
        assert len(caught.value.collection.failed) == 3
        service.synthesize.assert_not_called()
        history.save.assert_not_called()
    else:
        result = await worker.research("q")
        assert len(result.bundle.used) == 3 - failed
        assert len(result.warnings) == failed


@pytest.mark.asyncio
async def test_all_empty_has_different_diagnostics(ports):
    collector, service, history = ports
    service.fetch.side_effect = None
    service.fetch.return_value = []
    with pytest.raises(NoSourcesError) as caught:
        await ResearchOrchestrator(collector, service, history).research("q")
    assert len(caught.value.collection.empty) == 3
    assert caught.value.collection.failed == []
    service.synthesize.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [StorageUnavailableError("private"), ConnectionError(), TimeoutError()]
)
async def test_history_failure_preserves_answer_without_retry(ports, error):
    collector, service, history = ports
    history.save.side_effect = error
    result = await ResearchOrchestrator(collector, service, history).research("q")
    assert result.answer.answer and result.session_id is None
    assert result.history_status == (
        "unknown" if isinstance(error, TimeoutError) else "failed"
    )
    assert result.warnings[-1].stage == "history"
    assert "private" not in result.model_dump_json()
    history.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_history_timeout_is_bounded(ports):
    collector, service, history = ports

    async def slow(record):
        await asyncio.sleep(30)

    history.save.side_effect = slow
    result = await asyncio.wait_for(
        ResearchOrchestrator(
            collector, service, history, history_timeout_seconds=0.01
        ).research("q"),
        1,
    )
    assert result.history_status == "unknown"


@pytest.mark.asyncio
async def test_synthesis_deadline_does_not_persist(ports):
    collector, service, history = ports

    async def slow(question, sources):
        await asyncio.sleep(30)

    service.synthesize.side_effect = slow
    with pytest.raises(SynthesisError, match="deadline"):
        await ResearchOrchestrator(
            collector, service, history, synthesis_timeout_seconds=0.01
        ).research("q")
    history.save.assert_not_called()


@pytest.mark.asyncio
async def test_synthesis_error_is_typed(ports):
    collector, service, history = ports
    service.synthesize.side_effect = ProviderError("private")
    with pytest.raises(SynthesisError) as caught:
        await ResearchOrchestrator(collector, service, history).research("q")
    assert "private" not in str(caught.value)
    history.save.assert_not_called()


@pytest.mark.asyncio
async def test_overall_deadline_cleans_source_tasks(ports):
    collector, service, history = ports
    stopped = []

    async def slow(name, question, client):
        try:
            await asyncio.sleep(30)
        finally:
            stopped.append(name)

    service.fetch.side_effect = slow
    with pytest.raises(ResearchTimeoutError):
        await ResearchOrchestrator(
            collector, service, history, research_timeout_seconds=0.01
        ).research("q")
    assert len(stopped) == 3
    service.synthesize.assert_not_called()
    history.save.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_requests_are_isolated(ports):
    collector, service, history = ports
    worker = ResearchOrchestrator(collector, service, history)
    one, two = await asyncio.gather(worker.research("one"), worker.research("two"))
    assert one.request_id != two.request_id
    assert one.answer.question == "one" and two.answer.question == "two"
    assert {c.args[0].question for c in history.save.await_args_list} == {"one", "two"}


@pytest.mark.asyncio
async def test_caller_cancel_during_synthesis_does_not_save(ports):
    collector, service, history = ports
    entered = asyncio.Event()

    async def slow(question, sources):
        entered.set()
        await asyncio.sleep(30)

    service.synthesize.side_effect = slow
    task = asyncio.create_task(
        ResearchOrchestrator(collector, service, history).research("q")
    )
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    history.save.assert_not_called()


@pytest.mark.parametrize(
    "text,indices",
    [
        ("[99]", []),
        ("[0]", [0]),
        ("[1]", [1, 1]),
        ("uncited", []),
        ("", []),
        ("[1]", [2]),
    ],
)
def test_invalid_citations_are_rejected(text, indices):
    sources = [sample("web")]
    answer = AnswerWithCitations(
        question="q",
        answer=text,
        citations=[Citation(index=i, source=sources[0]) for i in indices],
    )
    with pytest.raises(InvalidAnswerError):
        validate_answer(answer, "q", sources)


def test_grouped_citations_and_wrong_source_mapping():
    sources = [sample("wikipedia"), sample("arxiv"), sample("web")]
    answer = AnswerWithCitations(
        question="q",
        answer="[1, 3] and [3]",
        citations=[Citation(index=i, source=sources[i - 1]) for i in [1, 3]],
    )
    validate_answer(answer, "q", sources)
    answer.citations = [
        Citation(index=1, source=sources[2]),
        Citation(index=3, source=sources[2]),
    ]
    with pytest.raises(InvalidAnswerError, match="wrong source"):
        validate_answer(answer, "q", sources)


def test_explicit_uncited_answer_policy_and_question_mismatch():
    answer = AnswerWithCitations(question="q", answer="Insufficient evidence.")
    validate_answer(answer, "q", [sample("web")], require_citations=False)
    with pytest.raises(InvalidAnswerError):
        validate_answer(answer, "different", [sample("web")])
    with pytest.raises(TypeError):
        validate_answer({}, "q", [sample("web")])


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_id", [None, True, 0])
async def test_broken_history_contract_propagates(ports, wrong_id):
    collector, service, history = ports
    history.save.return_value = wrong_id
    with pytest.raises(TypeError):
        await ResearchOrchestrator(collector, service, history).research("q")


@pytest.mark.asyncio
async def test_callback_history_adapter(ports):
    collector, service, history = ports
    adapter = HistoryCallbackAdapter(history.save)
    assert (
        await ResearchOrchestrator(collector, service, adapter).research("q")
    ).session_id == 10
