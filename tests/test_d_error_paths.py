"""Covers cli.py's show_history/main() error branches and application.py's
render_answer branches and run_ask cleanup paths that fakes-only tests miss.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncpg
import pytest
from pydantic import BaseModel, ValidationError as PydanticValidationError

from ai.providers.base import ProviderError
from ai.schemas import AnswerWithCitations, Citation, Source
from researcher.application import render_answer, run_ask
from researcher.cli import main, show_history
from researcher.concurrency.models import (
    CollectionResult,
    OperationWarning,
    ResearchResult,
    ResearchTimings,
    SourceOutcome,
)
from researcher.concurrency.research import OrchestrationError
from researcher.exceptions import ResearcherError


class _Tiny(BaseModel):
    x: int


def _settings_error() -> PydanticValidationError:
    try:
        _Tiny(x="not-an-int")
    except PydanticValidationError as exc:
        return exc
    raise AssertionError("expected a validation error")


def _make_source(origin: str) -> Source:
    return Source(
        title=f"{origin} title",
        url=f"https://example.org/{origin}",
        snippet="excerpt",
        origin=origin,
    )


def _make_result(*, failed=(), empty=(), warnings=(), history_status="saved",
                  session_id=1) -> ResearchResult:
    source = _make_source("web")
    outcomes = [SourceOutcome(name="web", sources=[source], status="ok")]
    for name in failed:
        outcomes.append(SourceOutcome(name=name, status="error", error_code="x"))
    for name in empty:
        outcomes.append(SourceOutcome(name=name, status="empty"))
    bundle = CollectionResult(request_id="r1", outcomes=outcomes, duration_ms=10)
    answer = AnswerWithCitations(
        question="Q?", answer="A [1].",
        citations=[Citation(index=1, source=source)],
    )
    return ResearchResult(
        request_id="r1",
        answer=answer,
        bundle=bundle,
        timings=ResearchTimings(
            collection_ms=1, synthesis_ms=1, answer_ready_ms=1,
            history_ms=1, total_ms=1,
        ),
        warnings=list(warnings),
        history_status=history_status,
        session_id=session_id,
    )


# ---- application.render_answer branches (lines 71, 76, 81, 88) ----

def test_render_notes_failed_sources():
    out = render_answer(_make_result(failed=["wikipedia"]))
    assert "Unavailable sources: wikipedia" in out


def test_render_notes_empty_sources():
    out = render_answer(_make_result(empty=["arxiv"]))
    assert "Sources with no results: arxiv" in out


def test_render_includes_warnings():
    warning = OperationWarning(code="w1", stage="collection", message="slow")
    out = render_answer(_make_result(warnings=[warning]))
    assert "Warning: w1: slow" in out


def test_render_flags_unknown_history_as_unconfirmed():
    out = render_answer(_make_result(history_status="unknown", session_id=None))
    assert "History: unknown" in out
    assert "unconfirmed" in out
    assert "Session:" not in out


# ---- cli.show_history (lines 57-108) ----

@pytest.mark.asyncio
async def test_show_history_prints_sessions(monkeypatch, capsys):
    session = SimpleNamespace(id=1, created_at="2026-09-19", question="Q?", answer="A.")
    monkeypatch.setattr(
        "researcher.storage.db.get_pool", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(
        "researcher.storage.history.list_sessions",
        AsyncMock(return_value=[session]),
    )
    monkeypatch.setattr("researcher.storage.db.close_pool", AsyncMock())

    assert await show_history(10) == 0
    out = capsys.readouterr().out
    assert "#1" in out and "Q?" in out


@pytest.mark.asyncio
async def test_show_history_empty(monkeypatch, capsys):
    monkeypatch.setattr(
        "researcher.storage.db.get_pool", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(
        "researcher.storage.history.list_sessions", AsyncMock(return_value=[])
    )
    monkeypatch.setattr("researcher.storage.db.close_pool", AsyncMock())

    assert await show_history(10) == 0
    assert "No previous queries found." in capsys.readouterr().out


@pytest.mark.asyncio
async def test_show_history_settings_error(monkeypatch, capsys):
    monkeypatch.setattr(
        "researcher.storage.db.get_pool",
        AsyncMock(side_effect=_settings_error()),
    )
    monkeypatch.setattr("researcher.storage.db.close_pool", AsyncMock())

    assert await show_history(10) == 1
    assert "Configuration error" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_show_history_db_error(monkeypatch, capsys):
    monkeypatch.setattr(
        "researcher.storage.db.get_pool",
        AsyncMock(side_effect=asyncpg.PostgresError("down")),
    )
    monkeypatch.setattr("researcher.storage.db.close_pool", AsyncMock())

    assert await show_history(10) == 1
    assert "Cannot read history" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_show_history_cleanup_failure_does_not_hide_result(monkeypatch, capsys):
    monkeypatch.setattr(
        "researcher.storage.db.get_pool", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(
        "researcher.storage.history.list_sessions", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        "researcher.storage.db.close_pool",
        AsyncMock(side_effect=RuntimeError("cleanup boom")),
    )

    assert await show_history(10) == 0
    assert "No previous queries found." in capsys.readouterr().out


# ---- cli.main() dispatch and outer error handling (lines 136-196) ----

def test_main_history_command_success(monkeypatch):
    async def fake_show_history(limit):
        assert limit == 5
        return 0

    monkeypatch.setattr("researcher.cli.show_history", fake_show_history)
    assert main(["history", "--limit", "5"]) == 0


def test_main_maps_researcher_error_to_exit_1(monkeypatch, capsys):
    async def boom(args):
        raise ResearcherError("no usable sources")

    monkeypatch.setattr("researcher.application.run_ask", boom)
    assert main(["ask", "A valid question"]) == 1
    assert "Research failed" in capsys.readouterr().err


def test_main_maps_orchestration_error_to_exit_1(monkeypatch, capsys):
    async def boom(args):
        raise OrchestrationError("collection failed")

    monkeypatch.setattr("researcher.application.run_ask", boom)
    assert main(["ask", "A valid question"]) == 1
    assert "Research failed" in capsys.readouterr().err


def test_main_maps_provider_error_to_exit_1(monkeypatch, capsys):
    async def boom(args):
        raise ProviderError("provider down")

    monkeypatch.setattr("researcher.application.run_ask", boom)
    assert main(["ask", "A valid question"]) == 1


def test_main_maps_settings_error_to_exit_1(monkeypatch, capsys):
    async def boom(args):
        raise _settings_error()

    monkeypatch.setattr("researcher.application.run_ask", boom)
    assert main(["ask", "A valid question"]) == 1
    assert "Invalid configuration" in capsys.readouterr().err


def test_main_maps_missing_dependency_to_exit_1(monkeypatch, capsys):
    async def boom(args):
        raise ModuleNotFoundError(name="asyncpg")

    monkeypatch.setattr("researcher.application.run_ask", boom)
    assert main(["ask", "A valid question"]) == 1
    assert "Missing dependency: asyncpg" in capsys.readouterr().err


def test_main_maps_keyboard_interrupt_to_exit_130(monkeypatch, capsys):
    async def boom(args):
        raise KeyboardInterrupt

    monkeypatch.setattr("researcher.application.run_ask", boom)
    assert main(["ask", "A valid question"]) == 130
    assert "Cancelled" in capsys.readouterr().err


# ---- application.run_ask cleanup/error paths (lines 156-229) ----

@pytest.mark.asyncio
async def test_run_ask_reports_storage_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        "researcher.config.get_settings",
        lambda: SimpleNamespace(
            per_source_timeout_seconds=2,
            cache_ttl_seconds=60,
            max_sources_per_query=3,
        ),
    )
    monkeypatch.setattr(
        "researcher.storage.db.get_pool",
        AsyncMock(side_effect=asyncpg.PostgresError("down")),
    )
    monkeypatch.setattr("researcher.storage.db.close_pool", AsyncMock())

    args = SimpleNamespace(question="Q?", sources=None, no_cache=False)
    assert await run_ask(args) == 1
    assert "check database connectivity" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_run_ask_cleanup_failure_does_not_replace_result(monkeypatch):
    monkeypatch.setattr(
        "researcher.config.get_settings",
        lambda: SimpleNamespace(
            per_source_timeout_seconds=2,
            cache_ttl_seconds=60,
            max_sources_per_query=3,
        ),
    )
    monkeypatch.setattr(
        "researcher.storage.db.get_pool", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(
        "researcher.storage.db.close_pool",
        AsyncMock(side_effect=RuntimeError("cleanup boom")),
    )

    async def fake_execute_ask(*a, **k):
        return 0

    monkeypatch.setattr("researcher.application.execute_ask", fake_execute_ask)

    args = SimpleNamespace(question="Q?", sources=None, no_cache=False)
    assert await run_ask(args) == 0