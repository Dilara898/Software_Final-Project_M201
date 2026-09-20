"""Covers researcher/ui.py (the Streamlit UI) via Streamlit's AppTest
harness, which re-executes the script -- including its top-level
`from X import Y` imports -- on every `.run()`, so monkeypatching the
*source* module (e.g. `researcher.application.run_research`) before
`.run()` takes effect exactly like it does for the CLI's own tests.

No real network/DB call is made: `run_research`, `get_pool`/`close_pool`
and `list_sessions` are always faked.
"""
from pathlib import Path
from unittest.mock import AsyncMock

import asyncpg
from streamlit.testing.v1 import AppTest

import researcher.application as application_module
import researcher.storage.db as db_module
import researcher.storage.history as history_module
from ai.schemas import AnswerWithCitations, Citation, Source
from researcher.concurrency.models import (
    CollectionResult,
    ResearchResult,
    ResearchTimings,
    SourceOutcome,
)
from researcher.exceptions import NoSourcesError

# AppTest.from_file resolves a RELATIVE path against the file that calls
# it -- i.e. tests/ -- not against the working directory, so a relative
# "researcher/ui.py" is looked up as tests/researcher/ui.py and never
# found. Anchor it to the repository root instead.
UI_SCRIPT = str(Path(__file__).resolve().parents[1] / "researcher" / "ui.py")


def _make_result(*, failed=(), empty=(), warnings=(), history_status="saved", session_id=5):
    source = Source(title="Photosynthesis", url="https://example.org/x", snippet="s", origin="web")
    outcomes = [SourceOutcome(name="web", sources=[source], status="ok")]
    for name in failed:
        outcomes.append(SourceOutcome(name=name, status="error", error_code="x"))
    for name in empty:
        outcomes.append(SourceOutcome(name=name, status="empty"))
    bundle = CollectionResult(request_id="r1", outcomes=outcomes, duration_ms=10)
    answer = AnswerWithCitations(
        question="What is photosynthesis",
        answer="It is a process [1].",
        citations=[Citation(index=1, source=source)],
    )
    return ResearchResult(
        request_id="r1",
        answer=answer,
        bundle=bundle,
        timings=ResearchTimings(
            collection_ms=1, synthesis_ms=1, answer_ready_ms=1, history_ms=1, total_ms=1
        ),
        warnings=list(warnings),
        history_status=history_status,
        session_id=session_id,
    )


def _at() -> AppTest:
    at = AppTest.from_file(UI_SCRIPT, default_timeout=15)
    at.run()
    return at


def test_initial_render_has_no_errors():
    at = _at()

    assert at.error == []
    assert at.exception == []
    assert len(at.text_input) == 1
    assert len(at.button) == 2  # Ask, Refresh


def test_ask_rejects_too_short_question_without_calling_research(monkeypatch):
    called = False

    async def fake_run_research(args):
        nonlocal called
        called = True
        return _make_result()

    monkeypatch.setattr(application_module, "run_research", fake_run_research)

    at = _at()
    at.text_input[0].set_value("ab").run()  # below MIN_QUESTION_LENGTH
    at.button[0].click().run()

    assert not called
    assert len(at.error) == 1
    assert "at least" in at.error[0].value


def test_ask_success_renders_answer_and_references(monkeypatch):
    async def fake_run_research(args):
        assert args.question == "What is photosynthesis"
        return _make_result()

    monkeypatch.setattr(application_module, "run_research", fake_run_research)

    at = _at()
    at.text_input[0].set_value("What is photosynthesis").run()
    at.button[0].click().run()

    assert at.error == []
    assert at.exception == []
    markdown_text = "\n".join(m.value for m in at.markdown)
    assert "It is a process [1]." in markdown_text
    assert "[1]" in markdown_text and "Photosynthesis" in markdown_text


def test_ask_notes_failed_and_empty_sources(monkeypatch):
    async def fake_run_research(args):
        return _make_result(failed=["wikipedia"], empty=["arxiv"])

    monkeypatch.setattr(application_module, "run_research", fake_run_research)

    at = _at()
    at.text_input[0].set_value("What is photosynthesis").run()
    at.button[0].click().run()

    warning_text = "\n".join(w.value for w in at.warning)
    info_text = "\n".join(i.value for i in at.info)
    assert "wikipedia" in warning_text
    assert "arxiv" in info_text


def test_ask_reports_no_sources_error_generically(monkeypatch):
    async def fake_run_research(args):
        raise NoSourcesError("no usable sources")

    monkeypatch.setattr(application_module, "run_research", fake_run_research)

    at = _at()
    at.text_input[0].set_value("What is photosynthesis").run()
    at.button[0].click().run()

    assert len(at.error) == 1
    assert "connectivity" in at.error[0].value


def test_ask_reports_asyncpg_interface_error_as_connectivity(monkeypatch):
    # asyncpg.InterfaceError is a sibling of PostgresError, not a subclass --
    # regression test for a real incident where it fell through to the
    # generic "Unexpected error" branch instead of the connectivity one.
    async def fake_run_research(args):
        raise asyncpg.InterfaceError("cannot perform operation: pool is closed")

    monkeypatch.setattr(application_module, "run_research", fake_run_research)

    at = _at()
    at.text_input[0].set_value("What is photosynthesis").run()
    at.button[0].click().run()

    assert len(at.error) == 1
    assert "connectivity" in at.error[0].value


def test_ask_reports_unexpected_error_generically(monkeypatch):
    async def fake_run_research(args):
        raise RuntimeError("something truly unexpected")

    monkeypatch.setattr(application_module, "run_research", fake_run_research)

    at = _at()
    at.text_input[0].set_value("What is photosynthesis").run()
    at.button[0].click().run()

    assert len(at.error) == 1
    assert "Unexpected error" in at.error[0].value


def test_history_shows_no_previous_queries(monkeypatch):
    monkeypatch.setattr(db_module, "get_pool", AsyncMock(return_value=object()))
    monkeypatch.setattr(db_module, "close_pool", AsyncMock())
    monkeypatch.setattr(history_module, "list_sessions", AsyncMock(return_value=[]))

    at = _at()
    at.button[1].click().run()  # Refresh, in the History tab

    assert at.error == []
    info_text = "\n".join(i.value for i in at.info)
    assert "No previous queries found." in info_text


def test_history_lists_past_sessions(monkeypatch):
    from types import SimpleNamespace

    session = SimpleNamespace(id=1, created_at="2026-09-19", question="Q?", answer="A.")
    monkeypatch.setattr(db_module, "get_pool", AsyncMock(return_value=object()))
    monkeypatch.setattr(db_module, "close_pool", AsyncMock())
    monkeypatch.setattr(
        history_module, "list_sessions", AsyncMock(return_value=[session])
    )

    at = _at()
    at.button[1].click().run()

    assert at.error == []
    expander_labels = [e.label for e in at.expander]
    assert any("Q?" in label for label in expander_labels)
