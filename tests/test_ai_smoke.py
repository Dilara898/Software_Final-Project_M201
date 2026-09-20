"""Smoke tests for the provided Topic 4 AI module.

Exercises the public interface end-to-end with fake providers (no network).
Students MUST NOT delete or weaken these tests — they are part of the
grading contract. Add your own tests in tests/test_*.py.
"""

from __future__ import annotations

import pytest

from ai import (
    Source, Citation, AnswerWithCitations,
    fetch_web, synthesize,
)
from ai.synthesizer import _extract_cited_indices
from ai.sources import _parse_arxiv_atom


# --- Source model ---------------------------------------------------------


def test_source_rejects_invalid_origin():
    with pytest.raises(ValueError):
        Source(title="t", url="u", snippet="s", origin="reddit")


def test_source_rejects_empty_title():
    with pytest.raises(ValueError):
        Source(title="  ", url="u", snippet="s", origin="web")


def test_source_is_frozen():
    s = Source(title="t", url="u", snippet="s", origin="web")
    with pytest.raises(Exception):
        s.title = "new"  # type: ignore


# --- synthesize -----------------------------------------------------------


def test_synthesize_returns_answer_with_citations(fake_llm, sample_sources):
    result = synthesize("What is photosynthesis?", sample_sources, llm=fake_llm)
    assert isinstance(result, AnswerWithCitations)
    assert result.question == "What is photosynthesis?"
    assert "[1]" in result.answer
    assert {c.index for c in result.citations} == {1, 2}


def test_synthesize_drops_out_of_range_citations(fake_llm, sample_sources):
    fake_llm.response = "Some claim [1] and another [99]."
    result = synthesize("Q", sample_sources, llm=fake_llm)
    indices = [c.index for c in result.citations]
    assert indices == [1]  # 99 is dropped


def test_synthesize_handles_no_citations_in_answer(fake_llm, sample_sources):
    fake_llm.response = "I cannot answer from the given sources."
    result = synthesize("Q", sample_sources, llm=fake_llm)
    assert result.citations == []
    assert result.answer == "I cannot answer from the given sources."


def test_synthesize_rejects_empty_question(fake_llm, sample_sources):
    with pytest.raises(ValueError):
        synthesize("   ", sample_sources, llm=fake_llm)


def test_synthesize_rejects_empty_sources(fake_llm):
    with pytest.raises(ValueError):
        synthesize("Q", [], llm=fake_llm)


def test_synthesize_passes_sources_to_prompt(fake_llm, sample_sources):
    synthesize("What is photosynthesis?", sample_sources, llm=fake_llm)
    prompt = fake_llm.calls[0]
    assert "Photosynthesis (Wikipedia)" in prompt
    assert "Calvin cycle (Wikipedia)" in prompt


# --- citation parsing -----------------------------------------------------


def test_extract_cited_indices_handles_grouped_citations():
    # "[1,2]" is a single bracket, "[3]" is another
    assert _extract_cited_indices("Claim [1,2] and [3].", n_sources=5) == [1, 2, 3]


def test_extract_cited_indices_dedupes_and_sorts():
    assert _extract_cited_indices("[2] then [1] then [2].", n_sources=3) == [1, 2]


# --- fetch_web -----------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_web_uses_provider(fake_web):
    out = await fetch_web("photosynthesis", provider=fake_web, max_results=3)
    assert fake_web.calls == ["photosynthesis"]
    assert all(s.origin == "web" for s in out)


@pytest.mark.asyncio
async def test_fetch_web_empty_query_short_circuits(fake_web):
    out = await fetch_web("   ", provider=fake_web)
    assert out == []
    assert fake_web.calls == []


# --- arXiv XML parsing ---------------------------------------------------


def test_parse_arxiv_atom_extracts_entries():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>A Study of Async Pipelines</title>
    <summary>We study async pipelines.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <title>Another Study</title>
    <summary>Different topic.</summary>
  </entry>
</feed>"""
    sources = _parse_arxiv_atom(xml)
    assert len(sources) == 2
    assert sources[0].origin == "arxiv"
    assert sources[0].title == "A Study of Async Pipelines"
    assert sources[0].url == "http://arxiv.org/abs/2401.00001v1"


def test_parse_arxiv_atom_empty_feed():
    xml = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert _parse_arxiv_atom(xml) == []


def test_source_rejects_extra_fields():
    """Pydantic ConfigDict(extra='forbid') enforces the schema contract."""
    with pytest.raises(Exception):  # pydantic.ValidationError
        Source(
            title="t", url="u", snippet="s", origin="web",
            totally_unknown_field=42,  # type: ignore[call-arg]
        )