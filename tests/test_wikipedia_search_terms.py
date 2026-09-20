"""Wikipedia's opensearch endpoint matches title prefixes, not free text.

A whole question therefore matches nothing and `ai.fetch_wikipedia` returns an
empty list without raising -- a successful, empty fetch that is indistinguishable
from "no such article". Measured against the live API before this was added:
all five questions in data/research_questions.json returned zero titles, while
the same topics as short phrases returned three each.

`ai/` is supplied code we must not modify, and the brief forbids calling
Wikipedia's API directly, so the only lever is the string we pass in. These
tests pin the term generation and the retry-until-non-empty behaviour around it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai.schemas import Source
from researcher.core.logic import wikipedia_search_terms
from researcher.services import ai_service
from researcher.services.ai_service import AIFetchService


def source(title: str) -> Source:
    return Source(
        title=title,
        url="https://en.wikipedia.org/wiki/Example",
        snippet="example",
        origin="wikipedia",
    )


class TestWikipediaSearchTerms:
    def test_question_words_are_dropped(self):
        terms = wikipedia_search_terms("What is photosynthesis?")
        assert terms == ["photosynthesis"]

    def test_longest_span_comes_first(self):
        terms = wikipedia_search_terms(
            "How does CRISPR-Cas9 gene editing work at a molecular level?"
        )
        # Most specific candidate first: a three-word title beats a one-word one.
        assert terms[0] == "CRISPR-Cas9 gene editing"

    def test_both_ends_of_a_question_are_tried(self):
        terms = wikipedia_search_terms(
            "What were the main causes of the 2008 financial crisis?"
        )
        # The topic sits at the end here, so the rightmost span must be reached
        # even though a leftmost span of the same length is tried first.
        assert "2008 financial crisis" in terms
        assert terms.index("causes 2008 financial") < terms.index("2008 financial crisis")

    def test_short_spans_survive_a_long_question(self):
        """A long question must not spend every candidate on failing long spans."""
        terms = wikipedia_search_terms(
            "How do transformer-based language models handle long context windows?"
        )
        # "context windows" is the span that actually resolves against
        # Wikipedia; capping per length rather than overall is what keeps it
        # in the list at all.
        assert "context windows" in terms

    def test_casing_and_hyphens_are_preserved(self):
        terms = wikipedia_search_terms("How does CRISPR-Cas9 work?")
        assert "CRISPR-Cas9" in terms

    def test_terms_are_unique(self):
        terms = wikipedia_search_terms("What is fusion energy research about?")
        assert len(terms) == len(set(t.lower() for t in terms))

    def test_a_question_of_only_stopwords_still_yields_something(self):
        # Never hand the caller an empty list -- it would have nothing to send.
        assert wikipedia_search_terms("What is it?") == ["What is it"]

    def test_empty_question_yields_no_terms(self):
        assert wikipedia_search_terms("   ") == []

    def test_plain_topic_is_returned_unchanged(self):
        assert wikipedia_search_terms("photosynthesis") == ["photosynthesis"]


class TestFetchUsesTermsInOrder:
    @pytest.mark.asyncio
    async def test_first_non_empty_term_wins(self, monkeypatch):
        seen: list[str] = []

        async def fake(query, *, max_results, client):
            seen.append(query)
            # The whole question and the first candidate match nothing;
            # a later, shorter candidate does.
            return [source("Photosynthesis")] if query == "photosynthesis" else []

        monkeypatch.setitem(ai_service._FETCHERS, "wikipedia", fake)
        result = await AIFetchService(min_interval_seconds=0.0).fetch(
            "wikipedia",
            "What is photosynthesis and what are its main stages?",
            client=SimpleNamespace(headers={}),
        )

        assert [s.title for s in result] == ["Photosynthesis"]
        assert seen == ["photosynthesis stages", "photosynthesis"]

    @pytest.mark.asyncio
    async def test_all_terms_empty_returns_empty_without_raising(self, monkeypatch):
        calls = {"n": 0}

        async def always_empty(query, *, max_results, client):
            calls["n"] += 1
            return []

        monkeypatch.setitem(ai_service._FETCHERS, "wikipedia", always_empty)
        result = await AIFetchService(min_interval_seconds=0.0).fetch(
            "wikipedia", "What is photosynthesis?", client=SimpleNamespace(headers={})
        )

        # An empty Wikipedia result is a degraded source, not an error: the
        # other two sources must still be able to answer.
        assert result == []
        assert calls["n"] >= 1

    @pytest.mark.asyncio
    async def test_other_sources_receive_the_question_unchanged(self, monkeypatch):
        seen: list[str] = []

        async def fake(query, *, max_results, client):
            seen.append(query)
            return [source("Paper")]

        monkeypatch.setitem(ai_service._FETCHERS, "arxiv", fake)
        question = "What is photosynthesis and what are its main stages?"
        await AIFetchService(min_interval_seconds=0.0).fetch("arxiv", question, client=object())

        # arXiv and the web provider do full-text search; rewriting their query
        # would lose information rather than gain it.
        assert seen == [question]

    @pytest.mark.asyncio
    async def test_a_hit_on_the_first_term_makes_only_one_request(self, monkeypatch):
        fetcher = AsyncMock(return_value=[source("CRISPR-Cas9 gene editing")])
        monkeypatch.setitem(ai_service._FETCHERS, "wikipedia", fetcher)

        await AIFetchService(min_interval_seconds=0.0).fetch(
            "wikipedia",
            "How does CRISPR-Cas9 gene editing work at a molecular level?",
            client=SimpleNamespace(headers={}),
        )

        assert fetcher.await_count == 1

    @pytest.mark.asyncio
    async def test_each_candidate_gets_its_own_timeout_budget(self, monkeypatch):
        """A question needing several candidates must not exhaust one deadline.

        The sweep first shared a single `timeout_seconds` window, so a question
        whose working term came late timed out before that term was ever tried.
        Observed live on "How do transformer-based language models handle long
        context windows?", which reported `source_timeout: wikipedia`.
        """
        seen: list[str] = []

        async def slow_then_hit(query, *, max_results, client):
            seen.append(query)
            # Every candidate costs most of one attempt window; only the
            # fourth returns anything. A shared deadline would expire first.
            await asyncio.sleep(0.04)
            return [source("Context window")] if query == "context windows" else []

        monkeypatch.setitem(ai_service._FETCHERS, "wikipedia", slow_then_hit)
        service = AIFetchService(timeout_seconds=0.08, min_interval_seconds=0.0)

        result = await service.fetch(
            "wikipedia",
            "How do transformer-based language models handle long context windows?",
            client=SimpleNamespace(headers={}),
        )

        assert [s.title for s in result] == ["Context window"]
        assert "context windows" in seen
        assert len(seen) > 1  # earlier candidates were tried and came back empty
