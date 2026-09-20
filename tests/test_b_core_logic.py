"""Tests for researcher/core/logic.py (B6, B7) -- source selection
(alias/dedup/validation) and reference formatting, matching the external
review's named scenarios: wiki->wikipedia alias, duplicate names, empty
selection -> all sources, unknown name -> ValidationError, stable order,
and "only used references" for a subset like [1, 3].
"""

from __future__ import annotations

import pytest

from ai.schemas import AnswerWithCitations, Citation, Source
from researcher.core.logic import (
    format_references,
    normalize_source_name,
    select_sources,
    used_references,
)
from researcher.exceptions import ValidationError


def make_source(origin: str, title: str = "t") -> Source:
    return Source(title=title, url=f"https://example.invalid/{title}", snippet="s", origin=origin)


class TestNormalizeSourceName:
    def test_wiki_alias_resolves_to_wikipedia(self):
        assert normalize_source_name("wiki") == "wikipedia"

    def test_canonical_name_passes_through(self):
        assert normalize_source_name("arxiv") == "arxiv"

    def test_case_and_whitespace_insensitive(self):
        assert normalize_source_name("  WIKI  ") == "wikipedia"

    def test_unknown_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            normalize_source_name("not-a-real-source")


class TestSelectSources:
    def test_empty_selection_returns_all_sources(self):
        assert select_sources(None) == ["wikipedia", "arxiv", "web"]
        assert select_sources([]) == ["wikipedia", "arxiv", "web"]

    def test_wiki_alias_in_selection(self):
        assert select_sources(["wiki"]) == ["wikipedia"]

    def test_duplicate_names_collapse_to_one(self):
        assert select_sources(["wiki", "wikipedia", "wiki"]) == ["wikipedia"]

    def test_alias_and_canonical_together_collapse(self):
        assert select_sources(["web", "internet", "search"]) == ["web"]

    def test_output_order_is_always_canonical_regardless_of_input_order(self):
        assert select_sources(["web", "wikipedia", "arxiv"]) == ["wikipedia", "arxiv", "web"]
        assert select_sources(["arxiv", "wiki"]) == ["wikipedia", "arxiv"]

    def test_unknown_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            select_sources(["wikipedia", "not-a-real-source"])


class TestUsedReferencesAndFormatting:
    def test_used_references_sorted_ascending(self):
        s1, s2 = make_source("wikipedia", "A"), make_source("arxiv", "B")
        answer = AnswerWithCitations(
            question="q",
            answer="Supported [3][1].",
            citations=[Citation(index=3, source=s2), Citation(index=1, source=s1)],
        )
        refs = used_references(answer)
        assert [c.index for c in refs] == [1, 3]

    def test_only_cited_indices_are_present_not_a_full_candidate_list(self):
        # Mirrors the review's "[1, 3] -> only used references" scenario:
        # ai.synthesizer already restricts citations to used indices, so
        # this just confirms the adapter doesn't add anything back in.
        s1, s3 = make_source("wikipedia", "One"), make_source("web", "Three")
        answer = AnswerWithCitations(
            question="q",
            answer="Claim [1] and another [3].",
            citations=[Citation(index=1, source=s1), Citation(index=3, source=s3)],
        )
        refs = used_references(answer)
        assert [c.index for c in refs] == [1, 3]
        assert {c.source.title for c in refs} == {"One", "Three"}

    def test_format_references_renders_numbered_list(self):
        s1 = make_source("wikipedia", "Python (programming language)")
        answer = AnswerWithCitations(
            question="q",
            answer="Supported [1].",
            citations=[Citation(index=1, source=s1)],
        )
        text = format_references(answer)
        assert text == (
            "[1] Python (programming language) -- "
            "https://example.invalid/Python (programming language)"
        )

    def test_format_references_empty_when_no_citations(self):
        answer = AnswerWithCitations(question="q", answer="No citations here.", citations=[])
        assert format_references(answer) == ""