"""Pydantic schemas for Topic 4 — Async Research Assistant."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Source(BaseModel):
    """A retrieved excerpt from one of the research sources.

    `origin` is one of: "wikipedia", "arxiv", "web". This lets the SE layer
    apply origin-specific weighting or filtering.

    Frozen so that the SE layer cannot mutate a Source produced by a fetcher.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    url: str
    snippet: str
    origin: str

    @field_validator("title")
    @classmethod
    def _title_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title must be non-empty")
        return v

    @field_validator("url")
    @classmethod
    def _url_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("url must be non-empty")
        return v

    @field_validator("origin")
    @classmethod
    def _origin_allowed(cls, v: str) -> str:
        if v not in ("wikipedia", "arxiv", "web"):
            raise ValueError(f"origin must be wikipedia|arxiv|web, got {v!r}")
        return v


class Citation(BaseModel):
    """One numbered citation in the final answer.

    `index` is 1-based and matches the `[N]` markers in `AnswerWithCitations.answer`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    source: Source


class AnswerWithCitations(BaseModel):
    """Final synthesized answer with bracketed citations.

    The `answer` text contains inline `[N]` markers; `citations` lists the
    sources keyed by those numbers.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        # Custom shape: citations are flattened with source fields inlined,
        # which is what the SE layer
        # and the demo render.
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [
                {
                    "index": c.index,
                    "title": c.source.title,
                    "url": c.source.url,
                    "origin": c.source.origin,
                }
                for c in self.citations
            ],
        }
