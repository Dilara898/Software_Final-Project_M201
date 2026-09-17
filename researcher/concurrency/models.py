"""C-owned results and diagnostics; adapters can map these to shared models."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai.schemas import AnswerWithCitations, Source

SourceName = Literal["wikipedia", "arxiv", "web"]
SOURCE_NAMES: tuple[SourceName, ...] = ("wikipedia", "arxiv", "web")
SourceStatus = Literal["ok", "empty", "timeout", "error", "invalid", "queue_timeout"]
CacheStatus = Literal["hit", "miss", "bypass", "read_error"]


class ResultModel(BaseModel):
    """Reject accidental fields and attribute reassignment at boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class OperationWarning(ResultModel):
    code: str
    stage: str
    source: SourceName | None = None
    message: str


class SourceTimings(ResultModel):
    cache_read_ms: float = Field(default=0, ge=0)
    queue_ms: float = Field(default=0, ge=0)
    fetch_ms: float = Field(default=0, ge=0)
    cache_write_ms: float = Field(default=0, ge=0)
    total_ms: float = Field(default=0, ge=0)


class SourceOutcome(ResultModel):
    name: SourceName
    sources: list[Source] = Field(default_factory=list)
    status: SourceStatus
    cache_status: CacheStatus = "bypass"
    warnings: list[OperationWarning] = Field(default_factory=list)
    timings: SourceTimings = Field(default_factory=SourceTimings)
    error_code: str | None = None

    @model_validator(mode="after")
    def consistent_sources(self) -> Self:
        if bool(self.sources) != (self.status == "ok"):
            raise ValueError("only successful outcomes must contain sources")
        if any(source.origin != self.name for source in self.sources):
            raise ValueError("source origin must match the outcome name")
        return self

    @property
    def cached(self) -> bool:
        return self.cache_status == "hit"

    @property
    def elapsed_seconds(self) -> float:
        return self.timings.total_ms / 1000


class CacheStats(ResultModel):
    hits: int = 0
    misses: int = 0
    bypasses: int = 0
    read_errors: int = 0
    write_errors: int = 0


class CollectionResult(ResultModel):
    """Derived lists are computed from outcomes, so they cannot drift apart."""

    request_id: str
    outcomes: list[SourceOutcome]
    duration_ms: float = Field(ge=0)

    @property
    def sources(self) -> list[Source]:
        return [source for outcome in self.outcomes for source in outcome.sources]

    @property
    def used(self) -> list[str]:
        return [outcome.name for outcome in self.outcomes if outcome.status == "ok"]

    @property
    def failed(self) -> list[str]:
        return [
            outcome.name
            for outcome in self.outcomes
            if outcome.status not in ("ok", "empty")
        ]

    @property
    def empty(self) -> list[str]:
        return [outcome.name for outcome in self.outcomes if outcome.status == "empty"]

    @property
    def warnings(self) -> list[OperationWarning]:
        return [warning for outcome in self.outcomes for warning in outcome.warnings]

    @property
    def cache_stats(self) -> CacheStats:
        return CacheStats(
            hits=sum(o.cache_status == "hit" for o in self.outcomes),
            misses=sum(o.cache_status == "miss" for o in self.outcomes),
            bypasses=sum(o.cache_status == "bypass" for o in self.outcomes),
            read_errors=sum(o.cache_status == "read_error" for o in self.outcomes),
            write_errors=sum(w.stage == "cache_write" for w in self.warnings),
        )


class SessionRecord(ResultModel):
    """History port input, not a replacement for A's ResearchSession model."""

    request_id: str
    question: str
    answer: str
    sources_used: list[str]
    sources_failed: list[str]
    duration_ms: int = Field(
        ge=0, description="Time until the validated answer is ready"
    )


class ResearchTimings(ResultModel):
    collection_ms: float = Field(ge=0)
    synthesis_ms: float = Field(ge=0)
    answer_ready_ms: float = Field(ge=0)
    history_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class ResearchResult(ResultModel):
    request_id: str
    answer: AnswerWithCitations
    bundle: CollectionResult
    timings: ResearchTimings
    warnings: list[OperationWarning]
    history_status: Literal["saved", "failed", "unknown"]
    session_id: int | None = None

    @property
    def cache_stats(self) -> CacheStats:
        return self.bundle.cache_stats
