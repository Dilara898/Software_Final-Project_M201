"""Reproducible collection measurements; live services are injected by callers."""

from __future__ import annotations

import csv
import io
import statistics
import time
from collections.abc import Callable
from typing import Literal, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from researcher.concurrency.contracts import FetchService
from researcher.concurrency.models import CollectionResult, ResultModel
from researcher.concurrency.orchestrator import SourceOrchestrator


class Question(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", max_length=80)
    text: str = Field(min_length=1, max_length=500)

    @field_validator("text")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question text must not be whitespace")
        return value


class QuestionSet(BaseModel):
    questions: list[Question] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique")
        return self


class BenchmarkOptions(ResultModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    repeats: int = Field(default=3, ge=1, le=20, strict=True)
    concurrency: int = Field(default=3, ge=1, le=20, strict=True)
    source_timeout_seconds: float = Field(default=10, gt=0)
    queue_timeout_seconds: float = Field(default=10, gt=0)
    max_results_per_source: int = Field(default=3, ge=1, le=100, strict=True)
    warmup: bool = False


class QuestionMeasurement(ResultModel):
    question_id: str
    collection: CollectionResult


class BatchMeasurement(ResultModel):
    mode: Literal["sequential", "parallel"]
    repeat: int
    setup_ms: float
    collection_ms: float
    teardown_ms: float
    questions: list[QuestionMeasurement]


class BenchmarkReport(ResultModel):
    options: BenchmarkOptions
    batches: list[BatchMeasurement]

    @property
    def comparable_repeats(self) -> list[int]:
        comparable = []
        for repeat in range(1, self.options.repeats + 1):
            pair = [b for b in self.batches if b.repeat == repeat]
            if len(pair) != 2:
                continue
            signatures = []
            for batch in pair:
                signatures.append(
                    [
                        (q.question_id, o.name, o.status, len(o.sources))
                        for q in batch.questions
                        for o in q.collection.outcomes
                    ]
                )
            if signatures[0] == signatures[1] and all(
                s[2] == "ok" for s in signatures[0]
            ):
                comparable.append(repeat)
        return comparable

    def median_ms(self, mode: str) -> float | None:
        comparable = self.comparable_repeats
        values = [
            b.collection_ms
            for b in self.batches
            if b.mode == mode and b.repeat in comparable
        ]
        return statistics.median(values) if values else None

    @property
    def speedup(self) -> float | None:
        sequential, parallel = self.median_ms("sequential"), self.median_ms("parallel")
        return sequential / parallel if sequential is not None and parallel else None

    def csv_text(self) -> str:
        """All runs survive, including failures; repeated batch fields are not additive."""
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(
            [
                "repeat",
                "mode",
                "question_id",
                "request_id",
                "source",
                "status",
                "result_count",
                "cache_status",
                "queue_ms",
                "fetch_ms",
                "cache_read_ms",
                "cache_write_ms",
                "source_total_ms",
                "error_code",
                "batch_setup_ms",
                "batch_collection_ms",
                "batch_teardown_ms",
            ]
        )
        for batch in self.batches:
            for question in batch.questions:
                for outcome in question.collection.outcomes:
                    t = outcome.timings
                    writer.writerow(
                        [
                            batch.repeat,
                            batch.mode,
                            question.question_id,
                            question.collection.request_id,
                            outcome.name,
                            outcome.status,
                            len(outcome.sources),
                            outcome.cache_status,
                            t.queue_ms,
                            t.fetch_ms,
                            t.cache_read_ms,
                            t.cache_write_ms,
                            t.total_ms,
                            outcome.error_code or "",
                            batch.setup_ms,
                            batch.collection_ms,
                            batch.teardown_ms,
                        ]
                    )
        return output.getvalue()

    def markdown(self, label: str) -> str:
        lines = [
            f"# {label}",
            "",
            "Collection only; synthesis and history excluded.",
            "Cache reads/writes bypassed. One shared client per batch; question order is identical.",
            "Setup and teardown are measured separately from collection in both modes.",
            "Only pairs with all sources successful and matching result counts enter the speedup.",
            "Matching counts do not guarantee identical live source content.",
            "",
            f"Options: `{self.options.model_dump_json()}`",
            "",
            "| Repeat | Mode | Setup ms | Collection ms | Teardown ms |",
            "|---|---|---:|---:|---:|",
        ]
        for b in self.batches:
            lines.append(
                f"| {b.repeat} | {b.mode} | {b.setup_ms:.3f} | {b.collection_ms:.3f} | {b.teardown_ms:.3f} |"
            )
        lines += [
            "",
            f"Comparable repeats: {self.comparable_repeats}",
            f"Median sequential ms: {self.median_ms('sequential')}",
            f"Median parallel ms: {self.median_ms('parallel')}",
            f"Speedup: {self.speedup:.3f}x"
            if self.speedup is not None
            else "Speedup: unavailable (incomplete or incomparable runs)",
            "",
            "Per-source observations (fetch includes the service's retries and pacing):",
        ]
        for name in ("wikipedia", "arxiv", "web"):
            values = [
                o.timings.fetch_ms
                for b in self.batches
                if b.mode == "parallel"
                for q in b.questions
                for o in q.collection.outcomes
                if o.name == name and o.status == "ok"
            ]
            if values:
                lines.append(
                    f"- {name}: median successful parallel fetch {statistics.median(values):.3f} ms"
                )
        lines += [
            "",
            "CSV retains every measured run. Batch timing columns repeat on each source row; do not sum those columns.",
            "There is no guaranteed speedup threshold. Offline results do not establish live provider performance.",
        ]
        return "\n".join(lines) + "\n"


async def run_benchmark(
    questions: QuestionSet,
    options: BenchmarkOptions,
    *,
    service_factory: Callable[[], FetchService],
    client_factory: Callable[[], httpx.AsyncClient],
) -> BenchmarkReport:
    """Alternate order; each paired batch owns an equivalent fresh service/client."""
    batches: list[BatchMeasurement] = []
    # A separate excluded warm-up pair is opt-in, especially for live API quotas.
    first = 0 if options.warmup else 1
    for repeat in range(first, options.repeats + 1):
        modes = (False, True) if repeat % 2 else (True, False)
        for concurrent in modes:
            setup_started = time.perf_counter()
            service = service_factory()
            async with client_factory() as client:
                collector = SourceOrchestrator(
                    service,
                    client=client,
                    timeout_seconds=options.source_timeout_seconds,
                    max_concurrency=options.concurrency,
                    queue_timeout_seconds=options.queue_timeout_seconds,
                    max_results_per_source=options.max_results_per_source,
                )
                setup_ms = (time.perf_counter() - setup_started) * 1000
                collection_started = time.perf_counter()
                measurements = []
                for question in questions.questions:
                    result = await collector.collect(
                        question.text, use_cache=False, concurrent=concurrent
                    )
                    measurements.append(
                        QuestionMeasurement(question_id=question.id, collection=result)
                    )
                collection_ms = (time.perf_counter() - collection_started) * 1000
                teardown_started = time.perf_counter()
            teardown_ms = (time.perf_counter() - teardown_started) * 1000
            if repeat:
                batches.append(
                    BatchMeasurement(
                        mode="parallel" if concurrent else "sequential",
                        repeat=repeat,
                        setup_ms=setup_ms,
                        collection_ms=collection_ms,
                        teardown_ms=teardown_ms,
                        questions=measurements,
                    )
                )
    return BenchmarkReport(options=options, batches=batches)
