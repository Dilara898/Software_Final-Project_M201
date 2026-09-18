# Role C: concurrency, research workflow and benchmark

## Scope and boundaries

C owns `researcher/concurrency/`, its tests, `scripts/benchmark.py`,
`scripts/c_offline.py`, `scripts/degradation_demo.py` and these documents.
The supplied `ai/` and shared A/B/D files are unchanged. A owns application
settings/models/storage; B owns provider integration, retries and rate limits;
D owns the CLI and Docker composition. Production integration remains pending
those implementations. Python 3.11+ syntax is used; runtime verification is on
Python 3.14 Windows (3.11 is not installed here).

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-concurrency.txt
New-Item -ItemType Directory -Force .cache | Out-Null
.\.venv\Scripts\python.exe -m pytest tests/test_orchestrator.py tests/test_c_lifecycle.py tests/test_c_research.py tests/test_c_benchmark.py tests/test_c_review_regressions.py tests/test_c_integration.py --basetemp=.cache/pytest-c -q --cov=researcher.concurrency --cov-branch --cov-report=term-missing
.\.venv\Scripts\python.exe scripts/benchmark.py --offline --repeats 3 --out artefacts/benchmark-offline-v2.md --csv artefacts/benchmark-offline-v2.csv
.\.venv\Scripts\python.exe scripts/degradation_demo.py
```

These commands need no API key or DB. The scoped requirements are development
dependencies for C, to be reconciled by D with the final application requirements.

## Integration contract

Construct one `SourceOrchestrator` per shared concurrency budget in one event
loop. Inject an open `httpx.AsyncClient`; the caller owns and closes it after all
research calls finish. Without injection, each collection owns a temporary client.

- `service.fetch(source, query, client) -> list[ai.schemas.Source]` is async.
- Source names are `wikipedia`, `arxiv`, `web`; aliases are resolved by B/D.
- Output order is canonical regardless of task completion order.
- `timeout_seconds` maps to A's per-source setting. `max_concurrency` bounds
  simultaneous fetch operations, including B's retries. It is not an RPS limiter.
- Queue (default 10s), fetch, cache read and cache write (each default 0.5s)
  have separate deadlines. Cache work does not hold semaphore slots.
- Inject `cache.get(key)` / `cache.set(key, sources)` and A's
  `cache_key(source, question)` together. `use_cache=False` skips both operations.
- Expected storage connectivity failures must be mapped by A's adapter to
  `StorageUnavailableError`. Unexpected programming errors propagate.
- A cache read failure falls back to fetch. A cache write failure retains good
  sources and adds a warning. Invalid cached entries are fetched again.
- Empty provider results, invalid data, provider errors, fetch deadlines and
  queue deadlines are distinct outcomes. Invalid items are removed, results
  limited (default three per source), with visible warnings.

`CollectionResult.sources`, `.used`, `.failed`, `.empty`, `.warnings`, and
`.cache_stats` are derived Python properties. JSON contains the underlying
`outcomes`; D should explicitly serialize derived properties if needed.
C's result models do not replace A's canonical application models.

Compose `ResearchOrchestrator(collector, synthesis, history)` where:

- `synthesis.synthesize(question, sources) -> AnswerWithCitations` is async.
- `history.save(SessionRecord) -> positive int` is async. A can use
  `HistoryCallbackAdapter` with a callback converting C's record into A's model.
  Map fields explicitly: question, answer, sources_used, sources_failed,
  duration_ms; preserve request_id for correlation if A supports it.
- Defaults: synthesis 20s, history 2s, complete research 45s. Inject agreed A
  settings at composition time; C does not read environment variables.
- `await runner.research(question)` returns a typed answer, collection, warnings,
  timings and history status. No usable source raises `NoSourcesError` before
  synthesis/history. D maps C's `OrchestrationError` family to CLI diagnostics.
- Citation validation checks numeric markers against the reference list and
  actual ordered sources, not factual truth. Citations are required by default;
  explicitly disable that policy for uncited insufficient-evidence answers.
- History outages preserve the answer. Timeout is `unknown`, since an insert
  may have committed; C never retries that insert automatically. The overall
  research deadline may still abort a request during history persistence.
- `SessionRecord.duration_ms` measures answer-ready time. Returned timings also
  include history and total wall time. Do not treat them as identical metrics.

Cancellation cleans up async source tasks and releases semaphore slots. Neither
asyncio cancellation nor a caller timeout forcibly stops a synchronous SDK
thread. B must configure SDK timeouts and prevent blocking the event loop.
Cleanup avoids sending a redundant cancellation to already-cancelling tasks.
A second explicit caller cancellation can still interrupt cleanup; adapters must
cooperate with cancellation and release their own resources.

## Benchmark interpretation

The v2 benchmark shares one client per batch, separates setup/collection/teardown,
alternates sequential/parallel order, disables caching, and retains every source
outcome in CSV. All questions use all three sources; dataset expected_sources
metadata does not change the workload. Speedup uses only paired repeats with
all sources successful and identical result counts. Content can still change
between live runs. Failed/incomparable measured batches cause exit code 2 after
artifacts are written. No performance threshold is asserted by unit tests.

Live invocation requires `--live --service-factory package.module:factory`.
The HTTP client uses the selected source timeout; the source deadline still
bounds the complete fetch including retries. Service-level timeout overrides
must be recorded separately. Output paths cannot replace the questions dataset.
The B-owned synchronous factory must return an async FetchService ready for use
with the injected client; it must not require unhandled async setup or teardown.
Each batch creates a service. B must ensure provider pacing survives across
batches, especially warmup boundaries; do not reset a global rate limit on each
factory call. Retries, limits and service configuration must match both modes.
Record those settings with the final live report. Warmup is optional and consumes
live quota. Offline mode never falls back to live HTTP; live never falls back to
fake data. Do not include secrets in factory specs or command-line arguments,
since the reproduction command is recorded.

`artefacts/benchmark-offline.txt` is the historical first measurement. The v2
artifacts are new simulated measurements, not evidence of live API speedup.
Synthesis and database performance require separate integration measurements.

## Remaining team work

1. A/B/D compose their real adapters with the contracts above.
2. Verify the real SDK threading/timeouts and storage error mapping.
3. Run real DB integration and complete CLI/Docker end-to-end tests.
4. Run the live paired benchmark with agreed provider pacing and credentials.
5. Include real measurements, limitations and AI assistance disclosure in the
   team report. Current implementation and documents used Codex assistance.

The original distributed smoke tests are not present in this repository's
starter commit. They were copied unchanged into ignored `.cache/ai-contract/`
for local verification; the shared setup owner should restore them unchanged
in the final repository.

## Adapters for the current A/D interfaces

`researcher.concurrency.integration` now provides:

- `StorageCacheAdapter(store)`: wrap A's InMemoryCacheStore or PostgresCacheStore.
  PostgreSQL connection/unavailability errors become C's StorageUnavailableError;
  invalid Pydantic cached Source data triggers fallback. Timeouts/cancellation
  keep their original meaning; SQL schema and programming errors propagate.
- `SessionHistoryAdapter(pool)`: maps C's SessionRecord fields to A's
  ResearchSession and calls A's save_session exactly once. The pool remains
  owned by application startup/shutdown; this adapter never calls get_pool.
- `sources_from_cli(args.sources)`: delegates D's comma-separated selection to
  B's `core.logic.select_sources`, including aliases, canonical ordering and
  shared `researcher.exceptions.ValidationError`. None means all sources. Pass
  `use_cache=not args.no_cache` to research.

Composition at the application's existing startup point, after obtaining the
pool, settings and HTTP client (imports omitted):

```python
cache = StorageCacheAdapter(PostgresCacheStore(pool, settings.cache_ttl_seconds))
fetch_service = AIFetchService(
    max_results=settings.max_sources_per_query,
    timeout_seconds=settings.per_source_timeout_seconds,
    min_interval_seconds=provider_interval_seconds,
)
synthesis_service = AISynthesisService(llm_factories=llm_factories)
collector = SourceOrchestrator(
    fetch_service, client=client, cache=cache, cache_key=cache_key,
    timeout_seconds=settings.per_source_timeout_seconds,
    max_concurrency=concurrency_limit,
    max_results_per_source=settings.max_sources_per_query,
)
runner = ResearchOrchestrator(collector, synthesis_service, SessionHistoryAdapter(pool))
try:
    result = await runner.research(
        question, sources_from_cli(args.sources), use_cache=not args.no_cache,
    )
finally:
    synthesis_service.shutdown()
```

This is a composition example, not a standalone entry point. B's services are
merged on main 386f3d4; final CLI lifecycle remains D-owned. The application owns
the HTTP client, pool and synthesis service; do not construct a fresh synthesis
service for each request. Shut it down once at application teardown, after all
requests finish. Shutdown does not stop a running synchronous SDK call.
The application supplies concurrency_limit, provider_interval_seconds and
llm_factories from its agreed configuration; A's current Settings does not expose
all these fields. B's default pacing interval is zero (disabled). Its Wikipedia
transport retries and whole-fetch retries can both consume the source deadline.
The outer C deadline covers pacing plus all attempts, not each attempt separately.

Offline A/C compatibility tests live in tests/test_c_integration.py. They exercise
A's actual memory/Postgres cache and history functions; DB operations are mocked.
Real B wrappers and supplied Wikipedia/arXiv/synthesis functions are also exercised
in tests/test_c_b_integration.py with HTTPX MockTransport and a fake LLM. Tests
cover D parser mapping, cold/warm/bypass cache, summary retry, permanent source
failure, citations, history and all-failed collection without constructing an LLM.
Run `python -m pytest tests/test_c_integration.py tests/test_c_b_integration.py -q`.

Main 7f10a7b fixes A's import-time settings failure with lazy `get_settings()`;
the application should call it at startup, after parsing and validating input.
The full offline suite now passes: 231 tests (including 136 C tests).
B's shared exceptions now exist, so D validation imports successfully.
C's NoSourcesError retains collection diagnostics; D should catch/map C's
OrchestrationError family separately from B's ResearcherError family. A blank
collection is stopped by C before B synthesis, so the two NoSourcesError types do
not need to be merged or replaced.
