# Role B: AI service and business logic

## Scope and boundaries

B owns `researcher/services/` (this file's code), its tests
(`tests/test_b_ai_service.py`), `requirements-b.txt` and this document. The
supplied `ai/` package and Part C's `researcher/concurrency/` are unchanged.
A owns application settings/models/storage; C owns concurrency, the research
workflow and the benchmark; D owns the CLI and Docker composition.

## What this module does

`ai/providers/base.py` says plainly: the provider adapters have "no retries,
no caching, no logging. Those are SE-layer concerns and belong to the
student's wrapper code." `researcher/services/` is that wrapper code:

- `researcher/services/retry.py` — one shared Tenacity retry/backoff policy
  (`call_with_retry`), reused identically by fetch and synthesis so behavior
  doesn't drift between the two. Retries `ProviderError`, `TimeoutError` and
  `ConnectionError`; everything else (bad input, malformed provider payloads)
  propagates on the first attempt. Every attempt has its own hard timeout via
  `asyncio.wait_for`. Logging only ever includes exception *types*/short
  codes (`error_code_for`), never `str(exc)` — SDK errors can embed API keys
  or full request URLs.
- `researcher/services/ai_service.py`:
  - `AIFetchService` implements C's `FetchService` port
    (`fetch(source, query, client) -> list[Source]`), dispatching to
    `ai.sources.fetch_wikipedia` / `fetch_arxiv` / `fetch_web` by name. These
    fetchers are already async/httpx-native, so no thread offloading is
    needed. Unknown source names or malformed fetcher output raise
    `UpstreamDataError` (C's port error for "provider returned unusable
    data") without being retried.
  - `AISynthesisService` implements C's `SynthesisService` port
    (`synthesize(question, sources) -> AnswerWithCitations`), wrapping the
    supplied, synchronous `ai.synthesizer.synthesize` (and whichever
    synchronous SDK it calls) in a small, dedicated `ThreadPoolExecutor`
    (`_BoundedThreadRunner`, default `max_workers=1`) rather than
    `asyncio.to_thread`. **Multi-provider failover (bonus):** if
    constructed with more than one LLM factory, each is tried in order; a
    provider only moves to the next after exhausting its own retry/backoff
    cycle. `default_llm_chain()` builds this from `LLM_PROVIDER` plus an
    explicit list of fallback provider names.
  - `make_fetch_service()` / `make_synthesis_service()` — zero-argument
    factories for composition. `make_fetch_service` is the target C's live
    benchmark expects: `--service-factory
    researcher.services.ai_service:make_fetch_service`.

## Integration contract

- Both service classes are `async def` end-to-end, matching C's `Protocol`
  definitions in `researcher/concurrency/contracts.py` exactly; C's
  orchestrator does runtime type checks on the return values
  (`FetchService.fetch must return list[Source]`) that will hard-fail on any
  deviation.
- `AIFetchService`/`AISynthesisService`'s own `timeout_seconds` /
  `max_attempts` /backoff bounds must fit **inside** C's own outer deadlines
  (`SourceOrchestrator.timeout_seconds`, `ResearchOrchestrator
  .synthesis_timeout_seconds`) — they are an inner budget, not a replacement
  for C's.
- A blocking SDK call cannot be forcibly killed by `asyncio.wait_for`/
  cancellation, full stop — no `timeout_seconds` value, however small,
  guarantees a stuck synchronous call actually stops. Keeping
  `AISynthesisService.timeout_seconds` at or below the provider SDK's own
  configured client timeout does not by itself prevent overlap either: if
  the SDK's own call finishes *after* the caller's deadline for unrelated
  reasons (network stall, server-side slowness), the two can still overlap.
  What `AISynthesisService` actually guarantees, via `_BoundedThreadRunner`,
  is *concurrency*, not cancellation: at most `max_concurrent_calls`
  (default 1) of its own threads are ever running at once, regardless of
  how many retries pile up against a stuck call — a retry whose own
  timeout fires while it is still queued (not yet running) is cleanly
  cancelled and never executes at all. See `ai_service.py`'s
  `_BoundedThreadRunner` docstring for the full mechanism. (SDK-side client
  timeout configuration was not added to the shared `ai/providers/*.py`
  files in this pass, to avoid touching supplied/shared code without team
  agreement — flag this if it needs revisiting; it would reduce how often a
  thread runs past its caller's deadline in the first place, which this
  fix does not prevent, only bound.)
- Sources returned by `AIFetchService.fetch` must keep `origin == source`;
  C's `valid_source()` check discards anything else as invalid.
- Read configuration (provider name, timeouts) from A's `researcher.config
  .Settings` where the team wires this together in production, rather than
  hardcoding values — the constructors here only default to environment
  variables (via `ai.providers.factory.get_llm`) for standalone use and
  testing.

## Run locally

```bash
pip install -r requirements-b.txt
python -m pytest tests/test_b_ai_service.py -q
```

No API key or database is required — every test uses fakes for the
underlying `ai.sources` fetchers and `ai.synthesizer.synthesize`, exactly
like `tests/test_c_research.py` does for the concurrency layer.

## Fixed after external review (2026-09-17, review of commit 3e52746)

An external technical review of `feat/b-ai-service` at commit `3e52746`
identified several real, reproduced issues (not style opinions). Fixed in
this revision:

- **B-02 (P1) — synthesis timeout retries piled up concurrent threads.**
  `asyncio.to_thread` uses the event loop's shared default executor; a
  timed-out retry started a *new* thread on top of the still-running
  previous one (reproduced: 3 attempts -> 3 concurrently running threads
  against a real LLM call). Fixed via `_BoundedThreadRunner`, above. This
  was also silently responsible for the ~30s Part B test suite runtime
  (each timeout-related test's stray thread delayed process exit); the
  suite now runs in under 1 second.

Not yet fixed, tracked for follow-up (see the review for full detail):
B-01 (shared `researcher/exceptions.py`), B-03 (retry classifier doesn't
distinguish permanent 401/404 from transient 429/5xx), B-04 (a bare
`except Exception` in `AIFetchService._call` intercepts retryable
`ConnectionError` before it reaches the retry policy), B-05 (no
provider-level pacing/rate limiting), B-06 (empty `sources` still triggers
one LLM factory call before failing), B-07 (Wikipedia's internal
per-title summary-fetch swallows HTTP errors before they reach this
wrapper's retry layer), plus the missing `logging_setup.py` and
`core/logic.py` (source alias/dedup/validation) deliverables.

## Known cross-role gap (not fixed here, flagged for the team)

`researcher/config.py` builds a module-level `Settings()` eagerly at import
time (`settings = Settings()`), so *any* import of `researcher.config`
without `DATABASE_URL` set raises immediately — this makes
`tests/test_config.py::test_missing_database_url_raises` fail even without
any Part B changes (confirmed against a clean checkout). This is A's file;
raised here for visibility rather than fixed unilaterally.

## Remaining team work

1. Wire `AIFetchService`/`AISynthesisService` into the real composition root
   (D) alongside A's cache/history adapters and C's orchestrators.
2. Agree on and add SDK-level client timeouts in `ai/providers/*.py` if the
   team wants a harder guarantee against a leaked thread on a stuck
   synchronous call.
3. Reconcile `requirements-b.txt` into the project's final `requirements.txt`
   (D).
4. Exercise a real provider outage/failover scenario against live
   credentials once available, in addition to the fake-based tests here.