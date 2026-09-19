# AI-ENG-110 – Final Project Report

**Topic 4 — Async Research Assistant**

| | |
|---|---|
| **Team** | *TODO: team name* |
| **Repo** | https://github.com/Dilara898/Software_Final-Project_M201 |
| **Date** | 2026-09-19 *(update to actual submission date)* |
| **Tag** | v1.0-final *(not yet tagged — see §7/§8)* |
| **Members** | Dilara Rzazada (`@Dilara898`), Nezrin Ibrahimova (`@NezrinIbrahimova`), Sanan Hajiyev (`@20hajiyev`), Melek Askerova (`@` *— confirm handle*) *(verify all four handles against actual GitHub profiles before submission; inferred from commit author metadata, not confirmed)* |

> **Note on team size.** Git history shows four distinct contributors. Per the brief, teams of 4 require the instructor's prior written approval (§3.1) — confirm this was obtained and reference it here if so.

---

## Executive Summary

We built an async research assistant that answers a natural-language question by querying Wikipedia, arXiv, and Tavily web search concurrently, then synthesizes one Gemini-generated answer with inline, verifiable `[N]` citations — exposed through both a CLI (`python -m researcher ask`) and a Streamlit UI that call the exact same pipeline. The concurrency design (bounded `asyncio.gather` over the three sources) measured a **2.02x speedup** against a sequential baseline (942ms → 468ms median, 5 questions × 3 repeats). Our headline robustness story is not synthetic: during live testing we genuinely exhausted Gemini's free-tier daily quota multiple times, and each time the system degraded exactly as designed — three retries with exponential backoff, a clean `SynthesisError` instead of a crash, and the other in-flight questions in the same batch completing unaffected. The most surprising thing we learned came from that same testing: a second, unrelated failure (`asyncpg.InterfaceError`, from two concurrent UI sessions racing on one shared connection pool) revealed that our storage layer's per-request "create pool, use it, tear it down" lifecycle — correct for the one-shot CLI — is not safe under genuinely concurrent long-running UI sessions. If we started over, we would design the storage layer's connection lifecycle around the *process* (one pool, opened once, held for the app's lifetime) rather than around the *request*, from day one, instead of discovering the gap through a live incident.

---

## 1. Project Overview

### 1.1 Problem framing & scope

Topic 4 asks for a system that takes a natural-language research question, retrieves supporting material from multiple independent sources in parallel, and returns a single synthesized answer whose claims are traceable back to those sources by numbered citation. Our system does exactly this over three sources — Wikipedia (REST API), arXiv (Atom API), and web search (Tavily) — all provided, unmodified, by the course's `ai/` package.

**Inputs.** A free-text question, an optional subset of sources (`wiki`, `arxiv`, `web`, comma-separated or multi-selected), and an optional cache-bypass flag.

**Outputs.** A synthesized answer with inline `[N]` citations, a reference list (title + URL + origin per citation), a per-source outcome summary (which sources succeeded, returned nothing, or failed), cache/timing statistics, and a persisted history record.

**User-facing entry points.**
- CLI: `python -m researcher ask "<question>" [--sources wiki,arxiv,web] [--no-cache]` and `python -m researcher history [--limit N]`.
- Streamlit UI (`streamlit run researcher/ui.py`): an "Ask" tab mirroring the CLI's `ask` command and a "History" tab mirroring `history`.
- A scripted batch runner, `scripts/demo.py`, which runs every question in `data/research_questions.json` through the identical pipeline and writes artefacts (used as our grading-demo reproduction and, incidentally, as the run that first surfaced the quota incident described in §4.2).

There is deliberately **no HTTP API**. The brief lists an HTTP API as "recommended" for Topic 4 but mandatory only for Topics 1–2; we judged that the two front ends we do have (CLI + UI) already demonstrate the required "CLI is mandatory, HTTP is optional" split more thoroughly than a third, unused HTTP surface would, and put that time into the concurrency/robustness/testing work the rubric weights far more heavily (22% + 10% + 8% + 7% vs. 0% for an API specifically).

**Provider stack.** `LLM_PROVIDER=gemini` (Google Gemini, via `ai.providers.google.GeminiLLM`), `WEB_SEARCH_PROVIDER=tavily`. No embedding provider is used — Topic 4's spec does not call for embeddings anywhere in its pipeline (unlike Topics 1/3), so we did not configure or store any embedding vectors.

**Where we tightened the spec.** The brief's citation requirement ("each fact carries a numeric reference") is enforced, not just hoped for: `ResearchOrchestrator.validate_answer()` (§2.1/§4.1) rejects an answer whose `[N]` markers don't exactly match its declared citation list, or that cites an index outside the source list — an LLM that hallucinates a citation number fails validation rather than silently shipping a broken reference.

**Where we loosened it.** The brief's example CLI usage shows `--sources wiki,arxiv` restricting to a subset; we additionally accept the full canonical names (`wikipedia`) and common synonyms (`internet`, `search` → `web`) via `core/logic.py`'s alias table, on the reasoning that a CLI a human actually types into should tolerate the name they naturally reach for, not just the exact three tokens documented.

### 1.2 Provider choice and the AI module contract

We use Google Gemini for synthesis (`LLM_PROVIDER=gemini`; `LLM_MODEL` has been re-pointed three times during live testing — `gemini-3.6-flash` → `gemini-flash-latest` → `gemini-flash-lite-latest` — each time because the *previous* model's free-tier daily quota was exhausted by our own testing, not because of a code defect; see §4.2) and Tavily for web search (`WEB_SEARCH_PROVIDER=tavily`). We call exactly four functions from the provided `ai/` package: `ai.sources.fetch_wikipedia`, `ai.sources.fetch_arxiv`, `ai.sources.fetch_web`, and `ai.synthesizer.synthesize` (which internally resolves the LLM via `ai.providers.factory.get_llm`). **We did not modify the public interface of `ai/` at any point** — the only exception is one file, `ai/providers/openai.py`, that mypy flags with a pre-existing argument-type issue (§6.1); we deliberately left it untouched rather than "fixing" supplied code, per §4.8 of the brief.

Two independent defenses sit between a provider response and anything downstream ever seeing it. First, **shape validation**: `AIFetchService._call` checks that a fetcher actually returned `list[Source]` before it's allowed past the retry boundary — anything else raises `UpstreamDataError` and is not retried, since retrying a malformed-response shape can't fix it. Second, **semantic validation over the synthesized answer itself**: `ResearchOrchestrator.validate_answer` doesn't trust that a syntactically valid `AnswerWithCitations` is *semantically* correct — it independently re-extracts every `[N]` marker from the answer text via regex, and rejects the answer (`InvalidAnswerError`) if those markers don't exactly match the declared `citations` list, if any citation index falls outside `1..len(sources)`, or if a citation's embedded `Source` doesn't equal the actual source at that position. This is the "valid JSON that is semantically wrong" case the template asks about, concretely handled: a model that cites `[7]` when only 3 sources exist, or a citation object that's been silently reordered, is caught here — not downstream in a broken reference list.

## 2. Architecture

### 2.1 Module map

```
ai/                                  PROVIDED — unmodified
  sources.py, synthesizer.py, providers/{base,factory,google,...}.py

researcher/
  config.py            settings boundary (env -> typed Settings, lazy-loaded)
  models.py            SE-layer pydantic models (ResearchSession, SourceBundle)
  exceptions.py        shared CLI/UI-facing exception family
  validation.py        input validation (question length, history limit)
  core/logic.py         source-name aliasing, reference formatting
  services/            <-- wraps ai/ --------------------------------
    ai_service.py        AIFetchService, AISynthesisService (retry, timeout,
                          rate limiting, multi-provider failover)
    retry.py             shared Tenacity policy + HTTP-status classifier
    transport.py         transport-level retry for Wikipedia's swallowed errors
    logging_setup.py     structured logging, secrets never logged
  concurrency/          <-- orchestration --------------------------
    contracts.py          Protocol ports: FetchService, SynthesisService,
                           Cache, HistoryStore
    orchestrator.py        SourceOrchestrator: bounded concurrent fetch
    research.py             ResearchOrchestrator: collect -> synthesize ->
                             validate -> persist
    integration.py          adapters: C's ports <-> A's storage
  storage/               <-- persistence ----------------------------
    db.py                  asyncpg pool (process-global, lock-guarded)
    cache_store.py          CacheStore(ABC) + Postgres/in-memory adapters
    history.py               session history persistence
  application.py        composition root: builds the real pipeline
  cli.py, __main__.py    entry point: `python -m researcher ask|history`
  ui.py                  Streamlit UI (same pipeline, different presentation)
```

**Boundary crossings.** Every arrow that crosses the `ai/` boundary passes through exactly one of `AIFetchService.fetch` or `AISynthesisService.synthesize` — no other file imports `ai.sources` or `ai.synthesizer` directly. Every arrow that crosses the storage boundary passes through the `Cache`/`HistoryStore` `Protocol`s in `contracts.py`, implemented concretely by `StorageCacheAdapter`/`SessionHistoryAdapter` in `integration.py`. **If we swapped `LLM_PROVIDER` from `gemini` to `anthropic`, zero files would change** — the swap is a single environment variable, resolved at runtime by `ai.providers.factory.get_llm`; our own code only ever depends on the `LLMProvider` abstract interface, never a concrete provider class.

### 2.2 OOP design & key abstractions

Our clearest example of an abstract base class we designed ourselves (not `ai/`'s) is the cache backend in `researcher/storage/cache_store.py`:

```python
class CacheStore(ABC):
    @abstractmethod
    async def get(self, key: str) -> list[Source] | None: ...

    @abstractmethod
    async def set(self, key: str, sources: list[Source]) -> None: ...


class InMemoryCacheStore(CacheStore):
    """TTL-based dict cache -- used by tests, no network or DB required."""
    ...


class PostgresCacheStore(CacheStore):
    """JSONB-backed cache with an expiry column, used in production."""
    ...
```

**Why an ABC and not a `Protocol` here specifically**, when the concurrency layer uses `Protocol`s (`contracts.py`) for the exact same kind of substitutability: the two serve different callers. `contracts.py`'s `Protocol`s exist so that C's orchestrators never import A's or B's concrete modules at all — structural typing lets three people build against an interface none of them owns a copy of. `CacheStore`, by contrast, is owned and instantiated entirely within the storage layer itself (`PostgresCacheStore`/`InMemoryCacheStore` are the only two implementations that will ever exist, and both are defined right next to the ABC); an ABC's `@abstractmethod` gives us a constructor-time guarantee — you cannot instantiate a `CacheStore` subclass that forgot to implement `set()` — that a `Protocol` (which is only checked at the call site, not at construction) doesn't. We reserved `Protocol` for the cross-team seam and ABC for the single-owner extension point, rather than picking one style project-wide.

A second, deliberately different pattern — **composition over inheritance** — appears in `AISynthesisService`: `TokenBudgetLLM` wraps whatever `LLMProvider` the failover chain produces, raising its `max_tokens` floor to 4096 without needing to be a subclass of anything (`self._provider = provider`, `self.complete(...)` delegates and adjusts one argument). Inheriting from each concrete provider to add this behavior would have meant three subclasses (`TokenBudgetAnthropicLLM`, `TokenBudgetOpenAILLM`, ...) for one cross-cutting concern; composition needed exactly one class regardless of how many providers exist.

### 2.3 Data flow across module boundaries

Our own SE-layer pydantic models: `ResearchSession`, `SourceBundle` (`researcher/models.py`); `SourceOutcome`, `SourceTimings`, `CollectionResult`, `CacheStats`, `SessionRecord`, `ResearchTimings`, `ResearchResult`, `OperationWarning` (`researcher/concurrency/models.py`). Every one of these — and every `ai/`-provided model (`Source`, `Citation`, `AnswerWithCitations`) that flows alongside them — is a pydantic model with `extra="forbid"`. **No naked dictionary ever crosses a module boundary in this codebase.**

The hardest call was `SessionRecord` vs. `ResearchSession`: C's `ResearchOrchestrator` needed a history-write shape (`SessionRecord`) that carries a `request_id` for correlation and doesn't know anything about A's actual database row shape (`ResearchSession`, which additionally carries `id`/`created_at`, populated only after an insert). Rather than have C construct or import A's model directly — which would mean the concurrency layer's tests need a real (or faked) `ResearchSession` just to type-check — we kept them as two distinct models and wrote one explicit adapter, `SessionHistoryAdapter.save()` in `integration.py`, whose entire job is the field-by-field translation between them. The trigger for wrapping, in general, was ownership: any model that a *different* layer would otherwise have to import to call into ours got its own thin, local shape instead, translated at the seam.

## 3. Concurrency & Performance

### 3.1 Why this concurrency model

We use **`asyncio.gather` bounded by an `asyncio.Semaphore`** for the three-source fan-out (`SourceOrchestrator.collect`), and, separately, a small **`ThreadPoolExecutor`** (`_BoundedThreadRunner`, `max_workers=1` by default) for the LLM synthesis call specifically. These are not competing choices — they solve two different problems in the same request. Fetching Wikipedia/arXiv/Tavily is genuinely I/O-bound (`httpx.AsyncClient` under the hood), so `asyncio` is the correct primitive: three concurrent HTTP round-trips block on network I/O, not the interpreter, so there is no GIL contention to work around and no reason to pay thread-creation overhead. `ai.synthesizer.synthesize`, however, calls a **synchronous** provider SDK — `asyncio.to_thread` would work, but it uses the loop's shared default executor (capacity `min(32, cpu_count + 4)`), which means a timed-out retry starts a *second* concurrent thread on top of the still-running first one; three retries against a stuck call would mean three simultaneous, separately-billed API calls for one logical request. Routing every attempt through our own single-worker pool instead means a queued retry cannot start until the previous attempt's thread actually finishes, so a stuck call costs at most one real running thread no matter how many retries pile up (`researcher/services/ai_service.py`, `_BoundedThreadRunner`).

The bound is `max_concurrency=3` (`SourceOrchestrator`, set in `application.py`'s composition root), chosen to exactly match the number of sources — every fetch for a single request can run at once, and the semaphore's real job is to be *the* place that bound lives, so it is trivially adjustable (a config value, not a hardcoded fan-out) if a future change added more sources or ran multiple requests through one orchestrator instance.

**Threads instead of asyncio, for the fetch side, would have worked but wasted capacity**: `httpx`'s async client already gives us non-blocking I/O for free, so a thread pool would only add context-switch and GIL-acquisition overhead for no concurrency benefit over three already-non-blocking coroutines. The smallest workload where the concurrent design starts winning is any request touching more than one source at all — even two sources overlap their network latency instead of summing it, and our benchmark shows the win is already ~2x at exactly three sources. Setting the semaphore to 1 would fully serialize the three fetches (equivalent to the sequential baseline in §3.2); setting it to 100 would make no observable difference for this workload, since only three tasks are ever submitted per request — the ceiling only matters once concurrent *requests* (not concurrent *sources within one request*), which nothing in the current composition root bounds (see the incident in §4.2 and the honest limitation in §8).

### 3.2 Sequential-vs-concurrent benchmark

Measured with `scripts/benchmark.py --offline --repeats 3`, using 5 questions from `data/research_questions.json`, 3 sources each, cache disabled, one shared `httpx.AsyncClient` per batch, on Windows / Python 3.14 (per the artefact's recorded runtime; the rest of this codebase targets 3.11+ syntax). "Offline" here means source responses come from `scripts/c_offline.py`'s fixed synthetic delays (40/80/40 ms per source), not live providers — this isolates the orchestration overhead itself from provider/network variance, and is explicitly not a claim about live-provider wall-clock time.

| Workload | N (repeats) | Sequential (median) | Concurrent (median) | Speedup |
|---|---:|---:|---:|---:|
| 5 questions × 3 sources, cache off | 3 | 942.3 ms | 467.6 ms | **2.02x** |

Reproduce: `python scripts/benchmark.py --offline --repeats 3 --out artefacts/benchmark-offline-v2.md --csv artefacts/benchmark-offline-v2.csv` (full per-repeat numbers and per-source medians in `artefacts/benchmark-offline-v2.md`).

**Honest interpretation.** With 3 sources we cannot reach 3x — synthetic per-source delays are 40/80/40 ms, so the sequential sum (~160 ms of *pure* fetch time, plus per-request overhead) is bounded below by the slowest single source (80 ms, arXiv) once parallel; the *measured* ~2x rather than a higher ratio reflects that setup/teardown and queueing are measured separately and are not zero, and that the offline harness's synthetic delays are deliberately modest. The new bottleneck under concurrency is exactly what asyncio theory predicts: **the slowest single source** (arXiv, median parallel fetch 93.057 ms vs. Wikipedia/web's ~46 ms each) — the whole batch cannot finish faster than its slowest member. We additionally have one **live** run (`artefacts/live-20260919T093055786500Z/`) confirming the pipeline completes all 5 questions successfully end-to-end against real providers, though we did not capture a paired live sequential-vs-concurrent timing (a gap noted in §8).

### 3.3 Rate limit & politeness

Two independent mechanisms, not one: a **min-interval pacer** (`RateLimiter` in `researcher/services/retry.py`) applied per source name inside `AIFetchService`, configured at `min_interval_seconds=1.0` in the composition root — every call to the same source, including retries, waits at least 1 second since that source's last call before proceeding; and **exponential-backoff retry** (`call_with_retry`, Tenacity-based) applied to both fetch and synthesis, `max_attempts=3`, jittered exponential backoff between an `initial_wait` and `max_wait` that scale with the configured per-source timeout budget (`per_source_timeout_seconds`, default 10s → `initial_wait≈0.25s`, `max_wait≈0.5s` for fetch; synthesis defaults to `initial_wait=0.5s`, `max_wait=6.0s`). The retry predicate is HTTP-status-aware, not a blanket type check: a `429` or `5xx` is retried, a `401`/`404` is not, since retrying a bad key or a malformed request only burns the timeout budget on an outcome retries cannot change (`_is_transient_status` in `retry.py`).

**This is not untested in anger — it is the single most-exercised path in the whole project.** During live testing this session we repeatedly and genuinely exhausted Gemini's free-tier daily quota (see §4.2 for the full incident), which meant we watched the real retry/backoff sequence fire dozens of times against a real `429 RESOURCE_EXHAUSTED` response, not a mocked one. A representative log excerpt from one such run:

```
INFO  provider_call_attempt | attempt=1 operation='synthesize'
WARN  provider_call_retry   | attempt=1 error_code='provider_error_status_429' wait_seconds=0.559
INFO  provider_call_attempt | attempt=2 operation='synthesize'
WARN  provider_call_retry   | attempt=2 error_code='provider_error_status_429' wait_seconds=1.76
INFO  provider_call_attempt | attempt=3 operation='synthesize'
WARN  llm_provider_exhausted | error_code='provider_error_status_429' provider_index=0
ERROR UI request failed (SynthesisError)
```

Exactly three attempts, growing jittered backoff, then a clean typed failure — no crash, no hang, no silent retry-forever.

## 4. Robustness & Error Handling

### 4.1 Wrapping the AI module

`AIFetchService` and `AISynthesisService` (`researcher/services/ai_service.py`) are the entire wrapper. **Retries:** `ProviderError`, `asyncio.TimeoutError`, `ConnectionError`, and `httpx.TransportError` are retryable by type; within `ProviderError` specifically, the wrapper walks the exception's `__cause__` chain to recover an HTTP status code (if any) and only retries `429`/`5xx` — a `401`/`404` propagates immediately (`retry.py::_should_retry`). Three attempts, jittered exponential backoff (§3.3). **Timeouts:** every attempt gets its own hard `asyncio.wait_for` deadline (fetch: `per_source_timeout_seconds / 4` ≈ 2.5s per attempt by default; synthesis: 15s per attempt), separate from the outer per-source/per-request deadlines the orchestration layer enforces — an attempt cannot silently run past its own budget even if the caller's overall deadline is longer. **Structured logging:** every attempt, retry, success, and exhaustion logs at INFO/WARNING with `extra={operation, attempt, error_code, duration_ms, source}` fields — never `str(exc)`, since provider SDK exceptions can embed API keys or full request URLs in their message text; `error_code_for()` maps any exception to a short, closed-vocabulary code (e.g. `provider_error_status_429`, `provider_timeout`) instead. **Input validation:** before calling, `AIFetchService.fetch` rejects an unknown source name as a non-retryable `UpstreamDataError`; on the response, it checks the fetcher actually returned `list[Source]` (not just "didn't raise") before treating the call as successful.

**Walking one path end-to-end — a 429 on synthesis, which we observed live, repeatedly:** `AISynthesisService.synthesize` calls `call_with_retry`; attempt 1 raises `ProviderError` wrapping a `429`; `_should_retry` recognizes it as transient and schedules a jittered backoff; attempt 2 and 3 repeat; after attempt 3 fails, `call_with_retry` re-raises the original exception (never Tenacity's own wrapper, so callers see types they already handle); `AISynthesisService` logs `llm_provider_exhausted` and, since no fallback provider was configured, raises `UpstreamDataError`; `ResearchOrchestrator._run` catches that alongside `ProviderError`/`httpx.HTTPError` and re-raises it as `SynthesisError`; the CLI/UI's outer handler catches the shared `OrchestrationError`/`ProviderError`/`ResearcherError` family and prints one clean, secret-free message. At no point does a raw traceback, an API key, or the model's internal error JSON reach the terminal or the browser.

### 4.2 Failure-mode analysis (one concrete incident)

**The failure mode:** Gemini free-tier daily quota exhaustion (`429 RESOURCE_EXHAUSTED`), discovered live, not injected synthetically — we did not need to simulate this; ordinary development testing produced it three separate times against three different models in succession (`gemini-3.6-flash`, then `gemini-flash-latest`, then `gemini-flash-lite-latest`), because each model's free tier caps at roughly 20 requests/day and our own iterative testing used that up each time.

**(i) How it was triggered.** No fault injection was needed — running `scripts/demo.py` against all 5 example questions, plus repeated manual UI testing on the same day, was sufficient to exhaust the quota organically.

**(ii) What the system did.** Source collection (Wikipedia/arXiv/web) succeeded normally in every case — the failure is isolated to the synthesis step. `AISynthesisService` retried 3 times with backoff (§3.3), then raised `UpstreamDataError` → `SynthesisError`. Critically, in the `scripts/demo.py` batch run, **the failure of one question did not affect the others**: 3 of 5 questions (q1, q2, q4) completed successfully with real cited answers persisted to Postgres; q3 and q5 failed cleanly and independently, and the script moved on and reported `Completed: 3/5` rather than aborting the batch.

**(iii) What the user saw.** In the CLI/UI, a single clean line: *"Research failed. Check provider settings, available sources and connectivity."* — no stack trace, no leaked error JSON, no partial/garbled answer.

**(iv) What the logs showed.** Exactly the sequence quoted in §3.3: three timestamped `provider_call_attempt`/`provider_call_retry` pairs with the real `error_code='provider_error_status_429'`, then `llm_provider_exhausted`, then `ERROR ... UI request failed (SynthesisError)` — sufficient on its own, with zero re-execution, to diagnose the exact cause (we confirmed the diagnosis independently by calling the Gemini SDK directly with the same key/model outside the app, and it returned the identical `429 RESOURCE_EXHAUSTED` with a quota message naming the exact model and the free-tier limit).

**What we expected vs. what happened, and a design gap it surfaced.** We expected retries+backoff to be the whole story. It mostly was — except that the *same* testing session, under the same load, also surfaced a **second, unrelated failure**: `asyncpg.InterfaceError` (a sibling of `asyncpg.PostgresError` in asyncpg's exception hierarchy, not a subclass of it) during a history write, which our UI's `render_error()` did not classify and showed as an unhelpful generic "Unexpected error" instead. Root cause: two overlapping research requests (visible in the log as two `research_started` events milliseconds apart — almost certainly two browser tabs or a double-click) raced on the single process-global connection pool that `run_research()`'s `finally` block closes after *every* request; the second request's database call landed on a pool the first request had already torn down. We fixed the immediate symptom (added `asyncpg.InterfaceError` to the classified-error branch, with a regression test) but explicitly did **not** fix the underlying per-request pool lifecycle in the time available — see §8.

### 4.3 Input validation

| Entry point | Rule | User-facing message on failure |
|---|---|---|
| CLI/UI `question` | 3–500 characters after whitespace normalization; empty rejected | `"Question must be at least 3 characters long."` / `"...cannot exceed 500 characters (yours is N)."` |
| CLI/UI `--sources` / source multiselect | Must resolve (via alias table) to `wikipedia`/`arxiv`/`web`; unknown name rejected outright, not silently dropped | `"Unknown source 'X'. Expected one of: wikipedia, arxiv, web (aliases: ...)."` |
| CLI `history --limit` / UI "Show last N" | Integer, 1–100 | `"--limit must be between 1 and 100 (yours is N)."` |
| `DATABASE_URL` (env) | Required, no default | `pydantic.ValidationError` → mapped to `"Invalid configuration. Check DATABASE_URL and .env settings."` |
| Missing Python dependency | N/A | `"Missing dependency: <name>. Install the project requirements."` |

**Adversarial questions from the template, answered honestly rather than force-fit:** a malformed *image* cannot affect this system — Topic 4 has no image input anywhere in its pipeline (that's Topics 1/2's surface). A 100 MB JSON request is not applicable either, since we deliberately did not build an HTTP API (§1.1); the only request bodies that exist are CLI argv and Streamlit widget values, neither of which accepts an arbitrary request payload. A 50,000-character query is explicitly rejected by the 500-character cap above, well before it reaches any provider call. A path that escapes the storage directory does not apply — this system writes nothing to the filesystem from user input at all; every persisted value goes through parameterized SQL (`asyncpg`'s `$1`/`$2` placeholders) into Postgres, never a file path built from user text.

## 5. Storage & Caching

### 5.1 Schema

We use **PostgreSQL** (via `asyncpg`), per Topic 4's actual storage requirement — not SQLite, which appears only in this template's illustrative boilerplate.

```sql
CREATE TABLE IF NOT EXISTS cache_entries (
    key TEXT PRIMARY KEY,
    payload JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries (expires_at);

CREATE TABLE IF NOT EXISTS research_sessions (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources_used TEXT[] NOT NULL DEFAULT '{}',
    sources_failed TEXT[] NOT NULL DEFAULT '{}',
    duration_ms INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

`cache_entries.expires_at` and `research_sessions.created_at`/`id` are the only derived/system-assigned fields; everything else is source-of-truth data supplied by the request itself. There is no foreign key between the two tables — a cache entry and a history record are independent facts about the same question, deliberately not joined, since a cache entry's lifetime (TTL-bound) and a history record's lifetime (permanent audit trail) are different concerns.

**Why `JSONB`, not a `BLOB`:** `cache_entries.payload` stores a JSON-serialized `list[Source]` (`json.dumps([s.model_dump() for s in sources])`), not opaque bytes, because the cached value is structured, human-inspectable data (titles, URLs, snippets) that benefits from Postgres's native JSON type — debuggable with a plain `SELECT`, not just a hex dump. **The embedding-dimension question does not apply to our system**: Topic 4's pipeline never computes or stores an embedding vector anywhere (no `ai.embedding` calls exist in our dependency graph at all), so there is no dimension-compatibility concern to design around.

### 5.2 Caching policy

We cache **per-source fetch results**, keyed by `(source, normalized_question)` (`cache_key()` in `cache_store.py`: lowercased, whitespace-collapsed, trailing punctuation stripped, then SHA-256'd, prefixed with the source name so a partial `--sources wiki` request never gets cached as if it were a full three-source result). Default TTL is `CACHE_TTL_SECONDS=86400` (24h), configurable via `.env`. Invalidation is passive expiry, checked on read (`PostgresCacheStore.get` deletes and returns `None` for an expired row rather than serving stale data) — there is no background eviction process. A cache **write** failure never discards a successful fetch (`SourceOrchestrator._one` logs a warning and proceeds with the already-fetched sources); a cache **read** failure or an invalid cached shape (e.g. tampered/corrupt JSON) falls back to a live fetch rather than raising. We have not run a demo specifically instrumented to report a single hit-rate number for this report — the live UI's fact-pill row does report per-request cache hits/misses/bypasses, but we did not aggregate a session-level rate (noted as a gap in §8 rather than invented here).

**Staleness in production:** a Wikipedia summary or arXiv abstract cached for up to 24h could go stale if the underlying page changes within that window — acceptable for a research assistant's freshness bar, but not for time-sensitive queries. In production we would add a short manual "force refresh" affordance (already partially present as `--no-cache`/"Skip cache") plus, if this saw real traffic, a shorter TTL for fast-moving topics detected heuristically (e.g. news-adjacent queries) rather than one global TTL.

## 6. Testing

### 6.1 Strategy & coverage

Reported by `pytest --cov=researcher --cov-branch` (statement + branch coverage), per module:

| Module | Cov. | Module | Cov. |
|---|---:|---|---:|
| `config.py` | 100% | `models.py` | 100% |
| `exceptions.py` | 100% | `validation.py` | 100% |
| `core/logic.py` | 100% | `services/logging_setup.py` | 100% |
| `services/retry.py` | 97% | `services/transport.py` | 100% |
| `services/ai_service.py` | 97% | `concurrency/contracts.py` | 100% |
| `concurrency/models.py` | 99% | `concurrency/orchestrator.py` | 99% |
| `concurrency/research.py` | 100% | `concurrency/integration.py` | 95% |
| `concurrency/benchmarking.py` | 99% | `storage/cache_store.py` | 94% |
| `storage/db.py` | 91% | `storage/history.py` | 100% |
| `application.py` | 100% | `cli.py` | 98% |
| `__main__.py` | 80% | `ui.py` | 91% |

**Total: 98%** (307 tests). Provided `ai/` smoke tests (`tests/test_ai_smoke.py`, 16 tests): **passing — unmodified**.

All 307 tests run offline. The AI module is mocked with hand-written fakes (`FakeLLM`, `FakeWebSearch` in `tests/conftest.py`) rather than a network-mocking library, since `ai.sources`/`ai.synthesizer`'s public functions accept an injectable `client`/`llm` parameter specifically for this purpose; the HTTP layer for lower-level transport tests (e.g. Wikipedia's transient-retry wrapper) uses `httpx.MockTransport`. Zero tests touch a real network or a real database — storage-layer tests use fakes for `asyncpg`'s pool interface.

The two lowest-covered modules are both explainable rather than neglected: `__main__.py` (80%) is a 3-line `if __name__ == "__main__"` guard where only the guard branch itself, not its non-`__main__` counterpart, is meaningfully testable; `storage/db.py` (91%) is the pool-lifecycle code directly implicated in the incident in §4.2 — its remaining gap is exactly the concurrent-access branch that a deterministic offline test cannot easily reproduce (see "what we didn't test," §6.2).

**Lines we decided not to cover, and why:** a handful of provider-construction branches in `ai_service.py::_llm_factory_for` (the actual `AnthropicLLM()`/`OpenAILLM()`/`GeminiLLM()` calls inside each closure) are exercised only up to "the factory is callable," not "calling it succeeds," since actually invoking them requires real API keys for providers we don't use by default — testing that Anthropic's constructor works is effectively testing the *provided* `ai/` package, which already has its own contract tests. **A mock we had to redesign mid-project:** `researcher/storage/db.py`'s `get_pool()`/`close_pool()` were originally only ever tested by wholesale-monkeypatching the two functions (used everywhere else in the suite); we added a dedicated `tests/test_storage_db.py` late, specifically because that wholesale-mocking pattern meant the pool's own creation/caching/lock logic had *zero* direct coverage — the exact code path that turned out to matter in the live incident in §4.2.

### 6.2 Notable tests

**(i) Happy-path end-to-end:** `tests/test_d_integration.py`'s `test_execute_ask_...` constructs real `SourceOrchestrator`/`ResearchOrchestrator` instances wired to fakes end-to-end and asserts `execute_ask(...) == 0` with a real rendered answer — the same composition the CLI and UI actually run in production, not a shortcut around it.

**(ii) Concurrency test:** the orchestrator's cancellation-cleanup suite (`tests/test_c_lifecycle.py`, `tests/test_c_review_regressions.py`) verifies that when `asyncio.gather` is cancelled mid-flight or one source task raises, in-flight sibling tasks are cleanly cancelled and awaited exactly once (no redundant cancellation of an already-cancelling task), and that partial failure (1 or 2 of 3 sources down) still returns the successful sources' results rather than losing everything.

**(iii) Error-path test that surfaced a real bug:** `tests/test_ui.py::test_ask_reports_asyncpg_interface_error_as_connectivity` — written *after* discovering, through live testing (§4.2), that `asyncpg.InterfaceError` fell through `render_error()`'s classification into an unhelpful generic message. This is the clearest example in the whole project of a test that exists because production use, not code review, found the bug.

**What we didn't test:** the concurrent-session race itself (two overlapping `run_research()` calls genuinely racing on the shared global pool) has no automated reproduction — it was found live, fixed cosmetically (§4.2), but not captured as a deterministic test, since reliably reproducing a cross-thread event-loop race in `pytest-asyncio` would need real threads or a redesigned pool lifecycle, either of which was out of scope for this pass. That is the test we most wish we'd had time to write.

## 7. Deployment & Reproducibility

### 7.1 Docker image

```bash
docker compose up -d db
docker compose build app
docker compose run --rm app ask "What is photosynthesis?"
```

Single-stage build, `python:3.12-slim` base, non-root `appuser`, `ENTRYPOINT ["python", "-m", "researcher"]`. **Image size was not measured for this report** — Docker was not available in the environment used to prepare it; before submission, run `docker images` after building and record the actual size here. We did not attempt a multi-stage build (the +1 bonus item) in this pass.

**Trade-off worth naming honestly:** `requirements.txt` is currently shared between the CLI (the only thing the Docker image runs) and the Streamlit UI, so the image installs `streamlit` and its full dependency tree (pandas/pyarrow/altair, etc.) even though the containerized entrypoint never imports `researcher.ui`. This is the single most obvious quick win if we revisit the Dockerfile: either split `requirements.txt` (CLI vs. UI) or move to a multi-stage build that only copies the packages the CLI target actually needs.

### 7.2 Configuration

| Variable | Required | If missing |
|---|---|---|
| `LLM_PROVIDER` | no (default `gemini`) | falls back to default |
| `LLM_MODEL` | no | provider's own default model |
| `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | yes, matching `LLM_PROVIDER` | provider construction raises `ProviderError`, surfaced as a clean CLI/UI error |
| `WEB_SEARCH_PROVIDER` | no (default `tavily`) | falls back to default |
| `TAVILY_API_KEY` | yes, if using Tavily | `ProviderError` at web-search construction |
| `DATABASE_URL` | **yes, no default** | `pydantic.ValidationError` at first settings access, mapped to a clean "check DATABASE_URL" message, not a raw traceback |
| `LOG_LEVEL` | no (default `INFO`) | defaults to `INFO` |
| `CACHE_TTL_SECONDS` | no (default `86400`) | defaults to 24h |
| `PER_SOURCE_TIMEOUT_SECONDS` | no (default `10`) | defaults to 10s |
| `MAX_SOURCES_PER_QUERY` | no (default `3`) | defaults to 3 |

## 8. Limitations & Future Work

- **Storage pool lifecycle is request-scoped, not process-scoped**, which is correct for the one-shot CLI but caused a real `asyncpg.InterfaceError` under concurrent UI sessions (§4.2). The fix — hold one pool for the life of the Streamlit process instead of recreating/closing it per request — is scoped but was not implemented this pass.
- **No cross-request rate limiting.** The `max_concurrency=3` semaphore bounds fan-out *within* one request; nothing currently stops multiple concurrent UI sessions from each independently hammering the same provider at the same time, which is part of why our free-tier Gemini quota was exhausted faster than a single-user testing session would predict.
- **The Docker image is unnecessarily large** for what it runs, since `requirements.txt` isn't split between the CLI and UI surfaces (§7.1) — not yet measured, but a known and named inefficiency rather than an unknown one.
- **No live (as opposed to offline-synthetic) sequential-vs-concurrent benchmark table was captured**, only a single live correctness run (5/5 or 3/5 depending on quota state at the time). The offline number (§3.2) is honest about isolating orchestration overhead from provider variance, but a paired live number would be a stronger report artifact.
- **Worst design decision, in hindsight:** sharing one `requirements.txt` (and by extension one Docker image target) between the CLI and the UI was expedient when the UI was added late, but it's exactly the kind of shortcut that compounds — the same instinct that made the pool-lifecycle bug invisible until a live UI session exercised it.

## 9. Tools & Acknowledgements

Substantial AI-assistant use is disclosed here rather than glossed over. Claude (via Claude Code) was used across most of this session's work, specifically:

- **`researcher/services/ai_service.py`, `retry.py`, `transport.py`** — drafted by Claude following an external code review's findings (documented in `docs/ROLE_B.md`'s "Fixed after external review" section); the team reviewed and integrated each fix.
- **`researcher/application.py` (composition root), `researcher/ui.py` (Streamlit UI), `.streamlit/config.toml`** — drafted by Claude, including the design-token theming system; reviewed against the full test suite (which was already in place and unmodified in behavior) before acceptance.
- **This report** — the section-by-section content was drafted by Claude directly from the project's verified test output, logs, and source code (not invented), and should be reviewed by the team for accuracy and voice before submission.
- **The `asyncpg.InterfaceError` fix and its regression test (§4.2, §6.2)** — diagnosed and written by Claude from a live production log during this session; a genuine example of an AI-assisted fix that traces to a real, observed incident rather than a hypothetical one.

## References

1. Google Gemini API documentation — https://ai.google.dev/gemini-api/docs
2. `tenacity` library — https://tenacity.readthedocs.io/
3. Streamlit theming configuration reference — https://docs.streamlit.io/
4. `asyncpg` documentation, exception hierarchy — https://magicstack.github.io/asyncpg/

## A. Appendix — Reproducing the benchmark

```bash
git clone https://github.com/Dilara898/Software_Final-Project_M201.git
cd Software_Final-Project_M201
git checkout ui-service   # or main, once merged
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/benchmark.py --offline --repeats 3 \
  --out artefacts/benchmark-offline-v2.md --csv artefacts/benchmark-offline-v2.csv
```

No API key or database is required for the offline benchmark. Expect numbers within ~20% of §3.2's table on comparable hardware.

---

*End of report.*
