# Async Research Assistant

AI-ENG-110 Final Project — Topic 4. The user asks a research question; the
system queries **Wikipedia**, **arXiv**, and a **web-search API** (Tavily)
concurrently, retrieves relevant excerpts, and synthesizes a single answer
with inline `[N]` citations using an LLM (Gemini by default, with optional
Anthropic/OpenAI failover).

Two front ends share one pipeline: a **CLI** (`python -m researcher`) and a
**Streamlit UI** (`researcher/ui.py`). Neither talks to the other over HTTP —
both call the same `researcher.application` composition directly.

## Architecture

```
ai/                    # PROVIDED — do not modify. Fetchers, synthesizer, provider adapters.
researcher/
├── config.py          # .env -> typed Settings (pydantic-settings, lazy-loaded)
├── models.py           # SE-layer pydantic models (ResearchSession, SourceBundle)
├── exceptions.py       # Shared CLI-facing exception family
├── validation.py       # Input validation (question length, history limit)
├── core/logic.py        # Source-name aliasing, reference formatting
├── services/
│   ├── ai_service.py    # AIFetchService / AISynthesisService — wraps ai/ with
│   │                     #   retry+backoff, timeouts, rate limiting, failover
│   ├── retry.py          # Shared Tenacity policy, HTTP-status-aware classifier
│   ├── transport.py      # Transport-level retry for Wikipedia's swallowed errors
│   └── logging_setup.py  # Structured logging, no secrets ever logged
├── concurrency/
│   ├── contracts.py      # Protocol ports (FetchService, SynthesisService, Cache, HistoryStore)
│   ├── orchestrator.py   # SourceOrchestrator — bounded concurrent fetch + cache-aside
│   ├── research.py       # ResearchOrchestrator — collect -> synthesize -> validate -> persist
│   └── integration.py    # Adapters wiring C's ports to A's storage
├── storage/
│   ├── db.py             # asyncpg pool (lazy, lock-guarded)
│   ├── cache_store.py     # In-memory + Postgres cache backends
│   └── history.py         # Session history persistence
├── application.py       # Composition root: builds the real pipeline for CLI + UI
├── cli.py / __main__.py  # `python -m researcher ask|history`
└── ui.py                 # Streamlit UI (`streamlit run researcher/ui.py`)
```

Each concern is a `Protocol`-typed port (`researcher/concurrency/contracts.py`),
so the AI module, storage, and concurrency layers compose without any layer
importing another's internals — the same abstract-base-class shape the
provided `ai/providers/base.py` uses, applied to storage/cache/fetch/synthesis.

## Setup

Requires **Python 3.11+** and a PostgreSQL database.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
cp .env.example .env              # then fill in the values below
```

### Environment variables (`.env`)

| Variable | Required | Default | Notes |
|---|---|---|---|
| `LLM_PROVIDER` | no | `gemini` | `anthropic` \| `openai` \| `gemini` |
| `LLM_MODEL` | no | provider default | e.g. `gemini-flash-latest` |
| `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | yes (one, matching `LLM_PROVIDER`) | — | Never commit real keys |
| `WEB_SEARCH_PROVIDER` | no | `tavily` | `tavily` \| `serper` \| `duckduckgo` |
| `TAVILY_API_KEY` | yes (if using Tavily) | — | Free tier: 1000 req/month |
| `DATABASE_URL` | **yes** | — | `postgresql://user:pass@host:port/db` |
| `LOG_LEVEL` | no | `INFO` | stdlib `logging` level |
| `CACHE_TTL_SECONDS` | no | `86400` | Source-fetch cache TTL |
| `PER_SOURCE_TIMEOUT_SECONDS` | no | `10` | Outer per-source fetch deadline |
| `MAX_SOURCES_PER_QUERY` | no | `3` | Results kept per source |

### Database

Create the database and apply the schema once:

```bash
psql -h <host> -p <port> -U <user> -c "CREATE DATABASE researcher;"
psql -h <host> -p <port> -U <user> -d researcher -f researcher/storage/schema.sql
```

Or use the bundled Postgres via Docker Compose (see below), which applies
`schema.sql` automatically on first start.

## Running

**CLI:**

```bash
python -m researcher ask "What is photosynthesis?"
python -m researcher ask "How does CRISPR-Cas9 work?" --sources wiki,arxiv
python -m researcher ask "..." --no-cache
python -m researcher history --limit 5
```

**Streamlit UI** (Ask + History tabs, same pipeline as the CLI):

```bash
streamlit run researcher/ui.py
# open http://localhost:8501
```

**Scripted 5-question demo** (matches the assignment's grading demo — runs
every question in `data/research_questions.json` through the real pipeline):

```bash
python scripts/demo.py --out artefacts/demo-run
```

## Testing

```bash
pytest --cov=researcher --cov-branch --cov-report=term-missing
```

All tests run fully offline (the AI module and all HTTP/DB calls are
mocked/faked) — zero live-network dependencies in CI.

Current results on this codebase:

- **306 tests passing**, **97% statement+branch coverage** of `researcher/`
  (target: ≥60%).
- The provided AI smoke tests (`tests/test_ai_smoke.py`, 16 tests) pass
  unmodified.
- `ruff check .` — clean, aside from one pre-existing unused import inside
  the provided smoke-test file itself.
- `mypy researcher --ignore-missing-imports` — 2 findings: one inside the
  *provided* `ai/` package (out of scope to fix), one a known
  `pydantic-settings` + mypy false positive in `config.py`.

## Concurrency benchmark

`scripts/benchmark.py` compares sequential vs. concurrent source collection
(3 sources × 5 questions × 3 repeats, cache disabled, one shared HTTP client
per batch — see `artefacts/benchmark-offline-v2.md` for the full report and
raw CSV):

| Mode | Median collection time |
|---|---:|
| Sequential | 942.3 ms |
| Concurrent (`asyncio.gather`, bounded semaphore) | 467.6 ms |
| **Speedup** | **2.02x** |

Parallelism is bounded by a configurable `asyncio.Semaphore`
(`max_concurrency` on `SourceOrchestrator`) so fan-out never exceeds provider
rate limits, independent of retries.

A live run against real providers (`artefacts/live-20260919T093055786500Z/`)
completed all 5 example questions successfully end-to-end (Wikipedia + arXiv
+ Tavily + Gemini synthesis + Postgres history).

## Docker

```bash
docker compose up -d db                              # Postgres, schema applied automatically
docker compose build app
docker compose run --rm app ask "What is photosynthesis?"
docker compose run --rm app history --limit 5
```

`app` and `db` share the compose network, so `DATABASE_URL` resolves to the
`db` service automatically (set in `docker-compose.yml`) — no host-networking
flags needed. The image entrypoint is `python -m researcher`, so any CLI
subcommand works the same way inside the container as it does locally.

## Known limitations

- Gemini's free tier caps requests per model per day; hitting that limit
  surfaces as a clean `SynthesisError` (retried 3x with backoff first), not
  a crash — the pipeline degrades gracefully and other questions in a batch
  are unaffected.
- The Streamlit UI and CLI share the same Postgres connection pool logic but
  are separate processes; there is no shared cache between a running UI
  session and a concurrent CLI invocation beyond what Postgres itself
  provides.

## Tools & Acknowledgements

Substantial portions of the SE layer (service wrappers, concurrency
orchestration, the Streamlit UI, and this README) were developed with AI
assistance (Claude). All code was reviewed and is understood by the team.
