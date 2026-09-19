from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.providers.base import ProviderError
"""Streamlit UI for the Async Research Assistant.

Reuses the exact same business logic the CLI uses -- `researcher.application
.run_research` calls the same `build_result` pipeline as `researcher.cli`'s
`ask` command (A's storage adapters, B's retrying AI services, C's
concurrent orchestrators) -- so this is a second presentation of one
pipeline, not a parallel implementation.

Visual design follows the project's editorial/technical design system
(warm-neutral palette, Instrument Sans + JetBrains Mono, restrained
monochrome with color reserved for genuine errors). Most of that system is
declared in `.streamlit/config.toml` via Streamlit's native theming API;
`inject_theme()` below covers the handful of things that API can't express
(the dotted background grid, and small presentational components -- tags,
section headers, fact pills -- for content this module renders itself).

Run with:
    streamlit run researcher/ui.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from types import SimpleNamespace

import asyncpg
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError as SettingsError

# On Streamlit Community Cloud (or any host using st.secrets instead of a
# .env file), secrets configured in the platform's UI land in `st.secrets`,
# not in the process environment -- but every other module in this project
# (pydantic-settings, os.getenv) only ever reads os.environ. Bridge the two
# before anything else runs. Locally, with no secrets.toml configured, this
# is a silent no-op and .env (below) is the only source, unchanged.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:  # noqa: BLE001 - no secrets.toml configured; nothing to bridge
    pass

# One-time process bootstrap, mirroring researcher.cli.main()'s own setup:
# existing process environment values take priority over .env, and empty
# optional values must not override a provider's own default.
load_dotenv(".env", override=False)
for _name in ("LLM_MODEL", "LLM_PROVIDER", "WEB_SEARCH_PROVIDER"):
    if _name in os.environ and not os.environ[_name].strip():
        os.environ.pop(_name)
os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("WEB_SEARCH_PROVIDER", "tavily")

from ai.providers.base import ProviderError  # noqa: E402
from researcher.application import run_research  # noqa: E402
from researcher.concurrency.models import ResearchResult  # noqa: E402
from researcher.concurrency.research import OrchestrationError  # noqa: E402
from researcher.exceptions import ResearcherError, ValidationError  # noqa: E402
from researcher.services.logging_setup import configure_logging  # noqa: E402
from researcher.storage.db import close_pool, get_pool  # noqa: E402
from researcher.storage.history import list_sessions  # noqa: E402
from researcher.validation import validate_limit, validate_question  # noqa: E402

configure_logging()
log = logging.getLogger(__name__)

st.set_page_config(page_title="Async Research Assistant", page_icon="\U0001f50e")


# ---------------------------------------------------------------------------
# Design system: tokens + the small set of custom components Streamlit's
# theme config can't express on its own. Kept intentionally minimal --
# colors, radii, fonts and borders already come from .streamlit/config.toml.
# ---------------------------------------------------------------------------

def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ara-grid: #DEDCD8;
          --ara-hairline: #E7E5E1;
          --ara-hairline-2: #D6D3CD;
          --ara-ink: #14130F;
          --ara-ink-2: #52504A;
          --ara-ink-3: #8B887F;
          --ara-solid: #14130F;
          --ara-solid-ink: #FAFAF9;
        }
        @media (prefers-color-scheme: dark) {
          :root {
            --ara-grid: #26241F;
            --ara-hairline: #2B2925;
            --ara-hairline-2: #3A3833;
            --ara-ink: #F7F5F1;
            --ara-ink-2: #B3AFA6;
            --ara-ink-3: #7C786F;
            --ara-solid: #F7F5F1;
            --ara-solid-ink: #14130F;
          }
        }
        @media (prefers-reduced-motion: reduce) {
          * { transition: none !important; animation-duration: 0.01ms !important; }
        }

        .stApp, [data-testid="stAppViewContainer"] {
          background-image: radial-gradient(var(--ara-grid) 1px, transparent 1px);
          background-size: 22px 22px;
          background-position: -1px -1px;
        }

        .ara-eyebrow {
          display: inline-flex; align-items: center; gap: 8px;
          font-family: "JetBrains Mono", monospace; font-size: 11px;
          letter-spacing: .1em; text-transform: uppercase;
          color: var(--ara-ink-2);
          border: 1px solid var(--ara-hairline-2);
          border-radius: 999px; padding: 5px 12px; margin-bottom: 16px;
        }
        .ara-eyebrow .dot {
          width: 6px; height: 6px; border-radius: 50%;
          background: var(--ara-ink); display: inline-block;
        }

        .ara-hero h1 {
          font-size: clamp(28px, 4.4vw, 38px); line-height: 1.06;
          letter-spacing: -.03em; font-weight: 700; margin: 0 0 10px;
          color: var(--ara-ink);
        }
        .ara-hero h1 em {
          font-style: italic; font-weight: 600; color: var(--ara-ink-2);
        }
        .ara-lede {
          font-size: 15px; color: var(--ara-ink-2); max-width: 62ch;
          margin: 0 0 4px; line-height: 1.55;
        }

        .ara-shead {
          display: flex; align-items: baseline; gap: 12px;
          margin: 8px 0 14px; flex-wrap: wrap;
        }
        .ara-snum {
          font-family: "JetBrains Mono", monospace; font-size: 11px;
          font-weight: 700; letter-spacing: .12em; color: var(--ara-ink-3);
        }
        .ara-shead h2 {
          font-size: 18px; letter-spacing: -.02em; font-weight: 700;
          margin: 0; color: var(--ara-ink);
        }
        .ara-snote { font-size: 12.5px; color: var(--ara-ink-3); }

        .ara-tag {
          display: inline-flex; align-items: center;
          font-family: "JetBrains Mono", monospace;
          font-size: 10px; letter-spacing: .08em; text-transform: uppercase;
          border: 1px solid var(--ara-hairline-2); border-radius: 999px;
          padding: 3px 9px; margin: 0 6px 0 0; color: var(--ara-ink-2);
        }
        .ara-tag.solid {
          background: var(--ara-solid); color: var(--ara-solid-ink);
          border-color: var(--ara-solid);
        }

        .ara-facts { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 2px; }
        .ara-fact {
          font-family: "JetBrains Mono", monospace; font-size: 11px;
          letter-spacing: .05em; color: var(--ara-ink-2);
          border: 1px solid var(--ara-hairline-2); border-radius: 999px;
          padding: 5px 11px;
        }
        .ara-fact b { color: var(--ara-ink); font-weight: 700; }

        .ara-citerow {
          padding: 9px 2px; border-top: 1px solid var(--ara-hairline);
          font-size: 13.5px; color: var(--ara-ink);
        }
        .ara-citerow:first-child { border-top: none; padding-top: 0; }
        .ara-citerow a { color: var(--ara-ink); font-weight: 600; }

        [data-testid="stTabs"] button[role="tab"] p {
          font-family: "JetBrains Mono", monospace; font-size: 11.5px;
          letter-spacing: .08em; text-transform: uppercase;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(number: str, title: str, note: str = "") -> None:
    note_html = f'<span class="ara-snote">{note}</span>' if note else ""
    st.markdown(
        f'<div class="ara-shead"><span class="ara-snum">{number}</span>'
        f"<h2>{title}</h2>{note_html}</div>",
        unsafe_allow_html=True,
    )


def tag(text: str, *, solid: bool = False) -> str:
    cls = "ara-tag solid" if solid else "ara-tag"
    return f'<span class="{cls}">{text}</span>'


def fact(label: str, value: str) -> str:
    return f'<span class="ara-fact">{label} <b>{value}</b></span>'


# ---------------------------------------------------------------------------
# Application logic (unchanged from the pre-redesign version): building the
# args namespace, calling the shared pipeline, and mapping outcomes/errors
# to a presentation. Only the presentation below has changed.
# ---------------------------------------------------------------------------

def _ask(question: str, sources: list[str], no_cache: bool) -> ResearchResult:
    args = SimpleNamespace(
        question=question,
        sources=",".join(sources) if sources else None,
        no_cache=no_cache,
    )
    return asyncio.run(run_research(args))


async def _fetch_history(limit: int):
    pool = await get_pool()
    try:
        return await list_sessions(pool, limit=limit)
    finally:
        await close_pool()


def render_result(result: ResearchResult) -> None:
    st.markdown('<div class="ara-snote" style="margin-bottom:4px">ANSWER</div>', unsafe_allow_html=True)
    st.write(result.answer.answer)

    if result.answer.citations:
        st.markdown(
            '<div class="ara-snote" style="margin:18px 0 4px">REFERENCES</div>',
            unsafe_allow_html=True,
        )
        for citation in sorted(result.answer.citations, key=lambda c: c.index):
            source = citation.source
            st.markdown(
                '<div class="ara-citerow">'
                + tag(f"[{citation.index}]", solid=True)
                + tag(source.origin)
                + f'<a href="{source.url}">{source.title}</a>'
                + "</div>",
                unsafe_allow_html=True,
            )

    bundle = result.bundle
    if bundle.failed:
        st.warning("Unavailable sources: " + ", ".join(bundle.failed))
    if bundle.empty:
        st.info("Sources with no results: " + ", ".join(bundle.empty))
    for warning in result.warnings:
        st.warning(f"{warning.code}: {warning.message}")

    stats = result.cache_stats
    st.markdown(
        '<div class="ara-facts">'
        + fact("Documents", str(len(bundle.sources)))
        + fact("Providers", str(len(bundle.used)))
        + fact("Time", f"{result.timings.total_ms / 1000:.2f}s")
        + fact("Cache hits", str(stats.hits))
        + "</div>",
        unsafe_allow_html=True,
    )

    caption = f"History: {result.history_status}"
    if result.session_id is not None:
        caption += f" (session #{result.session_id})"
    st.caption(caption)
    if result.history_status == "unknown":
        st.info(
            "History write is unconfirmed; it may have committed. "
            "Do not retry automatically."
        )


def render_error(exc: Exception) -> None:
    # Never surface str(exc) directly: provider/DB errors can embed secrets
    # (API keys, connection strings). Only the exception type is logged,
    # matching researcher.cli.main()'s own discipline.
    log.error("UI request failed (%s)", type(exc).__name__)

    if isinstance(exc, ValidationError):
        st.error(str(exc))
    elif isinstance(exc, (ResearcherError, OrchestrationError, ProviderError)):
        st.error(
            "Research failed. Check provider settings, available sources "
            "and connectivity."
        )
    elif isinstance(exc, SettingsError):
        st.error("Invalid configuration. Check DATABASE_URL and .env settings.")
    elif isinstance(exc, ModuleNotFoundError):
        st.error(f"Missing dependency: {exc.name}. Install the project requirements.")
    elif isinstance(exc, (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError)):
        st.error("Research failed: check database connectivity, tables and network access.")
    else:
        st.error("Unexpected error. Check the application logs.")


def ask_tab() -> None:
    section_header("01", "Ask a question", "Wikipedia + arXiv + web, synthesized with citations")

    question = st.text_input("Question", placeholder="What is photosynthesis?")
    sources = st.multiselect(
        "Sources",
        options=["wiki", "arxiv", "web"],
        default=["wiki", "arxiv", "web"],
        help="Same aliases the CLI accepts: wiki, arxiv, web.",
    )
    no_cache = st.checkbox("Skip cache", value=False)

    if not st.button("Ask", type="primary", disabled=not question.strip()):
        return

    try:
        clean_question = validate_question(question)
    except ValidationError as exc:
        render_error(exc)
        return

    with st.spinner("Researching..."):
        try:
            result = _ask(clean_question, sources, no_cache)
        except Exception as exc:  # noqa: BLE001 - single, logged UI error boundary
            render_error(exc)
            return

    render_result(result)


def history_tab() -> None:
    section_header("02", "History", "Past research sessions")

    limit = st.number_input("Show last N", min_value=1, max_value=100, value=10, step=1)

    if not st.button("Refresh"):
        return

    try:
        clean_limit = validate_limit(int(limit))
        sessions = asyncio.run(_fetch_history(clean_limit))
    except Exception as exc:  # noqa: BLE001 - single, logged UI error boundary
        render_error(exc)
        return

    if not sessions:
        st.info("No previous queries found.")
        return

    for session in sessions:
        with st.expander(f"#{session.id} · {session.created_at} -- {session.question}"):
            st.write(session.answer)


def main() -> None:
    inject_theme()

    st.markdown(
        '<div class="ara-eyebrow"><span class="dot"></span> AI-ENG-110 &middot; TOPIC 4</div>'
        '<div class="ara-hero"><h1>Async <em>Research</em> Assistant</h1>'
        '<p class="ara-lede">Wikipedia, arXiv and web search run concurrently; '
        "one cited answer comes back even if a source fails.</p></div>",
        unsafe_allow_html=True,
    )

    tab_ask, tab_history = st.tabs(["Ask", "History"])
    with tab_ask:
        ask_tab()
    with tab_history:
        history_tab()


main()
