"""Covers researcher/storage/history.py's save_session/list_sessions, which
were only exercised indirectly (via SessionHistoryAdapter) before."""
import pytest

from researcher.models import ResearchSession
from researcher.storage.history import list_sessions, save_session


class _FakePool:
    def __init__(self, *, fetchval_return=None, fetch_return=None):
        self._fetchval_return = fetchval_return
        self._fetch_return = fetch_return or []
        self.fetchval_calls: list[tuple] = []
        self.fetch_calls: list[tuple] = []

    async def fetchval(self, query, *args):
        self.fetchval_calls.append(args)
        return self._fetchval_return

    async def fetch(self, query, *args):
        self.fetch_calls.append(args)
        return self._fetch_return


@pytest.mark.asyncio
async def test_save_session_returns_new_id_and_passes_fields():
    pool = _FakePool(fetchval_return=7)
    session = ResearchSession(
        question="Q?",
        answer="A.",
        sources_used=["wikipedia"],
        sources_failed=["arxiv"],
        duration_ms=100,
    )

    assert await save_session(pool, session) == 7
    assert pool.fetchval_calls == [
        ("Q?", "A.", ["wikipedia"], ["arxiv"], 100)
    ]


@pytest.mark.asyncio
async def test_list_sessions_maps_rows_to_research_session():
    row = {
        "id": 1,
        "question": "Q?",
        "answer": "A.",
        "sources_used": ["wikipedia"],
        "sources_failed": [],
        "duration_ms": 50,
        "created_at": None,
    }
    pool = _FakePool(fetch_return=[row])

    sessions = await list_sessions(pool, limit=5)

    assert pool.fetch_calls == [(5,)]
    assert len(sessions) == 1
    assert isinstance(sessions[0], ResearchSession)
    assert sessions[0].question == "Q?"
    assert sessions[0].id == 1


@pytest.mark.asyncio
async def test_list_sessions_empty_returns_empty_list():
    pool = _FakePool(fetch_return=[])

    assert await list_sessions(pool) == []
