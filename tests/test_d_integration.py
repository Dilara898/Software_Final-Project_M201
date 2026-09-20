"""Offline tests for D's CLI wiring using the real C orchestrator."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from ai.schemas import Source, Citation, AnswerWithCitations
from researcher.application import execute_ask, clean
from researcher.cli import main


@pytest.mark.parametrize('argv', [
    ['ask', ''], ['ask', 'ab'], ['ask', 'x' * 501],
    ['history', '--limit', '0'], ['history', '--limit', '101'],
])
def test_invalid_input_without_io(argv, monkeypatch):
    async def unexpected(*args):
        pytest.fail('Invalid input must not open resources')
    monkeypatch.setattr('researcher.cli.show_history', unexpected)
    monkeypatch.setattr('researcher.application.run_ask', unexpected)
    assert main(argv) == 1


@pytest.mark.parametrize('no_cache', [True, False])
@pytest.mark.parametrize('cache_delay', [0, 0.6])
@pytest.mark.asyncio
async def test_ask_subset_cache_and_history(no_cache, cache_delay, capsys):
    class Fetch:
        names = []
        async def fetch(self, name, question, client):
            self.names.append(name)
            return [Source(title=name, url='https://example.org/' + name,
                           snippet='A source excerpt', origin=name)]

    class Synthesis:
        async def synthesize(self, question, sources):
            return AnswerWithCitations(question=question, answer='Answer [1].',
                                       citations=[Citation(index=1, source=sources[0])])

    async def read_cache(*args):
        # Remote connection acquisition can exceed C's standalone 0.5s default.
        await asyncio.sleep(cache_delay)
        return None

    pool = SimpleNamespace(fetchrow=AsyncMock(side_effect=read_cache),
                           execute=AsyncMock(), fetchval=AsyncMock(return_value=7))
    settings = SimpleNamespace(per_source_timeout_seconds=2,
                               cache_ttl_seconds=60, cache_timeout_seconds=3,
                               max_sources_per_query=3)
    args = SimpleNamespace(question='What is photosynthesis?',
                           sources='wiki,arxiv', no_cache=no_cache)
    fetch = Fetch()
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: pytest.fail('Unexpected HTTP request')
    )) as client:
        assert await execute_ask(args, settings, pool, client, fetch, Synthesis()) == 0
    assert set(fetch.names) == {'wikipedia', 'arxiv'}
    assert pool.fetchrow.await_count == (0 if no_cache else 2)
    assert pool.execute.await_count == (0 if no_cache else 2)
    pool.fetchval.assert_awaited_once()
    out = capsys.readouterr().out
    assert '[1] (wikipedia)' in out and 'History: saved' in out
    assert '0 read errors' in out
    assert '2 bypasses' in out if no_cache else '2 misses' in out


def test_terminal_controls_removed():
    assert clean('\x1b[31mSalam\x1b[0m\x00\nə') == 'Salam\nə'
