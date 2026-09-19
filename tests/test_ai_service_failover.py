"""Covers researcher/services/ai_service.py's provider-selection branches
(_llm_factory_for, default_llm_chain) and the failover path where a
provider fails to *construct* (not just fails to answer)."""
import pytest

from ai.providers.base import LLMProvider, ProviderError
from researcher.services.ai_service import (
    AISynthesisService,
    _llm_factory_for,
    default_llm_chain,
)


def test_llm_factory_for_unknown_name_raises():
    with pytest.raises(ProviderError):
        _llm_factory_for("bogus")


@pytest.mark.parametrize("name", ["anthropic", "openai", "google", "gemini"])
def test_llm_factory_for_known_names_return_callables(name):
    factory = _llm_factory_for(name)
    assert callable(factory)


def test_default_llm_chain_skips_blank_and_duplicate_names(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    chain = default_llm_chain(["", "anthropic", "openai", "openai"])

    # primary (anthropic, via get_llm) + exactly one openai factory
    assert len(chain) == 2


def test_default_llm_chain_maps_gemini_alias_to_google_for_dedup(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    # "google" is gemini's canonical name, already the primary -> skipped
    chain = default_llm_chain(["google"])

    assert len(chain) == 1


@pytest.mark.asyncio
async def test_synthesize_skips_factory_that_fails_to_construct(
    sample_source, fake_llm
):
    def bad_factory():
        raise ProviderError("no api key configured")

    def good_factory() -> LLMProvider:
        return fake_llm

    service = AISynthesisService(
        llm_factories=[bad_factory, good_factory], max_attempts=1
    )

    result = await service.synthesize("Q?", [sample_source])

    assert result.answer == fake_llm.response
