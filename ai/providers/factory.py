"""Factory functions that select the active provider based on environment variables.

Environment variables consulted
-------------------------------
LLM_PROVIDER       : "anthropic" | "openai" | "gemini"  (default: "anthropic")
LLM_MODEL          : provider-specific model id          (optional)
EMBEDDING_PROVIDER : "openai" | "gemini"                 (default: "openai")
EMBEDDING_MODEL    : provider-specific model id          (optional)

Plus the corresponding API_KEY variables. See each provider module for details.
"""

from __future__ import annotations

import os

from ai.providers.base import (
    VLMProvider,
    LLMProvider,
    EmbeddingProvider,
    ProviderError,
)


def get_llm() -> LLMProvider:
    """Return the configured text-only LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()
    if provider == "anthropic":
        from ai.providers.anthropic import AnthropicLLM
        return AnthropicLLM()
    if provider == "openai":
        from ai.providers.openai import OpenAILLM
        return OpenAILLM()
    if provider in ("google", "gemini"):
        from ai.providers.google import GeminiLLM
        return GeminiLLM()
    raise ProviderError(
        f"Unknown LLM_PROVIDER={provider!r}. Expected anthropic | openai | gemini."
    )


def get_vlm() -> VLMProvider:
    """Return the configured VLM provider."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()
    if provider == "anthropic":
        from ai.providers.anthropic import AnthropicVLM
        return AnthropicVLM()
    if provider == "openai":
        from ai.providers.openai import OpenAIVLM
        return OpenAIVLM()
    if provider in ("google", "gemini"):
        from ai.providers.google import GeminiVLM
        return GeminiVLM()
    raise ProviderError(
        f"Unknown LLM_PROVIDER={provider!r}. Expected anthropic | openai | gemini."
    )


def get_embedder() -> EmbeddingProvider:
    """Return the configured embedding provider."""
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower().strip()
    if provider == "openai":
        from ai.providers.openai import OpenAIEmbedding
        return OpenAIEmbedding()
    if provider in ("google", "gemini"):
        from ai.providers.google import GeminiEmbedding
        return GeminiEmbedding()
    raise ProviderError(
        f"Unknown EMBEDDING_PROVIDER={provider!r}. Expected openai | gemini. "
        "(Anthropic does not currently offer a first-party embedding endpoint.)"
    )
