"""OpenAI provider adapters (GPT-4o family for VLM/LLM, text-embedding-3-* for embeddings)."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import numpy as np

from ai.providers.base import VLMProvider, LLMProvider, EmbeddingProvider, ProviderError


def _make_openai_client(api_key: str | None):
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        raise ProviderError("OPENAI_API_KEY (or LLM_API_KEY) is not set.")
    try:
        import openai  # type: ignore
    except ImportError as e:
        raise ProviderError(
            "The `openai` package is required. Install with `pip install openai`."
        ) from e
    return openai.OpenAI(api_key=key)


class OpenAILLM(LLMProvider):
    """OpenAI Chat Completions (text only)."""

    def __init__(self, model: str | None = None, *, api_key: str | None = None) -> None:
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self._client = _make_openai_client(api_key)

    def complete(
        self,
        prompt: str,
        *,
        json_schema: dict | None = None,
        max_tokens: int = 1024,
    ) -> str:
        full_prompt = prompt
        if json_schema is not None:
            full_prompt = (
                prompt
                + "\n\nReturn ONLY valid JSON matching this schema "
                "(no prose, no markdown fences):\n"
                + json.dumps(json_schema, indent=2)
            )
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:  # pragma: no cover - network path
            raise ProviderError(f"OpenAI call failed: {e}") from e
        return (resp.choices[0].message.content or "").strip()


class OpenAIVLM(VLMProvider):
    """GPT-4o via the OpenAI Chat Completions API."""

    def __init__(self, model: str | None = None, *, api_key: str | None = None) -> None:
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self._client = _make_openai_client(api_key)

    def describe(
        self,
        image_path: str,
        prompt: str,
        *,
        json_schema: dict | None = None,
    ) -> str:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(image_path)
        media_type = _guess_media_type(path)
        b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        data_url = f"data:{media_type};base64,{b64}"

        full_prompt = prompt
        if json_schema is not None:
            full_prompt = (
                prompt
                + "\n\nReturn ONLY valid JSON matching this schema "
                "(no prose, no markdown fences):\n"
                + json.dumps(json_schema, indent=2)
            )

        kwargs: dict = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": 1024,
        }
        # Use OpenAI's structured-output mode when a schema is provided.
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:  # pragma: no cover - network path
            raise ProviderError(f"OpenAI call failed: {e}") from e
        content = resp.choices[0].message.content or ""
        return content.strip()


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI text-embedding-3-small / -large."""

    def __init__(self, model: str | None = None, *, api_key: str | None = None) -> None:
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self._api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("EMBEDDING_API_KEY")
        )
        if not self._api_key:
            raise ProviderError("OPENAI_API_KEY (or EMBEDDING_API_KEY) is not set.")
        try:
            import openai  # type: ignore
        except ImportError as e:
            raise ProviderError(
                "The `openai` package is required for OpenAIEmbedding."
            ) from e
        self._client = openai.OpenAI(api_key=self._api_key)
        # Known dimensions for OpenAI's standard embedding models.
        self._dim = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }.get(self.model, 1536)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Cannot embed empty string.")
        try:
            resp = self._client.embeddings.create(model=self.model, input=text)
        except Exception as e:  # pragma: no cover - network path
            raise ProviderError(f"OpenAI embedding call failed: {e}") from e
        vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            raise ProviderError("Provider returned a zero vector.")
        return vec / norm


def _guess_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    raise ProviderError(f"Unsupported image type: {suffix}")
