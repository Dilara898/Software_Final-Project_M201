"""Google Gemini provider adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ai.providers.base import VLMProvider, LLMProvider, EmbeddingProvider, ProviderError


def _make_gemini_client(api_key: str | None):
    key = (
        api_key
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("LLM_API_KEY")
    )
    if not key:
        raise ProviderError("GOOGLE_API_KEY (or LLM_API_KEY) is not set.")
    try:
        from google import genai  # type: ignore
    except ImportError as e:
        raise ProviderError(
            "The `google-genai` package is required. "
            "Install with `pip install google-genai`."
        ) from e
    return genai.Client(api_key=key)


class GeminiLLM(LLMProvider):
    """Gemini text-only completion."""

    def __init__(self, model: str | None = None, *, api_key: str | None = None) -> None:
        self.model = model or os.getenv("LLM_MODEL", "gemini-2.0-flash")
        self._client = _make_gemini_client(api_key)

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
        try:
            from google.genai import types  # type: ignore
            resp = self._client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json" if json_schema else None,
                    max_output_tokens=max_tokens,
                ),
            )
        except Exception as e:  # pragma: no cover
            raise ProviderError(f"Gemini call failed: {e}") from e
        return (resp.text or "").strip()


class GeminiVLM(VLMProvider):
    """Gemini 2.x via the google-genai SDK."""

    def __init__(self, model: str | None = None, *, api_key: str | None = None) -> None:
        self.model = model or os.getenv("LLM_MODEL", "gemini-2.0-flash")
        self._client = _make_gemini_client(api_key)

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
        full_prompt = prompt
        if json_schema is not None:
            full_prompt = (
                prompt
                + "\n\nReturn ONLY valid JSON matching this schema "
                "(no prose, no markdown fences):\n"
                + json.dumps(json_schema, indent=2)
            )

        try:
            from google.genai import types  # type: ignore

            uploaded = self._client.files.upload(file=str(path))
            resp = self._client.models.generate_content(
                model=self.model,
                contents=[uploaded, full_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json" if json_schema else None,
                ),
            )
        except Exception as e:  # pragma: no cover - network path
            raise ProviderError(f"Gemini call failed: {e}") from e
        return (resp.text or "").strip()


class GeminiEmbedding(EmbeddingProvider):
    """Gemini embedding model."""

    def __init__(self, model: str | None = None, *, api_key: str | None = None) -> None:
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-004")
        # We accept EMBEDDING_API_KEY as an additional fallback name.
        if api_key is None and os.getenv("EMBEDDING_API_KEY"):
            api_key = os.getenv("EMBEDDING_API_KEY")
        self._client = _make_gemini_client(api_key)
        self._dim = 768  # text-embedding-004 default

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Cannot embed empty string.")
        try:
            resp = self._client.models.embed_content(
                model=self.model,
                contents=text,
            )
        except Exception as e:  # pragma: no cover
            raise ProviderError(f"Gemini embedding call failed: {e}") from e
        # google-genai returns an Embedding object with .values
        vec = np.asarray(resp.embeddings[0].values, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            raise ProviderError("Provider returned a zero vector.")
        return vec / norm
