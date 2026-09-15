"""Anthropic Claude provider adapters.

Uses the Anthropic Messages API for vision via base64 image upload.
Anthropic does not offer a first-party embedding endpoint, so the
embedding adapter is intentionally absent from this module — set
EMBEDDING_PROVIDER=openai (or another provider) when LLM_PROVIDER=anthropic.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from ai.providers.base import VLMProvider, LLMProvider, ProviderError


class _AnthropicMixin:
    """Shared client setup for Anthropic VLM and LLM."""

    def _setup(self, model: str | None, api_key: str | None) -> None:
        self.model = model or os.getenv("LLM_MODEL", "claude-sonnet-4-6")
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY")
        if not self._api_key:
            raise ProviderError(
                "ANTHROPIC_API_KEY (or LLM_API_KEY) is not set. "
                "Either export it or switch LLM_PROVIDER."
            )
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise ProviderError(
                "The `anthropic` package is required. "
                "Install it with `pip install anthropic`."
            ) from e
        self._client = anthropic.Anthropic(api_key=self._api_key)


class AnthropicLLM(_AnthropicMixin, LLMProvider):
    """Claude (text-only) via the Messages API."""

    def __init__(self, model: str | None = None, *, api_key: str | None = None) -> None:
        self._setup(model, api_key)

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
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": full_prompt}],
            )
        except Exception as e:  # pragma: no cover - network path
            raise ProviderError(f"Anthropic call failed: {e}") from e
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()


class AnthropicVLM(_AnthropicMixin, VLMProvider):
    """Claude (vision) via the Messages API."""

    def __init__(self, model: str | None = None, *, api_key: str | None = None) -> None:
        self._setup(model, api_key)

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

        full_prompt = prompt
        if json_schema is not None:
            full_prompt = (
                prompt
                + "\n\nReturn ONLY valid JSON matching this schema "
                "(no prose, no markdown fences):\n"
                + json.dumps(json_schema, indent=2)
            )

        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": full_prompt},
                        ],
                    }
                ],
            )
        except Exception as e:  # pragma: no cover - network path
            raise ProviderError(f"Anthropic call failed: {e}") from e

        # Concatenate any text blocks into a single string.
        parts: list[str] = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts).strip()


def _guess_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    raise ProviderError(f"Unsupported image type: {suffix}")
