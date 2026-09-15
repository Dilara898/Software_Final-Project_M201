"""Abstract base classes for provider-agnostic AI calls.

These bases define the contract every concrete provider must implement.
They are intentionally minimal: no retries, no caching, no logging. Those are
SE-layer concerns and belong to the student's wrapper code.
"""

from __future__ import annotations

import abc
import numpy as np


class LLMProvider(abc.ABC):
    """Contract for text-only language model providers.

    Used by topics that summarize / categorize / synthesize text without
    images (Topics 3 and 4).
    """

    @abc.abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        json_schema: dict | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Return the model's response as plain text (or JSON-encoded text)."""
        raise NotImplementedError


class VLMProvider(abc.ABC):
    """Contract for vision-language model providers.

    A VLM provider takes an image (path or bytes) plus an optional textual
    prompt and returns a JSON string that conforms to a caller-supplied schema.
    """

    @abc.abstractmethod
    def describe(
        self,
        image_path: str,
        prompt: str,
        *,
        json_schema: dict | None = None,
    ) -> str:
        """Return the model's response as a JSON-encoded string.

        Parameters
        ----------
        image_path : str
            Path to a JPEG or PNG file on disk.
        prompt : str
            Instructional text shown to the model alongside the image.
        json_schema : dict | None
            Optional JSON Schema the response must conform to. The provider
            should attempt to use the model's structured-output / tool-calling
            feature if it has one; otherwise it must include the schema in the
            prompt and rely on the caller to validate the output.
        """
        raise NotImplementedError


class EmbeddingProvider(abc.ABC):
    """Contract for text-embedding providers."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the vectors this provider returns."""
        raise NotImplementedError

    @abc.abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D float32 numpy array for `text`.

        The returned vector must be **unit-normalized** so that downstream
        cosine similarity is just a dot product.
        """
        raise NotImplementedError


class ProviderError(RuntimeError):
    """Raised when a provider call fails for any reason.

    The SE layer should treat this as a transient/unknown failure and decide
    whether to retry based on its own policy.
    """
