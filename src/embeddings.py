"""Embedding provider backed by Ollama."""

from __future__ import annotations

import logging

import ollama

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


class OllamaEmbedder:
    """Wraps Ollama embeddings with a small, replaceable interface."""

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url
        self._client = ollama.Client(host=base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns one vector per text."""
        if not texts:
            return []
        try:
            response = self._client.embed(model=self.model, input=texts)
        except Exception as exc:  # ollama raises various errors
            raise EmbeddingError(
                f"Failed to generate embeddings with model '{self.model}'. "
                f"Is Ollama running at {self.base_url} and the model pulled? Error: {exc}"
            ) from exc

        # ollama-python returns {"embeddings": [[...], ...]}
        embeddings = response.get("embeddings") if isinstance(response, dict) else getattr(response, "embeddings", None)
        if not embeddings:
            raise EmbeddingError("Ollama returned an empty embedding response.")
        return [list(map(float, vec)) for vec in embeddings]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def health_check(self) -> None:
        """Raise EmbeddingError if the model is not reachable."""
        try:
            self._client.embed(model=self.model, input=["ping"])
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding model '{self.model}' is not available. "
                f"Run: ollama pull {self.model}. Error: {exc}"
            ) from exc