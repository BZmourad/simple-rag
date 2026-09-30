"""Ollama LLM client with streaming support."""

from __future__ import annotations

import logging
from typing import Iterable, Iterator

import ollama

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url
        self._client = ollama.Client(host=base_url)

    def generate(self, system: str, messages: Iterable[dict]) -> str:
        payload = [{"role": "system", "content": system}, *messages]
        try:
            response = self._client.chat(model=self.model, messages=payload)
        except Exception as exc:
            raise LLMError(
                f"Failed to call Ollama LLM '{self.model}'. "
                f"Ensure Ollama is running and the model is pulled. Error: {exc}"
            ) from exc

        if isinstance(response, dict):
            content = (response.get("message") or {}).get("content", "")
        else:
            content = getattr(getattr(response, "message", None), "content", "") or ""
        if not content:
            raise LLMError("Ollama returned an empty response.")
        return content.strip()

    def stream_generate(self, system: str, messages: Iterable[dict]) -> Iterator[str]:
        """Yield tokens as they are produced."""
        payload = [{"role": "system", "content": system}, *messages]
        try:
            stream = self._client.chat(model=self.model, messages=payload, stream=True)
        except Exception as exc:
            raise LLMError(
                f"Failed to stream from Ollama LLM '{self.model}'. "
                f"Ensure Ollama is running. Error: {exc}"
            ) from exc

        for chunk in stream:
            if isinstance(chunk, dict):
                content = (chunk.get("message") or {}).get("content", "")
            else:
                content = getattr(getattr(chunk, "message", None), "content", "") or ""
            if content:
                yield content

    def health_check(self) -> None:
        try:
            self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                options={"num_predict": 1},
            )
        except Exception as exc:
            raise LLMError(
                f"LLM model '{self.model}' is not available. "
                f"Run: ollama pull {self.model}. Error: {exc}"
            ) from exc