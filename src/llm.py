"""Ollama LLM client with a strict, replaceable interface."""

from __future__ import annotations

import logging
from typing import Iterable

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
        """Run a chat completion with the given system prompt and messages."""
        payload = [{"role": "system", "content": system}, *messages]
        try:
            response = self._client.chat(model=self.model, messages=payload)
        except Exception as exc:
            raise LLMError(
                f"Failed to call Ollama LLM '{self.model}' at {self.base_url}. "
                f"Make sure Ollama is running and the model is pulled. Error: {exc}"
            ) from exc

        if isinstance(response, dict):
            message = response.get("message") or {}
            content = message.get("content", "")
        else:
            message = getattr(response, "message", None)
            content = getattr(message, "content", "") if message else ""

        if not content:
            raise LLMError("Ollama returned an empty response.")
        return content.strip()

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