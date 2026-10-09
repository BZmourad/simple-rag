"""llama.cpp server client (OpenAI-compatible API)."""

from __future__ import annotations

import json
import logging
from typing import Iterable, Iterator

import httpx

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LlamaCppLLM:
    """Client for llama.cpp's llama-server.

    llama-server exposes an OpenAI-compatible /v1/chat/completions endpoint,
    so the same request/response shape works across providers [citation:1][citation:8].
    """

    def __init__(self, base_url: str, model: str = "local", timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer not-needed",  # llama.cpp ignores it
        }

    def generate(self, system: str, messages: Iterable[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": 0.2,
            "stream": False,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self._headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"llama.cpp error {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except Exception as exc:
            raise LLMError(f"llama.cpp request failed: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected llama.cpp response: {data}") from exc

    async def agenerate(self, system: str, messages: Iterable[dict]) -> str:
        """Async variant using httpx.AsyncClient."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": 0.2,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self._headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as exc:
                raise LLMError(
                    f"llama.cpp error {exc.response.status_code}: {exc.response.text[:200]}"
                ) from exc
            except Exception as exc:
                raise LLMError(f"llama.cpp request failed: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected llama.cpp response: {data}") from exc

    def stream_generate(self, system: str, messages: Iterable[dict]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": True,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    headers=self._headers,
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        try:
                            data = json.loads(chunk)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except Exception as exc:
            raise LLMError(f"llama.cpp stream failed: {exc}") from exc

    async def astream_generate(self, system: str, messages: Iterable[dict]):
        """Async streaming variant."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    headers=self._headers,
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        try:
                            data = json.loads(chunk)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
            except Exception as exc:
                raise LLMError(f"llama.cpp stream failed: {exc}") from exc

    def health_check(self) -> None:
        try:
            self.generate("ping", [{"role": "user", "content": "hi"}])
        except LLMError:
            raise