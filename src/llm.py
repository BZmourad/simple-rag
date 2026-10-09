"""LLM factory: switch between llama.cpp, OpenRouter, and Gemini via .env."""

from __future__ import annotations

import logging
from typing import Iterator

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def create_llm(cfg):
    """Return the configured LLM provider.

    Switch via LLM_PROVIDER in .env:
      - "llamacpp"  → local llama.cpp server
      - "openrouter" → OpenRouter cloud API
      - "gemini"     → Google Gemini cloud API
    """
    provider = cfg.llm_provider.lower()

    if provider == "llamacpp":
        from src.llm_llamacpp import LlamaCppLLM
        return LlamaCppLLM(
            base_url=cfg.llamacpp_base_url,
            model=cfg.llamacpp_model,
            timeout=cfg.llamacpp_timeout,
        )

    if provider == "openrouter":
        from src.llm_cloud import OpenRouterLLM
        return OpenRouterLLM(
            api_key=cfg.openrouter_api_key,
            model=cfg.openrouter_model,
            base_url=cfg.openrouter_base_url,
        )

    if provider == "gemini":
        from src.llm_cloud import GeminiLLM
        return GeminiLLM(api_key=cfg.gemini_api_key, model=cfg.gemini_model)

    raise LLMError(
        f"Unknown LLM_PROVIDER='{provider}'. "
        f"Valid options: llamacpp, openrouter, gemini"
    )