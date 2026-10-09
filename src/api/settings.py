"""Settings endpoints: read, update, and test provider connectivity."""

from __future__ import annotations

import logging

import httpx
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from src.settings_store import SettingsStore

logger = logging.getLogger(__name__)


class SettingsUpdate(BaseModel):
    values: dict[str, str]


def register_settings_routes(app: FastAPI, store: SettingsStore, auth) -> None:

    @app.get("/settings")
    async def get_settings(user=Depends(auth.require_auth)):
        """Return all mutable settings (secrets masked)."""
        return {"settings": store.all_mutable()}

    @app.put("/settings")
    async def update_settings(req: SettingsUpdate, user=Depends(auth.require_auth)):
        """Update settings. Persists to .env.runtime and applies to os.environ."""
        applied = store.set_many(req.values)
        return {"applied": list(applied.keys()), "count": len(applied)}

    @app.post("/settings/test")
    async def test_provider(user=Depends(auth.require_auth)):
        """Ping the currently configured LLM provider."""
        provider = store.get("LLM_PROVIDER", "llamacpp").lower()

        try:
            if provider == "llamacpp":
                url = store.get("LLAMACPP_BASE_URL", "http://127.0.0.1:8080")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    r = await client.get(f"{url.rstrip('/')}/health")
                    r.raise_for_status()
                    return {"provider": provider, "status": "ok", "detail": r.json()}

            if provider == "openrouter":
                url = store.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                key = store.get("OPENROUTER_API_KEY", "")
                if not key:
                    return {"provider": provider, "status": "error", "detail": "Missing API key"}
                async with httpx.AsyncClient(timeout=10.0) as client:
                    r = await client.get(
                        f"{url.rstrip('/')}/models",
                        headers={"Authorization": f"Bearer {key}"},
                    )
                    r.raise_for_status()
                    return {"provider": provider, "status": "ok", "detail": "models endpoint reachable"}

            if provider == "gemini":
                key = store.get("GEMINI_API_KEY", "")
                if not key:
                    return {"provider": provider, "status": "error", "detail": "Missing API key"}
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
                async with httpx.AsyncClient(timeout=10.0) as client:
                    r = await client.get(url)
                    r.raise_for_status()
                    return {"provider": provider, "status": "ok", "detail": "models endpoint reachable"}

            return {"provider": provider, "status": "error", "detail": f"Unknown provider: {provider}"}
        except Exception as exc:
            logger.exception("Provider test failed")
            return {"provider": provider, "status": "error", "detail": str(exc)}

    @app.post("/settings/reset")
    async def reset_settings(user=Depends(auth.require_auth)):
        """Clear runtime overrides — falls back to .env values."""
        store.clear()
        return {"status": "cleared"}