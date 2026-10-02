"""API key auth + Redis-based rate limiting for FastAPI."""

from __future__ import annotations

import logging
import time
from typing import Optional

import redis
from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader, APIKeyQuery, HTTPBearer

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)
_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
_query_scheme = APIKeyQuery(name="api_key", auto_error=False)


def _extract_key(request: Request, bearer, header_key: Optional[str], query_key: Optional[str]) -> Optional[str]:
    if bearer and bearer.credentials:
        return bearer.credentials
    if header_key:
        return header_key
    if query_key:
        return query_key
    return None


class AuthManager:
    def __init__(self, enabled: bool, valid_keys: list[str], rate_limit_enabled: bool, redis_url: str,
                 per_minute: int, per_day: int) -> None:
        self.enabled = enabled
        self.valid_keys = set(valid_keys)
        self.rate_limit_enabled = rate_limit_enabled
        self.per_minute = per_minute
        self.per_day = per_day
        self._redis: redis.Redis | None = None
        if rate_limit_enabled:
            try:
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("Rate limiter connected to Redis at %s", redis_url)
            except Exception as exc:
                logger.warning("Rate limiter Redis unavailable (%s); rate limiting disabled", exc)
                self._redis = None

    def _check_rate(self, key: str) -> None:
        """Sliding-window rate limit using Redis sorted sets."""
        if not self._redis:
            return
        now = time.time()
        minute_key = f"rl:min:{key}"
        day_key = f"rl:day:{key}"

        pipe = self._redis.pipeline()
        # Minute window
        pipe.zremrangebyscore(minute_key, 0, now - 60)
        pipe.zadd(minute_key, {str(now): now})
        pipe.zcard(minute_key)
        pipe.expire(minute_key, 120)
        # Day window
        pipe.zremrangebyscore(day_key, 0, now - 86400)
        pipe.zadd(day_key, {str(now): now})
        pipe.zcard(day_key)
        pipe.expire(day_key, 90000)
        results = pipe.execute()

        minute_count = results[2]
        day_count = results[5]
        if minute_count > self.per_minute:
            raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {self.per_minute}/min")
        if day_count > self.per_day:
            raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {self.per_day}/day")

    async def require_auth(
        self,
        request: Request,
        bearer=Depends(_bearer_scheme),
        header_key: Optional[str] = Depends(_header_scheme),
        query_key: Optional[str] = Depends(_query_scheme),
    ) -> dict:
        if not self.enabled:
            return {"key": "anonymous", "plan": "unlimited"}

        raw_key = _extract_key(request, bearer, header_key, query_key)
        if not raw_key:
            raise HTTPException(status_code=401, detail="Missing API key. Use Authorization: Bearer <key>, X-API-Key header, or api_key query param.")
        if raw_key not in self.valid_keys:
            raise HTTPException(status_code=401, detail="Invalid API key.")

        self._check_rate(raw_key)
        request.state.api_key = raw_key
        return {"key": raw_key, "plan": "default"}