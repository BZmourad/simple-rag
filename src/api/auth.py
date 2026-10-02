"""JWT + API-key auth with Redis-based rate limiting."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
import redis
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, APIKeyQuery, HTTPBearer
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)
_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
_query_scheme = APIKeyQuery(name="api_key", auto_error=False)

_password_hash = PasswordHash.recommended()


class AuthManager:
    def __init__(
        self,
        mode: str,
        valid_keys: list[str],
        jwt_secret: str,
        jwt_algorithm: str,
        access_expire_minutes: int,
        refresh_expire_days: int,
        rate_limit_enabled: bool,
        redis_url: str,
        per_minute: int,
        per_day: int,
    ) -> None:
        self.mode = mode
        self.valid_keys = set(valid_keys)
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.access_expire_minutes = access_expire_minutes
        self.refresh_expire_days = refresh_expire_days
        self.rate_limit_enabled = rate_limit_enabled
        self.per_minute = per_minute
        self.per_day = per_day
        self._redis: redis.Redis | None = None
        if rate_limit_enabled:
            try:
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("Rate limiter connected to Redis")
            except Exception as exc:
                logger.warning("Rate limiter Redis unavailable: %s", exc)
                self._redis = None

    # ------------------------------------------------------------------ #
    # JWT
    # ------------------------------------------------------------------ #
    def create_access_token(self, subject: str, scopes: list[str] | None = None) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": subject,
            "iat": now,
            "exp": now + timedelta(minutes=self.access_expire_minutes),
            "scopes": scopes or ["ask", "stats"],
            "type": "access",
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def create_refresh_token(self, subject: str) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": subject,
            "iat": now,
            "exp": now + timedelta(days=self.refresh_expire_days),
            "type": "refresh",
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def decode_token(self, token: str) -> dict:
        try:
            return jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
        except InvalidTokenError as exc:
            raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Password hashing
    # ------------------------------------------------------------------ #
    @staticmethod
    def hash_password(password: str) -> str:
        return _password_hash.hash(password)

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        return _password_hash.verify(plain, hashed)

    # ------------------------------------------------------------------ #
    # Rate limiting
    # ------------------------------------------------------------------ #
    def _check_rate(self, key: str) -> None:
        if not self._redis:
            return
        import time
        now = time.time()
        minute_key = f"rl:min:{key}"
        day_key = f"rl:day:{key}"
        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(minute_key, 0, now - 60)
        pipe.zadd(minute_key, {str(now): now})
        pipe.zcard(minute_key)
        pipe.expire(minute_key, 120)
        pipe.zremrangebyscore(day_key, 0, now - 86400)
        pipe.zadd(day_key, {str(now): now})
        pipe.zcard(day_key)
        pipe.expire(day_key, 90000)
        results = pipe.execute()
        if results[2] > self.per_minute:
            raise HTTPException(status_code=429, detail=f"Rate limit: {self.per_minute}/min")
        if results[5] > self.per_day:
            raise HTTPException(status_code=429, detail=f"Rate limit: {self.per_day}/day")

    # ------------------------------------------------------------------ #
    # FastAPI dependency
    # ------------------------------------------------------------------ #
    async def require_auth(
        self,
        request: Request,
        bearer=Depends(_bearer_scheme),
        header_key: Optional[str] = Depends(_header_scheme),
        query_key: Optional[str] = Depends(_query_scheme),
    ) -> dict:
        if self.mode == "jwt":
            if not bearer or not bearer.credentials:
                raise HTTPException(status_code=401, detail="Missing JWT. Use Authorization: Bearer <token>")
            payload = self.decode_token(bearer.credentials)
            if payload.get("type") != "access":
                raise HTTPException(status_code=401, detail="Not an access token")
            identity = payload.get("sub", "unknown")
            self._check_rate(identity)
            request.state.identity = identity
            return {"subject": identity, "scopes": payload.get("scopes", [])}

        # Legacy API-key mode
        raw = (bearer.credentials if bearer and bearer.credentials else None) or header_key or query_key
        if not raw:
            raise HTTPException(status_code=401, detail="Missing API key")
        if raw not in self.valid_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        self._check_rate(raw)
        request.state.identity = raw
        return {"subject": raw, "scopes": ["ask", "stats"]}