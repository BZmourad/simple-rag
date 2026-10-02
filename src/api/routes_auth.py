"""Auth endpoints: /auth/login and /auth/refresh."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


def create_auth_router(auth) -> APIRouter:
    router = APIRouter(prefix="/auth", tags=["auth"])

    @router.post("/login", response_model=TokenResponse)
    async def login(form: OAuth2PasswordRequestForm = Depends()):
        admin_username = os.getenv("ADMIN_USERNAME", "admin")
        admin_hash = os.getenv("ADMIN_PASSWORD_HASH", "")

        if not admin_hash:
            raise HTTPException(
                status_code=503,
                detail="ADMIN_PASSWORD_HASH not configured. Generate one with: "
                       "python -c \"from pwdlib import PasswordHash; print(PasswordHash.recommended().hash('yourpassword'))\"",
            )

        if form.username != admin_username or not auth.verify_password(form.password, admin_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access = auth.create_access_token(subject=form.username, scopes=["ask", "stats", "ingest"])
        refresh = auth.create_refresh_token(subject=form.username)
        return TokenResponse(
            access_token=access,
            refresh_token=refresh,
            expires_in=auth.access_expire_minutes * 60,
        )

    @router.post("/refresh", response_model=TokenResponse)
    async def refresh(req: RefreshRequest):
        payload = auth.decode_token(req.refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Not a refresh token")
        subject = payload.get("sub", "unknown")
        access = auth.create_access_token(subject=subject, scopes=["ask", "stats", "ingest"])
        new_refresh = auth.create_refresh_token(subject=subject)
        return TokenResponse(
            access_token=access,
            refresh_token=new_refresh,
            expires_in=auth.access_expire_minutes * 60,
        )

    return router