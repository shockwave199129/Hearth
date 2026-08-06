"""Local API token check — see desktop/src-tauri/src/main.rs and SEC-3."""

from __future__ import annotations

import secrets

from fastapi import Request
from fastapi.responses import JSONResponse

from app import config


def token_matches(supplied: str | None) -> bool:
    # Compared as bytes: secrets.compare_digest raises TypeError on a str
    # containing non-ASCII, and header values are attacker-controlled, so
    # the str form turns a junk header into a 500 instead of a 401.
    if supplied is None:
        return False
    return secrets.compare_digest(supplied.encode("utf-8"), config.API_TOKEN.encode("utf-8"))


async def require_api_token(request: Request, call_next):
    # Registered before CORSMiddleware so CORS ends up the outer layer: a
    # rejected request still needs the CORS headers, or the webview reports an
    # opaque network failure instead of the 401. Preflight carries no custom
    # headers by definition, so OPTIONS is checked by the CORS layer alone.
    # Reads config.API_TOKEN at call time so tests can monkeypatch it.
    if not config.API_TOKEN or request.method == "OPTIONS":
        return await call_next(request)
    if not token_matches(request.headers.get(config.API_TOKEN_HEADER)):
        return JSONResponse({"detail": "missing or invalid local API token"}, status_code=401)
    return await call_next(request)
