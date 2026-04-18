"""
Auth API routes (JWT-based).
"""

from datetime import datetime, timedelta, timezone
import os
import secrets
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel, Field

router = APIRouter()
security = HTTPBearer(auto_error=False)

JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRES_MINUTES = int(os.getenv("JWT_EXPIRES_MINUTES", "60"))
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", f"{ADMIN_USERNAME}@admin.com")
AUTH_COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "access_token")
AUTH_COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true"
AUTH_COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "lax")
AUTH_COOKIE_DOMAIN = os.getenv("AUTH_COOKIE_DOMAIN") or None

class LoginRequest(BaseModel):
    # Accept either 'username' or 'email' — both resolve to the admin account
    username: Optional[str] = Field(default=None, min_length=1, max_length=128)
    email: Optional[str] = Field(default=None, min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


def _create_access_token(subject: str) -> str:
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT secret not configured")
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRES_MINUTES)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_access_token(token: str) -> Dict[str, Any]:
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT secret not configured")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username}
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


def _extract_token(request: Request, credentials: Optional[HTTPAuthorizationCredentials]) -> Optional[str]:
    if credentials and credentials.credentials:
        return credentials.credentials
    cookie_token = request.cookies.get(AUTH_COOKIE_NAME)
    if cookie_token:
        return cookie_token
    return None


def validate_access_token(token: str) -> Dict[str, Any]:
    return _decode_access_token(token)


def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    token = _extract_token(request, credentials)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return _decode_access_token(token)


@router.post("/login")
async def login(request: LoginRequest, response: Response):
    """Login and issue JWT. Accepts username or email."""
    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        raise HTTPException(status_code=500, detail="Auth not configured")
    if not request.username and not request.email:
        raise HTTPException(status_code=422, detail="Provide username or email")

    # Resolve identifier — accept username OR email
    identifier = request.username or request.email or ""
    valid = (
        secrets.compare_digest(identifier, ADMIN_USERNAME) or
        secrets.compare_digest(identifier, ADMIN_EMAIL)
    )
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not secrets.compare_digest(request.password, ADMIN_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = _create_access_token(ADMIN_USERNAME)
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite=AUTH_COOKIE_SAMESITE,
        domain=AUTH_COOKIE_DOMAIN,
        max_age=JWT_EXPIRES_MINUTES * 60,
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return {
        "authenticated": True,
        "user": {
            "username": ADMIN_USERNAME,
            "email": ADMIN_EMAIL,
        },
    }

@router.get("/profile")
async def profile(response: Response, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Return the authenticated user profile."""
    response.headers["Cache-Control"] = "no-store"
    return {
        "authenticated": True,
        "user": {
            "username": current_user["username"],
            "email": ADMIN_EMAIL,
        },
    }


@router.post("/logout")
async def logout(response: Response):
    """Logout and clear the auth cookie."""
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        domain=AUTH_COOKIE_DOMAIN,
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return {"message": "Logged out successfully"}
