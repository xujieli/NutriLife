"""身份鉴权与安全工具：JWT 签发/校验、登录态依赖注入。

设计要点：
    - JWT 采用 HS256，密钥来自 ``APP_SECRET_KEY``，有效期默认 7 天。
    - 令牌存放在 HttpOnly + SameSite Cookie 中，前端脚本无法读取，从而防 XSS。
    - 提供 ``get_current_user``（必需登录）与 ``get_current_user_optional``
      （可选登录，用于匿名对话）两个 FastAPI 依赖。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_async_session
from app.core.models import User

# JWT 标准异常集合（用于统一降级处理）。
_JWT_ERRORS = (jwt.ExpiredSignatureError, jwt.InvalidTokenError, jwt.PyJWTError)


def _auth_settings():
    return get_settings().auth


def create_access_token(user: User) -> str:
    """为指定用户签发 JWT 令牌，有效期默认 7 天。

    Args:
        user: 已持久化的用户对象。

    Returns:
        str: 签名的 JWT 字符串。
    """
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": str(user.id),
        "phone": user.phone,
        "iat": now,
        "exp": now + timedelta(days=settings.auth.jwt_expire_days),
    }
    return jwt.encode(payload, settings.app.secret_key, algorithm="HS256")


def decode_access_token(token: str) -> dict:
    """校验并解码 JWT 令牌，返回 payload；无效或过期时抛出 ``jwt`` 异常。"""
    settings = get_settings()
    return jwt.decode(token, settings.app.secret_key, algorithms=["HS256"])


def set_auth_cookie(response: Response, token: str) -> None:
    """将 JWT 写入 HttpOnly 安全 Cookie。"""
    auth = _auth_settings()
    response.set_cookie(
        key=auth.cookie_name,
        value=token,
        max_age=auth.jwt_expire_days * 24 * 60 * 60,
        httponly=True,
        secure=auth.cookie_secure,
        samesite=auth.cookie_samesite,
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    """清除登录态 Cookie。"""
    auth = _auth_settings()
    response.delete_cookie(key=auth.cookie_name, path="/")


def _extract_user_id(request: Request) -> str | None:
    """从请求 Cookie 中解析并返回用户 ID；无令牌或令牌非法时返回 None。"""
    token = request.cookies.get(_auth_settings().cookie_name)
    if not token:
        return None
    try:
        payload = decode_access_token(token)
    except _JWT_ERRORS:
        return None
    return payload.get("sub")


async def get_current_user_optional(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> User | None:
    """可选登录依赖：未登录或令牌失效时返回 None（不阻断匿名对话）。"""
    user_id = _extract_user_id(request)
    if user_id is None:
        return None
    try:
        return await session.get(User, uuid.UUID(user_id))
    except (ValueError, TypeError):
        return None


async def get_current_user(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> User:
    """必需登录依赖：未登录或令牌失效时抛出 401。"""
    user = await get_current_user_optional(request, session)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录或登录已过期，请重新登录",
        )
    return user
