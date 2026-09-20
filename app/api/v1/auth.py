"""身份鉴权路由：手机号 + 验证码登录。

验证码策略：默认取手机号后 6 位（``settings.auth.code_length``），
本地自托管场景无需接入短信服务，便于开发与验收。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_async_session
from app.core.models import User
from app.core.security import (
    clear_auth_cookie,
    create_access_token,
    get_current_user,
    set_auth_cookie,
)
from app.schemas.user import LoginRequest, LoginResponse, SendCodeRequest, UserResponse

router = APIRouter(prefix="/auth", tags=["Auth"])

DbSession = Annotated[AsyncSession, Depends(get_async_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]


def _to_user_response(user: User) -> UserResponse:
    """ORM 用户对象 → API 响应模型。"""
    return UserResponse(
        id=str(user.id),
        phone=user.phone,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )


@router.post("/send-code")
async def send_code(payload: SendCodeRequest) -> dict[str, str]:
    """发送登录验证码（演示实现：验证码为手机号后 6 位）。"""
    code_length = get_settings().auth.code_length
    return {
        "message": "验证码已发送",
        "hint": f"验证码默认为手机号后 {code_length} 位",
    }


@router.post("/login", response_model=LoginResponse)
async def login(
    payload: LoginRequest,
    response: Response,
    session: DbSession,
) -> LoginResponse:
    """手机号 + 验证码登录，成功后签发 7 天 JWT 并写入 HttpOnly Cookie。"""
    settings = get_settings()
    expected_code = payload.phone[-settings.auth.code_length :]

    # 验证码比对（非空校验已由 Pydantic min_length=1 保证）
    if payload.code != expected_code:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="验证码错误，请输入手机号后 6 位",
        )

    try:
        result = await session.execute(
            select(User).where(User.phone == payload.phone)
        )
        user = result.scalar_one_or_none()

        now = datetime.now(UTC)
        if user is None:
            # 首次登录：自动注册
            user = User(phone=payload.phone, last_login_at=now)
            session.add(user)
            await session.flush()
            logger.info("新用户注册: phone={}", payload.phone)
        else:
            user.last_login_at = now

        await session.commit()
        await session.refresh(user)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        await session.rollback()
        logger.error("登录写入用户失败: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="用户数据库暂不可用",
        ) from exc

    token = create_access_token(user)
    set_auth_cookie(response, token)
    return LoginResponse(user=_to_user_response(user))


@router.get("/me", response_model=UserResponse)
async def me(user: CurrentUser) -> UserResponse:
    """返回当前登录用户信息。"""
    return _to_user_response(user)


@router.post("/logout")
async def logout(response: Response) -> dict[str, str]:
    """退出登录，清除鉴权 Cookie。"""
    clear_auth_cookie(response)
    return {"message": "已退出登录"}
