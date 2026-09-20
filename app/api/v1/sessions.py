"""会话历史路由：分页查询会话列表、创建会话、查看会话消息。"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from loguru import logger
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.models import Message, SessionMetadata, User
from app.core.security import get_current_user, get_current_user_optional
from app.schemas.chat import SessionCreateRequest, SessionResponse
from app.schemas.session import (
    BatchDeleteRequest,
    MessageResponse,
    SessionListResponse,
)

router = APIRouter(prefix="/sessions", tags=["Sessions"])

DbSession = Annotated[AsyncSession, Depends(get_async_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[User | None, Depends(get_current_user_optional)]

DEFAULT_PAGE_SIZE = 20


def _session_response(row: SessionMetadata) -> SessionResponse:
    """ORM 会话行 → API 响应。"""
    return SessionResponse(
        session_id=row.session_id,
        title=row.title,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    session: DbSession,
    user: CurrentUser,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = DEFAULT_PAGE_SIZE,
) -> SessionListResponse:
    """分页返回当前用户的历史会话，按最近更新时间倒序。"""
    try:
        total = (
            await session.execute(
                select(func.count())
                .select_from(SessionMetadata)
                .where(SessionMetadata.user_id == user.id)
            )
        ).scalar_one()

        result = await session.execute(
            select(SessionMetadata)
            .where(SessionMetadata.user_id == user.id)
            .order_by(SessionMetadata.updated_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        rows = list(result.scalars().all())
    except Exception as exc:  # noqa: BLE001
        await session.rollback()
        logger.error("查询历史会话失败: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="会话数据库暂不可用",
        ) from exc

    return SessionListResponse(
        items=[_session_response(row) for row in rows],
        page=page,
        page_size=page_size,
        total=total,
        has_more=page * page_size < total,
    )


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    payload: SessionCreateRequest,
    session: DbSession,
    user: OptionalUser,
) -> SessionResponse:
    """创建新会话；已登录时绑定到当前用户，未登录时 user_id 为 NULL。"""
    row = SessionMetadata(
        session_id=str(uuid.uuid4()),
        user_id=user.id if user else None,
        title=(payload.title or "").strip() or "新对话",
    )
    try:
        session.add(row)
        await session.commit()
        await session.refresh(row)
    except Exception as exc:  # noqa: BLE001
        await session.rollback()
        logger.error("创建会话失败: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="会话数据库暂不可用",
        ) from exc

    return _session_response(row)


@router.get("/{session_id}/messages", response_model=list[MessageResponse])
async def list_messages(
    session_id: str,
    session: DbSession,
    user: CurrentUser,
) -> list[MessageResponse]:
    """返回指定会话的完整聊天记录（按时间正序），并校验会话归属。"""
    try:
        ownership = (
            await session.execute(
                select(SessionMetadata.session_id).where(
                    SessionMetadata.session_id == session_id,
                    SessionMetadata.user_id == user.id,
                )
            )
        ).scalar_one_or_none()

        if ownership is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="会话不存在或无权访问",
            )

        result = await session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
        )
        rows = list(result.scalars().all())
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        await session.rollback()
        logger.error("查询会话消息失败: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="消息数据库暂不可用",
        ) from exc

    return [
        MessageResponse(
            id=str(row.id),
            role=row.role,
            content=row.content,
            created_at=row.created_at,
        )
        for row in rows
    ]


# LangGraph checkpoint 表：以 ``thread_id``（即 session_id）存储 Agent 对话记忆。
_CHECKPOINT_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")


async def _delete_sessions(
    session: AsyncSession,
    user_id: uuid.UUID,
    session_ids: list[str],
) -> int:
    """在单事务内删除会话主记录、关联消息与 Agent 记忆（checkpoint）。

    数据一致性保障：
        - ``messages`` 由数据库级 ``ON DELETE CASCADE`` 随主记录级联删除；
        - LangGraph checkpoint 三张表按 ``thread_id``（即 session_id）手动清理；
        - 全部写操作位于同一事务，任一步失败整体回滚，避免脏数据。

    Returns:
        实际删除的会话数量（仅统计当前用户拥有的会话）。
    """
    if not session_ids:
        return 0

    try:
        # 1. 清理 Agent 记忆（checkpoint_writes / checkpoint_blobs / checkpoints）
        for sid in session_ids:
            for table in _CHECKPOINT_TABLES:
                await session.execute(
                    text(f"DELETE FROM {table} WHERE thread_id = :sid"),
                    {"sid": sid},
                )

        # 2. 删除会话主记录（messages 由 DB 级 ON DELETE CASCADE 级联删除）
        result = await session.execute(
            delete(SessionMetadata).where(
                SessionMetadata.session_id.in_(session_ids),
                SessionMetadata.user_id == user_id,
            )
        )
        await session.commit()
        return result.rowcount or 0
    except Exception as exc:  # noqa: BLE001
        await session.rollback()
        logger.error("删除会话失败: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="删除会话失败，请稍后重试",
        ) from exc


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    session: DbSession,
    user: CurrentUser,
) -> Response:
    """删除单条会话（含关联消息与 Agent 记忆），仅允许删除自己的会话。"""
    deleted = await _delete_sessions(session, user.id, [session_id])
    if deleted == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在或无权删除",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def batch_delete_sessions(
    payload: BatchDeleteRequest,
    session: DbSession,
    user: CurrentUser,
) -> Response:
    """批量删除会话，仅删除当前用户拥有的会话，未拥有的会话自动忽略。"""
    await _delete_sessions(session, user.id, payload.session_ids)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
