"""会话历史相关的 Pydantic 响应模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.chat import SessionResponse


class MessageResponse(BaseModel):
    """单条聊天消息（供历史会话回看）。"""

    id: str
    role: str
    content: str
    created_at: datetime


class SessionListResponse(BaseModel):
    """分页的历史会话列表。"""

    items: list[SessionResponse]
    page: int
    page_size: int
    total: int
    has_more: bool


class BatchDeleteRequest(BaseModel):
    """批量删除会话请求体。"""

    session_ids: list[str] = Field(min_length=1, description="待删除的会话 ID 列表")

