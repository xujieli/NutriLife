"""对话 API 请求模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /api/chat 请求体。"""

    user_input: str = Field(min_length=1, description="用户输入")
    session_id: str = Field(
        default="",
        description="会话 ID；为空时由服务端生成新的 session_id",
    )


class SessionCreateRequest(BaseModel):
    """POST /api/v1/sessions 请求体。"""

    title: str | None = Field(
        default=None,
        max_length=255,
        description="可选会话标题；为空时使用默认标题",
    )


class SessionResponse(BaseModel):
    """历史会话列表项。"""

    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
