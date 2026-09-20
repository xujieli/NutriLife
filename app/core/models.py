"""NutriLife 核心 ORM 模型。

包含三张核心业务表：
    - ``users``：用户表（手机号、注册时间、最后登录时间）。
    - ``session_metadata``：会话表（关联用户，存储标题、创建/更新时间）。
    - ``messages``：消息表（关联会话，存储内容、发送角色、发送时间）。

索引策略：
    - ``users.phone`` 唯一索引：登录时按手机号反查用户，保证手机号唯一。
    - ``session_metadata.user_id`` 普通索引：按用户分页拉取历史会话。
    - ``messages.session_id`` 普通索引：按会话查询完整聊天记录。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class User(Base):
    """用户表，存储手机号与登录时间信息。"""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    phone: Mapped[str] = mapped_column(
        String(11),
        unique=True,
        index=True,
        nullable=False,
        comment="手机号（中国大陆 11 位，唯一索引）",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="注册时间",
    )
    last_login_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="最后登录时间",
    )

    sessions: Mapped[list[SessionMetadata]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class SessionMetadata(Base):
    """会话表：前端历史会话列表 + LangGraph 线程元数据。"""

    __tablename__ = "session_metadata"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="关联用户 ID；未登录会话为 NULL",
    )
    title: Mapped[str] = mapped_column(String(255), default="新对话", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped[User | None] = relationship(back_populates="sessions")
    messages: Mapped[list[Message]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class Message(Base):
    """消息表：存储单个会话内的完整聊天记录。"""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("session_metadata.session_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
        comment="关联会话 ID",
    )
    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="发送角色：user / assistant",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="消息内容",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="发送时间",
    )

    session: Mapped[SessionMetadata] = relationship(back_populates="messages")
