"""PostgreSQL 异步数据库连接池与依赖注入。

本模块使用 SQLAlchemy 2.0 async engine + ``async_sessionmaker``，
所有数据库访问均在异步上下文中执行。连接串优先读取环境变量
``DATABASE_URL``；未配置时使用项目默认的本地 PostgreSQL 占位符。
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

load_dotenv()


class Base(DeclarativeBase):
    """所有 SQLAlchemy ORM 模型的声明式基类。"""


DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://postgres:postgres@localhost:5432/nutrilife"
)
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=1800,
)

AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def init_db() -> None:
    """创建缺失的数据表。

    仅用于开发 / 测试环境的轻量初始化；生产环境建议使用 Alembic 迁移。
    任何建表失败都会记录日志，但不会中断应用启动。
    """
    try:
        from app.core import models  # noqa: F401

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("数据库表初始化完成")
    except Exception as exc:  # noqa: BLE001
        logger.error("数据库表初始化失败（应用将以降级模式继续运行）: {}", exc)


async def close_db() -> None:
    """关闭数据库连接池，释放底层连接。"""
    try:
        await engine.dispose()
        logger.info("数据库连接池已关闭")
    except Exception as exc:  # noqa: BLE001
        logger.error("关闭数据库连接池失败: {}", exc)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：为单个请求提供异步数据库会话。"""
    async with AsyncSessionFactory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
