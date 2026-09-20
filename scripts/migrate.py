#!/usr/bin/env python3
"""NutriLife 数据库迁移脚本（幂等）。

职责：
    1. 通过 ``create_all`` 确保 ``users`` / ``session_metadata`` / ``messages``
       三张表存在（含模型定义的索引与外键）。
    2. 兼容旧库：为既有的 ``session_metadata`` 表补充 ``user_id`` 列与索引
       （``ADD COLUMN IF NOT EXISTS`` 幂等）。

使用方式::

    python scripts/migrate.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，使脚本可直接导入 app 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.core import models  # noqa: F401, E402  # 导入以注册所有 ORM 模型
from app.core.database import Base, engine  # noqa: E402


async def migrate() -> None:
    """执行幂等迁移。"""
    async with engine.begin() as conn:
        # 1. 创建缺失的表（含模型声明的唯一/普通索引与外键）
        await conn.run_sync(Base.metadata.create_all)
        logger.info("基础表结构已确保存在（users / session_metadata / messages）")

        # 2. 旧库兼容：补充 user_id 列（新列时一并建立外键）
        await conn.execute(
            text(
                "ALTER TABLE session_metadata "
                "ADD COLUMN IF NOT EXISTS user_id UUID "
                "REFERENCES users(id) ON DELETE SET NULL"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_session_metadata_user_id "
                "ON session_metadata (user_id)"
            )
        )
        logger.info("session_metadata.user_id 列与索引已确保存在")


async def main() -> None:
    """脚本主入口。"""
    logger.info("开始执行 NutriLife 数据库迁移...")
    await migrate()
    await engine.dispose()
    logger.info("数据库迁移完成")


if __name__ == "__main__":
    asyncio.run(main())
