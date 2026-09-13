"""LangFuse 可观测性初始化模块（SDK v3，本地自托管）。

职责：
- 初始化 LangFuse v3 客户端（指向本地 Docker 实例，默认 http://localhost:3000）。
- 初始化 LangChain 兼容的 CallbackHandler（注入 LangGraph config，实现全链路追踪）。
- 未配置（缺少 Key 或显式禁用）时优雅降级，返回 None，不影响主流程。

v3 说明：
    - 凭据通过 ``Langfuse(public_key=..., secret_key=..., host=...)`` 配置单例客户端，
      不再传给 ``CallbackHandler`` 构造器（v3 的 handler 基本无参）。
    - trace 属性（session_id / user_id / tags）通过 LangChain config 的 ``metadata``
      字段传入（``langfuse_session_id`` 等键），见 ``app/api/v1/chat.py``。
"""

from __future__ import annotations

import functools
from typing import Any

from loguru import logger

from app.core.config import get_settings


@functools.lru_cache(maxsize=1)
def _langfuse_credentials() -> tuple[bool, str, str, str]:
    """返回 (enabled, secret_key, public_key, host)，供客户端与 Handler 复用。"""
    cfg = get_settings().langfuse
    enabled = cfg.enabled and bool(cfg.secret_key) and bool(cfg.public_key)
    return enabled, cfg.secret_key, cfg.public_key, cfg.host


def get_langfuse_client() -> Any | None:
    """获取 LangFuse v3 客户端；未配置 / 初始化失败时返回 None。

    Returns:
        Langfuse | None: 客户端实例。
    """
    enabled, secret_key, public_key, host = _langfuse_credentials()
    if not enabled:
        logger.info("LangFuse 未配置或已禁用，跳过客户端初始化")
        return None

    try:
        from langfuse import Langfuse

        return Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    except Exception as exc:  # noqa: BLE001
        logger.error("LangFuse 客户端初始化失败: {}", exc)
        return None


def get_langfuse_handler() -> Any | None:
    """获取 LangFuse v3 CallbackHandler，用于注入 LangGraph / LangChain config。

    v3 中 handler 无构造参数：凭据由单例客户端提供，trace 属性（session_id /
    user_id / tags）由调用方的 config metadata 提供。

    Returns:
        CallbackHandler | None: 已配置的 Handler；未配置时返回 None。
    """
    enabled, secret_key, public_key, host = _langfuse_credentials()
    if not enabled:
        return None

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # 配置单例客户端（handler 内部会读取它）
        Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        return CallbackHandler()
    except Exception as exc:  # noqa: BLE001
        logger.error("LangFuse CallbackHandler 初始化失败: {}", exc)
        return None
