"""
NutriLife LLM 客户端初始化模块。

职责：
- 初始化指向本地 LM Studio 的 ChatOpenAI 实例
- 暴露工厂函数，供 Agent 各节点按需获取

注意：
    本模块对象均为"懒加载"——仅在首次调用工厂函数时初始化，
    避免在 import 阶段依赖外部服务（LM Studio）是否在线。
    LangFuse 追踪已统一迁移至 ``app.core.langfuse`` 模块。
"""

from __future__ import annotations

import functools

import numpy as np
from langchain_openai import ChatOpenAI
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from loguru import logger
from pydantic import SecretStr

from app.core.config import get_settings


class NormalizedBGEEmbedding(OpenAILikeEmbedding):
    """
    终极版 BGE Embedding：强制 L2 归一化。
    彻底绕过 LlamaIndex 的 Score 转换 Bug。
    """

    def _l2_normalize(self, vec: list[float]) -> list[float]:
        """强制 L2 归一化，使向量模长严格等于 1"""
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            return [float(x) for x in (arr / norm)]
        return vec

    def _get_text_embedding(self, text: str) -> list[float]:
        raw_vec = super()._get_text_embedding(text)
        return self._l2_normalize(raw_vec)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        raw_vec = await super()._aget_text_embedding(text)
        return self._l2_normalize(raw_vec)


@functools.lru_cache(maxsize=8)
def get_chat_llm(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool | None = None,
) -> ChatOpenAI:
    """
    获取指向本地 LM Studio 的 ChatOpenAI 客户端单例。

    LM Studio 暴露与 OpenAI 兼容的 REST API，因此可直接复用
    ``langchain_openai.ChatOpenAI``，仅需替换 ``base_url``。

    Args:
        temperature: 采样温度，覆盖 ``settings.llm.temperature``。
            不传则使用配置文件中的默认值（0.1）。
        max_tokens: 最大生成 Token 数，覆盖 ``settings.llm.max_tokens``。
        streaming: 是否开启流式输出，覆盖 ``settings.llm.streaming``。

    Returns:
        ChatOpenAI: 已配置的 LangChain LLM 客户端实例。

    Example::

        from app.core.llm import get_chat_llm

        llm = get_chat_llm()
        response = llm.invoke("你好，请介绍 NutriLife 的功能")
        print(response.content)
    """
    settings = get_settings()
    lm = settings.lm_studio
    llm_cfg = settings.llm

    _temperature = temperature if temperature is not None else llm_cfg.temperature
    _max_tokens = max_tokens if max_tokens is not None else llm_cfg.max_tokens
    _streaming = streaming if streaming is not None else llm_cfg.streaming

    logger.info(
        "初始化 ChatOpenAI → model={}, base_url={}, temperature={}, max_tokens={}",
        lm.model,
        lm.base_url,
        _temperature,
        _max_tokens,
    )

    return ChatOpenAI(
        model=lm.model,
        base_url=lm.base_url,
        api_key=SecretStr(lm.api_key),  # LM Studio 不校验 key，填任意字符串
        temperature=_temperature,
        max_completion_tokens=_max_tokens,
        streaming=_streaming,
        timeout=llm_cfg.request_timeout,
    )
