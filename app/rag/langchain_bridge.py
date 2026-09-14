"""LlamaIndex 与 LangChain/LangGraph 之间的适配桥。

这个模块负责把 RAG 的两套类型系统隔离在同一个边界上：

* LlamaIndex 检索器产出 ``NodeWithScore``。
* LangChain/LangGraph 期望 ``Document`` 或纯文本 ``str``。

边界规则：

    1. 转换函数是纯函数，不依赖网络、数据库或全局状态；
    2. :class:`LlamaIndexLangChainRetriever` 实现 LangChain 的
       :class:`~langchain_core.retrievers.BaseRetriever` 接口，
       LangGraph 节点只需调用 ``ainvoke`` / ``invoke``；
    3. 节点内容在进入 LangChain 之前统一完成长度裁剪与元数据规整，
       避免编排层再次接触 LlamaIndex 内部对象。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from pydantic import Field

DEFAULT_MAX_LENGTH_PER_NODE = 1500

_SOURCE_METADATA_KEYS = ("source", "file_name", "filename")


def _node_content(node: NodeWithScore) -> str:
    """安全读取 LlamaIndex 节点的纯文本内容。"""
    try:
        content = node.get_content()
    except Exception:  # noqa: BLE001 - 兼容不同 LlamaIndex 版本的字段差异
        try:
            content = node.node.get_content()
        except Exception:  # noqa: BLE001
            content = ""
    return str(content or "").strip()


def _resolve_source(metadata: dict[str, Any]) -> str:
    """从元数据中解析来源文件名，找不到时使用统一兜底值。"""
    for key in _SOURCE_METADATA_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "未知文档"


def node_with_score_to_document(
    node: NodeWithScore,
    *,
    max_length_per_node: int | None = DEFAULT_MAX_LENGTH_PER_NODE,
    include_node_metadata: bool = True,
) -> Document:
    """将单个 LlamaIndex ``NodeWithScore`` 转换为 LangChain ``Document``。

    Args:
        node: LlamaIndex 检索结果。
        max_length_per_node: 单节点内容的截断长度；``None`` 表示不截断。
        include_node_metadata: 是否把 LlamaIndex 原始元数据合并进
            ``Document.metadata``。保留 ``source`` / ``file_path`` /
            ``node_id`` / ``score`` 等标准字段便于溯源。

    Returns:
        已规整的 LangChain Document。
    """
    content = _node_content(node)
    if max_length_per_node is not None and max_length_per_node > 0:
        content = content[:max_length_per_node]

    node_metadata = dict(node.node.metadata or {})
    source = _resolve_source(node_metadata)
    document_metadata: dict[str, Any] = {
        "source": source,
        "file_path": str(node_metadata.get("file_path", "") or ""),
        "score": float(node.score or 0.0),
        "node_id": str(node.node.node_id or ""),
    }

    if include_node_metadata:
        for key, value in node_metadata.items():
            document_metadata.setdefault(key, value)

    return Document(page_content=content, metadata=document_metadata)


def nodes_with_scores_to_documents(
    nodes: Sequence[NodeWithScore],
    *,
    max_length_per_node: int | None = DEFAULT_MAX_LENGTH_PER_NODE,
    include_node_metadata: bool = True,
) -> list[Document]:
    """批量转换 LlamaIndex 检索结果。"""
    return [
        node_with_score_to_document(
            node,
            max_length_per_node=max_length_per_node,
            include_node_metadata=include_node_metadata,
        )
        for node in nodes
    ]


def _coerce_retrieved_nodes(raw_nodes: Any) -> list[NodeWithScore]:
    """把不同 LlamaIndex 检索器的返回值统一成 ``list[NodeWithScore]``。"""
    if raw_nodes is None:
        return []
    if isinstance(raw_nodes, NodeWithScore):
        return [raw_nodes]
    return list(raw_nodes)


def build_context_str_from_documents(
    documents: Sequence[Document],
    *,
    max_length_per_node: int | None = None,
) -> str:
    """把 LangChain Document 列表拼成可直接填入 Prompt 的上下文字符串。"""
    chunks: list[str] = []
    for document in documents:
        content = (document.page_content or "").strip()
        if not content:
            continue
        if max_length_per_node is not None and max_length_per_node > 0:
            content = content[:max_length_per_node]
        chunks.append(content)
    return "\n\n".join(chunks)


class LlamaIndexLangChainRetriever(BaseRetriever):
    """把任意 LlamaIndex 检索器包装成 LangChain ``BaseRetriever``。

    示例::

        from app.rag.langchain_bridge import LlamaIndexLangChainRetriever

        langchain_retriever = LlamaIndexLangChainRetriever(
            llama_index_retriever=hybrid_retriever,
        )
        docs = await langchain_retriever.ainvoke("痛风能吃豆制品吗")

    Attributes:
        llama_index_retriever: 底层 LlamaIndex 检索器。需要至少实现
            ``retrieve(query)``；若同时实现 ``aretrieve(query)``，
            LangGraph 节点可直接获得原生异步检索能力。
        max_length_per_node: 转换时每个节点的最大内容长度。
        include_node_metadata: 是否保留 LlamaIndex 原始节点元数据。
    """

    llama_index_retriever: Any = Field(
        ...,
        description="底层 LlamaIndex 检索器，至少提供 retrieve(query) 方法。",
        exclude=True,
    )
    max_length_per_node: int | None = Field(
        default=DEFAULT_MAX_LENGTH_PER_NODE,
        ge=0,
        description="转换后单个 Document 的最大字符长度。",
    )
    include_node_metadata: bool = Field(
        default=True,
        description="是否将 LlamaIndex 节点元数据合并到 Document.metadata。",
    )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """同步检索并转换为 LangChain Document。"""
        del run_manager
        retrieve = getattr(self.llama_index_retriever, "retrieve", None)
        if retrieve is None:
            raise TypeError(
                "llama_index_retriever 未实现 retrieve(query)，无法作为 LangChain Retriever 使用。"
            )
        nodes = _coerce_retrieved_nodes(retrieve(query))
        return nodes_with_scores_to_documents(
            nodes,
            max_length_per_node=self.max_length_per_node,
            include_node_metadata=self.include_node_metadata,
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """异步检索：优先使用原生 ``aretrieve``，否则线程池隔离同步调用。"""
        del run_manager
        aretrieve = getattr(self.llama_index_retriever, "aretrieve", None)
        if aretrieve is not None:
            nodes = _coerce_retrieved_nodes(await aretrieve(query))
        else:
            retrieve = getattr(self.llama_index_retriever, "retrieve", None)
            if retrieve is None:
                raise TypeError(
                    "llama_index_retriever 未实现 retrieve(query)，"
                    "无法作为 LangChain Retriever 使用。"
                )
            nodes = _coerce_retrieved_nodes(await asyncio.to_thread(retrieve, query))

        return nodes_with_scores_to_documents(
            nodes,
            max_length_per_node=self.max_length_per_node,
            include_node_metadata=self.include_node_metadata,
        )


__all__ = [
    "DEFAULT_MAX_LENGTH_PER_NODE",
    "LlamaIndexLangChainRetriever",
    "build_context_str_from_documents",
    "node_with_score_to_document",
    "nodes_with_scores_to_documents",
]
