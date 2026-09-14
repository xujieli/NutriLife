from typing import Any

from loguru import logger

from app.rag.langchain_bridge import build_context_str_from_documents
from app.rag.query_engine import (
    extract_source_references_from_documents,
    get_langchain_retriever,
    source_references_to_state,
)
from app.schemas.state import AgentState, get_latest_user_text


async def rag_retrieve_node(state: AgentState) -> dict[str, Any]:
    """RAG 检索节点（LangGraph 编排 → 只消费 LangChain 检索器）。

    编排职责（本节点）：
        - 从重写后的 rag_query 或最新用户消息取 query
        - try-except 捕获检索异常 → 降级为空 context
        - 把结果写入 ``AgentState.rag_context`` / ``sources``
    桥接职责（委托 query_engine / langchain_bridge 数据层）：
        - ``get_langchain_retriever`` 返回 LangChain ``BaseRetriever``
        - ``ainvoke`` 内部把 LlamaIndex ``NodeWithScore`` 转成 ``Document``
        - 引用溯源提取（``extract_source_references_from_documents`` 纯函数）
        - sources → state dict 转换（``source_references_to_state``）
        - Context 字符串拼接（``build_context_str_from_documents`` 纯函数）
    """
    query = state.get("rag_query") or get_latest_user_text(state)
    logger.info("RAG Retrieve: '{}'", query[:80])

    try:
        # ── LangChain 边界：节点不再接触 NodeWithScore ─────────────
        retriever = get_langchain_retriever()
        documents = await retriever.ainvoke(query)

        # ── 数据层：纯函数拼接 Context、提取结构化引用 ───────────
        context = build_context_str_from_documents(documents)
        source_refs = extract_source_references_from_documents(documents)
        sources = source_references_to_state(source_refs)

        logger.info("RAG Retrieve: 命中 {} 个 Document", len(documents))
        return {"rag_context": context, "sources": sources}
    except Exception as exc:  # noqa: BLE001
        # ── 编排层：异常降级（不影响后续节点，走拒答路径） ───────
        logger.error("RAG 检索失败: {}", exc)
        return {"rag_context": "", "sources": []}
