"""NutriLife 主 Agent 图（LangGraph 编排逻辑，LlamaIndex 处理数据）。

拓扑：

                          ┌────────────┐
             entry ──────▶│   router   │
                          └─────┬──────┘
                    conditional_edges (route_after_router)
        ┌──────────────────────┼────────────────────────┐
        ▼                      ▼                         ▼
  ┌─────────────┐      ┌───────────────┐         ┌───────────────┐
  │ rewrite_query│      │   workflow     │         │ memory_optimize│
  └──────┬──────┘      └───────┬───────┘         └───────┬───────┘
         ▼                     ▼                        │
  ┌─────────────┐            END             ┌───────────▼───────────┐
  │ rag_retrieve │                            │ route_after_memory   │
  └──────┬──────┘                            └──────┬─────────┬──────┘
         ▼                                          ▼         ▼
  ┌──────────────┐                        ┌──────────────┐ ┌──────────────┐
  │memory_optimize│                        │ rag_generate │ │ general_chat │
  └──────┬───────┘                        └──────┬───────┘ └──────┬───────┘
         ▼                                       ▼                ▼
      END                                       END              END

**数据 / 编排 职责分离（重构后）：**
    - **LlamaIndex 数据层**（``app/rag/query_engine.py``）：
      RAG 共享 Prompt 模板字符串、``FALLBACK_RESPONSE`` 拒答话术常量、
      ``is_fallback_response()`` 不确定性检测纯函数、
      ``extract_source_references_from_documents()`` /
      ``source_references_to_state()`` 引用与 Context 构造纯函数、
      ``get_langchain_retriever()`` / ``get_synthesizer()`` 懒加载单例工厂。
    - **LangGraph 编排层**（本模块）：
      条件路由、节点依赖、``AgentState`` 字段写入、try-except 降级兜底、
      RAG 流式生成调用（保留小模型 astream UX）。
    - **不再重复**：本地 ``_RAG_SYSTEM_PROMPT`` / ``_FALLBACK_RESPONSE`` /
      ``_UNCERTAINTY_PATTERNS`` / ``_is_fallback_response`` / 手写 sources dict
      全部删除，改为 import 数据层单一真相源。
"""

from __future__ import annotations

import asyncio
import inspect
import os
from typing import TYPE_CHECKING, Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from loguru import logger

from app.agents.nodes import (
    general_chat_node,
    rag_generate_node,
    rag_retrieve_node,
    rewrite_query_node,
)
from app.agents.parallel_analysis import build_parallel_analysis_graph
from app.agents.react_agent import react_agent_node
from app.agents.router import route_after_router, router_node
from app.agents.workflow import build_workflow_graph
from app.core.database import DATABASE_URL
from app.core.memory_manager import memory_optimize_node
from app.schemas.router import Intent
from app.schemas.state import AgentState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


RECURSION_LIMIT = 15


def route_after_memory_optimize(state: AgentState) -> str:
    """共享 memory_optimize 节点后的条件路由。"""
    intent = state.get("current_intent", Intent.GENERAL_CHAT.value)
    if intent == Intent.RAG_QUERY.value:
        return "rag_generate"
    return "general_chat"


def build_graph(checkpointer: Any | None = None) -> CompiledStateGraph:
    """组装并编译完整 Agent 图。

    Args:
        checkpointer: LangGraph checkpointer。传入 AsyncPostgresSaver 时持久化到
            PostgreSQL；None 时由调用方在外部传入 MemorySaver 兜底。
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("rag_retrieve", rag_retrieve_node)
    graph.add_node("rag_generate", rag_generate_node)
    graph.add_node("memory_optimize", memory_optimize_node)
    graph.add_node("general_chat", general_chat_node)
    graph.add_node("workflow", build_workflow_graph())
    graph.add_node("parallel", build_parallel_analysis_graph())
    graph.add_node("react", react_agent_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "rag": "rewrite_query",
            "workflow": "workflow",
            "parallel": "parallel",
            "react": "react",
            "general_chat": "memory_optimize",
        },
    )
    graph.add_edge("rewrite_query", "rag_retrieve")
    graph.add_edge("rag_retrieve", "memory_optimize")
    graph.add_conditional_edges(
        "memory_optimize",
        route_after_memory_optimize,
        {
            "rag_generate": "rag_generate",
            "general_chat": "general_chat",
        },
    )
    graph.add_edge("rag_generate", END)
    graph.add_edge("general_chat", END)
    graph.add_edge("workflow", END)
    graph.add_edge("parallel", END)
    graph.add_edge("react", END)

    return graph.compile(checkpointer=checkpointer)


# ──────────────────────────────────────────────────────────────────
# Checkpointer 懒加载与单例图
# ──────────────────────────────────────────────────────────────────

_compiled_graph: CompiledStateGraph | None = None
_postgres_checkpointer: Any | None = None
_postgres_context: Any | None = None
_graph_lock = asyncio.Lock()


def _checkpoint_database_url() -> str:
    """把 SQLAlchemy URL 转换为 langgraph-checkpoint-postgres 可接受的形式。"""

    def _normalize(url: str) -> str:
        return url.replace("postgresql+psycopg://", "postgresql://").replace(
            "postgresql+asyncpg://", "postgresql://"
        )

    configured = os.getenv("LANGGRAPH_CHECKPOINT_URL", "").strip()
    if configured:
        return _normalize(configured)
    return _normalize(DATABASE_URL)


async def _ensure_postgres_checkpointer() -> Any | None:
    """初始化 AsyncPostgresSaver；失败时返回 None 由上层降级。"""
    global _postgres_checkpointer, _postgres_context

    if _postgres_checkpointer is not None:
        return _postgres_checkpointer

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        raw = AsyncPostgresSaver.from_conn_string(_checkpoint_database_url())
        if inspect.isawaitable(raw):
            raw = await raw

        if hasattr(raw, "__aenter__"):
            saver: Any = await raw.__aenter__()
            _postgres_context = raw
        else:
            saver = raw

        if hasattr(saver, "setup"):
            await saver.setup()

        _postgres_checkpointer = saver
        logger.info("AsyncPostgresSaver 初始化成功")
        return _postgres_checkpointer
    except Exception as exc:  # noqa: BLE001
        if _postgres_context is not None and hasattr(_postgres_context, "__aexit__"):
            try:
                await _postgres_context.__aexit__(None, None, None)
            except Exception as close_exc:  # noqa: BLE001
                logger.error("清理失败的 AsyncPostgresSaver 上下文出错: {}", close_exc)
        _postgres_context = None
        _postgres_checkpointer = None
        logger.error(
            "AsyncPostgresSaver 初始化失败，将降级为 MemorySaver: {}",
            exc,
        )
        return None


async def close_checkpointer() -> None:
    """关闭 PostgreSQL checkpointer 连接。"""
    global _postgres_context, _postgres_checkpointer

    if _postgres_context is not None and hasattr(_postgres_context, "__aexit__"):
        try:
            await _postgres_context.__aexit__(None, None, None)
        except Exception as exc:  # noqa: BLE001
            logger.error("关闭 AsyncPostgresSaver 失败: {}", exc)
    _postgres_context = None
    _postgres_checkpointer = None


async def get_graph() -> CompiledStateGraph:
    """获取编译后的 Agent 图单例，优先使用 PostgreSQL checkpointer。"""
    global _compiled_graph

    if _compiled_graph is not None:
        return _compiled_graph

    async with _graph_lock:
        if _compiled_graph is None:
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = await _ensure_postgres_checkpointer()
            fallback = checkpointer or MemorySaver()
            _compiled_graph = build_graph(checkpointer=fallback)
            if checkpointer is None:
                logger.warning("当前使用 MemorySaver，对话历史不会跨进程持久化")
    return _compiled_graph


async def arun(
    state: AgentState | dict,
    *,
    config: RunnableConfig | None = None,
) -> AgentState:
    """异步执行一次完整 Agent 图。"""
    graph = await get_graph()
    merged: RunnableConfig = {"recursion_limit": RECURSION_LIMIT, **(config or {})}
    return cast(AgentState, await graph.ainvoke(state, merged))


def run(
    state: AgentState | dict,
    *,
    config: RunnableConfig | None = None,
) -> AgentState:
    """同步执行一次完整 Agent 图（仅用于脚本 / 测试环境）。"""
    import asyncio

    return asyncio.run(arun(state, config=config))
