"""NutriLife 主 Agent 图。

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

说明：
    - 图使用 AsyncPostgresSaver 持久化对话状态，支持多轮对话与断点续传。
    - RAG 分支先重写查询，再检索上下文，最后经记忆优化后生成答案。
    - 数据库不可用时降级为 MemorySaver，保证服务仍可运行。
"""

from __future__ import annotations

import asyncio
import inspect
import os
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from loguru import logger

from app.agents.nodes.rewrite_query import rewrite_query_node
from app.agents.router import route_after_router, router_node
from app.agents.workflow import build_workflow_graph
from app.core.database import DATABASE_URL
from app.core.llm import get_chat_llm
from app.core.memory_manager import memory_optimize_node
from app.schemas.router import Intent
from app.schemas.state import AgentState, get_latest_user_text

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


RECURSION_LIMIT = 15

_GENERAL_SYSTEM_PROMPT = """\
你是 NutriLife 营养健康助手。用简洁、友好的中文回答用户。
如果问题与营养健康无关，礼貌地说明你的职责范围，不要编造信息。
"""

_FALLBACK_RESPONSE = (
    "抱歉，我的知识库中没有关于这个问题的准确信息，建议咨询专业医生。"
)

_RAG_SYSTEM_PROMPT = """\
<instruction>
你是 NutriLife 的专业营养健康顾问。请严格遵守以下规则回答用户问题：

1. 只能依据下方「参考资料」中的内容回答，不得使用参考资料以外的任何知识，不得编造数据、研究结论或医学事实。
2. 如果参考资料与用户问题完全无关，或参考资料为空，你必须且只能回复以下固定话术（一字不改）：
"抱歉，我的知识库中没有关于这个问题的准确信息，建议咨询专业医生。"
3. 回答要简洁、专业、使用中文；对关键结论引用参考资料中的具体内容；涉及医疗建议时提醒用户以专业医生意见为准。
</instruction>

参考资料：
---------------------
{context_str}
---------------------

用户问题：{query_str}

回答：
"""

_UNCERTAINTY_PATTERNS = (
    "无法回答",
    "没有相关信息",
    "知识库中没有",
    "无法从提供的信息",
    "上下文中没有",
    "context does not",
    "cannot answer",
    "not enough information",
    "i don't know",
    "no relevant",
    "the provided context",
)


def _is_fallback_response(text: str) -> bool:
    """判断 LLM 是否表达了“无法回答”。"""
    lowered = text.lower()
    return any(pattern in lowered for pattern in _UNCERTAINTY_PATTERNS)


async def rag_retrieve_node(state: AgentState) -> dict[str, Any]:
    """RAG 检索节点：基于重写后的查询召回文档片段。"""
    query = state.get("rag_query") or get_latest_user_text(state)
    logger.info("RAG Retrieve: '{}'", query[:80])

    try:
        def _retrieve() -> list[Any]:
            from app.core.config import get_settings
            from app.rag.retriever import HybridRetriever

            retriever = HybridRetriever(config=get_settings().rag).initialize()
            return retriever.retrieve(query)

        nodes = await asyncio.to_thread(_retrieve)
        context = "\n\n".join(node.get_content().strip()[:1500] for node in nodes)
        sources: list[dict[str, Any]] = []
        for node in nodes:
            meta = node.node.metadata or {}
            source_name = (
                meta.get("source")
                or meta.get("file_name")
                or meta.get("filename")
                or "未知文档"
            )
            sources.append(
                {
                    "source": source_name,
                    "snippet": node.get_content()[:200].replace("\n", " ").strip(),
                    "score": round(node.score or 0.0, 4),
                }
            )
        logger.info("RAG Retrieve: 命中 {} 个片段", len(nodes))
        return {"rag_context": context, "sources": sources}
    except Exception as exc:  # noqa: BLE001
        logger.error("RAG 检索失败: {}", exc)
        return {"rag_context": "", "sources": []}


async def rag_generate_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """RAG 生成节点：基于检索上下文和记忆优化后的消息生成答案。"""
    query = state.get("rag_query") or get_latest_user_text(state)
    context = state.get("rag_context", "")

    if not context.strip():
        answer = _FALLBACK_RESPONSE
    else:
        prompt = _RAG_SYSTEM_PROMPT.format(context_str=context, query_str=query)
        messages: list[BaseMessage] = [
            *(state.get("optimized_messages") or []),
            SystemMessage(content=prompt),
        ]
        try:
            llm = get_chat_llm(temperature=0.1)
            chunks: list[str] = []
            async for chunk in llm.astream(messages, config=config):
                content = getattr(chunk, "content", None)
                if isinstance(content, str):
                    chunks.append(content)
            answer = "".join(chunks).strip()
            if not answer:
                raise ValueError("RAG LLM 返回空回答")
        except Exception as exc:  # noqa: BLE001
            logger.error("RAG 生成失败: {}", exc)
            answer = _FALLBACK_RESPONSE

    if answer != _FALLBACK_RESPONSE and _is_fallback_response(answer):
        answer = _FALLBACK_RESPONSE

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


async def general_chat_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """通用闲聊节点：LLM 流式生成，失败时兜底话术。"""
    question = get_latest_user_text(state)
    logger.info("GeneralChat node: '{}'", question[:60])

    optimized_messages = state.get("optimized_messages") or []
    if (
        optimized_messages
        and isinstance(optimized_messages[0], SystemMessage)
        and str(optimized_messages[0].content).startswith("【历史对话摘要】")
    ):
        messages: list[BaseMessage] = [
            optimized_messages[0],
            SystemMessage(content=_GENERAL_SYSTEM_PROMPT),
            *optimized_messages[1:],
        ]
    else:
        messages = [
            SystemMessage(content=_GENERAL_SYSTEM_PROMPT),
            *(optimized_messages or [HumanMessage(content=question)]),
        ]

    try:
        llm = get_chat_llm(temperature=0.3)
        chunks: list[str] = []
        async for chunk in llm.astream(messages, config=config):
            content = getattr(chunk, "content", None)
            if isinstance(content, str):
                chunks.append(content)
        answer = "".join(chunks).strip()
        if not answer:
            raise ValueError("GeneralChat LLM 返回空回答")
    except Exception as exc:  # noqa: BLE001
        logger.error("GeneralChat 生成失败: {}", exc)
        answer = (
            "你好！我是 NutriLife 营养健康助手，"
            "可以帮你记录饮食、计算热量或解答营养问题。"
        )

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def route_after_memory_optimize(state: AgentState) -> str:
    """共享 memory_optimize 节点后的条件路由。"""
    intent = state.get("current_intent", Intent.GENERAL_CHAT.value)
    if intent == Intent.RAG_QUERY.value:
        return "rag_generate"
    return "general_chat"


def build_graph(checkpointer: Any | None = None) -> "CompiledStateGraph":
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

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "rag": "rewrite_query",
            "workflow": "workflow",
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

    return graph.compile(checkpointer=checkpointer)


# ──────────────────────────────────────────────────────────────────
# Checkpointer 懒加载与单例图
# ──────────────────────────────────────────────────────────────────

_compiled_graph: "CompiledStateGraph | None" = None
_postgres_checkpointer: Any | None = None
_postgres_context: Any | None = None
_graph_lock = asyncio.Lock()


def _checkpoint_database_url() -> str:
    """把 SQLAlchemy URL 转换为 langgraph-checkpoint-postgres 可接受的形式。"""
    def _normalize(url: str) -> str:
        return (
            url.replace("postgresql+psycopg://", "postgresql://")
            .replace("postgresql+asyncpg://", "postgresql://")
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
            saver = await raw.__aenter__()
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


async def get_graph() -> "CompiledStateGraph":
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
    config: dict | None = None,
) -> AgentState:
    """异步执行一次完整 Agent 图。"""
    graph = await get_graph()
    merged = {"recursion_limit": RECURSION_LIMIT, **(config or {})}
    return await graph.ainvoke(state, merged)


def run(
    state: AgentState | dict,
    *,
    config: dict | None = None,
) -> AgentState:
    """同步执行一次完整 Agent 图（仅用于脚本 / 测试环境）。"""
    import asyncio

    return asyncio.run(arun(state, config=config))
