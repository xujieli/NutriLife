"""对话路由：POST /api/chat，通过 SSE 流式返回。

流式协议（每行一条 ``data: <json>``）：
    - ``type``: text / tool_call / tool_result / intent / done
    - ``content``: 文本 token / 工具名 / 工具结果 / 意图 / 最终回答
    - ``node``: 产生该事件的 LangGraph 节点名
    - ``final``: 仅 text 事件携带，标记是否属于"最终回答"（True）还是"思考中"（False）

前端可据此区分"思考中"与"最终回答"：
    - ``final=False`` 的 text / tool_call / tool_result 为中间过程（思考中）
    - ``final=True`` 的 text 或 ``done`` 事件为最终回答
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import StreamEvent
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import RECURSION_LIMIT, get_graph
from app.core.config import get_settings
from app.core.database import AsyncSessionFactory, get_async_session
from app.core.langfuse import get_langfuse_handler
from app.core.models import Message, SessionMetadata, User
from app.core.security import get_current_user_optional
from app.schemas.chat import ChatRequest

if TYPE_CHECKING:
    # 静态类型检查时，Pylance 只走这里，认为它永远是 langgraph 的类
    from langgraph.errors import GraphRecursionError
else:
    try:
        from langgraph.errors import GraphRecursionError
    except ImportError:  # pragma: no cover —— 版本差异兜底

        class GraphRecursionError(Exception):
            """LangGraph 递归超限异常（版本兜底）。"""


router = APIRouter()

# 产生"最终回答"的节点（其余节点视为"思考中"）。
FINAL_ANSWER_NODES = frozenset({"rag_generate", "general_chat", "generate_advice"})

# LM Studio 健康检查缓存（避免每个请求额外打一次网络、拖慢首 token）
_HEALTH_CACHE: dict[str, float | bool] = {"checked_at": 0.0, "healthy": True}
_HEALTH_TTL = 20.0


async def _lm_studio_healthy() -> bool:
    """探测 LM Studio 服务是否在线（短 TTL 缓存）。"""
    now = time.monotonic()
    if now - _HEALTH_CACHE["checked_at"] < _HEALTH_TTL:
        return bool(_HEALTH_CACHE["healthy"])

    base_url = get_settings().lm_studio.base_url.rstrip("/")
    healthy = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base_url}/models")
            healthy = resp.status_code < 500
    except Exception as exc:  # noqa: BLE001
        logger.warning("LM Studio 健康检查失败: {}", exc)
        healthy = False

    _HEALTH_CACHE.update(checked_at=now, healthy=healthy)
    return healthy


def _sse(payload: dict[str, Any]) -> str:
    """将字典序列化为一条 SSE ``data:`` 行。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _title_from_input(user_input: str) -> str:
    """从首条用户输入生成会话标题。"""
    compact = " ".join(user_input.strip().split())
    return compact[:30] or "新对话"


async def _save_session_metadata(
    session: AsyncSession,
    session_id: str,
    user_input: str,
    user_id: str | None = None,
) -> None:
    """写入或更新会话元数据；数据库不可用时只记录日志。

    Args:
        user_id: 已登录用户的 ``str(user.id)``，未登录为 None。
    """
    try:
        result = await session.execute(
            select(SessionMetadata).where(SessionMetadata.session_id == session_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            session.add(
                SessionMetadata(
                    session_id=session_id,
                    user_id=user_id,
                    title=_title_from_input(user_input),
                )
            )
        else:
            row.updated_at = datetime.now(UTC)
            # 用户此前匿名创建了会话，后续登录后继续对话时补绑账号
            if user_id is not None and row.user_id is None:
                row.user_id = user_id
        await session.commit()
    except Exception as exc:  # noqa: BLE001
        await session.rollback()
        logger.error("会话元数据写入失败（对话仍会继续）: {}", exc)


async def _save_message(session_id: str, role: str, content: str) -> None:
    """持久化单条聊天消息；数据库不可用时只记录日志，不中断对话。"""
    if not content:
        return
    try:
        async with AsyncSessionFactory() as session:
            session.add(
                Message(session_id=session_id, role=role, content=content)
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.error("消息持久化失败（对话仍会继续）: {}", exc)


def _map_event(event: StreamEvent) -> dict[str, Any] | None:
    """将 LangGraph ``astream_events`` 事件映射为统一的流式协议。"""
    kind = event.get("event")
    if kind is None:
        return None

    metadata = event.get("metadata") or {}
    node = metadata.get("langgraph_node", "")
    data = event.get("data") or {}

    if kind == "on_chat_model_stream":
        chunk = data.get("chunk")
        content = getattr(chunk, "content", None)
        if isinstance(content, str) and content:
            return {
                "type": "text",
                "content": content,
                "node": node,
                "final": node in FINAL_ANSWER_NODES,
            }
        return None

    if kind == "on_tool_start":
        name = event.get("name", "tool")
        tool_input = data.get("input") or {}
        return {"type": "tool_call", "content": name, "node": node, "input": tool_input}

    if kind == "on_tool_end":
        output = data.get("output")
        return {"type": "tool_result", "content": str(output)[:1000], "node": node}

    # 路由完成：下发意图，供前端尽早显示「意图徽章」
    if kind == "on_chain_end" and node == "router":
        output = data.get("output")
        if isinstance(output, dict) and output.get("current_intent"):
            return {
                "type": "intent",
                "content": output["current_intent"],
                "confidence": output.get("router_confidence", 0.0),
                "node": node,
            }

    return None


def _is_root_end(event: StreamEvent) -> bool:
    """判断是否为根图（整次运行）的结束事件。"""
    return event.get("event") == "on_chain_end" and "langgraph_node" not in (
        event.get("metadata") or {}
    )


@router.post("/chat", response_model=None)
async def chat(
    request: ChatRequest,
    session: AsyncSession = Depends(get_async_session),
    user: User | None = Depends(get_current_user_optional),
) -> StreamingResponse | JSONResponse:
    """流式对话接口。

    每次请求创建一个 LangFuse trace（注入 CallbackHandler 到 LangGraph config），
    以 SSE 形式返回 Agent 的执行过程与最终回答。
    若本地模型服务（LM Studio）宕机，返回 503。
    """
    session_id = request.session_id or str(uuid.uuid4())
    user_input = request.user_input

    # ── 前置检查：LM Studio 宕机则直接 503，避免进入流后崩溃 ──
    if not await _lm_studio_healthy():
        return JSONResponse(
            status_code=503,
            content={
                "detail": "本地模型服务（LM Studio）暂不可用，请确认服务已启动并加载模型后重试。",
                "session_id": session_id,
            },
        )

    # 绑定用户（未登录为 None）并持久化用户消息
    await _save_session_metadata(
        session, session_id, user_input, user_id=str(user.id) if user else None
    )
    await _save_message(session_id, "user", user_input)

    # ── LangFuse trace：v3 凭据由 handler 单例客户端提供，trace 属性走 config metadata ──
    handler = get_langfuse_handler()
    callbacks = [handler] if handler is not None else []

    config: RunnableConfig = {
        "callbacks": callbacks,
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": session_id},
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_user_id": str(user.id) if user else "anonymous",
            "langfuse_tags": ["nutrilife", "chat"],
        },
    }

    graph = await get_graph()
    initial_state = {"messages": [HumanMessage(content=user_input)]}

    async def event_generator() -> AsyncIterator[str]:
        streamed_text: list[str] = []
        final_state: dict[str, Any] = {}

        try:
            async for event in graph.astream_events(
                initial_state, config=config, version="v2"
            ):
                # 捕获根图结束事件中的最终状态（含 final_answer / current_intent）
                if _is_root_end(event):
                    output = (event.get("data") or {}).get("output")
                    if isinstance(output, dict) and "final_answer" in output:
                        final_state = output

                payload = _map_event(event)
                if payload is not None:
                    if payload["type"] == "text":
                        streamed_text.append(payload["content"])
                    yield _sse(payload)
        except GraphRecursionError:
            # recursion_limit 触发：优雅截断，返回已生成的部分内容
            logger.warning(
                "触发 recursion_limit={}，返回已生成的部分内容", RECURSION_LIMIT
            )
            partial = "".join(streamed_text).strip()
            answer = partial or "抱歉，回答因处理步骤超限而被截断，请尝试更简洁的问题。"
            await _save_message(session_id, "assistant", answer)
            yield _sse(
                {
                    "type": "done",
                    "content": answer,
                    "session_id": session_id,
                    "intent": final_state.get("current_intent", ""),
                    "truncated": True,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent 流式执行失败: {}", exc)
            yield _sse(
                {
                    "type": "done",
                    "content": "抱歉，服务暂时不可用，请稍后再试。",
                    "session_id": session_id,
                    "error": str(exc),
                }
            )
        else:
            answer = final_state.get("final_answer") or "".join(streamed_text)
            await _save_message(session_id, "assistant", answer)
            yield _sse(
                {
                    "type": "done",
                    "content": answer,
                    "session_id": session_id,
                    "intent": final_state.get("current_intent", ""),
                    "sources": final_state.get("sources", []),
                }
            )
        finally:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
