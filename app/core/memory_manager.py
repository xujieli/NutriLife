"""长期记忆优化策略：滑动窗口 + LLM 摘要压缩。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger
from pydantic import BaseModel, Field

from app.core.llm import get_chat_llm
from app.schemas.state import AgentState


MEMORY_KEEP_RECENT = 6
MEMORY_THRESHOLD = 10
SUMMARY_MAX_LENGTH = 200


class ConversationSummary(BaseModel):
    """历史对话摘要结构化输出。"""

    summary: str = Field(
        description="不超过 200 字的历史对话摘要，只保留事实、约束和未完成事项",
        max_length=SUMMARY_MAX_LENGTH,
    )


_SUMMARY_SYSTEM_PROMPT = """\
你是对话记忆压缩器。请把历史对话压缩为一段不超过 200 字的中文摘要。

要求：
1. 只保留对后续对话有帮助的信息：用户身份/健康约束、饮食偏好、已记录事项、未完成事项。
2. 不要复述寒暄，不要编造对话中不存在的信息。
3. 使用第三人称或“用户”作为主语，语言简洁。
"""


def _content_to_text(content: Any) -> str:
    """将 LangChain 消息 content 转为纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _format_messages(messages: list[BaseMessage]) -> str:
    """把历史消息转换为紧凑文本，供摘要 LLM 读取。"""
    lines: list[str] = []
    for message in messages:
        role = "用户" if isinstance(message, HumanMessage) else "助手"
        text = _content_to_text(message.content)
        if text.strip():
            lines.append(f"{role}: {text.strip()}")
    return "\n".join(lines)


async def optimize_memory(
    messages: list[BaseMessage],
    llm: Any,
) -> list[BaseMessage]:
    """对历史消息执行「滑动窗口 + 摘要压缩」。

    规则：
    - 消息数不超过 ``MEMORY_THRESHOLD`` 时原样返回；
    - 超过时保留最近 ``MEMORY_KEEP_RECENT`` 条完整消息，将更早消息压缩为
      一条 ``SystemMessage`` 并置于列表最前；
    - LLM 摘要失败时降级为直接截断（仅保留最近窗口）。
    """
    if len(messages) <= MEMORY_THRESHOLD:
        return list(messages)

    older_messages = messages[:-MEMORY_KEEP_RECENT]
    recent_messages = messages[-MEMORY_KEEP_RECENT:]

    try:
        history_text = _format_messages(older_messages)
        structured_llm = llm.with_structured_output(ConversationSummary)
        result: ConversationSummary = await structured_llm.ainvoke(
            [
                SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=f"历史对话：\n{history_text}"),
            ]
        )
        summary = result.summary.strip()
        if not summary:
            raise ValueError("LLM 返回的摘要为空")

        logger.info(
            "记忆压缩完成：旧消息 {} 条 → 摘要 {} 字，保留最近 {} 条",
            len(older_messages),
            len(summary),
            len(recent_messages),
        )
        return [
            SystemMessage(content=f"【历史对话摘要】：{summary}"),
            *recent_messages,
        ]
    except Exception as exc:  # noqa: BLE001
        logger.error("历史摘要生成失败，降级为滑动窗口截断: {}", exc)
        return list(recent_messages)


async def memory_optimize_node(
    state: AgentState,
    config: RunnableConfig | None = None,
) -> dict[str, list[BaseMessage]]:
    """LangGraph 节点：为最终生成节点准备压缩后的消息。"""
    messages = state.get("messages", [])
    try:
        llm = get_chat_llm(temperature=0.0, streaming=False)
        optimized = await optimize_memory(messages, llm)
    except Exception as exc:  # noqa: BLE001
        logger.error("记忆优化节点失败，使用完整历史消息兜底: {}", exc)
        optimized = list(messages)
    return {"optimized_messages": optimized}
