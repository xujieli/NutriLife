"""RAG 查询重写节点。

利用最近历史上下文，把用户当前指代不清的问题改写成独立、完整、适合向量检索的查询。
针对 12B 小模型，优先使用 instructor 强制 JSON 结构化输出；若 instructor 不可用，
则降级到 LangChain ``with_structured_output``。
"""

from __future__ import annotations

from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from loguru import logger
from pydantic import BaseModel, Field

from app.core.llm import get_chat_llm
from app.schemas.state import AgentState, get_latest_user_text


class RewrittenQuery(BaseModel):
    """查询重写结构化输出。"""

    query: str = Field(description="改写后的独立完整检索查询")
    reasoning: str = Field(description="改写理由，一句话")


_REWRITE_SYSTEM_PROMPT = """\
你是 NutriLife 的查询改写器。你的任务是把用户当前的、可能依赖上下文的问句，改写成适合向量检索的独立查询。

要求：
1. 结合最近 3 条历史对话补全指代、省略和隐含约束。
2. 只做查询改写，不要回答问题，不要编造历史中不存在的信息。
3. 输出 query 与 reasoning 两个字段。

Few-shot：
历史：
用户：我尿酸高
当前用户输入：能吃这个吗
改写：query="尿酸高患者可以吃这个吗"，reasoning="补全了用户的尿酸高健康约束"

历史：
用户：我今天吃了两个鸡蛋和一杯牛奶
当前用户输入：热量超标了吗
改写：query="两个鸡蛋和一杯牛奶的热量是否超标"，reasoning="把指代的“热量”补全为具体食物"
"""


def format_history(state: AgentState, limit: int = 3) -> str:
    """提取当前输入之前的最近 N 条历史消息（供 RAG 与路由器复用）。"""
    messages: list[BaseMessage] = list(state.get("messages", []))
    if messages and isinstance(messages[-1], HumanMessage):
        messages = messages[:-1]

    recent = messages[-limit:]
    lines: list[str] = []
    for message in recent:
        role = "用户" if isinstance(message, HumanMessage) else "助手"
        content = message.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("text")
            )
        else:
            text = str(content)
        if text.strip():
            lines.append(f"{role}: {text.strip()}")
    return "\n".join(lines)


async def _rewrite_with_structured_output(
    llm: BaseChatModel,
    messages: list[BaseMessage],
) -> RewrittenQuery:
    """instructor 不可用时的结构化输出降级方案。"""
    structured_llm = llm.with_structured_output(RewrittenQuery)
    return cast(RewrittenQuery, await structured_llm.ainvoke(messages))


async def rewrite_query(user_input: str, history_text: str) -> str:
    """结合历史，把当前问题改写成独立、完整、适合检索的问题。

    该函数是「问题改写」的唯一抽象，供两处复用：
        - ``rewrite_query_node``：RAG 分支检索前的查询改写；
        - 路由器（``app.agents.router``）：意图识别前的问题澄清。

    任何失败都降级返回原始输入，保证主流程不中断。

    Args:
        user_input: 当前用户输入。
        history_text: 格式化后的历史对话文本（无历史时传空字符串）。

    Returns:
        str: 改写后的独立完整问题；失败时返回原始输入。
    """
    prompt_messages: list[BaseMessage] = [
        SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"历史对话：\n{history_text or '（无历史）'}\n\n"
                f"当前用户输入：{user_input}"
            )
        ),
    ]

    try:
        llm = get_chat_llm(temperature=0.0, streaming=False)
        result = await _rewrite_with_structured_output(llm, prompt_messages)

        query = result.query.strip()
        if not query:
            raise ValueError("重写结果为空")

        logger.info(
            "RewriteQuery → '{}' (reasoning='{}')",
            query[:80],
            result.reasoning[:60],
        )
        return query
    except Exception as exc:  # noqa: BLE001
        logger.error("查询重写完全失败，降级使用原始输入: {}", exc)
        return user_input


async def rewrite_query_node(state: AgentState) -> dict[str, str]:
    """重写查询并写入 ``state["rag_query"]``（复用 ``rewrite_query`` 抽象）。"""
    user_input = get_latest_user_text(state)
    history_text = format_history(state, limit=3)
    logger.info("RewriteQuery: 输入 '{}'", user_input[:60])

    query = await rewrite_query(user_input, history_text)
    return {"rag_query": query}
