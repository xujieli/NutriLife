from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.llm import get_chat_llm
from app.schemas.state import AgentState, get_latest_user_text

_GENERAL_SYSTEM_PROMPT = """\
你是 NutriLife 营养健康助手。用简洁、友好的中文回答用户。
如果问题与营养健康无关，礼貌地说明你的职责范围，不要编造信息。
"""


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
