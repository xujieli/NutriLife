from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.llm import get_chat_llm
from app.rag.query_engine import (
    FALLBACK_RESPONSE,
    RAG_SYSTEM_PROMPT_TEMPLATE_STR,
    is_fallback_response,
)
from app.schemas.state import AgentState, get_latest_user_text


async def rag_generate_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """RAG 生成节点（LangGraph 编排 → 复用数据层 Prompt/拒答 常量与校验）。

    编排职责（本节点，LangGraph 特有的流式交互）：
        - 空 Context → 直接拒答（不调 LLM）
        - 使用共享模板组装 SystemMessage（含 memory_optimize 注入的历史）
        - 流式 ``astream`` 生成（小模型 UX 所需，保留在编排层）
        - 兜底异常 → 拒答
    数据职责（委托 query_engine 数据层，消除重复）：
        - ``FALLBACK_RESPONSE`` 拒答话术常量（原 ``_FALLBACK_RESPONSE`` 删除）
        - ``RAG_SYSTEM_PROMPT_TEMPLATE_STR`` 共享 Prompt 模板字符串
          （原本地 ``_RAG_SYSTEM_PROMPT`` 删除）
        - ``is_fallback_response`` 不确定性校验（原本地 ``_is_fallback_response`` 删除）
    """
    query = state.get("rag_query") or get_latest_user_text(state)
    context = state.get("rag_context", "")

    # ── 编排决策：空 Context 直接拒答 ──────────────────────────────
    if not context.strip():
        answer = FALLBACK_RESPONSE
    else:
        # ── 数据层：共享 Prompt 模板字符串（单一真相源） ──────────
        prompt = RAG_SYSTEM_PROMPT_TEMPLATE_STR.format(
            context_str=context, query_str=query
        )
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
            answer = FALLBACK_RESPONSE

    # ── 数据层：不确定性二次校验（共享纯函数） ────────────────────
    if answer != FALLBACK_RESPONSE and is_fallback_response(answer):
        answer = FALLBACK_RESPONSE

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }
