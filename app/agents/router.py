"""意图路由器节点。

针对 gemma-4-12b 小模型的能力边界做专门设计：

1. **绝不自由输出 JSON**——使用 ``with_structured_output(RouterOutput)`` 走
   function-calling 通道，由推理服务在 schema 层强制枚举值，避免小模型
   生成的 JSON 结构漂移导致的解析失败。
2. **双层兜底**：
   - 结构化输出调用/解析失败（如服务不支持 function calling）→ 回退 GENERAL_CHAT；
   - 置信度低于阈值 → 回退 GENERAL_CHAT。
3. **few-shot 提示**——系统 Prompt 内置三类意图的具体示例，用例子锚定边界，
   降低小模型对长指令的理解成本。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.core.llm import get_chat_llm
from app.schemas.router import Intent, RouterOutput
from app.schemas.state import AgentState, get_latest_user_text

# 置信度阈值：低于此值一律回退 GENERAL_CHAT。
# 小模型普遍高估自身置信度，故该阈值是"粗粒度安全网"，真正的护栏是
# 枚举强约束 + few-shot 示例。
ROUTER_CONFIDENCE_THRESHOLD = 0.6

_ROUTER_SYSTEM_PROMPT = """\
<instruction>
你是 NutriLife 的意图路由器。根据用户输入，判断它属于以下三种意图之一：

1. RAG_QUERY —— 营养学 / 医学知识问答，需要检索专业知识库。
2. WORKFLOW_TASK —— 记录饮食或执行多步骤任务（提取食物 → 计算热量 → 给建议）。
3. GENERAL_CHAT —— 日常寒暄或与营养健康无关的闲聊。

判断要点：
- 提到「记录 / 吃了 / 喝了 / 计算热量 / 超标 / 摄入」等饮食记录意图 → WORKFLOW_TASK。
- 询问「能不能吃 / 为什么 / 多少 / 症状 / 缺乏」等知识问题 → RAG_QUERY。
- 其余（寒暄、无关闲聊）→ GENERAL_CHAT。
- 无法确定时选择 GENERAL_CHAT，confidence 给较低值（如 0.3）。
</instruction>

<examples>
输入："痛风能吃豆制品吗" → intent=RAG_QUERY, confidence=0.95
输入："维生素D缺乏有什么症状" → intent=RAG_QUERY, confidence=0.95
输入："记录午餐：汉堡、薯条，计算热量并给建议" → intent=WORKFLOW_TASK, confidence=0.95
输入："我今天吃了两个鸡蛋和一杯牛奶，帮我算算热量" → intent=WORKFLOW_TASK, confidence=0.9
输入："你好" → intent=GENERAL_CHAT, confidence=0.9
输入："今天天气怎么样" → intent=GENERAL_CHAT, confidence=0.9
</examples>

<output_format>
必须输出 intent（三者之一）与 confidence（0~1 之间的数值），不要输出其他内容。
</output_format>
"""


async def router_node(state: AgentState) -> dict[str, str | float | list[str]]:
    """路由节点：结构化输出 + 置信度兜底。

    Returns:
        dict: 写入 ``current_intent`` 与 ``router_confidence``。
    """
    question = get_latest_user_text(state)
    logger.info("Router: 输入 '{}'", question[:60])

    # ── 结构化输出（function calling 强约束枚举）──────────────────
    try:
        # 路由需确定性：temperature 拉到最低，并关闭流式输出
        llm = get_chat_llm(temperature=0.0, streaming=False)
        structured_llm = llm.with_structured_output(RouterOutput)
        result: RouterOutput = await structured_llm.ainvoke(
            [
                SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
        )
    except Exception as exc:  # noqa: BLE001 —— 任何失败都走兜底，不允许抛出
        logger.warning("Router 结构化输出失败，回退 GENERAL_CHAT: {}", exc)
        return {
            "current_intent": Intent.GENERAL_CHAT.value,
            "router_confidence": 0.0,
            **_reset_transient_state(),
        }

    intent = result.intent if result.intent else Intent.GENERAL_CHAT
    confidence = result.confidence

    # ── 置信度兜底 ───────────────────────────────────────────────
    if confidence < ROUTER_CONFIDENCE_THRESHOLD:
        logger.info(
            "Router 置信度 {:.2f} 低于阈值 {:.2f}，回退 GENERAL_CHAT",
            confidence,
            ROUTER_CONFIDENCE_THRESHOLD,
        )
        return {
            "current_intent": Intent.GENERAL_CHAT.value,
            "router_confidence": confidence,
            **_reset_transient_state(),
        }

    logger.info(
        "Router → intent={}, confidence={:.2f}, reasoning='{}'",
        intent.value,
        confidence,
        result.reasoning[:60],
    )
    return {
        "current_intent": intent.value,
        "router_confidence": confidence,
        **_reset_transient_state(),
    }


def _reset_transient_state() -> dict[str, str | list[str]]:
    """清空跨轮次持久化时不应保留的临时字段。"""
    return {
        "rag_query": "",
        "rag_context": "",
        "sources": [],
        "optimized_messages": [],
        "final_answer": "",
        "error": "",
    }


def route_after_router(state: AgentState) -> str:
    """条件边：根据路由结果决定下一个节点。

    Returns:
        str: 条件边的路由键（``rag`` / ``workflow`` / ``general_chat``）。
    """
    intent = state.get("current_intent", Intent.GENERAL_CHAT.value)
    if intent == Intent.RAG_QUERY.value:
        return "rag"
    if intent == Intent.WORKFLOW_TASK.value:
        return "workflow"
    return "general_chat"
