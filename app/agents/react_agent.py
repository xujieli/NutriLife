"""REACT_TASK：复杂开放式营养任务的 ReAct 执行节点。

这类任务（营养分析、膳食规划、综合评估等）步骤不确定、需要动态推理并
在「知识检索」与「热量查询」等多个工具之间自主选择，无法用固定的
Plan-and-Execute 流程（``workflow.py``）完成，因此采用 ReAct 循环：

    Reason → Act（调用工具）→ Observe（观察结果）→ Reason → ... → 最终回答

生产稳定性设计：
    - ``MAX_REACT_STEPS`` 硬上限，防止小模型在工具调用上无限循环；
    - 每次 LLM 调用、每个工具调用都 try-except 兜底，失败降级为可读话术；
    - 未知工具名 / 空回答 / LLM 离线等异常均有对应降级路径；
    - 所有 Tool Calling 请求/响应按 Banner 格式记录（可观测性）。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from loguru import logger

from app.core.llm import get_chat_llm
from app.schemas.state import AgentState, get_latest_user_text
from app.tools.nutrition_knowledge_tool import nutrition_knowledge_search_tool
from app.tools.openfoodfacts_mcp_tools import NUTRITION_CALORIE_TOOLS

# ReAct 最大迭代轮数：小模型可能反复调用工具，必须显式封顶。
MAX_REACT_STEPS = 6

REACT_TOOLS: list[BaseTool] = [
    *NUTRITION_CALORIE_TOOLS,
    nutrition_knowledge_search_tool,
]

_REACT_TOOL_BY_NAME: dict[str, BaseTool] = {tool.name: tool for tool in REACT_TOOLS}

_REACT_SYSTEM_PROMPT = """\
你是 NutriLife 的营养健康顾问，负责处理需要多步推理与工具协作的复杂任务。

请用 ReAct 方式工作：先思考，再调用合适的工具，观察结果后继续推理，直到能给出完整回答。

可用工具：
1. nutrition_calorie_lookup —— 查询单个食物每 100g 的热量（千卡）。
2. nutrition_knowledge_search —— 检索营养学 / 医学知识库。

规则：
1. 需要营养学 / 医学知识（如疾病饮食禁忌、推荐摄入量、营养素作用）时，先调用 nutrition_knowledge_search。
2. 需要食物热量数据时，调用 nutrition_calorie_lookup。
3. 不要编造任何数据或事实，所有结论必须来自工具返回结果。
4. 如果工具不可用或知识库没有相关信息，如实说明，不要编造。
5. 涉及疾病、用药或体重管理时，提醒用户以专业医生 / 营养师意见为准。
6. 最终用简洁的中文回答，不要输出推理过程。
"""


def _stringify(value: Any) -> str:
    """把工具返回结果安全地转成可注入 ToolMessage 的文本。"""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _fallback_answer(question: str) -> str:
    """ReAct 全链路失败时的兜底话术。"""
    if not question.strip():
        return "抱歉，我没有理解你的问题，请再描述一下你的需求。"
    return (
        "抱歉，我暂时无法完成这项分析。你可以换一种方式描述问题，"
        "或直接告诉我具体吃了什么，我可以先帮你计算热量。"
    )


async def _execute_tool_calls(
    ai_msg: AIMessage,
    config: RunnableConfig,
) -> list[ToolMessage]:
    """执行 AI 消息中的全部 ``tool_calls``，并为每个调用生成 ``ToolMessage``。

    每个工具独立 try-except：单个工具失败不会拖垮整个 ReAct 循环，
    而是把错误作为观察结果返回给模型，让模型继续推理或调整。
    """
    tool_messages: list[ToolMessage] = []
    for tc in getattr(ai_msg, "tool_calls", None) or []:
        tc_id = str(tc.get("id", ""))
        tc_name = str(tc.get("name", ""))
        tc_args = tc.get("args", {}) or {}

        tool = _REACT_TOOL_BY_NAME.get(tc_name)
        if tool is None:
            content = f"未知工具 '{tc_name}'，请改用可用工具之一：{list(_REACT_TOOL_BY_NAME)}"
            logger.warning("React: 模型调用了未知工具 '{}'", tc_name)
        else:
            logger.info("React: 执行工具 '{}' args={}", tc_name, tc_args)
            try:
                raw = await tool.ainvoke(tc_args, config=config)
                content = _stringify(raw)
            except Exception as exc:  # noqa: BLE001
                logger.error("React: 工具 '{}' 调用失败: {}", tc_name, exc)
                content = f"工具 '{tc_name}' 调用失败：{exc}"

        tool_messages.append(
            ToolMessage(content=content, tool_call_id=tc_id, name=tc_name)
        )
    return tool_messages


def _banner_request(step: int, messages: list[BaseMessage], llm: Any) -> None:
    """Banner 格式打印 ReAct LLM Tool Calling 请求。"""
    logger.info(
        "=" * 68
        + "\n"
        + f"│ React: LLM Tool Calling REQUEST (step {step})            │\n"
        + "│ 工具: {}\n".format(", ".join(_REACT_TOOL_BY_NAME))
        + "│ 历史消息数: {}  模型: {}\n".format(
            len(messages), getattr(llm, "model_name", "?")
        )
        + "=" * 68
    )


def _banner_response(ai_msg: AIMessage) -> None:
    """Banner 格式打印 ReAct LLM Tool Calling 响应。"""
    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    logger.info(
        "=" * 68
        + "\n"
        + "│ React: LLM Tool Calling RESPONSE                         │\n"
        + "│ tool_calls 数量: {}  内容长度: {}\n".format(
            len(tool_calls), len(str(ai_msg.content or ""))
        )
        + "=" * 68
    )
    for idx, tc in enumerate(tool_calls, 1):
        logger.info(
            "  [{:02d}] id={} name={} args={}",
            idx,
            tc.get("id", "-"),
            tc.get("name", "-"),
            tc.get("args", {}),
        )


async def react_agent_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """ReAct 节点：动态推理 + 多工具协作，直至产出最终回答。"""
    question = get_latest_user_text(state)
    logger.info("React: 输入 '{}'", question[:60])

    messages: list[BaseMessage] = [
        SystemMessage(content=_REACT_SYSTEM_PROMPT),
        *list(state.get("messages", [])),
    ]

    try:
        llm = get_chat_llm(temperature=0.0, streaming=False)
        llm_with_tools = llm.bind_tools(REACT_TOOLS)
    except Exception as exc:  # noqa: BLE001
        logger.error("React: 初始化 LLM 失败: {}", exc)
        answer = _fallback_answer(question)
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

    for step in range(1, MAX_REACT_STEPS + 1):
        _banner_request(step, messages, llm)

        try:
            ai_msg: AIMessage = await llm_with_tools.ainvoke(messages, config=config)
        except Exception as exc:  # noqa: BLE001
            logger.error("React: LLM 调用失败，终止循环: {}", exc)
            answer = _fallback_answer(question)
            return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

        messages.append(ai_msg)
        _banner_response(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            answer = str(ai_msg.content or "").strip() or _fallback_answer(question)
            logger.info("React: 产出最终回答（{} 轮后收敛）", step)
            return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

        tool_messages = await _execute_tool_calls(ai_msg, config)
        messages.extend(tool_messages)

    # 达到硬上限仍未收敛：放弃继续推理，返回兜底话术。
    logger.warning("React: 达到最大迭代次数 {}，强制结束", MAX_REACT_STEPS)
    answer = _fallback_answer(question)
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
