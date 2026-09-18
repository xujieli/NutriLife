"""WORKFLOW_TASK 子图：饮食记录 → 提取食物 → **Tool Calling 计算热量** → 阈值判断 → 生成建议。

执行链（线性为主，提取为空时短路）：

    extract_food ──(有食物)──> calculate_calories(LLM Tool Calling 环节) ──> check_threshold ──> generate_advice
         │
         └────────(无食物)───────────────────────────────────────────────────────────────> generate_advice

关键设计——``calculate_calories_node`` 升级为标准 LangChain Tool Calling 流程：
    1. LLM 通过 ``bind_tools`` 绑定 ``nutrition_calorie_lookup_tool``。
    2. LLM 自主判断是否需要调用工具、为每个食物生成独立的 ``tool_calls``。
    3. 逐个执行工具（内置重试 + Open Food Facts MCP / 默认估算降级）。
    4. 汇总所有工具结果，计算总热量写入 ``state["workflow_data"]``。
    5. 若 LLM 未生成任何 ``tool_calls``（小模型 Tool Calling 能力不足时），
       降级为"按食物列表直接调用工具"，保证功能可用。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from loguru import logger

from app.core.llm import get_chat_llm
from app.core.memory_manager import memory_optimize_node
from app.schemas.nutrition import FoodExtraction
from app.schemas.state import AgentState, get_latest_user_text
from app.tools.openfoodfacts_mcp_tools import (
    NUTRITION_CALORIE_TOOLS,
    NutritionCalorieLookupOutput,
    get_fallback_calories,
    nutrition_calorie_lookup_tool,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

# 单餐热量阈值（千卡），超过则提示超标
MEAL_CALORIE_THRESHOLD = 700.0

_EXTRACT_PROMPT = """\
你是营养记录助手。从用户消息中提取食物及其估算克数。

few-shot 示例：
输入："记录午餐：汉堡、薯条，计算热量并给建议"
输出：{"foods": [{"name": "汉堡", "grams": 200}, {"name": "薯条", "grams": 150}]}

输入："我今天吃了两个鸡蛋和一杯牛奶"
输出：{"foods": [{"name": "鸡蛋", "grams": 100}, {"name": "牛奶", "grams": 250}]}

规则：
1. 只提取用户明确提到的食物，不要凭空添加。
2. grams 为估算值；无法确定时给常见分量（一个汉堡约 200g，一杯约 250g）。
3. 若用户没有提到任何食物，返回空列表。
"""

_ADVICE_PROMPT = """\
你是 NutriLife 的营养顾问。根据以下饮食记录生成简短、实用的建议。

饮食记录：
{record}

要求：
1. 用中文，2~4 句话。
2. 若总热量超标，明确指出并给出可执行的改进建议；若未超标，给予肯定并提示均衡饮食。
3. 涉及疾病或体重管理时，提醒用户以专业医生/营养师的意见为准。
"""


# ──────────────────────────────────────────────────────────────
# Tool Calling 辅助：构造 Prompt / 解析 tool_calls / 执行工具
# ──────────────────────────────────────────────────────────────

_CALCULATE_TOOL_CALL_SYSTEM_PROMPT = """\
你是 NutriLife 的营养计算助手。用户会提供一份待查询热量的食物清单。

请按以下规则操作：
1. **必须** 对清单中的 **每一个** 食物分别调用 ``nutrition_calorie_lookup`` 工具，
   查询其每 100g 的热量值（千卡）。
2. 一次可以生成多个 tool_calls，为每个食物生成一条独立调用。
3. 不要编造任何热量数值，必须通过工具查询。
4. 不要回答任何自然语言，只需生成工具调用即可。
"""


def _build_tool_call_messages(foods: list[dict[str, Any]]) -> list[BaseMessage]:
    """根据已提取的食物清单，构造 Tool Calling 阶段的对话消息。"""
    food_lines = []
    for idx, item in enumerate(foods, start=1):
        name = item.get("name", "未知食物")
        grams = float(item.get("grams", 100.0))
        food_lines.append(f"{idx}. {name}（{grams:.0f}g）")
    food_list_str = "\n".join(food_lines)

    user_content = (
        f"以下是需要查询热量的食物清单，请逐个调用工具查询每 100g 的热量：\n\n"
        f"{food_list_str}"
    )
    return [
        SystemMessage(content=_CALCULATE_TOOL_CALL_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]


_DEFAULT_TOOL: BaseTool = cast(BaseTool, nutrition_calorie_lookup_tool)


async def _execute_tool_calls(
    ai_msg: AIMessage,
    foods: list[dict[str, Any]],
    tool: BaseTool = _DEFAULT_TOOL,
) -> list[NutritionCalorieLookupOutput]:
    """解析并执行 LLM 生成的 ``tool_calls``，返回结构化结果列表。

    执行策略：
        - 先按 LLM 的 ``tool_calls`` 顺序执行。
        - 若 LLM 遗漏了某个食物（小模型 Tool Calling 能力不足时），
          补充使用"直接 invoke 工具"的方式补齐，保证不遗漏。
        - 若调用抛异常，降级到本地 fallback + 默认估算。
    """
    results: dict[str, NutritionCalorieLookupOutput] = {}

    # ── 1. 执行 LLM 显式生成的 tool_calls ──────────────────────────
    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    for tc in tool_calls:
        try:
            tc_name = tc.get("name", "")
            tc_args = tc.get("args", {}) or {}
            tc_id = tc.get("id", "")
            if tc_name != tool.name:
                logger.warning(
                    "Workflow.calories: 忽略未知工具调用 '{}'（期望 '{}'）",
                    tc_name,
                    tool.name,
                )
                continue

            food_name_in_call: str = str(tc_args.get("food_name", "")).strip()
            if not food_name_in_call:
                continue

            logger.info(
                "Workflow.calories: 执行 LLM tool_call[{}] → food='{}'",
                tc_id or "-",
                food_name_in_call,
            )
            raw = await tool.ainvoke(tc_args)
            parsed = NutritionCalorieLookupOutput.model_validate(raw)
            results[parsed.food_name] = parsed
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Workflow.calories: 执行 tool_call 失败: {}，将降级补全",
                exc,
            )

    # ── 2. 对 LLM 遗漏的食物，降级为直接调用工具补齐 ──────────────────
    for item in foods:
        name = str(item.get("name", "")).strip()
        if not name or name in results:
            continue
        logger.warning(
            "Workflow.calories: LLM 未为 '{}' 生成 tool_call，直接调用工具补齐",
            name,
        )
        try:
            raw = await tool.ainvoke({"food_name": name})
            parsed = NutritionCalorieLookupOutput.model_validate(raw)
            results[name] = parsed
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Workflow.calories: 直接调用工具也失败 '{}': {}，终极降级 fallback",
                name,
                exc,
            )
            results[name] = NutritionCalorieLookupOutput(
                food_name=name,
                kcal_per_100g=get_fallback_calories(name),
                source="default",
                message="终极降级：Open Food Facts MCP 调用失败，使用默认估算值",
            )

    return list(results.values())


async def extract_food_node(state: AgentState) -> dict:
    """提取食物：结构化输出，失败则空列表兜底。"""
    text = get_latest_user_text(state)
    logger.info("Workflow.extract_food: '{}'", text[:60])

    try:
        llm = get_chat_llm(temperature=0.0, streaming=False)
        structured = llm.with_structured_output(FoodExtraction)
        result: FoodExtraction = cast(
            FoodExtraction,
            await structured.ainvoke(
                [SystemMessage(content=_EXTRACT_PROMPT), HumanMessage(content=text)]
            ),
        )
        foods = result.foods
    except Exception as exc:  # noqa: BLE001
        logger.warning("食物提取失败，使用空列表兜底: {}", exc)
        foods = []

    logger.info("Workflow.extract_food: 提取到 {} 个食物", len(foods))
    return {
        "workflow_data": {
            "foods": [f.model_dump() for f in foods],
        },
    }


def route_after_extract(state: AgentState) -> str:
    """条件边：未提取到食物时直接跳转生成建议（短路）。"""
    foods = state.get("workflow_data", {}).get("foods", [])
    return "memory_optimize" if not foods else "calculate_calories"


async def calculate_calories_node(state: AgentState) -> dict:
    """计算热量节点——标准 LangChain Tool Calling 流程。

    **Tool Calling 执行流程**（本节点是 Workflow 子图中 Tool Calling 的"请求环节"）：

        1. 构造 Prompt（System + Human，列出食物清单）。
        2. 通过 ``llm.bind_tools(NUTRITION_CALORIE_TOOLS)`` 将热量查询工具绑定到 LLM。
        3. LLM 自主推理并生成 ``tool_calls`` 列表（每个食物一条调用）。
        4. 解析 ``tool_calls``，逐个 ``await tool.ainvoke()`` 执行工具。
        5. 对 LLM 遗漏的食物，降级为"直接按列表 ainvoke 工具"补齐。
        6. 汇总所有结果 → 按克数计算实际热量 → 写入 state。

    多级兜底（小模型 + 远程服务双重不可靠场景）：
        - LLM 未生成 ``tool_calls`` → 直接按食物列表循环 ``await tool.ainvoke``。
        - 单个工具调用失败 → 降级到默认热量估算。
    """
    workflow_data = state.get("workflow_data", {})
    foods = workflow_data.get("foods", [])

    if not foods:
        logger.info("Workflow.calories: 无食物，跳过 Tool Calling")
        return {
            "workflow_data": {
                **workflow_data,
                "foods": [],
                "total_calories": 0.0,
            },
        }

    # ── Step 1：构造 Tool Calling Prompt + 绑定工具到 LLM ───────────
    messages: list[BaseMessage] = _build_tool_call_messages(foods)
    try:
        llm = get_chat_llm(temperature=0.0, streaming=False)
        llm_with_tools = llm.bind_tools(NUTRITION_CALORIE_TOOLS)

        # ── Banner 格式打印请求日志（满足闭环验证可观测性要求） ──────
        logger.info(
            "=" * 68
            + "\n"
            + "│ Workflow.calories: LLM Tool Calling REQUEST               │\n"
            + "│ 工具: {}\n".format(
                ", ".join(cast(BaseTool, t).name for t in NUTRITION_CALORIE_TOOLS)
            )
            + "│ 食物数: {}  模型: {}\n".format(
                len(foods), getattr(llm, "model_name", "?")
            )
            + "=" * 68
        )
        for idx, food in enumerate(foods, 1):
            logger.info(
                "  [{:02d}] name='{}' grams={:.0f}",
                idx,
                food.get("name", "?"),
                float(food.get("grams", 0.0)),
            )

        # ── Step 2：LLM 生成 tool_calls（Tool Calling "请求环节"本体） ─
        ai_msg: AIMessage = cast(AIMessage, await llm_with_tools.ainvoke(messages))
        tool_calls_raw = getattr(ai_msg, "tool_calls", None) or []

        logger.info(
            "=" * 68
            + "\n"
            + "│ Workflow.calories: LLM Tool Calling RESPONSE              │\n"
            + "│ tool_calls 数量: {}  内容长度: {}\n".format(
                len(tool_calls_raw),
                len(str(ai_msg.content or "")),
            )
            + "=" * 68
        )
        for idx, tc in enumerate(tool_calls_raw, 1):
            logger.info(
                "  [{:02d}] id={} name={} args={}",
                idx,
                tc.get("id", "-"),
                tc.get("name", "-"),
                tc.get("args", {}),
            )

        # ── Step 3：执行 tool_calls + 补齐遗漏 ────────────────────────
        parsed_results = await _execute_tool_calls(ai_msg, foods)

    except Exception as exc:  # noqa: BLE001
        # LLM 整体失败（如 LM Studio 离线）→ 直接调用工具做终极兜底
        logger.error(
            "Workflow.calories: LLM Tool Calling 整体失败，降级为直接 invoke 工具: {}",
            exc,
        )
        parsed_results = await _fallback_direct_tool_invoke(foods)

    # ── Step 4：按克数汇总实际热量 ─────────────────────────────────
    result_map: dict[str, NutritionCalorieLookupOutput] = {
        r.food_name: r for r in parsed_results
    }
    total = 0.0
    enriched: list[dict[str, Any]] = []
    for item in foods:
        name = str(item.get("name", ""))
        grams = float(item.get("grams", 100.0))
        lookup = result_map.get(name)
        if lookup is None:
            # 理论上不会走到这里（_execute_tool_calls 已做终极降级），
            # 再加一层保险，保证即使极端异常也不会 crash。
            kcal_per_100g = get_fallback_calories(name)
            source = "default"
        else:
            kcal_per_100g = lookup.kcal_per_100g
            source = lookup.source

        kcal = kcal_per_100g * grams / 100.0
        total += kcal
        enriched.append(
            {
                "name": name,
                "grams": grams,
                "kcal": round(kcal, 1),
                "source": source,
            }
        )

    logger.info(
        "Workflow.calories: 总热量 {:.1f} kcal（{} 个食物）", total, len(enriched)
    )
    return {
        "workflow_data": {
            **workflow_data,
            "foods": enriched,
            "total_calories": round(total, 1),
        },
    }


async def _fallback_direct_tool_invoke(
    foods: list[dict[str, Any]],
) -> list[NutritionCalorieLookupOutput]:
    """终极兜底：跳过 LLM，直接对每个食物调用 ``nutrition_calorie_lookup_tool``。

    当 LLM 不可用（LM Studio 离线 / 超时 / 返回格式损坏）时使用，
    保证"计算热量"这一核心功能在极端场景下仍可用。
    """
    results: list[NutritionCalorieLookupOutput] = []
    for item in foods:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            raw = await _DEFAULT_TOOL.ainvoke({"food_name": name})
            results.append(NutritionCalorieLookupOutput.model_validate(raw))
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Workflow.calories: 终极兜底 invoke 也失败 '{}': {}，使用默认估算",
                name,
                exc,
            )
            results.append(
                NutritionCalorieLookupOutput(
                    food_name=name,
                    kcal_per_100g=get_fallback_calories(name),
                    source="default",
                    message="终极兜底：Open Food Facts MCP 调用失败，使用默认估算值",
                )
            )
    return results


def check_threshold_node(state: AgentState) -> dict:
    """判断是否超标。"""
    workflow_data = state.get("workflow_data", {})
    total = float(workflow_data.get("total_calories", 0.0))
    exceeded = total > MEAL_CALORIE_THRESHOLD
    logger.info(
        "Workflow.check_threshold: total={:.1f}, exceeded={}",
        total,
        exceeded,
    )
    return {
        "workflow_data": {
            **workflow_data,
            "exceeded": exceeded,
            "threshold": MEAL_CALORIE_THRESHOLD,
        },
    }


async def generate_advice_node(state: AgentState, config: RunnableConfig) -> dict:
    """生成建议（含无食物/LLM 失败时的兜底话术）。"""
    workflow_data = state.get("workflow_data", {})
    foods = workflow_data.get("foods", [])
    total = float(workflow_data.get("total_calories", 0.0))
    exceeded = bool(workflow_data.get("exceeded", False))

    if not foods:
        answer = (
            "抱歉，我没能从你的消息里识别出食物。请告诉我具体吃了什么、大约多少量。"
        )
    else:
        record_lines = [
            f"- {f['name']} {f['grams']:.0f}g ≈ {f['kcal']:.0f} kcal" for f in foods
        ]
        record = "\n".join(record_lines) + f"\n总热量：{total:.0f} kcal"
        if exceeded:
            record += f"（已超过单餐建议阈值 {MEAL_CALORIE_THRESHOLD:.0f} kcal）"

        try:
            llm = get_chat_llm(temperature=0.3)
            chunks: list[str] = []
            messages = [
                *(state.get("optimized_messages") or []),
                SystemMessage(content=_ADVICE_PROMPT.format(record=record)),
            ]
            # 流式生成：使 astream_events 能捕获 on_chat_model_stream token
            async for chunk in llm.astream(messages, config=config):
                if isinstance(chunk.content, str):
                    chunks.append(chunk.content)
            answer = "".join(chunks)
        except Exception as exc:  # noqa: BLE001
            logger.error("建议生成失败，使用兜底话术: {}", exc)
            answer = _fallback_advice(total, exceeded)

    logger.info("Workflow.generate_advice: '{}'", answer[:60])
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def _fallback_advice(total: float, exceeded: bool) -> str:
    """建议生成的本地兜底话术（LLM 不可用时）。"""
    if exceeded:
        return (
            f"本餐热量约 {total:.0f} 千卡，已超过建议阈值，"
            "建议减少油炸食物与含糖饮料，并增加蔬菜摄入。"
        )
    return f"本餐热量约 {total:.0f} 千卡，处于合理范围，注意荤素搭配、控制总摄入即可。"


def build_workflow_graph() -> CompiledStateGraph:
    """构建并编译 WORKFLOW_TASK 子图。"""
    workflow = StateGraph(AgentState)
    workflow.add_node("extract_food", extract_food_node)
    workflow.add_node("calculate_calories", calculate_calories_node)
    workflow.add_node("check_threshold", check_threshold_node)
    workflow.add_node("memory_optimize", memory_optimize_node)
    workflow.add_node("generate_advice", generate_advice_node)

    workflow.set_entry_point("extract_food")
    workflow.add_conditional_edges(
        "extract_food",
        route_after_extract,
        {
            "calculate_calories": "calculate_calories",
            "memory_optimize": "memory_optimize",
        },
    )
    workflow.add_edge("calculate_calories", "check_threshold")
    workflow.add_edge("check_threshold", "memory_optimize")
    workflow.add_edge("memory_optimize", "generate_advice")
    workflow.add_edge("generate_advice", END)

    return workflow.compile()
