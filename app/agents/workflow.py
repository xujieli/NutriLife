"""WORKFLOW_TASK 子图：饮食记录 → 提取食物 → 计算热量 → 阈值判断 → 生成建议。

执行链（线性为主，提取为空时短路）：

    extract_food ──(有食物)──> calculate_calories ──> check_threshold ──> generate_advice
         │
         └────────(无食物)───────────────────────────────────────────────> generate_advice

针对小模型的兜底策略：
    - 食物提取失败 → 空列表，下游优雅处理（提示用户补充信息）。
    - 热量查询（远程 Tool）失败 → Tenacity 重试 3 次（指数退避）→ 仍失败则
      降级到本地硬编码的粗略热量表。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.llm import get_chat_llm
from app.core.memory_manager import memory_optimize_node
from app.schemas.nutrition import FoodExtraction
from app.schemas.state import AgentState, get_latest_user_text
from app.tools.nutrition_tools import (
    NutritionToolError,
    get_calories_per_100g,
    get_fallback_calories,
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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
    retry=retry_if_exception_type(NutritionToolError),
    reraise=True,
)
def _call_calorie_tool(food_name: str) -> float:
    """带重试地调用远程热量查询 Tool（最多 3 次，指数退避）。"""
    return get_calories_per_100g(food_name)


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


def calculate_calories_node(state: AgentState) -> dict:
    """计算热量：远程 Tool 重试 3 次，最终失败降级到本地粗略热量表。"""
    workflow_data = state.get("workflow_data", {})
    foods = workflow_data.get("foods", [])

    total = 0.0
    enriched: list[dict] = []
    for item in foods:
        name = str(item.get("name", ""))
        grams = float(item.get("grams", 100.0))
        try:
            kcal_per_100g = _call_calorie_tool(name)
            source = "remote"
        except NutritionToolError as exc:
            logger.warning(
                "热量查询重试失败，降级到本地粗略热量表: {}",
                exc,
            )
            kcal_per_100g = get_fallback_calories(name)
            source = "fallback"

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

    logger.info("Workflow.calculate: 总热量 {:.1f} kcal", total)
    return {
        "workflow_data": {
            **workflow_data,
            "foods": enriched,
            "total_calories": round(total, 1),
        },
    }


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
