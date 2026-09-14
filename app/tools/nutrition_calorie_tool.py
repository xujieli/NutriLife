"""营养热量查询工具——标准 LangChain Tool 实现。

本模块实现了一个符合 LangChain Tool 规范的 ``NutritionCalorieLookup`` 工具，
供 Workflow 子图中的 ``calculate_calories_node`` 通过标准 Tool Calling 流程
（``bind_tools`` → LLM 生成 ``tool_calls`` → 解析执行 → 结果回填）调用。

架构说明（与旧版 ``nutrition_tools.py`` 的关系）：
    - 旧版 :mod:`app.tools.nutrition_tools` 提供的是"裸函数"级能力
      （``get_calories_per_100g`` / ``get_fallback_calories``），
      仅能被直接 import 调用，不支持 LLM 自主 Tool Calling。
    - 本模块 **不重复造轮子**：在裸函数能力之上做 LangChain Tool 封装，
      在 ``_run`` 中编排 Tenacity 重试 + 本地 Fallback 降级策略，
      对外暴露统一的结构化输出（含 ``source`` 字段标识数据来源）。

典型调用链（Workflow 侧）::

    llm = get_chat_llm()
    llm_with_tools = llm.bind_tools([nutrition_calorie_lookup_tool])
    ai_msg = llm_with_tools.invoke([...])         # LLM 决定是否调用工具、生成 tool_calls
    for tc in ai_msg.tool_calls:                   # 解析 LLM 的工具调用请求
        tool_result = nutrition_calorie_lookup_tool.invoke(tc["args"])   # 真正执行
        ...  # 把结果写回 state/workflow_data
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from loguru import logger
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.tools.nutrition_tools import (
    DEFAULT_UNKNOWN_KCAL_PER_100G,
    FALLBACK_CALORIE_TABLE,
    NutritionToolError,
    get_calories_per_100g,
)

# ──────────────────────────────────────────────────────────────
# 1. Pydantic 输入 Schema：让 LLM 通过 function-calling 严格传参
# ──────────────────────────────────────────────────────────────


class NutritionCalorieLookupInput(BaseModel):
    """查询单个食物每 100g 热量的输入参数。

    作为 ``@tool(args_schema=...)`` 的结构化入参，确保 LLM 在发起
    Tool Calling 请求时不会遗漏 ``food_name`` 字段、也不会传类型错误的值。
    """

    food_name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description=(
            "要查询热量的食物名称（中文），例如：汉堡、米饭、鸡胸肉。"
            "必须是单个食物，不要同时传入多个食物名。"
        ),
    )


# ──────────────────────────────────────────────────────────────
# 2. Tool 输出结构（Pydantic）：统一结构化返回，便于下游汇总
# ──────────────────────────────────────────────────────────────


class NutritionCalorieLookupOutput(BaseModel):
    """工具执行后的结构化输出。"""

    food_name: str = Field(description="查询的食物名称（与输入一致）")
    kcal_per_100g: float = Field(description="每 100g 的热量（千卡）")
    source: str = Field(
        description=(
            "数据来源："
            "remote=远程营养数据库；"
            "fallback=本地硬编码粗略热量表；"
            "default=未知食物默认估算值"
        ),
    )
    message: str = Field(
        default="",
        description="可选的附加提示信息（如降级原因说明），默认空字符串",
    )


# ──────────────────────────────────────────────────────────────
# 3. 带重试的远程查询内层函数（Tenacity 装饰器只能作用于普通函数）
# ──────────────────────────────────────────────────────────────


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
    retry=retry_if_exception_type(NutritionToolError),
    reraise=True,
)
def _call_remote_calorie_tool(food_name: str) -> float:
    """带重试地调用远程营养数据库查询（最多 3 次指数退避）。

    注意：
        Tenacity ``@retry`` 装饰 **普通函数** 比装饰 ``@tool`` 函数更稳定，
        因此把重试逻辑独立于此私有函数，外层 Tool 在捕获最终异常后做降级。
    """
    return get_calories_per_100g(food_name)


# ──────────────────────────────────────────────────────────────
# 4. 核心 Tool：使用 @tool 装饰器注册为 LangChain 标准 Tool
# ──────────────────────────────────────────────────────────────


@tool(
    "nutrition_calorie_lookup",
    description=(
        "查询单个食物每 100 克所含的热量（单位：千卡/kcal）。"
        "适用于饮食记录、热量计算、营养评估等场景。"
        "输入必须是单个中文食物名称，返回结构化结果含热量值与数据来源标识。"
    ),
    args_schema=NutritionCalorieLookupInput,
)
def nutrition_calorie_lookup_tool(food_name: str) -> dict[str, Any]:
    """查询单个食物每 100g 的热量（标准 LangChain Tool）。

    执行策略（生产级兜底）：
        1. 优先调用 **远程营养数据库**（Tenacity 重试 3 次，指数退避）。
        2. 远程仍失败 → **降级** 到本地硬编码的 ``FALLBACK_CALORIE_TABLE``。
        3. 本地表也未收录 → 返回 ``DEFAULT_UNKNOWN_KCAL_PER_100G`` 作为默认估算。

    Args:
        food_name: 单个食物的中文名称，例如 "汉堡"、"米饭"。

    Returns:
        dict: 符合 :class:`NutritionCalorieLookupOutput` 结构的字典，
        含 ``food_name``、``kcal_per_100g``、``source``、``message`` 四字段。
    """
    name = food_name.strip()
    if not name:
        # 极端兜底：LLM 传了空字符串时，返回默认值并附带提示
        return NutritionCalorieLookupOutput(
            food_name=food_name,
            kcal_per_100g=DEFAULT_UNKNOWN_KCAL_PER_100G,
            source="default",
            message="食物名称为空，已使用默认估算值",
        ).model_dump()

    # ── 策略 1：远程数据库（带重试） ──────────────────────────────
    try:
        kcal_per_100g = _call_remote_calorie_tool(name)
        logger.debug(
            "NutritionCalorieTool[remote]: '{}' = {:.1f} kcal/100g",
            name,
            kcal_per_100g,
        )
        return NutritionCalorieLookupOutput(
            food_name=name,
            kcal_per_100g=kcal_per_100g,
            source="remote",
        ).model_dump()
    except NutritionToolError as exc:
        logger.warning(
            "NutritionCalorieTool: 远程查询重试失败，准备降级。food='{}', exc={}",
            name,
            exc,
        )

    # ── 策略 2：本地硬编码 Fallback 表 ─────────────────────────────
    fallback_value = FALLBACK_CALORIE_TABLE.get(name)
    if fallback_value is not None:
        logger.debug(
            "NutritionCalorieTool[fallback]: '{}' = {:.1f} kcal/100g",
            name,
            fallback_value,
        )
        return NutritionCalorieLookupOutput(
            food_name=name,
            kcal_per_100g=fallback_value,
            source="fallback",
            message=(
                "远程营养数据库暂时不可用或未收录该食物，"
                "已使用本地粗略热量表作为降级结果"
            ),
        ).model_dump()

    # ── 策略 3：未知食物默认估算（终极兜底） ───────────────────────
    logger.warning(
        "NutritionCalorieTool[default]: '{}' 本地表也未收录，使用默认估算 {} kcal/100g",
        name,
        DEFAULT_UNKNOWN_KCAL_PER_100G,
    )
    return NutritionCalorieLookupOutput(
        food_name=name,
        kcal_per_100g=DEFAULT_UNKNOWN_KCAL_PER_100G,
        source="default",
        message=(
            "远程与本地均未收录该食物，已使用通用默认估算值（150 kcal/100g），"
            "建议用户补充更准确的食物名称"
        ),
    ).model_dump()


# ──────────────────────────────────────────────────────────────
# 5. 导出便捷引用：供 Workflow 节点 ``import`` 使用
# ──────────────────────────────────────────────────────────────

# 工具单例（langchain bind_tools 接受的是 Tool 实例列表）
NUTRITION_CALORIE_TOOLS = [nutrition_calorie_lookup_tool]

__all__ = [
    "NutritionCalorieLookupInput",
    "NutritionCalorieLookupOutput",
    "nutrition_calorie_lookup_tool",
    "NUTRITION_CALORIE_TOOLS",
]
