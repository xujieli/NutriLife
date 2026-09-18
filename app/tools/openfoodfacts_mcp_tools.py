"""基于 Open Food Facts MCP 服务的营养热量查询工具。

本模块替代原先的 ``nutrition_tools.py`` / ``nutrition_calorie_tool.py``：

    - ``get_calories_per_100g`` 通过 MCP 的 ``searchProducts`` 与
      ``getProductByBarcode`` 工具查询真实食品数据；
    - ``nutrition_calorie_lookup_tool`` 仍保持 LangChain Tool 接口，
      供 Workflow 子图通过 ``bind_tools`` 使用；
    - 不再依赖本地硬编码热量表，仅在 MCP 服务不可用或未收录时给出
      ``DEFAULT_UNKNOWN_KCAL_PER_100G`` 默认估算值。
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import StructuredTool
from loguru import logger
from pydantic import BaseModel, Field

from app.tools.mcp_client import OpenFoodFactsMCPError, call_tool_json


class NutritionToolError(RuntimeError):
    """营养工具异常：MCP 服务不可用或食物未收录。"""


# 当 Open Food Facts 未收录食物或 MCP 服务不可用时，使用的通用估算值。
DEFAULT_UNKNOWN_KCAL_PER_100G = 150.0


class NutritionCalorieLookupInput(BaseModel):
    """查询单个食物每 100g 热量的输入参数。"""

    food_name: str = Field(
        ...,
        min_length=1,
        max_length=80,
        description=(
            "要查询热量的食物名称（中文或英文），例如：汉堡、米饭、鸡胸肉、"
            "Nutella。必须是单个食物，不要同时传入多个食物名。"
        ),
    )


class NutritionCalorieLookupOutput(BaseModel):
    """工具执行后的结构化输出。"""

    food_name: str = Field(description="查询的食物名称（与输入一致）")
    kcal_per_100g: float = Field(description="每 100g 的热量（千卡）")
    source: str = Field(
        description=(
            "数据来源：mcp=Open Food Facts MCP 查询结果；"
            "default=未知食物或服务不可用时的默认估算值"
        ),
    )
    message: str = Field(
        default="",
        description="可选的附加提示信息（如降级原因说明），默认空字符串",
    )


def _to_positive_float(value: Any) -> float | None:
    """把 MCP 返回的营养值安全转换为正浮点数。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed > 0 else None


def _extract_kcal_per_100g(product: Any) -> float | None:
    """从 ``getProductByBarcode`` 返回的产品数据中提取每 100g 千卡。"""
    if not isinstance(product, dict):
        return None

    nutrition_facts = product.get("nutritionFacts")
    if isinstance(nutrition_facts, dict):
        for key in ("energy", "energy-kcal_100g", "energy-kcal"):
            value = _to_positive_float(nutrition_facts.get(key))
            if value is not None:
                return value

    for key in ("energy-kcal_100g", "energy_100g", "energy-kcal", "energy"):
        value = _to_positive_float(product.get(key))
        if value is not None:
            return value

    nutriments = product.get("nutriments")
    if isinstance(nutriments, dict):
        for key in ("energy-kcal_100g", "energy_100g", "energy-kcal", "energy"):
            value = _to_positive_float(nutriments.get(key))
            if value is not None:
                return value

    return None


def _barcode_from_search_product(product: Any) -> str | None:
    """从 searchProducts 的单个结果中提取条码。"""
    if not isinstance(product, dict):
        return None
    for key in ("barcode", "code", "id"):
        value = product.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


async def get_calories_per_100g(food_name: str) -> float:
    """通过 Open Food Facts MCP 查询食物每 100g 热量。

    Args:
        food_name: 食物名称。

    Returns:
        float: 每 100g 热量（千卡）。

    Raises:
        NutritionToolError: MCP 服务不可用、搜索失败或未找到可解析的营养数据。
    """
    name = food_name.strip()
    if not name:
        raise NutritionToolError("食物名称不能为空")

    try:
        search_result = await call_tool_json(
            "searchProducts",
            {"query": name, "page": 1, "pageSize": 3},
        )
    except OpenFoodFactsMCPError as exc:
        raise NutritionToolError(f"Open Food Facts MCP 搜索失败: {exc}") from exc

    if not isinstance(search_result, dict):
        raise NutritionToolError(
            f"Open Food Facts MCP searchProducts 返回异常结构: {search_result!r}"
        )

    products = search_result.get("products")
    if not isinstance(products, list) or not products:
        raise NutritionToolError(f"Open Food Facts 未收录食物: {name}")

    last_error: Exception | None = None
    for product in products[:3]:
        barcode = _barcode_from_search_product(product)
        if not barcode:
            continue

        try:
            detail = await call_tool_json(
                "getProductByBarcode",
                {"barcode": barcode},
            )
        except OpenFoodFactsMCPError as exc:
            last_error = exc
            continue

        kcal = _extract_kcal_per_100g(detail)
        if kcal is not None:
            logger.debug(
                "OpenFoodFacts MCP 命中: '{}' -> barcode={} -> {:.1f} kcal/100g",
                name,
                barcode,
                kcal,
            )
            return kcal

    if last_error is not None:
        raise NutritionToolError(
            f"Open Food Facts MCP 产品详情查询失败: {last_error}"
        ) from last_error
    raise NutritionToolError(f"Open Food Facts 未找到食物 '{name}' 的热量数据")


def get_fallback_calories(_food_name: str) -> float:
    """返回 MCP 不可用或未收录时的默认热量估算值。

    保留该函数是为了兼容 Workflow 的终极兜底路径；不再维护本地硬编码表。
    """
    return DEFAULT_UNKNOWN_KCAL_PER_100G


async def _call_remote_calorie_tool_with_retry(food_name: str) -> float:
    """带简单指数退避地调用 MCP 热量查询。"""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return await get_calories_per_100g(food_name)
        except NutritionToolError as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(0.5 * (2**attempt))

    if last_error is None:
        raise NutritionToolError(f"查询食物 '{food_name}' 失败")
    raise last_error


async def _lookup_calories(food_name: str) -> dict[str, Any]:
    """LangChain 工具的异步执行体。"""
    name = food_name.strip()
    if not name:
        return NutritionCalorieLookupOutput(
            food_name=food_name,
            kcal_per_100g=DEFAULT_UNKNOWN_KCAL_PER_100G,
            source="default",
            message="食物名称为空，已使用默认估算值",
        ).model_dump()

    try:
        kcal_per_100g = await _call_remote_calorie_tool_with_retry(name)
        return NutritionCalorieLookupOutput(
            food_name=name,
            kcal_per_100g=kcal_per_100g,
            source="mcp",
        ).model_dump()
    except NutritionToolError as exc:
        logger.warning(
            "OpenFoodFacts MCP 查询失败，使用默认估算值。food='{}', exc={}",
            name,
            exc,
        )
        return NutritionCalorieLookupOutput(
            food_name=name,
            kcal_per_100g=DEFAULT_UNKNOWN_KCAL_PER_100G,
            source="default",
            message=(
                "Open Food Facts MCP 服务不可用或未收录该食物，已使用通用默认估算值"
            ),
        ).model_dump()


nutrition_calorie_lookup_tool = StructuredTool.from_function(
    name="nutrition_calorie_lookup",
    description=(
        "查询单个食物每 100 克所含的热量（单位：千卡/kcal）。"
        "数据来源为 Open Food Facts MCP 服务。"
        "输入必须是单个食物名称（中文或英文），返回结构化结果，"
        "包含热量值与数据来源标识。"
    ),
    args_schema=NutritionCalorieLookupInput,
    coroutine=_lookup_calories,
)

NUTRITION_CALORIE_TOOLS = [nutrition_calorie_lookup_tool]

__all__ = [
    "DEFAULT_UNKNOWN_KCAL_PER_100G",
    "NUTRITION_CALORIE_TOOLS",
    "NutritionCalorieLookupInput",
    "NutritionCalorieLookupOutput",
    "NutritionToolError",
    "get_calories_per_100g",
    "get_fallback_calories",
    "nutrition_calorie_lookup_tool",
]
