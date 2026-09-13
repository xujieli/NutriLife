"""营养查询工具（供 Workflow 的 calculate_calories_node 调用）。

设计说明：
    - ``get_calories_per_100g`` 模拟"远程营养数据库"查询，食物未收录时抛出
      :class:`NutritionToolError`，由上层节点配合 Tenacity 做重试。
    - ``FALLBACK_CALORIE_TABLE`` 是本地硬编码的粗略热量表，作为远程失败后的
      降级兜底（满足"生产兜底"要求：远程依赖不可用时仍能给出粗略估算）。
"""

from __future__ import annotations


class NutritionToolError(RuntimeError):
    """营养工具异常：远程服务不可用或食物未收录。"""


# 模拟"远程营养数据库"（每 100g 热量，单位：千卡）。
# 真实场景中这里替换为外部营养 API 调用。
_REMOTE_CALORIE_DB: dict[str, float] = {
    "汉堡": 254.0,
    "薯条": 312.0,
    "可乐": 42.0,
    "米饭": 116.0,
    "鸡胸肉": 165.0,
    "鸡蛋": 144.0,
    "苹果": 52.0,
    "西兰花": 34.0,
    "牛奶": 54.0,
}

# 本地硬编码粗略热量表（降级兜底，每 100g 千卡）。覆盖范围比远程表更广，
# 保证常见食物在远程失败时仍能给出近似值。
FALLBACK_CALORIE_TABLE: dict[str, float] = {
    "汉堡": 250.0,
    "薯条": 300.0,
    "可乐": 40.0,
    "米饭": 116.0,
    "鸡胸肉": 165.0,
    "鸡蛋": 143.0,
    "苹果": 52.0,
    "西兰花": 34.0,
    "牛奶": 54.0,
    "披萨": 266.0,
    "面条": 137.0,
    "香蕉": 89.0,
    "三明治": 250.0,
    "沙拉": 120.0,
}

# 未知食物的默认热量估算（每 100g 千卡）。
DEFAULT_UNKNOWN_KCAL_PER_100G = 150.0


def get_calories_per_100g(food_name: str) -> float:
    """查询食物每 100g 热量（模拟调用远程营养数据库 Tool）。

    Args:
        food_name: 食物名称。

    Returns:
        float: 每 100g 热量（千卡）。

    Raises:
        NutritionToolError: 食物未被远程数据库收录（视作远程服务失败）。
    """
    value = _REMOTE_CALORIE_DB.get(food_name)
    if value is None:
        raise NutritionToolError(f"远程数据库未收录食物: {food_name}")
    return value


def get_fallback_calories(food_name: str) -> float:
    """本地粗略热量兜底：优先查硬编码表，未知食物返回默认估算值。

    Args:
        food_name: 食物名称。

    Returns:
        float: 每 100g 粗略热量（千卡）。
    """
    return FALLBACK_CALORIE_TABLE.get(food_name, DEFAULT_UNKNOWN_KCAL_PER_100G)
