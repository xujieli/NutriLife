"""营养与饮食记录相关数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FoodItem(BaseModel):
    """单个食物条目。"""

    name: str = Field(description="食物名称，如：汉堡、薯条")
    grams: float = Field(
        default=100.0,
        gt=0,
        description="估算重量（克），无法确定时给常见分量",
    )


class FoodExtraction(BaseModel):
    """从用户输入中提取的食物列表（Workflow 提取节点的结构化输出）。"""

    foods: list[FoodItem] = Field(
        default_factory=list,
        description="提取到的食物条目；未识别到食物时为空列表",
    )
