"""意图路由的输出模型。

针对 gemma-4-12b 小模型设计：用强类型枚举替代自由 JSON 输出，
配合 ``with_structured_output`` 在推理服务的 function-calling 层强制约束取值。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Intent(StrEnum):
    """用户意图枚举（三种互斥路由目标）。"""

    RAG_QUERY = "RAG_QUERY"
    WORKFLOW_TASK = "WORKFLOW_TASK"
    GENERAL_CHAT = "GENERAL_CHAT"


class RouterOutput(BaseModel):
    """Router 节点的结构化输出。

    Attributes:
        intent: 路由目标意图。
        confidence: 分类置信度（0~1）。小模型常高估置信度，故路由侧会
            配合阈值做二次兜底——低于阈值一律回退 GENERAL_CHAT。
        reasoning: 分类理由（一句话，便于可观测性/调试）。
    """

    intent: Intent = Field(description="用户意图分类")
    confidence: float = Field(ge=0.0, le=1.0, description="分类置信度（0~1）")
    reasoning: str = Field(default="", description="分类理由（一句话）")
