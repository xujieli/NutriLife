"""意图路由的输出模型。

针对 gemma-4-12b 小模型设计：用强类型枚举替代自由 JSON 输出，
配合 ``with_structured_output`` 在推理服务的 function-calling 层强制约束取值。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Intent(StrEnum):
    """用户意图枚举（四种互斥路由目标）。

    任务类意图按「任务复杂度」进一步区分：
        - ``WORKFLOW_TASK``：结构化、步骤已知的饮食记录任务，走
          Plan-and-Execute（P&E）固定流程。
        - ``REACT_TASK``：开放式、需要动态推理与多工具协作的复杂营养任务，
          走 ReAct 循环。
    """

    RAG_QUERY = "RAG_QUERY"
    WORKFLOW_TASK = "WORKFLOW_TASK"  # P&E（Plan-and-Execute）流程
    REACT_TASK = "REACT_TASK"  # ReAct 流程
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
