"""LangGraph 全局共享状态（AgentState）定义。"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Agent 全图共享状态。

    使用 ``TypedDict(total=False)`` 使所有字段可选，节点按需写入。
    ``messages`` 通过 ``add_messages`` reducer 自动追加历史消息，其余字段
    由各节点返回的字典做浅合并（同名 key 整体覆盖）。

    Attributes:
        messages: 完整对话历史（add_messages reducer 自动追加）。
        current_intent: 路由器判定的意图（RAG_QUERY / WORKFLOW_TASK / GENERAL_CHAT）。
        router_confidence: 路由器输出的置信度。
        rag_query: RAG 分支中经过上下文重写后的独立检索查询。
        rag_context: RAG 检索到的上下文（引用来源摘要）。
        sources: RAG 检索到的结构化引用来源列表（source/snippet/score），供前端展示。
        workflow_data: Workflow 子图内部流转的结构化数据（食物列表、总热量等）。
        workflow_step: Workflow 子图中当前已完成、待 replan 验证的步骤名。
        workflow_replans: 当前步骤的 replan（重做）次数，用于封顶防止死循环。
        optimized_messages: 经「滑动窗口 + 摘要压缩」处理后、传入最终生成节点的消息。
        final_answer: 最终回答文本。
        error: 错误信息（用于可观测性与降级诊断）。
    """

    messages: Annotated[list[BaseMessage], add_messages]
    current_intent: str
    router_confidence: float
    rag_query: str
    rag_context: str
    sources: list[dict[str, Any]]
    workflow_data: dict[str, Any]
    workflow_step: str
    workflow_replans: int
    optimized_messages: list[BaseMessage]
    final_answer: str
    error: str


def get_latest_user_text(state: AgentState) -> str:
    """从状态中提取最新一条用户消息的纯文本。

    兼容 ``HumanMessage.content`` 为 ``str`` 或 ``list[dict]``（多模态）两种形态，
    优先取最后一条人类消息。

    Args:
        state: Agent 全局状态。

    Returns:
        str: 最新用户消息文本；无人类消息时返回空字符串。
    """
    for msg in reversed(state.get("messages", [])):
        if not isinstance(msg, HumanMessage):
            continue
        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return str(part.get("text", ""))
    return ""
