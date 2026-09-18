"""基于 NutriLife RAG 知识库的营养知识检索工具。

把 LlamaIndex 检索器包装成 LangChain Tool，供 ReAct agent 在需要查证
营养学 / 医学知识时动态调用，与 ``nutrition_calorie_lookup_tool``
（热量查询）共同构成 ReAct 的工具集。
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from loguru import logger
from pydantic import BaseModel, Field

from app.rag.langchain_bridge import build_context_str_from_documents
from app.rag.query_engine import get_langchain_retriever


class NutritionKnowledgeSearchInput(BaseModel):
    """营养知识检索工具的输入参数。"""

    query: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="需要检索的营养学/医学知识问题，尽量具体、独立完整",
    )


async def _search_knowledge(query: str) -> str:
    """检索知识库并返回拼接后的参考资料文本。"""
    query = query.strip()
    if not query:
        return "检索查询不能为空。"

    try:
        retriever = get_langchain_retriever()
        documents = await retriever.ainvoke(query)
        if not documents:
            return "知识库中没有检索到与该问题相关的信息。"

        context = build_context_str_from_documents(documents)
        if not context.strip():
            return "知识库中没有检索到与该问题相关的信息。"

        logger.info("NutritionKnowledgeSearch 命中 {} 个片段", len(documents))
        return context
    except Exception as exc:  # noqa: BLE001
        logger.error("NutritionKnowledgeSearch 检索失败: {}", exc)
        return f"知识库检索暂时不可用，请基于已有信息谨慎回答。原因：{exc}"


nutrition_knowledge_search_tool = StructuredTool.from_function(
    name="nutrition_knowledge_search",
    description=(
        "检索 NutriLife 营养学/医学知识库，返回与问题相关的专业参考资料片段。"
        "当需要查证营养知识、疾病饮食禁忌、营养素推荐摄入量等事实时调用。"
    ),
    args_schema=NutritionKnowledgeSearchInput,
    coroutine=_search_knowledge,
)

NUTRITION_KNOWLEDGE_TOOLS = [nutrition_knowledge_search_tool]

__all__ = [
    "NUTRITION_KNOWLEDGE_TOOLS",
    "NutritionKnowledgeSearchInput",
    "nutrition_knowledge_search_tool",
]
