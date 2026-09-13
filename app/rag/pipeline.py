"""
NutriLife RAG 管道门面模块。

提供统一的入口给 API 层调用，屏蔽底层 retriever / query_engine 的细节。
"""

from __future__ import annotations

from app.core.config import RAGSettings
from app.rag.query_engine import NutriLifeQueryEngine, RAGResponse, get_query_engine


class NutriLifeRAGPipeline:
    """
    RAG 管道高层封装。

    在 FastAPI 应用启动时初始化，通过 DI 注入到路由处理函数。

    Example::

        pipeline = NutriLifeRAGPipeline()
        result = pipeline.ask("维生素D缺乏有什么症状？")
        print(result.answer)
    """

    def __init__(self, config: RAGSettings | None = None) -> None:
        self._engine: NutriLifeQueryEngine = get_query_engine(
            retriever_config=config
        )

    def ask(self, question: str) -> RAGResponse:
        """同步查询接口。"""
        return self._engine.query(question)

    async def aask(self, question: str) -> RAGResponse:
        """异步查询接口（供 FastAPI async 路由使用）。"""
        return await self._engine.aquery(question)
