"""
NutriLife RAG 管道门面模块（API 层入口，与新"数据/编排分离"架构对齐）。

重构后的职责分配：
    - **门面层（本模块）**：保持对 FastAPI 路由层暴露的 ``ask`` / ``aask`` 不变；
      做"轻量编排"（空 Context 拒答 → 合成 → 不确定性校验 → 组装 RAGResponse）
      因为 API 层可能不经 LangGraph 直接调用本门面，不具备 LangGraph 节点编排环境。
    - **数据层（app/rag/query_engine.py）**：提供共享常量 + 纯函数 + 懒加载单例组件
      （``get_rag_components`` 返回 retriever + synthesizer，本门面直接 consume）。
    - **编排层（app/agents/graph.py）**：当 RAG 通过主 Agent 图走时，LangGraph 接管
      全部编排，不再经过本门面模块。

关于"为什么门面里仍然有编排逻辑"：
    这是一个 **刻意的双入口设计**：
        (1) 主入口：主 Agent 图（LangGraph 编排）——用户聊天、多轮对话、意图路由。
        (2) 独立入口：本门面（轻量编排）——单独的 RAG API、脚本测试、离线评估。
    两处编排最终都消费同一个数据层（常量/Prompt/校验/组件完全共享），
    从而在单一真相源前提下满足两种运行模式。
"""

from __future__ import annotations

from loguru import logger

from app.core.config import RAGSettings
from app.rag.query_engine import (
    FALLBACK_RESPONSE,
    RAGComponents,
    RAGResponse,
    extract_source_references,
    get_rag_components,
    is_fallback_response,
)


class NutriLifeRAGPipeline:
    """
    RAG 管道高层封装（API 层门面）。

    在 FastAPI 应用启动时初始化，通过 DI 注入到路由处理函数。

    **新架构**：构造时获取 ``RAGComponents``（retriever + synthesizer 单例），
    每次 ``ask`` 时独立消费组件；不再依赖旧的 "NutriLifeQueryEngine 大而全包"
    风格，而是遵循 "LlamaIndex 数据组件 + 门面内轻量编排" 分层。

    Example::

        pipeline = NutriLifeRAGPipeline()
        result = pipeline.ask("维生素D缺乏有什么症状？")
        print(result.answer)
        print(result.is_fallback)
        for src in result.sources:
            print(f"  [{src.source}] {src.text_snippet[:60]}")
    """

    def __init__(self, config: RAGSettings | None = None) -> None:
        # ── 新架构：直接消费 LlamaIndex 数据组件单例 ─────────────────
        self._components: RAGComponents = get_rag_components(
            retriever_config=config,
        )
        logger.info(
            "NutriLifeRAGPipeline 初始化完成（retriever={} , synthesizer={}）",
            type(self._components.retriever).__name__,
            type(self._components.synthesizer).__name__,
        )

    # ----------------------------------------------------------------
    # 轻量编排（独立入口场景）：同 LangGraph rag_retrieve → memory_optimize →
    # rag_generate 的逻辑等价，只是不经过 StateGraph + AgentState。
    # ----------------------------------------------------------------

    def ask(self, question: str) -> RAGResponse:
        """同步 RAG 查询接口（独立入口，门面级轻量编排）。"""
        logger.info("NutriLifeRAGPipeline.ask: '{}'", question[:80])
        retriever = self._components.retriever
        synthesizer = self._components.synthesizer

        # ── Step 1（数据层）：混合检索 ──────────────────────────────
        nodes = retriever.retrieve(question)
        raw_count = len(nodes)

        # ── Step 2（编排决策）：空 Context → 直接拒答 ──────────────
        if raw_count == 0:
            logger.info("空 Context，直接拒答")
            return RAGResponse(
                answer=FALLBACK_RESPONSE,
                is_fallback=True,
                sources=[],
                query=question,
                raw_context_count=0,
            )

        # ── Step 3（数据层）：调用 LlamaIndex Synthesizer 合成回答 ──
        try:
            answer_text = synthesizer.synthesize(question, nodes)
        except Exception as exc:  # noqa: BLE001
            logger.error("RAG Synthesizer 合成失败: {}", exc)
            return RAGResponse(
                answer=FALLBACK_RESPONSE,
                is_fallback=True,
                sources=extract_source_references(nodes),
                query=question,
                raw_context_count=raw_count,
            )

        # ── Step 4（编排决策 + 数据层校验）：不确定性 → 拒答 ────────
        is_fallback = FALLBACK_RESPONSE in answer_text or is_fallback_response(
            answer_text
        )
        if is_fallback:
            logger.info("检测到拒答关键词，返回标准拒答话术")
            answer_text = FALLBACK_RESPONSE

        # ── Step 5（数据层）：提取结构化引用溯源 ────────────────────
        sources = extract_source_references(nodes)

        logger.info(
            "NutriLifeRAGPipeline.ask 完成: is_fallback={}, sources={}, answer_len={}",
            is_fallback,
            len(sources),
            len(answer_text),
        )
        return RAGResponse(
            answer=answer_text,
            is_fallback=is_fallback,
            sources=sources if not is_fallback else [],
            query=question,
            raw_context_count=raw_count,
        )

    async def aask(self, question: str) -> RAGResponse:
        """异步 RAG 查询接口（供 FastAPI async 路由使用）。

        内部通过 ``asyncio.to_thread`` 把同步 ask() 丢到线程池，
        避免阻塞事件循环；当 LangGraph 接管 RAG 时不再使用本函数。
        """
        import asyncio

        return await asyncio.to_thread(self.ask, question)
