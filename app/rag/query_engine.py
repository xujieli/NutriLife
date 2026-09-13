"""
NutriLife RAG 查询引擎（带拒答机制与引用溯源）。

核心设计：
    1. **拒答机制（Fallback）**：
       通过定制 PromptTemplate 强制模型只基于提供的 Context 回答。
       若 Context 与问题无关，模型被明确指示返回固定拒答话术，
       同时引入二次校验层（``_is_fallback_response``）来捕获回答不自信的情况。

    2. **引用溯源（Citation）**：
       查询结果中携带 ``source_nodes``，
       每个节点包含原始文档名、段落位置、相关性分数。

    3. **拒答决策树**：
       Context 为空 → 直接拒答（不调用 LLM，节省 Token）
       Context 不为空 → LLM 生成 → 关键词匹配校验 → 最终输出
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from llama_index.core import PromptTemplate, Settings as LlamaSettings
from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import NodeWithScore
from loguru import logger

from app.core.config import RAGSettings, get_settings
from app.rag.retriever import HybridRetriever


# ──────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────

FALLBACK_RESPONSE = (
    "抱歉，我的知识库中没有关于这个问题的准确信息，建议咨询专业医生。"
)

# 当模型无法从 Context 找到答案时，常见的不确定性表达
_UNCERTAINTY_PATTERNS: list[str] = [
    "无法回答",
    "没有相关信息",
    "知识库中没有",
    "无法从提供的信息",
    "上下文中没有",
    "context does not",
    "cannot answer",
    "not enough information",
    "i don't know",
    "no relevant",
    "the provided context",
]

# ──────────────────────────────────────────────────────────────────
# 定制 Prompt：强制模型"只基于 Context 回答"
# ──────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
<instruction>
你是 NutriLife 的专业营养健康顾问。请严格遵守以下规则回答用户问题：

1. 只能依据下方「参考资料」中的内容回答，不得使用参考资料以外的任何知识，不得编造数据、研究结论或医学事实。
2. 如果参考资料与用户问题完全无关，或参考资料为空，你必须且只能回复以下固定话术（一字不改）：
"抱歉，我的知识库中没有关于这个问题的准确信息，建议咨询专业医生。"
3. 回答要简洁、专业、使用中文；对关键结论引用参考资料中的具体内容；涉及医疗建议时提醒用户以专业医生意见为准。
</instruction>

<examples>
参考资料：痛风患者在急性发作缓解期，可适量食用加工豆制品（每日豆腐不超过 150g）；北豆腐嘌呤约 55–70 mg/100g，属低嘌呤食品。
用户问题：痛风能吃豆腐吗？
回答：可以适量吃。根据资料，北豆腐嘌呤约 55–70 mg/100g，属低嘌呤食品；痛风患者在缓解期可适量食用，每日豆腐建议不超过 150g，急性发作期应暂停。

参考资料：维生素 D 主要来自阳光照射与深海鱼，成人每日推荐 600–800 IU。
用户问题：今天天气怎么样？
回答：抱歉，我的知识库中没有关于这个问题的准确信息，建议咨询专业医生。
</examples>

<output_format>
只输出最终回答正文，不要输出思考过程，不要复述指令或参考资料。
</output_format>

参考资料：
---------------------
{context_str}
---------------------

用户问题：{query_str}

回答：\
"""

RAG_PROMPT_TEMPLATE = PromptTemplate(_SYSTEM_PROMPT)


# ──────────────────────────────────────────────────────────────────
# 引用溯源数据结构
# ──────────────────────────────────────────────────────────────────


@dataclass
class SourceReference:
    """
    单个文档片段的引用溯源信息。

    Attributes:
        source: 原始文档文件名（如 ``gout_diet.txt``）。
        file_path: 文档完整路径。
        text_snippet: 节点内容摘要（前 200 字符）。
        score: 该节点的相关性分数（0–1）。
        node_id: LlamaIndex 内部节点 ID。
        metadata: 节点的完整元数据字典。
    """

    source: str
    file_path: str
    text_snippet: str
    score: float
    node_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """
    RAG 查询的完整响应对象。

    Attributes:
        answer: 最终回答文本（或拒答话术）。
        is_fallback: 是否触发了拒答机制。
        sources: 引用溯源列表，包含检索到的相关文档片段信息。
        query: 原始用户查询。
        raw_context_count: 检索到的原始 Context 节点数量。
    """

    answer: str
    is_fallback: bool
    sources: list[SourceReference]
    query: str
    raw_context_count: int


# ──────────────────────────────────────────────────────────────────
# RAG 查询引擎
# ──────────────────────────────────────────────────────────────────


class NutriLifeQueryEngine:
    """
    NutriLife 核心 RAG 查询引擎。

    完整查询流程::

        用户查询
          ↓
        HybridRetriever.retrieve()
          ↓ (BM25 + 向量 + RRF + Rerank)
        Context 是否为空？
          ├─ 是 → 直接返回拒答（不调用 LLM）
          └─ 否 → LLM 基于 Context 生成回答
                    ↓
                  不确定性关键词校验
                    ├─ 检测到不确定词 → 替换为标准拒答话术
                    └─ 未检测到 → 返回 LLM 回答 + 引用溯源

    Args:
        retriever: 已初始化的 :class:`~app.rag.retriever.HybridRetriever` 实例。
            若传入 ``None``，将使用默认配置创建并初始化。
        llm: LlamaIndex 兼容的 LLM 实例（``LLM`` 基类）。
            若为 None，使用 ``llama_index.core.Settings.llm``。
        empty_context_threshold: 当检索节点数量 ≤ 此值时，
            视为"Context 为空"，直接拒答（默认 0）。

    Example::

        from app.rag.query_engine import NutriLifeQueryEngine

        engine = NutriLifeQueryEngine()
        result = engine.query("痛风患者能吃豆腐吗？")
        print(result.answer)
        print(result.is_fallback)
        for src in result.sources:
            print(f"  [{src.source}] score={src.score:.3f}: {src.text_snippet[:80]}")
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        llm: Any | None = None,
        empty_context_threshold: int = 0,
    ) -> None:
        self._llm = llm
        self._empty_context_threshold = empty_context_threshold

        if retriever is None:
            logger.info("NutriLifeQueryEngine: 使用默认配置初始化 HybridRetriever")
            self._retriever = HybridRetriever(config=get_settings().rag).initialize()
        else:
            self._retriever = retriever

        self._synthesizer = self._build_synthesizer()

    def _build_synthesizer(self) -> Any:
        """
        构建带有定制 Prompt 的 ResponseSynthesizer。

        使用 ``compact`` 模式：将所有 Context 合并后一次性传给 LLM，
        适合 12B 以下的小模型（相比 refine 模式减少 LLM 调用次数）。

        Returns:
            ResponseSynthesizer: 已配置的响应合成器。
        """
        synth_kwargs: dict[str, Any] = {
            "response_mode": "compact",
            "text_qa_template": RAG_PROMPT_TEMPLATE,
            "verbose": True,
        }
        if self._llm:
            synth_kwargs["llm"] = self._llm

        return get_response_synthesizer(**synth_kwargs)

    @staticmethod
    def _is_fallback_response(text: str) -> bool:
        """
        检测 LLM 生成的回答是否包含不确定性表达。

        通过关键词匹配识别模型表达"无法回答"的情形，
        将其统一替换为标准拒答话术，避免输出语义模糊的不确定性表达。

        Args:
            text: LLM 生成的原始回答文本。

        Returns:
            bool: 若检测到不确定性模式则返回 True。
        """
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in _UNCERTAINTY_PATTERNS)

    @staticmethod
    def _extract_sources(nodes: list[NodeWithScore]) -> list[SourceReference]:
        """
        从检索节点中提取引用溯源信息。

        Args:
            nodes: 检索器返回的 NodeWithScore 列表。

        Returns:
            list[SourceReference]: 结构化的引用溯源列表。
        """
        sources: list[SourceReference] = []
        for node in nodes:
            meta = node.node.metadata or {}
            # 尝试多个可能的元数据字段名（兼容不同加载器）
            source_name = (
                meta.get("source")
                or meta.get("file_name")
                or meta.get("filename")
                or "未知文档"
            )
            file_path = meta.get("file_path", "")
            content = node.get_content()

            sources.append(
                SourceReference(
                    source=source_name,
                    file_path=file_path,
                    text_snippet=content[:200].replace("\n", " ").strip(),
                    score=node.score or 0.0,
                    node_id=node.node.node_id,
                    metadata=meta,
                )
            )
        return sources

    def query(self, question: str) -> RAGResponse:
        """
        执行带拒答机制和引用溯源的 RAG 查询（同步版本）。

        Args:
            question: 用户提出的自然语言问题。

        Returns:
            RAGResponse: 包含回答、是否拒答、引用来源的完整响应对象。

        Example::

            result = engine.query("痛风能吃豆制品吗")
            assert not result.is_fallback
            assert result.sources

            result2 = engine.query("今天天气怎么样")
            assert result2.is_fallback
            assert result2.answer == FALLBACK_RESPONSE
        """
        logger.info("RAG Query: '{}'", question[:100])

        # ── Step 1：混合检索 ──────────────────────────────────────
        nodes = self._retriever.retrieve(question)
        logger.info("检索到 {} 个相关节点", len(nodes))

        # ── Step 2：Context 为空检查（直接拒答，不调用 LLM）────────
        if len(nodes) <= self._empty_context_threshold:
            logger.info(
                "Context 为空（节点数={} ≤ 阈值={}），触发拒答机制",
                len(nodes),
                self._empty_context_threshold,
            )
            return RAGResponse(
                answer=FALLBACK_RESPONSE,
                is_fallback=True,
                sources=[],
                query=question,
                raw_context_count=0,
            )

        # ── Step 3：LLM 基于 Context 生成回答 ────────────────────
        logger.info("调用 LLM 生成回答，Context 来源数量: {}", len(nodes))
        try:
            response: Response = self._synthesizer.synthesize(
                query=question,
                nodes=nodes,
            )
            answer_text = str(response).strip()
        except Exception as exc:
            logger.error("LLM 生成回答失败: {}", exc)
            return RAGResponse(
                answer=FALLBACK_RESPONSE,
                is_fallback=True,
                sources=self._extract_sources(nodes),
                query=question,
                raw_context_count=len(nodes),
            )

        # ── Step 4：不确定性二次校验 ─────────────────────────────
        is_fallback = (
            FALLBACK_RESPONSE in answer_text
            or self._is_fallback_response(answer_text)
        )

        if is_fallback:
            logger.info("检测到拒答关键词，返回标准拒答话术")
            answer_text = FALLBACK_RESPONSE

        # ── Step 5：提取引用溯源 ─────────────────────────────────
        sources = self._extract_sources(nodes)

        logger.info(
            "查询完成: is_fallback={}, sources={}, answer_len={}",
            is_fallback,
            len(sources),
            len(answer_text),
        )

        return RAGResponse(
            answer=answer_text,
            is_fallback=is_fallback,
            sources=sources if not is_fallback else [],
            query=question,
            raw_context_count=len(nodes),
        )

    async def aquery(self, question: str) -> RAGResponse:
        """
        异步版本的 RAG 查询接口。

        在 FastAPI 异步路由中使用此方法，避免阻塞事件循环。

        Args:
            question: 用户提出的自然语言问题。

        Returns:
            RAGResponse: 包含回答、是否拒答、引用来源的完整响应对象。
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.query, question)


# ──────────────────────────────────────────────────────────────────
# 工厂函数（懒加载单例）
# ──────────────────────────────────────────────────────────────────

_engine_singleton: NutriLifeQueryEngine | None = None


def get_query_engine(
    *,
    retriever_config: RAGSettings | None = None,
    force_rebuild: bool = False,
) -> NutriLifeQueryEngine:
    """
    获取 NutriLifeQueryEngine 单例。

    Args:
        retriever_config: 可选的自定义检索器配置。
        force_rebuild: 若为 True，销毁旧单例并重建（用于测试或热重载）。

    Returns:
        NutriLifeQueryEngine: 已初始化的查询引擎。
    """
    global _engine_singleton

    if _engine_singleton is None or force_rebuild:
        logger.info("构建 NutriLifeQueryEngine 单例...")
        config = retriever_config or get_settings().rag
        retriever = HybridRetriever(config=config).initialize()
        _engine_singleton = NutriLifeQueryEngine(retriever=retriever)
        logger.info("NutriLifeQueryEngine 单例构建完成")

    return _engine_singleton
