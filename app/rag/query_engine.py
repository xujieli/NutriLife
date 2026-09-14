"""
NutriLife RAG **数据组件层**（LlamaIndex 处理数据，LangGraph 编排逻辑）。

架构原则（重构后）：
    ╔══════════════════════════════════════════════════════════════════╗
    ║  LlamaIndex 负责「处理数据」（本模块）                           ║
    ║      ├─ Prompt Template / 拒答话术 / 不确定性模式 等共享常量     ║
    ║      ├─ SourceReference / RAGResponse 数据结构                  ║
    ║      ├─ NodeWithScore → SourceReference 纯函数提取              ║
    ║      ├─ 不确定性关键词检测（纯函数 _is_fallback_response）       ║
    ║      ├─ ResponseSynthesizer 构建 + 纯 synthesize() 调用         ║
    ║      └─ HybridRetriever / Synthesizer 单例工厂（懒加载）        ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  LangGraph 负责「编排逻辑」（app/agents/graph.py）              ║
    ║      ├─ Context 是否为空 → 是否直接拒答                         ║
    ║      ├─ LLM 生成后 → 不确定性二次校验                           ║
    ║      ├─ memory_optimize 节点的插入与条件路由                    ║
    ║      └─ State 各字段（rag_query / rag_context / sources）写入  ║
    ╚══════════════════════════════════════════════════════════════════╝

旧版 :class:`NutriLifeQueryEngine` 的 ``query()`` 方法里写了 5 步决策树编排（
空 Context 检查 → LLM 生成 → 不确定性校验 → 引用提取），这部分已剥离到
LangGraph 编排层。本模块 **只提供原子化的 LlamaIndex 数据组件**，由编排层按需要组合。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    get_response_synthesizer,
)
from llama_index.core.schema import NodeWithScore
from loguru import logger

from app.core.config import RAGSettings, get_settings
from app.rag.langchain_bridge import (
    DEFAULT_MAX_LENGTH_PER_NODE,
    LlamaIndexLangChainRetriever,
    build_context_str_from_documents,
    nodes_with_scores_to_documents,
)
from app.rag.retriever import HybridRetriever

# ──────────────────────────────────────────────────────────────────
# 1. 共享常量（LangGraph 编排层直接 import 复用，杜绝重复定义）
# ──────────────────────────────────────────────────────────────────

# 固定拒答话术（graph.py、pipeline.py 都直接 import 这个常量使用）
FALLBACK_RESPONSE = "抱歉，我的知识库中没有关于这个问题的准确信息，建议咨询专业医生。"

# 当模型无法从 Context 找到答案时，常见的不确定性表达。
# graph.py 不再本地复制一份，改为调用 :func:`_is_fallback_response`。
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
# 2. 定制 Prompt：强制模型"只基于 Context 回答"
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

# 供 LangGraph 编排层（graph.py）直接做字符串格式化时使用的原始模板字符串。
# 避免编排层再次手写 Prompt 造成两个真相源漂移。
RAG_SYSTEM_PROMPT_TEMPLATE_STR = _SYSTEM_PROMPT


# ──────────────────────────────────────────────────────────────────
# 3. 引用溯源与响应数据结构（LangGraph 编排层直接使用这些类型）
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

    用于 LangGraph 之外、需要直接返回给 API 层的场景（如
    :class:`~app.rag.pipeline.NutriLifeRAGPipeline`）。
    LangGraph 内部使用的是 ``AgentState.sources`` / ``rag_context`` 字段。

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
# 4. 纯函数工具（不含状态、不调用 LLM，可被任意层复用）
# ──────────────────────────────────────────────────────────────────


def is_fallback_response(text: str) -> bool:
    """检测 LLM 生成的回答是否包含不确定性表达（纯函数）。

    通过关键词匹配识别模型表达"无法回答"的情形。
    **graph.py 不再本地复制不确定性关键词列表，统一调用本函数。**

    Args:
        text: LLM 生成的原始回答文本。

    Returns:
        bool: 若检测到不确定性模式则返回 True。
    """
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in _UNCERTAINTY_PATTERNS)


def extract_source_references_from_documents(
    documents: Sequence[Document],
) -> list[SourceReference]:
    """从 LangChain ``Document`` 列表中提取引用溯源信息。

    LangGraph 检索节点在 bridge 层已经拿到 Document，因此这是主入口；
    旧的 :func:`extract_source_references` 仅作为 LlamaIndex 兼容入口。
    """
    sources: list[SourceReference] = []
    for document in documents:
        meta = dict(document.metadata or {})
        source_name = (
            meta.get("source")
            or meta.get("file_name")
            or meta.get("filename")
            or "未知文档"
        )
        content = document.page_content or ""
        score = meta.get("score", 0.0)

        sources.append(
            SourceReference(
                source=source_name,
                file_path=str(meta.get("file_path", "") or ""),
                text_snippet=content[:200].replace("\n", " ").strip(),
                score=float(score or 0.0),
                node_id=str(meta.get("node_id", "") or ""),
                metadata=meta,
            )
        )
    return sources


def extract_source_references(nodes: list[NodeWithScore]) -> list[SourceReference]:
    """从 LlamaIndex ``NodeWithScore`` 中提取引用溯源信息（兼容入口）。"""
    documents = nodes_with_scores_to_documents(nodes)
    return extract_source_references_from_documents(documents)


def source_references_to_state(
    refs: list[SourceReference],
) -> list[dict[str, Any]]:
    """把 :class:`SourceReference` 列表转为 ``AgentState.sources`` 的 dict 列表。

    由于 ``AgentState`` 是 TypedDict（不可持久化任意 dataclass），
    LangGraph 编排层在写入 state 时需要通过这个转换函数。
    """
    return [
        {
            "source": r.source,
            "snippet": r.text_snippet,
            "score": round(r.score, 4),
            "file_path": r.file_path,
            "node_id": r.node_id,
        }
        for r in refs
    ]


def build_context_str(
    nodes: list[NodeWithScore],
    *,
    max_len_per_node: int = DEFAULT_MAX_LENGTH_PER_NODE,
) -> str:
    """把 LlamaIndex 检索节点拼成 Prompt 上下文（兼容入口）。

    实际转换统一委托给 bridge 模块，确保 NodeWithScore 与 Document 两条
    路径的截断、清理规则完全一致。
    """
    documents = nodes_with_scores_to_documents(
        nodes,
        max_length_per_node=max_len_per_node,
    )
    return build_context_str_from_documents(documents)


# ──────────────────────────────────────────────────────────────────
# 5. ResponseSynthesizer 包装（纯 LlamaIndex 数据合成，无编排决策）
# ──────────────────────────────────────────────────────────────────


class NutriLifeSynthesizer:
    """基于 LlamaIndex ``ResponseSynthesizer`` 的**纯数据合成器**。

    职责：
        给定 query + nodes → 调用 LLM 生成回答字符串
    不做：
        Context 空检查（编排层在调用前判断）、不确定性二次校验（编排层在调用后判断）

    使用 ``compact`` 模式：将所有 Context 合并后一次性传给 LLM，
    适合 12B 以下小模型（相比 refine 模式减少 LLM 调用次数）。
    """

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm
        self._synthesizer: BaseSynthesizer = self._build_synthesizer()

    # -- 内部构建 ---------------------------------------------------

    def _build_synthesizer(self) -> BaseSynthesizer:
        synth_kwargs: dict[str, Any] = {
            "response_mode": "compact",
            "text_qa_template": RAG_PROMPT_TEMPLATE,
            "verbose": True,
        }
        if self._llm is not None:
            synth_kwargs["llm"] = self._llm
        return get_response_synthesizer(**synth_kwargs)

    # -- 公开 API：纯合成 ------------------------------------------

    def synthesize(self, query: str, nodes: list[NodeWithScore]) -> str:
        """调用 LlamaIndex ResponseSynthesizer 生成回答。

        Args:
            query: 用户问题（原样传入合成器）。
            nodes: 检索到的上下文节点（**调用方保证非空**）。

        Returns:
            str: 模型生成的回答字符串（可能包含不确定性表达，
            由调用方进一步使用 :func:`is_fallback_response` 校验）。

        Raises:
            Exception: LLM 调用异常会原样抛出，由编排层兜底。
        """
        response = self._synthesizer.synthesize(query=query, nodes=nodes)
        return str(response).strip()


# ──────────────────────────────────────────────────────────────────
# 6. RAG 数据组件的单例容器（懒加载，编排层通过 get_rag_components 获取）
# ──────────────────────────────────────────────────────────────────


@dataclass
class RAGComponents:
    """打包后返回的 LlamaIndex 数据组件集合。

    编排层拿到这个对象后，可以：
        1. 用 ``retriever.retrieve(query)`` 召回节点
        2. 用 ``extract_source_references(nodes)`` 转引用
        3. 用 ``synthesizer.synthesize(query, nodes)`` 合成回答
        4. 用 ``is_fallback_response(answer)`` 做二次校验
    """

    retriever: HybridRetriever
    synthesizer: NutriLifeSynthesizer


_retriever_singleton: HybridRetriever | None = None
_langchain_retriever_singleton: LlamaIndexLangChainRetriever | None = None
_synthesizer_singleton: NutriLifeSynthesizer | None = None


def get_retriever(
    config: RAGSettings | None = None,
    *,
    force_rebuild: bool = False,
) -> HybridRetriever:
    """获取懒加载初始化的 :class:`HybridRetriever` 单例。

    Args:
        config: 可选的自定义配置。
        force_rebuild: True 时销毁旧实例并重建（用于测试或热重载）。
    """
    global _retriever_singleton, _langchain_retriever_singleton
    if force_rebuild:
        # 底层 LlamaIndex retriever 重建后，已缓存的 LangChain 包装也必须失效。
        _langchain_retriever_singleton = None
    if _retriever_singleton is None or force_rebuild:
        cfg = config or get_settings().rag
        logger.info("构建 HybridRetriever 单例（RAG 数据层）...")
        _retriever_singleton = HybridRetriever(config=cfg).initialize()
        logger.info("HybridRetriever 单例构建完成")
    return _retriever_singleton


def get_langchain_retriever(
    config: RAGSettings | None = None,
    *,
    force_rebuild: bool = False,
) -> LlamaIndexLangChainRetriever:
    """获取 LangGraph 节点应使用的 LangChain 检索器单例。

    这是 LlamaIndex → LangChain 的正式桥接入口。LangGraph 编排层只依赖
    LangChain 的 ``invoke`` / ``ainvoke``，不直接接触 ``NodeWithScore``。
    """
    global _langchain_retriever_singleton
    if _langchain_retriever_singleton is None or force_rebuild:
        llama_index_retriever = get_retriever(
            config,
            force_rebuild=force_rebuild,
        )
        _langchain_retriever_singleton = LlamaIndexLangChainRetriever(
            llama_index_retriever=llama_index_retriever,
        )
        logger.info("LlamaIndexLangChainRetriever 单例构建完成")
    return _langchain_retriever_singleton


def get_synthesizer(
    llm: Any | None = None,
    *,
    force_rebuild: bool = False,
) -> NutriLifeSynthesizer:
    """获取懒加载的 :class:`NutriLifeSynthesizer` 单例。

    Args:
        llm: 可选的自定义 LlamaIndex LLM 实例。指定时会触发 force_rebuild。
        force_rebuild: True 时重建（用于测试或切换 LLM）。
    """
    global _synthesizer_singleton
    # 传入自定义 llm 时视为需要重建（否则单例可能绑定旧 LLM）
    if llm is not None:
        force_rebuild = True
    if _synthesizer_singleton is None or force_rebuild:
        logger.info("构建 NutriLifeSynthesizer 单例（RAG 数据层）...")
        _synthesizer_singleton = NutriLifeSynthesizer(llm=llm)
        logger.info("NutriLifeSynthesizer 单例构建完成")
    return _synthesizer_singleton


def get_rag_components(
    *,
    retriever_config: RAGSettings | None = None,
    llm: Any | None = None,
    force_rebuild: bool = False,
) -> RAGComponents:
    """获取打包好的 RAG 数据组件集合（编排层最常用入口）。"""
    return RAGComponents(
        retriever=get_retriever(retriever_config, force_rebuild=force_rebuild),
        synthesizer=get_synthesizer(llm, force_rebuild=force_rebuild),
    )


# ──────────────────────────────────────────────────────────────────
# 7. 向后兼容 API（避免外部调用方崩溃，仅供过渡使用）
# ──────────────────────────────────────────────────────────────────


class NutriLifeQueryEngine:
    """兼容旧 API 的过渡包装器——**内部实际走"数据组件 + 轻量编排"**。

    新代码请直接使用 :func:`get_rag_components` 或独立调用
    :func:`get_retriever` + :func:`get_synthesizer`。

    本类保留的唯一目的是：让仍在使用 ``NutriLifeQueryEngine(...)`` 的
    旧代码（如 pipeline.py、测试脚本、文档示例）无需立即改动也能工作。
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        llm: Any | None = None,
        empty_context_threshold: int = 0,
    ) -> None:
        self._empty_context_threshold = empty_context_threshold
        self._llm = llm
        self._retriever = retriever or get_retriever()
        # Synthesizer 延迟到真正需要生成时才创建，避免空 Context 拒答路径
        # 也去触碰 LlamaIndex 的全局 LLM 配置。
        self._synthesizer: Any | None = None

    def _get_synthesizer(self) -> Any:
        """懒加载合成器；测试可提前替换 ``_synthesizer`` 注入 Mock。"""
        if self._synthesizer is None:
            self._synthesizer = (
                NutriLifeSynthesizer(llm=self._llm)
                if self._llm is not None
                else get_synthesizer()
            )
        return self._synthesizer

    # ----------------------------------------------------------------
    # 下面的 query/aquery 是旧 API 兼容：内部走"轻量编排"
    # （真实项目里编排层应该是 LangGraph，这里是兼容层的简化版编排）
    # ----------------------------------------------------------------

    def query(self, question: str) -> RAGResponse:
        """旧接口兼容：执行完整 RAG 查询。"""
        logger.info("NutriLifeQueryEngine[兼容] query: '{}'", question[:100])

        nodes = self._retriever.retrieve(question)
        raw_count = len(nodes)

        # 空 Context 直接拒答（编排逻辑）
        if raw_count <= self._empty_context_threshold:
            logger.info(
                "Context 为空（节点数={} ≤ 阈值={}），触发拒答",
                raw_count,
                self._empty_context_threshold,
            )
            return RAGResponse(
                answer=FALLBACK_RESPONSE,
                is_fallback=True,
                sources=[],
                query=question,
                raw_context_count=0,
            )

        # 调用数据合成器（纯数据层）
        try:
            answer_text = self._get_synthesizer().synthesize(question, nodes)
            answer_text = str(answer_text).strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("NutriLifeQueryEngine[兼容] 合成失败: {}", exc)
            return RAGResponse(
                answer=FALLBACK_RESPONSE,
                is_fallback=True,
                sources=extract_source_references(nodes),
                query=question,
                raw_context_count=raw_count,
            )

        # 不确定性二次校验（编排逻辑）
        is_fallback = FALLBACK_RESPONSE in answer_text or is_fallback_response(
            answer_text
        )
        if is_fallback:
            logger.info("NutriLifeQueryEngine[兼容] 检测到拒答关键词")
            answer_text = FALLBACK_RESPONSE

        sources = extract_source_references(nodes)
        logger.info(
            "NutriLifeQueryEngine[兼容] 完成: is_fallback={}, sources={}",
            is_fallback,
            len(sources),
        )
        return RAGResponse(
            answer=answer_text,
            is_fallback=is_fallback,
            sources=sources if not is_fallback else [],
            query=question,
            raw_context_count=raw_count,
        )

    async def aquery(self, question: str) -> RAGResponse:
        """旧接口兼容：异步版本（线程池跑同步实现）。"""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.query, question)


# 旧工厂函数兼容（新代码用 get_rag_components）
_engine_singleton: NutriLifeQueryEngine | None = None


def get_query_engine(
    *,
    retriever_config: RAGSettings | None = None,
    force_rebuild: bool = False,
) -> NutriLifeQueryEngine:
    """旧接口兼容：获取 NutriLifeQueryEngine 单例。

    新代码建议使用 :func:`get_rag_components` 获取解耦后的数据组件。
    """
    global _engine_singleton
    if _engine_singleton is None or force_rebuild:
        logger.info("NutriLifeQueryEngine[兼容] 构建单例...")
        config = retriever_config or get_settings().rag
        retriever = get_retriever(config, force_rebuild=force_rebuild)
        _engine_singleton = NutriLifeQueryEngine(retriever=retriever)
        logger.info("NutriLifeQueryEngine[兼容] 单例构建完成")
    return _engine_singleton
