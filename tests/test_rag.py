"""
NutriLife RAG 模块测试套件。

测试策略：
    使用 pytest + unittest.mock 对所有外部依赖（ChromaDB、LM Studio、LlamaIndex）
    进行完整 Mock，确保测试在没有任何外部服务的情况下可离线运行。

测试场景覆盖：
    1. 拒答机制 - 无关问题（天气）→ 必须返回拒答话术
    2. 拒答机制 - Context 为空 → 直接拒答，不调用 LLM
    3. 拒答机制 - LLM 返回不确定性文本 → 规范化为标准拒答话术
    4. 正常回答 - 营养学相关问题 → 返回有效回答 + 引用溯源
    5. 引用溯源 - 检索结果携带正确的 source 字段
    6. MockReranker - 验证按分数降序排列
    7. RAGSettings - 默认值校验
    8. RAGResponse - 数据类字段完整性
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

from llama_index.core.schema import NodeWithScore, TextNode

from app.core.config import RAGSettings, get_settings
from app.rag.query_engine import (
    FALLBACK_RESPONSE,
    NutriLifeQueryEngine,
    RAGResponse,
    SourceReference,
    _UNCERTAINTY_PATTERNS,
)
from app.rag.retriever import MockReranker


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


def _make_node(
    text: str,
    score: float = 0.85,
    source: str = "gout_diet.txt",
    node_id: str = "node-001",
) -> NodeWithScore:
    """创建带有完整元数据的 NodeWithScore 测试 Fixture。"""
    node = TextNode(
        text=text,
        id_=node_id,
        metadata={
            "source": source,
            "file_path": f"/data/knowledge_base/{source}",
            "file_type": "txt",
        },
    )
    return NodeWithScore(node=node, score=score)


@pytest.fixture
def nutrition_node() -> NodeWithScore:
    """营养学相关内容节点。"""
    return _make_node(
        text=(
            "大豆原料嘌呤含量约 150–200 mg/100g，但经过加工制成豆腐后，"
            "嘌呤含量显著降低至 55–70 mg/100g。"
            "多项队列研究显示，植物来源的嘌呤与痛风发作风险的关联显著弱于动物来源嘌呤。"
            "痛风患者在缓解期可适量食用加工豆制品（每日豆腐不超过 150g）。"
        ),
        score=0.92,
        source="gout_diet.txt",
        node_id="node-gout-001",
    )


@pytest.fixture
def mock_retriever_with_nodes(nutrition_node: NodeWithScore) -> MagicMock:
    """返回营养学节点的 Mock 检索器。"""
    retriever = MagicMock()
    retriever.retrieve.return_value = [nutrition_node]
    return retriever


@pytest.fixture
def mock_retriever_empty() -> MagicMock:
    """返回空结果的 Mock 检索器（模拟知识库无相关内容）。"""
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    return retriever


@pytest.fixture
def mock_synthesizer_with_answer() -> MagicMock:
    """返回有效回答的 Mock Synthesizer。"""
    synth = MagicMock()
    synth.synthesize.return_value = MagicMock(
        __str__=lambda self: (
            "根据知识库资料，痛风患者在缓解期可以适量食用豆腐（每日不超过150g）。"
            "豆腐嘌呤含量约55-70mg/100g，属于低嘌呤食品，"
            "且植物来源嘌呤对痛风发作风险的影响显著低于动物来源。"
        )
    )
    return synth


@pytest.fixture
def mock_synthesizer_fallback() -> MagicMock:
    """返回不确定性文本的 Mock Synthesizer（模拟无法回答的情形）。"""
    synth = MagicMock()
    synth.synthesize.return_value = MagicMock(
        __str__=lambda self: "无法从提供的信息中找到关于天气的相关内容。"
    )
    return synth


# ──────────────────────────────────────────────────────────────────
# 测试 1：拒答机制 - 无关问题（空 Context）
# ──────────────────────────────────────────────────────────────────


class TestFallbackMechanism:
    """拒答机制测试组。"""

    def test_empty_context_triggers_fallback_immediately(
        self, mock_retriever_empty: MagicMock
    ) -> None:
        """
        场景：检索器返回空结果（知识库中无相关内容）。
        期望：直接返回标准拒答话术，不调用 LLM。
        """
        engine = NutriLifeQueryEngine(retriever=mock_retriever_empty)

        result = engine.query("今天天气怎么样")

        assert isinstance(result, RAGResponse)
        assert result.is_fallback is True
        assert result.answer == FALLBACK_RESPONSE
        assert "建议咨询专业医生" in result.answer
        assert result.sources == []
        assert result.query == "今天天气怎么样"
        assert result.raw_context_count == 0

        # 验证 LLM（synthesizer）未被调用
        mock_retriever_empty.retrieve.assert_called_once_with("今天天气怎么样")

    def test_fallback_response_exact_text(
        self, mock_retriever_empty: MagicMock
    ) -> None:
        """
        场景：拒答话术必须与定义的 FALLBACK_RESPONSE 完全一致（不能有任何修改）。
        """
        engine = NutriLifeQueryEngine(retriever=mock_retriever_empty)
        result = engine.query("明天会下雨吗")

        assert result.answer == FALLBACK_RESPONSE, (
            f"拒答话术不匹配。\n期望: {FALLBACK_RESPONSE!r}\n实际: {result.answer!r}"
        )

    def test_llm_uncertainty_text_normalized_to_fallback(
        self,
        mock_retriever_with_nodes: MagicMock,
        mock_synthesizer_fallback: MagicMock,
    ) -> None:
        """
        场景：检索器返回了节点，但 LLM 生成了不确定性文本（"无法从提供的信息..."）。
        期望：二次校验检测到不确定性关键词，规范化替换为标准拒答话术。
        """
        engine = NutriLifeQueryEngine(retriever=mock_retriever_with_nodes)
        engine._synthesizer = mock_synthesizer_fallback

        result = engine.query("今天适合跑步吗")

        assert result.is_fallback is True
        assert result.answer == FALLBACK_RESPONSE

    def test_multiple_irrelevant_queries_all_trigger_fallback(
        self, mock_retriever_empty: MagicMock
    ) -> None:
        """
        场景：多个不同的无关问题，全部应触发拒答。
        """
        irrelevant_questions = [
            "今天天气怎么样",
            "帮我写一首诗",
            "最新的 iPhone 多少钱",
            "如何学习 Python",
            "世界杯冠军是谁",
        ]
        engine = NutriLifeQueryEngine(retriever=mock_retriever_empty)

        for question in irrelevant_questions:
            result = engine.query(question)
            assert result.is_fallback is True, (
                f"问题 '{question}' 应触发拒答，但实际未触发"
            )
            assert result.answer == FALLBACK_RESPONSE

    def test_fallback_response_contains_doctor_recommendation(
        self, mock_retriever_empty: MagicMock
    ) -> None:
        """
        场景：拒答话术必须包含"建议咨询专业医生"提示。
        """
        engine = NutriLifeQueryEngine(retriever=mock_retriever_empty)
        result = engine.query("股票今天涨了吗")

        assert "建议咨询专业医生" in result.answer


# ──────────────────────────────────────────────────────────────────
# 测试 2：正常回答 - 营养学相关问题
# ──────────────────────────────────────────────────────────────────


class TestNormalResponse:
    """正常回答测试组（营养学相关问题）。"""

    def test_nutrition_question_returns_answer_not_fallback(
        self,
        mock_retriever_with_nodes: MagicMock,
        mock_synthesizer_with_answer: MagicMock,
    ) -> None:
        """
        场景：营养学相关问题，检索器返回相关节点，LLM 生成有效回答。
        期望：is_fallback=False，回答包含有效内容。
        """
        engine = NutriLifeQueryEngine(retriever=mock_retriever_with_nodes)
        engine._synthesizer = mock_synthesizer_with_answer

        result = engine.query("痛风能吃豆制品吗")

        assert result.is_fallback is False
        assert result.answer != FALLBACK_RESPONSE
        assert len(result.answer) > 10
        assert result.raw_context_count == 1

    def test_nutrition_question_has_sources(
        self,
        mock_retriever_with_nodes: MagicMock,
        mock_synthesizer_with_answer: MagicMock,
    ) -> None:
        """
        场景：正常回答应附带引用溯源信息。
        期望：sources 列表非空，每个 source 包含 source 字段和 score。
        """
        engine = NutriLifeQueryEngine(retriever=mock_retriever_with_nodes)
        engine._synthesizer = mock_synthesizer_with_answer

        result = engine.query("痛风能吃豆制品吗")

        assert len(result.sources) > 0
        for src in result.sources:
            assert isinstance(src, SourceReference)
            assert src.source  # 文件名非空
            assert 0.0 <= src.score <= 1.0
            assert src.text_snippet  # 摘要非空


# ──────────────────────────────────────────────────────────────────
# 测试 3：引用溯源提取
# ──────────────────────────────────────────────────────────────────


class TestSourceExtraction:
    """引用溯源信息提取测试组。"""

    def test_extract_sources_from_nodes(self) -> None:
        """验证 _extract_sources 正确解析 source 元数据。"""
        nodes = [
            _make_node("豆腐嘌呤含量低", score=0.9, source="gout_diet.txt", node_id="n1"),
            _make_node("维生素D来源", score=0.75, source="nutrition_basics.txt", node_id="n2"),
        ]

        sources = NutriLifeQueryEngine._extract_sources(nodes)

        assert len(sources) == 2
        assert sources[0].source == "gout_diet.txt"
        assert sources[0].score == pytest.approx(0.9)
        assert sources[0].node_id == "n1"
        assert sources[1].source == "nutrition_basics.txt"
        assert sources[1].score == pytest.approx(0.75)

    def test_extract_sources_handles_missing_metadata(self) -> None:
        """当节点缺少 source 元数据时，应降级返回"未知文档"。"""
        node = TextNode(text="some content", id_="node-xyz", metadata={})
        nodes = [NodeWithScore(node=node, score=0.6)]

        sources = NutriLifeQueryEngine._extract_sources(nodes)

        assert sources[0].source == "未知文档"

    def test_source_snippet_truncated_to_200_chars(self) -> None:
        """验证 text_snippet 被截断至 200 字符。"""
        long_text = "A" * 500
        nodes = [_make_node(long_text, score=0.7, node_id="node-long")]

        sources = NutriLifeQueryEngine._extract_sources(nodes)

        assert len(sources[0].text_snippet) <= 200

    def test_fallback_response_has_empty_sources(
        self, mock_retriever_empty: MagicMock
    ) -> None:
        """拒答响应不应包含引用来源（sources 必须为空列表）。"""
        engine = NutriLifeQueryEngine(retriever=mock_retriever_empty)
        result = engine.query("今天几号")

        assert result.sources == []


# ──────────────────────────────────────────────────────────────────
# 测试 4：MockReranker
# ──────────────────────────────────────────────────────────────────


class TestMockReranker:
    """MockReranker 测试组。"""

    def test_reranker_returns_top_n(self) -> None:
        """验证 MockReranker 按 top_n 截断输出。"""
        nodes = [
            _make_node(f"内容 {i}", score=float(i) / 10, node_id=f"node-{i}")
            for i in range(10)
        ]
        reranker = MockReranker(top_n=3)
        result = reranker._postprocess_nodes(nodes, None)

        assert len(result) == 3

    def test_reranker_sorts_by_score_descending(self) -> None:
        """验证 MockReranker 按分数降序排列。"""
        nodes = [
            _make_node("low", score=0.3, node_id="node-low"),
            _make_node("high", score=0.9, node_id="node-high"),
            _make_node("mid", score=0.6, node_id="node-mid"),
        ]
        reranker = MockReranker(top_n=3)
        result = reranker._postprocess_nodes(nodes, None)

        scores = [n.score for n in result]
        assert scores == sorted(scores, reverse=True)
        assert result[0].node.node_id == "node-high"

    def test_reranker_handles_none_score(self) -> None:
        """验证 score=None 的节点被视为 0.0 排序。"""
        node_none_score = TextNode(text="no score", id_="node-none")
        node_with_score = _make_node("with score", score=0.5, node_id="node-scored")

        nodes = [
            NodeWithScore(node=node_none_score, score=None),
            node_with_score,
        ]
        reranker = MockReranker(top_n=2)
        result = reranker._postprocess_nodes(nodes, None)

        assert result[0].node.node_id == "node-scored"


# ──────────────────────────────────────────────────────────────────
# 测试 5：RAGSettings 默认值
# ──────────────────────────────────────────────────────────────────


class TestRAGSettings:
    """RAGSettings 测试组。"""

    def test_default_values(self) -> None:
        """验证默认配置值符合设计文档要求。"""
        config = get_settings().rag

        assert config.vector_top_k == 10
        assert config.bm25_top_k == 10
        assert config.fusion_top_k == 8
        assert config.rerank_top_n == 5
        assert config.similarity_cutoff == 0.3
        assert config.num_queries == 1
        assert config.qdrant_collection_name == "nutrilife_kb"

    def test_custom_config(self) -> None:
        """验证自定义配置覆盖。"""
        config = RAGSettings(rerank_top_n=3, vector_top_k=20)

        assert config.rerank_top_n == 3
        assert config.vector_top_k == 20
        assert config.bm25_top_k == 10  # 未覆盖，保持默认


# ──────────────────────────────────────────────────────────────────
# 测试 6：RAGResponse 数据模型
# ──────────────────────────────────────────────────────────────────


class TestRAGResponseModel:
    """RAGResponse 数据模型测试组。"""

    def test_rag_response_fields(self) -> None:
        """验证 RAGResponse 字段类型正确。"""
        response = RAGResponse(
            answer=FALLBACK_RESPONSE,
            is_fallback=True,
            sources=[],
            query="测试问题",
            raw_context_count=0,
        )

        assert isinstance(response.answer, str)
        assert isinstance(response.is_fallback, bool)
        assert isinstance(response.sources, list)
        assert isinstance(response.query, str)
        assert isinstance(response.raw_context_count, int)

    def test_source_reference_fields(self) -> None:
        """验证 SourceReference 字段类型正确。"""
        source = SourceReference(
            source="gout_diet.txt",
            file_path="/data/gout_diet.txt",
            text_snippet="豆腐嘌呤含量低...",
            score=0.88,
            node_id="node-001",
            metadata={"page": 1},
        )

        assert source.source == "gout_diet.txt"
        assert source.score == pytest.approx(0.88)
        assert source.metadata["page"] == 1


# ──────────────────────────────────────────────────────────────────
# 测试 7：不确定性模式检测
# ──────────────────────────────────────────────────────────────────


class TestUncertaintyDetection:
    """_is_fallback_response 方法测试组。"""

    @pytest.mark.parametrize("uncertain_text", [
        "无法回答这个问题",
        "上下文中没有相关信息",
        "The provided context does not contain information about weather",
        "I cannot answer based on the given context",
        "Context does not mention this topic",
        "no relevant information found",
    ])
    def test_detects_uncertainty_patterns(self, uncertain_text: str) -> None:
        """验证不确定性关键词被正确检测。"""
        assert NutriLifeQueryEngine._is_fallback_response(uncertain_text) is True

    @pytest.mark.parametrize("confident_text", [
        "痛风患者可以适量食用豆腐",
        "维生素D主要来源于阳光照射",
        "成人每日推荐钙摄入量为800-1200mg",
    ])
    def test_confident_answers_not_flagged(self, confident_text: str) -> None:
        """验证正常的确定性回答不会被误判为拒答。"""
        assert NutriLifeQueryEngine._is_fallback_response(confident_text) is False
