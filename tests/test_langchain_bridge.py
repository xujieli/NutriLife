"""LlamaIndex → LangChain bridge 单元测试。

不依赖 Qdrant / LM Studio，仅用 LlamaIndex 的 ``TextNode`` 与一个假的
LlamaIndex 检索器验证类型转换和异步适配行为。
"""

from __future__ import annotations

from langchain_core.documents import Document
from llama_index.core.schema import NodeWithScore, TextNode

from app.rag.langchain_bridge import (
    LlamaIndexLangChainRetriever,
    build_context_str_from_documents,
    node_with_score_to_document,
    nodes_with_scores_to_documents,
)


def _make_node(
    text: str,
    *,
    score: float | None = 0.82,
    source: str = "gout_diet.txt",
    node_id: str = "node-001",
) -> NodeWithScore:
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


class _FakeLlamaIndexRetriever:
    """模拟 LlamaIndex 检索器，暴露 retrieve 与 aretrieve。"""

    def __init__(self, nodes: list[NodeWithScore]) -> None:
        self.nodes = nodes

    def retrieve(self, query: str) -> list[NodeWithScore]:
        return self.nodes

    async def aretrieve(self, query: str) -> list[NodeWithScore]:
        return self.nodes


def test_node_with_score_to_document_preserves_trace_metadata() -> None:
    """转换后应保留 source / score / node_id，并包装为标准 Document。"""
    node = _make_node("痛风患者在缓解期可适量食用豆腐。", score=0.91)

    document = node_with_score_to_document(node)

    assert isinstance(document, Document)
    assert document.page_content == "痛风患者在缓解期可适量食用豆腐。"
    assert document.metadata["source"] == "gout_diet.txt"
    assert document.metadata["score"] == 0.91
    assert document.metadata["node_id"] == "node-001"
    assert document.metadata["file_path"].endswith("gout_diet.txt")


def test_node_with_score_to_document_truncates_content() -> None:
    """超长节点内容应按 max_length_per_node 截断。"""
    node = _make_node("A" * 100)

    document = node_with_score_to_document(node, max_length_per_node=12)

    assert len(document.page_content) == 12
    assert document.page_content == "A" * 12


def test_nodes_with_scores_to_documents_batch() -> None:
    """批量转换应保持输入顺序。"""
    nodes = [
        _make_node("内容一", node_id="n1"),
        _make_node("内容二", node_id="n2"),
    ]

    documents = nodes_with_scores_to_documents(nodes)

    assert [document.page_content for document in documents] == ["内容一", "内容二"]
    assert [document.metadata["node_id"] for document in documents] == ["n1", "n2"]


def test_build_context_str_from_documents_joins_content() -> None:
    """Document 列表应被拼接为 Prompt 可用的上下文字符串。"""
    documents = [
        Document(page_content="片段一", metadata={}),
        Document(page_content="片段二", metadata={}),
    ]

    context = build_context_str_from_documents(documents)

    assert context == "片段一\n\n片段二"


async def test_langchain_retriever_async_bridge() -> None:
    """LangGraph 节点通过 ainvoke 拿到的是 Document，而不是 NodeWithScore。"""
    llama_index_retriever = _FakeLlamaIndexRetriever(
        [_make_node("豆腐嘌呤约 55-70mg/100g", node_id="n-gout")]
    )
    retriever = LlamaIndexLangChainRetriever(
        llama_index_retriever=llama_index_retriever,
    )

    documents = await retriever.ainvoke("痛风能吃豆腐吗")

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert "豆腐嘌呤" in documents[0].page_content
    assert documents[0].metadata["node_id"] == "n-gout"


async def test_langchain_retriever_falls_back_to_threaded_retrieve() -> None:
    """底层只有 retrieve 时，异步适配应自动降级到线程池并保持正确类型。"""

    class SyncOnlyRetriever:
        def __init__(self, nodes: list[NodeWithScore]) -> None:
            self.nodes = nodes

        def retrieve(self, query: str) -> list[NodeWithScore]:
            return self.nodes

    llama_index_retriever = SyncOnlyRetriever(
        [_make_node("维生素D主要来源于阳光", node_id="n-vitd")]
    )

    retriever = LlamaIndexLangChainRetriever(
        llama_index_retriever=llama_index_retriever,
    )

    documents = await retriever.ainvoke("维生素D从哪里来")

    assert len(documents) == 1
    assert documents[0].metadata["node_id"] == "n-vitd"
