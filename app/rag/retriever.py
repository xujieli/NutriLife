"""
NutriLife 混合检索模块（Hybrid Retriever）。

架构：
    ┌─────────────────────────────────────────────┐
    │           HybridRetriever                   │
    │  ┌─────────────────┐  ┌──────────────────┐  │
    │  │ VectorRetriever │  │  BM25Retriever   │  │
    │  │  (Qdrant)       │  │ (Keyword-based)  │  │
    │  └────────┬────────┘  └────────┬─────────┘  │
    │           └──────────┬─────────┘            │
    │              QueryFusionRetriever            │
    │          (RRF 倒数排名融合 + 去重)             │
    │                      │                      │
    │              RerankerPostProcessor           │
    │          (Cross-Encoder 重排序)               │
    └─────────────────────────────────────────────┘

说明：
    - 向量检索负责语义相似度召回
    - BM25 负责关键词精确匹配召回（对小模型 Embedding 能力的重要补充）
    - QueryFusionRetriever 通过 Reciprocal Rank Fusion (RRF) 融合两路结果
    - Reranker 对融合后的候选集进行精排，选出最相关的 Top-K 片段
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from loguru import logger

from app.core.config import RAGSettings, get_settings
from app.rag.indexer import load_or_create_index


# ──────────────────────────────────────────────────────────────────
# HybridRetriever — 核心实现
# ──────────────────────────────────────────────────────────────────


class HybridRetriever:
    """
    NutriLife 混合检索器。

    封装 BM25 + 向量检索 + RRF 融合 + Rerank 的完整检索流水线，
    对外暴露统一的 ``retrieve(query)`` 接口。

    Args:
        config: 检索器配置，使用 :class:`RAGSettings` 实例。
        reranker: 可插拔的 Reranker 后处理器实例。
            传入 ``None`` 则跳过重排序步骤。
        nodes: 用于初始化 BM25 索引的 TextNode 列表。
            若为 None，将从节点缓存文件加载已有 nodes。
        embed_model: LlamaIndex Embedding 模型实例。
            若为 None，使用 ``llama_index.core.Settings.embed_model``。

    Example::

        from app.rag.retriever import HybridRetriever

        retriever = HybridRetriever()
        results = retriever.retrieve("痛风能吃豆制品吗")
        for node in results:
            print(node.get_content()[:200])
    """

    def __init__(
        self,
        config: RAGSettings | None = None,
        reranker: BaseNodePostprocessor | None = None,
        nodes: list[Any] | None = None,
        embed_model: Any | None = None,
    ) -> None:
        self.config = config or get_settings().rag
        self._nodes = nodes
        self._embed_model = embed_model
        self._vector_index: VectorStoreIndex | None = None
        self._vector_retriever: BaseRetriever | None = None
        self._bm25_retriever: BM25Retriever | None = None
        self._reranker = reranker
        # self._fusion_retriever: QueryFusionRetriever | None = None
        self._initialized = False

    def initialize(self) -> "HybridRetriever":
        """
        懒加载初始化：构建向量索引和 BM25 索引。

        调用此方法后，检索器才真正可用。设计为链式调用：

            retriever = HybridRetriever().initialize()

        Returns:
            HybridRetriever: 已初始化的自身实例（支持链式调用）。

        Raises:
            RuntimeError: 当 Qdrant 集合为空且未提供 nodes 时。
        """
        if self._initialized:
            return self

        logger.info("初始化 HybridRetriever...")
        cfg = self.config

        # ── 1. 构建向量索引（基于 Qdrant）──────────────────────────
        self._vector_index = load_or_create_index(
            nodes=self._nodes,
            embed_model=self._embed_model,
            rag_config=cfg,
        )
        self._vector_retriever = self._vector_index.as_retriever(
            similarity_top_k=cfg.vector_top_k,
        )
        logger.info("向量检索器初始化完成 (top_k={})", cfg.vector_top_k)

        # ── 2. 构建 BM25 检索器 ──────────────────────────────────────
        bm25_nodes = self._nodes or self._load_nodes_from_cache()
        if not bm25_nodes:
            logger.warning("BM25 节点为空，BM25 检索器将不可用或可能报错，请检查节点缓存文件。")
            # 如果 BM25 必须有数据，这里可以 raise ValueError
        self._bm25_retriever = BM25Retriever.from_defaults(
            nodes=bm25_nodes,
            similarity_top_k=cfg.bm25_top_k,
        )
        logger.info("BM25 检索器初始化完成 (top_k={})", cfg.bm25_top_k)

        # ── 3. 构建 QueryFusionRetriever（RRF 融合）──────────────────
        # self._fusion_retriever = QueryFusionRetriever(
        #     retrievers=[self._vector_retriever, bm25_retriever],
        #     similarity_top_k=cfg.fusion_top_k,
        #     num_queries=cfg.num_queries,   # 设为 1 禁用 query augmentation
        #     mode="reciprocal_rerank",       # RRF 融合模式
        #     use_async=cfg.use_async,
        #     verbose=True,
        # )
        # logger.info(
        #     "QueryFusionRetriever 初始化完成 (fusion_top_k={}, mode=reciprocal_rerank)",
        #     cfg.fusion_top_k,
        # )

        # ── 3. 构建Rerank ──────────────────
        if self.config.enable_rerank:
            self._reranker = FlagEmbeddingReranker(
                model="BAAI/bge-reranker-v2-m3",
                top_n=self.config.rerank_top_n,
                use_fp16=False                    # 开启半精度加速（显著降低内存占用）
            )
            logger.info("BGE Reranker 初始化成功 (model=BAAI/bge-reranker-v2-m3)")

        self._initialized = True
        logger.info("HybridRetriever 初始化完成 ✓")
        return self

    def _load_nodes_from_cache(self) -> list[TextNode]:
        """
        从节点缓存 JSON 文件加载文档节点（用于 BM25 索引重建）。

        ingest.py 在切分文档后会同时把节点写入该缓存文件，避免 BM25
        依赖 Qdrant 内部 payload 结构，实现与向量库解耦。

        Returns:
            list[TextNode]: 从缓存恢复的 TextNode 列表。
        """
        cache_path = Path(self.config.node_cache_path)
        if not cache_path.exists():
            logger.warning(
                "节点缓存 '{}' 不存在，BM25 将在空语料上运行",
                cache_path,
            )
            return []

        try:
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            nodes = [
                TextNode(text=item.get("text", ""), metadata=item.get("metadata") or {})
                for item in data
            ]
            logger.info("从缓存加载 {} 个节点用于 BM25 索引", len(nodes))
            return nodes
        except Exception as exc:  # noqa: BLE001
            logger.error("节点缓存加载失败: {}", exc)
            return []

    def _manual_rrf_fusion(
        self, 
        retriever_results: list[list[NodeWithScore]], 
        k: int = 60
    ) -> list[NodeWithScore]:
        """
        手动实现 Reciprocal Rank Fusion (RRF)。
        比 LlamaIndex 的 QueryFusionRetriever 更轻量，且不会丢失原始检索次数。
        """
        scores = {}
        node_map = {}
        
        for results in retriever_results:
            for rank, node_with_score in enumerate(results):
                # 获取节点的唯一标识 (node_id)
                node_id = node_with_score.node.node_id
                
                # 计算 RRF 分数: 1 / (k + rank + 1)
                # rank 从 0 开始，所以 +1 保证分母不为 0
                rrf_score = 1.0 / (k + rank + 1)
                scores[node_id] = scores.get(node_id, 0.0) + rrf_score
                
                # 保留节点引用（用于后续组装结果）
                if node_id not in node_map:
                    node_map[node_id] = node_with_score.node
                    
        # 按 RRF 分数降序排序
        sorted_node_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # 重新打包为 NodeWithScore，截取 top_k
        fused_results = [
            NodeWithScore(node=node_map[node_id], score=scores[node_id])
            for node_id in sorted_node_ids[:self.config.fusion_top_k]
        ]
        return fused_results

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """
        执行混合检索并返回 Rerank 后的节点列表。

        完整流程：
            1. 向量相关性门控（最高余弦相似度低于阈值则直接返回空）
            2. BM25 + 向量检索，RRF 倒数排名融合，去重
            3. Reranker 精排，返回 Top-N

        Args:
            query: 用户查询字符串。

        Returns:
            list[NodeWithScore]: 经过混合检索和重排序的最终节点列表，
                按相关性分数降序排列。当判定知识库无相关内容时返回空列表。

        Raises:
            RuntimeError: 当检索器未初始化时。
        """
        if not self._initialized:
            raise RuntimeError(
                "HybridRetriever 尚未初始化，请先调用 .initialize()"
            )

        logger.info("执行混合检索: query='{}'", query[:80])
        query_bundle = QueryBundle(query_str=query)

        # ── Step 1：向量相关性门控 ────────────────────────────────
        # QueryFusionRetriever 的 RRF 分数是排名分数（1/(60+rank)，量级约 0.016），
        # 无法与余弦相似度阈值直接比较。这里单独用原始向量检索的余弦相似度作为
        # "知识库是否包含相关内容"的判定信号：低于阈值则返回空列表，由上层
        # query_engine 触发拒答，避免小模型在无关上下文上胡编。
        assert self._vector_retriever is not None
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        max_similarity = max(
            (n.score for n in vector_nodes if n.score is not None),
            default=0.0,
        )
        if max_similarity < self.config.similarity_cutoff:
            logger.info(
                "最高向量相似度 {:.3f} 低于阈值 {:.3f}，判定知识库无相关内容",
                max_similarity,
                self.config.similarity_cutoff,
            )
            return []

        # ── Step 2：执行 BM25 检索（仅执行 1 次） ─────────────────
        assert self._bm25_retriever is not None
        bm25_nodes = self._bm25_retriever.retrieve(query_bundle)

        # ── Step 3：手动执行 RRF 融合（替代 QueryFusionRetriever） ──
        fused_nodes = self._manual_rrf_fusion(
            [vector_nodes, bm25_nodes], 
            k=60  # RRF 常数 k，通常设为 60
        )
        logger.debug("RRF 融合返回 {} 个候选节点", len(fused_nodes))

        # ── Step 3：Rerank 精排 ──────────────────────────────────
        final_nodes = self._reranker.postprocess_nodes(fused_nodes, query_bundle) if self._reranker else fused_nodes
        logger.info(
            "混合检索完成：最终返回 {} 个节点",
            len(final_nodes),
        )
        return final_nodes
