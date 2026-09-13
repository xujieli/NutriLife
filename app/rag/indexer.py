"""
NutriLife 向量索引管理器。

封装"加载已有索引"或"从节点构建新索引"的逻辑，
供 retriever.py、query_engine.py 和 ingest.py 共享使用。
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.schema import BaseNode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import QdrantClient

from app.core.config import RAGSettings, get_settings


def load_or_create_index(
    nodes: list[BaseNode] | None = None,
    embed_model: EmbedType | None = None,
    collection_name: str | None = None,
    *,
    rag_config: RAGSettings | None = None,
) -> VectorStoreIndex:
    """
    从 Qdrant 加载已有向量索引，或从传入的 nodes 构建新索引。

    Args:
        nodes: TextNode 列表。传入时构建并写入向量索引；
            未传入时从现有 Qdrant 集合加载索引。
        embed_model: Embedding 模型实例。None 时由 LlamaIndex
            自动使用 ``Settings.embed_model``。
        collection_name: Qdrant Collection 名称。None 时使用
            ``rag_config`` 中的配置。
        rag_config: RAG 配置。None 时使用全局 ``get_settings().rag``。

    Returns:
        VectorStoreIndex: 向量索引实例。
    """
    cfg = rag_config or get_settings().rag
    target_collection_name = collection_name or cfg.qdrant_collection_name

    logger.info(
        "连接 Qdrant: url={}, collection={}",
        cfg.qdrant_url,
        target_collection_name,
    )
    qdrant_client = QdrantClient(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key or None,
    )
    if not qdrant_client.collection_exists(target_collection_name) and not nodes:
        logger.warning(
            "Qdrant 集合 '{}' 不存在，请先运行 `python scripts/ingest.py` 构建索引",
            target_collection_name,
        )

    vector_store = QdrantVectorStore(
        collection_name=target_collection_name,
        client=qdrant_client,
    )

    if nodes:
        logger.info("检测到传入 nodes，正在构建并写入向量索引...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model,
        )

    logger.info("未传入 nodes，正在从现有 Qdrant 集合加载向量索引...")
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
