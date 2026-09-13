#!/usr/bin/env python3
"""
NutriLife 知识库文档摄取脚本（ingest.py）。

功能：
    将 ``data/knowledge_base/`` 目录下的 PDF/TXT/MD 文档读取、
    切分后写入 Qdrant 向量库（本地 Docker），同时保存节点缓存供 BM25 检索。

使用方式::

    # 首次构建索引
    python scripts/ingest.py

    # 强制重建（清空旧索引）
    python scripts/ingest.py --rebuild

    # 指定知识库目录
    python scripts/ingest.py --kb-dir /path/to/docs
"""

from __future__ import annotations

from typing import List
import numpy as np
import argparse
import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，使脚本可直接调用 app 模块
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

# ─── LlamaIndex ────────────────────────────────────────────────────
from llama_index.core import Settings as LlamaSettings
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.core.llm import NormalizedBGEEmbedding

# ─── App 配置 ──────────────────────────────────────────────────────
from app.core.config import get_settings

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md", ".docx"}


def build_embed_model(settings: object) -> OpenAILikeEmbedding:
    """
    构建指向 LM Studio 的 OpenAI 兼容 Embedding 模型。

    LM Studio 同时支持 Embedding 推理，使用与 Chat 相同的 base_url。
    如果 LM Studio 当前加载的模型不支持 Embedding，可改为使用
    ``llama-index-embeddings-huggingface`` 本地 HuggingFace Embedding。

    Args:
        settings: 全局配置对象。

    Returns:
        OpenAILikeEmbedding: 已配置的 Embedding 模型实例。
    """
    cfg = settings.lm_studio  # type: ignore[union-attr]
    logger.info(
        "初始化 Embedding 模型 → base_url={}, model={}",
        cfg.base_url,
        cfg.embed_model,
    )
    return NormalizedBGEEmbedding(
        model_name=cfg.embed_model,
        api_base=cfg.base_url,
        api_key=cfg.api_key,
        embed_batch_size=4,          # 本地模型并发能力有限，batch 不宜过大
        timeout=60,
    )


def load_documents(kb_dir: Path) -> list:
    """
    从知识库目录递归加载所有支持格式的文档。

    Args:
        kb_dir: 知识库根目录路径。

    Returns:
        list[Document]: LlamaIndex Document 对象列表。

    Raises:
        FileNotFoundError: 当 kb_dir 不存在时。
        ValueError: 当目录内没有任何支持格式的文件时。
    """
    if not kb_dir.exists():
        raise FileNotFoundError(f"知识库目录不存在: {kb_dir}")

    supported_files = [
        f for f in kb_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not supported_files:
        raise ValueError(
            f"目录 {kb_dir} 中未找到任何支持的文档格式 "
            f"({', '.join(SUPPORTED_EXTENSIONS)})"
        )

    logger.info("发现 {} 个文档文件，开始加载...", len(supported_files))
    for f in supported_files:
        logger.debug("  → {}", f.name)

    reader = SimpleDirectoryReader(
        input_dir=str(kb_dir),
        recursive=True,
        required_exts=list(SUPPORTED_EXTENSIONS),
        # 保留文件元数据，用于后续引用溯源
        file_metadata=lambda path: {
            "source": Path(path).name,
            "file_path": str(path),
            "file_type": Path(path).suffix.lstrip("."),
        },
    )
    docs = reader.load_data()
    logger.info("成功加载 {} 个文档片段（分页/段落）", len(docs))
    return docs


def split_documents(docs: list, chunk_size: int = 512, chunk_overlap: int = 50) -> list:
    """
    使用 SentenceSplitter 对文档进行语义感知切分。

    SentenceSplitter 优先在句子边界切分，保证语义完整性，
    比简单的字符级切分效果更好。

    Args:
        docs: LlamaIndex Document 对象列表。
        chunk_size: 单个 Chunk 的最大 Token 数（默认 512）。
        chunk_overlap: 相邻 Chunk 的重叠 Token 数（默认 50）。

    Returns:
        list[TextNode]: 切分后的 Node 列表。
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",  # 优先在段落边界切分
    )
    nodes = splitter.get_nodes_from_documents(docs, show_progress=True)
    logger.info(
        "文档切分完成：{} 个原始文档 → {} 个 Chunks (chunk_size={}, overlap={})",
        len(docs),
        len(nodes),
        chunk_size,
        chunk_overlap,
    )
    return nodes


def build_qdrant_index(
    nodes: list,
    url: str,
    embed_model: OpenAILikeEmbedding,
    collection_name: str = "nutrilife_kb",
    rebuild: bool = False,
) -> VectorStoreIndex:
    """
    将切分后的 Nodes 写入 Qdrant 并构建 VectorStoreIndex。

    Args:
        nodes: 已切分的 TextNode 列表。
        url: Qdrant 服务地址。
        embed_model: Embedding 模型实例。
        collection_name: Qdrant Collection 名称。
        rebuild: 若为 True，删除旧集合后重建；否则追加写入。

    Returns:
        VectorStoreIndex: 已建立的 LlamaIndex 向量索引。
    """
    qdrant_client = QdrantClient(url=url)

    if rebuild and qdrant_client.collection_exists(collection_name):
        logger.warning("--rebuild 模式：删除旧集合 {}", collection_name)
        qdrant_client.delete_collection(collection_name)

    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        client=qdrant_client,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info("开始向量化并写入 Qdrant...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    points_count = qdrant_client.get_collection(collection_name).points_count
    logger.info(
        "Qdrant 索引构建完成！Collection='{}' 共 {} 个向量",
        collection_name,
        points_count,
    )
    return index


def save_node_cache(nodes: list, cache_path: Path) -> None:
    """将切分后的节点保存为 JSON 缓存，供 BM25 检索器重建语料。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"text": node.text, "metadata": node.metadata or {}} for node in nodes]
    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    logger.info("节点缓存已保存至 {}，共 {} 个节点", cache_path, len(nodes))


def main() -> None:
    """脚本主入口。"""
    parser = argparse.ArgumentParser(
        description="NutriLife 知识库文档摄取 & 向量化脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=None,
        help="知识库文档目录（默认从 .env 读取 KNOWLEDGE_BASE_DIR）",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help="Qdrant 服务地址（默认从 .env 读取 QDRANT_URL）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="文档切分的 Chunk 大小（Token 数）",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="相邻 Chunk 的重叠 Token 数",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="清空旧索引并完整重建（谨慎使用）",
    )
    args = parser.parse_args()

    settings = get_settings()

    kb_dir = args.kb_dir or Path(settings.rag.knowledge_base_dir)
    qdrant_url = args.qdrant_url or settings.rag.qdrant_url
    cache_path = Path(settings.rag.node_cache_path)

    logger.info("=" * 60)
    logger.info("NutriLife 知识库摄取开始")
    logger.info("  知识库目录: {}", kb_dir)
    logger.info("  Qdrant 地址: {}", qdrant_url)
    logger.info("  Chunk Size: {} tokens", args.chunk_size)
    logger.info("  Chunk Overlap: {} tokens", args.chunk_overlap)
    logger.info("=" * 60)

    # 1. 初始化 Embedding 模型
    embed_model = build_embed_model(settings)
    LlamaSettings.embed_model = embed_model

    # 2. 加载文档
    docs = load_documents(kb_dir)

    # 3. 切分文档
    nodes = split_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    # 4. 构建 Qdrant 向量索引 + 保存节点缓存（供 BM25 检索器）
    build_qdrant_index(
        nodes=nodes,
        url=qdrant_url,
        embed_model=embed_model,
        rebuild=args.rebuild,
    )
    save_node_cache(nodes, cache_path)

    logger.info("✅ 摄取完成！可以启动服务并使用 RAG 检索了。")


if __name__ == "__main__":
    main()
