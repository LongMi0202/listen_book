# -*- coding: utf-8 -*-
"""FastAPI 依赖注入 —— 管理全局单例"""

import logging
from functools import lru_cache

from knowledge.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL,
    EMBEDDING_MODEL, EMBEDDING_PROVIDER,
    CHROMA_DIR, SQLITE_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)

# 全局实例
_retriever = None
_llm_client = None
_meta_store = None
_parser = None
_chunker = None
_embedder = None


def _init_components():
    """延迟初始化所有组件"""
    global _retriever, _llm_client, _meta_store, _parser, _chunker, _embedder

    if _retriever is not None:
        return

    logger.info("=" * 50)
    logger.info("初始化 RAG 系统组件...")
    logger.info("=" * 50)

    # 嵌入模型
    from knowledge.src.embedding.embedder import Embedder
    if EMBEDDING_PROVIDER == "api":
        _embedder = Embedder(
            model_name=EMBEDDING_MODEL,
            provider="api",
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
    else:
        _embedder = Embedder(model_name=EMBEDDING_MODEL, provider="local")
    logger.info(f"  ✓ 嵌入模型: {EMBEDDING_MODEL} ({EMBEDDING_PROVIDER})")

    # 向量存储
    from knowledge.src.storage.vector_store import VectorStoreManager
    vector_store = VectorStoreManager(str(CHROMA_DIR))
    logger.info(f"  ✓ 向量存储: {CHROMA_DIR} ({vector_store.count()} 条)")

    # 检索器
    from knowledge.src.retrieval.retriever import HybridRetriever
    _retriever = HybridRetriever(vector_store=vector_store, embedder=_embedder)
    logger.info("  ✓ 检索器: 就绪")

    # LLM 客户端
    from knowledge.src.generation.llm_client import LLMClient
    _llm_client = LLMClient(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=LLM_MODEL,
    )
    logger.info(f"  ✓ LLM: {LLM_MODEL}")

    # 元数据存储
    from knowledge.src.storage.meta_store import MetaStore
    _meta_store = MetaStore(str(SQLITE_PATH))
    logger.info(f"  ✓ 元数据: {SQLITE_PATH}")

    # 解析器 & 分块器
    from knowledge.src.parser.markdown_parser import MarkdownParser
    from knowledge.src.splitter.chunker import AudiobookChunker
    _parser = MarkdownParser()
    _chunker = AudiobookChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    logger.info("  ✓ 解析器 & 分块器: 就绪")

    logger.info("=" * 50)
    logger.info("RAG 系统初始化完成!")
    logger.info("=" * 50)


def get_rag_pipeline():
    """获取 RAG 管道三件套: (retriever, llm_client, meta_store)"""
    _init_components()
    return _retriever, _llm_client, _meta_store


def get_parser_chunker_embedder():
    """获取导入工具三件套: (parser, chunker, embedder)"""
    _init_components()
    return _parser, _chunker, _embedder
