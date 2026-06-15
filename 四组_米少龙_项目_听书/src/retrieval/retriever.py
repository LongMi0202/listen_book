# -*- coding: utf-8 -*-
"""混合检索器 —— 语义搜索 + 元数据过滤"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from knowledge.src.embedding.embedder import Embedder
from knowledge.src.storage.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """检索结果文档"""
    content: str
    metadata: dict
    score: float = 0.0  # 相似度分数（越小越好，cosine distance）


class HybridRetriever:
    """
    混合检索器

    支持:
    - 纯语义搜索
    - 按书名/作者/内容类型/标签过滤
    - 多条件组合过滤
    """

    def __init__(self, vector_store: VectorStoreManager, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        book_name: Optional[str] = None,
        author_name: Optional[str] = None,
        content_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[RetrievedDocument]:
        """
        混合检索主入口

        策略:
        1. 先按元数据过滤检索 top_k*2 候选
        2. 如无过滤条件，直接语义检索
        """
        # 生成查询嵌入
        query_embedding = self.embedder.embed_query(query)

        # 检索（多取一些候选，留去重空间）
        fetch_k = top_k * 2

        has_filter = any([book_name, author_name, content_type, tags])

        if has_filter:
            result = self.vector_store.search_with_metadata_filter(
                query_embedding=query_embedding,
                n_results=fetch_k,
                book_name=book_name,
                author_name=author_name,
                content_type=content_type,
                tags=tags,
            )
        else:
            result = self.vector_store.search(
                query_embedding=query_embedding,
                n_results=fetch_k,
            )

        # 解析结果
        docs = self._parse_results(result)

        # 去重（按内容相似度去重，保留分数更高的）
        docs = self._deduplicate(docs, top_k)

        logger.debug(f"检索 '{query[:30]}...' → {len(docs)} 条结果")
        return docs

    def retrieve_for_recommend(
        self,
        query: str,
        top_k: int = 8,
        tags: Optional[list[str]] = None,
    ) -> list[RetrievedDocument]:
        """书籍推荐专用检索 —— 优先检索推荐运营资料和书籍简介"""
        query_embedding = self.embedder.embed_query(query)

        # 分两次检索：先搜推荐资料，再搜书籍简介，合并结果
        result_rec = self.vector_store.search_with_metadata_filter(
            query_embedding=query_embedding,
            n_results=top_k,
            content_type="推荐运营资料",
            tags=tags,
        )
        result_intro = self.vector_store.search_with_metadata_filter(
            query_embedding=query_embedding,
            n_results=top_k,
            content_type="书籍简介",
            tags=tags,
        )

        docs_rec = self._parse_results(result_rec)
        docs_intro = self._parse_results(result_intro)

        # 合并、去重
        all_docs = docs_rec + docs_intro
        return self._deduplicate(all_docs, top_k)

    def retrieve_for_detail(
        self,
        book_name: str,
        top_k: int = 10,
    ) -> list[RetrievedDocument]:
        """书籍详情专用检索 —— 检索指定书籍的所有内容类型"""
        query_embedding = self.embedder.embed_query(book_name)

        result = self.vector_store.search_with_metadata_filter(
            query_embedding=query_embedding,
            n_results=top_k,
            book_name=book_name,
        )
        return self._parse_results(result)

    def _parse_results(self, chroma_result: dict) -> list[RetrievedDocument]:
        """解析 ChromaDB 查询结果"""
        docs = []
        if not chroma_result.get("ids") or not chroma_result["ids"][0]:
            return docs

        ids_list = chroma_result["ids"][0]
        docs_list = chroma_result.get("documents", [[]])[0]
        metas_list = chroma_result.get("metadatas", [[]])[0]
        dists_list = chroma_result.get("distances", [[]])[0]

        for i in range(len(ids_list)):
            docs.append(RetrievedDocument(
                content=docs_list[i] if i < len(docs_list) else "",
                metadata=metas_list[i] if i < len(metas_list) else {},
                score=dists_list[i] if i < len(dists_list) else 0.0,
            ))

        return docs

    def _deduplicate(
        self,
        docs: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]:
        """按书籍+内容类型去重，保留相似度最高的"""
        seen = set()
        unique = []
        for doc in docs:
            key = (
                doc.metadata.get("book_name", ""),
                doc.metadata.get("content_type", ""),
            )
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique[:top_k]
