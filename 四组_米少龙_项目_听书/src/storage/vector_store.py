# -*- coding: utf-8 -*-
"""向量存储 —— ChromaDB 操作封装"""

import logging
from typing import Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    ChromaDB 向量存储管理器

    管理一个名为 audiobook_knowledge 的 Collection
    支持语义搜索 + 元数据过滤
    """

    COLLECTION_NAME = "audiobook_knowledge"

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self._get_or_create_collection()
        logger.info(f"向量存储已就绪: {persist_dir}")

    def _get_or_create_collection(self):
        """获取或创建 Collection"""
        try:
            collection = self.client.get_collection(self.COLLECTION_NAME)
            logger.info(f"Collection '{self.COLLECTION_NAME}' 已存在，当前 {collection.count()} 条")
            return collection
        except Exception:
            logger.info(f"创建新 Collection: {self.COLLECTION_NAME}")
            return self.client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ):
        """批量添加文档"""
        if not ids:
            return

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info(f"已添加 {len(ids)} 条向量")

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where_filter: Optional[dict] = None,
    ) -> dict:
        """
        语义搜索

        参数:
            query_embedding: 查询嵌入向量
            n_results: 返回结果数
            where_filter: ChromaDB 元数据过滤条件
                例如: {"content_type": "书籍简介"}
                     {"book_name": "三体"}
                     {"tags": {"$contains": "科幻"}}

        返回:
            ChromaDB 查询结果 {ids, documents, metadatas, distances}
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where_filter:
            kwargs["where"] = where_filter

        return self.collection.query(**kwargs)

    def search_with_metadata_filter(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        book_name: Optional[str] = None,
        author_name: Optional[str] = None,
        content_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> dict:
        """
        带元数据过滤的语义搜索

        支持按书名、作者、内容类型、标签组合过滤
        """
        conditions = []

        if book_name:
            conditions.append({"book_name": book_name})
        if author_name:
            conditions.append({"author_name": author_name})
        if content_type:
            conditions.append({"content_type": content_type})
        if tags:
            for tag in tags:
                conditions.append({"tags": {"$contains": tag}})

        if len(conditions) == 0:
            where_filter = None
        elif len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}

        return self.search(query_embedding, n_results, where_filter)

    def count(self) -> int:
        """返回当前 Collection 中的文档数量"""
        return self.collection.count()

    def reset(self):
        """清空 Collection（谨慎使用）"""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
            self.collection = self._get_or_create_collection()
            logger.info("Collection 已重置")
        except Exception as e:
            logger.warning(f"重置失败: {e}")
