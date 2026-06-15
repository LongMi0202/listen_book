# -*- coding: utf-8 -*-
"""搜索路由 —— 内容检索"""

import logging
from fastapi import APIRouter, Depends
from knowledge.src.api.schemas import SearchRequest, SearchResponse, SearchResult
from knowledge.src.api.dependencies import get_rag_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["搜索"])


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, pipeline=Depends(get_rag_pipeline)):
    """
    内容检索 —— 返回匹配的知识条目及来源信息

    支持按内容类型和书名过滤
    """
    retriever, _, _ = pipeline

    docs = retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        book_name=request.book_name,
        content_type=request.content_type,
    )

    results = []
    for doc in docs:
        results.append(SearchResult(
            book_name=doc.metadata.get("book_name", ""),
            author_name=doc.metadata.get("author_name", ""),
            content_type=doc.metadata.get("content_type", ""),
            source_file=doc.metadata.get("source_file", ""),
            excerpt=doc.content[:300],
            score=round(1 - doc.score, 4) if doc.score else 0,  # 转余弦距离为相似度
        ))

    return SearchResponse(results=results, total=len(results))
