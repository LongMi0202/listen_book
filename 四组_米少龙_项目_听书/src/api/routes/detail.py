# -*- coding: utf-8 -*-
"""详情路由 —— 书籍详情查询"""

import logging
from fastapi import APIRouter, Depends
from knowledge.src.api.schemas import DetailRequest, DetailResponse
from knowledge.src.api.dependencies import get_rag_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["详情"])


@router.post("/detail", response_model=DetailResponse)
async def detail(request: DetailRequest, pipeline=Depends(get_rag_pipeline)):
    """查询指定书籍的详细信息"""
    retriever, llm_client, _ = pipeline

    query = request.query or f"介绍《{request.book_name}》这本书的详细信息"

    # 检索指定书籍的所有内容
    docs = retriever.retrieve_for_detail(
        book_name=request.book_name,
        top_k=10,
    )

    # 生成回答
    response = llm_client.generate_answer(
        query=query,
        context_docs=docs,
        intent="detail",
        stream=False,
    )

    answer = response.choices[0].message.content
    sources = llm_client.extract_sources(docs)

    return DetailResponse(
        answer=answer,
        book_name=request.book_name,
        sources=sources,
    )
