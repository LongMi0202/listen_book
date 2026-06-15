# -*- coding: utf-8 -*-
"""推荐路由 —— 书籍推荐"""

import logging
from fastapi import APIRouter, Depends
from knowledge.src.api.schemas import RecommendRequest, RecommendResponse
from knowledge.src.api.dependencies import get_rag_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["推荐"])


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest, pipeline=Depends(get_rag_pipeline)):
    """书籍推荐"""
    retriever, llm_client, _ = pipeline

    # 检索推荐资料 + 书籍简介
    docs = retriever.retrieve_for_recommend(
        query=request.query,
        top_k=request.top_k,
        tags=request.tags if request.tags else None,
    )

    # 生成推荐
    response = llm_client.generate_answer(
        query=request.query,
        context_docs=docs,
        intent="recommend",
        stream=False,
    )

    answer = response.choices[0].message.content
    sources = llm_client.extract_sources(docs)

    return RecommendResponse(answer=answer, sources=sources)
