# -*- coding: utf-8 -*-
"""问答路由 —— 知识问答、流式输出"""

import json
import logging
from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
from knowledge.src.api.schemas import QARequest, QAResponse
from knowledge.src.api.dependencies import get_rag_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["问答"])


@router.post("/qa", response_model=QAResponse)
async def qa(request: QARequest, pipeline=Depends(get_rag_pipeline)):
    """知识问答（非流式）"""
    retriever, llm_client, meta_store = pipeline

    # 1. 检索
    docs = retriever.retrieve(query=request.query, top_k=8)

    # 2. 获取对话历史
    history = meta_store.get_recent_messages(request.session_id, limit=10)

    # 3. 生成
    response = llm_client.generate_answer(
        query=request.query,
        context_docs=docs,
        intent=request.intent,
        history=history,
        stream=False,
    )

    answer = response.choices[0].message.content
    sources = llm_client.extract_sources(docs)

    # 4. 保存对话
    meta_store.add_message(request.session_id, "user", request.query)
    meta_store.add_message(request.session_id, "assistant", answer, sources)

    return QAResponse(
        answer=answer,
        sources=sources,
        session_id=request.session_id,
    )


@router.post("/qa/stream")
async def qa_stream(request: QARequest, pipeline=Depends(get_rag_pipeline)):
    """知识问答（流式 SSE）"""
    retriever, llm_client, meta_store = pipeline

    # 1. 检索
    docs = retriever.retrieve(query=request.query, top_k=8)
    history = meta_store.get_recent_messages(request.session_id, limit=10)
    sources = llm_client.extract_sources(docs)

    # 保存用户消息
    meta_store.add_message(request.session_id, "user", request.query)

    async def event_generator():
        full_answer = ""

        try:
            # 2. 流式生成
            stream = llm_client.generate_answer(
                query=request.query,
                context_docs=docs,
                intent=request.intent,
                history=history,
                stream=True,
            )

            for token in stream:
                full_answer += token
                yield {"event": "token", "data": token}

            # 3. 推送引用来源
            yield {
                "event": "sources",
                "data": json.dumps(sources, ensure_ascii=False),
            }

            # 4. 保存回答
            meta_store.add_message(
                request.session_id, "assistant", full_answer, sources
            )

            yield {"event": "done", "data": json.dumps({"session_id": request.session_id})}

        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())
