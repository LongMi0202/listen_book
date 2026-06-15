# -*- coding: utf-8 -*-
"""对话管理路由"""

import logging
from fastapi import APIRouter, Depends
from knowledge.src.api.schemas import ConversationInfo
from knowledge.src.api.dependencies import get_rag_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["对话"])


@router.get("/conversations")
async def list_conversations(pipeline=Depends(get_rag_pipeline)):
    """列出所有会话"""
    _, _, meta_store = pipeline
    sessions = meta_store.list_sessions(limit=50)

    result = []
    for s in sessions:
        msgs = meta_store.get_recent_messages(s["id"], limit=100)
        result.append(ConversationInfo(
            id=s["id"],
            title=s.get("title", "新对话"),
            created_at=s.get("created_at", ""),
            updated_at=s.get("updated_at", ""),
            message_count=len(msgs),
        ))

    return {"conversations": result}


@router.get("/conversations/{session_id}")
async def get_conversation(session_id: str, pipeline=Depends(get_rag_pipeline)):
    """获取指定会话的全部消息"""
    _, _, meta_store = pipeline
    messages = meta_store.get_session_messages(session_id)
    return {"session_id": session_id, "messages": messages}


@router.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str, pipeline=Depends(get_rag_pipeline)):
    """删除指定会话"""
    _, _, meta_store = pipeline
    meta_store.delete_session(session_id)
    return {"status": "ok", "session_id": session_id}
