# -*- coding: utf-8 -*-
"""API 请求/响应 Schema"""

from pydantic import BaseModel, Field
from typing import Optional


# ==================== 推荐 ====================

class RecommendRequest(BaseModel):
    query: str = Field(..., description="用户查询，如'推荐科幻类有声书'")
    tags: list[str] = Field(default_factory=list, description="过滤标签")
    top_k: int = Field(default=5, ge=1, le=20)


class BookCard(BaseModel):
    book_name: str
    author_name: str
    tags: list[str]
    highlights: str = ""
    target_audience: str = ""
    reason: str = ""


class RecommendResponse(BaseModel):
    answer: str
    books: list[BookCard] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)


# ==================== 详情 ====================

class DetailRequest(BaseModel):
    book_name: str = Field(..., description="书名")
    query: str = Field(default="", description="具体查询，如'这本书适合什么人听'")


class DetailResponse(BaseModel):
    answer: str
    book_name: str
    sources: list[dict] = Field(default_factory=list)


# ==================== 搜索 ====================

class SearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询")
    content_type: Optional[str] = Field(default=None, description="内容类型过滤")
    book_name: Optional[str] = Field(default=None, description="书名过滤")
    top_k: int = Field(default=10, ge=1, le=50)


class SearchResult(BaseModel):
    book_name: str
    author_name: str
    content_type: str
    source_file: str
    excerpt: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)
    total: int = 0


# ==================== 问答 ====================

class QARequest(BaseModel):
    query: str = Field(..., description="用户问题")
    session_id: str = Field(default="default", description="会话 ID")
    intent: str = Field(default="qa", description="意图: recommend/detail/note/qa")


class QAResponse(BaseModel):
    answer: str
    sources: list[dict] = Field(default_factory=list)
    session_id: str = "default"


# ==================== 导入 ====================

class ImportRequest(BaseModel):
    directory: str = Field(default="./data/raw", description="要导入的目录路径")


class ImportStatus(BaseModel):
    task_id: str
    status: str
    total_files: int = 0
    processed_files: int = 0
    total_entries: int = 0
    errors: list[dict] = Field(default_factory=list)


# ==================== 对话 ====================

class ConversationInfo(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class MessageInfo(BaseModel):
    id: int
    role: str
    content: str
    sources: list[dict] = Field(default_factory=list)
    created_at: str
