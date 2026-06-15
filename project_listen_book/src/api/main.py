# -*- coding: utf-8 -*-
"""
FastAPI 主入口
听书知识库 RAG 系统 API 服务

启动方式:
    cd knowledge
    .venv\\Scripts\\Activate.ps1
    python -m knowledge.src.api.main

或:
    uvicorn knowledge.src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import logging
from pathlib import Path

# 项目根目录 (D:\MySpace\RAG_ListenBook) 添加到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from knowledge.src.api.routes import qa, recommend, detail, search, conversation, import_

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 创建应用
app = FastAPI(
    title="听书知识库 RAG 系统",
    description="智能听书知识库 —— 书籍推荐、详情查询、内容检索、知识问答",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(qa.router)
app.include_router(recommend.router)
app.include_router(detail.router)
app.include_router(search.router)
app.include_router(conversation.router)
app.include_router(import_.router)


@app.get("/")
async def root():
    return {
        "name": "听书知识库 RAG 系统",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/health")
async def health():
    """健康检查 + 系统状态"""
    from knowledge.src.api.dependencies import _retriever, _embedder, _meta_store
    return {
        "status": "ok",
        "vector_count": _retriever.vector_store.count() if _retriever else "未初始化",
        "embedding_dim": _embedder.dimension if _embedder else "未初始化",
    }


@app.get("/api/books")
async def list_books():
    """列出知识库中所有书籍及内容统计"""
    from knowledge.src.api.dependencies import _retriever
    if _retriever is None:
        return {"books": [], "note": "请先导入数据（POST /api/import）"}

    # 通过向量库元数据获取去重后的书籍列表
    vs = _retriever.vector_store
    collection = vs.collection
    all_meta = collection.get()["metadatas"]

    books = {}
    for meta in all_meta:
        name = meta.get("book_name", "")
        if name and name not in books:
            books[name] = {
                "book_name": name,
                "author_name": meta.get("author_name", ""),
                "tags": meta.get("tags", ""),
            }

    return {
        "books": list(books.values()),
        "total": len(books),
    }


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    from knowledge.config import API_HOST, API_PORT

    logger.info(f"启动听书知识库 API 服务: http://{API_HOST}:{API_PORT}")
    logger.info(f"API 文档: http://{API_HOST}:{API_PORT}/docs")

    uvicorn.run(
        "knowledge.src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
