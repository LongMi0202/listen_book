# -*- coding: utf-8 -*-
"""导入路由 —— 文档导入与任务状态"""

import uuid
import logging
from pathlib import Path
from fastapi import APIRouter, Depends
from knowledge.src.api.schemas import ImportRequest, ImportStatus
from knowledge.src.api.dependencies import get_rag_pipeline, get_parser_chunker_embedder
from knowledge.src.parser.markdown_parser import parse_all

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["导入"])


@router.post("/import", response_model=ImportStatus)
async def import_documents(
    request: ImportRequest,
    pipeline=Depends(get_rag_pipeline),
    pce=Depends(get_parser_chunker_embedder),
):
    """导入文档到知识库"""
    retriever, llm_client, meta_store = pipeline
    parser, chunker, embedder = pce

    task_id = str(uuid.uuid4())[:8]
    directory = request.directory

    # 检查目录
    dir_path = Path(directory)
    if not dir_path.exists():
        return ImportStatus(task_id=task_id, status="failed", errors=[{"error": f"目录不存在: {directory}"}])

    md_files = list(dir_path.glob("*.md"))
    if not md_files:
        return ImportStatus(task_id=task_id, status="failed", errors=[{"error": "目录中无 Markdown 文件"}])

    # 创建任务
    meta_store.create_import_task(task_id, len(md_files))
    logger.info(f"开始导入任务 {task_id}: {len(md_files)} 个文件")

    try:
        # 1. 解析
        entries, errors = parse_all(directory)
        meta_store.update_import_progress(task_id, len(md_files), len(entries))

        if not entries:
            meta_store.finish_import_task(task_id, errors)
            return ImportStatus(
                task_id=task_id, status="failed",
                total_files=len(md_files), errors=errors,
            )

        # 2. 分块
        chunks = chunker.chunk_entries(entries)
        logger.info(f"分块: {len(entries)} 条 → {len(chunks)} chunks")

        # 3. 嵌入
        texts = [c["page_content"] for c in chunks]
        embeddings = embedder.embed_documents(texts)

        # 4. 存入向量库
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [c["metadata"] for c in chunks]

        # 清空旧数据（仅首次导入时）
        from knowledge.src.storage.vector_store import VectorStoreManager
        vs = VectorStoreManager(str(Path("./db/chroma").resolve()))
        if vs.count() > 0:
            logger.info(f"向量库已有 {vs.count()} 条数据，追加导入...")

        vs.add_documents(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        # 5. 完成
        meta_store.finish_import_task(task_id, errors)

        return ImportStatus(
            task_id=task_id,
            status="completed" if not errors else "partial_failed",
            total_files=len(md_files),
            processed_files=len(md_files) - len(errors),
            total_entries=len(entries),
            errors=errors,
        )

    except Exception as e:
        logger.error(f"导入失败: {e}", exc_info=True)
        meta_store.finish_import_task(task_id, [{"error": str(e)}])
        return ImportStatus(task_id=task_id, status="failed", errors=[{"error": str(e)}])


@router.get("/import/status/{task_id}", response_model=ImportStatus)
async def get_import_status(task_id: str, pipeline=Depends(get_rag_pipeline)):
    """查询导入任务状态"""
    _, _, meta_store = pipeline
    task = meta_store.get_import_task(task_id)
    if task is None:
        return ImportStatus(task_id=task_id, status="not_found")
    return ImportStatus(**task)
