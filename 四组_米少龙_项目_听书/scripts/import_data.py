# -*- coding: utf-8 -*-
"""
数据导入脚本 —— 将原始 Markdown 文档导入知识库

用法:
    cd knowledge
    .venv\\Scripts\\Activate.ps1
    python scripts/import_data.py
    python scripts/import_data.py --input ../data/数据/
    python scripts/import_data.py --input ./data/raw --reset
"""

import sys
import argparse
import logging
import time
from pathlib import Path

# 项目根目录 (D:\MySpace\RAG_ListenBook) 添加到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="听书知识库数据导入工具")
    parser.add_argument("--input", "-i", default=str(PROJECT_ROOT / "knowledge" / "data" / "raw"),
                        help="要导入的 Markdown 文件目录 (默认: ./data/raw)")
    parser.add_argument("--reset", "-r", action="store_true",
                        help="导入前清空已有向量数据")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="文本分块大小 (默认: 500 字符)")
    parser.add_argument("--chunk-overlap", type=int, default=100,
                        help="分块重叠大小 (默认: 100 字符)")
    args = parser.parse_args()

    from knowledge.config import (
        EMBEDDING_MODEL, EMBEDDING_PROVIDER,
        CHROMA_DIR, SQLITE_PATH, DATA_DIR,
        DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    )

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"[ERROR] 目录不存在: {input_dir}")
        sys.exit(1)

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        print(f"[ERROR] 目录中无 Markdown 文件: {input_dir}")
        sys.exit(1)

    print()
    print("=" * 55)
    print("  听书知识库数据导入工具")
    print("=" * 55)
    print(f"  导入目录: {input_dir}")
    print(f"  文件数量: {len(md_files)}")
    print(f"  分块参数: {args.chunk_size} / {args.chunk_overlap}")
    print(f"  嵌入方式: {EMBEDDING_PROVIDER} ({EMBEDDING_MODEL})")
    print()

    # ===== 初始化嵌入模型 =====
    print("[1/5] 加载嵌入模型...")
    from knowledge.src.embedding.embedder import Embedder
    if EMBEDDING_PROVIDER == "api":
        embedder = Embedder(
            model_name=EMBEDDING_MODEL, provider="api",
            api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
        )
    else:
        embedder = Embedder(model_name=EMBEDDING_MODEL, provider="local")
    print(f"  -> 嵌入模型加载完成")

    # ===== 初始化向量存储 =====
    from knowledge.src.storage.vector_store import VectorStoreManager
    vector_store = VectorStoreManager(str(CHROMA_DIR))
    print(f"  -> 向量存储就绪 (已有 {vector_store.count()} 条)")

    if args.reset and vector_store.count() > 0:
        print("[WARN] 清空已有向量数据...")
        vector_store.reset()

    # ===== 解析文档 =====
    print()
    print("[2/5] 解析 Markdown 文档...")
    from knowledge.src.parser.markdown_parser import MarkdownParser
    mp = MarkdownParser()
    entries = []
    errors = []

    for i, f in enumerate(md_files):
        try:
            file_entries = mp.parse(str(f))
            entries.extend(file_entries)
            print(f"  [{i+1}/{len(md_files)}] OK {f.name} -> {len(file_entries)} 条")
        except Exception as e:
            errors.append({"file": str(f), "error": str(e)})
            print(f"  [{i+1}/{len(md_files)}] FAIL {f.name}: {e}")

    print(f"  -> 解析完成: {len(entries)} 条记录, {len(errors)} 个失败")

    if not entries:
        print("[ERROR] 没有可用的解析结果，导入中止")
        sys.exit(1)

    # ===== 分块 =====
    print()
    print("[3/5] 文本分块...")
    from knowledge.src.splitter.chunker import AudiobookChunker
    chunker = AudiobookChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = chunker.chunk_entries(entries)
    print(f"  -> 分块完成: {len(entries)} 条 -> {len(chunks)} chunks")

    # ===== 嵌入 =====
    print()
    print("[4/5] 生成嵌入向量...")
    texts = [c["page_content"] for c in chunks]
    t0 = time.time()
    print(f"  正在为 {len(texts)} 个文本块生成嵌入...")

    embeddings = embedder.embed_documents(texts)
    elapsed = time.time() - t0
    print(f"  -> 嵌入完成: {len(embeddings)} 个向量, 维度 {len(embeddings[0])}, 耗时 {elapsed:.1f}s")

    # ===== 写入向量库 =====
    print()
    print("[5/5] 写入向量数据库...")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [c["metadata"] for c in chunks]

    vector_store.add_documents(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print(f"  -> 已写入 {len(ids)} 条向量")
    print(f"  -> 向量库总量: {vector_store.count()} 条")

    # ===== 完成 =====
    print()
    print("=" * 55)
    print("  导入完成!")
    print("=" * 55)
    print()
    print("  接下来可以:")
    print("    1. 启动 API:    python scripts/run_api.py")
    print("    2. 启动演示:    streamlit run scripts/run_app.py")
    print("    3. 查看文档:    http://localhost:8000/docs")
    print()

    if errors:
        print(f"[WARN] {len(errors)} 个文件解析失败")


if __name__ == "__main__":
    main()
