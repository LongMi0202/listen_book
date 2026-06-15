# -*- coding: utf-8 -*-
"""端到端 RAG 管道测试"""
import sys
from pathlib import Path

# 项目根目录 (D:\MySpace\RAG_ListenBook) 添加到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.config import (
    EMBEDDING_MODEL, EMBEDDING_PROVIDER,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL,
    CHROMA_DIR,
)
from knowledge.src.embedding.embedder import Embedder
from knowledge.src.storage.vector_store import VectorStoreManager
from knowledge.src.retrieval.retriever import HybridRetriever
from knowledge.src.generation.llm_client import LLMClient

def main():
    # 初始化组件
    print("Initializing components...")
    embedder = Embedder(
        model_name=EMBEDDING_MODEL, provider="api",
        api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
    )
    vs = VectorStoreManager(str(CHROMA_DIR))
    retriever = HybridRetriever(vector_store=vs, embedder=embedder)
    llm = LLMClient(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, model=LLM_MODEL)

    print(f"Vector count: {vs.count()}")
    print(f"Embed dim: {embedder.dimension}")
    print()

    # ---- Test 1: Semantic Search ----
    print("=" * 60)
    print("TEST 1: Semantic Search - 'recommend sci-fi audiobooks'")
    print("=" * 60)
    docs = retriever.retrieve("推荐科幻类有声书", top_k=5)
    for i, d in enumerate(docs):
        print(f"  {i+1}. [{d.metadata['book_name']}] [{d.metadata['content_type']}] score={1-d.score:.3f}")

    # ---- Test 2: Filtered Search ----
    print()
    print("=" * 60)
    print("TEST 2: Filtered Search - content_type=FAQ")
    print("=" * 60)
    docs2 = retriever.retrieve("适合什么人", top_k=5, content_type="常见问答")
    for i, d in enumerate(docs2):
        print(f"  {i+1}. [{d.metadata['book_name']}] score={1-d.score:.3f}")

    # ---- Test 3: Book Detail ----
    print()
    print("=" * 60)
    print("TEST 3: Book Detail - 'San Ti' all contents")
    print("=" * 60)
    docs3 = retriever.retrieve_for_detail(book_name="三体", top_k=8)
    types = set(d.metadata['content_type'] for d in docs3)
    print(f"  Found {len(docs3)} docs covering {len(types)} content types: {types}")

    # ---- Test 4: Search with tags ----
    print()
    print("=" * 60)
    print("TEST 4: Tag filter - tags contains '科幻'")
    print("=" * 60)
    docs4 = retriever.retrieve("宏大的世界观", top_k=5, tags=["科幻"])
    for i, d in enumerate(docs4):
        print(f"  {i+1}. [{d.metadata['book_name']}] tags={d.metadata['tags']}")

    # ---- Test 5: RAG Generation (Recommend) ----
    print()
    print("=" * 60)
    print("TEST 5: RAG Recommend - 'recommend sci-fi books'")
    print("=" * 60)
    docs5 = retriever.retrieve_for_recommend(query="推荐科幻类有声书", top_k=5)
    print(f"  Retrieved {len(docs5)} docs for recommendation")
    try:
        resp = llm.generate_answer(
            query="推荐科幻类有声书",
            context_docs=docs5,
            intent="recommend",
            stream=False,
        )
        answer = resp.choices[0].message.content
        print(f"  LLM Response ({resp.usage.completion_tokens} tokens):")
        print("  " + answer[:400].replace("\n", "\n  "))
        if len(answer) > 400:
            print("  ...")
    except Exception as e:
        print(f"  LLM Error: {e}")

    # ---- Test 6: RAG Generation (QA) ----
    print()
    print("=" * 60)
    print("TEST 6: RAG Q&A - 'Is San Ti suitable for beginners?'")
    print("=" * 60)
    docs6 = retriever.retrieve("三体适合新手吗", top_k=5)
    print(f"  Retrieved {len(docs6)} docs")
    try:
        resp = llm.generate_answer(
            query="三体这本书适合科幻新手吗？",
            context_docs=docs6,
            intent="qa",
            stream=False,
        )
        answer = resp.choices[0].message.content
        print(f"  LLM Response ({resp.usage.completion_tokens} tokens):")
        print("  " + answer[:400].replace("\n", "\n  "))
        if len(answer) > 400:
            print("  ...")
    except Exception as e:
        print(f"  LLM Error: {e}")

    # ---- Summary ----
    print()
    print("=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    # Print sources from test 5
    print()
    print("--- Test 5 Sources ---")
    for s in llm.extract_sources(docs5):
        print(f"  {s['book_name']} / {s['author_name']} / {s['content_type']}")


if __name__ == "__main__":
    main()
