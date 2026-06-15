"""Quick RAG output test - saves to file to avoid GBK encoding issues"""
import sys
from pathlib import Path

# 项目根目录 (D:\MySpace\RAG_ListenBook) 添加到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.config import (
    EMBEDDING_MODEL, EMBEDDING_PROVIDER,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL, CHROMA_DIR,
)
from knowledge.src.embedding.embedder import Embedder
from knowledge.src.storage.vector_store import VectorStoreManager
from knowledge.src.retrieval.retriever import HybridRetriever
from knowledge.src.generation.llm_client import LLMClient

embedder = Embedder(
    model_name=EMBEDDING_MODEL,
    provider="api",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)
vs = VectorStoreManager(str(CHROMA_DIR))
retriever = HybridRetriever(vector_store=vs, embedder=embedder)
llm = LLMClient(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model=LLM_MODEL,
)

# Test 1: Recommend
print("Running Recommend test...")
docs = retriever.retrieve_for_recommend(query="推荐科幻类有声书", top_k=5)
resp = llm.generate_answer(
    query="推荐科幻类有声书",
    context_docs=docs,
    intent="recommend",
    stream=False,
)
answer = resp.choices[0].message.content
out_path = PROJECT_ROOT / "knowledge" / "tests" / "test_output_recommend.txt"
out_path.write_text(answer, encoding="utf-8")
print(f"Recommend: {resp.usage.completion_tokens} tokens -> {out_path}")

# Test 2: Q&A
print("Running Q&A test...")
docs2 = retriever.retrieve("三体适合新手吗", top_k=5)
resp2 = llm.generate_answer(
    query="三体这本书适合科幻新手吗？",
    context_docs=docs2,
    intent="qa",
    stream=False,
)
answer2 = resp2.choices[0].message.content
out_path2 = PROJECT_ROOT / "knowledge" / "tests" / "test_output_qa.txt"
out_path2.write_text(answer2, encoding="utf-8")
print(f"QA: {resp2.usage.completion_tokens} tokens -> {out_path2}")

print("\nAll tests passed!")
