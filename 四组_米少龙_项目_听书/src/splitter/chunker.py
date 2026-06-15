# -*- coding: utf-8 -*-
"""文本分块器 —— 将解析后的知识条目切分为适合嵌入检索的片段"""

import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge.src.parser.markdown_parser import KnowledgeEntry, ContentType

logger = logging.getLogger(__name__)


class AudiobookChunker:
    """
    听书知识库文本分块器

    策略：
    - 默认 500 字符/块，100 字符重叠
    - FAQ 类型不做进一步拆分（每个 QA 对已经是独立单元）
    - 极短内容（<200 字符）保持完整
    - 每块携带完整的元数据信息
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
            keep_separator=True,
        )

    def chunk_entries(self, entries: list[KnowledgeEntry]) -> list[dict]:
        """
        将知识条目列表转为带元数据的 chunk 列表

        每个 chunk 结构:
        {
            "page_content": str,   # 检索文本（含元数据前缀）
            "metadata": dict,      # 元数据（用于过滤和展示）
        }
        """
        chunks = []
        for i, entry in enumerate(entries):
            entry_chunks = self._chunk_single(entry, i)
            chunks.extend(entry_chunks)

        logger.info(
            f"分块完成: {len(entries)} 条记录 → {len(chunks)} 个 chunk "
            f"(平均 {len(entries)/max(len(chunks),1):.1f} 条/chunk)"
        )
        return chunks

    def _chunk_single(self, entry: KnowledgeEntry, entry_index: int) -> list[dict]:
        """对单条知识记录进行分块"""
        text = entry.to_page_content()
        metadata = entry.to_metadata_dict()

        # FAQ 和极短内容不拆分
        if entry.content_type == ContentType.FAQ or len(text) < self.chunk_size * 0.4:
            return [{
                "page_content": text,
                "metadata": {**metadata, "chunk_index": 0, "entry_index": entry_index},
            }]

        # 使用 LangChain 分块器
        try:
            split_texts = self.splitter.split_text(text)
        except Exception:
            # 降级：手动按段落拆分
            split_texts = [p.strip() for p in text.split('\n\n') if p.strip()]

        results = []
        for j, chunk_text in enumerate(split_texts):
            if not chunk_text.strip():
                continue
            results.append({
                "page_content": chunk_text,
                "metadata": {**metadata, "chunk_index": j, "entry_index": entry_index},
            })

        return results
