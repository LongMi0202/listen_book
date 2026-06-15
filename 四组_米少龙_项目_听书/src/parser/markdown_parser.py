# -*- coding: utf-8 -*-
"""
Markdown 文档解析器
解析听书知识库的 Markdown 文件，提取元数据和结构化章节
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """内容类型枚举 —— 对应需求中的 7 种类型"""
    AUDIOBOOK_INFO = "有声书信息"
    BOOK_INTRO = "书籍简介"
    AUTHOR_INTRO = "作者介绍"
    LISTENING_NOTE = "听书笔记"
    RECOMMENDATION = "推荐运营资料"
    USER_REVIEW = "用户评论摘要"
    FAQ = "常见问答"


@dataclass
class KnowledgeEntry:
    """单条知识记录"""
    content_type: ContentType
    body: str                            # 内容正文
    book_name: str                       # 书名
    author_name: str                     # 作者名
    entry_name: str = ""                 # 条目名称
    tags: list = field(default_factory=list)  # 类别/标签
    duration: Optional[str] = None       # 有声书时长
    source_file: str = ""                # 来源文件名
    source_path: str = ""                # 来源路径

    def to_metadata_dict(self) -> dict:
        """转为 ChromaDB metadata 字典"""
        return {
            "book_name": self.book_name,
            "author_name": self.author_name,
            "content_type": self.content_type.value,
            "tags": ", ".join(self.tags) if self.tags else "",
            "entry_name": self.entry_name,
            "duration": self.duration or "",
            "source_file": self.source_file,
        }

    def to_page_content(self) -> str:
        """生成包含元数据前缀的检索文本"""
        meta_prefix = (
            f"[书名：{self.book_name}] "
            f"[作者：{self.author_name}] "
            f"[类型：{self.content_type.value}] "
            f"[标签：{', '.join(self.tags) if self.tags else '无'}]"
        )
        return f"{meta_prefix}\n{self.body}"


class MarkdownParser:
    """
    听书知识库 Markdown 解析器

    支持的文档结构：
    # {书名}简介
    ## 元数据
    - 书名：xxx
    - 作者名：xxx
    ...
    ## 内容正文
    ...
    ## 作者介绍
    ...
    （其他章节）
    """

    # 章节标题 → 内容类型映射
    SECTION_MAPPING = {
        "内容正文": ContentType.BOOK_INTRO,
        "作者介绍": ContentType.AUTHOR_INTRO,
        "有声书信息": ContentType.AUDIOBOOK_INFO,
        "推荐语": ContentType.RECOMMENDATION,
        "核心看点": ContentType.RECOMMENDATION,
        "适合人群": ContentType.RECOMMENDATION,
        "收听建议": ContentType.RECOMMENDATION,
        "常见问答": ContentType.FAQ,
        "听书笔记": ContentType.LISTENING_NOTE,
        "用户评论摘要": ContentType.USER_REVIEW,
    }

    # 元数据字段映射
    META_KEY_MAP = {
        "书名": "book_name",
        "作者名": "author_name",
        "条目名称": "entry_name",
        "类别/标签": "tags",
    }

    def parse(self, file_path: str) -> list[KnowledgeEntry]:
        """解析单个 Markdown 文件，返回知识条目列表"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        text = path.read_text(encoding="utf-8")
        source_file = path.name
        source_path = str(path.resolve())

        # 提取全局元数据
        metadata = self._extract_metadata(text)
        book_name = metadata.get("book_name", "")
        author_name = metadata.get("author_name", "")
        entry_name = metadata.get("entry_name", "")
        tags = metadata.get("tags", [])

        # 提取有声书时长（从有声书信息章节）
        duration = self._extract_duration(text)

        # 按 ## 拆分章节
        sections = self._split_sections(text)

        entries: list[KnowledgeEntry] = []
        for section_title, section_body in sections:
            content_type = self.SECTION_MAPPING.get(section_title)
            if content_type is None:
                continue  # 跳过元数据等非内容章节

            if not section_body.strip():
                continue

            # 常见问答需要进一步拆分为独立 QA 对
            if content_type == ContentType.FAQ:
                qa_pairs = self._parse_faq(section_body)
                for qa in qa_pairs:
                    entries.append(KnowledgeEntry(
                        content_type=ContentType.FAQ,
                        body=qa,
                        book_name=book_name,
                        author_name=author_name,
                        entry_name=entry_name,
                        tags=tags,
                        duration=duration,
                        source_file=source_file,
                        source_path=source_path,
                    ))
            else:
                entries.append(KnowledgeEntry(
                    content_type=content_type,
                    body=section_body.strip(),
                    book_name=book_name,
                    author_name=author_name,
                    entry_name=entry_name,
                    tags=tags,
                    duration=duration if content_type == ContentType.AUDIOBOOK_INFO else None,
                    source_file=source_file,
                    source_path=source_path,
                ))

        logger.info(f"解析完成: {source_file} → {len(entries)} 条知识记录")
        return entries

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        """按 ## 二级标题拆分章节，返回 (标题, 正文) 列表"""
        # 匹配 ## 标题行
        pattern = r'^## (.+)$'
        lines = text.split('\n')
        sections = []
        current_title = None
        current_lines = []

        for line in lines:
            m = re.match(pattern, line.strip())
            if m:
                if current_title is not None:
                    sections.append((current_title, '\n'.join(current_lines)))
                current_title = m.group(1).strip()
                current_lines = []
            else:
                if current_title is not None:
                    current_lines.append(line)

        # 最后一个章节
        if current_title is not None:
            sections.append((current_title, '\n'.join(current_lines)))

        return sections

    def _extract_metadata(self, text: str) -> dict:
        """从元数据章节提取结构化的键值对"""
        # 定位 ## 元数据 章节
        meta_match = re.search(r'## 元数据\n(.*?)(?=\n## |\Z)', text, re.DOTALL)
        if not meta_match:
            logger.warning("未找到「元数据」章节")
            return {}

        meta_text = meta_match.group(1)
        result: dict = {"tags": []}

        for line in meta_text.strip().split('\n'):
            line = line.strip()
            # 匹配 - 键：值
            m = re.match(r'[-*]\s*(.+?)[：:]\s*(.+)', line)
            if not m:
                continue

            key_raw = m.group(1).strip()
            value = m.group(2).strip()

            mapped_key = self.META_KEY_MAP.get(key_raw)
            if mapped_key is None:
                continue

            if mapped_key == "tags":
                result["tags"] = [t.strip() for t in value.split(',') if t.strip()]
            else:
                result[mapped_key] = value

        return result

    def _extract_duration(self, text: str) -> Optional[str]:
        """从有声书信息章节提取时长"""
        # 匹配 "约 X 小时" 或 "X 小时" 等模式
        duration_match = re.search(r'时长[大约]*\s*(\d+[\d.]*\s*小时[^\n，。]*)', text)
        if duration_match:
            return duration_match.group(1).strip()

        # 备用匹配
        duration_match = re.search(r'(\d+[\d.]*\s*小时)', text)
        if duration_match:
            return duration_match.group(1).strip()

        return None

    def _parse_faq(self, text: str) -> list[str]:
        """解析常见问答章节，返回独立 QA 对列表"""
        qa_pairs = []
        # 匹配 "问题：... 回答：..." 模式
        pattern = r'问题[：:]\s*(.+?)\s*\n回答[：:]\s*(.+?)(?=\n问题|$)'
        for m in re.finditer(pattern, text, re.DOTALL):
            question = m.group(1).strip()
            answer = m.group(2).strip()
            qa_pairs.append(f"问：{question}\n答：{answer}")

        if not qa_pairs:
            # 如果没有匹配到 Q&A 格式，整段作为一个条目
            qa_pairs.append(text.strip())

        return qa_pairs


def parse_all(directory: str) -> tuple[list[KnowledgeEntry], list[dict]]:
    """
    批量解析目录下所有 Markdown 文件
    返回: (成功条目列表, 错误列表)
    单个文件解析失败不影响其他文件
    """
    parser = MarkdownParser()
    all_entries: list[KnowledgeEntry] = []
    errors: list[dict] = []

    data_path = Path(directory)
    md_files = list(data_path.glob("*.md"))

    if not md_files:
        logger.warning(f"目录中无 Markdown 文件: {directory}")
        return all_entries, errors

    logger.info(f"发现 {len(md_files)} 个 Markdown 文件，开始解析...")

    for file_path in sorted(md_files):
        try:
            entries = parser.parse(str(file_path))
            all_entries.extend(entries)
            logger.info(f"  ✓ {file_path.name} → {len(entries)} 条")
        except Exception as e:
            error_info = {"file": str(file_path), "error": str(e)}
            errors.append(error_info)
            logger.warning(f"  ✗ {file_path.name} 解析失败: {e}")

    logger.info(f"解析完成: 成功 {len(all_entries)} 条, 失败 {len(errors)} 个文件")
    return all_entries, errors
