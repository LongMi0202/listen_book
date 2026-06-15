# -*- coding: utf-8 -*-
"""LLM 客户端 —— 封装大模型 API 调用"""

import logging
from typing import Optional, Generator
from openai import OpenAI

from knowledge.src.retrieval.retriever import RetrievedDocument
from knowledge.src.generation.prompt_template import get_prompt

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM 客户端

    支持 SiliconFlow / DeepSeek / OpenAI 等兼容接口
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        model: str = "deepseek-ai/DeepSeek-V3",
    ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        logger.info(f"LLM 客户端已初始化: {base_url} / {model}")

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False,
    ):
        """调用 LLM 对话"""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Generator[str, None, None]:
        """流式对话，逐 token 返回"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def generate_answer(
        self,
        query: str,
        context_docs: list[RetrievedDocument],
        intent: str = "qa",
        history: Optional[list[dict]] = None,
        stream: bool = False,
    ):
        """
        RAG 生成管道

        参数:
            query: 用户查询
            context_docs: 检索到的文档列表
            intent: 意图类型 (recommend/detail/note/qa)
            history: 历史对话 [{"role": "user", "content": "..."}, ...]
            stream: 是否流式输出

        返回:
            stream=False 时返回完整响应对象
            stream=True 时返回生成器
        """
        # 格式化上下文
        context = self._format_context(context_docs)

        # 选择 Prompt 模板
        prompt_template = get_prompt(intent)
        system_prompt = prompt_template.format(query=query, context=context)

        # 构建消息
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages = messages + history
        messages.append({"role": "user", "content": query})

        logger.debug(f"RAG 生成: intent={intent}, context_docs={len(context_docs)}, stream={stream}")

        if stream:
            return self.chat_stream(messages)
        else:
            return self.chat(messages)

    def _format_context(self, docs: list[RetrievedDocument]) -> str:
        """格式化检索文档为上下文文本"""
        if not docs:
            return "（未检索到相关知识库内容）"

        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            header = (
                f"【参考 {i}】"
                f"《{meta.get('book_name', '未知')}》"
                f" — {meta.get('author_name', '未知作者')}"
                f" | 类型：{meta.get('content_type', '未知')}"
            )
            parts.append(f"{header}\n{doc.content}")

        return "\n\n---\n\n".join(parts)

    def extract_sources(self, docs: list[RetrievedDocument]) -> list[dict]:
        """从检索结果提取引用来源"""
        seen = set()
        sources = []
        for doc in docs:
            meta = doc.metadata
            key = (meta.get("book_name"), meta.get("content_type"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "book_name": meta.get("book_name", ""),
                    "author_name": meta.get("author_name", ""),
                    "content_type": meta.get("content_type", ""),
                    "source_file": meta.get("source_file", ""),
                    "excerpt": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                })
        return sources
