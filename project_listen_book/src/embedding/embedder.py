# -*- coding: utf-8 -*-
"""嵌入模型封装 —— 支持 API（SiliconFlow/OpenAI）和本地模型"""

import logging
import numpy as np
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class Embedder:
    """
    嵌入模型封装

    API 模式（默认）: 调用 SiliconFlow / OpenAI 兼容的嵌入 API
    本地模式: 使用 sentence-transformers 加载本地模型
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        provider: str = "api",
        api_key: str = "",
        base_url: str = "https://api.siliconflow.cn/v1",
    ):
        self.model_name = model_name
        self.provider = provider
        self._model = None
        self._client = None

        if provider == "api":
            if not api_key:
                raise ValueError("API 模式需要提供 api_key")
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"嵌入 API 客户端已初始化: {base_url} / {model_name}")
        elif provider == "local":
            self._init_local_model()
        else:
            raise ValueError(f"不支持的嵌入提供商: {provider}")

    def _init_local_model(self):
        """初始化本地嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载本地嵌入模型: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("嵌入模型加载完成")
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文档"""
        if self.provider == "api":
            return self._embed_api(texts)
        elif self._model is not None:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 50,
            )
            return embeddings.tolist()
        else:
            raise RuntimeError("嵌入模型未初始化")

    def embed_query(self, query: str) -> list[float]:
        """嵌入查询文本"""
        if self.provider == "api":
            results = self._embed_api([query])
            return results[0]
        elif self._model is not None:
            embedding = self._model.encode(query, normalize_embeddings=True)
            return embedding.tolist()
        else:
            raise RuntimeError("嵌入模型未初始化")

    def _embed_api(self, texts: list[str]) -> list[list[float]]:
        """
        通过 API 批量嵌入

        SiliconFlow embeddings API 兼容 OpenAI 格式:
        POST /v1/embeddings
        """
        # API 通常有单次请求大小限制，分批处理
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )
                # 按 index 排序确保顺序正确
                batch_embeddings = sorted(response.data, key=lambda x: x.index)
                all_embeddings.extend([
                    self._normalize(e.embedding) for e in batch_embeddings
                ])
            except Exception as e:
                logger.error(f"嵌入 API 调用失败 (batch {i // batch_size}): {e}")
                raise

        return all_embeddings

    def _normalize(self, vec: list[float]) -> list[float]:
        """L2 归一化"""
        arr = np.array(vec)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    @property
    def dimension(self) -> Optional[int]:
        """返回嵌入向量维度"""
        if self.provider == "api":
            # BGE-large-zh-v1.5 是 1024 维
            return 1024
        elif self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return None
