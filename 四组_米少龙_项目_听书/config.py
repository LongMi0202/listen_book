# -*- coding: utf-8 -*-
"""全局配置 —— 从 .env 读取环境变量"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent

# 加载 .env
load_dotenv(ROOT_DIR / ".env")

# ===== LLM 配置 =====
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3")

# ===== 嵌入模型配置 =====
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "api")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

# ===== 服务配置 =====
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ===== 数据路径 =====
DATA_DIR = ROOT_DIR / os.getenv("DATA_DIR", "./data/raw")
CHROMA_DIR = ROOT_DIR / os.getenv("CHROMA_DIR", "./db/chroma")
SQLITE_PATH = ROOT_DIR / os.getenv("SQLITE_PATH", "./db/metadata.db")

# ===== 分块参数 =====
CHUNK_SIZE = 500       # 每块字符数
CHUNK_OVERLAP = 100    # 重叠字符数

# ===== 检索参数 =====
RETRIEVAL_TOP_K = 8    # 默认返回 Top-K

# ===== 生成参数 =====
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2000
