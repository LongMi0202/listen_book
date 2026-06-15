# -*- coding: utf-8 -*-
"""
启动 API 服务

用法:
    cd knowledge
    .venv\\Scripts\\Activate.ps1
    python scripts/run_api.py
"""

import sys
from pathlib import Path

# 项目根目录 (D:\MySpace\RAG_ListenBook) 添加到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.config import API_HOST, API_PORT

if __name__ == "__main__":
    import uvicorn
    print(f"启动听书知识库 API 服务: http://{API_HOST}:{API_PORT}")
    print(f"API 文档: http://{API_HOST}:{API_PORT}/docs")
    uvicorn.run(
        "knowledge.src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
