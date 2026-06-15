# -*- coding: utf-8 -*-
"""
Streamlit 演示界面
听书知识库 —— 智能问答演示

用法:
    cd knowledge
    .venv\\Scripts\\Activate.ps1
    streamlit run scripts/run_app.py
"""

import sys
import json
from pathlib import Path

# 项目根目录 (D:\MySpace\RAG_ListenBook) 添加到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import requests

# ========== 配置 ==========
st.set_page_config(
    page_title="听书知识库",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://127.0.0.1:8000"

# ========== 样式 ==========
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: bold; color: #1f77b4; margin-bottom: 0; }
    .sub-header { color: #666; font-size: 0.9rem; margin-top: 0; }
    .book-card {
        background: #f8f9fa; border-radius: 10px; padding: 15px;
        margin: 10px 0; border-left: 4px solid #1f77b4;
    }
    .source-tag {
        background: #e8f4f8; color: #1f77b4; padding: 2px 8px;
        border-radius: 4px; font-size: 0.8rem; margin: 2px;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ========== 侧边栏 ==========
with st.sidebar:
    st.markdown("## 📚 听书知识库")
    st.markdown("---")

    mode = st.radio(
        "功能模式",
        ["💬 知识问答", "🎯 书籍推荐", "📖 书籍详情", "🔍 内容检索"],
    )

    st.markdown("---")
    st.markdown("### ⚙️ 设置")

    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]

    st.text_input("会话 ID", value=st.session_state.session_id, disabled=True)

    if st.button("🔄 新建会话"):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 系统状态")

    try:
        resp = requests.get(f"{API_BASE}/api/health", timeout=3)
        if resp.ok:
            data = resp.json()
            st.success(f"✅ 服务正常 | 向量库: {data.get('vector_count', '?')} 条")
        else:
            st.error("❌ 服务异常")
    except Exception:
        st.error("❌ 服务未连接")
        st.info("请先启动 API: `python scripts/run_api.py`")

# ========== 主界面 ==========
st.markdown('<p class="main-header">📚 听书知识库</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">智能书籍推荐 · 详情查询 · 知识问答</p>', unsafe_allow_html=True)

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📖 引用来源"):
                for s in msg["sources"]:
                    st.markdown(
                        f'<span class="source-tag">📚 《{s.get("book_name", "")}》</span> '
                        f'<span class="source-tag">✍ {s.get("author_name", "")}</span> '
                        f'<span class="source-tag">📂 {s.get("content_type", "")}</span>',
                        unsafe_allow_html=True,
                    )
                    if s.get("excerpt"):
                        st.caption(s["excerpt"][:150])

# ========== 输入处理 ==========
if prompt := st.chat_input("请输入你的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 根据模式选择 API
    mode_map = {
        "💬 知识问答": ("/api/qa/stream", "qa"),
        "🎯 书籍推荐": ("/api/recommend", "recommend"),
        "📖 书籍详情": ("/api/detail", "detail"),
        "🔍 内容检索": ("/api/search", "search"),
    }
    endpoint, intent = mode_map[mode]

    with st.chat_message("assistant"):
        if endpoint == "/api/qa/stream":
            # 流式输出
            response_placeholder = st.empty()
            full_answer = ""

            try:
                resp = requests.post(
                    f"{API_BASE}{endpoint}",
                    json={
                        "query": prompt,
                        "session_id": st.session_state.session_id,
                        "intent": intent,
                    },
                    stream=True,
                    timeout=60,
                )

                sources = []
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data:"):
                        data_str = line_str[5:].strip()
                        if data_str.startswith("{"):
                            data = json.loads(data_str)
                            if "session_id" in data:
                                pass  # done event
                        else:
                            full_answer += data_str
                            response_placeholder.markdown(full_answer + "▌")

                    elif line_str.startswith("event: sources"):
                        pass  # sources 在下一行
                    elif line_str.startswith("event: done"):
                        response_placeholder.markdown(full_answer)
                        break

            except Exception as e:
                response_placeholder.error(f"请求失败: {e}")
                full_answer = f"抱歉，请求失败了：{e}"

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": [],
            })

        elif endpoint == "/api/search":
            # 检索模式
            with st.spinner("检索中..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}{endpoint}",
                        json={"query": prompt, "top_k": 10},
                        timeout=30,
                    )
                    data = resp.json()
                    results = data.get("results", [])

                    if results:
                        st.markdown(f"**找到 {len(results)} 条相关结果:**")
                        for r in results:
                            with st.container():
                                st.markdown(f"""<div class="book-card">
                                <strong>📚 {r['book_name']}</strong> — {r['author_name']}<br>
                                <small>类型: {r['content_type']} | 文件: {r['source_file']}</small><br>
                                <p>{r['excerpt'][:200]}...</p>
                                </div>""", unsafe_allow_html=True)
                    else:
                        st.info("未找到相关内容")

                    full_answer = f"检索完成，共 {len(results)} 条结果"
                except Exception as e:
                    st.error(f"检索失败: {e}")
                    full_answer = f"检索失败: {e}"

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": results,
            })

        else:
            # 非流式（推荐/详情）
            with st.spinner("思考中..."):
                try:
                    body = {"query": prompt, "top_k": 5}
                    if mode == "📖 书籍详情":
                        # 尝试从问题中提取书名
                        body = {"book_name": prompt.replace("介绍", "").replace("查询", "").strip(), "query": prompt}

                    resp = requests.post(
                        f"{API_BASE}{endpoint}",
                        json=body,
                        timeout=60,
                    )
                    data = resp.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("📖 引用来源"):
                            for s in sources:
                                st.markdown(
                                    f'**《{s.get("book_name", "")}》** — {s.get("author_name", "")} '
                                    f'| {s.get("content_type", "")}',
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    st.error(f"请求失败: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"抱歉，请求失败: {e}",
                        "sources": [],
                    })
