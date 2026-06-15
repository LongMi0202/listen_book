# -*- coding: utf-8 -*-
"""元数据存储 —— SQLite 操作封装"""

import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class MetaStore:
    """
    SQLite 元数据存储

    管理表:
    - import_tasks:  导入任务状态
    - sessions:      会话记录
    - messages:      对话消息
    """

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()
        logger.info(f"元数据存储已就绪: {db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_tables(self):
        """初始化所有表"""
        with self._get_conn() as conn:
            conn.executescript("""
                -- 导入任务表
                CREATE TABLE IF NOT EXISTS import_tasks (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    total_files INTEGER DEFAULT 0,
                    processed_files INTEGER DEFAULT 0,
                    total_entries INTEGER DEFAULT 0,
                    errors TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    finished_at TIMESTAMP
                );

                -- 会话表
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT DEFAULT '新对话',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- 消息表
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
            """)
            conn.commit()

    # ==================== 导入任务 ====================

    def create_import_task(self, task_id: str, total_files: int) -> str:
        """创建导入任务，返回 task_id"""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO import_tasks (id, total_files, status) VALUES (?, ?, 'running')",
                (task_id, total_files),
            )
            conn.commit()
        return task_id

    def update_import_progress(self, task_id: str, processed: int, entries: int):
        """更新导入进度"""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE import_tasks SET processed_files = ?, total_entries = ? WHERE id = ?",
                (processed, entries, task_id),
            )
            conn.commit()

    def finish_import_task(self, task_id: str, errors: list[dict]):
        """完成导入任务"""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE import_tasks SET status = ?, errors = ?, finished_at = ? WHERE id = ?",
                ("completed" if not errors else "partial_failed",
                 json.dumps(errors, ensure_ascii=False),
                 datetime.now().isoformat(),
                 task_id),
            )
            conn.commit()

    def get_import_task(self, task_id: str) -> Optional[dict]:
        """查询导入任务状态"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM import_tasks WHERE id = ?", (task_id,)
            ).fetchone()
        if row:
            return dict(row)
        return None

    # ==================== 会话管理 ====================

    def create_session(self, session_id: str, title: str = "新对话") -> str:
        """创建会话"""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, title) VALUES (?, ?)",
                (session_id, title),
            )
            conn.commit()
        return session_id

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """列出最近会话"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str):
        """删除会话及其消息"""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()

    # ==================== 消息管理 ====================

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[list[dict]] = None,
    ):
        """添加一条消息"""
        # 确保会话存在
        self.create_session(session_id)

        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, sources) VALUES (?, ?, ?, ?)",
                (session_id, role, content, json.dumps(sources or [], ensure_ascii=False)),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (datetime.now().isoformat(), session_id),
            )
            conn.commit()

    def get_recent_messages(self, session_id: str, limit: int = 10) -> list[dict]:
        """获取会话最近 N 轮消息"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit * 2),  # user + assistant 各一条
            ).fetchall()
        # 按时间正序返回
        return [dict(r) for r in reversed(rows)]

    def get_session_messages(self, session_id: str) -> list[dict]:
        """获取会话全部消息"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]
