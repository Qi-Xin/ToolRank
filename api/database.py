import sqlite3
import json
import uuid
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

import os as _os
DB_PATH = _os.environ.get("DB_PATH", "toolrank.db")


def get_db_path() -> str:
    return DB_PATH


def set_db_path(path: str):
    global DB_PATH
    DB_PATH = path


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS tools (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT,
    install_cmd TEXT,
    url TEXT,
    api_schema TEXT,
    embedding TEXT,
    avg_success_rate REAL DEFAULT 0.0,
    avg_latency_ms REAL DEFAULT 0.0,
    avg_token_cost REAL DEFAULT 0.0,
    avg_llm_score REAL DEFAULT 0.0,
    total_calls INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    tool_id TEXT NOT NULL,
    rating INTEGER NOT NULL,
    success INTEGER NOT NULL,
    latency_ms REAL,
    token_cost INTEGER,
    use_case TEXT,
    pros TEXT,
    cons TEXT,
    agent_model TEXT,
    task_description TEXT,
    llm_score REAL,
    llm_reason TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (tool_id) REFERENCES tools(id)
);

CREATE TABLE IF NOT EXISTS execution_logs (
    id TEXT PRIMARY KEY,
    tool_id TEXT NOT NULL,
    success INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    token_input INTEGER,
    token_output INTEGER,
    error_message TEXT,
    input_query TEXT,
    output_summary TEXT,
    agent_model TEXT,
    evaluated INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (tool_id) REFERENCES tools(id)
);
"""


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# --- Tool CRUD ---

def insert_tool(tool_data: dict) -> dict:
    tool_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO tools (id, name, description, category, install_cmd, url, api_schema, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tool_id,
                tool_data["name"],
                tool_data["description"],
                tool_data.get("category"),
                tool_data.get("install_cmd"),
                tool_data.get("url"),
                json.dumps(tool_data.get("api_schema")) if tool_data.get("api_schema") else None,
                json.dumps(tool_data.get("embedding")) if tool_data.get("embedding") else None,
            ),
        )
    return get_tool(tool_id)


def get_tool(tool_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM tools WHERE id = ?", (tool_id,)).fetchone()
        return _tool_row_to_dict(row) if row else None


def get_tool_by_name(name: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM tools WHERE name = ?", (name,)).fetchone()
        return _tool_row_to_dict(row) if row else None


def list_tools(category: Optional[str] = None) -> list[dict]:
    with get_conn() as conn:
        if category:
            rows = conn.execute("SELECT * FROM tools WHERE category = ?", (category,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tools").fetchall()
        return [_tool_row_to_dict(r) for r in rows]


def update_tool_embedding(tool_id: str, embedding: list[float]):
    with get_conn() as conn:
        conn.execute(
            "UPDATE tools SET embedding = ? WHERE id = ?",
            (json.dumps(embedding), tool_id),
        )


def update_tool_stats(tool_id: str):
    """Recompute aggregated stats from execution_logs and reviews."""
    with get_conn() as conn:
        stats = conn.execute(
            """SELECT
                AVG(CAST(success AS REAL)) as avg_success,
                AVG(latency_ms) as avg_latency,
                AVG(token_input + COALESCE(token_output, 0)) as avg_tokens,
                COUNT(*) as total
               FROM execution_logs WHERE tool_id = ?""",
            (tool_id,),
        ).fetchone()

        llm = conn.execute(
            "SELECT AVG(llm_score) as avg_score FROM reviews WHERE tool_id = ? AND llm_score IS NOT NULL",
            (tool_id,),
        ).fetchone()

        conn.execute(
            """UPDATE tools SET
                avg_success_rate = ?,
                avg_latency_ms = ?,
                avg_token_cost = ?,
                avg_llm_score = ?,
                total_calls = ?
               WHERE id = ?""",
            (
                stats["avg_success"] or 0.0,
                stats["avg_latency"] or 0.0,
                stats["avg_tokens"] or 0.0,
                llm["avg_score"] or 0.0,
                stats["total"] or 0,
                tool_id,
            ),
        )


def _tool_row_to_dict(row) -> dict:
    d = dict(row)
    if d.get("api_schema"):
        d["api_schema"] = json.loads(d["api_schema"])
    if d.get("embedding"):
        d["embedding"] = json.loads(d["embedding"])
    return d


# --- Review CRUD ---

def insert_review(review_data: dict) -> dict:
    review_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO reviews
               (id, tool_id, rating, success, latency_ms, token_cost,
                use_case, pros, cons, agent_model, task_description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                review_id,
                review_data["tool_id"],
                review_data["rating"],
                int(review_data["success"]),
                review_data.get("latency_ms"),
                review_data.get("token_cost"),
                review_data.get("use_case"),
                review_data.get("pros"),
                review_data.get("cons"),
                review_data.get("agent_model"),
                review_data.get("task_description"),
            ),
        )
    return get_review(review_id)


def get_review(review_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        return dict(row) if row else None


def get_reviews_for_tool(tool_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE tool_id = ? ORDER BY created_at DESC",
            (tool_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# --- Execution Log CRUD ---

def insert_execution_log(log_data: dict) -> dict:
    log_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO execution_logs
               (id, tool_id, success, latency_ms, token_input, token_output,
                error_message, input_query, output_summary, agent_model)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                log_id,
                log_data["tool_id"],
                int(log_data["success"]),
                log_data["latency_ms"],
                log_data.get("token_input"),
                log_data.get("token_output"),
                log_data.get("error_message"),
                log_data.get("input_query"),
                log_data.get("output_summary"),
                log_data.get("agent_model"),
            ),
        )
    return get_execution_log(log_id)


def get_execution_log(log_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM execution_logs WHERE id = ?", (log_id,)).fetchone()
        return dict(row) if row else None


def get_unevaluated_logs(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM execution_logs WHERE evaluated = 0 ORDER BY created_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def mark_log_evaluated(log_id: str):
    with get_conn() as conn:
        conn.execute("UPDATE execution_logs SET evaluated = 1 WHERE id = ?", (log_id,))
