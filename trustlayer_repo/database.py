import sqlite3
import json
from datetime import datetime

DB_PATH = "audit_trail.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                step_name TEXT,
                retry_count INTEGER,
                decision TEXT,
                details_json TEXT
            )
        """)

def log_step(session_id: str, step_name: str, retry_count: int, decision: str, details: dict):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO audit_logs (session_id, timestamp, step_name, retry_count, decision, details_json) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, datetime.utcnow().isoformat(), step_name, retry_count, decision, json.dumps(details))
        )

def get_logs(session_id: str = None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if session_id:
            rows = conn.execute("SELECT * FROM audit_logs WHERE session_id = ? ORDER BY id ASC", (session_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 50").fetchall()
        return [dict(r) for r in rows]
